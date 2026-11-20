"""
Local stand-in for the Anthropic and OpenAI APIs, used by the offline tests.

It records every request body, so tests can check what the bot actually sends,
and returns canned answers that look like research, a forecast rationale, or a
parser result depending on the prompt.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORECAST_TEXT = {
    "binary": "Base rates and the news both point the same way.\n\nProbability: 37%",
    "multiple_choice": "Reasoning about each option.\n\n{options}",
    "numeric": (
        "Reasoning about the distribution.\n\n"
        "Percentile 10: {p10}\nPercentile 20: {p20}\nPercentile 40: {p40}\n"
        "Percentile 60: {p60}\nPercentile 80: {p80}\nPercentile 90: {p90}"
    ),
}


class MockState:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.fail_paths: set[str] = set()  # paths that should return HTTP 500
        self.fail_models: set[str] = set()
        self.lock = threading.Lock()


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_flatten(part) for part in content)
    if isinstance(content, dict):
        return _flatten(content.get("text") or content.get("content") or "")
    return ""


def _prompt_text(body: dict) -> str:
    parts = []
    if "system" in body:
        parts.append(_flatten(body["system"]))
    for key in ("messages", "input"):
        value = body.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(_flatten(m.get("content")) for m in value)
    return "\n".join(parts)


def _answer(prompt: str) -> str:
    if "convert the text to the following" in prompt:
        return _parser_answer(prompt)
    if "You do not produce forecasts yourself" in prompt:
        return (
            "- 2026-11-18: Mock news item relevant to the question "
            "(https://example.com/a).\n- Status quo has held for six months."
        )
    if "Percentile 10:" in prompt or "percentile" in prompt.lower():
        return FORECAST_TEXT["numeric"].format(
            p10=12, p20=18, p40=27, p60=35, p80=50, p90=64
        )
    match = re.search(r"The options are: (\[.*?\])", prompt, re.S)
    if match:
        try:
            options = json.loads(match.group(1).replace("'", '"'))
        except json.JSONDecodeError:
            options = ["A", "B"]
        share = round(100 / len(options))
        lines = "\n".join(f"{option}: {share}%" for option in options)
        return FORECAST_TEXT["multiple_choice"].format(options=lines)
    return FORECAST_TEXT["binary"]


def _parser_answer(prompt: str) -> str:
    text = prompt.split("START TEXT")[1].split("END TEXT")[0]
    if "BinaryPrediction" in prompt:
        number = re.search(r"Probability:\s*([\d.]+)%", text)
        value = float(number.group(1)) / 100 if number else 0.5
        return json.dumps({"prediction_in_decimal": value})
    if "PredictedOptionList" in prompt:
        pairs = re.findall(r"^(.+?):\s*([\d.]+)%\s*$", text, re.M)
        return json.dumps(
            {
                "predicted_options": [
                    {"option_name": name.strip(), "probability": float(p) / 100}
                    for name, p in pairs
                ]
            }
        )
    pairs = re.findall(r"Percentile (\d+):\s*([\d.]+)", text)
    return json.dumps(
        [{"percentile": int(p) / 100, "value": float(v)} for p, v in pairs]
    )


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            with state.lock:
                state.requests.append({"path": self.path, "body": body})
            model = body.get("model", "")
            if self.path in state.fail_paths or model in state.fail_models:
                self._send(500, {"error": {"type": "api_error", "message": "mock failure"}})
                return
            text = _answer(_prompt_text(body))
            tool_types = [str(t.get("type", "")) for t in body.get("tools") or []]
            searching = any(t.startswith("web_search") for t in tool_types)
            if self.path.endswith("/messages"):
                content = [{"type": "text", "text": text}]
                usage = {"input_tokens": 3000, "output_tokens": 2000}
                if searching:
                    # Shape of a real web search turn: the server-side tool
                    # call, its result block, then cited text.
                    content = [
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_mock",
                            "name": "web_search",
                            "input": {"query": "mock query"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_mock",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/a",
                                    "title": "Mock result",
                                    "encrypted_content": "abc",
                                    "page_age": "2 days ago",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": text,
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://example.com/a",
                                    "title": "Mock result",
                                    "encrypted_index": "xyz",
                                    "cited_text": "Mock news item",
                                }
                            ],
                        },
                    ]
                    usage["server_tool_use"] = {"web_search_requests": 1}
                payload = {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": content,
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": usage,
                }
            elif self.path.endswith("/chat/completions"):
                payload = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3000,
                        "completion_tokens": 2000,
                        "total_tokens": 5000,
                    },
                }
            elif self.path.endswith("/responses"):
                payload = {
                    "id": "resp_mock",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": model,
                    "output": (
                        [
                            {
                                "type": "web_search_call",
                                "id": "ws_mock",
                                "status": "completed",
                                "action": {"type": "search", "query": "mock query"},
                            }
                        ]
                        if searching
                        else []
                    )
                    + [
                        {
                            "type": "message",
                            "id": "msg_mock",
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": text, "annotations": []}
                            ],
                        }
                    ],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                    "usage": {
                        "input_tokens": 3000,
                        "output_tokens": 2000,
                        "total_tokens": 5000,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                }
            else:
                self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                return
            self._send(200, payload)

        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def start_mock_server() -> tuple[ThreadingHTTPServer, MockState, str]:
    state = MockState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state, f"http://127.0.0.1:{server.server_address[1]}"
