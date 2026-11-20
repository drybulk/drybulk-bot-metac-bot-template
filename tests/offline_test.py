"""
Offline end-to-end test of the ensemble bot. No API key and no network needed.

A local server stands in for the Anthropic and OpenAI APIs (see
mock_llm_server.py), so the real request-building, parsing, aggregation, cost
tracking and ledger code all run. What this cannot check is how the real APIs
answer, in particular web search turns and actual token usage.

Run:  uv run python tests/offline_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

for name in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
    os.environ.pop(name, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from mock_llm_server import start_mock_server  # noqa: E402

SERVER, STATE, BASE = start_mock_server()
os.environ.update(
    {
        "METACULUS_TOKEN": "mock-token",
        "ANTHROPIC_API_KEY": "sk-ant-mock",
        "OPENAI_API_KEY": "sk-mock",
        "ANTHROPIC_API_BASE": BASE,
        "OPENAI_API_BASE": BASE + "/v1",
        "OPENAI_BASE_URL": BASE + "/v1",
    }
)
for name in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
    os.environ.pop(name, None)

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    ForecastReport,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from ensemble_bot import Settings, build_bot  # noqa: E402
from spend_ledger import ProviderBudget, SpendLedger, cycle_bounds  # noqa: E402

COMMON = dict(
    background_info="Mock background.",
    resolution_criteria="Resolves according to the mock criteria.",
    fine_print="",
)


def questions():
    return [
        BinaryQuestion(
            question_text="Will the mock event happen before 2027?",
            id_of_question=1,
            id_of_post=1,
            page_url="https://example.com/q/1",
            **COMMON,
        ),
        MultipleChoiceQuestion(
            question_text="Which mock option will win?",
            id_of_question=2,
            id_of_post=2,
            page_url="https://example.com/q/2",
            options=["Red", "Green", "Blue", "Other"],
            **COMMON,
        ),
        NumericQuestion(
            question_text="What will the mock index be on 2026-12-31?",
            id_of_question=3,
            id_of_post=3,
            page_url="https://example.com/q/3",
            upper_bound=100.0,
            lower_bound=0.0,
            open_upper_bound=True,
            open_lower_bound=False,
            unit_of_measure="points",
            **COMMON,
        ),
    ]


def run_bot(ledger_path: Path, question_list=None):
    """Forecast without publishing and return (reports, new requests, ledger)."""
    os.environ["BOT_LEDGER_PATH"] = str(ledger_path)
    settings = Settings.from_env()
    ledger = SpendLedger(settings.ledger_path, settings.budgets)
    bot = build_bot(settings, ledger, publish=False, skip_done=False)
    start = len(STATE.requests)
    reports = asyncio.run(
        bot.forecast_questions(question_list or questions(), return_exceptions=True)
    )
    return reports, STATE.requests[start:], ledger


def model_counts(requests) -> Counter:
    return Counter(r["body"].get("model") for r in requests)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok: {message}")


def test_pacing() -> None:
    print("pacing and billing cycle")
    check(cycle_bounds(date(2026, 12, 20), 1) == (date(2026, 12, 1), date(2027, 1, 1)), "cycle from day 1")
    check(cycle_bounds(date(2026, 12, 5), 17) == (date(2026, 11, 17), date(2026, 12, 17)), "cycle from day 17, before reset")
    check(cycle_bounds(date(2027, 2, 28), 31) == (date(2027, 2, 28), date(2027, 3, 31)), "reset day clamps to month end")

    budgets = {"openai": ProviderBudget("openai", 100.0)}
    with tempfile.TemporaryDirectory() as folder:
        now = datetime(2026, 12, 16, 12, tzinfo=timezone.utc)  # 50% of December
        ledger = SpendLedger(Path(folder) / "l.json", budgets, now=now)
        for spent, expected in ((0, "full"), (74, "full"), (76, "reduced"), (94, "reduced"), (96, "minimal"), (100, "off")):
            ledger._entry("openai")["spend_usd"] = float(spent)
            check(ledger.choose_tier("openai") == expected, f"${spent} spent at mid-cycle -> {expected}")
        ledger.save()
        reloaded = SpendLedger(Path(folder) / "l.json", budgets, now=now)
        check(reloaded.spend("openai") == 100.0, "ledger survives save and reload")
        january = SpendLedger(Path(folder) / "l.json", budgets, now=datetime(2027, 1, 2, tzinfo=timezone.utc))
        check(january.spend("openai") == 0.0 and january.choose_tier("openai") == "full", "new cycle starts from zero")
        (Path(folder) / "broken.json").write_text("{not json")
        broken = SpendLedger(Path(folder) / "broken.json", budgets, now=now)
        check(broken.choose_tier("openai") == "full", "unreadable ledger does not stop the bot")


def test_full_ensemble() -> None:
    print("full ensemble on binary, multiple choice and numeric questions")
    with tempfile.TemporaryDirectory() as folder:
        reports, requests, ledger = run_bot(Path(folder) / "ledger.json")
        for report in reports:
            check(isinstance(report, ForecastReport), f"report produced: {getattr(report, 'question', report)!s:.60}")
        binary, multiple_choice, numeric = reports
        check(abs(binary.prediction - 0.37) < 1e-9, "binary forecasts aggregate to 37%")
        total = sum(o.probability for o in multiple_choice.prediction.predicted_options)
        check(abs(total - 1) < 1e-6, "multiple choice probabilities sum to 1")
        check(len(numeric.prediction.declared_percentiles) >= 6, "numeric distribution has the percentiles")
        check(all(not r.errors for r in reports), "no forecast in the ensemble failed")
        check("Forecaster model: anthropic/claude-opus-5-5" in binary.explanation, "explanation names the forecaster model")
        check("Web research (openai/gpt-6.2-sol)" in binary.explanation, "explanation includes the web research")

        counts = model_counts(requests)
        forecasts = [r for r in requests if "convert the text" not in json.dumps(r["body"]) and not r["body"].get("tools")]
        forecast_counts = model_counts(forecasts)
        check(forecast_counts["claude-opus-5-5"] == 9, "3 Claude Opus forecasts per question")
        check(forecast_counts["gpt-6.2-astra"] == 3 and forecast_counts["gpt-6.2-sol"] == 6, "1 Astra and 2 Sol forecasts per question")
        check(counts["claude-haiku-5-5"] > 0 and counts["gpt-6-luna"] > 0, "each provider parses with its own small model")

        claude_forecast = next(r["body"] for r in forecasts if r["body"]["model"] == "claude-opus-5-5")
        check("temperature" not in claude_forecast and "thinking" not in claude_forecast, "Claude request sends no temperature or thinking override")
        check(claude_forecast.get("output_config") == {"effort": "high"} and claude_forecast.get("max_tokens") == 16000, "Claude request sets effort high and max_tokens")
        gpt_forecast = next(r["body"] for r in forecasts if r["body"]["model"] == "gpt-6.2-astra")
        check("temperature" not in gpt_forecast, "GPT request sends no temperature")
        searches = [r for r in requests if r["body"].get("tools")]
        check(sum(r["path"].endswith("/messages") for r in searches) == 3 and sum(r["path"].endswith("/responses") for r in searches) == 3, "one web research call per provider per question")
        claude_search = next(r["body"] for r in searches if r["path"].endswith("/messages"))
        check(claude_search["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}], "Claude research uses the web search tool with 5 searches")

        saved = json.loads((Path(folder) / "ledger.json").read_text())
        for provider in ("anthropic", "openai"):
            entry = next(iter(saved["cycles"][provider].values()))
            check(entry["questions"] == 3 and entry["spend_usd"] > 0, f"ledger file records 3 questions and ${entry['spend_usd']:.3f} for {provider}")
        # Mock usage is 3000 input and 2000 output tokens per call.
        # Per question: 3 Opus forecasts, 1 Sonnet research call with one $0.01
        # web search, and 2 Haiku parser calls per forecast.
        anthropic_expected = 3 * (3 * 0.052 + 0.026 + 0.01 + 3 * 2 * 0.0013)
        check(abs(ledger.spend("anthropic") - anthropic_expected) < 0.005, f"Anthropic spend matches list prices, each call counted once (${ledger.spend('anthropic'):.3f} vs ${anthropic_expected:.3f})")
        openai_tokens_only = 3 * (0.104 + 2 * 0.026 + 0.026 + 3 * 2 * 0.0013)
        check(ledger.spend("openai") >= openai_tokens_only - 0.005, f"OpenAI spend covers at least the token cost (${ledger.spend('openai'):.3f} vs ${openai_tokens_only:.3f})")


def test_provider_over_budget() -> None:
    print("provider over its cycle budget")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ledger.json"
        os.environ["BOT_LEDGER_PATH"] = str(path)
        settings = Settings.from_env()
        seeded = SpendLedger(path, settings.budgets)
        seeded.record("anthropic", "full", 85.0)
        seeded.save()
        reports, requests, ledger = run_bot(path, questions()[:1])
        check(isinstance(reports[0], ForecastReport), "question still forecast")
        check(not any(str(m).startswith("claude") for m in model_counts(requests)), "no Claude call once the budget is used up")
        check(ledger.spend("anthropic") == 85.0, "no further spend recorded for the capped provider")

        seeded.record("openai", "full", 85.0)
        seeded.save()
        reports, requests, _ = run_bot(path, questions()[:1])
        check(isinstance(reports[0], BaseException) and not requests, "both capped: question fails cleanly without any API call")


def test_reduced_tier() -> None:
    print("provider ahead of pace")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "ledger.json"
        os.environ["BOT_LEDGER_PATH"] = str(path)
        os.environ["BOT_OPENAI_MONTHLY_BUDGET"] = "10"
        try:
            settings = Settings.from_env()
            seeded = SpendLedger(path, settings.budgets)
            seeded.record("openai", "full", 9.99)  # above every pace line, below budget
            seeded.save()
            reports, requests, _ = run_bot(path, questions()[:1])
        finally:
            del os.environ["BOT_OPENAI_MONTHLY_BUDGET"]
        forecasts = [r for r in requests if "convert the text" not in json.dumps(r["body"]) and not r["body"].get("tools")]
        counts = model_counts(forecasts)
        expected_tier = seeded.choose_tier("openai")
        check(isinstance(reports[0], ForecastReport), "question still forecast")
        check(counts["gpt-6.2-astra"] == 0, f"tier '{expected_tier}' drops the expensive OpenAI model")
        check(counts["claude-opus-5-5"] == 3, "the other provider stays at full")


def test_provider_outage() -> None:
    print("one provider's API failing")
    with tempfile.TemporaryDirectory() as folder:
        STATE.fail_paths.add("/v1/messages")
        try:
            reports, _, ledger = run_bot(Path(folder) / "ledger.json", questions()[:1])
        finally:
            STATE.fail_paths.clear()
        report = reports[0]
        check(isinstance(report, ForecastReport), "forecast produced from the working provider")
        check(abs(report.prediction - 0.37) < 1e-9 and len(report.errors) == 3, "3 failed Claude forecasts are reported as errors, not fatal")
        check(ledger.spend("openai") > 0 and ledger.spend("anthropic") == 0, "only the working provider is charged")


def test_claude_search_failure() -> None:
    print("Claude web searches failing")
    import ensemble_bot

    ensemble_bot._SEARCH_RETRY_PAUSE_SECONDS = 0
    with tempfile.TemporaryDirectory() as folder:
        os.environ["BOT_LEDGER_PATH"] = str(Path(folder) / "ledger.json")
        settings = Settings.from_env()
        ledger = SpendLedger(settings.ledger_path, settings.budgets)
        bot = build_bot(settings, ledger, publish=False, skip_done=False)
        start = len(STATE.requests)
        STATE.search_error_code = "too_many_requests"
        try:
            reports = asyncio.run(bot.forecast_questions(questions()[:1], return_exceptions=True))
        finally:
            STATE.search_error_code = None
        requests = STATE.requests[start:]
        claude_searches = [r for r in requests if r["body"].get("tools") and r["path"].endswith("/messages")]
        report = reports[0]
        check(isinstance(report, ForecastReport), "question still forecast")
        check(len(claude_searches) == 2, "the failed research call is repeated once")
        check("Web research (anthropic" not in report.explanation and "Web research (openai" in report.explanation, "a briefing without any search result is left out, the other one is kept")
        notes = bot.diagnostics[0]
        check(notes["claude_search_errors"] == {"too_many_requests": 2} and notes["claude_searches_ok"] == 0, "search errors are recorded for the run summary")
        check(notes["research_sources"] == "1/2" and notes["forecasts"] == "6/6", "summary shows 1 of 2 research sources and 6 of 6 forecasts")

        reports = asyncio.run(bot.forecast_questions(questions()[:1], return_exceptions=True))
        check(bot.diagnostics[1]["claude_searches_ok"] == 1 and bot.diagnostics[1]["research_sources"] == "2/2", "working searches are counted and both briefings are used")


def test_entry_point() -> None:
    print("command line entry point")
    import ensemble_bot
    from forecasting_tools import MetaculusClient

    calls: list = []

    def fake_fetch(self, tournament_id, *args, **kwargs):
        calls.append(tournament_id)
        fetched = questions()
        fetched[0].already_forecasted = True  # must be skipped in tournament mode
        return fetched

    original = MetaculusClient.get_all_open_questions_from_tournament
    MetaculusClient.get_all_open_questions_from_tournament = fake_fetch
    try:
        with tempfile.TemporaryDirectory() as folder:
            os.environ["BOT_LEDGER_PATH"] = str(Path(folder) / "ledger.json")
            os.environ["GITHUB_STEP_SUMMARY"] = str(Path(folder) / "summary.md")
            start = len(STATE.requests)
            code = asyncio.run(ensemble_bot.run("tournament", publish=False, max_questions=3))
            done = {r["body"]["model"] for r in STATE.requests[start:]}
            check(code == 0, "run exits with status 0")
            check(len(calls) == 2, "both the seasonal tournament and MiniBench are fetched")
            saved = json.loads((Path(folder) / "ledger.json").read_text())
            entry = next(iter(saved["cycles"]["openai"].values()))
            check(entry["questions"] == 3, "already-forecast questions are skipped and --max-questions caps the run at 3")
            summary_text = (Path(folder) / "summary.md").read_text()
            check("| openai |" in summary_text, "spend table is written to the workflow summary")
            check("| https://example.com/q/2 |" in summary_text and "6/6" in summary_text, "per-question diagnostics are written to the workflow summary")
            check("gpt-4o" not in " ".join(done), "no outdated default model is ever called")

            STATE.fail_paths.update({"/v1/messages", "/v1/chat/completions", "/v1/responses"})
            try:
                code = asyncio.run(ensemble_bot.run("minibench", publish=False, max_questions=1))
            finally:
                STATE.fail_paths.clear()
            check(code == 1, "run exits with status 1 when most questions fail")
    finally:
        MetaculusClient.get_all_open_questions_from_tournament = original
        os.environ.pop("GITHUB_STEP_SUMMARY", None)


if __name__ == "__main__":
    test_pacing()
    test_full_ensemble()
    test_provider_over_budget()
    test_reduced_tier()
    test_provider_outage()
    test_claude_search_failure()
    test_entry_point()
    print("\nAll offline tests passed.")
