"""
Ensemble forecasting bot built on the Metaculus template bot (main.py).

What it changes relative to the template:
  * Forecasts come from several models of two providers (Anthropic and OpenAI)
    and are combined by the framework's aggregation (median for binary).
  * Research combines AskNews (if credentials exist) with one web-search
    briefing per provider. At the full Anthropic tier, Claude also writes two
    deeper briefings (outside view, primary sources) when time allows.
  * Shadow forecasts made from the standard research alone are logged next to
    the submitted ones, so the value of the deeper research can be measured
    after questions resolve (state/forecast_log.jsonl).
  * A spend ledger paces each provider's monthly credit. When a provider runs
    ahead of pace the bot uses fewer and cheaper calls there, and when the
    budget is used up it skips that provider instead of failing.

The prompts and answer parsing are inherited unchanged from the template, so
upstream template fixes can be merged without touching this file.

Run:  python ensemble_bot.py --mode tournament
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import dotenv

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    ApiFilter,
    AskNewsSearcher,
    ForecastReport,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    clean_indents,
)
from forecasting_tools.forecast_bots.forecast_bot import (  # noqa: E402
    ResearchWithPredictions,
)

from main import FallTemplateBot2026  # noqa: E402
from spend_ledger import OFF, ProviderBudget, SpendLedger  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


def _env(name: str, default: str) -> str:
    """Environment value, treating unset, empty and placeholder values as unset."""
    value = (os.getenv(name) or "").strip()
    return value if value else default


def _has_key(name: str) -> bool:
    value = (os.getenv(name) or "").strip()
    return bool(value) and "1234567890" not in value and not value.startswith("your")


# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class TierPlan:
    forecasters: tuple[str, ...]  # "strong" or "mid", one entry per forecast
    web_searches: int  # max web searches in the research call, 0 disables it
    # Extra research calls on top of the standard briefing (keys of DEEP_BRIEFS).
    deep_briefs: tuple[str, ...] = ()
    # Whether this provider adds one shadow forecast (see _run_shadow).
    shadow: bool = False


# How much of the ensemble each provider contributes at each spend tier.
TIER_PLANS: dict[str, dict[str, TierPlan]] = {
    "anthropic": {
        "full": TierPlan(
            ("strong", "strong", "strong"),
            5,
            deep_briefs=("outside_view", "primary_sources"),
            shadow=True,
        ),
        "reduced": TierPlan(("strong", "mid"), 3),
        "minimal": TierPlan(("mid",), 2),
    },
    "openai": {
        "full": TierPlan(("strong", "mid", "mid"), 5, shadow=True),
        "reduced": TierPlan(("mid", "mid"), 3),
        "minimal": TierPlan(("mid",), 2),
    },
}


@dataclass(frozen=True)
class DeepBrief:
    label: str
    web_searches: int
    web_fetches: int  # 0 leaves the web fetch tool out
    instructions: str


# Each brief is its own API call. The server-side tool loop of one call stops
# after 10 model turns (the response then ends without a briefing), so a brief
# keeps its searches and fetches together below that.
DEEP_BRIEFS: dict[str, DeepBrief] = {
    "outside_view": DeepBrief(
        label="Outside view",
        web_searches=6,
        web_fetches=0,
        instructions="""
            Another assistant is already writing a briefing on the current status and the latest news.
            Do not repeat that work. Your job is the outside view. Search the web and report:
            1. Reference classes: for each relevant class of similar past cases, how many cases there were, how many ended each way, and over what period. Give the counts and the source. If you can only estimate a frequency, say so.
            2. Similar forecasting questions or prediction markets (Metaculus, Polymarket, Kalshi, Manifold, Good Judgment, bookmakers) on the same or a closely related event: how they resolved, or the price or probability they show now, with its date.
            3. For questions about a quantity: recent values, the typical change over a period as long as the time left, and the highest and lowest values in the relevant history, with dates.
            4. How often the status quo held over a window as long as the time left until resolution.

            Give the source URL for each fact. Say plainly when you found no usable base rate. Do not invent one.
            Keep the briefing under 400 words.
            """,
    ),
    "primary_sources": DeepBrief(
        label="Primary sources",
        web_searches=4,
        web_fetches=3,
        instructions="""
            Another assistant is already writing a briefing on the current status and the latest news.
            Do not repeat that work. Your job is to check the sources that decide the question:
            1. Identify the exact source, publication or data series that the resolution criteria name, or the most authoritative one if none is named. Open it and report the latest value or status it shows, with its date and URL. Quote the relevant figure or sentence exactly.
            2. Check the resolution criteria and fine print against that source: units, cut-off dates, time zones, revisions, and edge cases that could change how the question resolves.
            3. List the scheduled releases, decisions or events between today and the resolution date that can change the outcome, with their dates.
            4. State what would still have to happen for each possible outcome, given the latest value.

            Open pages with web fetch instead of relying on search snippets when a page is likely to hold the exact figure.
            Give the source URL for each fact. Say plainly if you could not open the deciding source.
            Keep the briefing under 400 words.
            """,
    ),
}

# A deep briefing shorter than this is treated as failed. A call whose tool
# loop was cut off returns only the few words written before the first search.
_MIN_DEEP_BRIEF_CHARS = 300


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    key_env: str
    strong: str
    mid: str
    parser: str
    budget: ProviderBudget


@dataclass(frozen=True)
class Settings:
    providers: dict[str, ProviderSettings]
    ledger_path: str
    max_concurrent_questions: int
    max_concurrent_claude_searches: int
    request_timeout: int
    asknews_enabled: bool
    deep_research: bool
    deep_min_minutes_left: float
    deep_max_per_run: int
    shadow_forecasts: bool
    forecast_log_path: str

    @classmethod
    def from_env(cls) -> Settings:
        def provider(
            name: str, key_env: str, strong: str, mid: str, parser: str, budget: str
        ) -> ProviderSettings:
            upper = name.upper()
            return ProviderSettings(
                name=name,
                key_env=key_env,
                strong=_env(f"BOT_{upper}_STRONG", strong),
                mid=_env(f"BOT_{upper}_MID", mid),
                parser=_env(f"BOT_{upper}_PARSER", parser),
                budget=ProviderBudget(
                    name=name,
                    budget_usd=float(_env(f"BOT_{upper}_MONTHLY_BUDGET", budget)),
                    reset_day=int(_env(f"BOT_{upper}_RESET_DAY", "1")),
                ),
            )

        candidates = [
            provider(
                "anthropic",
                "ANTHROPIC_API_KEY",
                "anthropic/claude-opus-5-5",
                "anthropic/claude-sonnet-5-5",
                "anthropic/claude-haiku-5-5",
                # Monthly credit of 200 USD (Ultra plan). On a 100 USD credit
                # set the repository variable BOT_ANTHROPIC_MONTHLY_BUDGET to 85.
                "185",
            ),
            provider(
                "openai",
                "OPENAI_API_KEY",
                "openai/gpt-6.2-astra",
                "openai/gpt-6.2-sol",
                "openai/gpt-6-luna",
                "85",
            ),
        ]
        return cls(
            providers={p.name: p for p in candidates if _has_key(p.key_env)},
            ledger_path=_env("BOT_LEDGER_PATH", "state/spend_ledger.json"),
            max_concurrent_questions=int(_env("BOT_MAX_CONCURRENT_QUESTIONS", "4")),
            max_concurrent_claude_searches=int(
                _env("BOT_MAX_CONCURRENT_CLAUDE_SEARCHES", "2")
            ),
            request_timeout=int(_env("BOT_REQUEST_TIMEOUT_SECONDS", "600")),
            asknews_enabled=(
                _has_key("ASKNEWS_CLIENT_ID") and _has_key("ASKNEWS_SECRET")
            )
            or _has_key("ASKNEWS_API_KEY"),
            deep_research=_env("BOT_DEEP_RESEARCH", "1") != "0",
            deep_min_minutes_left=float(_env("BOT_DEEP_MIN_MINUTES_LEFT", "60")),
            deep_max_per_run=int(_env("BOT_DEEP_MAX_PER_RUN", "12")),
            shadow_forecasts=_env("BOT_SHADOW_FORECASTS", "1") != "0",
            forecast_log_path=_env("BOT_FORECAST_LOG_PATH", "state/forecast_log.jsonl"),
        )

    @property
    def budgets(self) -> dict[str, ProviderBudget]:
        return {name: p.budget for name, p in self.providers.items()}


# ------------------------------------------------------------------- slots


@dataclass
class Slot:
    """One forecast of the ensemble: which model reasons and which one parses."""

    provider: str
    forecaster: CostedLlm
    parser: CostedLlm
    shadow: bool = False  # forecast from the standard research only, not submitted


_current_slot: ContextVar[Slot | None] = ContextVar("_current_slot", default=None)

# Per-question spend by provider. The dict is created when a question starts and
# is shared by every task spawned for that question.
_cost_sink: ContextVar[dict[str, float] | None] = ContextVar("_cost_sink", default=None)


# Web search outcome of the research call running in the current task:
# {"ok": successful searches, "errors": Counter of error codes}.
_search_report: ContextVar[dict | None] = ContextVar("_search_report", default=None)


class CostedLlm(GeneralLlm):
    """
    GeneralLlm that adds the cost of each completed call to the current
    question's spend, under its own provider.

    The framework's MonetaryCostManager is not used for the ledger because its
    callback can arrive after the call returned and then counts the same call
    twice when an outer manager is still open.
    """

    def __init__(self, provider: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.provider = provider

    async def _mockable_direct_call_to_model(self, prompt):  # type: ignore[override]
        response = await super()._mockable_direct_call_to_model(prompt)
        sink = _cost_sink.get()
        if sink is not None:
            sink[self.provider] += response.cost or 0.0
        return response

    async def _call_litellm(self, prompt):  # type: ignore[override]
        response = await super()._call_litellm(prompt)
        try:
            self._note_web_searches(response)
        except Exception as error:  # diagnostics must never break a forecast
            logger.debug(f"Could not read web search results: {error}")
        return response

    def _note_web_searches(self, response) -> None:
        """Count successful and failed Anthropic web searches of one call."""
        report = _search_report.get()
        if report is None:
            return
        message = response.choices[0].message
        fields = getattr(message, "provider_specific_fields", None) or {}
        for result in fields.get("web_search_results") or []:
            content = result.get("content") if isinstance(result, dict) else None
            if isinstance(content, dict) and content.get("error_code"):
                report["errors"][content["error_code"]] += 1
            else:
                report["ok"] += 1


class LlmFactory:
    """Builds each model wrapper once, with the request shape its API accepts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: dict[tuple, CostedLlm] = {}

    def _get(self, provider: str, key: tuple, **kwargs: Any) -> CostedLlm:
        if key not in self._cache:
            self._cache[key] = CostedLlm(
                provider=provider,
                timeout=self.settings.request_timeout,
                allowed_tries=2,
                **kwargs,
            )
        return self._cache[key]

    def forecaster(self, provider: str, strength: str) -> GeneralLlm:
        p = self.settings.providers[provider]
        model = p.strong if strength == "strong" else p.mid
        if provider == "anthropic":
            # Thinking is adaptive by default on current Claude models and
            # sampling parameters are rejected, so only effort is set.
            return self._get(
                provider,
                ("forecast", model),
                model=model,
                max_tokens=16000,
                output_config={"effort": "high"},
            )
        return self._get(provider, ("forecast", model), model=model)

    def parser(self, provider: str) -> GeneralLlm:
        model = self.settings.providers[provider].parser
        if provider == "anthropic":
            return self._get(provider, ("parser", model), model=model, max_tokens=4000)
        return self._get(provider, ("parser", model), model=model)

    def web_researcher(self, provider: str, max_searches: int) -> GeneralLlm:
        model = self.settings.providers[provider].mid
        if provider == "anthropic":
            return self._get(
                provider,
                ("research", model, max_searches),
                model=model,
                max_tokens=16000,
                tools=[
                    {
                        "type": _env("BOT_ANTHROPIC_WEB_SEARCH_TOOL", "web_search_20260209"),
                        "name": "web_search",
                        "max_uses": max_searches,
                    }
                ],
            )
        # OpenAI exposes web search as a Responses API tool. It has no
        # per-request search cap, so `max_searches` only applies to Anthropic.
        return self._get(
            provider,
            ("research", model),
            model=model,
            responses_api=True,
            tools=[{"type": "web_search"}],
        )

    def deep_researcher(self, brief_key: str) -> GeneralLlm:
        """Claude call for one deep briefing (Anthropic only), on the strong model."""
        brief = DEEP_BRIEFS[brief_key]
        model = self.settings.providers["anthropic"].strong
        tools: list[dict] = [
            {
                "type": _env("BOT_ANTHROPIC_WEB_SEARCH_TOOL", "web_search_20260209"),
                "name": "web_search",
                "max_uses": brief.web_searches,
            }
        ]
        if brief.web_fetches > 0:
            tools.append(
                {
                    "type": _env("BOT_ANTHROPIC_WEB_FETCH_TOOL", "web_fetch_20260209"),
                    "name": "web_fetch",
                    "max_uses": brief.web_fetches,
                    # A long page must not crowd the briefing out of the budget.
                    "max_content_tokens": 12000,
                }
            )
        return self._get(
            "anthropic",
            ("deep", brief_key, model),
            model=model,
            max_tokens=16000,
            tools=tools,
        )


# Per-question facts for the run summary (research sources, search errors,
# forecasts that succeeded, cost). Shared by the tasks of one question.
_question_notes: ContextVar[dict | None] = ContextVar("_question_notes", default=None)

# Individual forecasts of the current question, submitted and shadow, for the
# forecast log. Shared by the tasks of one question.
_slot_results: ContextVar[list | None] = ContextVar("_slot_results", default=None)

_SEARCH_RETRY_PAUSE_SECONDS = 30


def _jsonable_prediction(value: Any) -> Any:
    """Reduce a prediction to plain JSON: a number, option -> probability, or percentiles."""
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    options = getattr(value, "predicted_options", None)
    if options is not None:
        return {o.option_name: round(float(o.probability), 6) for o in options}
    percentiles = getattr(value, "declared_percentiles", None)
    if percentiles is not None:
        return [[round(float(p.percentile), 6), float(p.value)] for p in percentiles]
    return str(value)[:500]


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n\n[Research shortened for display.]"


def _short_time(value: Any) -> str:
    """Format a question time as 'YYYY-MM-DD HH:MM' in UTC, or '-' if unknown."""
    if value is None:
        return "-"
    try:
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%d %H:%M")
    except (AttributeError, ValueError):
        return str(value)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


COVERAGE_DAYS = 7


async def coverage_markdown(client: MetaculusClient, tournament_ids: list) -> str:
    """
    Count the tournament questions that opened in the last days and split them
    into forecast, missed (closed without a forecast from this account) and
    still open. Tournament questions stay open for a few hours only, so this
    shows whether the bot runs often enough.
    """
    now = datetime.now(timezone.utc)
    rows: list[str] = []
    missed_lines: list[str] = []
    for tournament_id in tournament_ids:
        api_filter = ApiFilter(
            allowed_tournaments=[tournament_id],
            open_time_gt=now - timedelta(days=COVERAGE_DAYS),
            # The library sends this bound to Metaculus as a date without a time,
            # so "now" would leave out every question that opened today (UTC).
            open_time_lt=now + timedelta(days=1),
        )
        questions = await client.get_questions_matching_filter(
            api_filter, num_questions=1000, error_if_question_target_missed=False
        )
        questions = [q for q in questions if (_as_utc(q.open_time) or now) <= now]
        forecast = [q for q in questions if q.already_forecasted]
        pending = [q for q in questions if not q.already_forecasted]
        missed = [
            q for q in pending if (_as_utc(q.close_time) or now + timedelta(1)) <= now
        ]
        rows.append(
            f"| {tournament_id} | {len(questions)} | {len(forecast)} | "
            f"{len(missed)} | {len(pending) - len(missed)} |"
        )
        for q in sorted(missed, key=lambda q: _as_utc(q.open_time) or now):
            missed_lines.append(
                f"| {q.page_url} | {_short_time(q.open_time)} | "
                f"{_short_time(q.close_time)} |"
            )
    text = (
        f"\n### Questions opened in the last {COVERAGE_DAYS} days\n\n"
        "| Tournament | Opened | Forecast | Missed | Still open, no forecast |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )
    if missed_lines:
        text += (
            "\n| Missed question | Opened (UTC) | Closed (UTC) |\n|---|---|---|\n"
            + "\n".join(missed_lines[-40:])
            + "\n"
        )
    return text


# --------------------------------------------------------------------- bot


class EnsembleBot(FallTemplateBot2026):
    def configure(self, settings: Settings, ledger: SpendLedger) -> EnsembleBot:
        self._settings = settings
        self._ledger = ledger
        self._factory = LlmFactory(settings)
        self._question_semaphore: asyncio.Semaphore | None = None
        self._claude_search_semaphore: asyncio.Semaphore | None = None
        self._asknews_lock: asyncio.Lock | None = None
        self.diagnostics: list[dict] = []  # one entry per question, for the run summary
        self.run_mode = "unknown"  # set by run(); written to the forecast log
        self._deep_started = 0  # questions given deep research in this run
        return self

    # The template's forecasting methods ask for the "default" and "parser"
    # models. Inside an ensemble slot those resolve to the slot's models.
    def get_llm(self, purpose: str = "default", guarantee_type=None):  # type: ignore[override]
        slot = _current_slot.get()
        if slot is not None and purpose in ("default", "parser"):
            llm = slot.forecaster if purpose == "default" else slot.parser
            return llm.model if guarantee_type == "string_name" else llm
        return super().get_llm(purpose, guarantee_type)

    # ------------------------------------------------------------ planning
    def _plan(self) -> dict[str, str]:
        """Spend tier per provider for the next question, e.g. {"openai": "full"}."""
        return {
            name: self._ledger.choose_tier(name) for name in self._settings.providers
        }

    def _slots(self, plan: dict[str, str]) -> list[Slot]:
        slots = []
        for provider, tier in plan.items():
            if tier == OFF:
                continue
            for strength in TIER_PLANS[provider][tier].forecasters:
                slots.append(
                    Slot(
                        provider=provider,
                        forecaster=self._factory.forecaster(provider, strength),
                        parser=self._factory.parser(provider),
                    )
                )
        return slots

    # ------------------------------------------------------------ research
    async def run_research(self, question: MetaculusQuestion) -> str:
        plan = self._plan()
        research, _, _ = await self._gather_research(
            question, plan, self._deep_briefs_for(question, plan)
        )
        return research

    @staticmethod
    def _question_block(question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

    @staticmethod
    def _web_research_prompt(question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            You are a research assistant to a professional forecaster.
            You do not produce forecasts yourself. Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.

            Search the web and write a factual briefing on the question below. Cover:
            1. The current status and the most recent developments, each with its date.
            2. What must happen for each possible resolution under the criteria, and whether any of it has already happened.
            3. Base rates or historical precedents that apply.
            4. Scheduled events and deadlines before the question resolves.
            5. Existing forecasts if there are any (prediction markets, bookmakers, analyst consensus), with their dates.

            Give the source URL for each fact. Say plainly when information is missing, uncertain or out of date.
            Keep the briefing under 500 words.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            {question.fine_print}
            """
        )

    @classmethod
    def _deep_research_prompt(cls, question: MetaculusQuestion, brief_key: str) -> str:
        header = clean_indents(
            f"""
            You are a research assistant to a professional forecaster.
            You do not produce forecasts yourself. Today is {datetime.now(timezone.utc).strftime("%Y-%m-%d")}.
            """
        )
        instructions = clean_indents(DEEP_BRIEFS[brief_key].instructions)
        return f"{header}\n\n{instructions}\n\n{cls._question_block(question)}"

    def _deep_briefs_for(
        self, question: MetaculusQuestion, plan: dict[str, str]
    ) -> tuple[str, ...]:
        """
        Deep briefings this question gets. They need the full Anthropic tier,
        enough time before the question closes, and room in the per-run cap, so
        a burst of questions or a late start falls back to the standard research
        instead of missing the question.
        """
        if not self._settings.deep_research:
            return ()
        tier = plan.get("anthropic", OFF)
        if tier == OFF:
            return ()
        briefs = TIER_PLANS["anthropic"][tier].deep_briefs
        if not briefs:
            return ()
        close_time = _as_utc(getattr(question, "close_time", None))
        if close_time is not None:
            minutes_left = (close_time - datetime.now(timezone.utc)).total_seconds() / 60
            if minutes_left < self._settings.deep_min_minutes_left:
                logger.info(
                    f"Standard research only for {question.page_url}: "
                    f"{minutes_left:.0f} minutes until it closes"
                )
                return ()
        if self._deep_started >= self._settings.deep_max_per_run:
            logger.info(
                f"Standard research only for {question.page_url}: "
                f"deep research cap of {self._settings.deep_max_per_run} per run reached"
            )
            return ()
        self._deep_started += 1
        return briefs

    async def _claude_search_call(
        self, question: MetaculusQuestion, llm: GeneralLlm, prompt: str, what: str
    ) -> tuple[str, dict]:
        """
        One Claude call that uses server-side web tools. Returns the text and
        {"ok": successful tool results, "errors": Counter of error codes}.

        Claude's web search is rate limited per organization, so only a few
        of these calls run at once, and a call whose searches all failed is
        repeated once after a pause.
        """
        if self._claude_search_semaphore is None:
            self._claude_search_semaphore = asyncio.Semaphore(
                self._settings.max_concurrent_claude_searches
            )
        totals = {"ok": 0, "errors": Counter()}
        text = ""
        async with self._claude_search_semaphore:
            for attempt in (1, 2):
                report = {"ok": 0, "errors": Counter()}
                _search_report.set(report)
                text = await llm.invoke(prompt)
                totals["ok"] += report["ok"]
                totals["errors"].update(report["errors"])
                if report["ok"] > 0 or not report["errors"] or attempt == 2:
                    break
                logger.warning(
                    f"Claude web search failed for {question.page_url} ({what}, "
                    f"{dict(report['errors'])}); retrying once in 30 seconds"
                )
                await asyncio.sleep(_SEARCH_RETRY_PAUSE_SECONDS)
        return text, totals

    async def _web_research(
        self, question: MetaculusQuestion, provider: str, max_searches: int
    ) -> str:
        llm = self._factory.web_researcher(provider, max_searches)
        prompt = self._web_research_prompt(question)
        if provider != "anthropic":
            return await llm.invoke(prompt)

        text, totals = await self._claude_search_call(question, llm, prompt, "briefing")
        notes = _question_notes.get()
        if notes is not None:
            notes["claude_searches_ok"] = totals["ok"]
            notes["claude_search_errors"] = dict(totals["errors"])
        if totals["ok"] == 0 and totals["errors"]:
            # A briefing written without any search result is the model's
            # memory presented as research, so it is left out.
            raise RuntimeError(
                f"All Claude web searches failed: {dict(totals['errors'])}"
            )
        return text

    async def _deep_research(self, question: MetaculusQuestion, brief_key: str) -> str:
        llm = self._factory.deep_researcher(brief_key)
        prompt = self._deep_research_prompt(question, brief_key)
        text, totals = await self._claude_search_call(question, llm, prompt, brief_key)
        notes = _question_notes.get()
        if notes is not None:
            notes["deep_tool_results_ok"] = notes.get("deep_tool_results_ok", 0) + totals["ok"]
            if totals["errors"]:
                errors = Counter(notes.get("deep_tool_errors", {}))
                errors.update(totals["errors"])
                notes["deep_tool_errors"] = dict(errors)
        if totals["ok"] == 0:
            # Without a single search or fetch result the text is the model's
            # memory presented as research, so it is left out.
            raise RuntimeError(
                f"Deep briefing '{brief_key}' got no web result: {dict(totals['errors'])}"
            )
        if len((text or "").strip()) < _MIN_DEEP_BRIEF_CHARS:
            raise RuntimeError(
                f"Deep briefing '{brief_key}' came back with {len((text or '').strip())} "
                "characters, probably cut off before it was written"
            )
        return text

    async def _asknews_research(self, question: MetaculusQuestion) -> str:
        if self._asknews_lock is None:
            self._asknews_lock = asyncio.Lock()
        # The free AskNews tier allows one call per 10 seconds, and the
        # searcher makes two calls, so questions take turns.
        async with self._asknews_lock:
            try:
                return await AskNewsSearcher().call_preconfigured_version(
                    "asknews/news-summaries", question.question_text
                )
            finally:
                await asyncio.sleep(12)

    async def _gather_research(
        self,
        question: MetaculusQuestion,
        plan: dict[str, str],
        deep_briefs: tuple[str, ...] = (),
    ) -> tuple[str, str, int]:
        """
        Run every research source at once. Returns (all research, the standard
        research without the deep briefings, number of deep briefings that
        succeeded).
        """
        # (label, coroutine, is a deep briefing)
        sources: list[tuple[str, Any, bool]] = []
        if self._settings.asknews_enabled:
            sources.append(
                ("News search (AskNews)", self._asknews_research(question), False)
            )
        for provider, tier in plan.items():
            if tier == OFF:
                continue
            max_searches = TIER_PLANS[provider][tier].web_searches
            if max_searches > 0:
                model = self._settings.providers[provider].mid
                sources.append(
                    (
                        f"Web research ({model})",
                        self._web_research(question, provider, max_searches),
                        False,
                    )
                )
        for brief_key in deep_briefs:
            model = self._settings.providers["anthropic"].strong
            sources.append(
                (
                    f"{DEEP_BRIEFS[brief_key].label} ({model})",
                    self._deep_research(question, brief_key),
                    True,
                )
            )
        if not sources:
            return "", "", 0
        results = await asyncio.gather(
            *(coroutine for _, coroutine, _ in sources), return_exceptions=True
        )
        sections: list[str] = []
        standard_sections: list[str] = []
        standard_total = sum(1 for _, _, deep in sources if not deep)
        deep_ok = 0
        for (label, _, deep), result in zip(sources, results):
            if isinstance(result, BaseException):
                # One failed source must not block the forecast.
                logger.warning(
                    f"Research source '{label}' failed for {question.page_url}: "
                    f"{result.__class__.__name__}: {result}"
                )
            elif result and result.strip():
                section = f"## {label}\n{result.strip()}"
                sections.append(section)
                if deep:
                    deep_ok += 1
                else:
                    standard_sections.append(section)
        notes = _question_notes.get()
        if notes is not None:
            notes["research_sources"] = f"{len(standard_sections)}/{standard_total}"
            if deep_briefs:
                notes["deep_briefs"] = f"{deep_ok}/{len(deep_briefs)}"
            notes["research_chars"] = sum(len(section) for section in sections)
        research = "\n\n".join(sections)
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research, "\n\n".join(standard_sections), deep_ok

    async def _ensemble_research(
        self, question: MetaculusQuestion, plan: dict[str, str]
    ) -> str:
        research, _, _ = await self._gather_research(question, plan)
        return research

    # --------------------------------------------------------- forecasting
    async def _run_slot(self, question: MetaculusQuestion, research: str, slot: Slot):
        _current_slot.set(slot)  # task-local: gather runs each slot in its own task
        prediction = await self._make_prediction(question, research)
        results = _slot_results.get()
        if results is not None:
            try:
                results.append(
                    {
                        "provider": slot.provider,
                        "model": slot.forecaster.model,
                        "shadow": slot.shadow,
                        "value": _jsonable_prediction(prediction.prediction_value),
                    }
                )
            except Exception as error:  # the log must never break a forecast
                logger.debug(f"Could not record a forecast for the log: {error}")
        prediction.reasoning = (
            f"*Forecaster model: {slot.forecaster.model}*\n\n{prediction.reasoning}"
        )
        return prediction

    def _shadow_slots(self, plan: dict[str, str]) -> list[Slot]:
        # Shadow forecasts only measure; they must never use credit that the
        # submitted forecasts will need later in the cycle. So they run only
        # while the provider's spend is close to the linear pace.
        slots = []
        for provider, tier in plan.items():
            if (
                tier != OFF
                and TIER_PLANS[provider][tier].shadow
                and self._ledger.within_pace(provider)
            ):
                slots.append(
                    Slot(
                        provider=provider,
                        forecaster=self._factory.forecaster(provider, "strong"),
                        parser=self._factory.parser(provider),
                        shadow=True,
                    )
                )
        return slots

    async def _run_shadow(
        self, question: MetaculusQuestion, standard_research: str, slots: list[Slot]
    ) -> int:
        """
        Forecast from the standard research alone, with the strong model of each
        provider. These forecasts are logged, never submitted: comparing them
        with the submitted forecasts of the same models after resolution shows
        what the deep briefings are worth. Returns how many succeeded.
        """
        results = await asyncio.gather(
            *(self._run_slot(question, standard_research, slot) for slot in slots),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        for failure in failures:
            logger.warning(
                f"Shadow forecast failed for {question.page_url}: "
                f"{failure.__class__.__name__}: {failure}"
            )
        return len(results) - len(failures)

    def _append_forecast_log(
        self, question: MetaculusQuestion, notes: dict, results: list
    ) -> None:
        entry = {
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": self.run_mode,
            "question_id": getattr(question, "id_of_question", None),
            "post_id": getattr(question, "id_of_post", None),
            "url": question.page_url,
            "type": type(question).__name__,
            "title": (question.question_text or "")[:200],
            "opened": notes.get("opened"),
            "closes": notes.get("closes"),
            "plan": notes.get("plan"),
            "research_sources": notes.get("research_sources"),
            "deep_briefs": notes.get("deep_briefs"),
            "research_chars": notes.get("research_chars"),
            "cost": notes.get("cost"),
            "submitted": [r for r in results if not r["shadow"]],
            "shadow": [r for r in results if r["shadow"]],
        }
        path = Path(self._settings.forecast_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    async def _research_and_make_predictions(self, question: MetaculusQuestion):  # type: ignore[override]
        if self._question_semaphore is None:
            self._question_semaphore = asyncio.Semaphore(
                self._settings.max_concurrent_questions
            )
        async with self._question_semaphore:
            notepad = await self._get_notepad(question)
            notepad.total_research_reports_attempted += 1

            plan = self._plan()
            slots = self._slots(plan)
            if not slots:
                raise RuntimeError(
                    "No provider is available: every provider is either missing "
                    f"its API key or over its cycle budget (plan: {plan})."
                )
            logger.info(f"Plan for {question.page_url}: {plan}")

            costs: dict[str, float] = defaultdict(float)
            notes: dict[str, Any] = {
                "url": question.page_url,
                "plan": dict(plan),
                # How long questions stay open decides how often the bot must
                # run, so the run summary shows both times.
                "opened": _short_time(getattr(question, "open_time", None)),
                "closes": _short_time(getattr(question, "close_time", None)),
            }
            slot_results: list = []
            _cost_sink.set(costs)  # inherited by the tasks started below
            _question_notes.set(notes)
            _slot_results.set(slot_results)
            valid_predictions: list = []
            shadow_task: asyncio.Task | None = None
            try:
                research, standard_research, deep_ok = await self._gather_research(
                    question, plan, self._deep_briefs_for(question, plan)
                )
                # Shadow forecasts only make sense when the submitted forecasts
                # saw something the standard research does not contain.
                if deep_ok > 0 and self._settings.shadow_forecasts:
                    shadow_slots = self._shadow_slots(plan)
                    if shadow_slots:
                        shadow_task = asyncio.create_task(
                            self._run_shadow(question, standard_research, shadow_slots)
                        )
                        notes["shadow"] = f"0/{len(shadow_slots)}"
                valid_predictions, errors, exception_group = (
                    await self._gather_results_and_exceptions(
                        [self._run_slot(question, research, s) for s in slots]
                    )
                )
                if shadow_task is not None:
                    try:
                        shadow_ok = await shadow_task
                        notes["shadow"] = f"{shadow_ok}/{notes['shadow'].split('/')[1]}"
                    except Exception as error:  # never let the shadow break a forecast
                        logger.warning(f"Shadow forecasts failed: {error}")
            finally:
                if shadow_task is not None and not shadow_task.done():
                    shadow_task.cancel()
                # Record spend even when the question fails, then persist it so
                # an interrupted run does not lose what was already spent.
                for provider, tier in plan.items():
                    if tier != OFF:
                        self._ledger.record(provider, tier, costs[provider])
                self._ledger.save()
                notes["forecasts"] = f"{len(valid_predictions)}/{len(slots)}"
                notes["cost"] = {name: round(value, 4) for name, value in costs.items()}
                self.diagnostics.append(notes)
                logger.info(f"Diagnostics: {notes}")
                if valid_predictions:
                    try:
                        self._append_forecast_log(question, notes, slot_results)
                    except Exception as error:  # the log must never break a forecast
                        logger.warning(f"Could not write the forecast log: {error}")

            if errors:
                logger.warning(f"Errors while predicting {question.page_url}: {errors}")
            if len(valid_predictions) == 0:
                assert exception_group, "Exception group should not be None"
                self._reraise_exception_with_prepended_message(
                    exception_group,
                    "Every forecast in the ensemble failed",
                )
            return ResearchWithPredictions(
                research_report=research,
                summary_report=_trim(research, 12000) or "No research was available.",
                errors=errors,
                predictions=valid_predictions,
            )


# -------------------------------------------------------------------- main


TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/futureeval/",
    "minibench": "https://www.metaculus.com/tournament/minibench/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def build_bot(settings: Settings, ledger: SpendLedger, publish: bool, skip_done: bool) -> EnsembleBot:
    first = next(iter(settings.providers.values()))
    factory = LlmFactory(settings)
    bot = EnsembleBot(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # the ensemble sets its own count
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_done,
        extra_metadata_in_explanation=True,
        enable_summarize_research=False,
        required_successful_predictions=0.0,
        llms={
            # Only used outside ensemble slots, which the bot never does in
            # normal operation. Set explicitly so no outdated default is picked.
            "default": factory.forecaster(first.name, "mid"),
            "summarizer": factory.parser(first.name),
            "researcher": "no_research",
            "parser": factory.parser(first.name),
        },
    )
    return bot.configure(settings, ledger)


async def run(mode: str, publish: bool, max_questions: int | None) -> int:
    settings = Settings.from_env()
    if not settings.providers:
        print("❌  Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set.")
        return 1
    ledger = SpendLedger(settings.ledger_path, settings.budgets)
    print(f"Providers with a key: {', '.join(settings.providers)}")
    print(f"AskNews: {'on' if settings.asknews_enabled else 'off (no credentials)'}")
    print(f"Spend before this run:\n{ledger.summary_markdown()}\n")

    skip_done = mode != "test_questions"
    bot = build_bot(settings, ledger, publish, skip_done)
    bot.run_mode = mode
    client = MetaculusClient()
    tournament_ids = {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "test_questions": ["bot-testing-area"],
    }[mode]

    reports: list[ForecastReport | BaseException] = []
    remaining = max_questions
    for tournament_id in tournament_ids:
        # The client method calls asyncio.run itself, so it runs in a thread.
        questions = await asyncio.to_thread(
            client.get_all_open_questions_from_tournament, tournament_id
        )
        if skip_done:
            questions = [q for q in questions if not q.already_forecasted]
        if remaining is not None:
            questions = questions[:remaining]
            remaining -= len(questions)
        if questions:
            reports += await bot.forecast_questions(questions, return_exceptions=True)

    ledger.save()
    bot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(
        reports, will_publish=publish, tournament_url=TOURNAMENT_URLS.get(mode)
    )
    failures = [r for r in reports if isinstance(r, BaseException)]
    summary = (
        f"### Forecast run ({mode})\n\n"
        f"Questions attempted: {len(reports)}, failed: {len(failures)}\n\n"
        f"{ledger.summary_markdown()}\n"
    )
    if bot.diagnostics:
        summary += (
            "\n| Question | Opened (UTC) | Closes (UTC) | Tiers | Research sources | "
            "Claude searches | Deep briefings | Shadow | Forecasts | Cost |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n"
        )
        for notes in bot.diagnostics:
            searches = f"{notes.get('claude_searches_ok', '-')} ok"
            if notes.get("claude_search_errors"):
                searches += f", errors {notes['claude_search_errors']}"
            cost = ", ".join(f"{k} ${v:.2f}" for k, v in notes.get("cost", {}).items())
            deep = notes.get("deep_briefs", "-")
            if notes.get("deep_tool_errors"):
                deep += f", errors {notes['deep_tool_errors']}"
            summary += (
                f"| {notes.get('url')} | {notes.get('opened', '-')} | "
                f"{notes.get('closes', '-')} | {notes.get('plan')} | "
                f"{notes.get('research_sources', '-')} | {searches} | "
                f"{deep} | {notes.get('shadow', '-')} | "
                f"{notes.get('forecasts', '-')} | {cost} |\n"
            )
    if mode != "test_questions":
        try:
            summary += await coverage_markdown(client, tournament_ids)
        except Exception as error:  # reporting must never fail the run
            logger.warning(f"Coverage check failed: {error}")
            summary += f"\nCoverage check failed: {type(error).__name__}\n"
    print(summary)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as file:
            file.write(summary)
    # Fail the workflow (GitHub then emails the repository owner) only when
    # most questions failed. Single failures are retried by the next run.
    return 1 if reports and len(failures) > len(reports) / 2 else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "minibench", "test_questions"],
        default="tournament",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Forecast without submitting anything to Metaculus",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Stop after this many questions (limits the cost of a trial run)",
    )
    args = parser.parse_args()

    check_environment(strict=True)
    publish = not args.no_publish
    print_startup_banner(args.mode, will_publish=publish)
    sys.exit(asyncio.run(run(args.mode, publish, args.max_questions)))


if __name__ == "__main__":
    main()
