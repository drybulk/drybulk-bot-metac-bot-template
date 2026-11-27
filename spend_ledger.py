"""
Spend ledger and pacing for the ensemble bot.

Each provider (anthropic, openai) has a monthly credit that does not roll over.
The ledger records what the bot spent in the current billing cycle and decides,
per question, how much of the ensemble each provider can afford:

    full     -> spend is at or below the linear pace through the cycle (+ burst)
    reduced  -> above the "full" pace but below a looser pace
    minimal  -> above both paces but still below the cycle budget
    off      -> the cycle budget is used up; the provider is skipped

Pacing is linear in time, so it needs no forecast of how many questions will
open. The ledger is a JSON file that the GitHub Actions workflow commits back
to the repository after every run, because each run starts on a fresh machine.
"""

from __future__ import annotations

import calendar
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

TIERS = ("full", "reduced", "minimal")
OFF = "off"


@dataclass(frozen=True)
class ProviderBudget:
    name: str
    budget_usd: float  # spend ceiling per billing cycle
    reset_day: int = 1  # day of month on which the credit resets
    full_burst: float = 0.25  # "full" allowed up to (elapsed + burst) of budget
    reduced_burst: float = 0.45


def cycle_bounds(today: date, reset_day: int) -> tuple[date, date]:
    """Return [start, next_start) of the billing cycle that contains `today`."""

    def clamp(year: int, month: int) -> date:
        return date(year, month, min(reset_day, calendar.monthrange(year, month)[1]))

    start = clamp(today.year, today.month)
    if today < start:
        year, month = (
            (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        )
        start = clamp(year, month)
    year, month = (
        (start.year, start.month + 1) if start.month < 12 else (start.year + 1, 1)
    )
    return start, clamp(year, month)


class SpendLedger:
    def __init__(
        self,
        path: str | os.PathLike,
        budgets: dict[str, ProviderBudget],
        now: datetime | None = None,
    ) -> None:
        self.path = Path(path)
        self.budgets = budgets
        self._fixed_now = now
        self.data: dict = {"version": 1, "cycles": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict) and isinstance(loaded.get("cycles"), dict):
                    self.data = loaded
            except (json.JSONDecodeError, OSError) as error:
                # A broken ledger must not stop forecasting. Starting from zero
                # can overspend the pace, but the provider-side credit limit
                # still applies.
                logger.error(f"Could not read spend ledger {self.path}: {error}")

    # ------------------------------------------------------------------ time
    def _now(self) -> datetime:
        return self._fixed_now or datetime.now(timezone.utc)

    def _cycle(self, provider: str) -> tuple[str, float]:
        """Return (cycle key, elapsed fraction of the cycle in [0, 1])."""
        now = self._now()
        start, end = cycle_bounds(now.date(), self.budgets[provider].reset_day)
        start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
        end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
        elapsed = (now - start_dt) / (end_dt - start_dt)
        return start.isoformat(), min(max(elapsed, 0.0), 1.0)

    def _entry(self, provider: str) -> dict:
        key, _ = self._cycle(provider)
        cycles = self.data["cycles"].setdefault(provider, {})
        return cycles.setdefault(
            key, {"spend_usd": 0.0, "questions": 0, "tiers": {}}
        )

    # --------------------------------------------------------------- queries
    def spend(self, provider: str) -> float:
        return float(self._entry(provider)["spend_usd"])

    def choose_tier(self, provider: str) -> str:
        budget = self.budgets[provider]
        _, elapsed = self._cycle(provider)
        spent = self.spend(provider)
        if budget.budget_usd <= 0 or spent >= budget.budget_usd:
            return OFF
        if spent <= budget.budget_usd * min(1.0, elapsed + budget.full_burst):
            return "full"
        if spent <= budget.budget_usd * min(1.0, elapsed + budget.reduced_burst):
            return "reduced"
        return "minimal"

    def within_pace(self, provider: str, burst: float = 0.10) -> bool:
        """
        True while spend is at or below the linear pace plus a small burst.
        Stricter than the "full" tier, so optional extras (shadow forecasts)
        stop well before they could push the submitted ensemble down a tier.
        """
        budget = self.budgets[provider]
        _, elapsed = self._cycle(provider)
        if budget.budget_usd <= 0:
            return False
        return self.spend(provider) <= budget.budget_usd * min(1.0, elapsed + burst)

    # --------------------------------------------------------------- updates
    def record(self, provider: str, tier: str, cost_usd: float) -> None:
        """Record what one question cost at one provider."""
        entry = self._entry(provider)
        entry["spend_usd"] = round(entry["spend_usd"] + cost_usd, 6)
        entry["questions"] += 1
        stats = entry["tiers"].setdefault(tier, {"n": 0, "cost_usd": 0.0})
        stats["n"] += 1
        stats["cost_usd"] = round(stats["cost_usd"] + cost_usd, 6)
        entry["updated_at"] = self._now().isoformat(timespec="seconds")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.path)

    # --------------------------------------------------------------- reports
    def summary_markdown(self) -> str:
        lines = [
            "| Provider | Cycle start | Spent | Budget | Cycle elapsed | Next tier | Questions |",
            "|---|---|---|---|---|---|---|",
        ]
        for provider, budget in self.budgets.items():
            key, elapsed = self._cycle(provider)
            entry = self._entry(provider)
            lines.append(
                f"| {provider} | {key} | ${entry['spend_usd']:.2f} | "
                f"${budget.budget_usd:.2f} | {elapsed:.0%} | "
                f"{self.choose_tier(provider)} | {entry['questions']} |"
            )
        return "\n".join(lines)
