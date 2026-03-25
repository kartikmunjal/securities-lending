"""NYSE trading-calendar helpers built on pandas_market_calendars."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
import pandas_market_calendars as mcal


@lru_cache(maxsize=1)
def _nyse() -> mcal.MarketCalendar:
    return mcal.get_calendar("NYSE")


def trading_dates(start: str | date, end: str | date) -> list[date]:
    """Return list of NYSE trading dates between *start* and *end* (inclusive)."""
    cal = _nyse()
    sched = cal.schedule(start_date=str(start), end_date=str(end))
    return [d.date() for d in sched.index]


def prev_trading_date(d: date | str | None = None) -> date:
    """Return the most recent NYSE trading day on or before *d* (default: today)."""
    if d is None:
        d = date.today()
    d = pd.Timestamp(d).date()
    for offset in range(10):
        candidate = d - timedelta(days=offset)
        dates = trading_dates(candidate, candidate)
        if dates:
            return dates[0]
    raise RuntimeError("Could not find a trading date within 10 days of %s" % d)


def is_trading_day(d: date | str) -> bool:
    """Return True if *d* is an NYSE trading day."""
    d = pd.Timestamp(d).date()
    return bool(trading_dates(d, d))


def business_dates_in_range(start: str | date, end: str | date) -> pd.DatetimeIndex:
    """Alias returning a DatetimeIndex for use with pandas date_range comparisons."""
    return pd.DatetimeIndex(trading_dates(start, end))
