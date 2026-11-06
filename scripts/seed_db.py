"""Seed the finance warehouse with deterministic Faker data.

Uses asyncpg ``COPY`` (binary protocol) rather than row-by-row INSERTs: ~100k trades load in
about a second. Re-running is safe: tables are truncated first. A fixed seed makes the data
(and therefore the SQL eval's expected results) reproducible.

    python -m scripts.seed_db --trades 100000
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
import zlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import asyncpg
from faker import Faker

from enterprise_rag.config import get_settings

SEED = 42
REGIONS = ("AMER", "EMEA", "APAC", "LATAM")
REGION_WEIGHTS = (0.40, 0.30, 0.22, 0.08)
SEGMENTS = ("Retail", "Private Wealth", "Institutional", "Corporate")
SEGMENT_WEIGHTS = (0.45, 0.25, 0.20, 0.10)
CURRENCIES = ("USD", "EUR", "GBP", "JPY", "CHF")
CURRENCY_BY_REGION = {
    "AMER": ("USD",),
    "EMEA": ("EUR", "GBP", "CHF", "USD"),
    "APAC": ("JPY", "USD"),
    "LATAM": ("USD",),
}

# (ticker, name, asset_class, sector, base_price)
INSTRUMENTS: tuple[tuple[str, str, str, str, float], ...] = (
    ("AAPL", "Apple Inc.", "Equity", "Technology", 180.0),
    ("MSFT", "Microsoft Corp.", "Equity", "Technology", 350.0),
    ("NVDA", "NVIDIA Corp.", "Equity", "Technology", 450.0),
    ("GOOGL", "Alphabet Inc. Class A", "Equity", "Communication Services", 135.0),
    ("META", "Meta Platforms Inc.", "Equity", "Communication Services", 320.0),
    ("AMZN", "Amazon.com Inc.", "Equity", "Consumer Discretionary", 140.0),
    ("TSLA", "Tesla Inc.", "Equity", "Consumer Discretionary", 230.0),
    ("HD", "Home Depot Inc.", "Equity", "Consumer Discretionary", 310.0),
    ("KO", "Coca-Cola Co.", "Equity", "Consumer Staples", 60.0),
    ("PG", "Procter & Gamble Co.", "Equity", "Consumer Staples", 150.0),
    ("WMT", "Walmart Inc.", "Equity", "Consumer Staples", 160.0),
    ("JPM", "JPMorgan Chase & Co.", "Equity", "Financials", 150.0),
    ("GS", "Goldman Sachs Group Inc.", "Equity", "Financials", 340.0),
    ("BAC", "Bank of America Corp.", "Equity", "Financials", 30.0),
    ("V", "Visa Inc.", "Equity", "Financials", 240.0),
    ("JNJ", "Johnson & Johnson", "Equity", "Health Care", 160.0),
    ("PFE", "Pfizer Inc.", "Equity", "Health Care", 35.0),
    ("UNH", "UnitedHealth Group Inc.", "Equity", "Health Care", 500.0),
    ("XOM", "Exxon Mobil Corp.", "Equity", "Energy", 105.0),
    ("CVX", "Chevron Corp.", "Equity", "Energy", 155.0),
    ("CAT", "Caterpillar Inc.", "Equity", "Industrials", 260.0),
    ("BA", "Boeing Co.", "Equity", "Industrials", 210.0),
    ("NEE", "NextEra Energy Inc.", "Equity", "Utilities", 65.0),
    ("AMT", "American Tower Corp.", "Equity", "Real Estate", 190.0),
    ("LIN", "Linde plc", "Equity", "Materials", 380.0),
    ("SAP", "SAP SE", "Equity", "Technology", 140.0),
    ("ASML", "ASML Holding NV", "Equity", "Technology", 650.0),
    ("NESN", "Nestle SA", "Equity", "Consumer Staples", 105.0),
    ("HSBA", "HSBC Holdings plc", "Equity", "Financials", 6.5),
    ("7203", "Toyota Motor Corp.", "Equity", "Consumer Discretionary", 2600.0),
    ("SPY", "SPDR S&P 500 ETF Trust", "ETF", "Broad Market", 450.0),
    ("QQQ", "Invesco QQQ Trust", "ETF", "Broad Market", 380.0),
    ("IWM", "iShares Russell 2000 ETF", "ETF", "Broad Market", 190.0),
    ("EFA", "iShares MSCI EAFE ETF", "ETF", "International", 72.0),
    ("EEM", "iShares MSCI Emerging Markets ETF", "ETF", "International", 40.0),
    ("XLF", "Financial Select Sector SPDR", "ETF", "Financials", 36.0),
    ("XLE", "Energy Select Sector SPDR", "ETF", "Energy", 88.0),
    ("TLT", "iShares 20+ Year Treasury Bond ETF", "Bond", "Government", 95.0),
    ("IEF", "iShares 7-10 Year Treasury Bond ETF", "Bond", "Government", 95.0),
    ("LQD", "iShares iBoxx IG Corporate Bond ETF", "Bond", "Corporate Credit", 108.0),
    ("HYG", "iShares iBoxx High Yield Corporate Bond ETF", "Bond", "Corporate Credit", 76.0),
    ("BND", "Vanguard Total Bond Market ETF", "Bond", "Aggregate", 72.0),
    ("GLD", "SPDR Gold Shares", "Commodity", "Precious Metals", 185.0),
    ("SLV", "iShares Silver Trust", "Commodity", "Precious Metals", 22.0),
    ("USO", "United States Oil Fund", "Commodity", "Energy", 72.0),
    ("DBA", "Invesco DB Agriculture Fund", "Commodity", "Agriculture", 21.0),
)

START = datetime(2022, 1, 1, tzinfo=UTC)
TRADE_START = datetime(2023, 1, 1, tzinfo=UTC)
END = datetime(2025, 6, 30, 23, 59, tzinfo=UTC)


def _rand_dt(rng: random.Random, start: datetime, end: datetime) -> datetime:
    span = (end - start).total_seconds()
    return start + timedelta(seconds=rng.random() * span)


def build_data(
    n_clients: int, n_trades: int
) -> tuple[list[tuple], list[tuple], list[tuple], list[tuple]]:  # type: ignore[type-arg]
    rng = random.Random(SEED)
    fake = Faker()
    Faker.seed(SEED)

    instruments = [(t, n, ac, s) for t, n, ac, s, _ in INSTRUMENTS]
    base_price = {t: p for t, _, _, _, p in INSTRUMENTS}
    # Popularity is skewed (Zipf-ish) so "most traded" questions have clear answers.
    tickers = [t for t, *_ in INSTRUMENTS]
    weights = [1.0 / (i + 1) ** 0.8 for i in range(len(tickers))]

    clients: list[tuple] = []  # type: ignore[type-arg]
    for cid in range(1, n_clients + 1):
        region = rng.choices(REGIONS, REGION_WEIGHTS)[0]
        segment = rng.choices(SEGMENTS, SEGMENT_WEIGHTS)[0]
        name = fake.company() if segment in ("Institutional", "Corporate") else fake.name()
        clients.append((cid, name, region, segment, _rand_dt(rng, START, TRADE_START)))

    accounts: list[tuple] = []  # type: ignore[type-arg]
    aid = 0
    for cid, _, region, _, onboarded in clients:
        for _ in range(rng.choices((1, 2, 3), (0.55, 0.35, 0.10))[0]):
            aid += 1
            ccy = rng.choice(CURRENCY_BY_REGION[region])
            opened = _rand_dt(rng, onboarded, onboarded + timedelta(days=180))
            accounts.append((aid, cid, ccy, opened))

    # Account activity is also skewed: a minority of accounts trade a lot.
    acct_weights = [rng.paretovariate(1.5) for _ in accounts]
    trades: list[tuple] = []  # type: ignore[type-arg]
    chosen_accounts = rng.choices(accounts, acct_weights, k=n_trades)
    chosen_tickers = rng.choices(tickers, weights, k=n_trades)
    total_days = (END - TRADE_START).days
    for tid, (acct, ticker) in enumerate(zip(chosen_accounts, chosen_tickers, strict=True), 1):
        traded_at = _rand_dt(rng, max(acct[3], TRADE_START), END)
        # Deterministic (crc32, not hash()) drift + noise so prices look like a trend over time.
        t = (traded_at - TRADE_START).days / total_days
        drift = 1.0 + 0.25 * t * (1 if zlib.crc32(ticker.encode()) % 3 else -0.5)
        price = base_price[ticker] * drift * rng.uniform(0.95, 1.05)
        qty = max(1, int(rng.lognormvariate(4.0, 1.2)))
        side = "BUY" if rng.random() < 0.54 else "SELL"
        trades.append((tid, acct[0], ticker, side, qty, Decimal(f"{price:.4f}"), traded_at))
    return instruments, clients, accounts, trades


async def seed(n_clients: int, n_trades: int) -> None:
    dsn = get_settings().database_url.replace("+asyncpg", "")
    t0 = time.perf_counter()
    instruments, clients, accounts, trades = build_data(n_clients, n_trades)
    t_gen = time.perf_counter() - t0

    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute("TRUNCATE trades, accounts, clients, instruments CASCADE")
            await conn.copy_records_to_table(
                "instruments",
                records=instruments,
                columns=["ticker", "name", "asset_class", "sector"],
            )
            await conn.copy_records_to_table(
                "clients",
                records=clients,
                columns=["id", "name", "region", "segment", "onboarded_at"],
            )
            await conn.copy_records_to_table(
                "accounts",
                records=accounts,
                columns=["id", "client_id", "currency", "opened_at"],
            )
            await conn.copy_records_to_table(
                "trades",
                records=trades,
                columns=["id", "account_id", "ticker", "side", "qty", "price", "traded_at"],
            )
        # Fresh statistics so the planner picks the indexes immediately.
        await conn.execute("ANALYZE instruments, clients, accounts, trades")
    finally:
        await conn.close()
    total = time.perf_counter() - t0
    print(
        f"seeded instruments={len(instruments)} clients={len(clients)} "
        f"accounts={len(accounts)} trades={len(trades)} "
        f"(generate {t_gen:.2f}s, total {total:.2f}s)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clients", type=int, default=2000)
    p.add_argument("--trades", type=int, default=100_000)
    args = p.parse_args()
    asyncio.run(seed(args.clients, args.trades))


if __name__ == "__main__":
    main()
