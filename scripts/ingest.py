"""Ingest every document under data/docs (idempotent: unchanged docs are skipped)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from enterprise_rag.config import get_settings
from enterprise_rag.container import Container


async def run(root: Path) -> None:
    c = Container.build(get_settings())
    try:
        r = await c.ingest.ingest_directory(root)
        print(
            f"seen={r.docs_seen} ingested={r.docs_ingested} skipped={r.docs_skipped} "
            f"chunks={r.chunks_written} embedded={r.embeddings_computed} "
            f"reused={r.embeddings_reused} errors={len(r.errors)} in {r.seconds:.1f}s"
        )
        for e in r.errors:
            print("  error:", e)
    finally:
        await c.aclose()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", default=get_settings().docs_dir)
    asyncio.run(run(Path(p.parse_args().path)))


if __name__ == "__main__":
    main()
