"""Command-line entry point: ``ai-harness serve|ask|demo|eval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    """Make the repo-root ``evals`` and ``scripts`` packages importable when the
    installed console script is invoked from an arbitrary working directory."""
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "evals").is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ai_harness.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from ai_harness.pipeline import Harness
    from ai_harness.schemas import ChatRequest

    harness = Harness()
    response = harness.run(ChatRequest(query=args.query, session_id=args.session))
    print(json.dumps(response.model_dump(), indent=2, default=str))
    return 0


def _cmd_demo(_: argparse.Namespace) -> int:
    _ensure_repo_on_path()
    from scripts.demo import main as demo_main

    return demo_main()


def _cmd_eval(_: argparse.Namespace) -> int:
    _ensure_repo_on_path()
    from evals.run_evals import main as eval_main

    return eval_main()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-harness", description="Production AI harness CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the FastAPI server.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    ask = sub.add_parser("ask", help="Send a single query through the harness.")
    ask.add_argument("query")
    ask.add_argument("--session", default="cli")
    ask.set_defaults(func=_cmd_ask)

    demo = sub.add_parser("demo", help="Run the scripted demo (generates traces).")
    demo.set_defaults(func=_cmd_demo)

    ev = sub.add_parser("eval", help="Run the offline evaluation harness.")
    ev.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
