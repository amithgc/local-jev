"""Command line entry point: ``local-jev serve``."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser


def serve(args):
    import uvicorn

    from .api import create_app
    from .engine import Engine

    if args.db:
        os.environ["LOCAL_JEV_DB"] = args.db
    engine = Engine(default=args.model, warmup=not args.no_warmup)
    card = engine.models[engine.default_model]
    problem = card.unavailable()
    if problem:
        sys.exit(f"Model {card.name!r} cannot be loaded: {problem}")
    from .models import on_disk
    print(f"  model     {card.name}   [{card.backend}]   (local-jev models lists the rest)")
    if not on_disk(card, engine.models):
        print("  fetching  its weights from Hugging Face; this happens once")
    started = time.time()
    engine.backend(engine.default_model)
    print(f"  loaded    {time.time() - started:.1f}s" + ("" if args.no_warmup else " (including warm-up)"))
    url = f"http://{args.host}:{args.port}"
    print(f"  api       {url}/v1/systemone   (interactive docs: {url}/docs)")
    print(f"  ui        {url}" if args.ui else "  ui        off (add --ui for the browser portal)")
    print(f"  sdk       export TYPESAFE_BASE_URL={url} TYPESAFE_API_KEY=local")
    if args.open:
        target = url if args.ui else f"{url}/docs"
        threading.Timer(1.0, lambda: webbrowser.open(target)).start()
    uvicorn.run(create_app(engine, ui=args.ui), host=args.host, port=args.port, log_level="warning")


def models(_args):
    from .backends import BACKENDS
    from .models import default_model, discover, on_disk
    cards = discover()
    chosen = default_model(cards)
    print(f"  {'name':22s} {'backend':9s} {'priority':>8s}  status")
    for card in sorted(cards.values(), key=lambda c: -c.priority):
        status = card.unavailable() or ("ready" if on_disk(card, cards) else "ready (weights download on first use)")
        print(f"  {card.name:22s} {card.backend:9s} {card.priority:8d}  {status}{'   <- default' if card.name == chosen else ''}")
    print()
    for kind, text in BACKENDS.items():
        print(f"  {kind:9s} {text}")


def main():
    parser = argparse.ArgumentParser(prog="local-jev", description="A local, Jev-compatible System One server.")
    sub = parser.add_subparsers(dest="command")
    s = sub.add_parser("serve", help="start the API and the project UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--model", default=None, help="default model; any name from `local-jev models`. Requests can still pick another.")
    s.add_argument("--db", default=None, help="path to the projects database (used with --ui)")
    s.add_argument("--ui", action="store_true", help="also serve the browser portal (projects, test panel, model comparison)")
    s.add_argument("--open", action="store_true", help="open a browser: the portal with --ui, otherwise the API docs")
    s.add_argument("--no-warmup", action="store_true", help="skip answering throwaway questions when each model loads")
    s.set_defaults(fn=serve)
    sub.add_parser("models", help="list models, their backends and whether they can be loaded").set_defaults(fn=models)
    args = parser.parse_args()
    if not args.command:
        args = parser.parse_args(["serve"])
    args.fn(args)


if __name__ == "__main__":
    main()
