"""Clash Royale opponent overlay.

    python -m croverlay                       watch the game window (default)
    python -m croverlay demo                  scripted opponent, no game needed
    python -m croverlay replay TRACKS.jsonl   replay Solar-CR recorded detections
    python -m croverlay check                 what was found: windows, model, fonts
    python -m croverlay shot TRACKS.jsonl --at 90 out.png    render one frame to a PNG
    python -m croverlay.evaluate TRACKS_DIR   score play extraction on recordings

Common options (see --help on each):
    --my-deck hog_rider musketeer ...   your own 8 cards: sharpens whose-play-is-whose
    --bot-dir PATH                      Solar-CR checkout (for its detector weights)
    --window TITLE                      which window, when several emulators are open
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import catalog, detector, render
from .match import Session
from .pipeline import LiveFeed, Pipeline, TracksFeed


def _deck(names: Sequence[str]) -> List[str]:
    out, bad = [], []
    for name in names:
        key = catalog.from_any(name)
        (out if key else bad).append(key or name)
    if bad:
        raise SystemExit(f"unknown card(s) in --my-deck: {', '.join(bad)}")
    if out and len(out) != 8:
        print(f"note: --my-deck has {len(out)} cards, not 8", file=sys.stderr)
    return out


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--my-deck", nargs="*", default=[], metavar="CARD",
                   help="your own eight cards (names like 'hog rider', 'the_log', 'Skarmy')")
    p.add_argument("--cards-dir", type=Path, help="folder of card images named by card")
    p.add_argument("--width", type=int, default=300, help="panel width in pixels at 100%% scale")
    p.add_argument("--left", action="store_true", help="dock on the left of the window")
    p.add_argument("--no-dock", action="store_true", help="free-floating panel you can drag")
    p.add_argument("--height", type=int, default=900, help="panel height when not docked")
    p.add_argument("--alpha", type=float, default=0.96, help="panel opacity 0.3-1")
    p.add_argument("--intro-delay", type=float, default=0.0,
                   help="seconds between the arena appearing and elixir starting")


def _art(args, bot_dir: Optional[Path]) -> render.CardArt:
    folder = args.cards_dir
    if folder is None and bot_dir is not None:
        guess = bot_dir / "vendor" / "ClashRoyaleBuildABot" / "clashroyalebuildabot" / "images" / "cards"
        folder = guess if guess.is_dir() else None
    return render.CardArt(folder)


def _show(args, session: Session, feed, bot_dir: Optional[Path] = None, dock: bool = True) -> int:
    from . import winapi
    from .overlay import Overlay
    winapi.set_dpi_aware()
    pipeline = Pipeline(feed, session, max_fps=getattr(args, "max_fps", 30.0))
    pipeline.start()
    try:
        ui = Overlay(session, pipeline, _art(args, bot_dir), width=args.width,
                     prefer_left=args.left, dock=dock and not args.no_dock,
                     height=args.height, alpha=max(0.3, min(1.0, args.alpha)))
    except ImportError as exc:
        raise SystemExit(f"tkinter is needed for the window ({exc}). On Windows it ships with "
                         "the python.org installer; on Linux install python3-tk.")
    ui.run()
    pipeline.stop()
    return 0


def cmd_run(args) -> int:
    bot_dir = args.bot_dir or detector.guess_bot_dir()
    weights = args.weights or detector.find_weights(bot_dir)
    if weights is None:
        raise SystemExit("no detector weights found. Pass --weights PATH (a YOLO .pt), or "
                         "--bot-dir pointing at your Solar-CR checkout (it looks in "
                         "tmp/yolo/runs/*/weights/best.pt and vendor/KataCR/runs).")
    side_model = None if args.no_side_model else (args.side_model or detector.find_side_model(bot_dir))
    print(f"detector: {weights}")
    det = detector.Detector(weights, side_model, confidence=args.conf, imgsz=args.imgsz,
                            device=args.device)
    print(f"side classifier: {side_model if det.sides else 'not used' + (' (' + det.side_error + ')' if det.side_error else '')}")
    if det.unknown_classes:
        print(f"note: {len(det.unknown_classes)} detector classes have no card entry: "
              + ", ".join(det.unknown_classes[:12]))

    from . import sources
    if args.source == "window":
        source = sources.WindowSource(args.window, args.hwnd, args.capture)
        print(f"watching {source.window}")
    elif args.source == "adb":
        source = sources.AdbSource(args.serial, args.adb)
    else:
        if not args.video:
            raise SystemExit("--source video needs --video PATH")
        source = sources.VideoSource(str(args.video))
    session = Session(_deck(args.my_deck), intro_delay_s=args.intro_delay)
    feed = LiveFeed(source, det)
    return _show(args, session, feed, bot_dir, dock=args.source == "window")


def cmd_demo(args) -> int:
    from .demo import DemoFeed
    deck = _deck(args.deck) if args.deck else None
    session = Session(_deck(args.my_deck))
    return _show(args, session, DemoFeed(deck, seed=args.seed, speed=args.speed), dock=False)


def cmd_replay(args) -> int:
    session = Session(_deck(args.my_deck), intro_delay_s=args.intro_delay)
    return _show(args, session, TracksFeed(args.tracks, speed=args.speed), dock=False)


def cmd_shot(args) -> int:
    """Render the panel at one moment of a recording, without opening a window."""
    from .evaluate import load_tracks
    from .match import Battle
    _, frames = load_tracks(args.tracks)
    battle = Battle(_deck(args.my_deck))
    for t, dets in frames:
        if t > args.at:
            break
        battle.update(t, dets)
    battle.t = max(battle.t, args.at)
    snap = battle.snapshot(status=f"{args.tracks.name} @ {args.at:.0f}s")
    image, _ = render.Renderer(args.width, args.width / 300.0, _art(args, None)).render(
        snap, int(args.height_px), snap.status)
    image.save(args.out)
    print(f"wrote {args.out}")
    return 0


def cmd_check(args) -> int:
    from . import winapi
    bot_dir = args.bot_dir or detector.guess_bot_dir()
    print(f"python        {sys.version.split()[0]} on {sys.platform}")
    print(f"bot dir       {bot_dir or 'not found (pass --bot-dir)'}")
    print(f"weights       {detector.find_weights(bot_dir) or 'not found'}")
    print(f"side model    {detector.find_side_model(bot_dir) or 'not found (optional)'}")
    for mod in ("tkinter", "ultralytics", "torch", "cv2"):
        try:
            __import__(mod)
            print(f"{mod:<13} ok")
        except Exception as exc:
            print(f"{mod:<13} missing ({type(exc).__name__})")
    print(f"font          {render.font(12, 'bold').getname() if hasattr(render.font(12), 'getname') else '?'}")
    if winapi.IS_WINDOWS:
        wins = list(winapi.candidates())
        print(f"game windows  {len(wins)}")
        for w in wins:
            print(f"  {w}  client {winapi.client_rect(w.hwnd)[2:]}")
    else:
        print("game windows  (window capture is Windows-only; use --source adb or video)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "demo", "replay", "check", "shot"}
    if not argv or argv[0] not in commands and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")

    ap = argparse.ArgumentParser(prog="python -m croverlay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="watch the game (default)")
    _common(p)
    p.add_argument("--source", choices=("window", "adb", "video"), default="window")
    p.add_argument("--window", help="part of the game window's title (emulator instance name)")
    p.add_argument("--hwnd", type=int, help="exact window handle")
    p.add_argument("--capture", choices=("auto", "printwindow", "screen"), default="auto")
    p.add_argument("--serial", help="adb device serial, e.g. 127.0.0.1:16384 for MuMu")
    p.add_argument("--adb", default="adb", help="adb executable")
    p.add_argument("--video", type=Path)
    p.add_argument("--bot-dir", type=Path, help="Solar-CR checkout")
    p.add_argument("--weights", type=Path, help="YOLO weights (.pt)")
    p.add_argument("--side-model", type=Path)
    p.add_argument("--no-side-model", action="store_true")
    p.add_argument("--conf", type=float, default=0.35, help="detector confidence floor")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="", help="'0' for the first GPU, 'cpu' to force CPU")
    p.add_argument("--max-fps", type=float, default=15.0)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("demo", help="scripted opponent, no game needed")
    _common(p)
    p.add_argument("--deck", nargs="*", help="the scripted opponent's deck")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=1.0)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("replay", help="replay recorded detector tracks")
    _common(p)
    p.add_argument("tracks", type=Path)
    p.add_argument("--speed", type=float, default=1.0)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("shot", help="render the panel at one moment of a recording")
    _common(p)
    p.add_argument("tracks", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--at", type=float, default=90.0, help="match second")
    p.add_argument("--height-px", type=int, default=996)
    p.set_defaults(func=cmd_shot)

    p = sub.add_parser("check", help="report what the overlay can find on this machine")
    p.add_argument("--bot-dir", type=Path)
    p.set_defaults(func=cmd_check)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
