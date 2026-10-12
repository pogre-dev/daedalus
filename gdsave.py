"""Read and write Geometry Dash's CCLocalLevels.dat (Windows).

    python gdsave.py list
    python gdsave.py export N level.txt     decoded level string of level N -> file
    python gdsave.py import N level.txt     file -> level N (backs up the save first)
    python gdsave.py roundtrip N            decode + re-encode level N in place

Add --dry-run to import/roundtrip to do everything except write the save.
N is the index shown by `list` (0 = top of the in-game list).

Save file:    XOR 11 -> url-safe base64 -> gzip -> plist-like XML
Level string: url-safe base64 -> gzip -> "header;obj;obj;..."   (key k4 of each level)
"""

import argparse
import base64
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

SAVE = Path(os.environ.get("LOCALAPPDATA", "")) / "GeometryDash" / "CCLocalLevels.dat"
XOR_KEY = 11
TAG = re.compile(rb"<(/?)([a-z]+)\s*(/?)>")


# ---------- codecs ----------

def _unb64(data: bytes) -> bytes:
    return base64.urlsafe_b64decode(data + b"=" * (-len(data) % 4))


def decode_save(raw: bytes) -> bytes:
    if raw.lstrip().startswith(b"<?xml"):  # GD also accepts an unencoded save
        return raw
    data = bytes(b ^ XOR_KEY for b in raw).rstrip(b"\x00\x0b\r\n")
    return zlib.decompress(_unb64(data), 47)


def encode_save(xml: bytes) -> bytes:
    data = base64.urlsafe_b64encode(gzip.compress(xml))
    return bytes(b ^ XOR_KEY for b in data)


def decode_level(k4: bytes) -> str:
    if not k4.startswith(b"H4sI"):  # very old levels are stored uncompressed
        return k4.decode()
    return zlib.decompress(_unb64(k4), 47).decode()


def encode_level(level: str) -> bytes:
    return base64.urlsafe_b64encode(gzip.compress(level.encode()))


# ---------- plist ----------
# The XML is never re-serialised: we only record byte spans of values so a
# level string can be spliced in and everything else stays byte-identical.

def read_dict(xml: bytes, pos: int):
    """Parse a dict whose contents start at pos.

    Returns ({key: (tag, value_start, value_end)}, position after the closing tag).
    """
    entries = {}
    while True:
        m = TAG.search(xml, pos)
        closing, tag, selfclosing = m.group(1), m.group(2), m.group(3)
        if closing:
            return entries, m.end()
        if tag != b"k":
            raise ValueError(f"expected <k> at byte {m.start()}, got <{tag.decode()}>")
        key_end = xml.index(b"</k>", m.end())
        key = xml[m.end():key_end].decode()

        m = TAG.search(xml, key_end + 4)
        tag = m.group(2)
        if m.group(3):  # <t />, <d />
            entries[key] = (tag, m.end(), m.end())
            pos = m.end()
        elif tag in (b"d", b"dict"):
            _, pos = read_dict(xml, m.end())
            entries[key] = (tag, m.end(), xml.rindex(b"</", 0, pos))
        else:
            close = xml.index(b"</" + tag + b">", m.end())
            entries[key] = (tag, m.end(), close)
            pos = close + len(tag) + 3


def read_levels(xml: bytes):
    """Return a list of per-level dicts {key: (tag, start, end)}, in list order."""
    root, _ = read_dict(xml, xml.index(b"<dict>") + 6)
    _, start, _ = root["LLM_01"]
    entries, _ = read_dict(xml, start)
    levels = []
    for key, (tag, vstart, _) in entries.items():
        if key.startswith("k_") and tag == b"d":
            levels.append((int(key[2:]), read_dict(xml, vstart)[0]))
    return [lvl for _, lvl in sorted(levels, key=lambda p: p[0])]


def value(xml: bytes, level: dict, key: str, default: str = "") -> str:
    if key not in level:
        return default
    _, start, end = level[key]
    return xml[start:end].decode(errors="replace")


def replace_level_string(xml: bytes, level: dict, level_string: str) -> bytes:
    if "k4" not in level:
        raise SystemExit("That level has no level string (k4) - open and save it in the editor first.")
    _, start, end = level["k4"]
    return xml[:start] + encode_level(level_string) + xml[end:]


# ---------- file I/O ----------

def load() -> bytes:
    if not SAVE.exists():
        raise SystemExit(f"Save file not found: {SAVE}")
    return decode_save(SAVE.read_bytes())


def game_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq GeometryDash.exe", "/NH"],
                         capture_output=True, text=True).stdout
    return "GeometryDash.exe" in out


def write(xml: bytes):
    if game_running():
        raise SystemExit("Geometry Dash is running. Close it first - it rewrites "
                         "CCLocalLevels.dat on exit and would discard this change.")
    backup = SAVE.with_name(f"{SAVE.name}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(SAVE, backup)
    print(f"Backup: {backup}")
    SAVE.write_bytes(encode_save(xml))
    if decode_save(SAVE.read_bytes()) != xml:
        shutil.copy2(backup, SAVE)
        raise SystemExit("Verification failed after writing; restored the backup.")
    print(f"Wrote:  {SAVE}")


def pick(xml: bytes, index: int) -> dict:
    levels = read_levels(xml)
    if not 0 <= index < len(levels):
        raise SystemExit(f"No level {index}; valid range is 0-{len(levels) - 1}.")
    return levels[index]


def object_count(level_string: str) -> int:
    return max(len([o for o in level_string.split(";") if o]) - 1, 0)  # first entry is the header


# ---------- commands ----------

def cmd_list(args):
    xml = load()
    levels = read_levels(xml)
    print(f"{len(levels)} levels in {SAVE}\n")
    print(f"{'#':>4}  {'objects':>8}  {'k4 bytes':>9}  name")
    for i, level in enumerate(levels):
        size = level["k4"][2] - level["k4"][1] if "k4" in level else 0
        print(f"{i:>4}  {value(xml, level, 'k48', '?'):>8}  {size:>9}  {value(xml, level, 'k2')}")


def cmd_export(args):
    xml = load()
    level = pick(xml, args.index)
    level_string = decode_level(value(xml, level, "k4").encode())
    Path(args.file).write_text(level_string, encoding="utf-8")
    print(f"'{value(xml, level, 'k2')}': {object_count(level_string)} objects -> {args.file}")


def cmd_import(args):
    xml = load()
    level = pick(xml, args.index)
    level_string = Path(args.file).read_text(encoding="utf-8").strip()
    apply(xml, level, level_string, args.dry_run)


def cmd_roundtrip(args):
    xml = load()
    level = pick(xml, args.index)
    apply(xml, level, decode_level(value(xml, level, "k4").encode()), args.dry_run)


def apply(xml: bytes, level: dict, level_string: str, dry_run: bool):
    name = value(xml, level, "k2")
    new_xml = replace_level_string(xml, level, level_string)

    # Re-parse what we are about to write and confirm it decodes to what we meant.
    index = read_levels(xml).index(level)
    check = read_levels(new_xml)[index]
    if decode_level(value(new_xml, check, "k4").encode()) != level_string:
        raise SystemExit("Self-check failed: re-encoded level does not decode back. Nothing written.")
    if decode_save(encode_save(new_xml)) != new_xml:
        raise SystemExit("Self-check failed: save does not survive encoding. Nothing written.")

    print(f"'{name}': {object_count(level_string)} objects, self-check OK")
    if dry_run:
        print("Dry run - save file not touched.")
    else:
        write(new_xml)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)

    p = sub.add_parser("export")
    p.add_argument("index", type=int)
    p.add_argument("file")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import")
    p.add_argument("index", type=int)
    p.add_argument("file")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("roundtrip")
    p.add_argument("index", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_roundtrip)

    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")
    args.func(args)


if __name__ == "__main__":
    main()
