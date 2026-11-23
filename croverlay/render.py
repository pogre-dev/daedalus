"""Draw the overlay panel as an image.

Rendering is plain PIL so it can be tested (and screenshotted) without a
window; `overlay.py` only shows the image and routes clicks back through the
hit regions returned alongside it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from . import catalog
from .cycle import UNSEEN
from .match import Snapshot

Region = Tuple[int, int, int, int, str, str]       # x1, y1, x2, y2, action, argument

# Palette: dark panel, purple for elixir (the game's own colour), gold for
# "next", green for "certainly in hand".
BG = (16, 18, 24)
PANEL = (27, 31, 42)
PANEL_HI = (36, 41, 56)
TEXT = (232, 234, 240)
MUTED = (139, 147, 167)
FAINT = (78, 85, 102)
ELIXIR = (199, 84, 230)
ELIXIR_DIM = (70, 40, 86)
GOLD = (242, 193, 78)
GREEN = (74, 222, 128)
AMBER = (245, 158, 11)
RED = (248, 113, 113)
KIND_BG = {"troop": (40, 58, 88), "building": (78, 60, 40), "spell": (66, 46, 92)}

_FONT_FILES = {
    "regular": ("segoeui.ttf", "DejaVuSans.ttf", "Arial.ttf", "arial.ttf"),
    "bold": ("segoeuib.ttf", "seguisb.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"),
}
_FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"),
              Path("/usr/share/fonts/TTF"), Path("/Library/Fonts"), Path("/System/Library/Fonts")]


@lru_cache(maxsize=64)
def font(size: int, weight: str = "regular") -> ImageFont.ImageFont:
    for name in _FONT_FILES[weight]:
        for folder in _FONT_DIRS:
            path = folder / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:                                     # Pillow < 10.1
        return ImageFont.load_default()


def clock_text(t: float) -> str:
    """The game's own clock: time left in regulation, then in overtime."""
    if t < 180:
        left, prefix = 180 - t, ""
    else:
        left, prefix = max(0.0, 300 - t), "OT "
    m, s = divmod(int(left + 0.999), 60)
    return f"{prefix}{m}:{s:02d}"


class CardArt:
    """Optional card images from a folder (e.g. the bot's BuildABot assets)."""

    def __init__(self, folder: Optional[Path]):
        self.folder = Path(folder) if folder else None
        self._cache: Dict[Tuple[str, int, int], Optional[Image.Image]] = {}

    def _path(self, key: str) -> Optional[Path]:
        if not self.folder or not self.folder.is_dir():
            return None
        card = catalog.CARDS.get(key)
        stems = {key, key.replace("_", "-"), key.replace("_", ""),
                 catalog.squash(card.name) if card else key}
        for stem in stems:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                path = self.folder / f"{stem}{ext}"
                if path.exists():
                    return path
        return None

    def get(self, key: str, w: int, h: int) -> Optional[Image.Image]:
        cache_key = (key, w, h)
        if cache_key not in self._cache:
            path = self._path(key)
            image = None
            if path:
                try:
                    src = Image.open(path).convert("RGB")
                    scale = max(w / src.width, h / src.height)
                    src = src.resize((max(1, round(src.width * scale)), max(1, round(src.height * scale))),
                                     Image.LANCZOS)
                    left, top = (src.width - w) // 2, (src.height - h) // 2
                    image = src.crop((left, top, left + w, top + h))
                except OSError:
                    image = None
            self._cache[cache_key] = image
        return self._cache[cache_key]


@dataclass
class Layout:
    width: int = 300
    scale: float = 1.0

    def s(self, v: float) -> int:
        return int(round(v * self.scale))


class Renderer:
    def __init__(self, width: int = 300, scale: float = 1.0, art: Optional[CardArt] = None):
        self.L = Layout(width, scale)
        self.art = art or CardArt(None)

    # ------------------------------------------------------------ primitives

    def _text(self, d: ImageDraw.ImageDraw, xy, text, size, fill=TEXT, weight="regular", anchor="la"):
        d.text(xy, text, font=font(self.L.s(size), weight), fill=fill, anchor=anchor)

    def _fit(self, text: str, size: float, max_w: int, weight="bold") -> List[str]:
        """Wrap a card label onto at most two lines inside max_w."""
        return self._fit_sized(text, size, max_w, weight)[0]

    def _fit_sized(self, text: str, size: float, max_w: int, weight="bold") -> Tuple[List[str], float]:
        """Like `_fit`, shrinking the font up to ~40% before truncating."""
        for trial in (size, size * 0.9, size * 0.8, size * 0.72, size * 0.64):
            f = font(self.L.s(trial), weight)
            if f.getlength(text) <= max_w:
                return [text], trial
            words = text.split()
            for i in range(len(words) - 1, 0, -1):
                a, b = " ".join(words[:i]), " ".join(words[i:])
                if f.getlength(a) <= max_w and f.getlength(b) <= max_w:
                    return [a, b], trial
        f = font(self.L.s(size * 0.64), weight)
        out = text
        while out and f.getlength(out + "…") > max_w:
            out = out[:-1]
        return [out + "…"], size * 0.64

    def _badge(self, d, cx, cy, r, text, fill, fg=(255, 255, 255)):
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=BG, width=max(1, self.L.s(1.5)))
        self._text(d, (cx, cy + self.L.s(0.5)), text, r * 1.25 / self.L.scale, fg, "bold", "mm")

    def card(self, img: Image.Image, box, key: Optional[str], *, border=None, dim=False,
             note: Optional[str] = None, note_fill=MUTED, guess: Optional[Tuple[str, float]] = None,
             badge: Optional[Tuple[str, tuple]] = None) -> None:
        """One card tile. `key=None` draws an unseen card (with an optional guess)."""
        d = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        radius = self.L.s(7)
        info = catalog.CARDS.get(key) if key else None
        if info is None:
            d.rounded_rectangle(box, radius, fill=PANEL, outline=FAINT, width=1)
            if guess:
                g = catalog.CARDS[guess[0]]
                lines = self._fit(g.label, 11, w - self.L.s(8), "regular")
                cy = y1 + h * 0.42 - (len(lines) - 1) * self.L.s(7)
                for i, line in enumerate(lines):
                    self._text(d, ((x1 + x2) / 2, cy + i * self.L.s(14)), line, 11, MUTED, "regular", "mm")
                self._text(d, ((x1 + x2) / 2, y2 - self.L.s(12)), f"likely {guess[1]:.0%}", 9, FAINT, "regular", "mm")
            else:
                self._text(d, ((x1 + x2) / 2, (y1 + y2) / 2), "?", 26, FAINT, "bold", "mm")
            return

        art = self.art.get(key, w, h)
        if art is not None:
            tile = art if not dim else Image.blend(art, Image.new("RGB", art.size, BG), 0.55)
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius, fill=255)
            img.paste(tile, (x1, y1), mask)
            # Name strip at the bottom so the card is readable at a glance.
            strip = self.L.s(18)
            d.rectangle((x1 + 1, y2 - strip, x2 - 1, y2 - 1), fill=(10, 12, 16))
            label = self._fit(info.label, 10, w - self.L.s(6))[0]
            self._text(d, ((x1 + x2) / 2, y2 - strip / 2), label, 10, TEXT if not dim else MUTED, "bold", "mm")
        else:
            bg = KIND_BG[info.kind]
            if dim:
                bg = tuple(int(c * 0.55 + b * 0.45) for c, b in zip(bg, BG))
            d.rounded_rectangle(box, radius, fill=bg)
            lines, size = self._fit_sized(info.label, 13 if w > self.L.s(56) else 11, w - self.L.s(8))
            cy = (y1 + y2) / 2 + self.L.s(4) - (len(lines) - 1) * self.L.s(size * 0.62)
            for i, line in enumerate(lines):
                self._text(d, ((x1 + x2) / 2, cy + i * self.L.s(size * 1.25)), line, size,
                           TEXT if not dim else MUTED, "bold", "mm")
        if border:
            d.rounded_rectangle(box, radius, outline=border, width=self.L.s(2.5))
        r = self.L.s(10 if w > self.L.s(56) else 8)
        self._badge(d, x1 + r + self.L.s(2), y1 + r + self.L.s(2), r, str(info.cost),
                    ELIXIR if not dim else ELIXIR_DIM)
        if badge:
            text, fill = badge
            br = self.L.s(9)
            cx, cy = x2 - br - self.L.s(2), y1 + br + self.L.s(2)
            if text == "check":
                self._badge(d, cx, cy, br, "", fill, BG)
                k = br * 0.5
                d.line([(cx - k, cy), (cx - k * 0.2, cy + k * 0.7), (cx + k, cy - k * 0.6)],
                       fill=BG, width=max(2, self.L.s(2.2)), joint="curve")
            else:
                self._badge(d, cx, cy, br, text, fill, BG)
        if note:
            self._text(d, ((x1 + x2) / 2, y2 + self.L.s(10)), note, 10, note_fill, "regular", "mm")

    def _section(self, d, y, title, right: str = "", right_fill=MUTED) -> int:
        pad = self.L.s(14)
        self._text(d, (pad, y), title, 10.5, MUTED, "bold")
        if right:
            self._text(d, (self.L.width - pad, y), right, 10.5, right_fill, "regular", "ra")
        return y + self.L.s(18)

    # ------------------------------------------------------------ the panel

    def render(self, snap: Optional[Snapshot], height: int, status: str = "",
               docked: bool = True) -> Tuple[Image.Image, List[Region]]:
        L = self.L
        W = L.width
        height = max(height, L.s(420))
        img = Image.new("RGB", (W, height), BG)
        d = ImageDraw.Draw(img)
        regions: List[Region] = []
        pad = L.s(14)

        # Header
        d.rectangle((0, 0, W, L.s(40)), fill=PANEL)
        self._text(d, (pad, L.s(20)), "OPPONENT", 12, TEXT, "bold", "lm")
        close = (W - L.s(32), L.s(8), W - L.s(8), L.s(32))
        self._text(d, ((close[0] + close[2]) / 2, (close[1] + close[3]) / 2), "×", 18, MUTED, "regular", "mm")
        regions.append((*close, "close", ""))
        if snap is not None:
            phase = {1: ("1× ELIXIR", MUTED), 2: ("2× ELIXIR", ELIXIR), 3: ("3× ELIXIR", RED)}[snap.multiplier]
            self._text(d, (W - L.s(40), L.s(20)), clock_text(snap.t), 12, TEXT, "bold", "rm")
            tw = font(L.s(12), "bold").getlength(clock_text(snap.t))
            self._text(d, (W - L.s(48) - tw, L.s(20)), phase[0], 10, phase[1], "bold", "rm")
        y = L.s(52)

        if snap is None:
            self._text(d, (W / 2, height / 2 - L.s(10)), "Waiting for a battle", 14, MUTED, "bold", "mm")
            self._text(d, (W / 2, height / 2 + L.s(14)), "starts when the arena is on screen", 10, FAINT, "regular", "mm")
            self._footer(d, img, height, status, regions)
            return img, regions

        # Elixir
        y = self._section(d, y, "ELIXIR (ESTIMATE)",
                          "" if snap.in_battle else "battle over", AMBER)
        value = snap.elixir
        self._text(d, (pad, y + L.s(2)), f"{value:.1f}", 44, ELIXIR, "bold", "la")
        num_w = font(L.s(44), "bold").getlength(f"{value:.1f}")
        info_x = pad + num_w + L.s(12)
        if value >= 9.95:
            line1, fill1 = "full - leaking", AMBER
        else:
            line1, fill1 = f"+1 in {snap.to_next:.1f}s", TEXT
        self._text(d, (info_x, y + L.s(10)), line1, 12, fill1, "bold")
        line2 = f"full in {snap.to_full:.0f}s" if value < 9.95 else "a play is likely due"
        self._text(d, (info_x, y + L.s(28)), line2, 10.5, MUTED)
        if snap.spent_10s:
            self._text(d, (info_x, y + L.s(44)), f"spent {snap.spent_10s:.0f} in last 10s", 10.5, MUTED)
        y += L.s(64)
        seg_gap = L.s(3)
        seg_w = (W - 2 * pad - 9 * seg_gap) / 10
        for i in range(10):
            sx = pad + i * (seg_w + seg_gap)
            box = (round(sx), y, round(sx + seg_w), y + L.s(12))
            d.rounded_rectangle(box, L.s(3), fill=ELIXIR_DIM)
            fill = max(0.0, min(1.0, value - i))
            if fill > 0:
                d.rounded_rectangle((box[0], box[1], round(sx + seg_w * fill), box[3]), L.s(3), fill=ELIXIR)
        y += L.s(26)

        # Hand
        view = snap.cycle
        guesses = list(snap.predicted)
        unseen = sum(1 for c in view.hand if c == UNSEEN)
        certain = view.hand_p >= 0.95 and unseen == 0
        if not view.plays:
            right = ""
        elif unseen == 4:
            right = "4 cards not seen yet"
        elif certain:
            right = "certain"
        elif view.hand_p >= 0.95:
            right = f"{unseen} not seen yet"
        else:
            right = f"{view.hand_p:.0%} likely"
        y = self._section(d, y, "HAND", right, GREEN if certain else MUTED)
        gap = L.s(8)
        tw = (W - 2 * pad - 3 * gap) // 4
        th = int(tw * 1.22)
        unseen_guesses = iter(g for g in guesses if g[0] not in view.hand)
        for i, card in enumerate(view.hand[:4]):
            box = (pad + i * (tw + gap), y, pad + i * (tw + gap) + tw, y + th)
            if card == UNSEEN:
                self.card(img, box, None, guess=next(unseen_guesses, None))
                continue
            st = view.statuses.get(card)
            p = st.p_hand if st else 0.0
            onfield = snap.champion_on_field == card
            self.card(img, box, card, border=GREEN if p >= 0.95 else None, dim=p < 0.6 or onfield,
                      note=("on field" if onfield else (None if p >= 0.95 else f"{p:.0%}")),
                      note_fill=AMBER)
            regions.append((*box, "play", card))
        y += th + L.s(22)

        # Next + the rest of the queue
        y = self._section(d, y, "NEXT", f"{view.next_p:.0%}" if view.plays and view.next_p < 0.95 else "")
        nw, nh = int(tw * 1.0), int(tw * 1.22)
        nbox = (pad, y, pad + nw, y + nh)
        if view.next_card == UNSEEN:
            self.card(img, nbox, None, guess=next(unseen_guesses, None))
        else:
            self.card(img, nbox, view.next_card, border=GOLD)
            regions.append((*nbox, "play", view.next_card))
        qx = pad + nw + L.s(16)
        self._text(d, (qx, y + L.s(2)), "then", 10, MUTED)
        qw = (W - pad - qx - 2 * L.s(6)) // 3
        qh = int(qw * 1.22)
        for i, card in enumerate(view.queue[1:4]):
            box = (qx + i * (qw + L.s(6)), y + L.s(18), qx + i * (qw + L.s(6)) + qw, y + L.s(18) + qh)
            self.card(img, box, None if card == UNSEEN else card, dim=True)
        y += max(nh, L.s(18) + qh) + L.s(16)

        # Deck
        known = list(view.deck)
        y = self._section(d, y, "DECK", f"{len(known)}/8 seen")
        dw = (W - 2 * pad - 3 * L.s(6)) // 4
        dh = int(dw * 1.1)
        fills = known + [None] * (8 - len(known))
        spare = iter(g for g in guesses if g[0] not in known)
        for i, card in enumerate(fills[:8]):
            col, row = i % 4, i // 4
            box = (pad + col * (dw + L.s(6)), y + row * (dh + L.s(22)),
                   pad + col * (dw + L.s(6)) + dw, y + row * (dh + L.s(22)) + dh)
            if card is None:
                guess = next(spare, None)
                self.card(img, box, None, guess=guess)
                if guess:
                    regions.append((*box, "play", guess[0]))
                continue
            st = view.statuses[card]
            back = st.plays_until_hand
            if st.p_hand >= 0.95:
                badge, note, fill = ("check", GREEN), "in hand", GREEN
            elif st.p_next >= 0.95:
                badge, note, fill = ("N", GOLD), "next", GOLD
            elif back:
                badge, note, fill = (str(back), MUTED), f"back in {back}", MUTED
            else:
                badge, note, fill = None, f"{st.p_hand:.0%} hand", MUTED
            self.card(img, box, card, dim=st.p_hand < 0.5, badge=badge, note=note, note_fill=fill)
            regions.append((*box, "play", card))
        y += 2 * (dh + L.s(22)) + L.s(4)

        # Recent plays
        footer_top = height - L.s(64)
        if y + L.s(40) < footer_top:
            y = self._section(d, y, "PLAYS", f"{view.plays} total" + (f" · {view.unknown_plays} unseen" if view.unknown_plays else ""))
            row_h = L.s(19)
            rows = max(0, (footer_top - y - L.s(6)) // row_h)
            for entry in list(reversed(snap.log))[:rows]:
                info = catalog.CARDS.get(entry.card)
                name = info.name if info else entry.card
                mark = "~ " if entry.confidence < 0.5 and not entry.manual else ""
                fill = GOLD if entry.manual else TEXT if entry.confidence >= 0.5 else MUTED
                if entry.manual:
                    name += " (you)"
                self._text(d, (pad, y), clock_text(entry.t), 10.5, FAINT)
                self._text(d, (pad + L.s(46), y), mark + name, 11, fill, "regular")
                self._text(d, (W - pad, y), f"−{entry.cost:g}", 11, ELIXIR, "bold", "ra")
                y += row_h

        self._footer(d, img, height, status, regions)
        return img, regions

    def _footer(self, d, img, height, status, regions) -> None:
        L = self.L
        W = L.width
        pad = L.s(14)
        top = height - L.s(58)
        d.rectangle((0, top, W, height), fill=PANEL)
        bw = (W - 2 * pad - L.s(8)) // 2
        for i, (label, action) in enumerate((("Undo last", "undo"), ("New battle", "reset"))):
            box = (pad + i * (bw + L.s(8)), top + L.s(8), pad + i * (bw + L.s(8)) + bw, top + L.s(32))
            d.rounded_rectangle(box, L.s(6), fill=PANEL_HI)
            self._text(d, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, 11, TEXT, "bold", "mm")
            regions.append((*box, action, ""))
        self._text(d, (pad, top + L.s(46)), status[:60], 9.5, FAINT, "regular", "lm")


def hit(regions: Sequence[Region], x: int, y: int) -> Optional[Tuple[str, str]]:
    for x1, y1, x2, y2, action, arg in regions:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return action, arg
    return None
