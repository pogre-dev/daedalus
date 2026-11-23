"""What every card costs, and what the detector sees when one is played.

Three vocabularies meet here and this module is the only place they are joined:

* **card keys** (`hog_rider`, `the_log`, `skeleton_army`) - what the overlay
  shows and what the cycle and elixir models count,
* **detector classes** (`hog-rider`, `skeleton`, `tesla-evolution`) - one per
  thing on the board, so a Skeleton Army is fifteen `skeleton` boxes and a
  Tombstone's output is *also* `skeleton` boxes,
* **game internal names** (`ice_golemite`, `barb_log`, `assassin`) - what the
  meta-deck statistics use.

Costs and unit counts come from the game's own card tables (RoyaleAPI's
cr-api-data dump, the same files Solar-CR's simulator reads), not from memory.
Cards newer than that dump are marked in `_NEWER` and are the ones to re-check
after a balance update.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

TROOP, BUILDING, SPELL = "troop", "building", "spell"


@dataclass(frozen=True)
class Card:
    key: str
    name: str
    cost: int
    kind: str                      # troop | building | spell
    short: str = ""                # label that fits on a small tile
    champion: bool = False
    # Units placed on the board by one play, by detector class. A card absent
    # from here places one unit whose class is the card key with dashes.
    units: Tuple[Tuple[str, int], ...] = ()

    @property
    def label(self) -> str:
        return self.short or self.name


# (key, display name, elixir, kind) - generated from cr-api-data cards.json.
_BASE = [
    ("knight", "Knight", 3, TROOP), ("archers", "Archers", 3, TROOP),
    ("goblins", "Goblins", 2, TROOP), ("giant", "Giant", 5, TROOP),
    ("pekka", "P.E.K.K.A", 7, TROOP), ("minions", "Minions", 3, TROOP),
    ("balloon", "Balloon", 5, TROOP), ("witch", "Witch", 5, TROOP),
    ("barbarians", "Barbarians", 5, TROOP), ("golem", "Golem", 8, TROOP),
    ("skeletons", "Skeletons", 1, TROOP), ("valkyrie", "Valkyrie", 4, TROOP),
    ("skeleton_army", "Skeleton Army", 3, TROOP), ("bomber", "Bomber", 2, TROOP),
    ("musketeer", "Musketeer", 4, TROOP), ("baby_dragon", "Baby Dragon", 4, TROOP),
    ("prince", "Prince", 5, TROOP), ("wizard", "Wizard", 5, TROOP),
    ("mini_pekka", "Mini P.E.K.K.A", 4, TROOP), ("spear_goblins", "Spear Goblins", 2, TROOP),
    ("giant_skeleton", "Giant Skeleton", 6, TROOP), ("hog_rider", "Hog Rider", 4, TROOP),
    ("minion_horde", "Minion Horde", 5, TROOP), ("ice_wizard", "Ice Wizard", 3, TROOP),
    ("royal_giant", "Royal Giant", 6, TROOP), ("guards", "Guards", 3, TROOP),
    ("princess", "Princess", 3, TROOP), ("dark_prince", "Dark Prince", 4, TROOP),
    ("three_musketeers", "Three Musketeers", 9, TROOP), ("lava_hound", "Lava Hound", 7, TROOP),
    ("ice_spirit", "Ice Spirit", 1, TROOP), ("fire_spirit", "Fire Spirit", 1, TROOP),
    ("miner", "Miner", 3, TROOP), ("sparky", "Sparky", 6, TROOP),
    ("bowler", "Bowler", 5, TROOP), ("lumberjack", "Lumberjack", 4, TROOP),
    ("battle_ram", "Battle Ram", 4, TROOP), ("inferno_dragon", "Inferno Dragon", 4, TROOP),
    ("ice_golem", "Ice Golem", 2, TROOP), ("mega_minion", "Mega Minion", 3, TROOP),
    ("dart_goblin", "Dart Goblin", 3, TROOP), ("goblin_gang", "Goblin Gang", 3, TROOP),
    ("electro_wizard", "Electro Wizard", 4, TROOP), ("elite_barbarians", "Elite Barbarians", 6, TROOP),
    ("hunter", "Hunter", 4, TROOP), ("executioner", "Executioner", 5, TROOP),
    ("bandit", "Bandit", 3, TROOP), ("royal_recruits", "Royal Recruits", 7, TROOP),
    ("night_witch", "Night Witch", 4, TROOP), ("bats", "Bats", 2, TROOP),
    ("royal_ghost", "Royal Ghost", 3, TROOP), ("ram_rider", "Ram Rider", 5, TROOP),
    ("zappies", "Zappies", 4, TROOP), ("rascals", "Rascals", 5, TROOP),
    ("cannon_cart", "Cannon Cart", 5, TROOP), ("mega_knight", "Mega Knight", 7, TROOP),
    ("skeleton_barrel", "Skeleton Barrel", 3, TROOP), ("flying_machine", "Flying Machine", 4, TROOP),
    ("wall_breakers", "Wall Breakers", 2, TROOP), ("royal_hogs", "Royal Hogs", 5, TROOP),
    ("goblin_giant", "Goblin Giant", 6, TROOP), ("fisherman", "Fisherman", 3, TROOP),
    ("magic_archer", "Magic Archer", 4, TROOP), ("electro_dragon", "Electro Dragon", 5, TROOP),
    ("firecracker", "Firecracker", 3, TROOP), ("mighty_miner", "Mighty Miner", 4, TROOP),
    ("elixir_golem", "Elixir Golem", 3, TROOP), ("battle_healer", "Battle Healer", 4, TROOP),
    ("skeleton_king", "Skeleton King", 4, TROOP), ("archer_queen", "Archer Queen", 5, TROOP),
    ("golden_knight", "Golden Knight", 4, TROOP), ("monk", "Monk", 5, TROOP),
    ("skeleton_dragons", "Skeleton Dragons", 4, TROOP), ("mother_witch", "Mother Witch", 4, TROOP),
    ("electro_spirit", "Electro Spirit", 1, TROOP), ("electro_giant", "Electro Giant", 7, TROOP),
    ("phoenix", "Phoenix", 4, TROOP), ("heal_spirit", "Heal Spirit", 1, TROOP),
    ("cannon", "Cannon", 3, BUILDING), ("goblin_hut", "Goblin Hut", 5, BUILDING),
    ("mortar", "Mortar", 4, BUILDING), ("inferno_tower", "Inferno Tower", 5, BUILDING),
    ("bomb_tower", "Bomb Tower", 4, BUILDING), ("barbarian_hut", "Barbarian Hut", 6, BUILDING),
    ("tesla", "Tesla", 4, BUILDING), ("elixir_collector", "Elixir Collector", 6, BUILDING),
    ("x_bow", "X-Bow", 6, BUILDING), ("tombstone", "Tombstone", 3, BUILDING),
    ("furnace", "Furnace", 4, BUILDING), ("goblin_cage", "Goblin Cage", 4, BUILDING),
    ("goblin_drill", "Goblin Drill", 4, BUILDING),
    ("fireball", "Fireball", 4, SPELL), ("arrows", "Arrows", 3, SPELL),
    ("rage", "Rage", 2, SPELL), ("rocket", "Rocket", 6, SPELL),
    ("goblin_barrel", "Goblin Barrel", 3, SPELL), ("freeze", "Freeze", 4, SPELL),
    ("mirror", "Mirror", 1, SPELL), ("lightning", "Lightning", 6, SPELL),
    ("zap", "Zap", 2, SPELL), ("poison", "Poison", 4, SPELL),
    ("graveyard", "Graveyard", 5, SPELL), ("the_log", "The Log", 2, SPELL),
    ("tornado", "Tornado", 3, SPELL), ("clone", "Clone", 3, SPELL),
    ("earthquake", "Earthquake", 3, SPELL), ("barbarian_barrel", "Barbarian Barrel", 2, SPELL),
    ("giant_snowball", "Giant Snowball", 2, SPELL), ("royal_delivery", "Royal Delivery", 3, SPELL),
]

# Released after the cr-api-data dump. Re-check these after a balance change.
_NEWER = [
    ("little_prince", "Little Prince", 3, TROOP), ("goblinstein", "Goblinstein", 5, TROOP),
    ("boss_bandit", "Boss Bandit", 6, TROOP), ("goblin_demolisher", "Goblin Demolisher", 4, TROOP),
    ("goblin_machine", "Goblin Machine", 5, TROOP), ("suspicious_bush", "Suspicious Bush", 2, TROOP),
    ("berserker", "Berserker", 2, TROOP), ("rune_giant", "Rune Giant", 4, TROOP),
    ("goblin_curse", "Goblin Curse", 2, SPELL), ("void", "Void", 3, SPELL),
]

CHAMPIONS = frozenset({
    "archer_queen", "golden_knight", "skeleton_king", "mighty_miner", "monk",
    "little_prince", "goblinstein", "boss_bandit",
})

# summon_number from cards_stats_troop.json. Goblin Gang and Rascals are two
# classes in one play.
_UNITS = {
    "archers": (("archer", 2),),
    "goblins": (("goblin", 4),),
    "minions": (("minion", 3),),
    "barbarians": (("barbarian", 5),),
    "skeletons": (("skeleton", 3),),
    "skeleton_army": (("skeleton", 15),),
    "spear_goblins": (("spear-goblin", 3),),
    "minion_horde": (("minion", 6),),
    "guards": (("guard", 3),),
    "three_musketeers": (("musketeer", 3),),
    "goblin_gang": (("goblin", 3), ("spear-goblin", 3)),
    "elite_barbarians": (("elite-barbarian", 2),),
    "royal_recruits": (("royal-recruit", 6),),
    "bats": (("bat", 5),),
    "zappies": (("zappy", 3),),
    "rascals": (("rascal-boy", 1), ("rascal-girl", 2)),
    "wall_breakers": (("wall-breaker", 2),),
    "royal_hogs": (("royal-hog", 4),),
    "skeleton_dragons": (("skeleton-dragon", 2),),
    "phoenix": (("phoenix-big", 1),),
    "elixir_golem": (("elixir-golem-big", 1),),
}

_SHORT = {
    "skeleton_army": "Skarmy", "three_musketeers": "3 Musk", "mini_pekka": "Mini PEKKA",
    "electro_wizard": "E-Wiz", "elite_barbarians": "E-Barbs", "inferno_dragon": "Inferno D",
    "inferno_tower": "Inferno T", "electro_dragon": "E-Drag", "electro_giant": "E-Giant",
    "electro_spirit": "E-Spirit", "barbarian_barrel": "Barb Barrel", "barbarian_hut": "Barb Hut",
    "elixir_collector": "Pump", "giant_skeleton": "Giant Skel", "skeleton_barrel": "Skel Barrel",
    "skeleton_dragons": "Skel Drags", "royal_recruits": "Recruits", "royal_delivery": "Delivery",
    "goblin_barrel": "Gob Barrel", "goblin_gang": "Gob Gang", "goblin_giant": "Gob Giant",
    "goblin_drill": "Drill", "goblin_cage": "Gob Cage", "goblin_hut": "Gob Hut",
    "giant_snowball": "Snowball", "flying_machine": "Flying Mach", "mother_witch": "Mother W",
    "battle_healer": "Healer", "mighty_miner": "M. Miner", "archer_queen": "Queen",
    "golden_knight": "Gold Knight", "skeleton_king": "Skel King", "little_prince": "L. Prince",
    "spear_goblins": "Spear Gobs", "minion_horde": "Horde", "wall_breakers": "Wall Brk",
    "magic_archer": "Magic Arch", "cannon_cart": "Cannon Cart", "night_witch": "Night W",
    "goblin_demolisher": "Demolisher", "goblin_machine": "Gob Machine", "suspicious_bush": "Bush",
    "boss_bandit": "Boss Bandit", "royal_giant": "Royal Giant", "royal_hogs": "Royal Hogs",
}


def _build() -> Dict[str, Card]:
    cards = {}
    for key, name, cost, kind in _BASE + _NEWER:
        cards[key] = Card(key=key, name=name, cost=cost, kind=kind,
                          short=_SHORT.get(key, ""), champion=key in CHAMPIONS,
                          units=_UNITS.get(key, ((key.replace("_", "-"), 1),)))
    return cards


CARDS: Dict[str, Card] = _build()


def card(key: str) -> Card:
    return CARDS[key]


def cost(key: Optional[str], default: float = 0.0) -> float:
    found = CARDS.get(key or "")
    return float(found.cost) if found else default


# --------------------------------------------------------------- the detector

@dataclass(frozen=True)
class UnitInfo:
    """What one detector class means for play extraction."""
    unit: str                                  # canonical class, dashes
    role: str                                  # card | child | spell | ignore | marker | structure
    cards: Tuple[Tuple[str, int], ...] = ()    # (card, units placed per play)
    parents: FrozenSet[str] = frozenset()      # classes whose output this can be
    parent_radius: float = 0.0                 # tiles
    evolved: bool = False


# Towers, health bars and UI. Never a play.
STRUCTURES = frozenset({
    "king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower",
    "tower", "ruins",
})
IGNORED = frozenset({
    "tower-bar", "king-tower-bar", "bar", "bar-level", "dagger-duchess-tower-bar",
    "skeleton-king-bar", "king-level", "clock", "emote", "selected",
    "evolution-symbol", "dirt", "blood", "background", "small-text", "big-text",
    "arrow",
    # Projectiles and effects that belong to something already counted.
    "bomb", "axe", "goblin-ball", "tesla-evolution-shock", "zap-evolution",
})
# Shown at the deploy point of *your own* card: the floating card name and the
# "-N" elixir drop. Seeing one next to a new unit says whose the unit is.
MARKERS = frozenset({"text", "elixir"})

# Spawner and death output: a class that appears out of something already on
# the board. (child class -> parent classes, radius in tiles)
_SPAWNED_BY = {
    "skeleton": ({"tombstone", "graveyard", "witch", "skeleton-king", "skeleton-king-skill",
                  "skeleton-barrel"}, 4.5),
    "bat": ({"night-witch"}, 3.0),
    "spear-goblin": ({"goblin-hut", "goblin-giant"}, 3.0),
    "goblin": ({"goblin-barrel", "goblin-drill"}, 3.5),
    "barbarian": ({"barbarian-hut", "battle-ram", "barbarian-barrel"}, 3.0),
    "fire-spirit": ({"furnace"}, 3.0),
    "royal-recruit": ({"royal-delivery"}, 3.0),
}
# Classes that only ever come out of something else.
_CHILD_ONLY = {
    "golemite": {"golem"}, "lava-pup": {"lava-hound"}, "phoenix-egg": {"phoenix-big"},
    "phoenix-small": {"phoenix-big", "phoenix-egg"}, "elixir-golem-mid": {"elixir-golem-big"},
    "elixir-golem-small": {"elixir-golem-big", "elixir-golem-mid"}, "hog": {"mother-witch"},
    "goblin-brawler": {"goblin-cage"}, "royal-guardian": {"little-prince"},
    "rascal-girl": {"rascal-boy"}, "skeleton-king-skill": {"skeleton-king"},
}

# Spells that travel away from the player who cast them: thrown from the king
# tower, or rolled forward. Which way one moves says whose it is.
DIRECTIONAL_SPELLS = frozenset({
    "fireball", "arrows", "rocket", "goblin_barrel", "giant_snowball", "the_log",
    "barbarian_barrel",
})
# Troops that are not restricted to their own half when deployed.
DEPLOY_ANYWHERE = frozenset({"miner", "goblin_drill"})

_EVOLVED = re.compile(r"[-_ ](evolution|evo|ev\d|hero)$")


def squash(name: str) -> str:
    """Comparable form of a class name: `Hog_Rider`, `hog-rider` -> `hogrider`."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _class_table() -> Dict[str, UnitInfo]:
    table: Dict[str, UnitInfo] = {}
    producers: Dict[str, list] = {}
    for c in CARDS.values():
        for unit, count in c.units:
            producers.setdefault(unit, []).append((c.key, count))
    for unit, cards in producers.items():
        if unit in _CHILD_ONLY:
            continue
        role = "spell" if all(CARDS[k].kind == SPELL for k, _ in cards) else "card"
        parents, radius = _SPAWNED_BY.get(unit, (set(), 0.0))
        table[unit] = UnitInfo(unit, role, tuple(cards), frozenset(parents), radius)
    for unit, parents in _CHILD_ONLY.items():
        table[unit] = UnitInfo(unit, "child", (), frozenset(parents), 4.0)
    for unit in STRUCTURES:
        table[unit] = UnitInfo(unit, "structure")
    for unit in IGNORED:
        table[unit] = UnitInfo(unit, "ignore")
    for unit in MARKERS:
        table[unit] = UnitInfo(unit, "marker")
    return table


_BY_UNIT = _class_table()
_BY_SQUASHED = {squash(u): info for u, info in _BY_UNIT.items()}
# Spellings other detectors use for the same thing.
_BY_SQUASHED.update({
    squash(alias): _BY_UNIT[unit] for alias, unit in {
        "log": "the-log", "minipekka": "mini-pekka", "phoenix": "phoenix-big",
        "phoenixlarge": "phoenix-big", "elixirgolem": "elixir-golem-big",
        "elixirgolemlarge": "elixir-golem-big", "elixirgolemmedium": "elixir-golem-mid",
        "skeletonwarrior": "guard", "xbow": "x-bow", "ramrider": "ram-rider",
        "zapmachine": "sparky", "minisparkys": "zappy", "elitebarbarians": "elite-barbarian",
    }.items()
})
_UNKNOWN: Dict[str, int] = {}


def classify(detector_class: str) -> Optional[UnitInfo]:
    """Interpret one detector class; None for a class this catalog cannot place.

    Unknown classes are counted in `unknown_classes()` rather than silently
    dropped, because a class with no entry is a card the overlay is blind to.
    """
    name = detector_class.strip().lower()
    evolved = False
    match = _EVOLVED.search(name)
    if match and squash(name) not in _BY_SQUASHED:
        name, evolved = name[:match.start()], True
    info = _BY_SQUASHED.get(squash(name))
    if info is None:
        _UNKNOWN[detector_class] = _UNKNOWN.get(detector_class, 0) + 1
        return None
    if evolved:
        return UnitInfo(info.unit, info.role, info.cards, info.parents,
                        info.parent_radius, evolved=True)
    return info


def unknown_classes() -> Dict[str, int]:
    return dict(_UNKNOWN)


# ------------------------------------------------------------ meta-deck names

META_ALIASES = {
    "angry_barbarians": "elite_barbarians", "archer": "archers", "assassin": "bandit",
    "axe_man": "executioner", "barb_log": "barbarian_barrel", "blowdart_goblin": "dart_goblin",
    "dart_barrell": "flying_machine", "dark_witch": "night_witch", "elite_archer": "magic_archer",
    "elixir _collector": "elixir_collector", "fire_spirits": "fire_spirit",
    "firespirit_hut": "furnace", "ghost": "royal_ghost", "giant_buffer": "rune_giant",
    "heal": "heal_spirit", "ice_golemite": "ice_golem", "ice_spirits": "ice_spirit",
    "log": "the_log", "minipekka": "mini_pekka", "mini_sparkys": "zappies",
    "moving_cannon": "cannon_cart", "rage_barbarian": "lumberjack",
    "skeleton_balloon": "skeleton_barrel", "skeleton_warriors": "guards",
    "snowball": "giant_snowball", "wallbreakers": "wall_breakers",
    "witch_mother": "mother_witch", "xbow": "x_bow", "zap_machine": "sparky",
}


def from_meta(name: str) -> Optional[str]:
    """A meta-deck card name as a card key, or None if this catalog lacks it."""
    key = META_ALIASES.get(name, name)
    return key if key in CARDS else None


def from_any(name: str) -> Optional[str]:
    """Best-effort card key from a user-typed or foreign name."""
    if name in CARDS:
        return name
    key = from_meta(name)
    if key:
        return key
    wanted = squash(name)
    for c in CARDS.values():
        if squash(c.key) == wanted or squash(c.name) == wanted or squash(c.label) == wanted:
            return c.key
    info = classify(name)
    if info and len(info.cards) == 1:
        return info.cards[0][0]
    return None


def candidates_for(units: Iterable[str]) -> Tuple[str, ...]:
    """Every card that places at least one of these classes."""
    found = []
    for unit in units:
        info = _BY_UNIT.get(unit)
        if info:
            found.extend(k for k, _ in info.cards if k not in found)
    return tuple(found)
