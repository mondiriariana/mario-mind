"""Objects on the Mario screen read from emulator RAM: Mario, the 5 enemy slots and the power-up slot.

``oracle_objects`` turns one RAM snapshot into a list of ``Obj`` boxes (screen pixels, with a type: goomba, koopa,
other enemy, item, scenery); ``order_objects`` puts Mario first and the rest by distance to him. Exact and free (no
detector). Used by the RL reward's game features (experiments/glimmer_imitation/rl.py) and the reasoning-trace
audit (experiments/odysseus/audit_dataset.py)."""
from __future__ import annotations

from dataclasses import dataclass

from .emulator import SCREEN_SHAPE

SCREEN_H, SCREEN_W = SCREEN_SHAPE[:2]  # 224 x 240
# The NES draws 256 x 240; stable-retro's screen drops 8 px on each side. RAM coordinates
# are in NES pixels, so screen = RAM - offset.
X_OFF, Y_OFF = 8, 8

TYPE_MARIO, TYPE_GOOMBA, TYPE_KOOPA, TYPE_ENEMY, TYPE_ITEM, TYPE_SCENERY = range(6)
N_TYPES = 6
TYPE_NAMES = ["mario", "goomba", "koopa", "enemy", "item", "scenery"]
GOOMBA_IDS = {0x06}
KOOPA_IDS = {0x00, 0x01, 0x03, 0x04, 0x09, 0x0E, 0x0F, 0x10}
SCENERY_MIN_ID = 0x24  # lifts / platforms are enemy-slot objects too
META = 6  # cx, cy, w, h, type, valid

def _obj_ram_vars() -> list[str]:
    """RAM variables oracle_objects reads: Mario, scroll, then each per-slot variable for the 6 slots."""
    names = ["player_x_posHi", "player_x_posLo", "player_y_pos", "player_y_screen", "player_size",
             "xscrollHi", "xscrollLo"]
    for i in range(6):
        names.append(f"enemy_drawn{15 + i}")
    for prefix in ("enemy_id_", "enemy_x_", "enemy_pageloc_", "enemy_y_"):
        for i in range(6):
            names.append(f"{prefix}{i}")
    for i in range(5):  # y_high is only checked for the 5 enemy slots, not the power-up slot
        names.append(f"enemy_y_high_{i}")
    return names


OBJ_RAM_VARS = _obj_ram_vars()


@dataclass
class Obj:
    type: int
    x0: float  # screen pixels, box of the object itself
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    slot: int = -1  # RAM enemy slot 0-5 (-1 for Mario)

    @property
    def center(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def _on_screen(o: Obj) -> bool:
    return o.x1 > 0 and o.x0 < SCREEN_W and o.y1 > 0 and o.y0 < SCREEN_H


def oracle_objects(ram: dict) -> list[Obj]:
    """Objects on screen from one frame's RAM values (dict of ints)."""
    g = lambda k: int(ram.get(k, 0))
    scroll = g("xscrollHi") * 256 + g("xscrollLo")
    out = []
    if g("player_y_screen") == 1:
        mx = g("player_x_posHi") * 256 + g("player_x_posLo") - scroll - X_OFF
        # player_y_pos is the top of a 32-px-tall slot; small Mario is drawn in its lower half (+16)
        my = g("player_y_pos") - Y_OFF
        big = g("player_size") == 0
        o = Obj(TYPE_MARIO, mx, my if big else my + 16, mx + 16, my + 32)
        if _on_screen(o):
            out.append(o)
    for i in range(6):
        if g(f"enemy_drawn{15 + i}") == 0:
            continue
        if i < 5 and g(f"enemy_y_high_{i}") != 1:
            continue
        ex = g(f"enemy_pageloc_{i}") * 256 + g(f"enemy_x_{i}") - scroll - X_OFF
        # enemy_y is already the screen row of the top of the 16-px body (no Y_OFF);
        # koopas are 24 px tall, so their box starts 8 px higher
        ey = g(f"enemy_y_{i}")
        eid = g(f"enemy_id_{i}")
        if i == 5:
            t = TYPE_ITEM
        elif eid in GOOMBA_IDS:
            t = TYPE_GOOMBA
        elif eid in KOOPA_IDS:
            t = TYPE_KOOPA
        elif eid >= SCENERY_MIN_ID:
            t = TYPE_SCENERY
        else:
            t = TYPE_ENEMY
        o = Obj(t, ex, ey - 8 if t == TYPE_KOOPA else ey, ex + 16, ey + 16, slot=i)
        if _on_screen(o):
            out.append(o)
    return out


def order_objects(objs: list[Obj], k: int) -> list[Obj]:
    """Mario first, then the rest by distance to Mario (screen centre if Mario is absent)."""
    mario = []
    for o in objs:
        if o.type == TYPE_MARIO:
            mario.append(o)
            break
    ref = mario[0].center if mario else (SCREEN_W / 2, SCREEN_H / 2)
    others = []
    for o in objs:
        if o not in mario:
            others.append(o)
    rest = sorted(others, key=lambda o: (o.center[0] - ref[0]) ** 2 + (o.center[1] - ref[1]) ** 2)
    return (mario + rest)[:k]
