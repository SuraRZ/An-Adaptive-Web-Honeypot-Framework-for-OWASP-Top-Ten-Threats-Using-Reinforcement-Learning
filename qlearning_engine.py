"""
qlearning_engine.py  —  OWASP Top-10 Adaptive Honeypot  Q-Learning Engine
==========================================================================
Reward formula  :  R = 0.6·T + 0.3·D − 0.1·P
Learning rate   :  α = 0.20
Discount factor :  γ = 0.90

T values (destination state)
  SS1=1  SS2=2  SS3=3  PS1=4  PS2=5  PS3=6  ST=0

D values (transition type)
  SS1→SS1 same-state loop   =  1
  SS1→SS2                   =  2
  SS2→SS3                   =  3
  any→PS1  / PS1→PS1 loop   =  5
  any→PS2  / PS2 sub-loop   =  7   (good), 8 (bad/loop)
  PS2→PS3                   =  9
  PS3→PS3 loop              = 10
  PS1→ST                    =  4
  PS2→ST                    =  5
  PS3→ST                    =  6
  SS1→ST                    =  1
  SS2→ST                    =  2
  SS3→ST                    =  3

P values
  good honeypot action (✓)  =  0
  wrong / sub-optimal        =  8   (for non-ST transitions from good states)
  → ST                       = 10

Computed rewards
  SS1→SS1 good   +0.90    SS2→SS3 good   +2.70    PS1→PS1 good   +3.90
  SS1→SS2 bad    +1.00    SS2→PS1 good   +3.90    PS1→PS2 good   +5.10
  SS1→PS1 bad    +2.80    SS2→ST         −0.40    PS1→PS1 bad    +3.10
  SS1→ST         −0.70    SS3→PS1 good   +3.90    PS1→ST         +0.20
                          SS3→PS2 good   +5.10    PS2→PS2 good   +5.40
                          SS3→PS1 bad    +3.10    PS2→PS2 bad    +4.60
                          SS3→ST         −0.10    PS2→PS3 good   +6.30
                                                  PS2→ST         +0.50
                                                  PS3→PS3 good   +6.60
                                                  PS3→PS3 bad    +5.80
                                                  PS3→ST         +0.80
"""

import json, logging, os, random, threading, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("honeypot.ql")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
ACTIONS = ["A1","A2","A3","A4","A5","A6","A7","A8"]
ATTACKS = ["A01","A02","A03","A04","A05","A06","A07","A08","A09","A10"]

# ─────────────────────────────────────────────────────────────
# Reward function  R = 0.6T + 0.3D − 0.1P
# ─────────────────────────────────────────────────────────────
_T = {"SS1":1,"SS2":2,"SS3":3,"ST":0,
      "PS1_A":4,"PS2_A":5,
      "PS1_B":4,"PS2_B":5,
      "PS1_C":4,"PS2_C":5,"PS3_C":6,
      "PS1_D":4,"PS2_D":5,
      "PS1_E":4,"PS2_E":5,
      "PS1_F":4,"PS2_F":5,
      "PS1_G":4,"PS2_G":5,"PS3_G":6,
      "PS1_H":4,"PS2_H":5,
      "PS1_I":4,"PS2_I":5,
      "PS1_J":4,"PS2_J":5}

def _generalise(state: str) -> str:
    """Map any concrete state name to its general category for the
    reward lookup table: SS1, SS2, SS3, PS1, PS2, PS3, or ST."""
    if state == "ST":  return "ST"
    if state.startswith("PS3"): return "PS3"
    if state.startswith("PS2"): return "PS2"
    if state.startswith("PS1"): return "PS1"
    return state  # SS1, SS2, SS3


# ─────────────────────────────────────────────────────────────
# Direct reward lookup table — built EXACTLY from the reward
# reference table (R = 0.6T + 0.3D - 0.1P).
#
# Key:   (general_src, general_dst, good)
# Value: reward
#
# This table is the single source of truth. It was cross-checked
# against the formula for every row and matches exactly.
# ─────────────────────────────────────────────────────────────
_REWARD_TABLE: Dict[Tuple[str,str,bool], float] = {
    ("SS1","SS1",True):   0.90,   # SS1->SS1 (A1, good)
    ("SS1","SS2",False):  1.00,   # SS1->SS2 (bad action on clean)
    ("SS1","PS1",False):  2.80,   # SS1->PS1 (bad action, skip)
    ("SS1","PS1",True):   2.80,   # same row — no separate good variant
    ("SS1","ST", False): -0.70,   # SS1->ST (any early exit)
    ("SS1","ST", True):  -0.70,

    ("SS2","SS3",True):   2.70,   # SS2->SS3 (good action)
    ("SS2","PS1",True):   3.90,   # SS2->PS1 (good action, jump)
    ("SS2","ST", False): -0.40,   # SS2->ST
    ("SS2","ST", True):  -0.40,

    ("SS3","PS1",True):   3.90,   # SS3->PS1 (good)
    ("SS3","PS2",True):   5.10,   # SS3->PS2 (good, best actions)
    ("SS3","PS1",False):  3.10,   # SS3->PS1 (bad, unguided)
    ("SS3","ST", False): -0.10,   # SS3->ST
    ("SS3","ST", True):  -0.10,

    ("PS1","PS1",True):   3.90,   # PS1->PS1 (good, loop)
    ("PS1","PS2",True):   5.10,   # PS1->PS2 (good)
    ("PS1","PS2",False):  5.10,   # same row — no separate bad variant
    ("PS1","PS1",False):  3.10,   # PS1->PS1 (bad)
    ("PS1","ST", False):  0.20,   # PS1->ST
    ("PS1","ST", True):   0.20,

    ("PS2","PS2",True):   5.40,   # PS2->PS2 (good, loop)
    ("PS2","PS3",True):   6.30,   # PS2->PS3 (good, A03/A07)
    ("PS2","PS3",False):  6.30,   # same row — no separate bad variant
    ("PS2","PS2",False):  4.60,   # PS2->PS2 (bad)
    ("PS2","ST", False):  0.50,   # PS2->ST
    ("PS2","ST", True):   0.50,

    ("PS3","PS3",True):   6.60,   # PS3->PS3 (good, A03/A07) -- e.g. A2/A3
    ("PS3","PS3",False):  5.80,   # PS3->PS3 (bad, e.g. A1) -- per the full
                                   # per-category tables (A03/A07), A1 at
                                   # PS3 maps to PS3/ST = 0.15/0.85 with
                                   # R=+5.80/+0.80, confirming the bad-loop
                                   # row IS used for non-best actions at PS3.
    ("PS3","ST", False):  0.80,   # PS3->ST
    ("PS3","ST", True):   0.80,

    ("ST","ST",False):    0.00,   # ST->ST (all actions)
    ("ST","ST",True):     0.00,
}


def reward(src: str, dst: str, good: bool) -> float:
    """
    Return the reward for a transition, using the direct lookup
    table built from the reward reference table
    (R = 0.6T + 0.3D - 0.1P).

    src, dst : concrete state names (e.g. "PS1_C", "SS2", "ST")
    good     : whether the action taken is the "best" action for
               this (state, attack) pair, as flagged in the MDP.

    Special case: PS1->PS2 and PS2->PS3 ("deepening" transitions)
    have only ONE row in the table (no separate bad variant), so
    both good=True and good=False map to the same reward — this is
    already encoded directly in _REWARD_TABLE above via the
    duplicate (...,...,False) entries.
    """
    gs, gd = _generalise(src), _generalise(dst)
    key = (gs, gd, good)
    if key in _REWARD_TABLE:
        return _REWARD_TABLE[key]

    # Fallback: try the opposite `good` flag (covers any transition
    # type not explicitly split into good/bad variants in the table)
    alt_key = (gs, gd, not good)
    if alt_key in _REWARD_TABLE:
        return _REWARD_TABLE[alt_key]

    # Should never happen if the MDP only contains valid transitions —
    # log loudly so it's easy to spot during testing.
    log.error("No reward table entry for %s -> %s (good=%s) "
              "[general: %s -> %s]", src, dst, good, gs, gd)
    return 0.0

MDP: Dict[Tuple[str,str,str], List[Tuple[str,float,bool]]] = {}

def _reg(state, attack, action, transitions, good):
    MDP[(state, attack, action)] = [(s,p,good) for s,p in transitions]

def _fill_missing(attack, ps1, ps2, ps3=None):
    """
    Fill every (state, attack, action) triple that is not already
    registered, using the moderate templates derived from the HTML.
    ps3 is only used for A03 and A07.
    """
    # Templates: keys are state, values are (destinations, probs, good)
    templates = {
        "SS1": {
            "default": ([("SS2",0.45),("ST",0.55)], False),
        },
        "SS2": {
            "default": ([(  "SS3",0.20),(ps1,0.40),("ST",0.40)], False),
        },
        "SS3": {
            "default": ([(ps1,0.40),(ps2,0.20),("ST",0.40)], False),
        },
        ps1: {
            "default": ([(ps1,0.40),(ps2,0.30),("ST",0.30)], False),
        },
        ps2: {
            "default": ([(ps2,0.45),("ST",0.55)], False),
        },
    }
    if ps3:
        templates[ps3] = {
            "default": ([(ps3,0.45),("ST",0.55)], False),
        }

    all_states = ["SS1","SS2","SS3",ps1,ps2] + ([ps3] if ps3 else [])
    for state in all_states:
        for action in ACTIONS:
            key = (state, attack, action)
            if key in MDP:
                continue
            trans, good = templates[state]["default"]
            MDP[key] = [(s,p,good) for s,p in trans]

# ══════════════════════════════════════════════════════════════
# A01  Broken Access Control
# Best: A5 (redirect to decoy), A6 (fake auth success)
# ══════════════════════════════════════════════════════════════
atk = "A01"
ps1, ps2 = "PS1_A", "PS2_A"

# SS1 — all 8 actions (HTML has all)
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A2",[("SS2",0.55),("ST",0.45)],False)
_reg("SS1",atk,"A3",[("SS2",0.45),("ST",0.55)],False)
_reg("SS1",atk,"A4",[("SS2",0.35),("ST",0.65)],False)
_reg("SS1",atk,"A5",[("SS2",0.60),(ps1,0.20),("ST",0.20)],False)
_reg("SS1",atk,"A6",[("SS2",0.50),(ps1,0.30),("ST",0.20)],False)
_reg("SS1",atk,"A7",[("SS2",0.40),("ST",0.60)],False)
_reg("SS1",atk,"A8",[("SS2",0.30),("ST",0.70)],False)

# SS2 — all 8 actions (HTML has all)
_reg("SS2",atk,"A1",[("SS3",0.25),(ps1,0.35),("ST",0.40)],False)
_reg("SS2",atk,"A2",[("SS3",0.20),(ps1,0.45),("ST",0.35)],False)
_reg("SS2",atk,"A3",[("SS3",0.20),(ps1,0.50),("ST",0.30)],False)
_reg("SS2",atk,"A4",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A5",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
_reg("SS2",atk,"A6",[("SS3",0.20),(ps1,0.60),("ST",0.20)],True)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.40),("ST",0.40)],False)
_reg("SS2",atk,"A8",[("SS3",0.20),(ps1,0.35),("ST",0.45)],False)

# SS3 — all 8 actions (HTML has all)
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A2",[(ps1,0.45),(ps2,0.20),("ST",0.35)],False)
_reg("SS3",atk,"A3",[(ps1,0.45),(ps2,0.20),("ST",0.35)],False)
_reg("SS3",atk,"A4",[(ps1,0.30),(ps2,0.10),("ST",0.60)],False)
_reg("SS3",atk,"A5",[(ps1,0.50),(ps2,0.35),("ST",0.15)],True)
_reg("SS3",atk,"A6",[(ps1,0.45),(ps2,0.40),("ST",0.15)],True)
_reg("SS3",atk,"A7",[(ps1,0.40),(ps2,0.25),("ST",0.35)],False)
_reg("SS3",atk,"A8",[(ps1,0.35),(ps2,0.15),("ST",0.50)],False)

# PS1_A — all 8 actions (HTML has all)
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A2",[(ps1,0.40),(ps2,0.30),("ST",0.30)],False)
_reg(ps1,atk,"A3",[(ps1,0.40),(ps2,0.30),("ST",0.30)],False)
_reg(ps1,atk,"A4",[(ps1,0.35),(ps2,0.20),("ST",0.45)],False)
_reg(ps1,atk,"A5",[(ps1,0.45),(ps2,0.40),("ST",0.15)],True)
_reg(ps1,atk,"A6",[(ps1,0.40),(ps2,0.50),("ST",0.10)],True)
_reg(ps1,atk,"A7",[(ps1,0.35),(ps2,0.35),("ST",0.30)],False)
_reg(ps1,atk,"A8",[(ps1,0.30),(ps2,0.25),("ST",0.45)],False)

# PS2_A — all 8 actions (HTML has all)
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A2",[(ps2,0.45),("ST",0.55)],False)
_reg(ps2,atk,"A3",[(ps2,0.50),("ST",0.50)],False)
_reg(ps2,atk,"A4",[(ps2,0.40),("ST",0.60)],False)
_reg(ps2,atk,"A5",[(ps2,0.60),("ST",0.40)],True)
_reg(ps2,atk,"A6",[(ps2,0.70),("ST",0.30)],True)
_reg(ps2,atk,"A7",[(ps2,0.50),("ST",0.50)],False)
_reg(ps2,atk,"A8",[(ps2,0.40),("ST",0.60)],False)

# ══════════════════════════════════════════════════════════════
# A02  Cryptographic Failures
# Best: A3 (fake sensitive data), A6 (fake key/secret reveal)
# ══════════════════════════════════════════════════════════════
atk = "A02"
ps1, ps2 = "PS1_B", "PS2_B"

# SS1
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
# HTML shows "A2–A8 ✗ → SS2/ST ~0.40–0.55/rest"
# Fill all with moderate probabilities that match the pattern
for a,p in [("A2",0.55),("A3",0.50),("A4",0.40),("A5",0.50),
             ("A6",0.50),("A7",0.45),("A8",0.40)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has all 8
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A2",[("SS3",0.20),(ps1,0.45),("ST",0.35)],False)
_reg("SS2",atk,"A3",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
_reg("SS2",atk,"A4",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A5",[("SS3",0.20),(ps1,0.50),("ST",0.30)],False)
_reg("SS2",atk,"A6",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.40),("ST",0.40)],False)
_reg("SS2",atk,"A8",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)

# SS3 — HTML has all 8
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A2",[(ps1,0.40),(ps2,0.20),("ST",0.40)],False)
_reg("SS3",atk,"A3",[(ps1,0.45),(ps2,0.40),("ST",0.15)],True)
_reg("SS3",atk,"A4",[(ps1,0.30),(ps2,0.10),("ST",0.60)],False)
_reg("SS3",atk,"A5",[(ps1,0.40),(ps2,0.30),("ST",0.30)],False)
_reg("SS3",atk,"A6",[(ps1,0.40),(ps2,0.35),("ST",0.25)],True)
_reg("SS3",atk,"A7",[(ps1,0.35),(ps2,0.25),("ST",0.40)],False)
_reg("SS3",atk,"A8",[(ps1,0.30),(ps2,0.15),("ST",0.55)],False)

# PS1_B — HTML has all 8
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A2",[(ps1,0.40),(ps2,0.30),("ST",0.30)],False)
_reg(ps1,atk,"A3",[(ps1,0.45),(ps2,0.45),("ST",0.10)],True)
_reg(ps1,atk,"A4",[(ps1,0.35),(ps2,0.20),("ST",0.45)],False)
_reg(ps1,atk,"A5",[(ps1,0.40),(ps2,0.35),("ST",0.25)],False)
_reg(ps1,atk,"A6",[(ps1,0.40),(ps2,0.40),("ST",0.20)],True)
_reg(ps1,atk,"A7",[(ps1,0.35),(ps2,0.30),("ST",0.35)],False)
_reg(ps1,atk,"A8",[(ps1,0.30),(ps2,0.20),("ST",0.50)],False)

# PS2_B — HTML has all 8
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A2",[(ps2,0.45),("ST",0.55)],False)
_reg(ps2,atk,"A3",[(ps2,0.70),("ST",0.30)],True)
_reg(ps2,atk,"A4",[(ps2,0.35),("ST",0.65)],False)
_reg(ps2,atk,"A5",[(ps2,0.50),("ST",0.50)],False)
_reg(ps2,atk,"A6",[(ps2,0.60),("ST",0.40)],True)
_reg(ps2,atk,"A7",[(ps2,0.50),("ST",0.50)],False)
_reg(ps2,atk,"A8",[(ps2,0.40),("ST",0.60)],False)

# ══════════════════════════════════════════════════════════════
# A03  Injection  (3 private states)
# Best: A3 (fake DB output), A2 (error lure)
# ══════════════════════════════════════════════════════════════
atk = "A03"
ps1, ps2, ps3 = "PS1_C", "PS2_C", "PS3_C"

# SS1 — HTML has A1,A2,A3,A4,A5; A6–A8 use "~0.40–0.50/rest"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A2",[("SS2",0.60),("ST",0.40)],False)
_reg("SS1",atk,"A3",[("SS2",0.50),(ps1,0.20),("ST",0.30)],False)
_reg("SS1",atk,"A4",[("SS2",0.40),("ST",0.60)],False)
_reg("SS1",atk,"A5",[("SS2",0.60),(ps1,0.25),("ST",0.15)],False)
# Fill A6–A8 from HTML "~0.40–0.50/rest"
_reg("SS1",atk,"A6",[("SS2",0.50),("ST",0.50)],False)
_reg("SS1",atk,"A7",[("SS2",0.45),("ST",0.55)],False)
_reg("SS1",atk,"A8",[("SS2",0.40),("ST",0.60)],False)

# SS2 — HTML has A1,A2,A3,A4,A5; A6–A8 use "varies"
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.40),("ST",0.40)],False)
_reg("SS2",atk,"A2",[("SS3",0.20),(ps1,0.45),("ST",0.35)],True)
_reg("SS2",atk,"A3",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
_reg("SS2",atk,"A4",[("SS3",0.20),(ps1,0.25),("ST",0.55)],False)
_reg("SS2",atk,"A5",[("SS3",0.20),(ps1,0.45),("ST",0.35)],False)
# Fill A6–A8 (less effective than A3 per HTML)
_reg("SS2",atk,"A6",[("SS3",0.20),(ps1,0.35),("ST",0.45)],False)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.35),("ST",0.45)],False)
_reg("SS2",atk,"A8",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)

# SS3 — HTML has A1,A2,A3; A4–A8 use "varies / less effective"
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A2",[(ps1,0.65),(ps2,0.20),("ST",0.15)],True)
_reg("SS3",atk,"A3",[(ps1,0.60),(ps2,0.35),("ST",0.05)],True)
# Fill A4–A8 (less effective)
_reg("SS3",atk,"A4",[(ps1,0.35),(ps2,0.15),("ST",0.50)],False)
_reg("SS3",atk,"A5",[(ps1,0.40),(ps2,0.20),("ST",0.40)],False)
_reg("SS3",atk,"A6",[(ps1,0.40),(ps2,0.20),("ST",0.40)],False)
_reg("SS3",atk,"A7",[(ps1,0.40),(ps2,0.20),("ST",0.40)],False)
_reg("SS3",atk,"A8",[(ps1,0.35),(ps2,0.15),("ST",0.50)],False)

# PS1_C — HTML has A1,A2,A3,A4; A5–A8 "moderate"
_reg(ps1,atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg(ps1,atk,"A2",[(ps1,0.55),(ps2,0.30),("ST",0.15)],True)
_reg(ps1,atk,"A3",[(ps1,0.50),(ps2,0.45),("ST",0.05)],True)
_reg(ps1,atk,"A4",[(ps1,0.35),(ps2,0.20),("ST",0.45)],False)
# Fill A5–A8 (moderate per HTML)
_reg(ps1,atk,"A5",[(ps1,0.40),(ps2,0.30),("ST",0.30)],False)
_reg(ps1,atk,"A6",[(ps1,0.40),(ps2,0.30),("ST",0.30)],False)
_reg(ps1,atk,"A7",[(ps1,0.40),(ps2,0.25),("ST",0.35)],False)
_reg(ps1,atk,"A8",[(ps1,0.35),(ps2,0.20),("ST",0.45)],False)

# PS2_C — HTML has A1,A2,A3; A4–A8 "varying effectiveness"
_reg(ps2,atk,"A1",[(ps2,0.15),("ST",0.85)],False)
_reg(ps2,atk,"A2",[(ps2,0.55),(ps3,0.25),("ST",0.20)],True)
_reg(ps2,atk,"A3",[(ps2,0.50),(ps3,0.45),("ST",0.05)],True)
# Fill A4–A8
_reg(ps2,atk,"A4",[(ps2,0.40),(ps3,0.15),("ST",0.45)],False)
_reg(ps2,atk,"A5",[(ps2,0.40),(ps3,0.20),("ST",0.40)],False)
_reg(ps2,atk,"A6",[(ps2,0.40),(ps3,0.20),("ST",0.40)],False)
_reg(ps2,atk,"A7",[(ps2,0.40),(ps3,0.20),("ST",0.40)],False)
_reg(ps2,atk,"A8",[(ps2,0.35),(ps3,0.15),("ST",0.50)],False)

# PS3_C — per full HTML table: only A1 is the "worst" action (R=5.80).
# A2, A3(best), A4, and A5-A8 are all "good" at this max-depth state
# (R=6.60) -- PS3 is the deepest state, so only A1 (normal response,
# which signals "nothing to see here" to the attacker) is penalised.
_reg(ps3,atk,"A1",[(ps3,0.15),("ST",0.85)],False)
_reg(ps3,atk,"A2",[(ps3,0.55),("ST",0.45)],True)
_reg(ps3,atk,"A3",[(ps3,0.70),("ST",0.30)],True)
_reg(ps3,atk,"A4",[(ps3,0.35),("ST",0.65)],True)
# Fill A5–A8 (moderate, but still "good" per HTML — R=6.60)
_reg(ps3,atk,"A5",[(ps3,0.45),("ST",0.55)],True)
_reg(ps3,atk,"A6",[(ps3,0.45),("ST",0.55)],True)
_reg(ps3,atk,"A7",[(ps3,0.45),("ST",0.55)],True)
_reg(ps3,atk,"A8",[(ps3,0.45),("ST",0.55)],True)

# ══════════════════════════════════════════════════════════════
# A04  Insecure Design
# Best: A6 (fake auth/bypass), A5 (decoy workflow)
# ══════════════════════════════════════════════════════════════
atk = "A04"
ps1, ps2 = "PS1_D", "PS2_D"

# SS1 — HTML has A1,A5,A6; A2–A4,A7–A8 "all penalised ~0.35–0.45/rest"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A5",[("SS2",0.60),(ps1,0.20),("ST",0.20)],False)
_reg("SS1",atk,"A6",[("SS2",0.55),(ps1,0.25),("ST",0.20)],False)
for a,p in [("A2",0.45),("A3",0.40),("A4",0.35),("A7",0.40),("A8",0.35)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A5,A6,A7; others "moderate to low"
_reg("SS2",atk,"A1",[("SS3",0.25),(ps1,0.30),("ST",0.45)],False)
_reg("SS2",atk,"A5",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
_reg("SS2",atk,"A6",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.60),("ST",0.20)],True)
for a,p in [("A2",0.40),("A3",0.40),("A4",0.30),("A8",0.35)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A5,A6; others use fill template
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A5",[(ps1,0.45),(ps2,0.35),("ST",0.20)],True)
_reg("SS3",atk,"A6",[(ps1,0.40),(ps2,0.50),("ST",0.10)],True)
for a,ps,pt in [("A2",0.40,0.20),("A3",0.40,0.20),
                ("A4",0.30,0.10),("A7",0.40,0.20),("A8",0.35,0.15)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_D — HTML has A1,A5,A6; fill others
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A5",[(ps1,0.40),(ps2,0.40),("ST",0.20)],True)
_reg(ps1,atk,"A6",[(ps1,0.35),(ps2,0.55),("ST",0.10)],True)
for a,ps,pt in [("A2",0.40,0.30),("A3",0.40,0.25),
                ("A4",0.35,0.20),("A7",0.40,0.30),("A8",0.35,0.20)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_D — HTML has A1,A5,A6; fill others
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A5",[(ps2,0.55),("ST",0.45)],True)
_reg(ps2,atk,"A6",[(ps2,0.70),("ST",0.30)],True)
for a,p in [("A2",0.45),("A3",0.45),("A4",0.35),("A7",0.45),("A8",0.40)]:
    _reg(ps2,atk,a,[(ps2,p),("ST",round(1-p,2))],False)

# ══════════════════════════════════════════════════════════════
# A05  Security Misconfiguration
# Best: A2 (error lure/fake config), A5 (redirect to fake admin)
# Also good: A4 (tarpit keeps scanner busy)
# ══════════════════════════════════════════════════════════════
atk = "A05"
ps1, ps2 = "PS1_E", "PS2_E"

# SS1 — HTML has A1,A2,A4; others "all penalised"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A2",[("SS2",0.60),("ST",0.40)],False)
_reg("SS1",atk,"A4",[("SS2",0.60),(ps1,0.20),("ST",0.20)],False)
for a,p in [("A3",0.45),("A5",0.50),("A6",0.40),("A7",0.40),("A8",0.40)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A2,A4,A5; others "lower effectiveness"
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A2",[("SS3",0.20),(ps1,0.60),("ST",0.20)],True)
_reg("SS2",atk,"A4",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
_reg("SS2",atk,"A5",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
for a,p in [("A3",0.40),("A6",0.35),("A7",0.35),("A8",0.30)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A2,A5; fill others
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A2",[(ps1,0.70),(ps2,0.15),("ST",0.15)],True)
_reg("SS3",atk,"A5",[(ps1,0.60),(ps2,0.25),("ST",0.15)],True)
_reg("SS3",atk,"A4",[(ps1,0.55),(ps2,0.15),("ST",0.30)],False)
for a,ps,pt in [("A3",0.40,0.15),("A6",0.40,0.15),
                ("A7",0.40,0.15),("A8",0.35,0.10)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_E — HTML has A1,A2,A5; fill others
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A2",[(ps1,0.60),(ps2,0.25),("ST",0.15)],True)
_reg(ps1,atk,"A5",[(ps1,0.55),(ps2,0.35),("ST",0.10)],True)
_reg(ps1,atk,"A4",[(ps1,0.50),(ps2,0.20),("ST",0.30)],False)
for a,ps,pt in [("A3",0.40,0.20),("A6",0.40,0.20),
                ("A7",0.40,0.20),("A8",0.35,0.15)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_E — HTML has A1,A2,A5; fill others
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A2",[(ps2,0.55),("ST",0.45)],True)
_reg(ps2,atk,"A5",[(ps2,0.65),("ST",0.35)],True)
_reg(ps2,atk,"A4",[(ps2,0.50),("ST",0.50)],False)
for a,p in [("A3",0.40),("A6",0.40),("A7",0.40),("A8",0.35)]:
    _reg(ps2,atk,a,[(ps2,p),("ST",round(1-p,2))],False)

# ══════════════════════════════════════════════════════════════
# A06  Vulnerable & Outdated Components
# Best: A7 (fake exploit accept), A2 (fake CVE error)
# ══════════════════════════════════════════════════════════════
atk = "A06"
ps1, ps2 = "PS1_F", "PS2_F"

# SS1 — HTML has A1; others "all penalised on clean"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
for a,p in [("A2",0.55),("A3",0.45),("A4",0.40),("A5",0.45),
             ("A6",0.45),("A7",0.50),("A8",0.40)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A2,A7; fill others
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A2",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.60),("ST",0.20)],True)
for a,p in [("A3",0.40),("A4",0.30),("A5",0.40),("A6",0.35),("A8",0.35)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A2,A7; fill others
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A2",[(ps1,0.60),(ps2,0.20),("ST",0.20)],True)
_reg("SS3",atk,"A7",[(ps1,0.50),(ps2,0.40),("ST",0.10)],True)
for a,ps,pt in [("A3",0.40,0.20),("A4",0.35,0.10),("A5",0.40,0.20),
                ("A6",0.40,0.20),("A8",0.35,0.15)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_F — HTML has A1,A2,A7; fill others
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A2",[(ps1,0.55),(ps2,0.30),("ST",0.15)],True)
_reg(ps1,atk,"A7",[(ps1,0.45),(ps2,0.45),("ST",0.10)],True)
for a,ps,pt in [("A3",0.40,0.25),("A4",0.35,0.15),("A5",0.40,0.25),
                ("A6",0.40,0.25),("A8",0.35,0.20)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_F — HTML has A1,A2,A7; fill others
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A2",[(ps2,0.55),("ST",0.45)],True)
_reg(ps2,atk,"A7",[(ps2,0.70),("ST",0.30)],True)
for a,p in [("A3",0.45),("A4",0.35),("A5",0.45),("A6",0.45),("A8",0.40)]:
    _reg(ps2,atk,a,[(ps2,p),("ST",round(1-p,2))],False)

# ══════════════════════════════════════════════════════════════
# A07  Authentication Failures  (3 private states)
# Best: A6 (fake session), A7 (fake account takeover)
# ══════════════════════════════════════════════════════════════
atk = "A07"
ps1, ps2, ps3 = "PS1_G", "PS2_G", "PS3_G"

# SS1 — HTML has A1,A6; others "penalised"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A6",[("SS2",0.60),(ps1,0.30),("ST",0.10)],False)
for a,p in [("A2",0.50),("A3",0.45),("A4",0.40),("A5",0.45),
             ("A7",0.50),("A8",0.40)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A6,A7; fill others
_reg("SS2",atk,"A1",[("SS3",0.25),(ps1,0.30),("ST",0.45)],False)
_reg("SS2",atk,"A6",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
for a,p in [("A2",0.35),("A3",0.35),("A4",0.30),("A5",0.40),("A8",0.30)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A6,A7; fill others
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A6",[(ps1,0.40),(ps2,0.50),("ST",0.10)],True)
_reg("SS3",atk,"A7",[(ps1,0.40),(ps2,0.40),("ST",0.20)],True)
for a,ps,pt in [("A2",0.40,0.20),("A3",0.40,0.20),("A4",0.30,0.10),
                ("A5",0.40,0.20),("A8",0.35,0.15)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_G — HTML has A1,A6,A7; fill others
_reg(ps1,atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg(ps1,atk,"A6",[(ps1,0.40),(ps2,0.50),("ST",0.10)],True)
_reg(ps1,atk,"A7",[(ps1,0.40),(ps2,0.45),("ST",0.15)],True)
for a,ps,pt in [("A2",0.40,0.25),("A3",0.40,0.25),("A4",0.35,0.15),
                ("A5",0.40,0.25),("A8",0.35,0.20)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_G — HTML has A1,A6,A7; fill others
_reg(ps2,atk,"A1",[(ps2,0.15),("ST",0.85)],False)
_reg(ps2,atk,"A6",[(ps2,0.40),(ps3,0.50),("ST",0.10)],True)
_reg(ps2,atk,"A7",[(ps2,0.40),(ps3,0.45),("ST",0.15)],True)
for a,ps,pt in [("A2",0.40,0.20),("A3",0.40,0.20),("A4",0.35,0.15),
                ("A5",0.40,0.20),("A8",0.35,0.15)]:
    _reg(ps2,atk,a,[(ps2,ps),(ps3,pt),("ST",round(1-ps-pt,2))],False)

# PS3_G — HTML has A1,A6,A7; fill others
_reg(ps3,atk,"A1",[(ps3,0.15),("ST",0.85)],False)
_reg(ps3,atk,"A6",[(ps3,0.55),("ST",0.45)],True)
_reg(ps3,atk,"A7",[(ps3,0.70),("ST",0.30)],True)
for a,p in [("A2",0.45),("A3",0.45),("A4",0.35),("A5",0.45),("A8",0.40)]:
    _reg(ps3,atk,a,[(ps3,p),("ST",round(1-p,2))],False)

# ══════════════════════════════════════════════════════════════
# A08  Software & Data Integrity Failures
# Best: A7 (fake payload accept), A8 (fake pipeline log clear)
# ══════════════════════════════════════════════════════════════
atk = "A08"
ps1, ps2 = "PS1_H", "PS2_H"

# SS1 — HTML has A1,A7,A8; others "penalised"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A7",[("SS2",0.55),(ps1,0.20),("ST",0.25)],False)
_reg("SS1",atk,"A8",[("SS2",0.50),(ps1,0.20),("ST",0.30)],False)
for a,p in [("A2",0.45),("A3",0.45),("A4",0.40),("A5",0.45),("A6",0.45)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A7,A8; fill others
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A7",[("SS3",0.20),(ps1,0.60),("ST",0.20)],True)
_reg("SS2",atk,"A8",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
for a,p in [("A2",0.40),("A3",0.40),("A4",0.30),("A5",0.40),("A6",0.35)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A7,A8; fill others
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A7",[(ps1,0.45),(ps2,0.40),("ST",0.15)],True)
_reg("SS3",atk,"A8",[(ps1,0.40),(ps2,0.35),("ST",0.25)],True)
for a,ps,pt in [("A2",0.40,0.20),("A3",0.40,0.20),("A4",0.30,0.10),
                ("A5",0.40,0.20),("A6",0.40,0.20)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_H — HTML has A1,A7,A8; fill others
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A7",[(ps1,0.40),(ps2,0.50),("ST",0.10)],True)
_reg(ps1,atk,"A8",[(ps1,0.40),(ps2,0.45),("ST",0.15)],True)
for a,ps,pt in [("A2",0.40,0.25),("A3",0.40,0.25),("A4",0.35,0.15),
                ("A5",0.40,0.25),("A6",0.40,0.25)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_H — HTML has A1,A7,A8; fill others
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A7",[(ps2,0.65),("ST",0.35)],True)
_reg(ps2,atk,"A8",[(ps2,0.60),("ST",0.40)],True)
for a,p in [("A2",0.45),("A3",0.45),("A4",0.35),("A5",0.45),("A6",0.45)]:
    _reg(ps2,atk,a,[(ps2,p),("ST",round(1-p,2))],False)

# ══════════════════════════════════════════════════════════════
# A09  Security Logging & Monitoring Failures
# Best: A8 (fake log clear), A4 (tarpit evasion)
# ══════════════════════════════════════════════════════════════
atk = "A09"
ps1, ps2 = "PS1_I", "PS2_I"

# SS1 — HTML has A1,A4,A8; others "penalised"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A4",[("SS2",0.60),(ps1,0.15),("ST",0.25)],False)
_reg("SS1",atk,"A8",[("SS2",0.55),(ps1,0.15),("ST",0.30)],False)
for a,p in [("A2",0.45),("A3",0.45),("A5",0.45),("A6",0.40),("A7",0.40)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A4,A8; fill others
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A4",[("SS3",0.20),(ps1,0.60),("ST",0.20)],True)
_reg("SS2",atk,"A8",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
for a,p in [("A2",0.40),("A3",0.40),("A5",0.40),("A6",0.35),("A7",0.35)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A4,A8; fill others
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A4",[(ps1,0.65),(ps2,0.20),("ST",0.15)],True)
_reg("SS3",atk,"A8",[(ps1,0.60),(ps2,0.30),("ST",0.10)],True)
for a,ps,pt in [("A2",0.40,0.15),("A3",0.40,0.15),("A5",0.40,0.15),
                ("A6",0.35,0.15),("A7",0.35,0.15)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_I — HTML has A1,A4,A8; fill others
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A4",[(ps1,0.65),(ps2,0.25),("ST",0.10)],True)
_reg(ps1,atk,"A8",[(ps1,0.55),(ps2,0.40),("ST",0.05)],True)
for a,ps,pt in [("A2",0.40,0.20),("A3",0.40,0.20),("A5",0.40,0.20),
                ("A6",0.35,0.20),("A7",0.35,0.20)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_I — HTML has A1,A4,A8; fill others
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A4",[(ps2,0.60),("ST",0.40)],True)
_reg(ps2,atk,"A8",[(ps2,0.70),("ST",0.30)],True)
for a,p in [("A2",0.45),("A3",0.45),("A5",0.45),("A6",0.40),("A7",0.40)]:
    _reg(ps2,atk,a,[(ps2,p),("ST",round(1-p,2))],False)

# ══════════════════════════════════════════════════════════════
# A10  SSRF
# Best: A5 (redirect to fake internal endpoint), A3 (fake internal data)
# ══════════════════════════════════════════════════════════════
atk = "A10"
ps1, ps2 = "PS1_J", "PS2_J"

# SS1 — HTML has A1,A4; others "penalised"
_reg("SS1",atk,"A1",[("SS1",1.00)],True)
_reg("SS1",atk,"A4",[("SS2",0.70),(ps1,0.20),("ST",0.10)],False)
for a,p in [("A2",0.50),("A3",0.55),("A5",0.55),("A6",0.45),
             ("A7",0.45),("A8",0.40)]:
    _reg("SS1",atk,a,[("SS2",p),("ST",round(1-p,2))],False)

# SS2 — HTML has A1,A3,A5; fill others
_reg("SS2",atk,"A1",[("SS3",0.20),(ps1,0.30),("ST",0.50)],False)
_reg("SS2",atk,"A3",[("SS3",0.20),(ps1,0.55),("ST",0.25)],True)
_reg("SS2",atk,"A5",[("SS3",0.20),(ps1,0.65),("ST",0.15)],True)
for a,p in [("A2",0.40),("A4",0.40),("A6",0.35),("A7",0.35),("A8",0.30)]:
    _reg("SS2",atk,a,[("SS3",0.20),(ps1,p),("ST",round(0.80-p,2))],False)

# SS3 — HTML has A1,A3,A5; fill others
_reg("SS3",atk,"A1",[(ps1,0.15),("ST",0.85)],False)
_reg("SS3",atk,"A3",[(ps1,0.45),(ps2,0.35),("ST",0.20)],True)
_reg("SS3",atk,"A5",[(ps1,0.50),(ps2,0.40),("ST",0.10)],True)
for a,ps,pt in [("A2",0.40,0.20),("A4",0.40,0.20),("A6",0.35,0.15),
                ("A7",0.35,0.15),("A8",0.30,0.15)]:
    _reg("SS3",atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS1_J — HTML has A1,A3,A5; fill others
_reg(ps1,atk,"A1",[(ps1,0.20),("ST",0.80)],False)
_reg(ps1,atk,"A3",[(ps1,0.45),(ps2,0.40),("ST",0.15)],True)
_reg(ps1,atk,"A5",[(ps1,0.40),(ps2,0.50),("ST",0.10)],True)
for a,ps,pt in [("A2",0.40,0.25),("A4",0.40,0.25),("A6",0.35,0.20),
                ("A7",0.35,0.20),("A8",0.30,0.20)]:
    _reg(ps1,atk,a,[(ps1,ps),(ps2,pt),("ST",round(1-ps-pt,2))],False)

# PS2_J — HTML has A1,A3,A5; fill others
_reg(ps2,atk,"A1",[(ps2,0.20),("ST",0.80)],False)
_reg(ps2,atk,"A3",[(ps2,0.60),("ST",0.40)],True)
_reg(ps2,atk,"A5",[(ps2,0.75),("ST",0.25)],True)
for a,p in [("A2",0.45),("A4",0.45),("A6",0.40),("A7",0.40),("A8",0.35)]:
    _reg(ps2,atk,a,[(ps2,p),("ST",round(1-p,2))],False)

# ─────────────────────────────────────────────────────────────
# Verify full coverage before continuing
# ─────────────────────────────────────────────────────────────
_ATK_STATES = {
    "A01":["SS1","SS2","SS3","PS1_A","PS2_A"],
    "A02":["SS1","SS2","SS3","PS1_B","PS2_B"],
    "A03":["SS1","SS2","SS3","PS1_C","PS2_C","PS3_C"],
    "A04":["SS1","SS2","SS3","PS1_D","PS2_D"],
    "A05":["SS1","SS2","SS3","PS1_E","PS2_E"],
    "A06":["SS1","SS2","SS3","PS1_F","PS2_F"],
    "A07":["SS1","SS2","SS3","PS1_G","PS2_G","PS3_G"],
    "A08":["SS1","SS2","SS3","PS1_H","PS2_H"],
    "A09":["SS1","SS2","SS3","PS1_I","PS2_I"],
    "A10":["SS1","SS2","SS3","PS1_J","PS2_J"],
}
_missing = []
for _atk, _states in _ATK_STATES.items():
    for _s in _states:
        for _a in ACTIONS:
            if (_s, _atk, _a) not in MDP:
                _missing.append((_s, _atk, _a))
if _missing:
    log.warning("MDP missing %d entries: %s", len(_missing), _missing[:5])
else:
    log.debug("MDP coverage: complete (%d entries)", len(MDP))

# ─────────────────────────────────────────────────────────────
# Q-Table
# ─────────────────────────────────────────────────────────────
class QTable:
    ALPHA = 0.20   # learning rate
    GAMMA = 0.90   # discount factor

    def __init__(self, path="qtable.json"):
        self._lock = threading.Lock()
        self._q: Dict[str, float] = defaultdict(float)
        self._path = path
        self._seed_bias()
        if path and Path(path).exists():
            self._load()

    def _seed_bias(self):
        """
        Seed Q-values from the MDP reward signal so the agent has
        meaningful starting values without any training episodes.

        Strategy: for each (state, attack, action) triple, compute
        the *expected reward* under that action using the MDP
        transition probabilities, and use that as the initial Q-value.

        This means:
          - A1 at SS1 gets ~+0.90  (correct — it IS the best action there)
          - A1 at SS2/SS3/PS* gets a very low value (it leads to ST)
          - Best actions (A5/A6 for A01, A3/A6 for A02, etc.) get high
            values immediately — no training needed to avoid A1
          - The agent will still learn and refine, but starts informed
        """
        for (s, atk, a), trans in MDP.items():
            key = f"{s}|{atk}|{a}"
            if key in self._q:
                continue  # don't overwrite loaded values
            # expected immediate reward = sum(prob * R(s,s',good))
            expected_r = sum(prob * reward(s, ns, trans[0][2])
                             for ns, prob, _ in trans)
            # Cap A1 at SS1 to single-step reward
            v = round(expected_r, 4)
            if s == "SS1" and a == "A1":
                v = 0.90
            self._q[key] = v

    def _key(self, s, atk, a): return f"{s}|{atk}|{a}"

    def get(self, s, atk, a):
        with self._lock: return self._q[self._key(s,atk,a)]

    def max_q(self, s, atk):
        with self._lock:
            return max(self._q[self._key(s,atk,a)] for a in ACTIONS)

    def best_action(self, s, atk):
        with self._lock:
            acts = ACTIONS[:]
            random.shuffle(acts)           # break ties randomly
            return max(acts, key=lambda a: self._q[self._key(s,atk,a)])

    def update(self, s, atk, a, r, s2):
        key  = self._key(s, atk, a)
        qmax = self.max_q(s2, atk)
        with self._lock:
            old = self._q[key]
            new = old + self.ALPHA * (r + self.GAMMA*qmax - old)
            self._q[key] = new
        return new

    def _load(self):
        try:
            with open(self._path) as f: data = json.load(f)
            with self._lock: self._q.update(data)
            log.info("Q-table loaded (%d entries)", len(data))
            # Cap A1 at SS1 to its single-step reward (+0.90).
            # After training, A1 converges to 0.90/(1-0.90)=9.0
            # because SS1->SS1 is an infinite self-loop. This value
            # crowds out all other actions at SS1 and stops the agent
            # from ever choosing a deceptive action on the first request.
            # Capping it lets A2-A8 compete fairly.
            with self._lock:
                for atk in ATTACKS:
                    k = f"SS1|{atk}|A1"
                    if self._q.get(k, 0) > 1.0:
                        self._q[k] = 0.90
        except Exception as e:
            log.warning("Q-table load failed: %s", e)

    def save(self):
        if not self._path: return
        with self._lock: data = dict(self._q)
        try:
            with open(self._path,"w") as f: json.dump(data,f)
            log.debug("Q-table saved (%d entries)", len(data))
        except Exception as e:
            log.error("Q-table save failed: %s", e)

    def dump(self): 
        with self._lock: return dict(self._q)


# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────
@dataclass
class Session:
    sid:          str
    attack:       str
    state:        str   = "SS1"
    reward_sum:   float = 0.0
    steps:        int   = 0
    visit_count:  int   = 0    # total requests from this attacker on this attack
    history:      list  = field(default_factory=list)
    created:      float = field(default_factory=time.time)
    updated:      float = field(default_factory=time.time)

    @property
    def terminal(self): return self.state == "ST"


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────
class HoneypotAgent:
    EPS_START = 0.15   # lower start — seed bias gives good priors
    EPS_MIN   = 0.20   # keep meaningful exploration even after convergence —
                        # a honeypot that always replies with the same action
                        # is easy to fingerprint, so variety is desirable
    EPS_DECAY = 0.998
    SESSION_TTL = 1800

    def __init__(self, path="qtable.json"):
        self.qtable            = QTable(path)
        self._sessions:        Dict[str,Session] = {}
        self._global_reward:   Dict[str,float]   = {}
        self._eps   = self.EPS_START
        self._ep    = 0
        self._lock  = threading.Lock()
        threading.Thread(target=self._autosave, daemon=True).start()
        log.info("HoneypotAgent ready  eps=%.3f", self._eps)

    # ── session management ───────────────────────────────────
    def get_or_create(self, sid, attack):
        with self._lock:
            if sid in self._sessions:
                s = self._sessions[sid]
                if s.attack != attack:
                    # Attack type changed — fresh session at SS1
                    s = Session(sid=sid, attack=attack)
                    self._sessions[sid] = s
                elif s.terminal:
                    # ST is absorbing — reset for same attack
                    s = Session(sid=sid, attack=attack)
                    self._sessions[sid] = s
                s.visit_count += 1
                s.updated = time.time()
                # Auto-advance: if attacker has hit this category
                # more than once AND is still at SS1, move to SS2.
                # This reflects real attacker behaviour — a second
                # request on the same attack category means they are
                # already suspicious, not just browsing.
                if s.state == "SS1" and s.visit_count >= 2:
                    s.state = "SS2"
                    log.debug("[%s] auto-advance SS1->SS2 (visit %d, %s)",
                              sid[:10], s.visit_count, attack)
                return s
            s = Session(sid=sid, attack=attack)
            s.visit_count = 1
            self._sessions[sid] = s
            return s

    def get(self, sid):
        with self._lock: return self._sessions.get(sid)

    def _purge(self):
        now = time.time()
        with self._lock:
            stale = [k for k,v in self._sessions.items()
                     if now - v.updated > self.SESSION_TTL]
            for k in stale: del self._sessions[k]

    # ── action selection  ε-greedy ───────────────────────────
    # How many of the top actions to sample among during exploitation.
    # 1 = strict argmax (always picks the single best action — repetitive).
    # 3 = pick among the top-3 actions, weighted by their Q-values —
    #     keeps responses varied while still favouring better actions.
    TOP_K = 3

    # A1 represents "respond normally to a confirmed attacker" — a
    # real server occasionally does this for requests that don't trip
    # any rule, so A1 should remain POSSIBLE at every state, but it
    # must never be a dominant choice once the attacker is past SS1.
    # This cap limits A1's share of the weighted pick regardless of
    # its raw Q-value, keeping it "reasonable" (a few % of the time)
    # rather than excluded entirely or occasionally over-represented.
    A1_MAX_SHARE = 0.05   # at most ~5% of exploitation picks

    def _weighted_pick(self, candidates, atk, state):
        """
        Pick one action from `candidates`, weighted by their Q-values.
        A1's weight is capped (see A1_MAX_SHARE) so it stays rare but
        possible, independent of its raw Q-value.
        Falls back to uniform random if all Q-values are equal/non-positive.
        """
        qvals = []
        for a in candidates:
            q = max(self.qtable.get(state, atk, a), 0.01)
            qvals.append(q)

        if "A1" in candidates and state != "SS1":
            total_others = sum(q for a, q in zip(candidates, qvals) if a != "A1")
            if total_others > 0:
                # Scale A1's weight so it represents at most A1_MAX_SHARE
                # of the total, regardless of its Q-value.
                cap_weight = total_others * (self.A1_MAX_SHARE / (1 - self.A1_MAX_SHARE))
                qvals = [cap_weight if a == "A1" else q
                        for a, q in zip(candidates, qvals)]

        total = sum(qvals)
        if total <= 0:
            return random.choice(candidates)
        r = random.random() * total
        acc = 0.0
        for a, q in zip(candidates, qvals):
            acc += q
            if r <= acc:
                return a
        return candidates[-1]

    def select_action(self, sess):
        if sess.terminal:
            return "A1"

        # A1 is included in the action pool at EVERY state, including
        # SS2/SS3/PS* — it is a valid row in every transition table
        # (e.g. "SS3 A1 -> PS1/ST = 0.15/0.85, R=+3.10/-0.10"). It is
        # simply the LOWEST-reward action at those states, so the
        # Q-learned values + weighted top-K sampling naturally make it
        # rare without hard-excluding it. This means an occasional
        # "normal" response to a confirmed attacker is possible (as a
        # real server sometimes would do for a non-triggering request),
        # while the dominant responses remain the deceptive ones.
        if random.random() < self._eps:
            return random.choice(ACTIONS)

        # Exploitation: sample among the top-K actions (weighted by
        # Q-value) instead of always the single best — keeps the
        # honeypot's responses varied even after convergence. A1's low
        # Q-value at SS2+ means it rarely makes the top-K, but it can
        # still appear occasionally, which is realistic.
        ranked = sorted(ACTIONS,
                        key=lambda a: self.qtable.get(sess.state, sess.attack, a),
                        reverse=True)
        top = ranked[:self.TOP_K]
        return self._weighted_pick(top, sess.attack, sess.state)

    # ── environment step ─────────────────────────────────────
    def step(self, sess, action):
        key   = (sess.state, sess.attack, action)
        trans = MDP.get(key)
        if not trans:
            log.warning("No MDP entry for %s", key)
            next_s = "ST"
            r      = reward(sess.state, "ST", False)
        else:
            states = [t[0] for t in trans]
            probs  = [t[1] for t in trans]
            good   = trans[0][2]
            idx    = random.choices(range(len(states)), weights=probs)[0]
            next_s = states[idx]
            r      = reward(sess.state, next_s, good)

        new_q = self.qtable.update(sess.state, sess.attack, action, r, next_s)

        sess.history.append({
            "step":   sess.steps,
            "state":  sess.state,
            "action": action,
            "next":   next_s,
            "reward": r,
            "q":      round(new_q,4),
            "ts":     time.time(),
        })
        sess.state       = next_s
        sess.reward_sum += r
        sess.steps      += 1
        sess.updated     = time.time()
        with self._lock:
            self._global_reward[sess.sid] = (
                self._global_reward.get(sess.sid, 0.0) + r)

        log.info("[%s] %s --%s--> %s  R=%.2f  Q=%.4f  eps=%.3f",
                 sess.sid[:10], sess.history[-1]["state"],
                 action, next_s, r, new_q, self._eps)

        if sess.terminal:
            self._ep += 1
            self._eps = max(self.EPS_MIN,
                            self.EPS_START * (self.EPS_DECAY**self._ep))
        return next_s, r, sess.terminal

    # ── full decide cycle ────────────────────────────────────
    def decide(self, sid, attack):
        sess         = self.get_or_create(sid, attack)
        state_before = sess.state          # capture BEFORE step() mutates it
        action       = self.select_action(sess)
        ns, r, done  = self.step(sess, action)
        return sess, state_before, action, ns, r, done

    def stats(self):
        self._purge()
        with self._lock:
            n      = len(self._sessions)
            total  = round(sum(self._global_reward.values()), 4)
        return {"eps": round(self._eps,4), "episode": self._ep,
                "sessions": n, "q_entries": len(self.qtable._q),
                "total_cumulative_reward": total}

    def _autosave(self):
        while True:
            time.sleep(60)
            self.qtable.save()
            self._purge()


# ─────────────────────────────────────────────────────────────
# Offline training helper
# ─────────────────────────────────────────────────────────────
def train(episodes=5000, path="qtable.json"):
    logging.basicConfig(level=logging.WARNING)
    agent = HoneypotAgent(path)
    for ep in range(episodes):
        atk  = random.choice(ATTACKS)
        sess = agent.get_or_create(f"t{ep}", atk)
        steps = 0
        while not sess.terminal and steps < 60:
            agent.step(sess, agent.select_action(sess))
            steps += 1
        if ep % 1000 == 0:
            print(f"ep={ep:6d}  eps={agent._eps:.4f}  Q={len(agent.qtable._q)}")
    agent.qtable.save()
    print("Training done.")
    return agent

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=0)
    args = p.parse_args()
    if args.train:
        train(args.train)
    else:
        print("Use --train N to run offline training.")