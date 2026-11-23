"""Initialize antezero embedding tables from RLBalatro text-embedding checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import antezero_sim as az
import torch

from ..model import Net

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# TagKind order in sim/src/data_gen.rs (`ALL_TAGS`).
ANTZERO_TAGS = (
    "tag_uncommon",
    "tag_rare",
    "tag_negative",
    "tag_foil",
    "tag_holo",
    "tag_polychrome",
    "tag_investment",
    "tag_voucher",
    "tag_boss",
    "tag_standard",
    "tag_charm",
    "tag_meteor",
    "tag_buffoon",
    "tag_handy",
    "tag_garbage",
    "tag_ethereal",
    "tag_coupon",
    "tag_double",
    "tag_juggle",
    "tag_d_six",
    "tag_top_up",
    "tag_skip",
    "tag_orbital",
    "tag_economy",
)

# BlindKind order in sim/src/data_gen.rs (`ALL_BLINDS`).
ANTZERO_BLINDS = (
    "bl_arm",
    "bl_big",
    "bl_club",
    "bl_eye",
    "bl_final_acorn",
    "bl_final_bell",
    "bl_final_heart",
    "bl_final_leaf",
    "bl_final_vessel",
    "bl_fish",
    "bl_flint",
    "bl_goad",
    "bl_head",
    "bl_hook",
    "bl_house",
    "bl_manacle",
    "bl_mark",
    "bl_mouth",
    "bl_needle",
    "bl_ox",
    "bl_pillar",
    "bl_plant",
    "bl_psychic",
    "bl_serpent",
    "bl_small",
    "bl_tooth",
    "bl_wall",
    "bl_water",
    "bl_wheel",
    "bl_window",
)


def _resolve_embed_path(path: str, d_model: int) -> Path:
    if path == "auto":
        path = f"data/text_embeddings_{d_model}_v2.pt"
    p = Path(path)
    if p.exists():
        return p
    for base in (Path("agent"), _DATA_DIR.parent):
        cand = base / p
        if cand.exists():
            return cand
    cand = _DATA_DIR / p.name
    if cand.exists():
        return cand
    raise FileNotFoundError(f"embed_from file not found: {path!r}")


def load_text_embeddings(model: Net, path: str, device: torch.device | str = "cpu") -> list[str]:
    """Copy RLBalatro text embeddings into matching ``Net`` / ``Run95Net`` embedding tables.

    Maps ``center.weight``, ``edition.weight``, ``seal.weight``, ``blind.weight``,
    ``boss.weight``, ``tag.weight``, and ``hand_emb.weight`` by name/order using
    ``agent/data/ids.json`` and ``antezero_sim.centers()``.
    """
    src = _resolve_embed_path(path, model.cfg.d_model)
    sd = torch.load(src, map_location=device)
    ids = json.loads((_DATA_DIR / "ids.json").read_text())

    d = model.cfg.d_model
    for key in ("center.weight", "blind.weight", "boss.weight", "tag.weight", "edition.weight", "seal.weight", "hand_emb.weight"):
        if key in sd and sd[key].shape[1] != d:
            raise ValueError(f"embed_from {src}: {key} has dim {sd[key].shape[1]}, expected d_model={d}")

    loaded: list[str] = []
    with torch.no_grad():
        if "center.weight" in sd:
            cw = sd["center.weight"].to(device)
            for name, tbl in (
                ("tok_emb[1]", model.tok_emb[1].weight),
                ("glob_emb[8]", model.glob_emb[8].weight),
                ("glob_emb[9]", model.glob_emb[9].weight),
            ):
                tbl[0].copy_(cw[0])
                tbl[1].copy_(cw[ids["center"]["c_base"] + 1])
                for cname, cid, _ in az.centers():
                    if cname in ids["center"]:
                        tbl[cid].copy_(cw[ids["center"][cname] + 1])
                if "tag.weight" in sd:
                    tw = sd["tag.weight"].to(device)
                    for i, tk in enumerate(ANTZERO_TAGS):
                        tbl[268 + i].copy_(tw[ids["tag"][tk] + 1])
                if "blind.weight" in sd:
                    bw = sd["blind.weight"].to(device)
                    for i, bk in enumerate(ANTZERO_BLINDS):
                        tbl[292 + i].copy_(bw[ids["blind"][bk] + 1])
                if "hand_emb.weight" in sd:
                    hw = sd["hand_emb.weight"].to(device)
                    tbl[322:334].copy_(hw[:12])
                loaded.append(name)

        if "edition.weight" in sd:
            model.tok_emb[5].weight[:5].copy_(sd["edition.weight"][:5].to(device))
            loaded.append("tok_emb[5]")

        if "seal.weight" in sd:
            model.tok_emb[6].weight[:5].copy_(sd["seal.weight"][:5].to(device))
            loaded.append("tok_emb[6]")

        if "blind.weight" in sd:
            bw = sd["blind.weight"].to(device)
            model.glob_emb[3].weight[0].copy_(bw[0])
            for i, bk in enumerate(ANTZERO_BLINDS):
                model.glob_emb[3].weight[1 + i].copy_(bw[ids["blind"][bk] + 1])
            loaded.append("glob_emb[3]")

        if "boss.weight" in sd:
            bsw = sd["boss.weight"].to(device)
            model.glob_emb[4].weight[0].copy_(bsw[0])
            for i, bk in enumerate(ANTZERO_BLINDS):
                model.glob_emb[4].weight[1 + i].copy_(bsw[ids["blind"][bk] + 1])
            loaded.append("glob_emb[4]")

        if "tag.weight" in sd:
            tw = sd["tag.weight"].to(device)
            for name, tbl in (("glob_emb[5]", model.glob_emb[5].weight), ("glob_emb[6]", model.glob_emb[6].weight)):
                tbl[0].copy_(tw[0])
                for i, tk in enumerate(ANTZERO_TAGS):
                    tbl[1 + i].copy_(tw[ids["tag"][tk] + 1])
                loaded.append(name)

        if "hand_emb.weight" in sd:
            hw = sd["hand_emb.weight"].to(device)
            model.glob_emb[7].weight[0].copy_(hw[12])
            model.glob_emb[7].weight[1:13].copy_(hw[:12])
            loaded.append("glob_emb[7]")

    return loaded
