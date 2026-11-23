import json
from pathlib import Path

import antezero_sim as az
import torch
import pytest

from antezero_agent.config import ModelConfig
from antezero_agent.models import Run95Net, load_text_embeddings

DATA = Path(__file__).resolve().parents[1] / "data"


def test_load_text_embeddings_maps_all_tables():
    cfg = ModelConfig(arch="run95", obs_v2=True, d_model=384, n_layers=2, n_heads=12, embed_from="auto")
    model = Run95Net(cfg)
    loaded = load_text_embeddings(model, cfg.embed_from)
    assert set(loaded) == {
        "tok_emb[1]",
        "tok_emb[5]",
        "tok_emb[6]",
        "glob_emb[3]",
        "glob_emb[4]",
        "glob_emb[5]",
        "glob_emb[6]",
        "glob_emb[7]",
        "glob_emb[8]",
        "glob_emb[9]",
    }

    sd = torch.load(DATA / "text_embeddings_384_v2.pt", map_location="cpu")
    ids = json.loads((DATA / "ids.json").read_text())
    cids = {name: cid for name, cid, _ in az.centers()}

    assert torch.equal(model.tok_emb[1].weight[1], sd["center.weight"][ids["center"]["c_base"] + 1])
    for name in ("j_joker", "j_blueprint", "c_hermit", "v_grabber"):
        assert torch.equal(model.tok_emb[1].weight[cids[name]], sd["center.weight"][ids["center"][name] + 1])
        assert torch.equal(model.glob_emb[8].weight[cids[name]], sd["center.weight"][ids["center"][name] + 1])
        assert torch.equal(model.glob_emb[9].weight[cids[name]], sd["center.weight"][ids["center"][name] + 1])

    assert torch.equal(model.tok_emb[1].weight[268], sd["tag.weight"][ids["tag"]["tag_uncommon"] + 1])
    assert torch.equal(model.tok_emb[1].weight[292], sd["blind.weight"][ids["blind"]["bl_arm"] + 1])
    assert torch.equal(model.tok_emb[1].weight[322], sd["hand_emb.weight"][0])

    assert torch.equal(model.tok_emb[5].weight[:5], sd["edition.weight"][:5])
    assert torch.equal(model.tok_emb[6].weight[:5], sd["seal.weight"][:5])

    assert torch.equal(model.glob_emb[3].weight[1], sd["blind.weight"][ids["blind"]["bl_arm"] + 1])
    assert torch.equal(model.glob_emb[4].weight[1], sd["boss.weight"][ids["blind"]["bl_arm"] + 1])
    assert torch.equal(model.glob_emb[5].weight[1], sd["tag.weight"][ids["tag"]["tag_uncommon"] + 1])
    assert torch.equal(model.glob_emb[6].weight[1], sd["tag.weight"][ids["tag"]["tag_uncommon"] + 1])

    assert torch.equal(model.glob_emb[7].weight[0], sd["hand_emb.weight"][12])
    assert torch.equal(model.glob_emb[7].weight[1:13], sd["hand_emb.weight"][:12])


def test_load_text_embeddings_dim_mismatch():
    cfg = ModelConfig(arch="run95", obs_v2=True, d_model=256, n_layers=2, n_heads=8)
    model = Run95Net(cfg)
    with pytest.raises(ValueError, match="d_model=256"):
        load_text_embeddings(model, str(DATA / "text_embeddings_384_v2.pt"))
