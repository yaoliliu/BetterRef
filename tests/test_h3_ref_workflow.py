"""Exercise the H3 reference workflow's clause binding and per-patch override rules on CPU."""

import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def core():
    name = "_h3_ref_workflow_test"
    pkg = ModuleType(name)
    pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "freefuse_core")]
    sys.modules[name] = pkg
    try:
        yield importlib.import_module(name + ".h3_ref_workflow")
    finally:
        for key in list(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


def role(name, target, reference="", mask=None):
    return dict(name=name, target=target, reference=reference, target_mask=mask, reference_mask=None)


def test_reference_scope_excludes_repeated_scene_phrase(core):
    scene = "花窗前，红衣公子向公子点头。"
    roles = [role("A", "公子", "红衣公子"), role("B", "红衣公子"), role("旁观者", "")]
    prompt, target, refs = core.bind_prompt(scene, roles, "花窗")
    assert len(target["A"]) == 2
    assert refs["B"] == target["B"]
    assert min(a for a, b in refs["A"]) >= len(scene.encode())
    assert all(prompt.encode()[a:b].decode() == "红衣公子" for a, b in refs["A"])
    assert "旁观者" not in target and "旁观者" not in refs
    with pytest.raises(ValueError, match="verbatim"):
        core.bind_prompt(scene, [role("A", "不存在")], "花窗")


def test_painted_role_reclaims_foreign_automatic_mask(core):
    refs = [dict(latent_h=4, latent_w=4)] * 2
    geom = core.packed_geometry(8, refs, ("A", "B"), 64, 64)
    one = torch.ones(1, 2, 2)
    edit = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    auto = {"target": {"A": one * 0, "B": one}, "refs": {n: {n: one} for n in ("A", "B")}}
    roles = [role("A", "first", mask=edit), role("B", "second")]
    masks, applied, conflict, names = core.merge_masks(geom, roles, auto)
    assert torch.equal(masks["target"]["A"], edit)
    assert torch.equal(masks["target"]["B"], 1 - edit)
    assert applied == ["A", "B"] and not names and not conflict.any()
    roles[1]["target_mask"] = edit
    roles[0]["reference_mask"] = roles[1]["reference_mask"] = one
    _, _, conflict, names = core.merge_masks(geom, roles, None)
    assert torch.equal(conflict, edit.bool()) and names == ["A", "B"]


def test_masks_can_be_fully_manual_with_an_unbound_reference(core):
    refs = [dict(latent_h=4, latent_w=4), dict(latent_h=4, latent_w=6)]
    geometry = core.packed_geometry(32, refs, ("宝玉", "旁观者"), 64, 96)
    foreground = torch.ones(1, 64, 96)
    by = role("宝玉", "红衣少年", mask=foreground)
    by["reference_mask"] = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    masks, applied, conflict, names = core.merge_masks(geometry, [by, role("旁观者", "")], None)
    assert applied == ["宝玉"] and not names and not conflict.any()
    assert masks["target"]["宝玉"].shape == (1, 2, 3)
    assert not masks["target"]["旁观者"].any()
    assert masks["refs"]["旁观者"]["__background__"].all()
    assert torch.equal(masks["refs"]["宝玉"]["宝玉"], by["reference_mask"])


def test_editor_mask_round_trip_through_generation_masks(core):
    import json

    editor = importlib.import_module(core.__package__ + ".h3_ref_editor")
    painted = torch.tensor([[[0., 1.], [1., 0.]]])
    empty = torch.zeros_like(painted)
    role_a = role("A", "first")
    previews = editor.editor_context(
        [dict(role_a, editor_id="5", image=torch.zeros(1, 2, 2, 3), external_masks={"target": empty})],
        None, {"target": {"A": empty}, "refs": {"A": {"A": painted}}}, 42, (1, 2, 2)
    )
    assert previews["seed"] == "42" and previews["width"] == 64
    ref = previews["roles"]["5"]
    edits = editor.decode_edits(json.dumps({"reference": ref["referenceAuto"], "target": ref["targetExternal"]}))
    assert torch.equal(edits["reference"], painted)
    assert torch.equal(edits["target"], empty)
    assert editor.decode_edits("{}") == {}
    geometry = core.packed_geometry(8, [dict(latent_h=4, latent_w=4)], ("A",), 64, 64)
    role_a.update(target_mask=edits["reference"], reference_mask=edits["reference"])
    masks, applied, _, _ = core.merge_masks(geometry, [role_a], None)
    assert applied == ["A"] and torch.equal(masks["target"]["A"], painted)
    with pytest.raises(ValueError):
        editor.decode_edits('{"target":"unexpected image format"}')
