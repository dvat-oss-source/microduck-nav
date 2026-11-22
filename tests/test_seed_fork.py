"""A provenance fork must fail on invalid unchanged tasks, without resampling."""
import copy
import pytest
from tools.fork_seeds import validate_tasks, fork


class Grid:
    def is_lethal(self, x, y):
        return x < 0


class Validator:
    rejections = {}
    def set_layout(self, objects):
        self.objects = objects
    def pose_ok(self, seed, pose, placements, exclude=None):
        return True
    def stable(self, seed, start, placements):
        return True


def test_fork_validation_preserves_tasks_and_rejects_invalid_goal():
    doc = {"nav": [{"seed": 1, "start": [1, 1, 0], "goal": [2, 2, 0],
                    "object_poses": {}}], "kick": [], "pick": []}
    before = copy.deepcopy(doc)
    result = validate_tasks(doc, Grid(), Validator())
    assert doc == before
    assert result["unchanged_tasks_checked"] == {"nav": 1, "kick": 0, "pick": 0}
    doc["nav"][0]["goal"][0] = -1
    with pytest.raises(RuntimeError, match="nav_000 goal invalid.*without resampling"):
        validate_tasks(doc, Grid(), Validator())


def test_fork_refuses_existing_destination_before_reading_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("immutable")
    out = tmp_path / "v2.json"
    out.write_text("also immutable")
    with pytest.raises(FileExistsError):
        fork(source, out, "v2", "unused")
    assert source.read_text() == "immutable"
    assert out.read_text() == "also immutable"
