"""Mock tests for LeRobotWriter -- LeRobotDataset is fully mocked."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from lerobot.datasets.utils import validate_frame

from dexim.recorder.metadata import EpisodeMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(index: int = 0) -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_index=index,
        task_id="pick_cup",
        task_description="Pick up the red cup",
        start_time=0.0,
        end_time=2.0,
        num_frames=3,
        num_topics=1,
        topics=["obs/arm/joint_state"],
    )


def _make_frames(
    n: int = 3, topic: str = "obs/arm/joint_state"
) -> list[dict[str, Any]]:
    return [{"timestamp": float(i), topic: [float(i)] * 6} for i in range(n)]


@pytest.fixture()
def mock_lerobot_dataset():
    """Mock LeRobotDataset with add_frame, save_episode, consolidate, push_to_hub."""
    ds = MagicMock()
    ds.add_frame = MagicMock()
    ds.save_episode = MagicMock()
    ds.finalize = MagicMock()
    ds.push_to_hub = MagicMock()
    ds.features = {}
    return ds


@pytest.fixture()
def patch_lerobot_dataset(mock_lerobot_dataset):
    """Patch LeRobotDataset.create to return the mock dataset."""
    mock_cls = MagicMock(return_value=mock_lerobot_dataset)
    mock_cls.create = MagicMock(return_value=mock_lerobot_dataset)
    with patch.dict(
        sys.modules,
        {
            "lerobot": MagicMock(),
            "lerobot.datasets": MagicMock(),
            "lerobot.datasets.lerobot_dataset": MagicMock(LeRobotDataset=mock_cls),
        },
    ):
        yield mock_lerobot_dataset, mock_cls


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_empty_topic_to_feature_raises(self, patch_lerobot_dataset) -> None:
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        with pytest.raises(ValueError, match="topic_to_feature"):
            LeRobotWriter(
                dataset_path="/tmp/ds",
                repo_id="org/ds",
                features=None,
                topic_to_feature={},
            )

    def test_missing_lerobot_raises_import_error(self) -> None:
        # Remove lerobot from sys.modules to simulate ImportError
        with patch.dict(sys.modules, {"lerobot": None}):
            # Need to force re-import so the ImportError is triggered

            if "dexim.recorder.backends.lerobot_writer" in sys.modules:
                del sys.modules["dexim.recorder.backends.lerobot_writer"]

            with pytest.raises((ImportError, AttributeError)):
                from dexim.recorder.backends.lerobot_writer import LeRobotWriter

                LeRobotWriter(
                    dataset_path="/tmp/ds",
                    repo_id="org/ds",
                    features=None,
                    topic_to_feature={"obs/arm/joint_state": "state"},
                )

    def test_list_shapes_normalized_to_tuples(self, patch_lerobot_dataset) -> None:
        _, mock_cls = patch_lerobot_dataset
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        features = {"state": {"dtype": "float32", "shape": [6]}}
        LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features=features,
            topic_to_feature={"obs/arm/joint_state": "state"},
        )

        call_kwargs = mock_cls.create.call_args.kwargs
        assert call_kwargs["features"]["state"]["shape"] == (6,)

    def test_none_features_normalized_to_empty_dict(
        self, patch_lerobot_dataset
    ) -> None:
        _, mock_cls = patch_lerobot_dataset
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features=None,
            topic_to_feature={"obs/arm/joint_state": "state"},
        )

        call_kwargs = mock_cls.create.call_args.kwargs
        assert call_kwargs["features"] == {}

    def test_creates_dataset_when_path_missing(
        self, patch_lerobot_dataset, tmp_path
    ) -> None:
        _, mock_cls = patch_lerobot_dataset
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        new_path = str(tmp_path / "new_ds")  # does not exist on disk
        LeRobotWriter(
            dataset_path=new_path,
            repo_id="org/ds",
            features=None,
            topic_to_feature={"obs/arm/joint_state": "state"},
        )

        mock_cls.create.assert_called_once()
        mock_cls.assert_not_called()

    def test_encoding_configuration_is_forwarded(
        self, patch_lerobot_dataset, tmp_path
    ) -> None:
        _, mock_cls = patch_lerobot_dataset
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        LeRobotWriter(
            dataset_path=str(tmp_path / "new_ds"),
            repo_id="org/ds",
            features=None,
            topic_to_feature={"obs/arm/joint_state": "state"},
            vcodec="h264",
            parallel_encoding=True,
            encoder_threads=2,
        )

        call_kwargs = mock_cls.create.call_args.kwargs
        assert call_kwargs["vcodec"] == "h264"
        assert call_kwargs["encoder_threads"] == 2

    def test_loads_dataset_when_path_exists(
        self, patch_lerobot_dataset, tmp_path
    ) -> None:
        _, mock_cls = patch_lerobot_dataset
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        # Populate the minimum files that _is_complete_dataset requires
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{}")
        (tmp_path / "meta" / "tasks.parquet").write_bytes(b"")
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)
        (tmp_path / "data" / "chunk-000" / "file-000.parquet").write_bytes(b"")

        LeRobotWriter(
            dataset_path=str(tmp_path),
            repo_id="org/ds",
            features=None,
            topic_to_feature={"obs/arm/joint_state": "state"},
        )

        mock_cls.assert_called_once()
        mock_cls.create.assert_not_called()

    def test_incomplete_dataset_raises(self, patch_lerobot_dataset, tmp_path) -> None:
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        # Stub dataset with only info.json (no tasks.parquet, no data/)
        (tmp_path / "meta").mkdir()
        sentinel = tmp_path / "meta" / "info.json"
        sentinel.write_text("{}")
        with pytest.raises(RuntimeError, match="Incomplete LeRobot dataset"):
            LeRobotWriter(
                dataset_path=str(tmp_path),
                repo_id="org/ds",
                features=None,
                topic_to_feature={"obs/arm/joint_state": "state"},
            )
        assert sentinel.read_text() == "{}"


# ---------------------------------------------------------------------------
# _map_frame
# ---------------------------------------------------------------------------


class TestMapFrame:
    def _make_writer(self, patch_lerobot_dataset):
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        return LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": None,
                },
                "action": {"dtype": "float32", "shape": [6], "names": None},
            },
            topic_to_feature={
                "obs/arm/joint_state": "observation.state",
                "action/arm/joint_cmd": "action",
            },
        )

    def test_joint_payload_extracts_q(self, patch_lerobot_dataset) -> None:
        writer = self._make_writer(patch_lerobot_dataset)
        frame = {
            "obs/arm/joint_state": {
                "q": [1.0] * 6,
                "qd": [0.1] * 6,
                "tau": [0.2] * 6,
                "stamp": 1.0,
            },
            "action/arm/joint_cmd": {
                "q": [0.5] * 6,
            },
        }

        result = writer._map_frame(frame)

        np.testing.assert_array_equal(
            result["observation.state"], np.asarray([1.0] * 6, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            result["action"], np.asarray([0.5] * 6, dtype=np.float32)
        )

    def test_mapped_topics_translated(self, patch_lerobot_dataset) -> None:
        writer = self._make_writer(patch_lerobot_dataset)
        frame = {
            "timestamp": 1.0,
            "obs/arm/joint_state": [1.0] * 6,
            "action/arm/joint_cmd": [0.5] * 6,
        }
        result = writer._map_frame(frame)
        assert "observation.state" in result
        assert "action" in result

    @pytest.mark.parametrize(
        ("topic", "feature_name", "dtype"),
        [
            ("action/nova_left/joint_cmd", "action.nova_left", "float32"),
            (
                "observation/nova_left/joint_state",
                "observation.state.nova_left",
                "float64",
            ),
            ("observation/nova_left/joint_state", "custom.joints", "float32"),
        ],
    )
    def test_numeric_joint_feature_follows_schema(
        self,
        patch_lerobot_dataset,
        topic: str,
        feature_name: str,
        dtype: str,
    ) -> None:
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        features = {feature_name: {"dtype": dtype, "shape": [6], "names": None}}
        writer = LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features=features,
            topic_to_feature={topic: feature_name},
        )

        result = writer._map_frame({topic: {"q": [0.5] * 6}})

        expected = np.asarray([0.5] * 6, dtype=dtype)
        np.testing.assert_array_equal(result[feature_name], expected)
        validate_frame(
            {feature_name: result[feature_name], "task": "test"},
            {feature_name: {**features[feature_name], "shape": (6,)}},
        )

    def test_unmapped_topics_dropped(self, patch_lerobot_dataset) -> None:
        writer = self._make_writer(patch_lerobot_dataset)
        frame = {
            "timestamp": 1.0,
            "obs/arm/joint_state": [1.0] * 6,
            "unknown/topic": "should_be_dropped",
        }
        result = writer._map_frame(frame)
        assert "unknown/topic" not in result
        assert "timestamp" not in result

    def test_none_values_not_included(self, patch_lerobot_dataset) -> None:
        writer = self._make_writer(patch_lerobot_dataset)
        frame = {"timestamp": 1.0, "obs/arm/joint_state": None}
        result = writer._map_frame(frame)
        assert "observation.state" not in result

    def test_empty_frame_returns_empty_dict(self, patch_lerobot_dataset) -> None:
        writer = self._make_writer(patch_lerobot_dataset)
        result = writer._map_frame({"timestamp": 0.0})
        assert result == {}


# ---------------------------------------------------------------------------
# write_episode
# ---------------------------------------------------------------------------


class TestWriteEpisode:
    def _make_writer(self, patch_lerobot_dataset):
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        return LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": None,
                }
            },
            topic_to_feature={"obs/arm/joint_state": "observation.state"},
        )

    def test_add_frame_called_per_frame(self, patch_lerobot_dataset) -> None:
        mock_ds, _ = patch_lerobot_dataset
        writer = self._make_writer(patch_lerobot_dataset)
        frames = _make_frames(n=4)

        writer.write_episode(frames, _make_metadata())

        assert mock_ds.add_frame.call_count == 4

    def test_save_episode_called_once(self, patch_lerobot_dataset) -> None:
        mock_ds, _ = patch_lerobot_dataset
        writer = self._make_writer(patch_lerobot_dataset)

        writer.write_episode(_make_frames(3), _make_metadata())

        mock_ds.save_episode.assert_called_once_with(parallel_encoding=False)

    def test_parallel_encoding_is_explicit_opt_in(self, patch_lerobot_dataset) -> None:
        mock_ds, _ = patch_lerobot_dataset
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        writer = LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features=None,
            topic_to_feature={"obs/arm/joint_state": "observation.state"},
            parallel_encoding=True,
        )

        writer.write_episode(_make_frames(1), _make_metadata())

        mock_ds.save_episode.assert_called_once_with(parallel_encoding=True)

    def test_empty_frames_skipped(self, patch_lerobot_dataset) -> None:
        mock_ds, _ = patch_lerobot_dataset
        writer = self._make_writer(patch_lerobot_dataset)

        writer.write_episode([], _make_metadata())

        mock_ds.add_frame.assert_not_called()
        mock_ds.save_episode.assert_not_called()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    def _make_writer(self, patch_lerobot_dataset, push_to_hub: bool = False):
        from dexim.recorder.backends.lerobot_writer import LeRobotWriter

        return LeRobotWriter(
            dataset_path="/tmp/ds",
            repo_id="org/ds",
            features=None,
            topic_to_feature={"obs/arm/joint_state": "observation.state"},
            push_to_hub=push_to_hub,
        )

    def test_consolidate_called(self, patch_lerobot_dataset) -> None:
        mock_ds, _ = patch_lerobot_dataset
        writer = self._make_writer(patch_lerobot_dataset)
        writer.close()
        mock_ds.finalize.assert_called_once()

    def test_push_to_hub_called_when_flag_true(self, patch_lerobot_dataset) -> None:
        mock_ds, _ = patch_lerobot_dataset
        writer = self._make_writer(patch_lerobot_dataset, push_to_hub=True)
        writer.close()
        mock_ds.push_to_hub.assert_called_once()

    def test_push_to_hub_not_called_when_flag_false(
        self, patch_lerobot_dataset
    ) -> None:
        mock_ds, _ = patch_lerobot_dataset
        writer = self._make_writer(patch_lerobot_dataset, push_to_hub=False)
        writer.close()
        mock_ds.push_to_hub.assert_not_called()
