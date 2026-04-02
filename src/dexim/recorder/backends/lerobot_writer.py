"""LeRobot v0.4.x storage backend.

Appends episodes to a persistent ``LeRobotDataset`` (Parquet + video).
All dataset operations are guarded by a ``threading.Lock`` to be safe
against future API changes that might call the backend concurrently.

Topic → LeRobot feature name mapping is the sole supported path
(``topic_to_feature`` config dict).  The legacy alphabetic-concatenation
path from the old ``data-recorder-node`` is intentionally not ported.
"""

from __future__ import annotations

import io
import shutil
import threading
from pathlib import Path
from typing import Any

from loguru import logger

from dexim.recorder.backends.base import StorageBackend
from dexim.recorder.metadata import EpisodeMetadata

__all__ = ["LeRobotWriter"]


class LeRobotWriter(StorageBackend):
    """Appends episodes to a persistent ``LeRobotDataset``.

    Args:
        dataset_path: Root path for the LeRobotDataset directory.
        repo_id: HuggingFace repo ID (e.g. ``"org/dataset-name"``).
        features: Feature schema dict required when creating a new dataset.
            Shapes may be lists (converted to tuples internally).
        topic_to_feature: Mapping from topic string to LeRobot feature name.
        fps: Dataset frame rate.
        push_to_hub: Push dataset to HuggingFace Hub on ``close()``.
        robot_type: Robot type string embedded in dataset metadata.

    Raises:
        ImportError: When ``lerobot`` is not installed.
        ValueError: When ``topic_to_feature`` is empty.
    """

    def __init__(
        self,
        dataset_path: str,
        repo_id: str,
        features: dict[str, Any],
        topic_to_feature: dict[str, str],
        fps: int = 30,
        push_to_hub: bool = False,
        robot_type: str = "dexim-dualarm",
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._LeRobotDataset = LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "lerobot is required for the LeRobot storage backend. "
                "Install it with: pip install lerobot"
            ) from exc

        if not topic_to_feature:
            raise ValueError(
                "topic_to_feature must map at least one topic to a LeRobot feature name"
            )

        self._topic_to_feature = topic_to_feature
        self._push_to_hub = push_to_hub
        self._lock = threading.Lock()

        # Normalise list shapes → tuples for the LeRobot API
        processed_features: dict[str, Any] = {}
        for k, v in (features or {}).items():
            if isinstance(v, dict) and isinstance(v.get("shape"), list):
                processed_features[k] = {**v, "shape": tuple(v["shape"])}
            else:
                processed_features[k] = v

        root = Path(dataset_path)
        if root.exists() and not self._is_complete_dataset(root):
            logger.warning(
                f"LeRobotWriter: incomplete dataset found at {dataset_path!r} — removing and recreating"
            )
            shutil.rmtree(root)

        if root.exists():
            self._dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_path,
                tolerance_s=1e-4,
            )
            logger.info(
                f"LeRobotWriter: loaded existing dataset at {dataset_path!r} "
                f"(repo_id={repo_id!r})"
            )
        else:
            self._dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=dataset_path,
                robot_type=robot_type,
                features=processed_features,
                tolerance_s=1e-4,
                use_videos=True,
            )
            logger.info(
                f"LeRobotWriter: dataset created at {dataset_path!r} "
                f"(repo_id={repo_id!r}, fps={fps})"
            )

    def write_episode(
        self,
        frames: list[dict[str, Any]],
        metadata: EpisodeMetadata,
    ) -> None:
        """Append aligned frames as a new LeRobot episode.

        Args:
            frames: Aligned frame list from ``align_episode_data``.
            metadata: Episode metadata snapshot (index, task, timing, topics).
        """
        if not frames:
            logger.warning(
                f"Episode {metadata.episode_index}: no frames to write — skipped"
            )
            return

        with self._lock:
            try:
                for frame in frames:
                    mapped = self._map_frame(frame)
                    if mapped:
                        mapped["task"] = metadata.task_id
                        self._dataset.add_frame(mapped)
                self._dataset.save_episode()
                logger.info(
                    f"Episode {metadata.episode_index} saved to LeRobotDataset "
                    f"({len(frames)} frames)"
                )
            except Exception as exc:
                logger.error(
                    f"LeRobotWriter: episode {metadata.episode_index} failed — {exc}",
                    exc_info=True,
                )
                raise

    def close(self) -> None:
        """Finalize the dataset and optionally push to HuggingFace Hub."""
        with self._lock:
            try:
                self._dataset.finalize()
                logger.info("LeRobotDataset finalized")
            except Exception as exc:
                logger.error(f"LeRobotWriter.close: finalize failed — {exc}")

            if self._push_to_hub:
                try:
                    self._dataset.push_to_hub()
                    logger.success("LeRobotDataset pushed to Hub")
                except Exception as exc:
                    logger.error(f"LeRobotWriter.close: push_to_hub failed — {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_complete_dataset(root: Path) -> bool:
        """Return True only if *root* contains a finalized LeRobot v3 dataset.

        The minimum required files are:
        - ``meta/info.json``
        - ``meta/tasks.parquet``
        - at least one ``data/`` parquet file
        """
        return (
            (root / "meta" / "info.json").exists()
            and (root / "meta" / "tasks.parquet").exists()
            and any((root / "data").rglob("*.parquet"))
        )

    def _map_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Convert a frame dict from topic keys to LeRobot feature keys.

        Topics absent from ``topic_to_feature`` are silently dropped.
        ``FrameObservation`` dicts (containing JPEG bytes under ``"color"``)
        are decoded to PIL Images before being handed to LeRobot.

        Args:
            frame: Aligned frame dict (topic → data, plus ``"timestamp"``).

        Returns:
            Dict keyed by LeRobot feature names.
        """
        mapped: dict[str, Any] = {}
        for topic, feature_name in self._topic_to_feature.items():
            value = frame.get(topic)
            if value is not None:
                mapped[feature_name] = self._coerce_image(value)
        return mapped

    @staticmethod
    def _coerce_image(value: Any) -> Any:
        """Convert a FrameObservation dict to a PIL Image, pass other values through.

        ``FrameObservation.to_dict()`` serialises the color frame as
        JPEG-compressed bytes under the ``"color"`` key.  LeRobot's
        ``add_frame()`` requires either a ``PIL.Image`` or a numpy array for
        image features, so we decode here.

        Args:
            value: Buffered value — either a raw data dict or a scalar/array.

        Returns:
            PIL RGB image when the value is a ``FrameObservation`` dict;
            otherwise the original value unchanged.
        """
        if not (isinstance(value, dict) and "color" in value):
            return value
        from PIL import Image  # lerobot already requires Pillow

        color_bytes: bytes = value["color"]
        return Image.open(io.BytesIO(color_bytes)).convert("RGB")
