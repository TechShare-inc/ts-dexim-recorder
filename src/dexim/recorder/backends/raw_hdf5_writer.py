"""Lossless raw HDF5 backend for deferred LeRobot conversion.

Each episode is stored before temporal alignment. Every source topic keeps its
own timestamps and MessagePack-encoded payloads. The LeRobot feature schema and
topic mapping are embedded in the file so an offline converter can reproduce
the configured dataset without relying on a separate YAML snapshot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import msgpack
import numpy as np
from loguru import logger

from dexim.recorder.backends.base import StorageBackend
from dexim.recorder.metadata import EpisodeMetadata

__all__ = ["RawHDF5Writer", "read_raw_episode"]

_SCHEMA_VERSION = 1
_VLEN_UINT8 = h5py.vlen_dtype(np.dtype("uint8"))


def _pack_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__dexim_ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported raw payload type: {type(value)!r}")


def _unpack_object(value: dict[str, Any]) -> Any:
    if value.get("__dexim_ndarray__") is True:
        array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
        return array.reshape(tuple(value["shape"])).copy()
    return value


def _encode_payload(value: Any) -> np.ndarray:
    packed = msgpack.packb(value, use_bin_type=True, default=_pack_default)
    return np.frombuffer(packed, dtype=np.uint8)


def _decode_payload(value: np.ndarray) -> Any:
    return msgpack.unpackb(
        bytes(value),
        raw=False,
        object_hook=_unpack_object,
    )


class RawHDF5Writer(StorageBackend):
    """Write one unaligned, lossless HDF5 file per episode."""

    requires_alignment = False

    def __init__(
        self,
        output_dir: str,
        *,
        topic_to_feature: dict[str, str] | None = None,
        features: dict[str, Any] | None = None,
        fps: int = 30,
        master_clock_topic: str | None = None,
        continuous_topics: list[str] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._topic_to_feature = dict(topic_to_feature or {})
        self._features = dict(features or {})
        self._fps = fps
        self._master_clock_topic = master_clock_topic
        self._continuous_topics = list(continuous_topics or [])

    @property
    def next_episode_index(self) -> int:
        """Return the first unused episode index in the output directory."""
        indices: list[int] = []
        for path in self._output_dir.glob("episode_*.raw.h5"):
            try:
                indices.append(int(path.name.removeprefix("episode_").split(".")[0]))
            except ValueError:
                continue
        return max(indices, default=-1) + 1

    def write_episode(
        self,
        frames: dict[str, list[tuple[float, Any]]],
        metadata: EpisodeMetadata,
    ) -> None:
        """Persist raw per-topic buffers without alignment or image decoding."""
        if not frames:
            logger.warning(
                f"Episode {metadata.episode_index}: no raw samples to write -- skipped"
            )
            return

        filepath = self._output_dir / f"episode_{metadata.episode_index:06d}.raw.h5"
        temporary = filepath.with_suffix(filepath.suffix + ".tmp")
        if filepath.exists():
            raise FileExistsError(f"Raw episode already exists: {filepath}")

        try:
            with h5py.File(temporary, "w") as hf:
                hf.attrs["storage_format"] = "dexim_raw_hdf5"
                hf.attrs["schema_version"] = _SCHEMA_VERSION
                hf.attrs["episode_index"] = metadata.episode_index
                hf.attrs["task"] = metadata.task_id
                hf.attrs["task_description"] = metadata.task_description
                hf.attrs["start_time"] = metadata.start_time
                hf.attrs["end_time"] = metadata.end_time
                hf.attrs["fps"] = self._fps
                hf.attrs["master_clock_topic"] = self._master_clock_topic or ""
                hf.attrs["topics_json"] = json.dumps(
                    sorted(frames), ensure_ascii=False
                )
                hf.attrs["topic_to_feature_json"] = json.dumps(
                    self._topic_to_feature, ensure_ascii=False, sort_keys=True
                )
                hf.attrs["features_json"] = json.dumps(
                    self._features, ensure_ascii=False, sort_keys=True
                )
                hf.attrs["continuous_topics_json"] = json.dumps(
                    self._continuous_topics, ensure_ascii=False
                )

                streams = hf.create_group("streams")
                total_samples = 0
                for stream_index, topic in enumerate(sorted(frames)):
                    samples = frames[topic]
                    group = streams.create_group(f"{stream_index:04d}")
                    group.attrs["topic"] = topic
                    group.attrs["feature"] = self._topic_to_feature.get(topic, "")
                    timestamps = np.asarray(
                        [timestamp for timestamp, _ in samples], dtype=np.float64
                    )
                    group.create_dataset("timestamp", data=timestamps)
                    payloads = group.create_dataset(
                        "payload",
                        shape=(len(samples),),
                        dtype=_VLEN_UINT8,
                    )
                    for sample_index, (_, payload) in enumerate(samples):
                        payloads[sample_index] = _encode_payload(payload)
                    total_samples += len(samples)
                hf.attrs["num_topics"] = len(frames)
                hf.attrs["num_samples"] = total_samples
                hf.flush()
            os.replace(temporary, filepath)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

        logger.info(
            f"Raw episode {metadata.episode_index} -> {filepath} "
            f"({len(frames)} topics, {total_samples} samples)"
        )

    def close(self) -> None:
        """No-op because every raw episode is an independent atomic file."""


def read_raw_episode(
    filepath: str | Path,
) -> tuple[dict[str, list[tuple[float, Any]]], EpisodeMetadata, dict[str, Any]]:
    """Read a raw episode for inspection or future offline conversion."""
    buffers: dict[str, list[tuple[float, Any]]] = {}
    with h5py.File(filepath, "r") as hf:
        if hf.attrs.get("storage_format") != "dexim_raw_hdf5":
            raise ValueError(f"Not a Dexim raw HDF5 episode: {filepath}")
        if int(hf.attrs.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported raw HDF5 schema version: "
                f"{hf.attrs.get('schema_version')!r}"
            )
        for group in hf["streams"].values():
            topic = str(group.attrs["topic"])
            timestamps = group["timestamp"][:]
            payloads = group["payload"][:]
            buffers[topic] = [
                (float(timestamp), _decode_payload(payload))
                for timestamp, payload in zip(timestamps, payloads, strict=True)
            ]
        metadata = EpisodeMetadata(
            episode_index=int(hf.attrs["episode_index"]),
            task_id=str(hf.attrs["task"]),
            task_description=str(hf.attrs["task_description"]),
            start_time=float(hf.attrs["start_time"]),
            end_time=float(hf.attrs["end_time"]),
            num_frames=0,
            num_topics=int(hf.attrs["num_topics"]),
            topics=sorted(buffers),
        )
        schema = {
            "fps": int(hf.attrs["fps"]),
            "master_clock_topic": str(hf.attrs["master_clock_topic"]) or None,
            "topic_to_feature": json.loads(hf.attrs["topic_to_feature_json"]),
            "features": json.loads(hf.attrs["features_json"]),
            "continuous_topics": json.loads(
                hf.attrs.get("continuous_topics_json", "[]")
            ),
        }
    return buffers, metadata, schema
