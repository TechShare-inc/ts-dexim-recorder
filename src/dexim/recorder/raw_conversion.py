"""Offline conversion from Dexim raw HDF5 episodes to LeRobot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from dexim.recorder.alignment import align_episode_data
from dexim.recorder.backends.lerobot_writer import LeRobotWriter
from dexim.recorder.backends.raw_hdf5_writer import read_raw_episode

__all__ = ["convert_raw_dataset"]


def _schema_signature(schema: dict[str, Any]) -> tuple[Any, ...]:
    return (
        schema["fps"],
        schema["master_clock_topic"],
        schema["topic_to_feature"],
        schema["features"],
        schema["continuous_topics"],
    )


def convert_raw_dataset(
    input_dir: str,
    dataset_path: str,
    repo_id: str,
    *,
    push_to_hub: bool = False,
    vcodec: str = "libsvtav1",
    parallel_encoding: bool = False,
    encoder_threads: int = 4,
) -> int:
    """Convert every raw episode in a directory into one LeRobot dataset.

    Returns the number of converted episodes. Source files are never modified.
    """
    files = sorted(Path(input_dir).glob("episode_*.raw.h5"))
    if not files:
        raise FileNotFoundError(f"No raw HDF5 episodes found in {input_dir}")
    output_root = Path(dataset_path)
    if output_root.exists():
        raise FileExistsError(
            f"LeRobot output already exists: {output_root}. "
            "Choose a new output directory to avoid duplicate episodes."
        )

    _, _, first_schema = read_raw_episode(files[0])
    writer = LeRobotWriter(
        dataset_path=dataset_path,
        repo_id=repo_id,
        features=first_schema["features"],
        topic_to_feature=first_schema["topic_to_feature"],
        fps=first_schema["fps"],
        push_to_hub=push_to_hub,
        vcodec=vcodec,
        parallel_encoding=parallel_encoding,
        encoder_threads=encoder_threads,
    )

    converted = 0
    try:
        expected_schema = _schema_signature(first_schema)
        for filepath in files:
            buffers, metadata, schema = read_raw_episode(filepath)
            if _schema_signature(schema) != expected_schema:
                raise ValueError(
                    f"Raw episode schema does not match the first file: {filepath}"
                )
            frames = align_episode_data(
                buffers,
                master_clock_topic=schema["master_clock_topic"],
                continuous_topics=schema["continuous_topics"],
            )
            writer.write_episode(frames, metadata)
            converted += 1
            logger.success(
                f"Converted raw episode {metadata.episode_index} "
                f"({len(frames)} aligned frames)"
            )
    finally:
        writer.close()
    return converted
