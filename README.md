# dexim-recorder

Multi-stream episode data collection for imitation learning in the DexImitate framework.

## Overview

`dexim-recorder` provides:

- **`DataRecorderNode`** — a `ManagedNode` that subscribes to N ZMQ PUB endpoints, buffers data by topic, and writes complete episodes to disk on `STOP_REC`
- **Temporal alignment** — multi-stream alignment to a master clock (video frame priority, linear interpolation for joint states)
- **HDF5 backend** — one `episode_NNNN.h5` file per episode with gzip-compressed per-topic datasets
- **LeRobot v3 backend** — appends to a persistent `LeRobotDataset` with thread-safe episode writes
- **`EpisodeWriterQueue`** — single daemon thread that serializes alignment + writes (eliminates unbounded thread growth)

## Installation

```bash
pip install dexim-recorder
```

## Usage

```python
from dexim.recorder import DataRecorderNode, RecorderNodeConfig, load_config

config = RecorderNodeConfig(
    node_id="recorder",
    data_endpoints=["tcp://localhost:5600", "tcp://localhost:5601"],
    storage_format="hdf5",
    output_dir="output/episodes",
    task="pick_and_place",
)
node = DataRecorderNode(config)
node.run()  # blocks; responds to CTRL messages for START_REC / STOP_REC / SHUTDOWN
```

Load config from YAML:

```python
config = load_config("config/recorder.yaml")
node = DataRecorderNode(config)
```

## Running Tests

```bash
pixi run test
```

## Architecture

```
dexim.recorder/
  config.py         RecorderNodeConfig dataclass + YAML loader
  alignment.py      align_episode_data() — temporal alignment
  writer_thread.py  EpisodeWriterQueue — serialised background writes
  node.py           DataRecorderNode(ManagedNode)
  backends/
    base.py         StorageBackend ABC
    hdf5_writer.py  HDF5Writer
    lerobot_writer.py  LeRobotWriter
```
# Raw HDF5 capture

Use `raw_hdf5` when consecutive episodes must be captured without waiting for
LeRobot video encoding. The recorder stores each topic's original timestamps
and payloads without temporal alignment, while embedding the configured
LeRobot `features` and `topic_to_feature` mappings in every file.

In an existing LeRobot recorder YAML, change the storage settings while
keeping the same feature schema and topic mapping:

```yaml
storage_format: raw_hdf5
output_dir: output/raw/my_session
fps: 30
master_clock_topic: observation/realsense_front/video_frame

# Keep the same topic_to_feature and features sections used by LeRobot.
```

This creates atomic, non-overwriting files such as:

```text
output/raw/my_session/episode_000000.raw.h5
output/raw/my_session/episode_000001.raw.h5
```

After data collection finishes, convert the complete session offline:

```powershell
dexim recorder convert-raw `
  --input-dir output/raw/my_session `
  --output-dir output/lerobot/my_session `
  --repo-id local/my-session
```

The source Raw HDF5 files are retained after conversion.
