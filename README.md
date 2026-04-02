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
