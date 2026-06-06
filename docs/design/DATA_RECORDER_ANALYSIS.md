# `data-recorder-node` — Architecture & Implementation Analysis

> Generated: 2026-04-01

---

## Package Overview

A **multi-stream timestamped data collection** library for robotics imitation learning. It subscribes to ZMQ data streams, temporally aligns them, and persists episodes in HDF5 or LeRobot v3 format. Designed as a pluggable, backend-agnostic framework.

**Source layout:**

```
src/data_recorder_node/
├── __init__.py                  # Exports DataRecorderNode, EpisodeWriterThread
├── recorder.py                  # Main DataRecorderNode class
├── episode_writer.py            # Background save thread
├── alignment.py                 # Temporal alignment (interpolation + nearest-neighbor)
├── lerobot_backend.py           # LeRobot-specific format logic (optional dep)
├── topic_feature_mapper.py      # Topic ↔ LeRobot feature name mapping
├── topic_validator.py           # Topic format parsing/validation
├── config_validator.py          # Configuration validation
├── validation.py                # Runtime feature data validation
├── discover_schema.py           # Auto schema discovery from live ZMQ streams
└── backends/
    ├── base.py                  # StorageBackend ABC
    ├── hdf5_writer.py           # HDF5 backend
    └── lerobot_writer.py        # LeRobot v3 backend
```

**Dependencies:**

| Package                | Version | Role                                         |
| ---------------------- | ------- | -------------------------------------------- |
| `core-node-framework`  | ≥0.1.0  | Base `RecorderNode` class                    |
| `shared-messages`      | ≥0.1.0  | ZMQ message contracts                        |
| `pyzmq`                | ≥25.0.0 | ZMQ transport                                |
| `numpy`                | ≥1.20.0 | Alignment interpolation, dtype conversion    |
| `h5py`                 | ≥3.0.0  | HDF5 file I/O                                |
| `loguru`               | ≥0.7.0  | Structured logging                           |
| `lerobot` _(optional)_ | ≥0.1.0  | LeRobotDataset, Parquet/MP4, HuggingFace Hub |

> **Migration note:** `core-node-framework` and `shared-messages` are legacy package names. Per the DexImitate architecture, these should migrate to `dexim.core.node_framework` and `dexim.core.messages` respectively when this package is consolidated into `dexim-recorder`.

---

## 1. Class Hierarchy

```
RecorderNode (core_node_framework)     ← ZMQ subscription, buffering, episode lifecycle
  └── DataRecorderNode                 ← Backend selection, writer spawning, finalization
        │
        ├── owns: StorageBackend (ABC)
        │     ├── HDF5Writer           ← Individual .h5 files per episode
        │     └── LeRobotWriter        ← Persistent LeRobotDataset (Parquet + MP4)
        │           └── delegates to: lerobot_backend (module-level functions)
        │           └── uses: TopicFeatureMapper
        │
        └── spawns: EpisodeWriterThread (daemon Thread)
              └── calls: align_episode_data() → backend.write_episode()
```

**Supporting classes / modules:**

| Class / Module                         | Role                                                              |
| -------------------------------------- | ----------------------------------------------------------------- |
| `StorageBackend` (ABC)                 | Contract for all storage backends                                 |
| `TopicFeatureMapper`                   | Config-driven topic → LeRobot feature name lookup (bidirectional) |
| `SchemaDiscovery`                      | Live ZMQ sampling → auto-generated YAML config                    |
| `ConfigValidator` / `ValidationResult` | Static config validation before launch                            |
| `validate_features()`                  | Runtime shape/dtype validation against feature schema             |
| `validate_topic_format()`              | Parse/validate `{category}_{type}_{device_id}` format             |

---

## 2. Data Flow Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  1. SUBSCRIPTION                                              │
│  RecorderNode subscribes to N ZMQ PUB endpoints               │
│  socket.setsockopt(zmq.SUBSCRIBE, b"")                        │
│  Multipart: [topic (bytes), data (pyobj/pickle)]              │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  2. BUFFERING (in-memory, main thread)                        │
│  episode_buffers[topic: bytes] = [(timestamp: float, data)]   │
│  Accumulates between start_episode() → stop_episode()         │
└──────────────┬───────────────────────────────────────────────┘
               ▼  deep copy on stop_episode()
┌──────────────────────────────────────────────────────────────┐
│  3. ALIGNMENT (EpisodeWriterThread, background daemon)        │
│  a) Select master clock:                                      │
│       explicit config > obs_image_cam1 > most-samples stream  │
│  b) Classify streams:                                         │
│       continuous (interpolate): "joint_state"/"state" in name │
│       discrete (nearest-neighbor): images, depth, actions     │
│  c) For each master timestamp, align all other streams        │
│     → list[dict[str, Any]]  (one dict per frame)              │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  4. STORAGE (backend.write_episode)                           │
│                                                               │
│  HDF5Writer:                                                  │
│    row → column, dtype conversion (pattern-based), gzip(4)    │
│    → episode_NNNN.h5  (one file per episode)                  │
│                                                               │
│  LeRobotWriter:                                               │
│    topic → feature mapping (TopicFeatureMapper or legacy)     │
│    dataset.add_frame() + save_episode()                       │
│    → Parquet + MP4 in persistent LeRobotDataset               │
│    → optional push_to_hub() on close()                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Episode Lifecycle

```
recorder.run()
    │
    ├─ recorder.start_episode()
    │     ├─ Init empty episode_buffers
    │     └─ _is_recording = True
    │
    │  [data streams in via ZMQ, buffered per topic]
    │
    └─ recorder.stop_episode()
          ├─ _is_recording = False
          ├─ Deep copy buffers
          ├─ Increment episode_number
          ├─ Spawn EpisodeWriterThread (daemon=True)  ──► runs in background
          └─ Clear buffer, ready for next episode

EpisodeWriterThread.run():
    ├─ align_episode_data(buffers, master_clock, continuous_topics)
    ├─ backend.write_episode(aligned_frames, episode_number, task)
    └─ (thread exits; errors are logged, not raised)
```

---

## 4. Temporal Alignment (`alignment.py`)

### Master clock selection

Priority order:

1. Explicitly configured `master_clock_topic`
2. `obs_image_cam1` if present in buffers
3. Stream with the most data points

### Stream classification

| Stream type | Detection                                                       | Alignment method                                  |
| ----------- | --------------------------------------------------------------- | ------------------------------------------------- |
| Continuous  | `"joint_state"` or `"state"` in topic name (or explicit config) | Linear interpolation                              |
| Discrete    | Everything else                                                 | Nearest-neighbor (`np.argmin(np.abs(times - t))`) |

### Interpolation details

```python
# Linear interpolation for numpy arrays and scalars:
alpha = (target_time - t_before) / (t_after - t_before)
result = (1 - alpha) * data_before + alpha * data_after

# Boundary handling:
# target_time ≤ first sample → return first sample
# target_time ≥ last sample  → return last sample
# non-numeric fallback       → return nearest by alpha threshold
```

**Complexity:** O(N × M) where N = master clock frames, M = number of streams.

---

## 5. Storage Backends

### HDF5Writer

- One `episode_NNNN.h5` per episode, flat key-value structure
- Row→column transposition before write
- Pattern-based dtype conversion (overridable via `topic_dtypes`):

| Topic pattern        | Target dtype | Notes                                     |
| -------------------- | ------------ | ----------------------------------------- |
| `obs_image_*`        | uint8        | Float images in [0,1] are normalized ×255 |
| `obs_depth_*`        | uint16       |                                           |
| `obs_joint_state_*`  | float32      |                                           |
| `action_joint_cmd_*` | float32      |                                           |

- `timestamp` saved as float64 dataset; `task` as HDF5 file attribute
- gzip compression level 4
- `close()` is a no-op (each episode is a standalone file)

### LeRobotWriter

- Persistent `LeRobotDataset` (Parquet + video), accumulates all episodes
- New datasets require a `features` schema dict (raises `ValueError` otherwise)
- Feature shapes converted from `list` → `tuple` for LeRobot API
- Created with: fps=30, `use_videos=True`, `robot_type="ts-dualarm"`, `tolerance_s=1e-4`
- `close()` calls `dataset.finalize()` then optionally `dataset.push_to_hub()`

### Topic → Feature Mapping (LeRobot only)

Two paths, selected by whether `TopicFeatureMapper` is configured:

**New path (preferred):** `TopicFeatureMapper` — config-driven direct dict lookup  
**Legacy path (fallback):** Four per-type functions with hardcoded conventions:

| Legacy function            | Input keys                 | Output feature keys                        |
| -------------------------- | -------------------------- | ------------------------------------------ |
| `map_image_features`       | `obs_image_{cam}`          | `observation.images.{cam}_rgb`             |
| `map_depth_features`       | `obs_depth_{cam}`          | `observation.state.{cam}_depth`            |
| `map_joint_state_features` | `obs_joint_state_{robot}`  | `observation.state` (concatenated, sorted) |
| `map_action_features`      | `action_joint_cmd_{robot}` | `action.{robot}`                           |

---

## 6. Topic Convention

Parsed by `validate_topic_format(topic: str) -> (category, type_str, device_id)`:

```
{category}_{type}_{device_id}
```

- `category` = `parts[0]` — typically `obs` or `action`
- `type_str` = `"_".join(parts[1:-1])` — e.g., `image`, `joint_state`, `joint_cmd`
- `device_id` = `parts[-1]` — last segment only

**Examples:**

| Topic                       | category | type             | device_id |
| --------------------------- | -------- | ---------------- | --------- |
| `obs_image_cam1`            | obs      | image            | cam1      |
| `obs_joint_state_nova_left` | obs      | joint_state_nova | left      |
| `action_joint_cmd_robot1`   | action   | joint_cmd        | robot1    |

---

## 7. Configuration & Validation

### Constructor arguments (no config file parsing in library)

```python
DataRecorderNode(
    node_id="data_recorder",
    data_endpoints=["tcp://localhost:5556", "tcp://localhost:5557"],
    storage_format="lerobot",          # or "hdf5"
    output_dir="output/episodes",
    lerobot_dataset_path="/data/datasets",
    lerobot_repo_id="ts-lerobot/dualarm",
    push_to_hub=False,
    features={...},                    # required for new LeRobot datasets
    topic_to_feature={...},            # TopicFeatureMapper config
    devices={...},                     # device_id → metadata
    master_clock_topic="obs_image_cam1",
    continuous_topics=["obs_joint_state_nova_left", ...],
    topic_dtypes={...},                # HDF5 dtype overrides
    task="pick_and_place",
)
```

### Validation layers

| Layer             | Module                    | Timing                      | Scope                                                              |
| ----------------- | ------------------------- | --------------------------- | ------------------------------------------------------------------ |
| Config validation | `ConfigValidator`         | Pre-launch                  | Topic format, feature references, device metadata, unmapped topics |
| Topic format      | `validate_topic_format()` | On access                   | `{category}_{type}_{device_id}` format (≥3 segments)               |
| Runtime feature   | `validate_features()`     | First frame of each episode | Shape/dtype vs feature schema                                      |

### Schema discovery

```bash
python discover_schema.py \
    --endpoints tcp://localhost:5556 tcp://localhost:5557 \
    --duration 10 \
    --output discovered_config.yaml
```

`SchemaDiscovery` connects to live ZMQ endpoints, samples for `duration` seconds, infers shapes/dtypes/continuity, picks master clock, generates device metadata, and writes a complete YAML config.

---

## 8. Threading Model

```
Main Thread                        EpisodeWriterThread (daemon)
    │                                       │
    ├─ ZMQ NOBLOCK recv loop                │
    ├─ Buffer per episode                   │
    │                                       │
    ├─ stop_episode()                       │
    │   ├─ deep copy buffers                │
    │   └─ Thread(target=run).start() ─────►│
    │                                       ├─ align_episode_data()
    ├─ (buffering next episode)             ├─ backend.write_episode()
    │                                       └─ exit (or log error)
    │
    └─ on_finalize() → backend.close()
```

**Properties:**

- Daemon threads — won't block process exit
- No inter-thread synchronization needed for HDF5 (separate files per episode)
- Deep copy of buffers at `stop_episode()` time prevents race conditions between the recording and writing threads

---

## 9. Error Handling Strategy

| Context                                   | Strategy                                                        |
| ----------------------------------------- | --------------------------------------------------------------- |
| Constructor (bad format, missing path)    | Raise `ValueError` immediately                                  |
| Missing optional dep (lerobot)            | Raise `ImportError` with install hint                           |
| Episode write failure (background thread) | Log `logger.error()`, thread exits silently — episode data lost |
| Backend finalization failure              | Log `logger.error()`, continue shutdown                         |
| Dtype mismatch / missing frame key        | Log `logger.warning()`, skip/fallback                           |

---

## 10. Known Issues & Observations

### 1. Topic parsing ambiguity with multi-word device IDs

`validate_topic_format("obs_joint_state_nova_left")` returns `("obs", "joint_state_nova", "left")` because `device_id = parts[-1]`. Device IDs like `nova_left` are unrepresentable with the current parser — `nova` becomes part of `type_str`. The schema discovery, legacy mapping functions, and `TopicFeatureMapper` each handle this differently, leading to inconsistent behavior across the codebase.

### 2. No bound on concurrent writer threads

Each `stop_episode()` call spawns a new daemon thread. Rapid repeated episodes (e.g., short demonstrations) can accumulate many live threads simultaneously with no backpressure mechanism. A bounded thread pool or single serialized write queue would prevent this.

### 3. LeRobot concurrent write safety

`LeRobotWriter` holds a single shared `LeRobotDataset` instance. If two `EpisodeWriterThread`s run concurrently (which is possible), `dataset.add_frame()` and `dataset.save_episode()` interleave without any locking. This is a data corruption risk for the LeRobot backend specifically.

### 4. Dual mapping path divergence

`lerobot_backend.py` has two codepaths that produce different output for the same input:

- **New path** (`map_frame_to_features` via `TopicFeatureMapper`): flat per-topic mapping
- **Legacy path** (`map_joint_state_features`): concatenates all `obs_joint_state_*` arrays alphabetically by robot name into a single `observation.state` vector

Users switching from legacy to new config may silently produce differently-shaped datasets.

### 5. `observation.state` concatenation order is implicit

In `map_joint_state_features()`, robot names are gathered from `observation.state.names` metadata and sorted alphabetically (`sorted(robot_names)`) before concatenation. If robot names change or are added, the vector layout shifts silently with no error.

### 6. Lost episodes on write failure

`EpisodeWriterThread` catches all exceptions and discards the episode. There is no retry, dead-letter queue, or user notification beyond a log line. A failed write is indistinguishable from success from the recording operator's perspective.

### 7. Pickle-based ZMQ deserialization in schema discovery

`discover_schema.py` calls `socket.recv_pyobj()` (pickle) on all incoming messages. This is standard in this codebase but is a known deserialization risk if any untrusted publisher exists on the network.

### 8. Legacy package names in `pyproject.toml`

`shared-messages` and `core-node-framework` should become `dexim-core` when migrated to the consolidated architecture (`dexim.core.messages`, `dexim.core.node_framework`).

---

## 11. Summary

| Aspect           | Implementation                                                              |
| ---------------- | --------------------------------------------------------------------------- |
| Entry point      | `DataRecorderNode` (subclass of `RecorderNode`)                             |
| Storage backends | 2 built-in: HDF5, LeRobot v3; extensible via `StorageBackend` ABC           |
| Alignment        | Time-based; linear interpolation (continuous) + nearest-neighbor (discrete) |
| Threading        | Daemon threads for non-blocking episode writes; deep copy for safety        |
| Configuration    | Constructor args; optional live schema discovery CLI                        |
| ZMQ pattern      | PUB-SUB to multiple endpoints; subscribe-all                                |
| Topic format     | `{category}_{type}_{device_id}` (validated)                                 |
| Dtype conversion | Pattern-based (HDF5) or feature-schema-based (LeRobot)                      |
| Validation       | Static config + runtime shape/dtype per episode                             |
| Logging          | `loguru` with success/info/debug/warning/error levels                       |
| Error strategy   | Fail fast in constructor; log + continue in background threads              |
