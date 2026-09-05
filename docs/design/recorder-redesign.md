# `dexim-recorder` — Redesign Report

> Date: 2026-04-02
> Status: Draft
> Package: `dexim-recorder` (`dexim.recorder`)
> Repository: `ts-dexim-recorder`

---

## 1. Purpose

`dexim-recorder` is a multi-stream data collection node for **imitation learning**
and **foundation model training**. It subscribes to ZMQ data endpoints, temporally
aligns heterogeneous sensor streams, and persists episodes in a format suitable
for downstream policy learning.

Key design goals:

- **LeRobot-first**: LeRobot v3 (Parquet + video) is the default storage format.
  HDF5 is retained as a fallback for environments where `lerobot` cannot be
  installed or for quick prototyping.
- **Runtime task metadata**: Task identity (name, description, language instruction)
  is supplied at recording time via control-plane commands, not baked into static
  config. This supports multi-task data collection sessions where the operator
  switches tasks without restarting the node.
- **Episode discard**: The operator can discard a bad episode instead of saving it.
  There is no pause-and-resume within an episode — an episode is either committed
  or discarded atomically.

---

## 2. Terminology

| Term | Definition |
|------|-----------|
| **Episode** | One continuous recording segment: START_REC → STOP_REC. Contains N temporally-aligned frames across all subscribed streams. |
| **Task** | A semantic label for what the robot is doing (e.g. `"pick_red_cup"`). A task has an `id` and a human-readable `description` (natural language instruction). Multiple episodes can share the same task. |
| **Frame** | One aligned sample across all streams at a single master-clock tick. |
| **Dataset** | The persistent collection of episodes on disk (LeRobot dataset directory or HDF5 output directory). |
| **Stream** | A single ZMQ topic carrying one sensor modality (e.g. `observation/cam_left/video_frame`). |

---

## 3. Architecture

### 3.1 Node Position in the System

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Nova Node   │     │ Inspire Node │     │ RealSense    │
│  (arm ctrl)  │     │ (hand ctrl)  │     │ (camera)     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │ PUB                │ PUB                │ PUB
       │  action/nova_left/ │  action/inspire_*/ │  observation/cam_*/
       │  joint_cmd         │  joint_cmd         │  video_frame
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────┐
│                   DataRecorderNode                        │
│  SUB ← endpoint_1, endpoint_2, …, endpoint_N             │
│                                                           │
│  ┌─────────┐   ┌───────────┐   ┌──────────────────────┐  │
│  │ Buffers │──►│ Alignment │──►│ StorageBackend (ABC)  │  │
│  │ (topic  │   │ (master   │   │  ├─ LeRobotWriter ◄─ │  │  default
│  │  → data)│   │  clock)   │   │  └─ HDF5Writer       │  │  fallback
│  └─────────┘   └───────────┘   └──────────────────────┘  │
│                                                           │
│  Heartbeat ──► Status Plane (PUSH)                        │
│  Control   ◄── Control Plane (SUB)                        │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Orchestrator   │
                │  (sends START/  │
                │   STOP/DISCARD  │
                │   + task info)  │
                └─────────────────┘
```

### 3.2 Class Hierarchy

```
ManagedNode (dexim.core.nodes.managed)
  └── DataRecorderNode (dexim.recorder.node)
        ├── owns: StorageBackend (ABC)
        │     ├── LeRobotWriter      ← default
        │     └── HDF5Writer         ← fallback
        ├── owns: EpisodeWriterQueue  (single daemon thread)
        └── owns: EpisodeMetadata     (current episode state)
```

The intermediate `RecorderNode` base class in `dexim.core.nodes.recorder_node`
is **deprecated**. `DataRecorderNode` extends `ManagedNode` directly and owns
all subscription, buffering, alignment, and episode lifecycle responsibilities.

---

## 4. Control Commands

### 4.1 Existing Commands (from `dexim.core.messages`)

| Command | Constant | Effect on Recorder |
|---------|----------|--------------------|
| `START` | `CTRL_START` | Node enters active mode, clears stale buffers |
| `PAUSE` | `CTRL_PAUSE` | Node-level pause (hold position) — recorder drains sockets but does not buffer |
| `STOP` | `CTRL_STOP` | Node-level stop |
| `SHUTDOWN` | `CTRL_SHUTDOWN` | Drain writer queue, close backend, exit |
| `START_REC` | `CTRL_START_REC` | Begin new episode — clear buffers, start buffering |
| `STOP_REC` | `CTRL_STOP_REC` | End episode — deep-copy buffers, submit to writer queue |

### 4.2 New Commands (to add)

| Command | Constant | Purpose |
|---------|----------|---------|
| `DISCARD_REC` | `CTRL_DISCARD_REC` | Discard current episode — clear buffers without writing. The episode counter does **not** increment. |
| `SET_TASK` | `CTRL_SET_TASK` | Set the active task for subsequent episodes. Payload: JSON with `task_id` and `task_description`. |

### 4.3 Command Payload Extension

Currently all control commands are simple string constants sent as
`[TOPIC_CTRL, command_bytes]`. The `SET_TASK` command needs a payload.
We extend the multipart format to support an optional third frame:

```
Control message format:
  Frame 0: TOPIC_CTRL (b"control")
  Frame 1: command string (e.g. b"SET_TASK")
  Frame 2: (optional) JSON payload bytes

Backward compatible: existing commands send 2 frames, new commands can
send 3. The handler ignores frame 2 when absent.
```

### 4.4 Episode State Machine

```
                     SET_TASK
                    (any time)
                        │
           ┌────────────▼────────────┐
           │         IDLE            │
           │  (not recording)        │
           └────────────┬────────────┘
                        │ START_REC
                        ▼
           ┌─────────────────────────┐
           │      RECORDING          │
           │  (buffering data)       │
           └───┬────────────────┬────┘
               │                │
          STOP_REC         DISCARD_REC
               │                │
               ▼                ▼
     ┌─────────────┐   ┌──────────────┐
     │  WRITING    │   │  DISCARDED   │
     │  (async)    │   │  (buffers    │
     │             │   │   cleared)   │
     └──────┬──────┘   └──────┬───────┘
            │                 │
            ▼                 ▼
           ┌─────────────────────────┐
           │         IDLE            │
           └─────────────────────────┘
```

**Rules:**
- `START_REC` is only valid in IDLE. If received while already recording, it is
  logged as a warning and ignored.
- `STOP_REC` commits the episode: deep-copies buffers, increments counter, submits
  to writer queue.
- `DISCARD_REC` drops the episode: clears buffers, does **not** increment counter,
  logs discarded episode info for traceability.
- `SET_TASK` is valid at any time. It updates the active task metadata for all
  subsequent episodes. If sent during RECORDING, it takes effect on the *next*
  episode (the current one retains the task that was active at START_REC).

---

## 5. Metadata Model

### 5.1 Task Info (Runtime, Not Config)

Task information is **not part of `RecorderNodeConfig`**. Instead, it is supplied
at runtime via the `SET_TASK` control command. This design reflects the reality
that:

- An operator may collect data for multiple tasks in a single session.
- Task descriptions may be refined on the fly.
- The node should not need to be restarted to switch tasks.

```python
@dataclass
class TaskInfo:
    """Active task metadata, set via SET_TASK command."""
    task_id: str = ""                  # e.g. "pick_red_cup"
    task_description: str = ""         # e.g. "Pick up the red cup and place it on the tray"
```

The recorder holds an `_active_task: TaskInfo` that is updated by `SET_TASK` and
snapshot-copied into each episode at `START_REC` time.

**Default behavior:** If no `SET_TASK` has been received, `task_id` and
`task_description` are empty strings. A config-level `default_task` field
can optionally provide a fallback that is used when the runtime task is unset.

### 5.2 Episode Metadata

Each committed episode carries metadata that is persisted alongside the data:

```python
@dataclass
class EpisodeMetadata:
    """Metadata for a single recorded episode."""
    episode_index: int                 # 0-based index within the dataset
    task_id: str                       # task active at START_REC
    task_description: str              # natural language instruction
    start_time: float                  # wall-clock epoch at START_REC
    end_time: float                    # wall-clock epoch at STOP_REC
    num_frames: int                    # after alignment
    num_topics: int                    # number of subscribed streams
    topics: list[str]                  # topic names present in this episode
```

### 5.3 Storage of Metadata

| Field | HDF5 | LeRobot |
|-------|------|---------|
| `task_id` | HDF5 file attribute `task` | `save_episode(task=task_id)` — stored in episode info |
| `task_description` | HDF5 file attribute `task_description` | `info.json` episode metadata or column in Parquet |
| `start_time` / `end_time` | HDF5 file attributes | `info.json` episode metadata |
| `topics` | HDF5 group names (implicit) | Implicit via feature keys |

---

## 6. Storage Backends

### 6.1 LeRobot Writer (Default)

LeRobot v3 is the default backend because:

- It is the de facto standard for imitation learning datasets in the community.
- Native support for video compression (MP4), multi-episode datasets, and
  HuggingFace Hub integration.
- Built-in episode/task metadata model aligns with our needs.
- Direct consumption by policy training frameworks (ACT, Diffusion Policy, etc.).

**Initialization:**

```python
LeRobotWriter(
    dataset_path="output/lerobot",
    repo_id="org/dataset-name",
    features={...},                # feature schema (shapes, dtypes)
    topic_to_feature={...},        # topic string → LeRobot feature name
    fps=30,
    push_to_hub=False,
    robot_type="dexim-bimanual",
)
```

**Feature mapping** uses the explicit `topic_to_feature` config dict. Example:

```yaml
topic_to_feature:
  observation/cam_left/video_frame: observation.images.cam_left
  observation/cam_right/video_frame: observation.images.cam_right
  observation/nova_left/joint_state: observation.state.nova_left_joint
  observation/nova_right/joint_state: observation.state.nova_right_joint
  observation/inspire_left/joint_state: observation.state.inspire_left_joint
  observation/inspire_right/joint_state: observation.state.inspire_right_joint
  action/nova_left/joint_cmd: action.nova_left_joint
  action/nova_right/joint_cmd: action.nova_right_joint
  action/inspire_left/joint_cmd: action.inspire_left_joint
  action/inspire_right/joint_cmd: action.inspire_right_joint
```

Joint-state and joint-command payloads use their `q` vector. Numeric values are
converted to numpy arrays using the dtype declared by the mapped feature's
schema; image payloads are decoded from their `color` bytes. If `dataset_path`
exists but does not contain a complete LeRobot dataset, initialization fails
without deleting the directory so an operator can move or recover it
explicitly.

### 6.2 HDF5 Writer (Fallback)

Retained for:

- Environments without the `lerobot` dependency.
- Quick prototyping / debugging where single-file-per-episode is convenient.
- Offline conversion workflows (record HDF5 → batch-convert to LeRobot later).

One `episode_NNNN.h5` per episode, gzip-compressed, pattern-based dtype mapping.

### 6.3 Config Changes

```python
@dataclass
class RecorderNodeConfig:
    # Identity
    node_id: str = "recorder"

    # Data plane
    data_endpoints: list[str] = field(default_factory=list)

    # Storage
    storage_format: str = "lerobot"        # ← changed default from "hdf5"
    output_dir: str = "output/episodes"    # HDF5 output directory

    # Task (fallback only — runtime SET_TASK is preferred)
    default_task: str = ""

    # Alignment
    master_clock_topic: str | None = None
    continuous_topics: list[str] = field(default_factory=list)

    # HDF5-specific
    topic_dtypes: dict[str, str] = field(default_factory=dict)

    # LeRobot-specific
    topic_to_feature: dict[str, str] = field(default_factory=dict)
    features: dict[str, Any] | None = None
    lerobot_dataset_path: str = "output/lerobot"
    lerobot_repo_id: str = ""
    push_to_hub: bool = False
    fps: int = 30

    # Control plane
    control_endpoint: str = CTRL_PUB_ENDPOINT
    status_endpoint: str = STATUS_PULL_ENDPOINT
```

Notable changes from current:
- `storage_format` default changed from `"hdf5"` to `"lerobot"`.
- `task` renamed to `default_task` to clarify it is a fallback, not the primary
  source of task metadata.
- Runtime `TaskInfo` is added separately (not a config field).

---

## 7. Data Flow Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  1. SUBSCRIPTION                                              │
│  DataRecorderNode subscribes to N ZMQ PUB endpoints.          │
│  Multipart: [topic_bytes, msgpack_payload]                    │
│  Decoded via unpack_data_message() → (topic, timestamp, data) │
└──────────────┬───────────────────────────────────────────────┘
               │
               │  is_recording == True?
               │  ├─ Yes → buffer
               │  └─ No  → drain and discard
               ▼
┌──────────────────────────────────────────────────────────────┐
│  2. BUFFERING (main thread, guarded by _buffer_lock)          │
│  _buffers[topic: str] = [(timestamp: float, data: Any), …]   │
│  Accumulates between START_REC → STOP_REC / DISCARD_REC      │
└──────────────┬───────────────────────────────────────────────┘
               │
               │  STOP_REC received?
               │  ├─ STOP_REC    → deep-copy, submit to writer
               │  └─ DISCARD_REC → clear buffers, log, done
               ▼
┌──────────────────────────────────────────────────────────────┐
│  3. ALIGNMENT (EpisodeWriterQueue, single daemon thread)      │
│  Master clock: explicit config > */video_frame > most samples │
│  Continuous streams (joint_state): linear interpolation       │
│  Discrete streams (images, actions): nearest-neighbour        │
│  → list[dict[str, Any]]  (one dict per master-clock frame)   │
└──────────────┬───────────────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────────────┐
│  4. STORAGE                                                   │
│                                                               │
│  LeRobotWriter (default):                                     │
│    topic → feature mapping → dataset.add_frame()              │
│    → save_episode(task=task_id)                               │
│    → Parquet + MP4 in persistent LeRobotDataset               │
│                                                               │
│  HDF5Writer (fallback):                                       │
│    row → column, dtype conversion, gzip(4)                    │
│    → episode_NNNN.h5                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Topic Convention

All new code uses the **hierarchical slash-delimited** format:

```
{category}/{device_id}/{data_type}
```

| Category | Examples |
|----------|----------|
| `observation` | `observation/cam_left/video_frame`, `observation/nova_left/joint_state` |
| `action` | `action/nova_left/joint_cmd`, `action/inspire_right/joint_cmd` |

This maps naturally to `TopicBuilder` in `dexim.core.messages`:

```python
topics = TopicBuilder()
topics.observation.joint_state("nova_left")   # b"observation/nova_left/joint_state"
topics.action.joint_cmd("inspire_right")      # b"action/inspire_right/joint_cmd"
```

Legacy flat topics (`obs_image_cam1`, `action_joint_cmd_robot1`) are still
recognized by the alignment module's pattern matching. No migration shim is
needed in the recorder itself — the topic strings are opaque keys. The dtype
patterns in `HDF5Writer` already support both formats.

---

## 9. Episode Writer Queue

The `EpisodeWriterQueue` serializes episode writes on a single daemon thread,
preventing unbounded thread accumulation:

```
Main Thread                          Writer Thread (daemon)
    │                                       │
    ├─ on_stop_recording()                  │
    │   ├─ deep-copy buffers                │
    │   ├─ snapshot EpisodeMetadata          │
    │   └─ queue.put((buffers, meta)) ─────►│
    │                                       ├─ align_episode_data()
    │   (continues polling)                 ├─ backend.write_episode()
    │                                       └─ log result
    │                                       │
    ├─ on_discard_recording()               │
    │   ├─ clear buffers                    │  (nothing queued)
    │   └─ log discard                      │
    │                                       │
    ├─ on_shutdown()                        │
    │   └─ queue.put(SHUTDOWN) ────────────►│
    │       thread.join(timeout)            └─ exit
```

### 9.1 Writer Queue Payload Change

Currently the queue accepts `(buffers, episode_number, task)`. This changes to
`(buffers, EpisodeMetadata)` to carry the full metadata snapshot:

```python
def submit(
    self,
    buffers: dict[str, list[tuple[float, Any]]],
    metadata: EpisodeMetadata,
) -> None:
```

---

## 10. Heartbeat & Status Reporting

Heartbeat is inherited from `ManagedNode.send_heartbeat_if_needed()` and reports
to the status plane at `heartbeat_interval` (default 1s).

The recorder enriches the heartbeat `info` dict with recording-specific state:

```python
{
    "is_recording": True,
    "episode_counter": 5,
    "active_task": "pick_red_cup",
    "buffer_topics": 8,
    "buffer_frames": 1234,        # total samples across all topics
    "writer_queue_depth": 0,      # episodes queued but not yet written
}
```

This gives the orchestrator and monitoring dashboard real-time visibility into
the recorder's state without polling.

---

## 11. CLI

The recorder exposes a CLI via `dexim-recorder` (standalone) or as a subcommand
of the umbrella `dexim` CLI:

```
dexim recorder run --config lab-bimanual
dexim recorder status --output-dir output/lerobot
dexim recorder config list
dexim recorder config show lab-bimanual
dexim recorder config validate lab-bimanual
```

No CLI changes are needed for runtime task metadata — `SET_TASK` is sent from
the orchestrator or a separate operator tool, not from the recorder's own CLI.

---

## 12. Impact on `dexim-core`

### 12.1 New Message Constants

Add to `dexim.core.messages`:

```python
CTRL_DISCARD_REC = "DISCARD_REC"
CTRL_SET_TASK = "SET_TASK"
```

### 12.2 ManagedNode Changes

Add handling for the new commands in `ManagedNode._handle_control_message()`:

```python
elif cmd == CTRL_DISCARD_REC:
    self.is_recording = False
    self.on_discard_recording()
    self.report_status(STATUS_HEALTHY)

elif cmd == CTRL_SET_TASK:
    payload = parts[2] if len(parts) > 2 else b"{}"
    task_info = json.loads(payload)
    self.on_set_task(task_info)
    self.report_status(STATUS_HEALTHY)
```

Add abstract hooks (with default no-op implementations so existing nodes don't
break):

```python
def on_discard_recording(self) -> None:
    """Called when DISCARD_REC command is received. Default: no-op."""

def on_set_task(self, task_info: dict[str, Any]) -> None:
    """Called when SET_TASK command is received. Default: no-op."""
```

### 12.3 Orchestrator Changes

The orchestrator (`dexim.core.nodes.orchestrator`) needs to expose methods /
CLI commands to send `DISCARD_REC` and `SET_TASK` on the control plane.

---

## 13. Migration Path

### Phase 1: Core changes (non-breaking)

1. Add `CTRL_DISCARD_REC` and `CTRL_SET_TASK` to `dexim.core.messages`.
2. Add `on_discard_recording()` and `on_set_task()` as default no-op hooks on
   `ManagedNode`.
3. Handle the new commands in `ManagedNode._handle_control_message()`.
4. Deprecate `RecorderNode` base class in `dexim.core.nodes.recorder_node` (it
   is already unused by the current `DataRecorderNode`).

### Phase 2: Recorder changes

5. Add `TaskInfo` and `EpisodeMetadata` dataclasses to `dexim.recorder`.
6. Implement `on_discard_recording()` and `on_set_task()` on `DataRecorderNode`.
7. Change `RecorderNodeConfig.storage_format` default to `"lerobot"`.
8. Rename `task` to `default_task` on `RecorderNodeConfig`.
9. Update `EpisodeWriterQueue.submit()` to accept `EpisodeMetadata`.
10. Update both backends to persist the enriched metadata.

### Phase 3: Orchestrator integration

11. Add `DISCARD_REC` and `SET_TASK` to the orchestrator's command vocabulary.
12. Wire up operator UI / CLI to send `SET_TASK` before recording sessions.

---

## 14. Open Questions

1. **Episode success/failure labelling** — Should we add a `LABEL_EPISODE`
   command for post-hoc labelling, or rely on external tooling to annotate
   datasets after collection?
2. **Auto-discard policy** — Should the recorder auto-discard episodes shorter
   than a configurable minimum duration or frame count? Or leave all filtering
   to post-processing?
3. **Multi-task within one episode** — The current design snapshots the task at
   `START_REC`. Should we support task changes mid-episode (e.g. for long-horizon
   compound tasks)? Likely not needed for v1.
4. **Schema validation** — Should the feature schema be mandatory for both
   backends (currently only required for LeRobot)? This would catch misconfigured
   topics early.
