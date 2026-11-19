# Three-camera LeRobot encoding check

Measured on Windows with 12 logical CPUs using one second of synthetic
1280x720 JPEG input at 30 fps per camera.

| Mode | Cameras | Elapsed |
| --- | ---: | ---: |
| LeRobot 0.4.4 default parallel SVT-AV1 | 1 | 1.069 s |
| LeRobot 0.4.4 default parallel SVT-AV1 | 3 | 13.944 s |
| Serial SVT-AV1 | 3 | 2.935 s |

The three-camera parallel run was 4.75 times slower than serial on this host.
Each parallel SVT-AV1 encoder independently selected five-way parallelism,
oversubscribing the available CPU. Wall-clock values are diagnostic rather
than CI thresholds because encoder performance depends on the host.

The Release Candidate Line therefore defaults to serial camera encoding and
four encoder threads. Both settings and the codec remain explicit recorder
configuration so other hardware can opt into measured parallel settings.
