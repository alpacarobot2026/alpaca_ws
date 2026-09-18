# Container for the alpaca stack

ROS 2 Humble + Nav2 + the five packages in this repo, prebuilt and sourced by
the entrypoint. Setup, model weights and launch commands are in the
[main README](../README.md).

## Build

From the repository root:

```bash
docker build -f docker/humble.Dockerfile -t ros:alpaca-humble .
```

Expect a long first build and a large image (roughly 15-20 GB).

## Run

```bash
docker compose -f docker/docker-compose.yml run --rm alpaca
```

Compose mounts `$ALPACA_MODELS_DIR` (default `~/alpaca_ws/models`) at
`/models` and `$ALPACA_MAPS_DIR` (default `~/alpaca_ws/data/maps`) at `/maps`,
and forwards X11 and PulseAudio. The plain `docker run` equivalent:

```bash
docker run --rm -it --gpus all --network host --ipc host --privileged \
    -v ~/alpaca_ws/models:/models:ro -v ~/alpaca_ws/data/maps:/maps:ro \
    -e DISPLAY=$DISPLAY -e XAUTHORITY=/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v <xauth file>:/root/.Xauthority:ro \
    ros:alpaca-humble
```

`--gpus all` needs nvidia-container-toolkit. For GUI tools (rviz2, rqt,
`--face-display`), check the host display with `who` and export `DISPLAY` to
match. The X auth cookie is usually *not* `~/.Xauthority`: on a GDM host it is
`$XDG_RUNTIME_DIR/gdm/Xauthority` (compose's default). Find the real path in
the `-auth` argument of `ps aux | grep Xorg` and set `ALPACA_XAUTHORITY` if it
differs.

## Adding packages

- **Python:** add runtime deps to the matching requirements file -
  `requirements-ros.txt` (system python3: ROS nodes, MPPI, SimpleTrack,
  TalkNCE), `requirements-hst.txt` (`/opt/venvs/hst`: prediction) or
  `requirements-rpea.txt` (`/opt/venvs/rpea`: RPEA detector) - and rebuild.
  Hard pins are deliberate where versions constrain each other (numpy, torch,
  TensorFlow, protobuf, numba, the torchsparse commit); everything else floats.
- **apt / ROS binaries:** add to one of the `apt-get install` blocks in
  `humble.Dockerfile`.
- **ROS source packages:** add to `underlay.repos` (pinned by commit); they are
  built into `/opt/underlay_ws`. Commits not reachable from any branch need an
  explicit `git fetch origin <sha>` in the Dockerfile, as done for
  `ranger_ros2`.
- **This repo's packages:** built into `/opt/alpaca_ws` with `--merge-install`;
  add new ones to the `colcon build --packages-select` list.

To tweak a baked-in params file without rebuilding, bind-mount over its
installed copy, e.g.
`-v ~/my_params.yaml:/opt/alpaca_ws/install/share/nav2_params/navigation_params.yaml:ro`.

## Limitations

- **ZED camera.** `zed_wrapper` is not built, since it needs the ZED SDK
  matched to the host driver. `zed_msgs` is installed, so nodes that only
  import the messages work. For the camera, install the SDK in a derived image
  or bind-mount `/usr/local/zed` (commented out in the compose file) and build
  the wrapper in the underlay.
- **Hardware access** (`ranger_base` over USB/serial, the Velodyne) relies on
  `privileged: true` and host networking.
- **Slow first RPEA inference** on RTX 40xx: `torchsparse`/`iou3d` are built
  for sm_70/75/80/86 + PTX, so Ada cards JIT the PTX on first use.
- **Model weights are not in the image**; they must be mounted at `/models`.
