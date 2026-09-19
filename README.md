# ALPACA robot software

Social navigation stack for the Alpaca robot: 3D LiDAR pedestrian
detection and tracking, human trajectory prediction, and an MPPI local planner
running under Nav2.

This branch is a standalone snapshot containing only first-party packages.

| Package | Contents |
| --- | --- |
| `alpaca_detection` | RPEA 3D pedestrian detector, SimpleTrack multi-object tracker |
| `alpaca_navigation` | MPPI controller, plan segmenter, trvjectory prediction (HST / EqMotion / AutoBots / MoFlow) |
| `alpaca_interaction` | Voice interaction node (OpenAI realtime API) |
| `alpaca_launch` | Robot bringup, TF publishers, pointcloud filtering |
| `nav2_params` | Nav2, SLAM and EKF parameter files |

<!-- ## Why there are three python environments

The dependency constraints are mutually exclusive, so everything cannot live in
one interpreter:

| Environment | Key pins | Used by |
| --- | --- | --- |
| system `python3` | torch 2.6 + cu124, numpy 1.26 | MPPI controller, SimpleTrack, all ROS nodes |
| HST venv | TensorFlow 2.13, torch 2.0.1, numpy 1.24.3 | prediction node (all backends) |
| RPEA venv | torch 1.13.1, torchsparse, iou3d | RPEA detector |

TensorFlow 2.13 caps numpy at 1.24.3, and torchsparse 1.x only builds against
torch 1.13. The launch files pick the right interpreter through the
`HST_PYTHON` and `RPEA_PYTHON` environment variables, so nothing needs editing
once they are set.

--- -->

# Option A: Docker (recommended)

Everything is prebuilt; no dependency setup at all. Requires an NVIDIA GPU,
Docker, and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
git clone -b submission <repo-url> alpaca_ws && cd alpaca_ws
docker build -f docker/humble.Dockerfile -t ros:alpaca-humble .   # slow, ~15-20 GB
docker compose -f docker/docker-compose.yml run --rm alpaca
```

Put the model weights in `~/alpaca_ws/models` (or set `ALPACA_MODELS_DIR`);
compose mounts that directory at `/models`. Maps go in `~/alpaca_ws/data/maps`
(or set `ALPACA_MAPS_DIR`), mounted at `/maps`. See [docker/README.md](docker/README.md)
for GUI/X11 setup and [Model weights](#model-weights) for the file list.

Inside the container, use the commands in [Running](#running) with the
workspace at `/opt/alpaca_ws` (e.g.
`params_file:=/opt/alpaca_ws/src/nav2_params/navigation_params.yaml`,
`map:=/maps/<map>.yaml`). Steps 1-9 below are already done in the image.

---

# Option B: Native install

## 1. System packages

Ubuntu 22.04 with [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html)
(`ros-humble-desktop`), then:

```bash
sudo apt update && sudo apt install -y \
    python3-colcon-common-extensions python3-rosdep python3-vcstool \
    ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-common \
    ros-humble-pointcloud-to-laserscan ros-humble-robot-localization \
    ros-humble-rtabmap-ros ros-humble-slam-toolbox ros-humble-topic-tools \
    ros-humble-velodyne ros-humble-vision-msgs ros-humble-zed-msgs \
    ros-humble-tf2-geometry-msgs ros-humble-cv-bridge \
    espeak-ng libasound2 libportaudio2 python3-gi python3-tk \
    gir1.2-gstreamer-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    pulseaudio-utils
```

`ros-humble-cv-bridge` and `pulseaudio-utils` are for TalkNCE active-speaker
detection (`alpaca_interaction/talknce/`, see [Running](#running)).

The audio packages are for the voice node (portaudio), speech synthesis
(espeak) and sound playback (GStreamer).

## 2. Workspace

Keep the `<workspace>/src/<package>` layout. The prediction code finds the
workspace by looking for sibling `src/` and `install/` directories, and loads
its HST model params from the source tree.

```bash
mkdir -p ~/alpaca_ws/src && cd ~/alpaca_ws/src
git clone -b submission <repo-url> .

# Third-party ROS packages this repo does not carry, pinned to known-good commits
vcs import < docker/underlay.repos

# ranger_ros2 at a commit not on any branch (has RCState.msg / /rc_state for
# --safety rc), so it must be fetched by SHA
git clone https://github.com/agilexrobotics/ranger_ros2.git
git -C ranger_ros2 fetch origin 4357bdfb9fda828e4b84880e9c34a874cf060b80
git -C ranger_ros2 checkout FETCH_HEAD
```

That pulls `ros2_numpy` (used by the RPEA node), `ugv_sdk`, and `ranger_ros2`
(`ranger_msgs`, `ranger_base`, `ranger_bringup`) for the robot base.

## 3. Main python environment

Installed system-wide so the colcon-installed nodes import it directly:

```bash
cd ~/alpaca_ws/src
pip3 install -r docker/requirements-ros.txt
```

## 4. Prediction (HST) environment

`--system-site-packages` keeps `rclpy` importable; the venv's own
numpy/torch/tensorflow take precedence.

```bash
python3 -m venv --system-site-packages ~/venvs/hst
~/venvs/hst/bin/pip install -r ~/alpaca_ws/src/docker/requirements-hst.txt
```

## 5. RPEA detector environment

This is the only step needing a CUDA toolkit: `torchsparse` and `iou3d` are
compiled from source against torch 1.13. Install
[CUDA toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)
and `sudo apt install -y libsparsehash-dev` first.

```bash
python3 -m venv --system-site-packages ~/venvs/rpea
source ~/venvs/rpea/bin/activate
pip install -r ~/alpaca_ws/src/docker/requirements-rpea.txt

# Ada GPUs (RTX 40xx) are sm_89, which CUDA 11.7 cannot target; the 8.6 PTX
# is JIT-compiled on first use instead.
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" FORCE_CUDA=1
pip install --no-build-isolation \
    "torchsparse @ git+https://github.com/mit-han-lab/torchsparse.git@e268836e64513b9a31c091cd1d517778d4c1b9e6"

cd ~/alpaca_ws/src/alpaca_detection/alpaca_detection/RPEA
pip install --no-build-isolation ./lib/iou3d   # CUDA extension
pip install --no-deps .                        # lidar_det (pure python)
deactivate
```

## 6. Build

```bash
cd ~/alpaca_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -y --skip-keys "zed_wrapper"
colcon build --symlink-install
source install/setup.bash
```

`zed_wrapper` is skipped because the ZED camera needs the Stereolabs SDK; see
[Limitations](#limitations).

## 7. Environment variables

Add to `~/.bashrc`:

```bash
export HST_PYTHON=$HOME/venvs/hst/bin/python
export RPEA_PYTHON=$HOME/venvs/rpea/bin/python
export ALPACA_MODELS_DIR=$HOME/alpaca_ws/models
export RPEA_CHECKPOINT=$ALPACA_MODELS_DIR/RPEA_JRDB2022.pth
export OPENAI_API_KEY=...        # only for alpaca_interaction
```

## 8. Fix the absolute paths in the Nav2 params

`nav2_params/navigation_params.yaml` points `default_nav_to_pose_bt_xml` and
`default_nav_through_poses_bt_xml` at an absolute path from the development
machine. Nav2's `bt_navigator` fails at runtime otherwise:

```bash
cd ~/alpaca_ws/src/nav2_params
sed -i "s|/home/user/alpaca_ws/src/nav2_params|$PWD|g" navigation_params.yaml
```

## 9. Retarget Nav2's velocity_smoother output

Stock `nav2_bringup` remaps `velocity_smoother`'s output to `/cmd_vel`, which
collides with the command that actually drives the robot (from
`mppi_controller_node`, not Nav2's controller). Send it to `/cmd_vel_debug`:

```bash
sudo sed -i "s/('cmd_vel_smoothed', 'cmd_vel')/('cmd_vel_smoothed', 'cmd_vel_debug')/g" \
    /opt/ros/humble/share/nav2_bringup/launch/navigation_launch.py
```

---

## Model weights

Not in the repo; put them in `$ALPACA_MODELS_DIR`, default is `~/alpaca_ws/models`:

| File | Needed by |
| --- | --- |
| `RPEA_JRDB2022.pth` (275 MB) | RPEA detector |
| `best_models_ade_autobots_da.pth`, `config_autobots_da.yaml` | `autobots` backend |
| `best_models_ade_student20_da.pth`, `config_student20_da.yaml`, `moflow_joint_student_eval.yaml` | `moflow` backend |
| `best_models_ade_eqmotion5.pth` + its config | `eqmotion` backend |
| `UniTalk_TalkNCE.model` (132 MB) | TalkNCE active-speaker detection |

The default `hst` backend needs nothing extra: its checkpoint ships in
`alpaca_navigation/alpaca_navigation/prediction/jrdb_no_keypoints/`.

## Verify the install

```bash
ros2 pkg list | grep -E "alpaca|nav2_params"          # expect 5 packages
python3 -c "import torch, pytorch_mppi; print(torch.__version__, torch.cuda.is_available())"
$HST_PYTHON  -c "import tensorflow, torch; print(tensorflow.__version__, torch.__version__)"
$RPEA_PYTHON -c "import torch, torchsparse, iou3d, lidar_det; print('rpea ok', torch.__version__)"
ros2 launch alpaca_detection RPEA_launch.py --show-args
```

## Running

```bash
# Robot bringup: base, LiDAR, camera, TF
ros2 launch alpaca_launch alpaca_bringup.launch.py use_lidar:=true use_camera:=false

# Detection + tracking
ros2 launch alpaca_detection RPEA_launch.py score_threshold:=0.8

# Trajectory prediction (backends: hst, cv, eqmotion, autobots, moflow)
ros2 launch alpaca_navigation hst_prediction_launch.py prediction_backend:=hst

ros2 launch nav2_bringup bringup_launch.py use_sim_time:=False \
    map:=/path/to/map.yaml \
    params_file:=$HOME/alpaca_ws/src/nav2_params/navigation_params.yaml

ros2 run alpaca_navigation mppi_controller_node \
    --params-file $HOME/alpaca_ws/src/alpaca_navigation/config/mppi_params.yaml \
    --pose-source amcl --use-hst --safety rc \
```

The `--safety` flag 
- `rc` - commands are published only while the RC transmitter's SWA switch reads 0 (flicked up). The node blocks at startup until something publishes `/rc_state`.

In every mode the unfiltered command is also published on `/cmd_vel_debug`,
which is useful for testing without moving the robot. See
`alpaca_navigation/mppi_controller_node.py` for the other flags
(`--publish-costs`, `--save-rollouts`, etc.).

`map` has no default; maps are not checked in.

Velodyne bringup arguments:

- `velodyne_device_ip` (default `192.168.3.201`) - the driver silently drops
  packets from any other address, so it looks like no data is arriving.
- `velodyne_organize_cloud` (default `false`) - `true` builds a structured
  cloud but costs enough CPU to drop `/velodyne_points` from ~10 Hz to ~1 Hz
  with the full stack running.

Active-speaker detection (TalkNCE), optional:

```bash
ros2 run alpaca_interaction talknce_node \
    --checkpoint $ALPACA_MODELS_DIR/UniTalk_TalkNCE.model \
    --image-topic /zed/zed_node/rgb_raw/image_raw_color/compressed \
    --output-topic /talknce/image_annotated
```

Audio is captured with `parec` from PulseAudio/PipeWire. Pass
`--audio-device <source>` (list with `--list-pulse-sources`, or set
`TALKNCE_AUDIO_DEVICE`), or omit it to use the default input.

## Limitations

- **ZED camera.** `zed_wrapper` is not built here; it needs the Stereolabs SDK
  matched to the host driver. `zed_msgs` comes from apt, so nodes that only
  import the messages work. Set `use_camera:=false` in bringup.
- **First RPEA inference is slow** on RTX 40xx cards: the CUDA kernels are
  JIT-compiled from PTX, then cached.
- **MoFlow backend** requests `numpy==2.2.4` upstream, but the HST environment
  is capped at 1.24.3 by TensorFlow. The vendored model code only does ordinary
  array work, so this is expected to be fine but has not been verified against
  a live checkpoint.
