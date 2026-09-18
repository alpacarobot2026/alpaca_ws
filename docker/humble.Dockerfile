# syntax=docker/dockerfile:1
#
# ROS 2 Humble + Nav2 + the alpaca packages, with three python environments:
#
#   system python3    ROS nodes, MPPI controller, SimpleTrack tracker
#   /opt/venvs/hst    TensorFlow 2.13 + torch 2.0.1  (HST prediction node)
#   /opt/venvs/rpea   torch 1.13.1 + torchsparse + iou3d  (RPEA detector)
#
# Three environments are needed because the version constraints are mutually
# exclusive: TensorFlow 2.13 caps numpy at 1.24.3, and torchsparse 1.x only
# builds against torch 1.13. Both venvs use --system-site-packages so rclpy and
# the ROS message packages stay importable.
#
# Build from the repository root:
#   docker build -f docker/humble.Dockerfile -t ros:alpaca-humble .

########################  stage 1: compile RPEA CUDA extensions  ##############
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS rpea-builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Detroit

# torchsparse needs sparsehash headers; iou3d needs nvcc and a C++ toolchain.
RUN apt-get update && apt-get install --yes --no-install-recommends \
        build-essential \
        git \
        libsparsehash-dev \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# CUDA 11.7 has no sm_89 target, so Ada cards (RTX 40xx) JIT the 8.6 PTX.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ENV FORCE_CUDA=1

RUN python3 -m venv /opt/build-venv
ENV PATH=/opt/build-venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip "setuptools<81" wheel \
    && pip install --no-cache-dir "numpy<2" torch==1.13.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

# Only the wheels are copied forward, so the runtime image needs no nvcc.
# The pinned commit does `from collections import Sequence`, removed in
# Python 3.10 (this image's interpreter); patch it before building the wheel.
RUN git clone --quiet https://github.com/mit-han-lab/torchsparse.git /src/torchsparse \
    && cd /src/torchsparse \
    && git checkout --quiet e268836e64513b9a31c091cd1d517778d4c1b9e6 \
    && sed -i 's/from collections import Sequence/from collections.abc import Sequence/' \
        torchsparse/utils/helpers.py
WORKDIR /wheels
RUN pip wheel --no-cache-dir --no-deps --no-build-isolation /src/torchsparse
COPY alpaca_detection/alpaca_detection/RPEA/lib/iou3d /src/iou3d
RUN pip wheel --no-cache-dir --no-deps --no-build-isolation /src/iou3d

########################  stage 2: runtime image  #############################
FROM osrf/ros:humble-desktop AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Detroit

# Commonly-used development tools. Jammy apt has CMake 3.22, so no manual install.
RUN apt-get update && apt-get install --yes \
        build-essential \
        ccache \
        clang \
        clang-format \
        clang-tidy \
        cmake \
        g++ \
        gdb \
        git \
        git-lfs \
        nano \
        ninja-build \
        valgrind \
        vim \
    && rm -rf /var/lib/apt/lists/*

# Commonly-used command-line tools.
RUN apt-get update && apt-get install --yes \
        curl \
        ffmpeg \
        iproute2 \
        iputils-ping \
        less \
        mesa-utils \
        net-tools \
        parallel \
        rsync \
        software-properties-common \
        tmux \
        tree \
        unzip \
        usbutils \
        wget \
        xxhash \
        zip \
        zsh \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Python tooling. python3-tk is for alpaca_face_display.
RUN apt-get update && apt-get install --yes \
        python-is-python3 \
        python3-dev \
        python3-pip \
        python3-tk \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Fonts for matplotlib.
RUN apt-get update && apt-get install --yes fonts-urw-base35 \
    && rm -rf /var/lib/apt/lists/*

# Audio stack: sounddevice needs portaudio, pyttsx3 needs espeak, playsound
# goes through GStreamer via PyGObject, and TalkNCE's parec/pactl audio
# capture (alpaca_interaction/talknce) needs pulseaudio-utils.
RUN apt-get update && apt-get install --yes \
        espeak-ng \
        gir1.2-gstreamer-1.0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        libasound2 \
        libportaudio2 \
        pulseaudio-utils \
        python3-gi \
    && rm -rf /var/lib/apt/lists/*

# ROS packages referenced by the alpaca launch files, plus build tooling.
RUN apt-get update && apt-get install --yes \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
        ros-humble-cv-bridge \
        ros-humble-cyclonedds \
        ros-humble-nav2-bringup \
        ros-humble-nav2-common \
        ros-humble-navigation2 \
        ros-humble-pointcloud-to-laserscan \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-robot-localization \
        ros-humble-rtabmap-ros \
        ros-humble-slam-toolbox \
        ros-humble-foxglove-bridge \
        ros-humble-teleop-twist-keyboard \
        ros-humble-tf2-geometry-msgs \
        ros-humble-topic-tools \
        ros-humble-twist-mux \
        ros-humble-velodyne \
        ros-humble-vision-msgs \
        ros-humble-zed-msgs \
    && rm -rf /var/lib/apt/lists/*

# Stock nav2_bringup remaps velocity_smoother's output cmd_vel_smoothed to
# /cmd_vel by default, colliding with twist_mux publishing the arbitrated
# command to that same topic (the external mppi_controller_node, not Nav2's
# own controller, is what actually drives the robot). Retarget it to
# cmd_vel_debug instead, in both the composed and non-composed node blocks.
RUN sed -i "s/('cmd_vel_smoothed', 'cmd_vel')/('cmd_vel_smoothed', 'cmd_vel_debug')/g" \
        /opt/ros/humble/share/nav2_bringup/launch/navigation_launch.py

# Grant passwordless sudo to the "sudo" group. Must be baked in at build
# time (root-owned layer) rather than bind-mounted at runtime, since sudo
# refuses to honor a sudoers.d file that isn't owned by uid 0, and
# enter_ros.sh runs the container as a non-root spoofed host user.
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/local-user \
    && chmod 0440 /etc/sudoers.d/local-user

# Main ROS python environment, installed system-wide.
COPY docker/requirements-ros.txt /tmp/requirements-ros.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements-ros.txt \
    && rm /tmp/requirements-ros.txt

# HST prediction environment. The venv's numpy/torch/tensorflow shadow the
# system ones, while rclpy stays visible through system site-packages.
COPY docker/requirements-hst.txt /tmp/requirements-hst.txt
RUN python3 -m venv --system-site-packages /opt/venvs/hst \
    && /opt/venvs/hst/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venvs/hst/bin/pip install --no-cache-dir -r /tmp/requirements-hst.txt \
    && rm /tmp/requirements-hst.txt

# RPEA environment, using the wheels compiled in the builder stage.
COPY docker/requirements-rpea.txt /tmp/requirements-rpea.txt
COPY --from=rpea-builder /wheels /tmp/wheels
RUN python3 -m venv --system-site-packages /opt/venvs/rpea \
    && /opt/venvs/rpea/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venvs/rpea/bin/pip install --no-cache-dir -r /tmp/requirements-rpea.txt \
    && /opt/venvs/rpea/bin/pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels /tmp/requirements-rpea.txt

# Third-party ROS packages the launch files import that this repo does not carry:
# ros2_numpy (RPEA node), ranger_msgs/ranger_base/ranger_bringup and ugv_sdk.
COPY docker/underlay.repos /tmp/underlay.repos
RUN mkdir -p /opt/underlay_ws/src \
    && vcs import --input /tmp/underlay.repos /opt/underlay_ws/src \
    # ranger_ros2 isn't in underlay.repos: the commit with RCState.msg and
    # /rc_state (needed by mppi_controller_node's --safety rc mode) isn't on
    # any branch, so a plain `git clone` (what vcs import's git driver does)
    # can't reach it - `git checkout <sha>` after that clone fails with
    # "reference is not a tree". GitHub does still serve the object by exact
    # SHA (`git fetch origin <sha>`) even though it's unreferenced by a
    # branch, so fetch+checkout it directly instead.
    && git clone https://github.com/agilexrobotics/ranger_ros2.git /opt/underlay_ws/src/ranger_ros2 \
    && cd /opt/underlay_ws/src/ranger_ros2 \
    && git fetch origin 4357bdfb9fda828e4b84880e9c34a874cf060b80 \
    && git checkout FETCH_HEAD \
    && cd /opt/underlay_ws \
    && apt-get update \
    && rosdep update --rosdistro humble \
    && rosdep install --from-paths /opt/underlay_ws/src --ignore-src -y --rosdistro humble \
    && . /opt/ros/humble/setup.sh \
    && colcon build --merge-install \
    && rm -rf /opt/underlay_ws/build /opt/underlay_ws/log /tmp/underlay.repos /var/lib/apt/lists/*

# The alpaca packages. The layout must stay /opt/alpaca_ws/src/<package>: the
# prediction code finds the workspace by walking up for sibling src/ and
# install/ dirs, and loads HST model params (.gin) from the source tree, which
# setuptools never installs.
COPY . /opt/alpaca_ws/src/
# nav2_params' BT xml paths are hardcoded to the development machine's
# checkout (/home/user/alpaca_ws/src/nav2_params); bt_navigator fails to
# load its tree files otherwise. Fix in place before the build so both this
# source-tree copy and the --merge-install'd copy get the corrected path.
RUN sed -i 's|/home/user/alpaca_ws/src/nav2_params|/opt/alpaca_ws/src/nav2_params|g' \
        /opt/alpaca_ws/src/nav2_params/navigation_params.yaml \
        /opt/alpaca_ws/src/nav2_params/navigation_params_atrium.yaml \
        /opt/alpaca_ws/src/nav2_params/navigation_params_sim.yaml
RUN . /opt/ros/humble/setup.sh \
    && . /opt/underlay_ws/install/setup.sh \
    && cd /opt/alpaca_ws \
    && colcon build --merge-install --packages-select \
        alpaca_detection \
        alpaca_interaction \
        alpaca_launch \
        alpaca_navigation \
        nav2_params \
    && rm -rf /opt/alpaca_ws/build /opt/alpaca_ws/log

# RPEA's own python package (lidar_det). Pure python, so no compiler needed.
RUN /opt/venvs/rpea/bin/pip install --no-cache-dir --no-deps \
        /opt/alpaca_ws/src/alpaca_detection/alpaca_detection/RPEA

# Model weights are too large to ship in the repo, so /models is a mount point.
# The symlink makes the prediction code's default <workspace>/models path resolve.
RUN mkdir -p /models && ln -s /models /opt/alpaca_ws/models

ENV ALPACA_MODELS_DIR=/models \
    HST_PYTHON=/opt/venvs/hst/bin/python \
    RPEA_CHECKPOINT=/models/RPEA_JRDB2022.pth \
    RPEA_PYTHON=/opt/venvs/rpea/bin/python

# ZED SDK is bind-mounted at runtime (see docker/README.md) rather than
# installed here, so it matches whatever SDK version is on the host.
RUN mkdir -p /usr/local/zed
ENV LD_LIBRARY_PATH=/usr/local/zed/lib

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
