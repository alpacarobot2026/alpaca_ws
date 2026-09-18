#!/usr/bin/env bash
# Sources ROS, the third-party underlay and the alpaca overlay, then runs the
# command. Keeps `docker run ... ros2 launch ...` working with no setup.
set -e

source /opt/ros/humble/setup.bash
[ -f /opt/underlay_ws/install/setup.bash ] && source /opt/underlay_ws/install/setup.bash
[ -f /opt/alpaca_ws/install/setup.bash ] && source /opt/alpaca_ws/install/setup.bash

exec "$@"
