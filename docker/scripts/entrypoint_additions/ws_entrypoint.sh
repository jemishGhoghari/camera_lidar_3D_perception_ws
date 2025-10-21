#!/bin/bash

# source ROS and Isaac ROS workspace
source /opt/ros/${ROS_DISTRO}/setup.bash
source ${ISAAC_ROS_WS}/install/setup.bash
export RCUTILS_COLORIZED_OUTPUT=1

# colcon command completion for bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
