#!/bin/bash

# add user to additional groups
adduser ${USERNAME} zed >/dev/null
adduser ${USERNAME} dialout >/dev/null

if [[ -n "$GPIO_GID" && "$GPIO_GID" =~ ^[0-9]+$ ]]; then
    if ! getent group ${GPIO_GID} > /dev/null; then
        groupadd -g ${GPIO_GID} gpio
    fi
    adduser ${USERNAME} gpio >/dev/null
fi

# source ROS and Isaac ROS workspace
source /opt/ros/${ROS_DISTRO}/setup.bash
source ${ISAAC_ROS_WS}/install/setup.bash
export RCUTILS_COLORIZED_OUTPUT=1

# colcon command completion for bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
