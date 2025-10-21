#!/bin/bash

# run isaac_ros docker without building
$HOME/workspaces/srd_acu_ws/src/isaac_ros_common/scripts/run_dev.sh \
--isaac_ros_dev_dir $HOME/workspaces/aici_gmbh_robotics_ws \
$@
