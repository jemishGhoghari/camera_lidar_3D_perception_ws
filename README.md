# AICI GmbH - ROS2 Development Workspace

This repository contains all packages that form the entire software stack of the ACU.

<table>
<tr>
<td>docker</td>
<td>docker related files</td>
</tr>
<tr>
<td>rosbags</td>
<td>collected sensor and software data as rosbags</td>
</tr>
<tr>
<td>src</td>
<td></td>
</tr>
<tr>
<td> multi_sensor_object_mapping</td>
<td>Camera detections and LiDAR based 3D Object detection</td>
</tr>
</table>

## How do I get set up?

### Prerequisites

There are two ways to run the Isaac ROS dev environment: on a desktop pc or on a Jetson device.

1. Install docker using the [instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository) or execute below commands. NOTE: skip this step, If you already have docker installed.

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
sudo apt-get update

# Install the Docker packages.
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

2. Install the nvidia-container-toolkit using the [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt).

```bash
sudo apt-get update && apt-get install -y --no-install-recommends \
   curl \
   gnupg2
```

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```bash
sudo apt-get update
```

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

### Installation

#### Isaac ROS Development Environment

1. Setup docker to run without sudo using the [instructions](https://docs.docker.com/engine/install/linux-postinstall/).
2. Configure the container runtime by using the nvidia-ctk command.

```bash
sudo nvidia-ctk runtime configure --runtime=docker && systemctl --user restart docker
```

> [!NOTE]
> To use the nvidia-container-toolkit on Jetson with Isaac ROS 3.0, you need to generate a [CDI specification](https://github.com/cncf-tags/container-device-interface) file. This file is used to configure the nvidia-container-toolkit to use the correct GPU devices and [PVA accelerators](https://docs.nvidia.com/vpi/architecture.html).
>
> ```bash
> sudo nvidia-ctk cdi generate --mode=csv --output=/etc/cdi/nvidia.yaml
> ```

3. Install git-lfs.

```bash
sudo apt-get install git-lfs
git lfs install --skip-repo
```

4. Clone the repository.

```bash
mkdir ~/workspaces && cd ~/workspaces
git clone git@github.com:jemishGhoghari/aici_gmbh_robotics_ws.git
```

5. Navigate to workspace dir.

```bash
cd ~/workspaces/aici_gmbh_robotics_ws
```

6. Link the config for the additional custom [Isaac ROS Development Environment](https://nvidia-isaac-ros.github.io/concepts/docker_devenv/index.html) image layer.

```bash
ln -s ~/workspaces/aici_gmbh_robotics_ws/docker/.isaac_ros_common-config ~/.isaac_ros_common-config
```

7. Setup the dev convenience script.

```bash
sudo ln -s ~/workspaces/aici_gmbh_robotics_ws/docker/startup/srd_dev_startup.sh ~/srd_dev_startup.sh
```

8. Build and run the docker image using the dev script. Check the [Isaac ROS Common Docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html) for more information.

```bash
~/srd_dev_startup.sh
```

## Development

1. To Launch the dev container, run this command:

```bash
~/srd_dev_startup.sh -b
```

2. To Stop the running container, use this command:

```bash
docker stop isaac_ros_dev-${PLATFORM}-container
```

> [!NOTE] 
>
> The `-b` flag will omit rebuilding the docker image and use the existing one.

1. Build the workspace inside the container.

```bash
colcon build --symlink-install
```
