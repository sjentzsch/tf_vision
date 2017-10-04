# tf_vision
Playground for deep learning with tensorflow in the field of robot vision (let's see where this leads to)

Derived from tensorflow research model object_detection:

https://github.com/tensorflow/models/tree/master/research/object_detection

For GPU-accelerated tensorflow under Ubuntu 16.04 x86_64, ensure you do or have done the following: 

1. You have a GPU with CUDA Compute Capability 3.0 or higher.  
   Check via https://developer.nvidia.com/cuda-gpus for a list of supported GPU cards.  
   The command `lspci | grep -i nvidia` shows your current GPU.

2. Check if you are using the latest (proprietary) NVIDIA binary drivers: 

   Check remotely available ones on http://www.nvidia.com/Download/index.aspx?lang=en-us

   Check locally installed one via `cat /proc/driver/nvidia/version`

   Add the Ubuntu PPA (https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa?field.series_filter=xenial):

   ```
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt-get update
   ```

   Then switch to respective latest stable version branch (System Settings -> Software & Updates -> Additional Drivers).

3. Install latest CUDA:

   Download the latest NVIDIA CUDA Toolkit at https://developer.nvidia.com/cuda-downloads (select `deb (local)` in the end), then follow the Installation Instructions below.

   For further notes, refer to: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

   To uninstall previous versions please refer to: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#handle-uninstallation

   Note: Our Ansible provisioning will take care of setting PATH and LD_LIBRARY_PATH environment variables properly.

3. Install latest NVIDIA CUDA Deep Neural Network library (cuDNN)

   Download latest version at https://developer.nvidia.com/rdp/cudnn-download

  * cuDNN v7.0 Library for Linux
  * cuDNN v7.0 Runtime Library for Ubuntu16.04 (Deb)
  * cuDNN v7.0 Developer Library for Ubuntu16.04 (Deb)

   See also the Installation Guide (`cuDNN Install Guide`)

# Installation

1. Setup log folder for Ansible (replace `sjentzsch` with your local user to run Ansible from):

```
sudo mkdir -p /var/log/ansible/ && sudo chown -R sjentzsch:adm /var/log/ansible/ && sudo chmod 2750 /var/log/ansible/
```

2. Create a new and clean folder called e.g. `ml` somewhere, and inside, checkout `setup_common_lib` and this GitHub repository (here with SSH keys):

```
git clone ssh://git@scr.bsh-sdd.com:7999/civtm/setup_common_lib.git
git clone git@github.com:sjentzsch/tf_vision.git
```

Note: Currently, the shared library `setup_common_lib.git` lives inside BSH SDD.

3. Inside `ml/tf_vision/ansible/` setup your computer by running Ansible via:

```
ansible-playbook -i hosts -v local.yml -K
```

Ansible will immediately ask you for your sudo password, as packages etc. will need to be installed through root user.


