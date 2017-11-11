# tf_vision

On 15th June 2017 Google released their first (pretrained) Computer Vision models via the TensorFlow Object Detection API.  
These models won the COCO detection challenge (http://cocodataset.org/) in 2016. COCO features a large-scale dataset with more than 200k labeled images, 80 object categories and 250k people with keypoints.  

This repository builds upon the Jupyter notebook of TensorFlow's Object Detection API, includes an Ansible playbook to automatically setup your system to run (pretrained) models, allowing for interesting use cases when it comes to infering on video data. Applying it to Robotics is one of the major goals, though.

Firstly, find here some highly recommended links:
* Google initial announcement: https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html
* COCO detection challenge (incl. dataset, results): http://cocodataset.org/
* CVPR 2017 Paper describing the models in detail: https://arxiv.org/abs/1611.10012
* TensorFlow research model and object_detection API: https://github.com/tensorflow/models/tree/master/research/object_detection

This repository assumes you have an Ubuntu 16.04 x86_64. You can choose between using either Python 2.7 or 3.5.

# Requirements for GPU Acceleration

> Note: As of current TensorFlow version 1.4.0: `All our prebuilt binaries have been built with CUDA 8 and cuDNN 6. We anticipate releasing TensorFlow 1.5 with CUDA 9 and cuDNN 7.` However, building from source with 1.4.0 using CUDA 9 and cuDNN 7 is working well already.

For GPU-accelerated tensorflow, ensure you do or have done the following: 

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

3. Install CUDA:

   Download the respective NVIDIA CUDA Toolkit at https://developer.nvidia.com/cuda-downloads (select `deb (local)` in the end), then follow the Installation Instructions below.

   For further notes, refer to: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

   To uninstall previous versions please refer to: http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#handle-uninstallation

   Note: Our Ansible provisioning will take care of setting PATH and LD_LIBRARY_PATH environment variables properly.

3. Install NVIDIA CUDA Deep Neural Network library (cuDNN)

   First, you need to register for the NVIDIA Developer Program at https://developer.nvidia.com/accelerated-computing-developer

   Then, download respective version at https://developer.nvidia.com/rdp/cudnn-download of the following Debian Files (e.g.):

  * cuDNN v7.0 Runtime Library for Ubuntu16.04 (Deb)
  * cuDNN v7.0 Developer Library for Ubuntu16.04 (Deb)
  * cuDNN v7.0 Code Samples and User Guide for Ubuntu16.04 (Deb)

   Install Debians via (e.g.):

   ```
   sudo dpkg -i libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
   sudo dpkg -i libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb
   sudo dpkg -i libcudnn7-doc_7.0.3.11-1+cuda9.0_amd64.deb
   ```

   For further information, see also the Installation Guide (`cuDNN Install Guide`) available at their site.

# Installation

1. Setup log folder for Ansible (replace `sjentzsch` with your local user to run Ansible from):

   ```
   sudo mkdir -p /var/log/ansible/ && sudo chown -R sjentzsch:adm /var/log/ansible/ && sudo chmod 2750 /var/log/ansible/
   ```

2. Create a new and clean folder called e.g. `ml` somewhere, and inside, clone `setup_common_lib` and this GitHub repository (here with SSH keys):

   ```
   git clone ssh://git@scr.bsh-sdd.com:7999/civtm/setup_common_lib.git
   git clone git@github.com:sjentzsch/tf_vision.git
   ```

   Note: Currently, the repository `setup_common_lib` lives inside restricted BSH SDD.

3. Inside `ml/tf_vision/ansible/` setup your machine by running Ansible:

   ```
   ansible-playbook -i hosts -v local.yml -K
   ```

   Ansible will immediately ask you for your sudo password, as packages etc. will need to be installed through root user.

# Run

Inside `ml/tf_vision/` simply run:

```
# Python 2.7
python obj_detect.py
# Python 3.5
python3 obj_detect.py
```
