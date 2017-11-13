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
   sudo mkdir -p /var/log/ansible/
   sudo chown -R sjentzsch:adm /var/log/ansible/
   sudo chmod 2750 /var/log/ansible/
   ```

2. Create a new and clean folder called e.g. `ml` somewhere, and inside, clone `setup_common_lib` and this GitHub repository (here with SSH keys):

   ```
   git clone ssh://git@scr.bsh-sdd.com:7999/civtm/setup_common_lib.git
   git clone git@github.com:sjentzsch/tf_vision.git
   ```

   Note: Currently, the repository `setup_common_lib` lives inside restricted BSH SDD.

3. Inside `ml/tf_vision/config/` the YAML file `config.local.sample.yml` contains the default configuration on how to provision your machine.

   This includes e.g. enabling or disabling TensorFlow GPU acceleration, TensorFlow version to use, OpenCV version to use, Python version to use.

   In order to overwrite default values, create a local copy `cp config.local.sample.yml config.local.yml` and modify values inside `config.local.yml` only.

4. Inside `ml/tf_vision/ansible/` setup your machine by running Ansible:

   ```
   ansible-playbook -i hosts -v local.yml -K
   ```

   Ansible will immediately ask you for your sudo password, as packages etc. will need to be installed through root user.

# Build TensorFlow from Source

> Note: As of now, if you configured to build TensorFlow from Source, you need to follow this guide after running the Ansible playbook. In the future, this should mostly be handled by Ansible.

First, make sure to remove previous builds again (choose pip2 for Python 2 or pip3 for Python 3), e.g.:

```
sudo pip3 uninstall tensorflow tensorflow-tensorboard
```

For a new build, inside `ml/tensorflow` (default location), run `./configure`.

Here is my sample dialogue and the answers I gave to the questions: 

> Note: Ensure to refer to the proper Python version you want to use! Here I compiled with Python 3.5.

```
Extracting Bazel installation...
You have bazel 0.7.0 installed.
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3


Found possible Python library paths:
  /usr/local/lib/python3.5/dist-packages
  /usr/lib/python3/dist-packages
Please input the desired Python library path to use.  Default is [/usr/local/lib/python3.5/dist-packages]

Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: 
jemalloc as malloc support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
No Google Cloud Platform support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
No Hadoop File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
No Amazon S3 File System support will be enabled for TensorFlow.

Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with GDR support? [y/N]: n
No GDR support will be enabled for TensorFlow.

Do you wish to build TensorFlow with VERBS support? [y/N]: n
No VERBS support will be enabled for TensorFlow.

Do you wish to build TensorFlow with OpenCL support? [y/N]: n
No OpenCL support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: y
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: 9.0


Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 


Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 7


Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:/usr/lib/x86_64-linux-gnu


Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 5.0]5.0


Do you want to use clang as CUDA compiler? [y/N]: n
nvcc will be used as CUDA compiler.

Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 


Do you wish to build TensorFlow with MPI support? [y/N]: n
No MPI support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 


Add "--config=mkl" to your bazel command to build with MKL support.
Please note that MKL on MacOS or windows is still not supported.
If you would like to use a local MKL instead of downloading, please set the environment variable "TF_MKL_ROOT" every time before build.
Configuration finished
```

Then trigger the building procedure (takes about 23 minutes for me):

```
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

Next create the package:

```
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

Finally install the package (choose proper pip version again! .whl-file can be named differently, investigate the folder!):

```
sudo pip3 install /tmp/tensorflow_pkg/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl
```

See also: https://www.tensorflow.org/install/install_sources

# Run

1. Inside `ml/tf_vision/config/` the YAML file `config.obj_detect.sample.yml` contains the default configuration to run.

   This includes e.g. the model to use for object detection, the source for the video stream to analyze upon, and if certain components (like the visualizer or the speech synthesis) should be enabled.

   In order to overwrite default values, create a local copy `cp config.obj_detect.sample.yml config.obj_detect.yml` and modify values inside `config.obj_detect.yml` only.

2. Inside `ml/tf_vision/` simply run:

   ```
   # Python 2.7
   python obj_detect.py
   # Python 3.5
   python3 obj_detect.py
   ```
