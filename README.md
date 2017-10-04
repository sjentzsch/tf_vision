# tf_vision
Playground for deep learning with tensorflow in the field of robot vision (let's see where this leads to)

Derived from tensorflow research model object_detection:

https://github.com/tensorflow/models/tree/master/research/object_detection

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


