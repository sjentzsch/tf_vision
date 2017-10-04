# tf_vision
Playground for deep learning with tensorflow in the field of robot vision (let's see where this leads to)

Derived from tensorflow research model object_detection:

https://github.com/tensorflow/models/tree/master/research/object_detection

# Installation

1. Create a new and clean folder called e.g. `ml` somewhere, and inside, checkout this GitHub repository (here with SSH keys):

```
git clone git@github.com:sjentzsch/tf_vision.git
```

2. Inside `ml/tf_vision/ansible/` setup your computer by running Ansible via:

```
ansible-playbook -i hosts -v local.yml -K
```

Ansible will immediately ask you for your sudo password, as packages etc. will need to be installed through root user.


