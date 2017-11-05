#!/bin/bash

echo "(1/2) config local:"
colordiff config.local.sample.yml config.local.yml

echo "(2/2) config obj_detect:"
colordiff config.obj_detect.sample.yml config.obj_detect.yml
