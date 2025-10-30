#!/bin/bash

IRSL_TAG=_new_rsl_rl
# IRSL_TAG=_latest
# IRSL_TAG=''

_UBUNTU_VER=22.04
_ROS_DISTRO=one
REPO=repo.irsl.eiiris.tut.ac.jp/
IRSL_SYSTEM_IMAGE_NAME=${REPO}irsl_system:22.04_one
OUTPUT_IMAGE_NAME=${REPO}genesis_with_irsl${IRSL_TAG}:one

# .irsl_docker_irsl_system/run.sh --image ${OUTPUT_IMAGE_NAME} -w ${PWD}/userdir${IRSL_TAG} "$@"
.irsl_docker_irsl_system/run.sh --image ${OUTPUT_IMAGE_NAME} -w ${PWD} "$@"
