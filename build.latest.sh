#!/bin/bash

# --build-system
# --build-version

set -x

UBUNTU_VER=22.04
_ROS_DISTRO=one
BASEIMAGE_TAG=12.1.0-cudnn8-devel-ubuntu${UBUNTU_VER}
CUDA_IMAGE=nvidia/cuda:${BASEIMAGE_TAG}
REPO=repo.irsl.eiiris.tut.ac.jp/
# IRSL_TAG=''
# IRSL_TAG=_latest
IRSL_TAG=_new_rsl_rl
# IRSL_BASE_IMAGE_NAME=${REPO}irsl_base:cuda_${BASEIMAGE_TAG}_${_ROS_DISTRO}
# IRSL_SYSTEM_IMAGE_NAME=${REPO}irsl_system:cuda_${BASEIMAGE_TAG}_${_ROS_DISTRO}
IRSL_SYSTEM_IMAGE_NAME=${REPO}irsl_system:${_ROS_DISTRO}
OUTPUT_IMAGE_NAME=${REPO}genesis_with_irsl${IRSL_TAG}:${_ROS_DISTRO}

DOCKER_OPT='--progress plain'

# make irsl_system with cuda
FIND_IMAGE=$(docker image ls --format '{{.Repository}}:{{.Tag}}' | grep ${IRSL_SYSTEM_IMAGE_NAME})
if [ -n "${BUILD_SYSTEM}" ]; then
    FIND_IMAGE=
else
    docker pull ${IRSL_SYSTEM_IMAGE_NAME}
fi
if [ -z "${FIND_IMAGE}" ]; then
    cd .irsl_docker_irsl_system/
    INPUT_IMAGE=${IRSL_BASE_IMAGE_NAME} BUILD_ROS=${_ROS_DISTRO} BUILD_UBUNTU=${UBUNTU_VER} ./build.sh ${IRSL_SYSTEM_IMAGE_NAME}
    cd ..
fi

docker build . ${DOCKER_OPT} --build-arg BASE_IMAGE=${IRSL_SYSTEM_IMAGE_NAME} -f Dockerfile.add_genesis${IRSL_TAG} -t ${OUTPUT_IMAGE_NAME}
