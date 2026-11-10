# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.description="Image containing all the dependencies required for building and testing 4C"
LABEL org.4c-multiphysics.project=4C

# Prevents tzdata asking for user feedback
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Set locale information:. region and timezone
RUN apt-get update && apt-get install -y locales && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

COPY dependencies/current/apt_packages_ubuntu24.04.txt /tmp/apt_packages.txt
RUN apt-get update && xargs -a /tmp/apt_packages.txt apt-get install -y \
  && rm -rf /var/lib/apt/lists/*

# Create directory for dependencies
ARG NPROCS=12
ENV NPROCS=$NPROCS \
    INSTALL_DIR="/opt/4C-dependencies"
RUN mkdir -p ${INSTALL_DIR}

COPY dependencies /dependencies
# Install cmake
RUN /dependencies/current/cmake/install.sh /usr/local

# Install superLU_dist
RUN /dependencies/current/superlu_dist/install.sh ${INSTALL_DIR}

# Install Trilinos. The variant selects which install script is used (current, oldest_supported,
# trilinos_develop, ...), TRILINOS_VERSION optionally overrides the checked-out ref, and
# TRILINOS_ENABLE_DEBUG overrides the variant's default assertion setting (empty = variant default).
ARG TRILINOS_VARIANT=current
ARG TRILINOS_VERSION=
ARG TRILINOS_ENABLE_DEBUG=
RUN TRILINOS_VERSION="${TRILINOS_VERSION}" \
    TRILINOS_ENABLE_DEBUG="${TRILINOS_ENABLE_DEBUG}" \
    /dependencies/${TRILINOS_VARIANT}/trilinos/install.sh ${INSTALL_DIR}

# Install deal.II (needs to happen after Trilinos)
RUN /dependencies/current/dealii/install.sh ${INSTALL_DIR}

# Install (optional) backtrace library
RUN /dependencies/current/backtrace/install.sh ${INSTALL_DIR}

# install (optional) gmsh library
RUN /dependencies/current/gmsh/install.sh ${INSTALL_DIR}

# Packages for testing
# Installation directory for dependencies concerning testing
ENV FOUR_C_TESTING_DEPENDENCIES_DIR="/opt/4C-dependencies-testing/"
RUN mkdir ${FOUR_C_TESTING_DEPENDENCIES_DIR}

# Install Mathjax
RUN /dependencies/testing/mathjax/install.sh ${FOUR_C_TESTING_DEPENDENCIES_DIR}

# Add dependencies hash
# The label is added at the end because the label causes a cache miss
ARG DEPENDENCIES_HASH
LABEL org.4c-multiphysics.dependencies_hash="${DEPENDENCIES_HASH}"
ENV DEPENDENCIES_HASH=${DEPENDENCIES_HASH}

# add and enable the default user
ENV USER=user
RUN adduser --disabled-password --shell '/usr/bin/bash' --gecos '' $USER
RUN adduser $USER sudo; echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

#make sure everything is in place
RUN chown -R $USER:$USER /home/$USER
USER $USER
ENV HOME=/home/$USER
ENV USER=$USER
# see https://github.com/open-mpi/ompi/issues/4948
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
WORKDIR $HOME
