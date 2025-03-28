# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

# Install deal.II

# Create directory for dependencies
ARG NPROCS=12
ENV NPROCS=$NPROCS \
    INSTALL_DIR="/opt/4C-dependencies"

COPY dependencies /dependencies
# Install cmake
RUN /dependencies/dealii/install.sh ${INSTALL_DIR}

# Copy everything to a new image to get rid of the previous layers
FROM scratch
COPY --from=0 / /
LABEL org.opencontainers.image.description="Image containing all the dependencies required for building and testing 4C with deal.II"
LABEL org.4c-multiphysics.project=4C

USER user
WORKDIR /home/user

CMD [ "/bin/bash" ]