# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:latest

USER root

# Install deal.II

# Create directory for dependencies
ARG NPROCS=12
ENV NPROCS=$NPROCS \
    INSTALL_DIR="/opt/4C-dependencies"

COPY dependencies /dependencies
# Install cmake
RUN /dependencies/current/dealii/install.sh ${INSTALL_DIR}

# Copy everything to a new image to get rid of the previous layers
FROM scratch
COPY --from=0 / /
LABEL org.opencontainers.image.description="Image containing the 4C executable"
LABEL org.4c-multiphysics.project=4C

USER user
WORKDIR /home/user

CMD [ "/bin/bash" ]