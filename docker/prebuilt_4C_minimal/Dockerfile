# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

FROM ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04:main

USER root

# Remove unnecessary packages
RUN apt-get update && apt-get -y remove \
      doxygen \
      clang* \
      llvm* \
  && apt-get -y autoremove \
  && rm -rf /var/lib/apt/lists/* \
  && rm -rf /usr/share/doc

# Remove testing dependencies
RUN rm -r /opt/4C-dependencies-testing/

RUN git clone --depth 1 https://github.com/4C-multiphysics/4C.git 4C-source

RUN mkdir 4C-source/build && cd 4C-source/build && \
  cmake ../ --fresh -DCMAKE_INSTALL_PREFIX=/home/user/4C --preset=docker_minimal && \
  cmake --build . --target install -- -j `nproc`

RUN ln -s /home/user/4C/bin/4C /usr/local/bin/4C

# Remove build artifacts
RUN rm -r /home/user/4C-source

# Copy everything to a new image to get rid of the intermediate layers
FROM scratch
COPY --from=0 / /
LABEL org.opencontainers.image.description="Image containing the 4C executable. All optional dependencies compatible with 4C's license are enabled. Built with release dependencies (no Trilinos assertions)."
LABEL org.4c-multiphysics.project=4C

USER user
WORKDIR /home/user

CMD [ "/bin/bash" ]
