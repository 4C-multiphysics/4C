#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install parmetis. It will directly install to brew folder. You may need to grant access when installing.
# Call with
# ./install.sh

# Exit the script at the first failure
set -e

INSTALL_DIR=/opt/homebrew
# Number of procs for building (default 4)
NPROCS=${NPROCS=4}
VERSION="4.0.3"
CHECKSUM="f2d9a231b7cf97f1fee6e8c9663113ebf6c240d407d3c118c55b3633d6be6e5f"

# Install parmetis 4.0.3
wget --no-verbose https://ftp.mcs.anl.gov/pub/pdetools/spack-pkgs/parmetis-${VERSION}.tar.gz
# Verify checksum
if [ $CHECKSUM = `sha256sum parmetis-${VERSION}.tar.gz | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

tar -xzf parmetis-${VERSION}.tar.gz
cd parmetis-${VERSION}
make config shared=1 cc=mpicc prefix=${INSTALL_DIR}
make -j${NPROCS} && make install
cd ../
rm -rf parmetis*
