#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install SuiteSparse
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS=4}
PREFIX="mumps-debian-"
VERSION="5.6.2-2"
CHECKSUM="5555eead9891938a54f12bf5c0cbd77e906648bb5409e97d69f29adaaf59a295"

wget --no-verbose https://salsa.debian.org/science-team/mumps/-/archive/debian/${VERSION}/${PREFIX}${VERSION}.tar.gz
# Verify checksum
if [ $CHECKSUM = `sha256sum ${PREFIX}${VERSION}.tar.gz | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

tar -xzf ${PREFIX}${VERSION}.tar.gz
cd ${PREFIX}${VERSION}
cp Make.inc/Makefile.inc.generic Makefile.inc
make -j${NPROCS} FC=mpif90 FL=mpif90 \
  INCPAR="-I/opt/homebrew/include -I$HOME/opt/parmetis/include" \
  LIBPAR="-L/opt/homebrew/lib -lmetis -lscalapack -llapack -L$HOME/opt/parmetis/lib -lparmetis" \
  ORDERINGSF="-Dpord -Dparmetis" \
  OPTF="-O3 -fallow-argument-mismatch"
mkdir -p ${INSTALL_DIR}
mkdir -p ${INSTALL_DIR}/include
mkdir -p ${INSTALL_DIR}/lib
cp include/*.h ${INSTALL_DIR}/include
cp lib/*.a ${INSTALL_DIR}/lib
cd ../
rm -rf MUMPS_*
