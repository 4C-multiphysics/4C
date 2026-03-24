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
VERSION="5.6.0"
CHECKSUM="3e08c1bdea7aaaba303d3cf03059f3b4336fa49bef93f4260f478f067f518289"

wget --no-verbose https://ftp.mcs.anl.gov/pub/petsc/externalpackages/MUMPS_${VERSION}.tar.gz
# Verify checksum
if [ $CHECKSUM = `sha256sum MUMPS_${VERSION}.tar.gz | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

tar -xzf MUMPS_${VERSION}.tar.gz
cd MUMPS_${VERSION}/
cp Make.inc/Makefile.inc.generic Makefile.inc
make -j${NPROCS} FC=mpif90 FL=mpif90 \
  INCPAR="-I/opt/homebrew/include" \
  LIBPAR="-L/opt/homebrew/lib -lmetis -lparmetis -lscalapack -llapack" \
  ORDERINGSF="-Dpord -Dparmetis" \
  OPTF="-O3 -fallow-argument-mismatch"
mkdir -p ${INSTALL_DIR}
mkdir -p ${INSTALL_DIR}/include
mkdir -p ${INSTALL_DIR}/lib
cp include/*.h ${INSTALL_DIR}/include
cp lib/*.a ${INSTALL_DIR}/lib
cd ../
rm -rf MUMPS_*
