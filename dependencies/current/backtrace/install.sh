#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install backtrace
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
# Number of procs for building (default 4)
NPROCS=${NPROCS=4}
VERSION="9ae4f4ae4481b1e69d38ed810980d33103544613"
CHECKSUM="7b8fed02dd67bf083f06c6c3a6b30d86b0467a996fd94670c18109a723ab76b7"

# Install backtrace master
wget --no-verbose https://github.com/ianlancetaylor/libbacktrace/archive/${VERSION}.zip
# Verify checksum
if [ $CHECKSUM = `sha256sum ${VERSION}.zip | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

unzip ${VERSION}.zip
cd libbacktrace-${VERSION}
./configure --prefix=${INSTALL_DIR}
make -j${NPROCS} && make install
cd ..
rm -rf libbacktrace-${VERSION} ${VERSION}.zip
