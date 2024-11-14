#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install mathjax
# Call with
# ./install.sh /path/to/install/dir

# Exit the script at the first failure
set -e

INSTALL_DIR="$1"
VERSION="2.7.9"
CHECKSUM="c9167279c87da044f2ff910ad573a02ce90354cb59440ae568eb86e1630f65df"

wget --no-verbose https://github.com/mathjax/MathJax/archive/refs/tags/${VERSION}.tar.gz
# Verify checksum
if [ $CHECKSUM = `sha256sum ${VERSION}.tar.gz | awk '{print $1}'` ]
then
  echo "Checksum matches"
else
  echo "Checksum does not match"
  exit 1
fi

tar -xzf ${VERSION}.tar.gz -C ${INSTALL_DIR}
rm -rf ${VERSION}.tar.gz
