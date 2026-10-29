#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install the oldest supported Trilinos version.
# Call with
# ./install.sh /path/to/install/dir
# The Trilinos commit ref can be overridden with the environment variable TRILINOS_VERSION.

# Exit the script at the first failure
set -e

# Location of this script
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# git sha from Trilinos repository (oldest supported version):
export TRILINOS_VERSION=${TRILINOS_VERSION:-"f4d642715185dca1b94c91f434a2cf6db9f82014"}

# Enable Trilinos debug assertions by default for the compatibility tests
export TRILINOS_ENABLE_DEBUG=${TRILINOS_ENABLE_DEBUG:-ON}

exec "$SCRIPT_DIR/../../current/trilinos/install.sh" "$1"
