#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install Trilinos from the develop branch.
# Call with
# ./install.sh /path/to/install/dir
# The Trilinos commit ref can be overridden with the environment variable TRILINOS_VERSION.

# Exit the script at the first failure
set -e

# Location of this script
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Checkout develop branch if no other version is specified
export TRILINOS_VERSION=${TRILINOS_VERSION:-develop}

# Enable Trilinos debug assertions by default
export TRILINOS_ENABLE_DEBUG=${TRILINOS_ENABLE_DEBUG:-ON}

exec "$SCRIPT_DIR/../../current/trilinos/install.sh" "$1"
