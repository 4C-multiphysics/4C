#!/bin/bash

# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Install the virtual Python environment necessary for code development in 4C.
# Call the script from the root directory of the repository:
#     ./utilities/set_up_dev_env.sh
# Optionally, you can specify the path to the python executable (>=3.8):
#     ./utilities/set_up_dev_env.sh /path/to/python

# Exit the script at the first failure
set -e

if [ ! -f "./utilities/set_up_dev_env.sh" ]; then
    echo "Please run this script from the root directory of the repository."
    exit 1
fi

# Path to the python virtual environment.
PYTHON_VENV="`dirname "$0"`/python-venv"

# If the virtual environment already exists, delete it.
if [ -d "$PYTHON_VENV" ]; then rm -Rf $PYTHON_VENV; fi

# Path to python
PYTHON_PATH=${1:-python3}

if ! $PYTHON_PATH -c "import sys; exit(sys.version_info < (3, 9))"; then
    echo "Provided Python version ${PYTHON_PATH} does not meet the minimum requirement (>=3.9)."
    echo "Please provide a compatible Python executable as an argument to this script."
    exit 1
fi

# Setup the virtual environment and source it.
$PYTHON_PATH -m venv "${PYTHON_VENV}"
source "${PYTHON_VENV}"/bin/activate

# Install all the modules defined in requirements.txt.
pip install --upgrade pip
pip install wheel
pip install -r utilities/requirements.txt

# Additionally store the hash of the ingredients for the virtual environment.
./utilities/code_checks/check_venv --update

# Install the pre-commit hooks.
pre-commit install

# Copy the commit-msg hook to the .git/hooks directory.
cp utilities/code_checks/commit-msg .git/hooks/commit-msg
