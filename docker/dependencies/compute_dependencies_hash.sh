#!/bin/bash
# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Exit the script at the first failure
set -e

# Calculate a hash from our dependencies. Call the script from the project root directory.
# - Find all files and exclude hidden files (like .DS_Store on macOS). Hidden files should not be part of the dependencies.
# - Calculate the SHA1 of every file. The SHA1 is good enough as we don't use it for cryptographic security. No need for SHA256
# - Calculate the final SHA1 from the filenames and their respective SHA1. We include the filename and not only the content to track changes to the dependency structure.
# - Use the first 8 characters of the SHA1 as hash. There is only a low collision probability with 8 characters as the hash is rarely updated.
find dependencies/current dependencies/oldest_supported dependencies/testing docker/dependencies docker/oldest_supported -type f -not -name '.*' -exec sha1sum {} \; | sort | sha1sum | cut -c -8
