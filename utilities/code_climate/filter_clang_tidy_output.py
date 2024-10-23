# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import re
import sys

begin_warning_regex = re.compile(r"^.* warning: .*\[.*\]$")
begin_note_regex = re.compile(r"^\/\S+\:\d+\:\d+\: note: .*$")
note_regex = re.compile(r"^note: .*$")
message_block_regex = re.compile(r"^ *\d* \|.*$")
number_of_warnings = re.compile(r"^\d+ warnings generated.$")
supressed_warnings_hint = re.compile(r"^Suppressed \d+ warnings")
display_warning_hint = re.compile(r"^Use -header-filter=\.\* to display errors")

clang_invocation = re.compile(r"^\/\S+\/clang-tidy ")

color_output_regex = re.compile(r"\033\[[0-9;]+m")

is_warning_or_note_active = False
for line in sys.stdin:
    line_removed_coloroutput = color_output_regex.sub("", line)

    # skip all warning message blocks
    if is_warning_or_note_active and message_block_regex.match(
        line_removed_coloroutput
    ):
        continue

    is_warning_or_note_active = False

    # skip warnings
    if begin_warning_regex.match(line_removed_coloroutput) or begin_note_regex.match(
        line_removed_coloroutput
    ):
        is_warning_or_note_active = True
        continue

    # ignore pure notes
    if note_regex.match(line_removed_coloroutput):
        continue

    # remove clang invokation message
    if clang_invocation.match(line_removed_coloroutput):
        continue

    # remove suppressed warnings
    if (
        number_of_warnings.match(line_removed_coloroutput)
        or supressed_warnings_hint.match(line_removed_coloroutput)
        or display_warning_hint.match(line_removed_coloroutput)
    ):
        continue

    # write all other lines
    sys.stdout.write(line)
