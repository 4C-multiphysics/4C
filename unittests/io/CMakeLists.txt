# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# These tests claim they test IO (part of core) but they use other parts as well,
# notably concrete materials. Therefore, the tests are attached to the mat module,
# until this problem is resolved.
four_c_auto_define_tests(MODULE mat)
