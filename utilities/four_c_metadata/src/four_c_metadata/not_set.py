# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from typing import TypeAlias, TypeVar


class NotSet:
    """Not set object."""

    def __init__(self, expected: object = object) -> None:
        """Not set object.

        Args:
            expected: Expected object to to display
        """
        self.expected = expected

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"NotSet({self.expected})"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"NotSet({self.expected})"


NOT_SET = NotSet()


def check_if_set(obj: object) -> bool:
    """Check if object or is NotSet.

    Args:
        obj: Object to check

    Returns:
        True if object is set
    """
    # Check if object is not of type _NotSet, i.e. it has a value
    return not isinstance(obj, NotSet)


T = TypeVar("T")
NotSetAlias: TypeAlias = NotSet | T
