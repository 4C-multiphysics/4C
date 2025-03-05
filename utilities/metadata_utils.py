# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Utilities to handle 4C metadata files."""
from dataclasses import dataclass, field
from pathlib import Path
from ruamel.yaml import YAML


# In order to allow None as defaults
class NotSet:
    def __str__(self):
        return "NOTSET singleton"


MISSING_DESCRIPTION = "No description yet."
FOURC_BASE_TYPES_TO_JSON_SCHEMA_DICT = {
    "double": "number",
    "int": "integer",
    "bool": "boolean",
    "string": "string",
    "path": "string",
}


@dataclass
class Parameter:
    """Base parameter."""

    name: str = None
    type: str = None
    required: bool = False
    description: str = None


@dataclass
class Primitive(Parameter):
    """Primitive parameter."""

    default: int = NotSet()
    noneable: bool = False

    def short_description(self):
        """Create short description."""
        description = ""
        if self.name:
            description += self.name + " "
        if self.noneable:
            description += f"({self.type} or null)"
        else:
            description += f"({self.type})"
        return description


@dataclass
class Vector(Primitive):
    """Vector parameter."""

    size: int = None
    value_type: str = None

    def __post_init__(self):
        # If dict create object
        if isinstance(self.value_type, dict):
            self.value_type = _metadata_object_from_dict(self.value_type)


@dataclass
class Map(Primitive):
    """Map parameter."""

    size: int = None
    value_type: str = None

    def __post_init__(self):
        # If dict create object
        if isinstance(self.value_type, dict):
            self.value_type = _metadata_object_from_dict(self.value_type)


@dataclass
class Enum(Primitive):
    """Enum parameter."""

    choices: list = field(default_factory=list)

    def __post_init__(self):
        for i, choice in enumerate(self.choices):
            if isinstance(choice, dict):
                self.choices[i] = choice["name"]


@dataclass
class Collection(Parameter):
    """Collection parameter."""

    def __post_init__(self):
        self.type = self._type
        for i, spec in enumerate(self.specs):
            if isinstance(spec, dict):
                spec = _metadata_object_from_dict(spec)
                self.specs[i] = spec

        self.specs = sorted(self.specs, key=lambda x: x.name)

    def one_ofs(self, pop=False):
        """Get all one_ofs.

        Args:
            pop (bool, optional): Delete entries from specs. Defaults to False.

        Returns:
            list: one_ofs specs
        """
        ids = []
        specs = []
        for i, k in enumerate(self.specs):
            if isinstance(k, One_Of):
                specs.append(k)
            else:
                ids.append(i)

        if pop:
            self.specs = [self.specs[i] for i in ids]
        return specs

    def all_ofs(self, pop=False):
        """Get all all_ofs.

        Args:
            pop (bool, optional): Delete entries from specs. Defaults to False.

        Returns:
            list: all_ofs specs
        """
        ids = []
        specs = []
        for i, k in enumerate(self.specs):
            if type(k) == All_Of:
                specs.append(k)
            else:
                ids.append(i)

        if pop:
            self.specs = [self.specs[i] for i in ids]
        return specs

    def named_specs(self, pop=False):
        """Get all named specs.

        Specs that are not one_of or all_of

        Args:
            pop (bool, optional): Delete entries from specs. Defaults to False.

        Returns:
            list: named specs
        """
        ids = []
        specs = []
        for i, k in enumerate(self.specs):
            if isinstance(k, (Primitive, Group, List)):
                specs.append(k)
            else:
                ids.append(i)

        if pop:
            self.specs = [self.specs[i] for i in ids]
        return specs

    def split_specs(self, pop=False):
        """Split specs into named specs, one_ofs and all_ofs

        Args:
            pop (bool, optional): Delete entries from specs. Defaults to False.

        Returns:
            tuple of lists: named_specs, one_ofs and all_ofs
        """
        return self.named_specs(pop), self.one_ofs(pop), self.all_ofs(pop)

    def parameter_names(self):
        """Names of the named specs.

        Returns:
            list: names of the named specs.
        """
        return [k.name for k in self.named_specs()]

    def is_empty(self):
        """Check for no entries.

        Returns:
            bool: True for no entries
        """
        return not bool(self.specs)


@dataclass
class One_Of(Collection):
    """One_of collection."""

    specs: list = field(default_factory=list)
    name: str = ""
    _type = "one_of"


@dataclass
class All_Of(Collection):
    """All_of collection."""

    specs: list = field(default_factory=list)
    name: str = ""
    _type = "all_of"

    def __post_init__(self):
        super().__post_init__()
        if len(self.one_ofs()) > 1:
            raise ValueError("More than one One_of is not supported.")


@dataclass
class Group(Collection):
    """Group."""

    specs: list = field(default_factory=list)
    name: str = None
    noneable: bool = False
    defaultable: bool = False
    _type = "group"

    def short_description(self):
        """Create short description."""
        return f"{self.name} ({self.type})"


@dataclass
class List(Parameter):
    """List."""

    size: int = None
    spec: Parameter = None
    noneable: bool = False
    _type = "list"

    def __post_init__(self):
        self.type = self._type
        if isinstance(self.spec, dict):
            self.spec = _metadata_object_from_dict(self.spec)

    def short_description(self):
        """Create short description."""
        return f"{self.name} ({self.type})"


# Named objects
NAMED_TYPES = (Primitive, List, Group)


def _metadata_object_from_dict(metadata_dict):
    """Create metadata object from dict.

    Args:
        metadata_dict (dict): Dictionary with the metadata

    Returns:
        Parameter: Metadata object
    """
    match metadata_dict["type"]:
        # Primitives
        case _ if metadata_dict["type"] in FOURC_BASE_TYPES_TO_JSON_SCHEMA_DICT:
            cls = Primitive
        case "enum":
            cls = Enum
        case "vector":
            cls = Vector
        case "map":
            cls = Map
        # Collections
        case "group":
            cls = Group
        case "list":
            cls = List
        case "all_of":
            cls = All_Of
        case "one_of":
            cls = One_Of
        case _:
            raise TypeError(
                f"Unknown type '{metadata_dict['type']}' for {metadata_dict}."
            )

    return cls(**metadata_dict)


def metadata_object_from_file(metadata_4C_path):
    """Load metadata from file and create metadata object.

    Args:
        metadata_4C_path (str, pathlib.Path): Path to metadata file.

    Returns:
        tuple: All_of and description section name and 4C code metadata
    """
    metadata = YAML().load(Path(metadata_4C_path).read_text())

    # Title section
    description_section_name = metadata["metadata"]["description_section_name"]
    metadata_description = f"Schema for 4C\nCommit hash: {metadata['metadata']['commit_hash']}\nVersion: {metadata['metadata']['version']}"

    # Legacy section
    sections = metadata["sections"]
    legacy_sections = [
        {
            "name": name,
            "description": name + f" [legacy section] ",
            "type": "vector",
            "value_type": {"type": "string"},
        }
        for name in metadata["legacy_string_sections"]
    ]

    # Combine sections with legacy sections
    all_of = All_Of(
        description=metadata_description,
        specs=sections + legacy_sections,
        name="Sections",
    )
    return all_of, description_section_name, metadata["metadata"]
