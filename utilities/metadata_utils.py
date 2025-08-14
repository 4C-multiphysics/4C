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
from typing import ClassVar


# In order to allow None as defaults
class NotSet:
    def __str__(self):
        return "NOTSET singleton"

    def __repr__(self):
        return "NOTSET singleton"


NOTSET = NotSet()

MISSING_DESCRIPTION = "No description yet."
FOURC_BASE_TYPES_TO_JSON_SCHEMA_DICT = {
    "double": "number",
    "int": "integer",
    "bool": "boolean",
    "string": "string",
    "path": "string",
}


def _check_for_multiple_one_ofs(metadata_object):
    """Check if object contains multiple one_ofs.

    Args:
        metadata_object (Group, All_of): object to check.
    """

    hint_text = """
    In general, this error indicates that the input construct is suboptimal. Transform the
    different One_ofs into Groups or Selections, e.g., in pseudo code for parameters a,b,c,d:

    From this:  All_of
                  {
                    One_of{a,b},
                    One_of{c,d}
                  }

    To this:    All_of
                  {
                    Group{"good_group_name", One_of{a,b}},
                    Group{"other_good_group_name", One_of{c,d}}
                  }
    """

    if len(metadata_object.one_ofs()) > 1:
        raise NotImplementedError(
            f"More than one One_of within an {type(metadata_object)} is not supported.\n"
            f"Here the object you provided: {metadata_object}\n\n{hint_text}"
        )


@dataclass
class Parameter:
    """Base parameter."""

    name: str = None
    type: str = None
    required: bool = False
    description: str = NOTSET


@dataclass
class Primitive(Parameter):
    """Primitive parameter."""

    default: int = NOTSET
    noneable: bool = False
    constant: object = NOTSET  # The primitive can only take this value
    validator: object = None

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

    def __post_init__(self):
        if self.validator is not None and not isinstance(self.validator, Validator):
            self.validator = validator_from_dict(self.validator)
            self.validator.check_type(self.type)


@dataclass
class Vector(Primitive):
    """Vector parameter."""

    size: int = None
    value_type: str = None

    def __post_init__(self):
        # If dict create object
        if isinstance(self.value_type, dict):
            self.value_type = _metadata_object_from_dict(self.value_type)
        if self.validator is not None and not isinstance(self.validator, Validator):
            self.validator = validator_from_dict(self.validator)


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
        super().__post_init__()
        combined_description = ""
        for i, choice in enumerate(self.choices):
            if isinstance(choice, dict):
                self.choices[i] = choice["name"]
                if d := choice.get("description", None):
                    combined_description += f"{choice['name']}: {d}\n"
        if self.description is not NOTSET:
            self.description += "\n" + combined_description
        elif combined_description != "":
            self.description = combined_description


@dataclass
class Collection(Parameter):
    """Collection parameter."""

    def __post_init__(self):
        self.type = type(self).__name__.lower()
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
            if isinstance(k, NAMED_TYPES):
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
        if not self.specs:
            return True

        if len(self.specs) == 1:
            if isinstance(ao := self.specs[0], All_Of):
                return ao.is_empty()

        return False


@dataclass
class One_Of(Collection):
    """One_of collection."""

    specs: list = field(default_factory=list)
    name: str = ""


@dataclass
class All_Of(Collection):
    """All_of collection."""

    specs: list = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        super().__post_init__()
        self._condense_all_ofs()
        _check_for_multiple_one_ofs(self)

    def _condense_all_ofs(self):
        """All_of in All_of can be condensed into a single All_of."""
        if self.all_ofs():
            all_ofs = self.all_ofs(pop=True)
            for ao in all_ofs:
                ao._condense_all_ofs()
                self.specs.extend(ao.specs)


@dataclass
class Group(Collection):
    """Group."""

    specs: list = field(default_factory=list)
    name: str = None
    noneable: bool = False

    def short_description(self):
        """Create short description."""
        return f"{self.name} ({self.type})"

    def __post_init__(self):
        super().__post_init__()
        _check_for_multiple_one_ofs(self)


@dataclass
class Choice(Parameter):
    spec: object = None
    type = "choice"


@dataclass
class Selection(Collection):
    """Selection."""

    choices: list = field(default_factory=list)
    name: str = None
    noneable: bool = False
    required: bool = False

    def __post_init__(self):
        self.type = "selection"
        for i, choice in enumerate(self.choices):
            if isinstance(choice, dict):
                spec = _metadata_object_from_dict(choice["spec"])

                self.choices[i] = Choice(
                    name=choice["name"],
                    description=f"Selector type {choice['name']} for {self.name} (string)",
                    spec=spec,
                )

        self.choices = sorted(self.choices, key=lambda x: x.name)

    def short_description(self):
        """Create short description."""
        return f"{self.name} ({self.type})"


@dataclass
class List(Parameter):
    """List."""

    size: int = None
    spec: Parameter = None
    noneable: bool = False

    def __post_init__(self):
        self.type = "list"
        if isinstance(self.spec, dict):
            self.spec = _metadata_object_from_dict(self.spec)

    def short_description(self):
        """Create short description."""
        return f"{self.name} ({self.type})"


# Named objects
NAMED_TYPES = (Primitive, List, Group, Selection)


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
        case "selection":
            cls = Selection
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

    metadata_4C_path = Path(metadata_4C_path)
    if not metadata_4C_path.exists():
        raise FileNotFoundError(
            f"Metadata file {metadata_4C_path.resolve()} not found."
        )

    try:
        metadata = YAML().load(metadata_4C_path.read_text(encoding="utf-8"))

        # Title section
        description_section_name = metadata["metadata"]["description_section_name"]
        metadata_description = f"Schema for 4C\nCommit hash: {metadata['metadata']['commit_hash']}\nVersion: {metadata['metadata']['version']}"

        # Legacy section
        sections = All_Of(**metadata["sections"])
        legacy_sections = All_Of(
            specs=[
                {
                    "name": name,
                    "description": name + " [legacy section] ",
                    "type": "vector",
                    "value_type": {"type": "string"},
                }
                for name in metadata["legacy_string_sections"]
            ]
        )

        sections.specs.extend(legacy_sections.specs)
        sections.description = metadata_description
        sections.name = "Sections"

    except Exception as exception:
        raise ValueError(
            "Could not read the 4C metadata file.\nThis generally indicates that input parameters"
            " were added or modified in a non-conforming way."
        ) from exception

    return sections, description_section_name, metadata["metadata"]


@dataclass
class Validator:
    allowed_types: ClassVar[tuple]

    def check_type(self, parameter_type):
        if parameter_type not in self.allowed_types:
            raise ValueError(
                f"Can not use {self} for {parameter_type}. Only {self.allowed_types} can be validated with {type(self)}"
            )


@dataclass
class RangeValidator(Validator):
    allowed_types: ClassVar[tuple] = ("int", "double")
    minimum: float
    maximum: float
    minimum_exclusive: bool
    maximum_exclusive: bool


@dataclass
class AllElementsValidator(Validator):
    inner_validator: object


def validator_from_dict(validator_dict):
    if len(validator_dict) > 1:
        raise ValueError(
            f"Currently only a single validator function is allowed, you provided {validator_dict}"
        )

    validator_type, validator_settings = list(validator_dict.items())[0]

    match validator_type:
        case "range":
            return RangeValidator(**validator_settings)
        case "all_elements":
            return AllElementsValidator(validator_from_dict(validator_settings))
        case _:
            raise ValueError("Validator '{}' not known.".format(validator_type))
