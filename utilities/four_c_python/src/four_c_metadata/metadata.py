# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Metadata objects from the 4C input metadata."""

from __future__ import annotations

import abc
import re
from collections.abc import Generator, Sequence
from typing import Callable, Literal, Protocol, TypeAlias, TypeVar

from four_c_metadata.not_set import NotSet, check_if_set, NotSetAlias

PRIMITIVES_PYTHON_TYPES = float | bool | int | str

ValidatorAlias: TypeAlias = Callable[[object], bool] | None

NotSetString: NotSetAlias[str] = NotSet(str)

T = TypeVar("T", int, float)
G = TypeVar("G", contravariant=True)
# NotSet
NotSetAlias: TypeAlias = NotSet | T


class InputSpec:
    """Input spec base class."""

    def __init__(
        self,
        spec_type: str,
        name: NotSetAlias[str] = NotSetString,
        description: NotSetAlias[str] = NotSetString,
        required: bool = False,
        noneable: bool = False,
        validator: ValidatorAlias = None,
    ) -> None:
        """Initialise InputSpec.

        Args:
            spec_type (str): Type of input spec
            name: Name
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            validator: Validator callable
        """
        self.spec_type = spec_type
        self.name = name
        self.description = description
        self.required = required
        self.noneable = noneable
        self.validator = validator

    @classmethod
    @abc.abstractmethod
    def from_4C_metadata(cls, data_dict: dict) -> InputSpec:
        """Create InputSpec from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            InputSpec
        """


class All_Of:
    def __init__(
        self,
        specs: Sequence[Spec],
        description: NotSetAlias[str] = NotSetString,
    ) -> None:
        """Initialise All_Of.

        Args:
            specs: Specs of the All_Of
            description: Description
        """
        self.specs: list[InputSpec | One_Of] = All_Of.condense(specs)
        self.description = description

    @staticmethod
    def condense(specs: Sequence[Spec]) -> list[InputSpec | One_Of]:
        """Condense All_Of.

        All_Ofs of All_Ofs are joined, aka flattened or condensed.

        Args:
            specs: Specs to condense

        Returns:
            Condensed specs
        """
        new_specs: list[InputSpec | One_Of] = []
        one_ofs: list[One_Of] = []

        for spec in specs:
            match spec:
                case InputSpec():
                    new_specs.append(spec)
                case All_Of():
                    # Check if it is an One_Of
                    if spec.is_one_of():
                        one_ofs.append(spec.specs[0])  # type: ignore
                    else:
                        new_specs.extend(All_Of.condense(spec.specs))
                case One_Of():
                    one_ofs.append(spec)
                case _:
                    raise TypeError(f"Invalid spec {spec}")

        if len(one_ofs) > 1:
            oo = "\n".join([str(oo) for oo in one_ofs])
            raise ValueError(f"More than one_of is not allowed {oo}")

        if one_ofs:
            new_specs = [one_ofs[0].add_specs(new_specs)]

        return new_specs

    def add_specs(self, input_specs: Sequence[Spec]) -> All_Of:
        """Add specs to the All_Of.

        Returns:
            All_Of object.
        """
        self.specs = All_Of.condense(self.specs + list(input_specs))
        return self

    def is_one_of(self) -> bool:
        """Check if All_Of only hold a One_Of.

        In essence this means that the All_Of is simply a One_Of.

        Returns:
            True if is a One_Of in disguise
        """
        return len(self.specs) == 1 and isinstance(self.specs[0], One_Of)

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> All_Of:
        """Create All_Of from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            All_Of
        """
        data_dict.pop("type", None)
        specs = data_dict.pop("specs", [])
        specs = [metadata_from_dict(spec) for spec in specs]
        return cls(specs=specs, **data_dict)

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return "All_of(" + ", ".join([str(spec) for spec in self.specs]) + ")"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return "All_of(" + ", ".join([str(spec) for spec in self.specs]) + ")"

    def __len__(self) -> int:
        """Len of the All_Of.

        Returns:
            number of specs
        """
        return len(self.specs)

    def __iter__(self) -> Generator[One_Of | InputSpec]:
        """Iterate All_Of.

        Yields:
            All_Of specs
        """
        yield from self.specs


class One_Of:
    def __init__(
        self,
        specs: Sequence[Spec],
        description: NotSetAlias[str] = NotSetString,
    ) -> None:
        """Initialise One_Of.

        Args:
            specs: Specs of the One_Of, i.e., each distinct branch
            description: Description
        """
        self.specs: list[All_Of] = One_Of.condense(specs)
        self.description = description

    @staticmethod
    def condense(specs: Sequence[Spec]) -> list[All_Of]:
        """Condense One_Of.

        One_Ofs of One_Ofs are combined to a single One_Of.

        Args:
            specs: Specs of the One_Of

        Returns:
            condesed specs
        """
        new_specs: list[All_Of] = []

        for spec in specs:
            match spec:
                case InputSpec():
                    new_specs.append(All_Of([spec]))
                case All_Of():
                    if spec.is_one_of():
                        # One_of in One_of
                        new_specs.extend(One_Of.condense(spec.specs[0].specs))  # type: ignore
                    else:
                        new_specs.append(spec)
                case One_Of():
                    # One_of in one_of
                    new_specs.extend(One_Of.condense(spec.specs))
                case _:
                    raise TypeError(f"Not supported type {type(spec)} for {spec}")

        return new_specs

    def add_specs(self, input_specs: Sequence[Spec]) -> One_Of:
        """Add specs to the One_Of.

        Returns:
            One_Of object.
        """
        for i in range(len(self.specs)):
            self.specs[i].add_specs(input_specs)
        self.specs = One_Of.condense(self.specs)
        return self

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> One_Of:
        """Create One_Of from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            One_Of
        """
        data_dict.pop("type", None)
        specs = [metadata_from_dict(spec) for spec in data_dict.pop("specs")]
        return cls(specs=specs, **data_dict)

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return "One_of(" + ", ".join([str(spec) for spec in self.specs]) + ")"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return "One_of(" + ", ".join([str(spec) for spec in self.specs]) + ")"

    def __len__(self) -> int:
        """Len of the One_Of.

        Returns:
            number of One_Of branches
        """
        return len(self.specs)

    def __iter__(self) -> Generator[All_Of]:
        """Iterate One_of.

        Yields:
            One_Of branches
        """
        yield from self.specs


class Primitive(InputSpec):
    PRIMITIVE_TYPES: list[str] = ["double", "bool", "int", "string", "path"]
    PRIMITIVE_TYPES_TO_PYTHON: dict[str, type] = {
        "double": float,
        "bool": bool,
        "int": int,
        "string": str,
        "path": str,
    }

    def __init__(
        self,
        spec_type: Literal["double", "int", "bool", "string", "path"],
        name: NotSetAlias[
            str
        ] = NotSetString,  # Name is only optional with vectors, maps, tuple
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        validator: ValidatorAlias = None,
        default: NotSetAlias[PRIMITIVES_PYTHON_TYPES] = NotSet("default"),
    ) -> None:
        """Initialise primitive.

        Args:
            spec_type: Type of primitive
            name: Name
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            validator: Validator callable
            default: Default value of the spec
        """
        if spec_type not in Primitive.PRIMITIVE_TYPES:
            raise TypeError(
                f"Spec type {spec_type} is not supported. Supported ones are: {Primitive.PRIMITIVE_TYPES}"
            )
        super().__init__(spec_type, name, description, required, noneable, validator)
        if check_if_set(default) and default is not None and False:
            if not isinstance(default, Primitive.PRIMITIVE_TYPES_TO_PYTHON[spec_type]):
                raise TypeError(
                    f"Default {default} for the parameter of type '{spec_type}' is not Python type {Primitive.PRIMITIVE_TYPES_TO_PYTHON[spec_type]}"
                )
        self.default = default

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Primitive:
        """Create primitive from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            primitive
        """
        spec_type = data_dict.pop("type")
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(spec_type=spec_type, **data_dict)

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name}"


class Enum(InputSpec):
    def __init__(
        self,
        choices: Sequence[str],
        name: NotSetAlias[str] = NotSetString,
        description: NotSetAlias[str] = NotSetString,
        choices_description: Sequence[str] = [],
        required: bool = True,
        noneable: bool = False,
        validator: ValidatorAlias = None,
        default: NotSetAlias[str] = NotSet("default"),
    ) -> None:
        """Initialise enum.

        Args:
            choices: Enum choices
            name: Name
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            validator: Validator callable
            default: Default value
        """
        super().__init__("enum", name, description, required, noneable, validator)
        if check_if_set(default):
            if default not in choices:
                raise ValueError(
                    f"Default choice {default} is not in the valid enum choices {choices} of {name}"
                )
        self.choices = choices
        self.default = default
        if len(choices_description) == 0:
            choices_description = [None] * len(choices)
        elif len(choices_description) != len(choices):
            raise ValueError(
                f"Number of descriptions ({len(choices_description)}) does not match the number of number of choices ({len(choices)}) in {name}"
            )
        self.choices_description = choices_description

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Enum:
        """Create enum from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            enum
        """
        data_dict.pop("type", None)
        choices = [c["name"] for c in data_dict.pop("choices")]
        # TODO add description from choices
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(choices=choices, **data_dict)

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name}"


class Vector(InputSpec):
    def __init__(
        self,
        value_type: NATIVE_CPP_ALIAS,
        size: int | None = None,
        name: NotSetAlias[str] = NotSetString,
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        validator: ValidatorAlias = None,
        default: NotSetAlias[Sequence[NATIVE_CPP_ALIAS]] = NotSet("default"),
    ) -> None:
        """Initialise vector.

        Args:
            value_type: Type of vector elements
            size: Vector dimension
            name: Name
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            validator: Validator callable
            default: Default value
        """
        super().__init__("vector", name, description, required, noneable, validator)
        if not isinstance(value_type, NATIVE_CPP_TYPES):
            raise TypeError(
                f"Value type {value_type} has to be of type: {NATIVE_CPP_TYPES}"
            )
        self.value_type = value_type
        self.size = size
        self.default = default

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Vector:
        """Create vector from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            vector
        """
        data_dict.pop("type", None)
        value_type: NATIVE_CPP_ALIAS = metadata_from_dict(data_dict.pop("value_type"))  # type: ignore
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(value_type=value_type, **data_dict)

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name}"

    def __iter__(self) -> Generator[NATIVE_CPP_ALIAS]:
        """Iterate vector.

        Yields:
            value type of the vector
        """
        yield self.value_type


class Map(InputSpec):
    def __init__(
        self,
        value_type: NATIVE_CPP_ALIAS,
        size: int | None = None,
        name: NotSetAlias[str] = NotSetString,
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        validator: ValidatorAlias = None,
        default: NotSetAlias[dict[str, NATIVE_CPP_ALIAS]] = NotSet("default"),
    ) -> None:
        """Initialise map.

        Args:
            value_type: Type of map elements
            size: Number of entries in the map
            name: Name
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            validator: Validator callable
            default: Default value
        """
        if not isinstance(value_type, NATIVE_CPP_TYPES):
            raise TypeError(
                f"Value type {value_type} has to be of type: {NATIVE_CPP_TYPES}"
            )
        super().__init__("map", name, description, required, noneable, validator)
        self.value_type: NATIVE_CPP_ALIAS = value_type
        self.size = size
        self.default = default

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Map:
        """Create map from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            map
        """
        data_dict.pop("type", None)
        value_type: NATIVE_CPP_ALIAS = metadata_from_dict(data_dict.pop("value_type"))  # type: ignore
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(value_type=value_type, **data_dict)

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name}"

    def __iter__(self) -> Generator[NATIVE_CPP_ALIAS]:
        """Iterate map.

        Yields:
            value type of the map
        """
        yield self.value_type


class Tuple(InputSpec):
    def __init__(
        self,
        value_types: Sequence[NATIVE_CPP_ALIAS],
        size: int,
        name: NotSetAlias[str] = NotSetString,
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        validator: ValidatorAlias = None,
        default: NotSetAlias[Sequence[NATIVE_CPP_ALIAS]] = NotSet("default"),
    ) -> None:
        """Initialise tuple.

        Args:
            value_types: Types of the tuple element
            size: Number of tuple entries
            name: Name
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            validator: Validator callable
            default: Default value
        """
        super().__init__("tuple", name, description, required, noneable, validator)
        for i, value_type in enumerate(value_types):
            if not isinstance(value_type, NATIVE_CPP_TYPES):
                raise TypeError(
                    f"Value type {value_type} (parameter number {i + 1} in {value_types}) has to be of type: {NATIVE_CPP_TYPES}"
                )
        self.value_types = value_types
        self.size = size
        if self.size != len(self.value_types):
            raise ValueError(
                f"Tuple size {self.size} does not match the number of value types "
                f"{len(self.value_types)}"
            )
        self.default = default

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Tuple:
        """Create tuple from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            tuple
        """
        data_dict.pop("type", None)
        value_types = [
            metadata_from_dict(value_type)
            for value_type in data_dict.pop("value_types")
        ]
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(value_types=value_types, **data_dict)  # type: ignore

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name}"

    def __iter__(self) -> Generator[NATIVE_CPP_ALIAS]:
        """Iterate tuple.

        Yields:
            value types of the tuple
        """
        yield from self.value_types


class Selection(InputSpec):
    def __init__(
        self,
        name: NotSetAlias[str],
        choices: dict[str, Spec],
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        defaultable: bool = False,
        validator: ValidatorAlias = None,
    ) -> None:
        """Initialise selection.

        Args:
            name: Name
            choices: Choices of the selection
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            defaultable: Can have a default
            validator: Validator callable
        """
        super().__init__("selection", name, description, required, noneable, validator)
        self.choices: dict[str, All_Of] = {k: All_Of([v]) for k, v in choices.items()}
        self.defautable = defaultable

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Selection:
        """Create selection from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            selection
        """
        data_dict.pop("type", None)

        choices = {}
        for choice in data_dict.pop("choices"):
            spec = choice.pop("spec")
            choices[choice.pop("name")] = metadata_from_dict(spec)
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(choices=choices, **data_dict)

    def __iter__(self) -> Generator[Spec]:
        """Iterate selection.

        Yields:
            choices of the selection
        """
        yield from self.choices.values()

    def __len__(self) -> int:
        """Len of the selection.

        Returns:
            number of choices
        """
        return len(self.choices)


class Group(InputSpec):
    def __init__(
        self,
        name: NotSetAlias[str],
        spec: Spec,
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        defaultable: bool = False,
        validator: ValidatorAlias = None,
    ) -> None:
        """Initialise group.

        Args:
            name: Name
            spec: Spec of the group
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            defaultable: Can have a default
            validator: Validator callable
        """
        super().__init__("group", name, description, required, noneable, validator)
        self.spec: All_Of = All_Of([spec])
        self.defautable = defaultable

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> Group:
        """Create group from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            group
        """
        data_dict.pop("type", None)
        specs = [metadata_from_dict(spec) for spec in data_dict.pop("specs", [])]
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(spec=All_Of(specs), **data_dict)

    def __iter__(self) -> Generator[All_Of]:
        """Iterate group.

        Yields:
            group spec
        """
        yield self.spec

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name} Group({self.spec})"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name} Group({self.spec})"


class List(InputSpec):
    def __init__(
        self,
        name: NotSetAlias[str],
        spec: Spec,
        size: int | None = None,
        description: NotSetAlias[str] = NotSetString,
        required: bool = True,
        noneable: bool = False,
        defaultable: bool = False,
        validator: ValidatorAlias = None,
    ) -> None:
        """Initialise list.

        Args:
            name: Name
            spec: Spec of the list entries
            size: Number of list entries
            description: Description
            required: True if parameter is required
            noneable: True if parameter can be None
            defaultable: Can have a default
            validator: Validator callable
        """
        super().__init__("list", name, description, required, noneable, validator)
        self.spec: All_Of = All_Of([spec])
        self.size = size
        self.defautable = defaultable

    @classmethod
    def from_4C_metadata(cls, data_dict: dict) -> List:
        """Create list from 4C metadata file.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            list
        """
        data_dict.pop("type", None)
        spec = metadata_from_dict(data_dict.pop("spec"))
        if "validator" in data_dict:
            data_dict["validator"] = validator_from_dict(data_dict["validator"])
        return cls(spec=All_Of([spec]), **data_dict)

    def __iter__(self) -> Generator[All_Of]:
        """Iterate list.

        Yields:
            spec
        """
        yield self.spec

    def __str__(self) -> str:  # pragma: no cover
        """String method."""
        return f"{self.name} List({self.spec})"

    def __repr__(self) -> str:  # pragma: no cover
        """Representation method."""
        return f"{self.name} List({self.spec})"


Spec: TypeAlias = InputSpec | All_Of | One_Of
NATIVE_CPP_ALIAS: TypeAlias = Primitive | Enum | Vector | Map | Tuple
NATIVE_CPP_TYPES = (Primitive, Enum, Vector, Map, Tuple)

_METADATA_TYPE_TO_CLASS: dict[str, type[Spec]] = {
    i.__name__.lower(): i
    for i in [Enum, Vector, Map, Tuple, Selection, Group, List, All_Of, One_Of]
} | {i: Primitive for i in Primitive.PRIMITIVE_TYPES}


def metadata_from_dict(data_dict: dict) -> Spec:
    """Create metadata object from 4C metadata dict.

    Args:
        data_dict: Data from 4C metadata file

    Returns:
        metadata object
    """
    entry_type: str = data_dict.get("type")  # type: ignore

    metadata_class: type[Spec] | None = _METADATA_TYPE_TO_CLASS.get(entry_type, None)

    if metadata_class is not None:
        return metadata_class.from_4C_metadata(data_dict)

    raise TypeError(
        f"Unknown type {entry_type} for {data_dict}, known ones are {list(_METADATA_TYPE_TO_CLASS.keys())}"
    )


class Validator(Protocol[G]):
    """Validator protocol."""

    def _validate(self, entry: G) -> bool:
        """Validate object.

        Args:
            entry: Object to validate

        Returns:
            True if valid
        """
        pass

    def __call__(self, entry: G) -> bool:
        """Call method."""
        return self._validate(entry)


class RangeValidator(Validator[T]):
    def __init__(
        self,
        minimum: T,
        maximum: T,
        minimum_exclusive: bool = False,
        maximum_exclusive: bool = False,
    ) -> None:
        """Range validator.

        Args:
            entry: Entry value
            minimum: Minimum value
            maximum: Maximum value
            minimum_exclusive: Minimum is exclusive
            maximum_exclusive: Maximum is exclusive
        """
        self.minimum: T = minimum
        self.maximum: T = maximum
        self.minimum_exclusive = minimum_exclusive
        self.maximum_exclusive = maximum_exclusive

    def _validate(self, entry: T) -> bool:
        """Validate if float or in is in range.

        Args:
            entry: Entry value
            minimum: Minimum value
            maximum: Maximum value
            minimum_exclusive: Minimum is exclusive
            maximum_exclusive: Maximum is exclusive

        Returns:
            True if in range
        """
        if self.minimum_exclusive:
            if entry <= self.minimum:
                return False
        else:
            if entry < self.minimum:
                return False

        if self.maximum_exclusive:
            if entry >= self.maximum:
                return False
        else:
            if entry > self.maximum:
                return False

        return True

    @classmethod
    def from_4C_metadata_dict(cls, data_dict: dict) -> RangeValidator:
        """Create validator from 4C metadata dict.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            range validator
        """
        return cls(**data_dict)


class AllEmementsValidator(Validator[Sequence]):
    def __init__(self, element_validator: Callable[[Sequence], bool]) -> None:
        """All elements validator.

        Validates each entry of a sequence with the `element_validator`.

        Args:
            element_validator: Validator for each entry
        """
        self.element_validator = element_validator

    def _validate(self, entry: Sequence) -> bool:
        """Validate sequence.

        Args:
            entry: Sequence to validate

        Returns:
            True if all entries are valid
        """
        return all([self.element_validator(e) for e in entry])

    @classmethod
    def from_4C_metadata_dict(cls, data_dict: dict) -> AllEmementsValidator:
        """Create validator from 4C metadata dict.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            all elements validator
        """
        element_validator = validator_from_dict(data_dict)
        return cls(element_validator)


class PatternValidator(Validator):
    def __init__(self, pattern: str):
        """Regex pattern validator.

        Args:
            pattern: Pattern to match
        """
        self.pattern = re.compile(pattern)

    def _validate(self, entry: str) -> bool:
        """Validate pattern.

        Args:
            entry: Entry to validate

        Returns:
            True if pattern matches
        """
        return bool(self.pattern.match(entry))

    @classmethod
    def from_4C_metadata_dict(cls, data_dict: dict) -> PatternValidator:
        """Create validator from 4C metadata dict.

        Args:
            data_dict: Data from 4C metadata file

        Returns:
            validator
        """
        return cls(**data_dict)


def validator_from_dict(data_dict: dict) -> Validator:
    """Create validator from dict.

    Args:
        data_dict: Data from 4C metadata file

    Returns:
        validator
    """
    if len(data_dict) == 0:
        raise ValueError(f"Can not construct value error from {data_dict}.")
    validator_name = list(data_dict.keys())[0]
    validator_data = data_dict[validator_name]
    match validator_name:
        case "range":
            return RangeValidator.from_4C_metadata_dict(validator_data)
        case "all_elements":
            return AllEmementsValidator.from_4C_metadata_dict(validator_data)
        case "pattern":
            return PatternValidator.from_4C_metadata_dict(validator_data)
        case _:
            raise KeyError(f"Unknown validator {data_dict}")
