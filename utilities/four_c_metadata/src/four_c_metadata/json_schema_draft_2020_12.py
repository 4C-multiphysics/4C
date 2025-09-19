# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""JSON schema utils for draft 2020-12.

This mostly follows the naming in https://www.learnjsonschema.com/2020-12/. Only the stuff we actually need is implemented.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, TypeAlias
from four_c_metadata.not_set import NotSet, NotSetAlias, check_if_set


def add_if_set(data: dict, key: str, value: object) -> None:
    """Set value if it set."""
    if check_if_set(value):
        data[key] = value


@dataclass
class Schema:
    schema_type: SchemaTypesAlias | Sequence[SchemaTypesAlias]
    title: NotSetAlias[str] | None = NotSet(str)
    description: NotSetAlias[str] | None = NotSet(str)
    default: NotSetAlias | None = NotSet()

    def to_dict(self) -> dict:
        def convert_type(schema_type, class_name) -> str:
            type_string = str(type(schema_type).__name__)
            type_string = type_string.split(class_name)[0].lower()
            return type_string

        def class_name_to_schema_keyword(object) -> str:
            string = str(type(object).__name__)
            return string[0].lower() + string[1:]

        data: dict = {}
        add_if_set(data, "title", self.title)
        if self.description is not None:
            add_if_set(data, "description", self.description)
        add_if_set(data, "default", self.default)

        if isinstance(self.schema_type, list):
            data["type"] = [convert_type(t, "Type") for t in self.schema_type]
        else:
            data["type"] = convert_type(self.schema_type, "Type")

        types = (
            self.schema_type
            if isinstance(self.schema_type, list)
            else [self.schema_type]
        )
        for t in types:
            for applicator in t.applicators:
                key = class_name_to_schema_keyword(applicator)
                data[key] = applicator.serialize()
            for validator in t.validators:
                key = class_name_to_schema_keyword(validator)
                data[key] = validator.data
            for core in t.core:
                data["$" + convert_type(core, "Core")] = core.data

        return data


# Types
@dataclass
class NullType:
    applicators: Sequence[OneOf] = field(default_factory=list)
    validators: Sequence[Enum | Const] = field(default_factory=list)
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


@dataclass
class BooleanType:
    applicators: Sequence[OneOf] = field(default_factory=list)
    validators: Sequence[Enum | Const] = field(default_factory=list)
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


@dataclass
class ObjectType:
    applicators: Sequence[
        OneOf | Properties | AdditionalProperties | PatternProperties
    ] = field(default_factory=list)
    validators: Sequence[Enum | Const | Required | MinProperties | MaxProperties] = (
        field(default_factory=list)
    )
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


@dataclass
class ArrayType:
    applicators: Sequence[OneOf | Items | PrefixItems] = field(default_factory=list)
    validators: Sequence[Enum | Const | MinItems | MaxItems] = field(
        default_factory=list
    )
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


@dataclass
class NumberType:
    applicators: Sequence[OneOf] = field(default_factory=list)
    validators: Sequence[
        Enum | Const | ExclusiveMinimum | ExclusiveMaximum | Minimum | Maximum
    ] = field(default_factory=list)
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


@dataclass
class IntegerType:
    applicators: Sequence[OneOf] = field(default_factory=list)
    validators: Sequence[
        Enum | Const | ExclusiveMinimum | ExclusiveMaximum | Minimum | Maximum
    ] = field(default_factory=list)
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


@dataclass
class StringType:
    applicators: Sequence[OneOf] = field(default_factory=list)
    validators: Sequence[Enum | Const | Pattern] = field(default_factory=list)
    core: Sequence[IdCore | SchemaCore] = field(default_factory=list)


SchemaTypesAlias: TypeAlias = (
    NullType
    | BooleanType
    | ObjectType
    | ArrayType
    | NumberType
    | IntegerType
    | StringType
)

AnyAlias: TypeAlias = (
    NullType
    | BooleanType
    | ObjectType
    | ArrayType
    | NumberType
    | IntegerType
    | StringType
)


class Types:
    null = NullType
    boolean = BooleanType
    object = ObjectType
    array = ArrayType
    number = NumberType
    integer = IntegerType
    string = StringType


# Validators
@dataclass
class Enum:
    data: Sequence


@dataclass
class Const:
    data: object


@dataclass
class Pattern:
    data: str


@dataclass
class ExclusiveMaximum:
    data: int | float


@dataclass
class ExclusiveMinimum:
    data: int | float


@dataclass
class Maximum:
    data: int | float


@dataclass
class Minimum:
    data: int | float


@dataclass
class MinItems:
    data: int


@dataclass
class MaxItems:
    data: int


@dataclass
class MinProperties:
    data: int


@dataclass
class MaxProperties:
    data: int


@dataclass
class Required:
    data: Sequence[str]


class Validators:
    enum = Enum
    const = Const
    pattern = Pattern
    exclusiveMaximum = ExclusiveMaximum
    exclusiveMinimum = ExclusiveMinimum
    maximum = Maximum
    minimum = Minimum
    minItems = MinItems
    maxItems = MaxItems
    minProperties = MinProperties
    maxProperties = MaxProperties
    required = Required


# Applicators
@dataclass
class OneOf:
    schemas: Sequence[Schema]

    def serialize(self) -> Sequence[dict]:
        return [s.to_dict() for s in self.schemas]


@dataclass
class Properties:
    schemas: dict[str, Schema]

    def serialize(self) -> dict[str, dict]:
        return {name: schema.to_dict() for name, schema in self.schemas.items()}


@dataclass
class AdditionalProperties:
    schemas: Sequence[Schema] | bool

    def serialize(self) -> bool | Sequence[dict[str, dict]]:
        if isinstance(self.schemas, bool):
            return self.schemas
        else:
            return [s.to_dict() for s in self.schemas]


@dataclass
class PatternProperties:
    schemas: dict[str, Schema]

    def serialize(self) -> dict[str, dict]:
        return {name: schema.to_dict() for name, schema in self.schemas.items()}


@dataclass
class Items:
    schema: Schema

    def serialize(self) -> dict[str, dict]:
        return self.schema.to_dict()


@dataclass
class PrefixItems:
    data: Sequence[Schema]

    def serialize(self) -> Sequence[str, dict]:
        return [s.to_dict() for s in self.data]


class Applicators:
    oneOf = OneOf
    properties = Properties
    additionalProperties = AdditionalProperties
    patternProperties = PatternProperties
    items = Items
    prefixItems = PrefixItems


# Core
@dataclass
class IdCore:
    data: str


@dataclass
class SchemaCore:
    data: str


class Core:
    id_core = IdCore
    schema = SchemaCore
