# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Create JSON schema for YAML validation."""

from collections.abc import Sequence
import json
import pathlib

from four_c_metadata.not_set import NotSet, check_if_set, NotSetAlias
from four_c_metadata.utils import load_yaml

from four_c_metadata.metadata import (
    One_Of,
    Enum,
    Group,
    List,
    Map,
    All_Of,
    Primitive,
    RangeValidator,
    PatternValidator,
    AllEmementsValidator,
    Selection,
    Tuple,
    Vector,
    InputSpec,
)
from four_c_metadata.json_schema_draft_2020_12 import (
    Types,
    Validators,
    Applicators,
    Schema,
    Core,
    TypeAlias,
)

METADATA_TO_SCHEMA_PRIMITIVE = {
    "double": Types.number,
    "bool": Types.boolean,
    "int": Types.integer,
    "string": Types.string,
    "path": Types.string,
}

MISSING_DESCRIPTION = "No description yet."


def set_description(description: NotSetAlias[str] | None) -> str:
    if check_if_set(description):
        return description
    return MISSING_DESCRIPTION


def short_name(spec: InputSpec) -> str:
    if check_if_set(spec.name) or spec.name is None:
        return spec.name + f" ({spec.spec_type})"
    return spec.spec_type


def create_schema(spec: InputSpec, schema_type: TypeAlias) -> Schema:

    if spec.validator is not None:
        validators = validator_to_schema(spec.validator)
        if validators is not None:
            schema_type.validators.extend(validators)

    if hasattr(spec, "noneable"):
        if spec.noneable:
            schema_type = [schema_type, Types.null()]

    schema = Schema(
        title=short_name(spec),
        description=set_description(spec.description),
        default=spec.default if hasattr(spec, "default") else NotSet(),
        schema_type=schema_type,
    )
    return schema


def create_schema_from_primitive(primitive: Primitive) -> Schema:
    schema_type = METADATA_TO_SCHEMA_PRIMITIVE[primitive.spec_type]()

    return create_schema(primitive, schema_type)


def create_schema_from_enum(enum: Enum) -> Schema:
    schema_type = Types.string(validators=[Validators.enum(enum.choices)])

    return create_schema(enum, schema_type)


def schema_from(spec: InputSpec | All_Of | One_Of) -> Schema:
    match spec:
        case Primitive():
            return create_schema_from_primitive(spec)
        case Enum():
            return create_schema_from_enum(spec)
        case Vector():
            return create_schema_from_vector(spec)
        case Map():
            return create_schema_from_map(spec)
        case Tuple():
            return create_schema_from_tuple(spec)
        case Group():
            return create_schema_from_group(spec)
        case List():
            return create_schema_from_list(spec)
        case Selection():
            return create_schema_from_selection(spec)
        case All_Of():
            return create_schema_from_all_of(spec)
        case One_Of():
            return create_schema_from_one_of(spec)
        case _:
            raise TypeError(f"Unknown type {type(spec)}: {spec}")


def create_schema_from_vector(vector: Vector) -> Schema:
    if isinstance(vector.validator, AllEmementsValidator):
        vector.value_type.validator = vector.validator.element_validator
        vector.validator = None
    validators = []
    if vector.size is not None:
        validators = [
            Validators.minItems(vector.size),
            Validators.maxItems(vector.size),
        ]

    if isinstance(vector.value_type, (Primitive, Map)):
        if not check_if_set(vector.value_type.description):
            vector.value_type.description = None
    items = schema_from(vector.value_type)

    schema = Types.array(applicators=[Applicators.items(items)], validators=validators)

    return create_schema(vector, schema)


def create_schema_from_map(map_spec: Map) -> Schema:
    validators = []
    if map_spec.size is not None:
        validators = [
            Validators.minProperties(map_spec.size),
            Validators.maxProperties(map_spec.size),
        ]

    pattern_property = schema_from(map_spec.value_type)

    schema = Types.object(
        applicators=[Applicators.patternProperties({"^.*$": pattern_property})],
        validators=validators,
    )

    return create_schema(map_spec, schema)


def create_schema_from_tuple(tuple_spec: Tuple) -> Schema:
    prefix_items = []
    validators = []

    for spec in tuple_spec:
        prefix_items.append(schema_from(spec))

    schema = Types.array(
        applicators=[Applicators.prefixItems(prefix_items)],
    )
    return create_schema(tuple_spec, schema)


def create_schema_from_all_of(all_of: All_Of) -> Schema:
    applicators = []
    validators = []

    # is a One_Of
    if all_of.is_one_of():
        return create_schema_from_one_of(all_of.specs[0])
    else:
        properties = {}
        required = []
        for spec in all_of:
            properties[spec.name] = schema_from(spec)
            if spec.required:
                required.append(spec.name)
        applicators.append(Applicators.properties(properties))
        if len(required):
            validators.append(Validators.required(required))
        applicators.append(Applicators.additionalProperties(False))

    schema_type = Types.object(validators=validators, applicators=applicators)
    description = all_of.description if check_if_set(all_of.description) else None

    return Schema(schema_type=schema_type, description=description)


def create_schema_from_one_of(one_of: One_Of) -> Schema:
    applicators = []

    all_ofs = []

    def sort_one_of_option_names(specs: One_Of) -> list:
        options = [[s.name for s in spec] for spec in specs]

        common_names = set(options[0])
        for l in options:
            common_names = common_names.intersection(set(l))

        if common_names:
            common_names_list = sorted(common_names)
            for i, l in enumerate(options):
                o = [name for name in l if name not in common_names]
                options[i] = o + common_names_list

        return [", ".join(l) for l in options]

    for all_of, name in zip(
        one_of.specs, sort_one_of_option_names(one_of.specs), strict=True
    ):

        schema_all_of = create_schema_from_all_of(all_of)
        schema_all_of.title = name
        schema_all_of.description = name
        all_ofs.append(schema_all_of)

    applicators.append(Applicators.oneOf(all_ofs))

    schema_type = Types.object(applicators=applicators)
    description = one_of.description if check_if_set(one_of.description) else None

    return Schema(schema_type=schema_type, description=description)


def create_schema_from_group(group: Group) -> Schema:
    if len(group.spec):
        schema_type = create_schema_from_all_of(group.spec).schema_type
    else:
        schema_type = Types.object(validators=[Validators.const({})])
    return create_schema(group, schema_type)


def create_schema_from_list(list_spec: List) -> Schema:
    items = create_schema_from_all_of(list_spec.spec)
    validators = []
    if list_spec.size is not None:
        validators.append(Validators.minProperties(list_spec.size))
        validators.append(Validators.maxProperties(list_spec.size))

    schema = Types.array(validators=validators, applicators=[Applicators.items(items)])

    return create_schema(list_spec, schema)


def create_schema_from_selection(selection: Selection) -> Schema:
    schemas = []

    for key, spec in selection.choices.items():
        all_of_schema = create_schema_from_all_of(spec)
        all_of_schema.title = key
        all_of_schema.description = spec.specs[0].description
        schemas.append(all_of_schema)

    schema = Types.object(applicators=[Applicators.oneOf(schemas)])
    return create_schema(selection, schema)


def validator_to_schema(
    validator: RangeValidator | PatternValidator,
) -> Sequence[
    Validators.exclusiveMinimum
    | Validators.exclusiveMaximum
    | Validators.minimum
    | Validators.maximum
    | Validators.pattern
]:
    match validator:

        case RangeValidator():
            validators = []
            if validator.minimum_exclusive:
                validators.append(Validators.exclusiveMinimum(validator.minimum))
            else:
                validators.append(Validators.minimum(validator.minimum))

            if validator.maximum_exclusive:
                validators.append(Validators.exclusiveMaximum(validator.maximum))
            else:
                validators.append(Validators.maximum(validator.maximum))

            return validators

        case PatternValidator():
            return [Validators.pattern(validator.pattern.pattern)]

        case _:
            if validator is not None:
                print(f"Validator {validator} not supported by JSON schema.")
            return None


def main(fourc_metadata_yaml_path: str, json_schema_path: str):
    json_schema_path = pathlib.Path(json_schema_path)

    # Load metadata
    metadata = load_yaml(fourc_metadata_yaml_path)

    # Title section
    description_section_name = metadata["metadata"]["description_section_name"]

    # Description
    metadata_description = f"Schema for 4C\nCommit hash: {metadata['metadata']['commit_hash']}\nVersion: {metadata['metadata']['version']}"

    metadata_object = All_Of.from_4C_metadata(metadata["sections"])
    metadata_object.description = metadata_description

    # Add legacy sections
    metadata_object.specs.extend(
        [
            Vector(
                value_type=Primitive("string", name=section_name),
                description=(
                    section_name[:-1].lower()
                    if section_name.endswith("S")
                    else section_name.lower() + " entry"
                ),
                name=section_name,
                required=False,
            )
            for section_name in metadata["legacy_string_sections"]
        ]
    )

    # Create JSON schema object
    schema = create_schema_from_all_of(metadata_object)

    # Append core data
    schema.schema_type.core.append(
        Core.schema("https://json-schema.org/draft/2020-12/schema")
    )
    schema.schema_type.core.append(Core.id_core(metadata["metadata"]["commit_hash"]))

    # Find properties
    properties = schema.schema_type.applicators[
        [
            i
            for i in range(len(schema.schema_type.applicators))
            if isinstance(schema.schema_type.applicators[i], Applicators.properties)
        ][0]
    ]

    # Add description section, can be anything.
    properties.schemas[description_section_name] = Schema(
        [
            Types.array(),
            Types.boolean(),
            Types.integer(),
            Types.null(),
            Types.number(),
            Types.object(),
            Types.string(),
        ],
        title=description_section_name,
        description="In this section metadata for information purposes can be added.",
    )

    # Move functions to pattern properties
    function_schema = properties.schemas.pop("FUNCT<n>")
    schema.schema_type.applicators.append(
        Applicators.patternProperties({"^FUNCT[1-9][0-9]*$": function_schema})
    )

    # Dump schema as JSON
    schema_dict = schema.to_dict()
    json_schema_path.write_text(json.dumps(schema_dict, indent=2))

    # Dump partial schema as JSON
    schema_dict.pop("required")
    json_schema_partial_path = json_schema_path.with_stem(
        json_schema_path.stem + "_partial"
    )
    json_schema_partial_path.write_text(json.dumps(schema_dict, indent=2))


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create JSON schema from 4C metadata yaml"
    )
    parser.add_argument(
        "fourc_metadata_yaml_path", help="Path to the yaml file generated by 4C."
    )
    parser.add_argument("json_schema_path", help="Path for the JSON schema.")
    args = parser.parse_args()
    main(**vars(args))


if __name__ == "__main__":
    cli()
