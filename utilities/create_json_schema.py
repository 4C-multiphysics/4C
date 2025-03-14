# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Create JSON schema for YAML validation."""

import json
import argparse
from pathlib import Path

from metadata_utils import (
    NotSet,
    NOTSET,
    NAMED_TYPES,
    MISSING_DESCRIPTION,
    FOURC_BASE_TYPES_TO_JSON_SCHEMA_DICT,
    Primitive,
    Vector,
    Map,
    Enum,
    Group,
    Selection,
    List,
    All_Of,
    One_Of,
    metadata_object_from_file,
)
from dataclasses import asdict


def set_description(schema, description):
    """Set the description.

    Args:
        schema (dict): JSON schema
        description (str): Description to be added.
    """
    # Add the missing description if nothing was set
    if isinstance(description, NotSet):
        schema["description"] = MISSING_DESCRIPTION
    else:
        if description is not None:
            # Add the description
            schema["description"] = description


def json_schema(
    *,
    schema_type="object",
    title=None,
    description=NOTSET,
    noneable=False,
    default=NOTSET,
    enum=None,
    const=NOTSET,
    properties=None,
    required=None,
    oneOf=None,
    minProperties=None,
    maxProperties=None,
    patternProperties=None,
    additionalProperties=False,
    items=None,
    minItems=None,
    maxItems=None,
):
    """Create JSON schema dict."""

    def set_if_not_none(schema, name, value):
        if value is not None:
            schema[name] = value

    def set_if_true(schema, name, value):
        if value:
            schema[name] = value

    def set_if_set(schema, name, value):
        if not isinstance(value, NotSet):
            schema[name] = value

    schema = {}

    # Properties for any kind of schema
    set_if_not_none(schema, "title", title)

    # Description is handled differently
    set_description(schema, description)

    set_if_not_none(schema, "type", schema_type)
    if schema_type:
        if noneable:
            schema["type"] = [schema_type, "null"]
        else:
            schema["type"] = schema_type
    set_if_true(schema, "enum", enum)

    # Allow None since it could be a desired value
    set_if_set(schema, "const", const)
    set_if_set(schema, "default", default)

    # Objects
    if schema_type == "object":
        set_if_true(schema, "required", required)
        set_if_true(schema, "properties", properties)
        set_if_true(schema, "oneOf", oneOf)
        if not additionalProperties:
            schema["additionalProperties"] = False
        set_if_true(schema, "patternProperties", patternProperties)
        set_if_not_none(schema, "minProperties", minProperties)
        set_if_not_none(schema, "maxProperties", maxProperties)

    # Arrays
    if schema_type == "array":
        set_if_not_none(schema, "items", items)
        set_if_not_none(schema, "maxItems", maxItems)
        set_if_not_none(schema, "minItems", minItems)

    return schema


def schema_from_base_type(primitive):
    """Create schema from Primitive with base type.

    Args:
        primitive (Primitive): Primitive made of base type.

    Returns:
        dict: JSON schema data
    """
    schema = json_schema(
        schema_type=FOURC_BASE_TYPES_TO_JSON_SCHEMA_DICT[primitive.type],
        description=primitive.description,
        title=primitive.short_description(),
        default=primitive.default,
        noneable=primitive.noneable,
        const=primitive.constant,
    )
    return schema


def schema_from_enum(enum):
    """Create schema from Enum.

    Args:
        enum (Enum): Enum parameter

    Returns:
        dict: JSON schema data
    """
    # One could also do this using a oneOf where the choices are strings with constant values.
    # This would allow to add a description for each option.
    schema = json_schema(
        schema_type=FOURC_BASE_TYPES_TO_JSON_SCHEMA_DICT["string"],
        title=enum.short_description(),
        description=enum.description,
        default=enum.default,
        enum=list(set(enum.choices)),
        noneable=enum.noneable,
    )

    return schema


def schema_from_group(group):
    """Create schema from Group.

    Args:
        group (Group): Group parameter

    Returns:
        dict: JSON schema data
    """
    # Empty group
    if group.is_empty():
        title = f"{group.name} (empty group)"
        schema = json_schema(
            title=title,
            description=group.description,
            properties={},
            const={},
            default={},
            noneable=group.noneable,
        )
        return schema

    # Use All_of to create the schema
    metadata_dict = asdict(group)
    noneable = metadata_dict.pop("noneable", False)
    metadata_dict.pop("defaultable", False)
    schema = schema_from_all_of(All_Of(**metadata_dict))
    schema["title"] = group.short_description()

    set_description(schema, group.description)

    if noneable:
        schema["type"] = [schema["type"], "null"]
        schema["title"] = f"{group.name} (group, null)"
    return schema


def schema_from_selection(selection):
    """Create selection schema.

    Args:
        selection (Selection): Metadata object.
    """

    specs = []

    for choice in selection.choices:
        # Add the selection parameter
        selector_variable = Primitive(
            name=selection.selector,
            type="string",
            required=True,
            description=f"Selector type {choice.name} for {selection.name} (string)",
            constant=choice.name,
        )

        choice = All_Of(
            name=choice.name,
            required=True,
            description=f"Selector type {choice.name} for {selection.name} (string)",
            specs=[selector_variable, choice.spec],
        )
        specs.append(choice)

    # The choices are options of a one_of
    schema = schema_from_one_of(One_Of(description=selection.description, specs=specs))

    schema["title"] = selection.short_description()

    # Add the options additionally as property to be able to chose between them
    schema["properties"] = {
        selection.selector: schema_from_base_type(
            Enum(
                name=selection.selector,
                type="string",
                required=True,
                description=f"Selector type for {selection.name} (string)\n"
                + "\nchoices are: "
                + ", ".join([choice.name for choice in selection.choices]),
                choices=[choice.name for choice in selection.choices],
            )
        )
    }

    # Selection can be noneable
    if selection.noneable:
        schema["type"] = [schema["type"], "null"]

    # Add all the choices as description
    schema["description"] = (
        schema.get("description", MISSING_DESCRIPTION)
        + "\nchoices are: "
        + ", ".join([choice.name for choice in selection.choices])
    )

    return schema


def array_schema(parameter, items):
    """Create array schema.

    Args:
        parameter (Vector, List): Metadata object
        items (dict): JSON schema for item type of the parameter

    Returns:
        dict: JSON schema data
    """
    schema = json_schema(
        title=parameter.short_description(),
        description=parameter.description,
        schema_type="array",
        items=items,
        maxItems=parameter.size,
        minItems=parameter.size,
        noneable=parameter.noneable,
    )

    return schema


def schema_from_vector(vector):
    """Create schema from vector.

    Args:
        vector (Vector): Vector parameter

    Returns:
        dict: JSON schema data
    """
    if isinstance(vector.description, NotSet):
        # ignore this entry
        vector.description = None
    items = get_schema(vector.value_type)
    return array_schema(vector, items)


def schema_from_list(list_metadata):
    """Create schema from list.

    Args:
        list_metadata (List): List parameter

    Returns:
        dict: JSON schema data
    """
    if isinstance(list_metadata.spec.description, NotSet):
        # Ignore this entry
        list_metadata.spec.description = None

    if isinstance(list_metadata.spec, NAMED_TYPES):
        items = json_schema(
            title=list_metadata.spec.short_description(),
            description=list_metadata.spec.description,
            required=[list_metadata.spec.name] if list_metadata.spec.required else None,
            properties={list_metadata.spec.name: get_schema(list_metadata.spec)},
        )
    # Essentially one_ofs remain
    else:
        items = get_schema(list_metadata.spec)

    return array_schema(list_metadata, items)


def schema_from_map(map_metadata):
    """Create schema from map.

    Args:
        map_metadata (Map): Map parameter

    Returns:
        dict: JSON schema data
    """
    value_schema = get_schema(map_metadata.value_type)

    schema = json_schema(
        title=map_metadata.short_description(),
        description=map_metadata.description,
        patternProperties={"^.*$": value_schema},
        minProperties=map_metadata.size,
        maxProperties=map_metadata.size,
        noneable=map_metadata.noneable,
    )
    return schema


def schema_from_one_of(one_of):
    """Create schema from one_of.

    Args:
        one_of (One_of): One_of collection

    Returns:
        dict: JSON schema data
    """
    schemas_options = []

    # One_of in One_of
    if one_of.one_ofs():
        for v in one_of.one_ofs():
            properties = schema_from_one_of(v)
            properties.pop("properties", None)
            schemas_options.append(properties)

    # named specs
    if named_specs := one_of.named_specs():
        named_specs_schema = [object_schema(spec) for spec in named_specs]
        schemas_options.extend(named_specs_schema)

    # all_of nesting
    if all_ofs := one_of.all_ofs():
        for v in all_ofs:
            properties = schema_from_all_of(v)
            schemas_options.append(properties)

    if not one_of.name:
        if specs_names := one_of.parameter_names():
            one_of.name = " or ".join(specs_names)
        else:
            one_of.name = None

    if isinstance(one_of.description, NotSet):
        if specs_names := one_of.parameter_names():
            one_of.description = " or ".join(specs_names)
        else:
            one_of.description = None

    schema = json_schema(
        title=one_of.name,
        description=one_of.description,
        oneOf=schemas_options,
        additionalProperties=True,
    )

    def flatten_one_ofs(schema):
        """A single oneOf in a oneOf is flatted."""
        if "oneOf" in schema and len(schema["oneOf"]) == 1:
            return flatten_one_ofs(schema["oneOf"][0])
        return schema

    return flatten_one_ofs(schema)


def schema_from_all_of(all_of):
    """Create schema from all_of.

    Args:
        all_of (All_of): All_of collection

    Returns:
        dict: JSON schema data
    """

    all_of = condense_all_of(all_of)

    # Check if is empty
    if all_of.is_empty():
        raise ValueError("Empty all_of not possible!")

    # Condensing might remove an unnecessart All_of and transformed it into a One_of
    if isinstance(all_of, One_Of):
        return schema_from_one_of(all_of)

    named_specs, one_ofs, all_ofs = all_of.split_specs(pop=True)

    # Sanity checks

    # All_ofs in all_ofs is not possible after condensing
    if all_ofs:
        raise TypeError("All_ofs in All_ofs are not possible anymore.")

    # One_ofs and named_specs are not possible after condensing
    if bool(one_ofs) == bool(named_specs):
        raise TypeError("One_ofs and named specs together are not possible.")

    # One_ofs
    if one_ofs:
        if len(one_ofs) == 1:
            raise TypeError("A single one_of had to be returned previously")
        else:
            raise NotImplementedError(
                f"Multiple parallel one_ofs not supported found {len(one_ofs)}. If this should become required anyOf in JSON schema files have to be introduced."
            )

    if named_specs:
        properties = {}
        required_properties = []
        for entry in named_specs:
            properties[entry.name] = get_schema(entry)
            if entry.required:
                required_properties.append(entry.name)

        if not all_of.name:
            all_of.name = " & ".join(properties.keys())

        if not all_of.description:
            all_of.description = " & ".join(properties.keys())

        # Return a simple object schema
        schema = json_schema(
            title=all_of.name,
            description=all_of.description,
            required=required_properties,
            properties=properties,
        )
        return schema

    raise ValueError("Empty all of is not supported!")


def get_schema(parameter):
    """Create schema from parameter.

    Args:
        parameter (Parameter): Parameter to create a schema for.

    Returns:
        dict: JSON schema data
    """
    schema = {}

    match parameter:
        case Enum():
            schema = schema_from_enum(parameter)
        case Vector():
            schema = schema_from_vector(parameter)
        case Map():
            schema = schema_from_map(parameter)
        case Primitive():
            schema = schema_from_base_type(parameter)
        case Selection():
            schema = schema_from_selection(parameter)
        case Group():
            schema = schema_from_group(parameter)
        case List():
            schema = schema_from_list(parameter)
        case All_Of():
            schema = schema_from_all_of(parameter)
        case One_Of():
            schema = schema_from_one_of(parameter)
        case _:
            raise TypeError(f"Unknown class {type(parameter)} for entry: {parameter}")

    return schema


def create_json_schema(schema_data, schema_id, schema_path):
    """Create JSON schema file.

    Args:
        schema_data (dict): Schema data
        schema_id (str): Id of the schema
        schema_path (str, pathlib.Path): Path to JSON schema
    """
    schema = {}
    schema["$id"] = schema_id
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema.update(schema_data)

    Path(schema_path).write_text(json.dumps(schema, indent=2), encoding="utf-8")


def object_schema(named_spec):
    """Create object schema for a single named spec.

    Args:
        named_spec (Parameter): Named spec to objectify

    Returns:
        dict: JSON schema data
    """
    schema = json_schema(
        title=named_spec.short_description(),
        description=named_spec.description,
        required=[named_spec.name] if named_spec.required else None,
        properties={named_spec.name: get_schema(named_spec)},
    )
    return schema


def add_named_specs(parameter, named_specs):
    """Add named specs to parameter.

    Args:
        parameter (Parameter): Parameter to add the named specs to
        named_specs (list): Named specs to add

    Returns:
        Parameter: parameter with the additional specs
    """
    # Adding specs to a named specs turns them into an all_of
    if isinstance(parameter, NAMED_TYPES):
        return All_Of(specs=[parameter] + named_specs)

    if isinstance(parameter, All_Of):

        # Only named specs already
        if not parameter.one_ofs() and not parameter.all_ofs():
            parameter.specs.extend(named_specs)
            return parameter

        # Make sure it only contains a single all_of
        parameter = condense_all_of(parameter)

        # Check if it was transformed into a single oneof
        if isinstance(parameter, One_Of):
            return add_named_specs(parameter, named_specs)

        if parameter.all_ofs():
            raise ValueError("oh no")

        # Add the named_specs of the all_of to the new named_specs
        named_specs += parameter.named_specs(pop=True)

        # Only remaining stuff is one_ofs
        for i, spec in enumerate(parameter.specs):
            parameter.specs[i] = add_named_specs(spec, named_specs)
        return parameter

    if isinstance(parameter, One_Of):
        for i, spec in enumerate(parameter.specs):
            parameter.specs[i] = add_named_specs(spec, named_specs)
        return parameter

    raise TypeError(f"Unknown metadata type {type(parameter)}: {parameter}")


def condense_all_of(all_of):
    """Condense All_ofs.

    The all_ofs in the metadata file are not natively supported in JSON schemas. Therefore, they are condensed by pushing named specs down to their options and avoid the mix of named specs and one_ofs.

    Args:
        all_of (All_of): All_of collection to condense.

    Returns:
        All_of or One_of: Condensed collection
    """
    # Empty all_of
    if all_of.is_empty():
        raise ValueError("Empty all_ofs are not allowed!")

    named_specs, one_ofs, all_ofs = all_of.split_specs()

    # Either only one_ofs or only named specs, easy
    if not all_ofs and (bool(one_ofs) != bool(named_specs)):
        return all_of

    # Extract the specs
    named_specs, one_ofs, all_ofs = all_of.split_specs(pop=True)

    # Ensure all the all_ofs a condensed and combine them into one all_of
    for ao in all_ofs:

        ao = condense_all_of(ao)
        if isinstance(ao, One_Of):
            raise TypeError(
                "All_of was converted into a One_of. This is not possible!."
            )

        new_named_specs, new_one_ofs, new_all_ofs = ao.split(pop=True)

        # Update the named specs
        named_specs += new_named_specs

        # update the one_ofs
        one_ofs += new_one_ofs

        if new_all_ofs:
            raise TypeError("No new all_ofs possible!")

    # No all_of left
    all_ofs = None

    if not one_ofs:
        raise TypeError(
            "There have to be one_ofs if this stage of the function is reached!"
        )

    # Add the named specs to the one_ofs
    for i, one_of in enumerate(one_ofs):
        one_ofs[i] = add_named_specs(one_of, named_specs)

    # If the only entry is a single one_of, return the one_of
    if len(one_ofs) == 1:
        return one_ofs[0]

    raise ValueError("This case should not be possible.")


def schema_for_description_section(section_name):
    """Create JSON schema for the description section.

    Args:
        section_name (str): Name of the description section

    Returns:
        dict: JSON schema data
    """
    # Allow any type
    schema = json_schema(
        title=section_name + " (no schema)",
        description=MISSING_DESCRIPTION + " (no schema)",
        additionalProperties=True,
        schema_type=None,
        noneable=True,
    )
    return schema


def main(metadata_path, json_schema_path):
    """Generate JSON schema from metadata file.

    Args:
        metadata_path (str, pathlib.Path): 4C emitted metadata file path
        json_schema_path (str, pathlib.Path): JSON schema file path
    """
    metadata_4C, title_section, code_metadata = metadata_object_from_file(metadata_path)
    schema = get_schema(metadata_4C)
    schema["properties"].update(
        {title_section: schema_for_description_section(title_section)}
    )
    schema["patternProperties"] = {
        "^FUNCT[1-9][0-9]*$": schema["properties"].pop("FUNCT<n>")
    }
    create_json_schema(schema, code_metadata["commit_hash"], json_schema_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create JSON schema from 4C metadata yaml"
    )
    parser.add_argument(
        "fourc_metadata_yaml_path", help="Path to the yaml file generated by 4C."
    )
    parser.add_argument("json_schema_path", help="Path for the JSON schema.")
    args = parser.parse_args()

    main(args.fourc_metadata_yaml_path, args.json_schema_path)
