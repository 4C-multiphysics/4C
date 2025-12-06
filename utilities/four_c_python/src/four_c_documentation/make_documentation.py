# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# The MIT License (MIT)
#
# Copyright (c) 2025 FourCIPP Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTAB    ILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import pathlib
import textwrap
import re
import time
from dataclasses import dataclass, field
from functools import partial

from four_c_common_utils.io import load_yaml

from four_c_metadata.metadata import (
    make_context,
    NATIVE_CPP_TYPES,
    All_Of,
    AllEmementsValidator,
    Enum,
    Group,
    List,
    Map,
    One_Of,
    PatternValidator,
    Primitive,
    RangeValidator,
    Selection,
    Tuple,
    Vector,
    NotSetString,
)
from four_c_metadata.not_set import check_if_set

DESCRIPTION_MISSING = '<span style="color:grey">*no description yet*</span>'
TOO_MANY_TESTS_TO_SHOW = 200
TESTS_TO_SHOW_IF_TOO_MANY = 20
USED_ANCHORS = set()


# Data class to store which sections belong to which chapter (and thus to which file)
@dataclass
class ReferenceChapter:
    title: str  # Title of the chapter
    filename: str  # Filename to write the chapter to (without .md)
    linkanchor: str  # Link anchor to link to this chapter
    pattern: list[
        str
    ]  # List of regular expressions to match section names to include in this chapter
    description: str = ""  # Description of the chapter (filled later)
    content: str = ""  # Content of the chapter (filled later)
    has_content: bool = False  # Whether the chapter has any content (filled later)


def sanitize_anchor(text: str):
    """
    Sanitize a string so it is safe and consistent to use as an explicit MyST anchor.

    - Strip trailing HTML (e.g. ", <span ...>").
    - Remove '<' and '>'.
    - Remove whitespace.
    - Replace '/' with '_'.
    """
    # cut off HTML decoration like ", <span style=...>"
    text = re.sub(r"\s*, <span.*", "", text).strip()
    text = text.replace("<", "").replace(">", "")
    text = text.replace(" ", "")
    text = text.replace("/", "_")
    return text


def make_anchor_base(option_name: str, path_prefix: str | None):
    """
    Build the base anchor from an option name and an optional hierarchical prefix.
    """
    base = sanitize_anchor(option_name)
    if path_prefix:
        base = f"{path_prefix}_{base}"
    return base


def make_unique_anchor(base: str):
    if base not in USED_ANCHORS:
        USED_ANCHORS.add(base)
        return base

    n = 2
    candidate = f"{base}-{n}"
    while candidate in USED_ANCHORS:
        n += 1
        candidate = f"{base}-{n}"

    USED_ANCHORS.add(candidate)
    return candidate


def make_collapsible(content, summary=None, open_content=None):
    documentation = []
    if open_content is None:
        open_content = len(content.split("\n")) < 100

    if open_content:
        documentation.append("<details open>")
    else:
        documentation.append("<details>")

    if summary is not None:
        documentation.append("")
        documentation.extend(["<summary>", summary, "</summary>"])
    documentation.append("")
    documentation.append(content)
    documentation.append("")
    documentation.append("</details>")
    documentation.append("")
    return "\n".join(documentation) + "\n"


def make_spec_description(
    name,
    spec_type,
    header_properties: dict,
    description: str,
    collapsible_properties: dict = None,
):
    header = ""
    if check_if_set(name):
        header += f"**{name}** "
    header += (
        f"(*{spec_type}*"
        + ", ".join([""] + [f"*{k}*: {v}" for k, v in header_properties.items()])
        + ")\\"
    )

    description = set_missing_description(description) + "\n"

    if collapsible_properties is not None:
        for k, v in collapsible_properties.items():
            description += "\n" + make_collapsible(textwrap.indent(v, ""), k)

    return header, description


def make_section_description(
    name,
    spec_type,
    header_properties: dict,
    description: str,
    collapsible_properties: dict = None,
    section_in_tests: dict = {},
    section_in_tutorials: dict = {},
):
    if name.startswith("FUNCT"):
        name = name.replace("<", "[").replace(">", "]")
        description = "Functions can be used for material properties and boundary/initial conditions. More information can be found in the {ref}`Analysis guide <functiondefinitions>`"
    string = "\n## " + name
    string += (
        f"\n*Type*: {spec_type}"
        + ", ".join([""] + [f"*{k}*: {v}" for k, v in header_properties.items()])
        + ""
    )

    string += "\n\n" + set_missing_description(description) + "\n"

    if name in section_in_tests:

        n_files = len(section_in_tests[name])

        summary = f"Used in {n_files} test"
        if n_files > 1:
            summary += "s"

        if n_files > TOO_MANY_TESTS_TO_SHOW:
            summary += f", showing only the first {TESTS_TO_SHOW_IF_TOO_MANY}."
            n_files_to_show = TESTS_TO_SHOW_IF_TOO_MANY
        else:
            n_files_to_show = n_files
        content = "\n - " + "\n - ".join(
            [
                f"[{n.split('/')[-1]}]({n})"
                for n in sorted(section_in_tests[name])[:n_files_to_show]
            ]
        )

        string += make_collapsible(content, summary, open_content=False)
    else:
        string += "\n Not used in any test."

    if name in section_in_tutorials:
        content = "\n - " + "\n - ".join(
            [f"[{n.split('/')[-1]}]({n})" for n in sorted(section_in_tutorials[name])]
        )

        n_files = len(section_in_tutorials[name])

        summary = f"Used in {n_files} tutorial"
        if n_files > 1:
            summary += "s"

        string += make_collapsible(content, summary, open_content=False)

    string += "\n"
    if collapsible_properties is not None:
        for k, v in collapsible_properties.items():
            string += "\n" + make_collapsible(textwrap.indent(v, " "), k)

    string + "\n"
    return string


def none_to_null(obj):
    if obj is None:
        return "null"
    return obj


def type_or_none(type_description, entry):
    if entry.noneable:
        return type_description + " or null"
    return type_description


def value_type_flatter(entry):
    def is_primitive(obj):
        if isinstance(obj, Primitive):
            return obj.spec_type

        return value_type_flatter(obj)

    if isinstance(entry, Vector):
        return "vector\\<" + is_primitive(entry.value_type) + ">"
    elif isinstance(entry, Map):
        return "map\\<str," + is_primitive(entry.value_type) + ">"
    elif isinstance(entry, Tuple):
        return (
            "tuple\\<" + ", ".join([is_primitive(v) for v in entry.value_types]) + ">"
        )
    else:
        raise ValueError(f"{entry.spec_type}, {entry}")


def flatten_size_vector(vector):
    text = str(vector.size)
    if vector.size is None:
        text = "n"

    if isinstance(vector.value_type, Primitive):
        return text

    elif isinstance(vector.value_type, Vector):
        return text + " x " + flatten_size_vector(vector.value_type)
    elif isinstance(vector.value_type, Tuple):
        return text
    else:
        raise ValueError(f"{vector}")


def primitive_to_md(primitive: Primitive, make_description=make_spec_description):
    type_description = type_or_none(primitive.spec_type, primitive)

    header_properties = {"required": primitive.required}
    header_properties.update(validator_to_header_argument(primitive.validator))
    if not primitive.required:
        header_properties["default"] = none_to_null(primitive.default)

    return make_description(
        primitive.name, type_description, header_properties, primitive.description
    )


def vector_or_map_to_md(
    entry: Vector | Map, make_description=make_spec_description, path_prefix: str = ""
):
    type_description = type_or_none(value_type_flatter(entry), entry)

    header_properties = {"size": flatten_size_vector(entry), "required": entry.required}
    if entry.validator is not None:
        header_properties.update(validator_to_header_argument(entry.validator))

    if not entry.required:
        header_properties["default"] = none_to_null(entry.default)

    collapsible_properties = None
    if not isinstance(entry.value_type, Primitive):
        collapsible_properties = {
            "Value type": all_of_to_md(
                All_Of([entry.value_type]), path_prefix=path_prefix
            )
        }

    return make_description(
        entry.name,
        type_description,
        header_properties,
        entry.description,
        collapsible_properties,
    )


def tuple_to_md(
    tuple_entry: Tuple, make_description=make_spec_description, path_prefix: str = ""
):
    type_description = type_or_none(value_type_flatter(tuple_entry), tuple_entry)

    header_properties = {"required": tuple_entry.required}
    header_properties.update(validator_to_header_argument(tuple_entry.validator))

    if not tuple_entry.required:
        header_properties["default"] = none_to_null(tuple_entry.default)

    collapsible_properties = {
        "Entries": all_of_to_md(
            All_Of(tuple_entry.value_types), indent=1, path_prefix=path_prefix
        )
    }

    return make_description(
        tuple_entry.name,
        type_description,
        header_properties,
        tuple_entry.description,
        collapsible_properties,
    )


def enum_to_md(enum: Enum, make_description=make_spec_description):
    type_description = type_or_none(enum.spec_type, enum)

    header_properties = {"required": enum.required}
    header_properties.update(validator_to_header_argument(enum.validator))

    if not enum.required:
        header_properties["default"] = none_to_null(enum.default)

    choices = ""
    for choice, choice_description in zip(enum.choices, enum.choices_description):
        choices += f"\n - **{choice}**"
        if choice_description is not None:
            choices += " : " + choice_description

    collapsible_properties = {"Choices": choices}

    return make_description(
        enum.name,
        type_description,
        header_properties,
        enum.description,
        collapsible_properties,
    )


def group_to_md(
    group: Group, make_description=make_spec_description, path_prefix: str = ""
):
    type_description = type_or_none(group.spec_type, group)

    collapsible_properties = {}
    if len(group.spec) > 0:
        collapsible_properties = {
            "Contains": all_of_to_md(group.spec, indent=1, path_prefix=path_prefix)
        }

    header_properties = validator_to_header_argument(group.validator)
    return make_description(
        group.name,
        type_description,
        header_properties,
        group.description,
        collapsible_properties,
    )


def list_to_md(
    list_entry: List, make_description=make_spec_description, path_prefix: str = ""
):
    type_description = type_or_none(list_entry.spec_type, list_entry)

    header_properties = {}
    if list_entry.size is not None:
        header_properties = {"size": list_entry.size}
    header_properties.update(validator_to_header_argument(list_entry.validator))

    return make_description(
        list_entry.name,
        type_description,
        header_properties,
        list_entry.description,
        collapsible_properties={
            "Each element contains": all_of_to_md(
                list_entry.spec, indent=1, path_prefix=path_prefix
            )
        },
    )


def selection_to_md(
    selection: Selection, make_description=make_spec_description, path_prefix: str = ""
):
    type_description = type_or_none(selection.spec_type, selection)

    choices = ""
    for choice, choice_spec in selection.choices.items():
        choices += "\n\n" + " " + f"- *{choice}*:\n\n"
        choices += all_of_to_md(choice_spec, indent=3, path_prefix=path_prefix)

    return make_description(
        selection.name,
        type_description,
        validator_to_header_argument(selection.validator),
        selection.description,
        collapsible_properties={"Choices": choices},
    )


def sort_one_of_option_names(one_of: One_Of) -> list:
    options = [[s.name for s in spec] for spec in one_of]

    common_names = set(options[0])
    for l in options:
        common_names = common_names.intersection(set(l))

    if common_names:
        common_names_list = [
            f'<span style="color:grey">{name}</span>' for name in sorted(common_names)
        ]
        for i, l in enumerate(options):
            o = [name for name in l if name not in common_names]
            options[i] = o + common_names_list

    return [", ".join(l) for l in options]


def one_of_to_md(one_of: One_Of, path_prefix: str = ""):
    header = "*One of*"

    description = ""
    if check_if_set(one_of.description):
        description += "\n" + one_of.description + "\n"

    open_content = len(one_of.specs) < 11

    names = sort_one_of_option_names(one_of)

    for i, spec in enumerate(one_of.specs):
        anchor_base = make_anchor_base(names[i], path_prefix)
        opt_anchor = make_unique_anchor(anchor_base)

        key = "Option (" + names[i] + ")"

        # Handle nested One Of
        nested_prefix = anchor_base
        content = textwrap.indent(
            all_of_to_md(spec, indent=1, path_prefix=nested_prefix),
            " ",
        )
        content = f"({opt_anchor})=\n\n" + content
        description += "\n" + make_collapsible(content, key, open_content)

    return header, description


def set_missing_description(description):
    if check_if_set(description):
        return "*" + description + "*"
    else:
        return DESCRIPTION_MISSING


def validator_to_header_argument(validator):
    if validator is None:
        return {}
    match validator:
        case RangeValidator():
            range_text = ""
            if validator.minimum_exclusive:
                range_text += "("
            else:
                range_text += "["

            range_text += f"{validator.minimum}, {validator.maximum}"

            if validator.maximum_exclusive:
                range_text += ")"
            else:
                range_text += "]"
            return {"range": range_text}
        case AllEmementsValidator():
            element_validator_name, element_validator_value = tuple(
                *validator_to_header_argument(validator.element_validator).items()
            )
            return {
                "each element must validate": f"{element_validator_name} {element_validator_value}",
            }

        case PatternValidator():
            return {"must match pattern": validator.pattern}
        case _:
            raise ValueError(f"Unknown validator {validator}")


def all_of_to_md(all_of: All_Of, indent=0, path_prefix: str = ""):
    entries = ""
    for entry in all_of:
        string_entry = None
        description_entry = None
        match entry:
            case Primitive():
                string_entry, description_entry = primitive_to_md(entry)
            case Vector() | Map():
                string_entry, description_entry = vector_or_map_to_md(
                    entry, path_prefix=path_prefix
                )
            case Tuple():
                string_entry, description_entry = tuple_to_md(
                    entry, path_prefix=path_prefix
                )
            case Enum():
                string_entry, description_entry = enum_to_md(entry)
            case Group():
                string_entry, description_entry = group_to_md(
                    entry, path_prefix=path_prefix
                )
            case List():
                string_entry, description_entry = list_to_md(
                    entry, path_prefix=path_prefix
                )
            case Selection():
                string_entry, description_entry = selection_to_md(
                    entry, path_prefix=path_prefix
                )
            case One_Of():
                string_entry, description_entry = one_of_to_md(
                    entry, path_prefix=path_prefix
                )
            case _:
                raise ValueError(type(entry))

        entries += (
            "\n\n"
            + indent * " "
            + "- "
            + string_entry
            + "\n"
            + textwrap.indent(description_entry, prefix=(indent + 2) * " ")
        )
    return entries


def create_section_markdown(section, section_in_tests, section_in_tutorials):
    section_link_anchor = "sec" + sanitize_anchor(section.name.lower())
    string_entry = "\n(" + section_link_anchor + ")=\n\n"

    section_prefix = sanitize_anchor(section.name.upper())

    create_section = partial(
        make_section_description,
        section_in_tests=section_in_tests,
        section_in_tutorials=section_in_tutorials,
    )

    match section:
        case Primitive():
            string_entry += primitive_to_md(section, create_section)
        case Vector() | Map():
            string_entry += vector_or_map_to_md(
                section, create_section, path_prefix=section_prefix
            )
        case Tuple():
            string_entry += tuple_to_md(
                section, create_section, path_prefix=section_prefix
            )
        case Enum():
            string_entry += enum_to_md(section, create_section)
        case Group():
            string_entry += group_to_md(
                section, create_section, path_prefix=section_prefix
            )
        case List():
            string_entry += list_to_md(
                section, create_section, path_prefix=section_prefix
            )
        case Selection():
            string_entry += selection_to_md(
                section, create_section, path_prefix=section_prefix
            )
        case One_Of():
            header, description = one_of_to_md(section, path_prefix=section_prefix)
            string_entry += header + "\n" + description + "\n"
        case _:
            raise ValueError(type(section))
    return string_entry + "\n\n"


def has_entity(section):
    def iterate(obj):
        if isinstance(obj, NATIVE_CPP_TYPES):
            yield obj
        else:
            for o in obj:
                yield from iterate(o)

    for o in iterate(section):
        if o.name == "ENTITY_TYPE":
            return True

    return False


def get_sections_in_files(directory):
    sections = {}
    for file in directory.glob("**/*.4C.yaml"):
        sections_names = [k.strip() for k in load_yaml(file).keys()]
        for s in sections_names:
            file_name = str(file).split("tests/")[-1]
            file_name = (
                f"https://github.com/4C-multiphysics/4C/blob/main/tests/{file_name}"
            )
            if s in sections:
                sections[s].append(file_name)
            else:
                sections[s] = [file_name]
    return sections


def create_category_file(section_list, name):
    documentation = []

    documentation.append(f"# {name}")
    documentation.append("")
    documentation.append("")
    for section in section_list:
        documentation.append(create_section_markdown(section))
    pathlib.Path(name.lower().replace(" ", "_") + ".md").write_text(
        "\n".join(documentation)
    )


def create_markdown_documentation(
    reference_directory, tests_directory, fourc_metadata_yaml_path
):
    """
    Args:
        reference_directory: Directory to write the reference files to
        tests_directory: Directory containing the tests (to check which sections are used in tests)
        fourc_metadata_yaml_path: Path to the fourc_metadata.yaml file (containing the metadata for the 4C input file format)

    Here we create markdown documentation for the 4C input file format (called from conf.py.in)
    Note particularly that the reference sections are split into different files.
    """

    start_time = time.time()
    section_in_tests = get_sections_in_files(
        pathlib.Path(tests_directory) / "input_files"
    )
    section_in_tutorials = get_sections_in_files(
        pathlib.Path(tests_directory) / "tutorials"
    )
    elapsed_time = time.time() - start_time
    print(f"Loading sections took {elapsed_time:.3f} seconds")

    fourc_metadata = load_yaml(fourc_metadata_yaml_path)

    # Here comes the list of ReferenceChapter objects, which defines the different files and which sections go into which file.
    # The sections are matched by regular expressions in the pattern attribute of each ReferenceChapter object.
    reference_documents = [
        ReferenceChapter(
            "Spatial discretization",
            "discretization_reference",
            "discretization_reference",
            [".* DOMAIN$", ".* ELEMENTS$", ".* GEOMETRY$", ".*KNOTVECTORS$"],
            "This is the reference containing the sections related to the spatial discretization of the problem, namely finite elements. "
            + "Note that the definition of elements or particles one by one through ``* ELEMENTS`` or ``PARTICLES`` is not yet contained due to an old input spec layout",
        ),
        ReferenceChapter(
            "Material information",
            "materials_reference",
            "materials_reference",
            ["^MATERIALS$", "^CLONING MATERIAL MAP$"],
            "This is the reference for all material models. "
            + "The Cloning Material Map is also included here, as it defines the situation where mesh definitions of one physics (SRC_FIELD) are used for another physics (TARGET_FIELD) and thus need different materials.",
        ),
        ReferenceChapter(
            "Boundary and constraint conditions",
            "condition_reference",
            "condition_reference",
            ["^DESIGN ", ".*CONDITIONS$"],
            "This is the reference for all boundary and constraint conditions.",
        ),
        ReferenceChapter(
            "General parameters",
            "general_reference",
            "general_reference",
            [".*"],
            "This is the reference for all parameters that do not fit into the other categories, often called header parameters.",
        ),
    ]

    for doc in reference_documents:
        doc.content = f"({doc.linkanchor})=\n\n" + f"# {doc.title}\n\n"
        doc.content += f"{set_missing_description(doc.description)}\n\n"

    context = make_context(fourc_metadata)
    obj = All_Of.from_4C_metadata(fourc_metadata["sections"], context)
    sorted_entries = sorted(list(obj), key=lambda e: getattr(e, "name", "").lower())

    for section in sorted_entries:
        for doc in reference_documents:
            if any(re.match(pattern, section.name) for pattern in doc.pattern):
                doc.content += create_section_markdown(
                    section, section_in_tests, section_in_tutorials
                )
                doc.has_content = True
                break
    with open(
        pathlib.Path(reference_directory) / "reference_documents.txt", "w"
    ) as reference_index_file:
        for doc in reference_documents:
            if doc.has_content:
                full_reference_filepath = pathlib.Path(reference_directory) / (
                    doc.filename + ".md"
                )
                full_reference_filepath.write_text(doc.content)
                print(f"Writing documentation to {doc.filename}.md finished")
                reference_index_file.write("\n" + doc.filename)
        reference_index_file.write("\n")


if __name__ == "__main__":
    create_markdown_documentation(
        "fourcipp_documentation.md", ".", "fourc_metadata.yaml"
    )
