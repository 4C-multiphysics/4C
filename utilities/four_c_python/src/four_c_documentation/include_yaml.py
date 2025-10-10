# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import yaml
import pathlib
import shutil
import jinja2
import argparse
import re

PATH_TO_TESTS = ""
RECOGNIZED_TYPES = ["rst", "md"]


def load_input_file(yaml_file):
    """Returns the dictionary of parameters for the given yaml file."""
    data = yaml.safe_load((PATH_TO_TESTS / pathlib.Path(yaml_file)).read_text())
    return data


def add_raw_text_from_file(text_file, indent=0):
    """Returns the dictionary of parameters for the given yaml file."""
    with open(PATH_TO_TESTS / pathlib.Path(text_file), "r") as file:
        input = file.readlines()
    if indent > 0:
        indent_string = " " * indent
        input = "".join([indent_string + line for line in input])
    else:
        input = "".join(input)
    return input


def load_meta_data():
    """Returns a dictionary of sections and their parameters from the meta file 4C_metadata.yaml."""
    metafile_data = yaml.safe_load(
        (PATH_TO_TESTS / ("../.." / pathlib.Path("4C_metadata.yaml"))).read_text()
    )
    section_dict = {}
    for section in metafile_data["sections"]["specs"]:
        params_avail = {}
        if "spec" in section:
            params_avail = section["spec"]["specs"]
        elif "specs" in section:
            params_avail = section["specs"][0]["specs"]
        else:
            KeyError(f'Parameters in section {section["name"]} not found.')

        def parse_parameters(params, parameters, section_name):
            for item in params:
                if item["type"] == "group":
                    # Recursively process nested group specs
                    parse_parameters(
                        item["specs"][0]["specs"], parameters, section_name
                    )
                else:
                    if "name" not in item:
                        print(
                            f"Currently, parameters in section {section_name} cannot be parsed."
                        )
                        continue

                    if item["required"]:
                        parameter_string = "# required parameter"
                    else:
                        parameter_string = f"{item['default']} # optional, the given value is the default"

                    parameters[item["name"]] = parameter_string

        parameter_dict = {}
        parse_parameters(params_avail, parameter_dict, section["name"])

        section_dict[section["name"]] = parameter_dict

    for section in metafile_data["legacy_string_sections"]:
        section_dict[section] = "legacy section, no parameters available."
    return section_dict


def yaml_dump(data, filetype="rst"):
    """Returns a string of the yaml file in the given filetype.
    As of now I can return markdown and restructuredText code blocks.
    """
    if filetype == "rst":
        rststring = ""
        yaml_data = yaml.safe_dump(data, sort_keys=False).split("\n")
        for line in yaml_data:
            rststring += "    " + line + "\n"
        rststring += "\n"
        return ".. code-block:: yaml\n\n" + rststring + "\n"
    elif filetype == "md":
        return "```yaml\n" + yaml.safe_dump(data, sort_keys=False) + "```\n"
    else:
        raise TypeError(f"Filetype {filetype} for yaml_dump cannot be recognized yet.")


def section_dump(input_file_section, section_names, filetype="rst"):
    """Returns a string of the given sections from the given dictionary of sections.
    The parameter section_names can be either a string (one section name) or a list of section names
    """
    if isinstance(section_names, str):
        section_names = [section_names]
    yaml_dict = {}
    for section in section_names:
        yaml_dict[section] = input_file_section[section]

    return yaml_dump(yaml_dict, filetype)


def find_sections_in_meta(
    meta_file_data,
    section_name_expressions,
    section_details="<parameters>",
    filetype="rst",
):
    """Returns a string of the given section name expressions from the given dictionary of sections.
    Besides the section names, the section_details can be specified. The default is just to print the string "<parameters>".
    It can take either one regular expression for a section name or a list of those
    """
    if isinstance(section_name_expressions, str):
        section_name_expressions = [section_name_expressions]
    if section_details == "none":
        section_names = []
        for section_name_expression in section_name_expressions:
            reg_expression = re.compile(section_name_expression)
            for section_name in filter(reg_expression.match, meta_file_data.keys()):
                section_names.append(section_name)
    elif section_details == "all":
        section_names = {}
        for section_name_expression in section_name_expressions:
            reg_expression = re.compile(section_name_expression)
            for section_name in filter(reg_expression.match, meta_file_data.keys()):
                section_names[section_name] = meta_file_data[section_name]
    else:
        section_names = {}
        for section_name_expression in section_name_expressions:
            reg_expression = re.compile(section_name_expression)
            section_names = section_names | {
                k: [section_details]
                for k in filter(reg_expression.match, meta_file_data.keys())
            }
    if len(section_names) == 0:
        exit("No sections found for the given regular expressions.")
    return yaml_dump(section_names, filetype)


def convert(template_path, rendering_path, input_file_path):
    global PATH_TO_TESTS
    PATH_TO_TESTS = input_file_path
    target_dir = pathlib.Path(rendering_path)
    template_list = []
    for suffix in ["*.j2", "*.rst", "*.md"]:
        template_list += list(pathlib.Path(template_path).glob(suffix))
    for template_file in template_list:

        try:
            template = jinja2.Template(template_file.read_text())
        except jinja2.exceptions.TemplateSyntaxError as e:
            print(f"Warning: Could not read {template_file}: {e}")
            continue
        docfile_name = (
            template_file.stem if template_file.suffix == ".j2" else template_file
        )
        docfile_fullpath = target_dir / docfile_name
        print(f"source: {docfile_name}, target: {docfile_fullpath}")
        docfile_fullpath.write_text(
            template.render(
                section_dump=section_dump,
                load_input_file=load_input_file,
                yaml_dump=yaml_dump,
                find_sections_in_meta=find_sections_in_meta,
                load_meta_data=load_meta_data,
                add_raw_text_from_file=add_raw_text_from_file,
                len=len,
            )
        )
    for filetype in RECOGNIZED_TYPES:
        for file in pathlib.Path(template_path).glob("*." + filetype):
            shutil.copy(file, target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render all tutorials in the given directory."
    )
    parser.add_argument(
        "template_path", type=str, help="Path to the tutorial templates"
    )
    parser.add_argument(
        "rendering_path", type=str, help="Path to the final tutorial files"
    )
    parser.add_argument("input_file_path", type=str, help="Path to the input files")
    # Parse the arguments
    args = parser.parse_args()

    convert(args.template_path, args.rendering_path, args.input_file_path)
