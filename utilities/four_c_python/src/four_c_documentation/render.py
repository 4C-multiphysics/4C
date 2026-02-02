# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dataclasses import dataclass
from ruamel.yaml import YAML
from four_c_common_utils.io import load_yaml
import pathlib
import shutil
import jinja2
import argparse
import re
import pyvista as pv
import json
import hashlib
import functools
import os

RECOGNIZED_TYPES = ["rst", "md"]


@dataclass
class Context:
    template_path: pathlib.Path
    base_rendering_path: pathlib.Path
    subdirectory: str
    input_file_path: pathlib.Path
    filetype: str


def hash_options(*args, **kwargs):
    data = json.dumps(
        {
            "args": [hash(id(arg)) for arg in args],
            "kwargs": {hash(id(key)): hash(id(value)) for key, value in kwargs.items()},
        },
        sort_keys=True,
    )
    return hashlib.sha256(data.encode()).hexdigest()


class PlotBuilder:

    def __init__(self, context: Context, aspect_ratio: float = 16 / 9):
        self.plotter = pv.Plotter()
        self.hash = "empty"
        self.context = context
        self.aspect_ratio = aspect_ratio

    def __getattr__(self, name):
        attr = getattr(self.plotter, name)

        assert callable(attr), f"Attribute {name} is not callable on plotter."

        # return a wrapper that calls the plotter method and returns self
        def method(*args, **kwargs):
            attr(*args, **kwargs)

            # we need to update the hash with the options used for this method call
            self.hash = hash_options(self.hash, name, *args, **kwargs)
            return self

        return method

    def make(self):
        # export as a standalone html file
        relative_path = f"plot_{self.hash}.html"

        output_file = (
            self.context.base_rendering_path
            / "_static"
            / self.context.subdirectory
            / relative_path
        )

        os.makedirs(output_file.parent, exist_ok=True)
        self.plotter.export_html(str(output_file))
        print("Generated plot at " + str(output_file))

        num_subdirs = len(self.context.subdirectory.split("/"))

        embedded_html = f'<div style="aspect-ratio: {self.aspect_ratio};"><iframe src="{"../"*num_subdirs}_static/{self.context.subdirectory}/{relative_path}" width="100%" height="100%" style="border:none;"></iframe></div>'

        if self.context.filetype == "rst":
            return f".. raw:: html\n\n   {embedded_html}"
        elif self.context.filetype == "md":
            return f"```{{raw}} html\n{embedded_html}\n```"

        raise ValueError(f"Unsupported filetype: {self.context.filetype}")


def load_input_file(context: Context, yaml_file):
    """Returns the dictionary of parameters for the given yaml file."""
    # load input files with ruamel.yaml to preserve comments
    data = YAML(typ="safe").load((context.input_file_path / yaml_file).read_text())
    return data


def add_raw_text_from_file(context: Context, text_file, indent=0):
    """Returns the dictionary of parameters for the given yaml file."""
    with open(context.input_file_path / pathlib.Path(text_file), "r") as file:
        input = file.readlines()
    if indent > 0:
        indent_string = " " * indent
        input = "".join([indent_string + line for line in input])
    else:
        input = "".join(input)
    return input


def load_meta_data(context: Context):
    """Returns a dictionary of sections and their parameters from the meta file 4C_metadata.yaml."""
    metafile_data = load_yaml(context.input_file_path / ("../../4C_metadata.yaml"))
    sections = [section["name"] for section in metafile_data["sections"]["specs"]]
    sections += metafile_data["legacy_string_sections"]

    return sections


def yaml_dump(context: Context, data):
    """Returns a string of the yaml file in the given filetype.
    As of now I can return markdown and restructuredText code blocks.
    """
    yaml = YAML(typ=["safe", "string"])
    if context.filetype == "rst":
        rststring = ""
        yaml_data = yaml.dump_to_string(data).split("\n")
        for line in yaml_data:
            rststring += "    " + line + "\n"
        rststring += "\n"
        return ".. code-block:: yaml\n\n" + rststring + "\n"
    elif context.filetype == "md":
        return "```yaml\n" + yaml.dump_to_string(data) + "\n```\n"
    else:
        raise TypeError(
            f"Filetype {context.filetype} for yaml_dump cannot be recognized yet."
        )


def section_dump(context: Context, input_file_section, section_names):
    """Returns a string of the given sections from the given dictionary of sections.
    The parameter section_names can be either a string (one section name) or a list of section names
    """
    if isinstance(section_names, str):
        section_names = [section_names]
    yaml_dict = {}
    for section in section_names:
        yaml_dict[section] = input_file_section[section]

    return yaml_dump(context, yaml_dict)


def find_sections_in_meta(
    context: Context,
    meta_file_data,
    section_name_expressions,
):
    """Returns a string of the given section name expressions from the given dictionary of sections.
    It can take either one regular expression for a section name or a list of those
    """
    if isinstance(section_name_expressions, str):
        section_name_expressions = [section_name_expressions]
    section_names = []
    for section_name_expression in section_name_expressions:
        reg_expression = re.compile(section_name_expression)

        section_names += filter(reg_expression.match, meta_file_data)
    if len(section_names) == 0:
        exit("No sections found for the given regular expressions.")
    return yaml_dump(context, section_names)


def plot(context: Context, input_file: str, aspect_ratio: float = 16 / 9, **kwargs):
    mesh = pv.read(context.input_file_path / pathlib.Path(input_file))
    return PlotBuilder(context, aspect_ratio).add_mesh(mesh, **kwargs).make()


def plotter(context: Context, aspect_ratio: float = 16 / 9):
    return PlotBuilder(context, aspect_ratio)


def render(template_path, base_rendering_path, subdirectory, input_file_path):
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
        ext = pathlib.Path(docfile_name).suffix[1:]
        assert ext == "rst" or ext == "md", f"File extension {ext} not recognized."

        context = Context(
            pathlib.Path(template_path),
            pathlib.Path(base_rendering_path),
            subdirectory,
            pathlib.Path(input_file_path),
            filetype=ext,
        )

        output_dir = context.base_rendering_path / context.subdirectory

        docfile_fullpath = output_dir / docfile_name
        print(f"source: {docfile_name}, target: {docfile_fullpath}")
        docfile_fullpath.write_text(
            template.render(
                section_dump=functools.partial(section_dump, context),
                load_input_file=functools.partial(load_input_file, context),
                yaml_dump=functools.partial(yaml_dump, context),
                find_sections_in_meta=functools.partial(find_sections_in_meta, context),
                load_meta_data=functools.partial(load_meta_data, context),
                add_raw_text_from_file=functools.partial(
                    add_raw_text_from_file, context
                ),
                len=len,
                plot=functools.partial(plot, context),
                plotter=functools.partial(plotter, context),
                read_mesh=lambda input_file: pv.read(
                    context.input_file_path / pathlib.Path(input_file)
                ),
            )
        )

    for filetype in RECOGNIZED_TYPES:
        for file in pathlib.Path(template_path).glob("*." + filetype):
            shutil.copy(file, pathlib.Path(base_rendering_path) / subdirectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render all tutorials in the given directory."
    )
    parser.add_argument(
        "template_path", type=str, help="Path to the tutorial templates"
    )
    parser.add_argument(
        "base_rendering_path", type=str, help="Path to the base rendering directory"
    )
    parser.add_argument(
        "subdirectory", type=str, help="Subdirectory for the rendered files"
    )
    parser.add_argument("input_file_path", type=str, help="Path to the input files")
    # Parse the arguments
    args = parser.parse_args()

    render(
        args.template_path,
        args.base_rendering_path,
        args.subdirectory,
        args.input_file_path,
    )
