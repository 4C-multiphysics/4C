# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import jsonschema_rs
import json
import pathlib
from four_c_metadata.utils import load_yaml


def cli():
    parser = argparse.ArgumentParser(
        description="Validate files against a JSON schema file.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        help="Path to the schema file to validate against.",
    )
    parser.add_argument(
        "yaml_files",
        nargs="+",
        type=str,
        help="Path(s) to the YAML file(s) to validate.",
    )

    args = parser.parse_args()

    json_schema = json.loads(pathlib.Path(args.schema).read_text())
    validator = jsonschema_rs.validator_for(json_schema)

    def format_dotted(left: str, right: str) -> str:
        dots = "." * (300 - len(left) - len(right))
        return f"{left}{dots}{right}"

    files_with_errors = []
    for yaml_file in args.yaml_files:
        try:
            data = load_yaml(yaml_file)
            validator.validate(data)
            print(format_dotted(yaml_file, "passed"))

        except jsonschema_rs.ValidationError as e:
            print(format_dotted(yaml_file, "failed"))
            print(f"{e}")
            files_with_errors.append(yaml_file)
        except Exception as e:
            print(format_dotted(yaml_file, "error"))
            print(f"{e}")

            files_with_errors.append(yaml_file)

    if files_with_errors:
        print(f"\nThe following {len(files_with_errors)} files failed validation:")
        for file in files_with_errors:
            print(f" - {file}")
        exit(1)
    else:
        print("\nAll files passed validation.")


if __name__ == "__main__":
    cli()
