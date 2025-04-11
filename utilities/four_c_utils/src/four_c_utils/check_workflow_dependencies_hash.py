# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import subprocess
import os
import yaml
import sys


def compute_dependencies_hash():
    result = subprocess.run(
        ["./docker/dependencies/compute_dependencies_hash.sh"],
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError("Failed to compute dependencies hash")

    return result.stdout.decode("utf-8").strip()


def main():
    dependencies_hash = compute_dependencies_hash()

    # loop over all workflows and check dependencies hash
    jobs_with_wrong_dependency_hash = {}
    workflow_path = ".github/workflows"
    for file in os.listdir(workflow_path):
        with open(os.path.join(workflow_path, file), "r") as f:
            workflow = yaml.safe_load(f)

            for job_name, job in workflow["jobs"].items():
                if "container" in job and "image" in job["container"]:
                    image_name = job["container"]["image"]
                    if image_name.startswith(
                        "ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04"
                    ) and not image_name.endswith(f":{dependencies_hash}"):
                        if file not in jobs_with_wrong_dependency_hash:
                            jobs_with_wrong_dependency_hash[file] = []

                        jobs_with_wrong_dependency_hash[file].append(job_name)

    if len(jobs_with_wrong_dependency_hash) > 0:
        print(f"Expecting all jobs to have the dependencies hash {dependencies_hash}.")
        print("")
        print("The following jobs do not have the correct hash:")
        for workflow_name, job_names in jobs_with_wrong_dependency_hash.items():
            print(f"In {os.path.join(workflow_path, workflow_name)}:")
            for job_name in job_names:
                print(f" * {job_name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
