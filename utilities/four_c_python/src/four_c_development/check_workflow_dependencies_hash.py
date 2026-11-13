# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import subprocess
import os
import sys

from four_c_common_utils.io import load_yaml

# All dependency images are published under this registry namespace.
DEPENDENCY_IMAGE_NAMESPACE = "ghcr.io/4c-multiphysics/4c-dependencies-"

# Dependency images that are built from distinct dependency sets and hence need
# their own hash computation script. The image name (without tag) must match a key
# exactly and its tag must equal the computed hash.
DEPENDENCY_IMAGE_HASH_SCRIPTS = {
    "ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04": "./docker/dependencies/compute_dependencies_hash.sh",
    "ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04-assertions": "./docker/dependencies/compute_dependencies_hash.sh",
    "ghcr.io/4c-multiphysics/4c-dependencies-ubuntu24.04-assertions-oldest-supported": "./docker/dependencies/compute_dependencies_hash.sh",
    "ghcr.io/4c-multiphysics/4c-dependencies-assertions-trilinos-kokkosparallel": "./docker/kokkosparallel/compute_dependencies_hash.sh",
}

# Images tagged with one of these intentionally track a moving image instead of
# a dependencies hash. They are reported but not considered an error.
FLOATING_TAGS = {"develop", "latest", "main", "stable"}


def compute_dependencies_hash(compute_hash_script):
    result = subprocess.run(
        [compute_hash_script],
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compute dependencies hash with {compute_hash_script}"
        )

    return result.stdout.decode("utf-8").strip()


def image_repository(image_name):
    """Return the image name without its tag or digest."""
    name = image_name.split("@", 1)[0]
    tag_separator = name.rfind(":")
    if tag_separator > name.rfind("/"):
        name = name[:tag_separator]
    return name


def image_tag(image_name):
    """Return the image tag, or an empty string if the image has no tag."""
    name = image_name.split("@", 1)[0]
    tag_separator = name.rfind(":")
    if tag_separator > name.rfind("/"):
        return name[tag_separator + 1 :]
    return ""


def find_hash_script(image_name):
    return DEPENDENCY_IMAGE_HASH_SCRIPTS.get(image_repository(image_name))


def iter_image_names(workflow):
    """Yield (job label, image name) for all container and service images in a workflow."""
    for job_name, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue

        container = job.get("container")
        if isinstance(container, dict) and "image" in container:
            yield job_name, container["image"]

        services = job.get("services")
        if isinstance(services, dict):
            for service_name, service in services.items():
                if isinstance(service, dict) and "image" in service:
                    yield f"{job_name}.{service_name}", service["image"]


def main():
    hash_cache = {}

    # file -> list of (job label, image name, expected hash)
    jobs_with_wrong_dependency_hash = {}
    # file -> list of (job label, image name)
    jobs_with_unmapped_image = {}
    # file -> list of (job label, image name, tag)
    jobs_with_floating_image = {}

    workflow_path = ".github/workflows"
    for file in sorted(os.listdir(workflow_path)):
        workflow = load_yaml(os.path.join(workflow_path, file))
        if not isinstance(workflow, dict):
            continue

        for job_name, image_name in iter_image_names(workflow):
            if not isinstance(image_name, str) or not image_name.startswith(
                DEPENDENCY_IMAGE_NAMESPACE
            ):
                continue

            tag = image_tag(image_name)

            if tag in FLOATING_TAGS:
                jobs_with_floating_image.setdefault(file, []).append(
                    (job_name, image_name)
                )
                continue

            compute_hash_script = find_hash_script(image_name)
            if compute_hash_script is None:
                jobs_with_unmapped_image.setdefault(file, []).append(
                    (job_name, image_name)
                )
                continue

            if compute_hash_script not in hash_cache:
                hash_cache[compute_hash_script] = compute_dependencies_hash(
                    compute_hash_script
                )
            expected_hash = hash_cache[compute_hash_script]

            if tag != expected_hash:
                jobs_with_wrong_dependency_hash.setdefault(file, []).append(
                    (job_name, image_name, expected_hash)
                )

    if jobs_with_floating_image:
        print("The following jobs intentionally use a floating image tag:")
        for workflow_name, entries in jobs_with_floating_image.items():
            print(f"In {os.path.join(workflow_path, workflow_name)}:")
            for job_name, image_name in entries:
                print(f" * {job_name}: {image_name}")
        print("")

    has_error = False

    if jobs_with_wrong_dependency_hash:
        has_error = True
        print("The following jobs do not have the correct dependencies hash:")
        for workflow_name, entries in jobs_with_wrong_dependency_hash.items():
            print(f"In {os.path.join(workflow_path, workflow_name)}:")
            for job_name, image_name, expected_hash in entries:
                print(
                    f" * {job_name}: expected tag {expected_hash} but got {image_name}"
                )

    if jobs_with_unmapped_image:
        has_error = True
        if jobs_with_wrong_dependency_hash:
            print("")
        print(
            "The following jobs use a dependency image without a known hash computation script:"
        )
        for workflow_name, entries in jobs_with_unmapped_image.items():
            print(f"In {os.path.join(workflow_path, workflow_name)}:")
            for job_name, image_name in entries:
                print(f" * {job_name}: {image_name}")
        print("")
        print(
            "Add the image to DEPENDENCY_IMAGE_HASH_SCRIPTS in "
            "check_workflow_dependencies_hash.py if it needs a hash check."
        )

    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
