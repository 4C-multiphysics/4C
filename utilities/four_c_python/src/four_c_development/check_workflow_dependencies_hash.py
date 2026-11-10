# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import functools
import subprocess
import os
import sys

from four_c_common_utils.io import load_yaml

# All 4C dependency images are published under this registry prefix. Images outside of
# it are not checked. Images inside it must be registered in DEPENDENCY_IMAGE_HASH_SCRIPTS
# (or use a floating tag), otherwise the check fails.
DEPENDENCY_IMAGE_PREFIX = "ghcr.io/4c-multiphysics/4c-dependencies-"

# Maps the image name (without tag) to the script computing its expected tag.
# The tag of each image must equal the output of that script.
DEPENDENCY_IMAGE_HASH_SCRIPTS = {
    f"{DEPENDENCY_IMAGE_PREFIX}ubuntu24.04": "./docker/dependencies/compute_dependencies_hash.sh",
    f"{DEPENDENCY_IMAGE_PREFIX}ubuntu24.04-assertions": "./docker/dependencies/compute_dependencies_hash.sh",
    f"{DEPENDENCY_IMAGE_PREFIX}ubuntu24.04-assertions-oldest-supported": "./docker/dependencies/compute_dependencies_hash.sh",
    f"{DEPENDENCY_IMAGE_PREFIX}assertions-trilinos-kokkosparallel": "./docker/kokkosparallel/compute_dependencies_hash.sh",
}

# Images tagged with one of these intentionally track a moving image instead of
# a dependencies hash. They are reported but not considered an error.
FLOATING_TAGS = {"develop", "latest", "main", "stable"}

WORKFLOW_PATH = ".github/workflows"


@functools.cache
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


def split_image_reference(full_image_name):
    """Split an image reference without digest into (image_name, tag).

    The tag is an empty string if the reference has no tag.
    """
    tag_separator = full_image_name.rfind(":")
    if tag_separator > full_image_name.rfind("/"):
        return full_image_name[:tag_separator], full_image_name[tag_separator + 1 :]
    return full_image_name, ""


def iter_image_names(workflow):
    """Yield (job label, image name) for all container and service images in a workflow.

    The container can be given as a string shorthand or as a dict with an `image` key.
    """
    for job_name, job in (workflow.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue

        container = job.get("container")
        if isinstance(container, str):
            yield job_name, container
        elif isinstance(container, dict) and isinstance(container.get("image"), str):
            yield job_name, container["image"]

        services = job.get("services")
        if isinstance(services, dict):
            for service_name, service in services.items():
                if isinstance(service, dict) and isinstance(service.get("image"), str):
                    yield f"{job_name}.{service_name}", service["image"]


def check_image(job_name, full_image_name):
    """Return (status, message) for a wrong dependency image, or None if it is fine.

    The status is "error" or "info". Images outside of DEPENDENCY_IMAGE_PREFIX are ignored.
    """
    if "${{" in full_image_name:
        return "info", f"{job_name}: {full_image_name} cannot be checked statically"

    if not full_image_name.startswith(DEPENDENCY_IMAGE_PREFIX):
        return None

    if "@" in full_image_name:
        return (
            "error",
            f"{job_name}: {full_image_name} digest pins are not supported, use a tag",
        )

    image_name, tag = split_image_reference(full_image_name)

    if tag in FLOATING_TAGS:
        return (
            "info",
            f"{job_name}: {full_image_name} intentionally uses a floating tag",
        )

    compute_hash_script = DEPENDENCY_IMAGE_HASH_SCRIPTS.get(image_name)
    if compute_hash_script is None:
        return (
            "error",
            f"{job_name}: {full_image_name} has no hash script, register it in "
            "DEPENDENCY_IMAGE_HASH_SCRIPTS",
        )

    expected_hash = compute_dependencies_hash(compute_hash_script)
    if tag != expected_hash:
        return (
            "error",
            f"{job_name}: {full_image_name} expected tag {expected_hash}, got {tag or 'no tag'}",
        )

    return None


def main():
    # Findings are (status, file, message) tuples, grouped by file when printed.
    findings = []

    for file in sorted(os.listdir(WORKFLOW_PATH)):
        path = os.path.join(WORKFLOW_PATH, file)
        workflow = load_yaml(path)
        if not isinstance(workflow, dict):
            findings.append(("error", path, "not a YAML mapping"))
            continue

        for job_name, full_image_name in iter_image_names(workflow):
            finding = check_image(job_name, full_image_name)
            if finding is not None:
                status, message = finding
                findings.append((status, path, message))

    has_error = False
    for status, title in [
        ("error", "The following checks failed:"),
        ("info", "The following informational notes apply:"),
    ]:
        entries = [
            (file, message) for stat, file, message in findings if stat == status
        ]
        if not entries:
            continue

        has_error = has_error or status == "error"
        print(f"\n{title}")
        current_file = None
        for file, message in entries:
            if file != current_file:
                print(f"In {file}:")
                current_file = file
            print(f" * {message}")

    if has_error:
        print(
            "\nSee DEPENDENCY_IMAGE_HASH_SCRIPTS in check_workflow_dependencies_hash.py "
            "and docker/README.md for how to update the hashes."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
