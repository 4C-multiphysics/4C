# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import argparse
import datetime
import requests
import os
import io
import zipfile
import json
import dataclasses


@dataclasses.dataclass
class Config:
    repo: str
    workflow_id: str


def get_headers(ci_token: str) -> dict:
    """
    Generate the headers for API requests, including the authorization token.
    """
    return {"Authorization": f"Bearer {ci_token}"}


def download_and_extract_json(url: str, ci_token: str):
    """
    Downloads the artifact zip file into memory, finds the JSON file,
    and parses it directly without saving the zip to disk.
    """
    response = requests.get(url, headers=get_headers(ci_token), timeout=10)
    response.raise_for_status()

    # Artifact is zipped on GitHub
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find your specific JSON file in the archive
        for filename in z.namelist():
            if filename.endswith(".json"):  # Or specify the exact filename
                with z.open(filename) as f:
                    return json.load(f)

    raise FileNotFoundError("No JSON file found in the artifact zip.")


def get_workflow_runs(
    config: Config, last_fetched_id: int, ci_token: str
) -> list[dict]:
    """Fetches successful workflow runs from a specific date onwards."""
    print(
        f"Fetching workflow runs for '{config.workflow_id}' since id {last_fetched_id}..."
    )

    url = f"https://api.github.com/repos/{config.repo}/actions/workflows/{config.workflow_id}/runs"
    # GitHub API allows filtering by creation date and status
    params = {
        "status": "success",
        "per_page": 100,
    }

    runs = []

    # Handle pagination in case there are hundreds of runs
    while url:
        response = requests.get(
            url, headers=get_headers(ci_token), params=params, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        page_runs = data.get("workflow_runs", [])
        for run in page_runs:
            if run["id"] <= last_fetched_id:
                print(
                    f"Reached previously fetched run ID ({run['id']}). Stopping search."
                )
                return runs

            runs.append(run)

        # Check if there is a 'next' page
        url = response.links.get("next", {}).get("url")
        params = None  # Params are already included in the 'next' URL

    return runs


def get_target_artifact_url(
    config: Config, artifact_name: str, run_id: int, ci_token: str
) -> str | None:
    """Finds the download URL for the specific artifact in a given run."""
    url = f"https://api.github.com/repos/{config.repo}/actions/runs/{run_id}/artifacts"

    params = {
        "per_page": 100,
    }
    while url:
        response = requests.get(
            url, headers=get_headers(ci_token), params=params, timeout=10
        )
        response.raise_for_status()

        artifacts = response.json().get("artifacts", [])

        for artifact in artifacts:
            if artifact["name"] == artifact_name:
                return artifact["archive_download_url"]

        # Check if there is a 'next' page
        url = response.links.get("next", {}).get("url")
        params = None  # Params are already included in the 'next' URL
    return None


def translate_old_performance_data_format(entry: dict) -> None:
    context = {"date": entry["date"]}

    for group_name, group in entry["data"].items():
        data = []
        for name, timing_data in group["Total times"].items():
            data.append(
                {
                    "name": name,
                    "unit": group["Time unit"],
                    "measurements": {
                        "min": timing_data["MinOverProcs"],
                        "max": timing_data["MaxOverProcs"],
                        "mean": timing_data["MeanOverProcs"],
                    },
                }
            )

        entry["data"][group_name] = {
            "context": context,
            "data": data,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Collect performance data for four_c_ci"
    )
    parser.add_argument(
        "repository", type=str, help="GitHub repository in owner/repo format"
    )
    parser.add_argument("workflow_name", type=str, help="Name of the workflow")
    parser.add_argument(
        "--artifact-name", type=str, required=True, help="Name of the artifact to fetch"
    )
    parser.add_argument(
        "--previous-data",
        type=str,
        help="Path to the previous performance data file.",
    )
    parser.add_argument(
        "--export-to",
        type=str,
        help="Path to the file where the collected data will be exported.",
    )
    parser.add_argument(
        "--expire-days",
        type=int,
        default=None,
        help="Number of days after which to expire old data.",
    )
    args = parser.parse_args()

    old_data = []
    # fill with all previous data
    if args.previous_data:
        with open(args.previous_data, "r") as f:
            old_data = json.load(f)

    # We might need to update the old data and convert it to the our new format (generalized data format, not the Trilinos format)
    if len(old_data) > 0:
        is_legacy = False
        for group in old_data[0]["data"].values():
            if "Output mode" in group:
                is_legacy = True
                break

        if is_legacy:
            print("Upgrading previous performance data to new format...")
            for entry in old_data:
                translate_old_performance_data_format(entry)

    largest_id = max((entry["run_id"] for entry in old_data), default=0)

    ci_token = os.getenv("GITHUB_TOKEN")
    config = Config(repo=args.repository, workflow_id=args.workflow_name)
    artifact_name = args.artifact_name

    if not ci_token:
        parser.error("GITHUB_TOKEN environment variable must be set")
    workflow_runs = get_workflow_runs(
        config,
        largest_id,
        ci_token,
    )

    # get all new data from nightly pipeline
    new_data = []
    for run in workflow_runs:
        run_id = run["id"]
        run_date = datetime.datetime.fromisoformat(
            run["created_at"].replace("Z", "+00:00"),
        )
        run_sha = run["head_sha"]
        print(f"Processing run {run_id} from {run_date}...")
        artifact_url = get_target_artifact_url(
            config,
            artifact_name,
            run_id,
            ci_token,
        )

        if artifact_url is None:
            print(f"Artifact '{artifact_name}' not found for run {run_id}")
            continue

        data = download_and_extract_json(artifact_url, ci_token)

        new_data.append(
            {
                "run_id": run_id,
                "date": run_date.isoformat(),
                "sha": run_sha,
                "data": data,
            },
        )

    if args.expire_days is not None:
        expire_threshold = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(days=args.expire_days)
        old_data = [
            entry
            for entry in old_data
            if datetime.datetime.fromisoformat(entry["date"]) >= expire_threshold
        ]

    if args.export_to:
        with open(args.export_to, "w") as f:
            json.dump(old_data + list(reversed(new_data)), f)


if __name__ == "__main__":
    main()
