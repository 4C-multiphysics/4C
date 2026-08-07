# Docker

This folder contains Dockerfiles for different configurations. The `Dockerfile` in the folder `dependencies` is the Dockerfile to
build the project with the current dependencies. Other configurations reside in different subfolders.

## Build docker images locally

Build the Docker image for the dependencies with the following command. **Note** that we need the correct build
context in order to copy the installation scripts. So, run the `docker build` in the projects root folder, e.g. for the standard Dockerfile:

```bash
cd <project_root>
docker build --tag 4c-dependencies --file docker/dependencies/Dockerfile .
```

## How to update the docker images for the Github Actions

1. Make your changes to the Dockerfile or dependencies
2. Compute the sha1 of the docker and dependencies folder.

```bash
cd <project_root>
./docker/dependencies/compute_dependencies_hash.sh
```

3. The kokkosparallel Docker image (used for Kokkos OpenMP and CUDA configurations) has a separate dependency hash, with some overlap in the files included in the computation:

```bash
cd <project_root>
./docker/kokkosparallel/compute_dependencies_hash.sh
```

4. Update all explicit mentions of the old hashes by search and replace in all workflows in `.github/workflows`. Keep the standard and kokkosparallel hashes separate.
5. Push a branch directly to the 4C repo (not your fork). The branch has to start with `docker-update`, e.g., `docker-update-issue-number-update-some-dependency`.
6. Open a PR (the automatically triggered workflows will fail)
7. Trigger the `Docker` workflow manually on the branch in the 4C repo to build the docker image. If the kokkosparallel docker was affected, also trigger the `Docker Kokkos parallel` workflow.
8. Update the branch such that the workflows are triggered again.
