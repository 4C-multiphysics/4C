# Docker

This folder contains Dockerfiles for different configurations. The `Dockerfile` in the folder `dependencies` builds the
project with the current dependencies and is also used, via build arguments, for the oldest-supported and
Trilinos-develop dependency images. The other configurations are:

- oldest supported: dependencies image with the oldest supported dependencies (currently only Trilinos), built with
  `TRILINOS_VARIANT=oldest_supported`.
- prebuilt 4C: docker image with ready-to-use 4C executable
- minimal prebuilt 4C: prebuilt 4C docker image with everything removed not need to execute 4C
- trilinos develop: dependencies image with Trilinos develop installation, built with
  `TRILINOS_VARIANT=trilinos_develop`. Currently not working due to Epetra removal.
- kokkos parallel: dependencies image with Trilinos installation with Kokkos OpenMP and Cuda.

The Trilinos build is customized with the build arguments `TRILINOS_VARIANT` (which install script to use),
`TRILINOS_VERSION` (optional commit ref override) and `TRILINOS_ENABLE_DEBUG` (empty uses the variant default).

There are two variants of the Ubuntu 24.04 dependencies image, both built from `docker/dependencies/Dockerfile`: the
release variant (Trilinos without debug assertions, used for performance tests and prebuilt images) and the assertions
variant (`...-assertions`, built with the build argument `TRILINOS_ENABLE_DEBUG=ON`, used for checking that we properly
use Trilinos within CI/CD testing). Since the image names differ, both variants share the same dependencies hash.

## Build docker images locally

Build the Docker image for the dependencies with the following command. **Note** that we need the correct build
context in order to copy the installation scripts. So, run the `docker build` in the projects root folder, e.g. for the
standard Dockerfile:

```bash
cd <project_root>
docker build --tag 4c-dependencies --file docker/dependencies/Dockerfile .
```

For the other variants, pass the corresponding build arguments:

```bash
docker build --tag 4c-dependencies-assertions \
  --build-arg TRILINOS_ENABLE_DEBUG=ON --file docker/dependencies/Dockerfile .
docker build --tag 4c-dependencies-oldest-supported \
  --build-arg TRILINOS_VARIANT=oldest_supported --file docker/dependencies/Dockerfile .
docker build --tag 4c-dependencies-trilinos-develop \
  --build-arg TRILINOS_VARIANT=trilinos_develop \
  --build-arg TRILINOS_VERSION=develop --file docker/dependencies/Dockerfile .
```

## How to update the docker images for the Github Actions

1. Make your changes to the Dockerfile or dependencies
2. Compute the sha1 of the docker and dependencies folder. Make sure that no unwanted files are in these folders.

```bash
cd <project_root>
./docker/dependencies/compute_dependencies_hash.sh
```

3. The kokkosparallel Docker image (used for Kokkos OpenMP and CUDA configurations) has a separate dependency hash, with
   some overlap in the files included in the computation:

```bash
cd <project_root>
./docker/kokkosparallel/compute_dependencies_hash.sh
```

4. Update all explicit mentions of the old hashes by search and replace in all workflows in `.github/workflows`. Keep
   the standard and kokkosparallel hashes separate.
5. Push a branch directly to the 4C repo (not your fork). The branch has to start with `docker-update`, e.g.,
   `docker-update-issue-number-update-some-dependency`.
6. Open a PR (the automatically triggered workflows will fail)
7. Trigger the `Docker` workflow manually on the branch in the 4C repo to build the docker image. If the kokkosparallel
   docker was affected, also trigger the `Docker Kokkos parallel` workflow.
8. Update the branch such that the workflows are triggered again.
