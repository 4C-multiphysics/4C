This folder contains the dependencies required to build the project. The current dependencies are in `current/`. To
install the dependencies just run

```
# Define the number of procs for building or use the default (4 procs)
export NPROCS=12
# Install each dependency
./current/<dependency>/install.sh <path/to/install/dir>
```

Common Trilinos CMake options are shared via `common/trilinos_options.sh`. The install scripts in `oldest_supported/` and
`trilinos_develop/` are thin wrappers that reuse `current/trilinos/install.sh` with a different version and assertion
default. The install scripts in `darwin/` and `kokkosparallel/` source the common options but add platform- or
backend-specific flags.

The Trilinos commit ref and debug assertions can be controlled with the environment variables
`TRILINOS_VERSION` and `TRILINOS_ENABLE_DEBUG`, e.g.

```
TRILINOS_VERSION=develop TRILINOS_ENABLE_DEBUG=ON ./current/trilinos/install.sh <path/to/install/dir>
```

If unset, each script falls back to its pinned default (the most recent supported version, except for
`oldest_supported/`, which uses the oldest supported version).
