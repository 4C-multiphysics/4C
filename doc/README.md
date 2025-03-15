# Documentation
On every change of the `main` branch, we automatically build a new version of our general documentation and the Doxygen documentation. Sometimes, it is necessary to build the documentation locally. Here, we give instructions on how to do it.

## Build the documentation
To build the documentation, you need the required dependencies (see the [README](../README.md)) and a `CMakeUserPresets.json` with the configured paths. For simplicity, the `docker` preset is used here.

### General documentation
```
cd <build-dir>
cmake <source-dir> --fresh --preset=docker
cmake --build . --target documentation
```

### Doxygen
```
cd <build-dir>
cmake <source-dir> --fresh --preset=docker -DFOUR_C_BUILD_DOXYGEN="ON"
cmake --build . --target doxygen
```
