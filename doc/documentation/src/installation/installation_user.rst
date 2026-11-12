Install 4C for simulations
==========================

This path is for users who want to run |FOURC| without maintaining a source checkout.

Run with Docker
---------------

The ready-to-use Docker image contains a pre-built |FOURC| executable and all optional dependencies compatible with
the 4C license. Install Docker according to the `Docker documentation <https://docs.docker.com/get-docker/>`_, then
start an interactive container::

    docker run --interactive --tty ghcr.io/4c-multiphysics/4c:main

The image is downloaded automatically on its first use. Run one of the bundled input files from inside the container::

    /home/user/4C/build/4C /home/user/4C/tests/input_files/<some-input-file>.4C.yaml output_name

To use local input files and retain output files, mount a local working directory::

    docker run --interactive --tty --mount type=bind,source="$PWD",target=/work ghcr.io/4c-multiphysics/4c:main
    /home/user/4C/build/4C /work/<input-file>.4C.yaml /work/output_name

Install with Spack
------------------

For a native installation of a released 4C version, Spack automatically resolves, builds, and installs 4C together
with all external dependencies. Starting with no existing Spack installation, clone Spack and initialize its shell
support::

    git clone https://github.com/spack/spack.git
    spack_root="$PWD/spack"
    . "$spack_root/share/spack/setup-env.sh"
    printf '. "%s/share/spack/setup-env.sh"\n' "$spack_root" >> ~/.bashrc

Clone and register the 4C Spack package repository::

    git clone https://github.com/spack/spack-packages.git
    spack repo add spack-packages/repos/spack_repo/builtin

The command above enables Spack automatically in future Bash sessions. See the
`Spack documentation <https://spack.readthedocs.io/>`_ for initialization of other shells and configuration options.

List the available 4C releases and install the required version::

    spack info 4c-multiphysics
    spack install 4c-multiphysics@<release-version> -j <number-of-processes>

The ``-j`` option sets the number of parallel build processes per package.

The installation can be made available in the current shell with::

    spack load 4c-multiphysics@<release-version>

Use ``spack find 4c-multiphysics`` to list installed versions. Select optional capabilities or a specific compiler by
adding Spack variants or a compiler constraint to the installation specification; see ``spack info 4c-multiphysics``
for the supported options.

Select optional capabilities
----------------------------

The base Spack specification provides the standard 4C functionality, including Qhull support. Add a ``+<variant>``
only when your simulations or workflow need the corresponding capability:

- ``+arborx``: required for contact-based beam-interaction simulations.
- ``+backtrace``: enables stack traces in runtime error messages.
- ``+dealii``: required for simulations that use the deal.II coupling or discretization functionality.
- ``+fftw``: required for simulations that use FFTW-based functionality.
- ``+gmsh``: required to use the Gmsh interface.
- ``+vtk``: required to use VTK-based output or visualization functionality.
- ``+mirco``: required for rigid rough-indenter contact using MIRCO.
- ``+python``: installs the Python-based utilities, tests, and documentation dependencies.
- ``+pybind11``: builds the py4C Python bindings; it also requires ``+python``.

For example, install and load release ``2026.3.0`` with VTK and Gmsh support using::

    spack install 4c-multiphysics@2026.3.0+vtk+gmsh
    spack load 4c-multiphysics@2026.3.0+vtk+gmsh

Spack release versions include the patch number, such as ``2026.3.0``.

Install only the variants that you need; optional variants can substantially increase installation time and dependency
count. When in doubt, begin with the base specification and add a variant when a tutorial, input file, or workflow
requires it.