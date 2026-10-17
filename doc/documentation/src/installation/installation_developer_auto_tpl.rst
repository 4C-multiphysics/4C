Set up 4C with automatic TPL installation
==========================================

This path is for contributors who work on 4C Multiphysics from a Git checkout and want to use an IDE.

Create a contributor checkout
-----------------------------

Fork the `4C repository <https://github.com/4C-multiphysics/4C>`_, then clone your fork. Use the fork as ``origin``
and configure the central repository as ``upstream``. Details about forks and cloning can be found in the
`GitHub Docs <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_.
Set the local ``main`` branch to track ``upstream/main``::

    git branch main --set-upstream-to=upstream/main

Set up the Python development environment and pre-commit hooks once in the source directory::

    ./utilities/set_up_dev_env.sh <optional-path-to-python-executable>

Install external dependencies with Spack
----------------------------------------

Spack automatically resolves and installs the 4C dependency stack while leaving the source checkout under your
control. It records the selected compiler, MPI implementation, package versions, and variants in ``spack.lock``.

Starting with no existing Spack installation, run the following commands in Bash to clone Spack and initialize it::

    git clone https://github.com/spack/spack.git
    spack_root="$PWD/spack"
    . "$spack_root/share/spack/setup-env.sh"
    printf '. "%s/share/spack/setup-env.sh"\n' "$spack_root" >> ~/.bashrc

.. note::

    The ``printf`` command enables Spack automatically in future Bash sessions.

.. note::

    For other shells, use the corresponding initialization instructions in `Spack's shell setup instructions
    <https://spack.readthedocs.io/en/latest/getting_started.html>`_.

Clone and register the 4C Spack package repository::

    git clone https://github.com/spack/spack-packages.git
    spack repo add spack-packages/repos/spack_repo/builtin

Create a checkout-local environment and install the dependencies, but not 4C itself::

    cd <4C_sourceDir>
    spack env create --dir .spack
    spack env activate .spack
    spack add '4c-multiphysics@main %gcc@13.3.0'
    spack concretize
    spack install --only dependencies -j <number-of-processes>
    ./utilities/python-venv/bin/generate-spack-cmake-presets

The ``%gcc@13.3.0`` constraint is an example. Use ``spack compiler find`` to register compilers available on your
machine, then use ``spack compiler list`` to select the compiler and version for the ``spack add`` specification.

The dependency-only installation deliberately excludes the 4C package and its source resources. During CMake
configuration, |FOURC| downloads its ryml source; a configuration with ``+mirco`` also downloads the MIRCO source.

The preset generator reads the concretized 4C Spack specification and writes matching CMake feature options to
``.spack/4CSpackPresets.json``. It also creates a minimal ``CMakeUserPresets.json`` that includes the generated file
when no user preset file exists. An existing ``CMakeUserPresets.json`` is never modified; add the following entry to
its top-level object if needed::

    {
        "version": 5,
        "include": [".spack/4CSpackPresets.json"]
    }

The generated ``spack-generated-debug`` and ``spack-generated-release`` presets use the dependency view at
``.spack/.spack-env/view``. No environment variables need to be set after the one-time installation. When changing
Spack variants, run ``spack concretize -f``, ``spack install --only dependencies``, and
``./utilities/python-venv/bin/generate-spack-cmake-presets`` again, then use fresh CMake build directories. Create
personal presets by inheriting from the generated presets.

Full-featured development build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable all optional 4C capabilities supported by the Spack package, replace the base ``spack add`` command above
with the following command when creating the environment::

    spack add '4c-multiphysics@main+arborx+backtrace+mirco+dealii+vtk+gmsh+fftw+python+pybind11 %gcc@13.3.0'

After concretizing and installing the environment, rerun
``./utilities/python-venv/bin/generate-spack-cmake-presets``. It enables the matching CMake capabilities
automatically.

To select only some optional capabilities, add the required ``+<variant>`` tokens to the 4C package name before the
compiler constraint. For example, to enable VTK output and the Gmsh interface, use::

    spack add '4c-multiphysics@main+vtk+gmsh %gcc@13.3.0'

Run ``spack info 4c-multiphysics`` to list the variants supported by the registered package, including variants added
in later package updates. Rerun the preset generator after changing variants so the generated CMake presets match.
See :doc:`the simulation-user installation guide <installation_user>` for the current capability descriptions.

Configure and build 4C using Spack
----------------------------------

List the available presets and configure the debug build from the source directory::

    cmake --list-presets
    cmake --preset spack-generated-debug

The preset creates its configured build directory automatically. Continue with the shared
:doc:`configuration, build, and IDE workflow <installation_developer_workflow>` and select the
``spack-generated-debug`` preset in your IDE.