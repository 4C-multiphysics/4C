Set up 4C with manual TPL installation
=======================================

This path is for contributors who work on 4C Multiphysics from a Git checkout and install external dependencies
manually.

|FOURC| is developed and used on Linux but other Unixes might work as well.
Additionally, |FOURC| can be compiled under Windows using the WSL (Windows Subsystem for Linux),
which provides a complete Linux Environment within the Windows operating system.

.. _external-dependencies:

External dependencies
---------------------

This page describes manual dependency installation.

|FOURC| requires the following general tools to write parallel C++ code:

- git
- C++ compiler with C++20 compatibility (e.g. gcc 13, clang 18)
- MPI
- CMake
- Ninja

|FOURC| is built on top of other powerful external dependencies.
The top-level directory ``dependencies`` contains information about these dependencies.
The following list shows the most important ones:

External solver and linear algebra:

- :ref:`Trilinos <trilinos>` (supported versions are listed in ``dependencies/supported_version/Trilinos.txt``, currently only supported version: 16.2)
- :ref:`SuiteSparse <suitesparse>` (optional)
- :ref:`SuperLUDist <superludist>` (optional, recommended version: 9.2.1)
- MUMPS (optional)
- ScaLAPACK (as a dependency of MUMPS)
- BLAS
- LAPACK

Graph and domain partitioner:

- Metis (recommended version: 5.1.0, included in SuiteSparse)
- ParMETIS (recommended version: 4.0.3)

Miscellaneous:

- Boost
- CLN
- HDF5
- magic_enum
- ryml
- zlib


Optional:

- :ref:`ArborX <arborx>`
- Backtrace
- dealii
- FFTW
- :ref:`MIRCO <mirco>`
- :ref:`Qhull <qhull>`


Build information
~~~~~~~~~~~~~~~~~

For many external dependencies, you'll find an installation file for the recommended version in the ``<4C_sourceDir>/dependencies/current`` directory.
Additionally, some dependencies are accepted in multiple versions. During configuration, |FOURC| will check the version of the installed dependency
and warn you if it is not a supported version.

.. _suitesparse:

**SuiteSparse**

|FOURC| uses SuiteSparse indirectly via the Trilinos package Amesos2 for directly solving linear systems of equations.
See the `SuiteSparse repository <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_ for details and downloads.

On Linux based systems the package can be installed using ``sudo apt install libsuitesparse-dev``.

.. _superludist:

**SuperLUDist**

|FOURC| uses SuperLUDist indirectly via the Trilinos package Amesos/Amesos2 for directly solving linear systems of equations in distributed memory fashion.
See the `superLU repository <https://github.com/xiaoyeli/superlu_dist>`_ for details and downloads.

.. _arborx:

**ArborX**

ArborX can be used as optional dependency inside |FOURC| for utilizing it's tree-based search algorithms.
See the `ArborX repository <https://github.com/arborx/ArborX>`_ for details and downloads.

Building |FOURC| with ArborX enabled automatically fetches the repository during the configure stage and later builds the library as dependency.

Note: For contact based beaminteraction simulations ArborX is a mandatory dependency.

.. _trilinos:

**Trilinos**

This external dependency can be downloaded from the `Trilinos Github repository <https://github.com/trilinos/Trilinos>`_ .
Currently supported versions are listed in ``<4C_sourceDir>/dependencies/supported_version/Trilinos.txt``. An older supported version will be supported for at least six months after its introduction. Afterwards, the version may be dropped at any time.

.. note:: As 4C is still depending on Epetra based Trilinos code, the last officially supported version is release 16.2.2.

.. _mirco:

**MIRCO**

MIRCO can be used as optional dependency inside |FOURC| to be used for linear elastic frictionless normal contact between a rigid rough indentor and an elastic half-space.
See the `MIRCO repository <https://github.com/imcs-compsim/MIRCO>`_ for details and downloads.

Building |FOURC| with MIRCO enabled automatically fetches the repository during the configure stage and later builds the library as dependency. Alternatively, an external MIRCO installation can be specified. In either case, MIRCO can make use of shared memory parallelism through Kokkos :ref:`when enabled <build4Cwithkokkosparallel>` in |FOURC|. Note that |FOURC| and MIRCO must depend on the same Kokkos installation. When using Kokkos with CUDA enabled, MIRCO must be built with ``CMAKE_POSITION_INDEPENDENT_CODE=ON``.

.. _qhull:

**Qhull**

Qhull can be used as optional dependency inside |FOURC| to calculate the convex hull of a set of points. This is mainly used for finite element procedures using cut operations.
See the `Qhull repository <https://github.com/qhull/qhull>`_ for details and downloads.

Building |FOURC| with Qhull enabled assumes that a version is already pre-installed.

On Linux based systems the package can be installed using ``sudo apt install libqhull-dev``.


.. _4Cinstallation:

Create a contributor checkout
-----------------------------

Fork the `4C repository <https://github.com/4C-multiphysics/4C>`_, then clone your fork. Use the fork as ``origin`` and
configure the central repository as ``upstream``. Details about forks and cloning can be found in the
`GitHub Docs <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_.

Set the local ``main`` branch to track ``upstream/main``::

    git branch main --set-upstream-to=upstream/main

Further, you need to create a python virtual environment for development.
In the source directory, execute::

    ./utilities/set_up_dev_env.sh <optional-path-to-python-executable>

If the optional path to the python executable is not given, the script will use ``python3`` as the default. If your
``python3`` is too old or not available, you can specify the path to a compatible python version. The script will let
you know if the python version is not compatible.

This script installs `pre-commit <https://pre-commit.com/>`_ and sets up the pre-commit hooks for the repository.

.. note::

    You only need to execute this command once.
    However, when changes in the virtual python environment have been made, you **must** generate a new environment locally as well.
    You will be reminded of this when you try to commit with an outdated virtual environment.

Continue with :doc:`configuring and building 4C <installation_developer_workflow>`.

.. _build4Cwithcustomtargets:

Build |FOURC| with custom targets
-----------------------------------

Above it was shown how to build all executables that come with |FOURC|.
However, one can also build just a subset or even specific libraries.
The command to build |FOURC| these specific targets is:

::

    ninja -j <numProcs> <customTarget>

where ``<numProcs>`` denotes the number of processors and ``<customTarget>`` the target you want to build (see below).

.. _custom_target_specifiers:

Custom target specifiers
~~~~~~~~~~~~~~~~~~~~~~~~

|FOURC| offers a variety of additional target specifiers <customTarget> (defined in ``CMakeLists.txt``) that can be used within the build command.
Here's a list of all valid custom target specifiers with a brief explanation:

Executables:

- ``4C`` generate the main |FOURC| executable only
- ``post_processor`` build the post-filters only
- ``post_monitor`` build a nodal data extraction application
- ``full`` generate all executable targets of |FOURC|

Documentation (for the documentation to be generated, you have to set the respective cmake variables in the presets file described :ref:`above<installation_configure>`):

- ``documentation`` create the main documentation (set ``FOUR_C_ENABLE_DOCUMENTATION=ON`` in the presets)
- ``doxygen`` create the (developer-oriented) Doxygen documentation (set ``FOUR_C_ENABLE_DOXYGEN=ON`` in the presets)

.. note::

    When omitting the custom target specifier in the build command, the default specifier 4C is used.


Installing |FOURC| for use in other projects
------------------------------------------------

|FOURC| can be installed the same way as any typical CMake project. In your CMake preset, set the ``CMAKE_INSTALL_PREFIX`` to the desired location.
Then, run the install command:

::

    ninja install

This will install |FOURC| in the specified location. You can then use the installed |FOURC| in another CMake project by finding it with the `find_package` command:

.. code-block:: cmake

    find_package(4C REQUIRED CONFIG HINTS <path-to-4C-installation>)

    # Link against the 4C library which was found by find_package.
    # This pulls in all the necessary dependencies and headers.
    target_link_libraries(<your-target> PRIVATE 4C::lib4C)

.. _build4Cwithkokkosparallel:

Building |FOURC| with OpenMP and CUDA support through Kokkos
------------------------------------------------------------

|FOURC| is primarily developed around MPI parallelism, but also offers the ability to use shared memory parallelism through `Kokkos <https://kokkos.org/>`_, enabling hybrid parallelism on the CPU through OpenMP and GPU acceleration through CUDA.

Kokkos (and Kokkos-Kernels) can be built within Trilinos or specified as an external TPL in Trilinos, and its configuration follows the usual procedure for the desired backend (see the `Kokkos configuration guide <https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html>`_). Trilinos then requires ``Trilinos_ENABLE_<Backend>=ON`` and, specifically for CUDA, ``TPL_ENABLE_CUDA=ON``. To prevent oversubscription and unwanted shared memory parallelism in |FOURC|, you should disable these backends for Tpetra with ``TPETRA_INST_<BACKEND>=OFF`` and explicitly set ``TPETRA_INST_SERIAL=ON``. When using the CUDA backend, you must also make sure to enable the correct architecture during the Kokkos or Trilinos configuration (e.g. ``Kokkos_ARCH_HOPPER90=ON``).

Building |FOURC| with Kokkos' CUDA backend enabled currently requires using a compiler wrapper, ``utilities/clangcuda++``, as the C++ compiler and clang as the C compiler. Nvidia's NVCC, which Kokkos uses by default, has issues compiling some modern C++ code in |FOURC|. To build |FOURC| with Kokkos' CUDA backend, you must first enable the ``FOUR_C_CLANGCUDA`` compile option. When using Open MPI, you should then set the following environment variables to point to the underlying compilers: ``export OMPI_CXX=</path/to/4C>/utilities/clangcuda++`` and ``export OMPI_CC=</path/to/clang>``. The ``clangcuda++`` compiler wrapper will detect and use the same architecture as the Kokkos installation, but you can explicitly override this by setting the ``CLANGCUDA_ARCH`` environment variable. The default clang++ and CUDA paths can also be overridden with the environment variables ``CLANGCUDA_CLANG_PATH`` and ``CLANGCUDA_CUDA_PATH``, respectively.

Due to incompatibility with the serial version of ArborX, it is recommended to disable ``FOUR_C_WITH_ARBORX`` when using shared memory Kokkos backends. It is also necessary to disable ``FOUR_C_WITH_VTK`` when using the CUDA backend, due to compilation errors.

For developers, it is important to know that any target in |FOURC| which contains Kokkos device code (e.g. ``Kokkos::parallel_for()`` or ``KOKKOS_LAMBDA``) must be marked with the ``CLANGCUDA_MODE_DEVICE`` compile definition for CUDA compilation to be possible.
