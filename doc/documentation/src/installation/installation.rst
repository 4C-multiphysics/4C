.. _installation:

Installation
============

Here you'll find some useful notes on setting up and running |FOURC|.

|FOURC| is developed and used on Linux but other Unixes might work as well.
Additionally, |FOURC| can be compiled under Windows using the WSL (Windows Subsystem for Linux),
which provides a complete Linux Environment within the Windows operating system.

.. _external-dependencies:

External dependencies
---------------------

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

- :ref:`Trilinos <trilinos>` (supported versions are listed in ``dependencies/supported_version/Trilinos.txt``)
- :ref:`SuiteSparse <suitesparse>`
- :ref:`SuperLUDist <superludist>` (recommended version: 7.2.0)
- BLAS
- LAPACK

Graph and domain partitioner:

- Metis (recommended version: 5.1.0, included in SuiteSparse)
- ParMETIS (recommended version: 4.0.3)

Miscellaneous:

- :ref:`ArborX <arborx>` (optional)
- Backtrace
- Boost
- CLN
- dealii (optional)
- FFTW
- HDF5
- magic_enum
- :ref:`MIRCO <mirco>` (optional)
- Qhull
- ryml
- zlib

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

.. _trilinos:

**Trilinos**

This external dependency can be downloaded from the `Trilinos Github repository <https://github.com/trilinos/Trilinos>`_ .
Currently supported versions are listed in ``<4C_sourceDir>/dependencies/supported_version/Trilinos.txt``. An older supported version will be supported for at least six months after its introduction. Afterwards, the version may be dropped at any time.

.. note:: As 4C is still depending on Epetra based Trilinos code, the last officially supported version is release 16.2.0.

.. _mirco:

**MIRCO**

MIRCO can be used as optional dependency inside |FOURC| to be used for linear elastic frictionless normal contact between a rigid rough indentor and an elastic half-space.
See the `MIRCO repository <https://github.com/imcs-compsim/MIRCO>`_ for details and downloads.

Building |FOURC| with MIRCO enabled automatically fetches the repository during the configure stage and later builds the library as dependency.

.. _4Cinstallation:

Download and install
--------------------

After you have installed all the external dependencies, you should download and install |FOURC| itself.

Access the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**If you only want to use** |FOURC| **without contributing to the project,** you can simply clone the repository:

::

    cd <someBaseDir>
    mkdir <4C_sourceDir>
    git clone git@github.com:4C-multiphysics/4C.git <4C_sourceDir>
    cd <4C_sourceDir>

where ``<someBaseDir>`` is some directory on your machine and ``<4C_sourceDir>`` will contain the |FOURC| source code.
You can choose names and locations of these directories freely.

Your directory tree should look like the following::

    <someBaseDir>/
        <4C_sourceDir>

**If you want to contribute to the project via pull requests,** you should fork the repository.
We recommend setting your forked repository as ``origin`` and the `4C-multiphysics/4C  <https://github.com/4C-multiphysics/4C>`_ repository as ``upstream``.
Details about forks, how to fork a repository, how to clone the forked repository,
and how to configure git to sync with the upstream repository can be found in the `GitHub Docs <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>`_.

You can set your local ``main`` branch to track the upstream``main`` branch with the following command::

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

.. _installation_configure:

Configure
~~~~~~~~~

|FOURC| enforces an out-of-source build, i.e. your build directory may not be the same as your source directory.
This is a good practice to keep your source directory clean. Instead, create a build directory.
Many development tools work well when the build directory is a subdirectory of the source directory.
A possible directory structure could look like this::

    <4C_sourceDir>/
       build/
         debug/
         release/
         other_configuration

Note that the directory name ``build`` is automatically excluded from the git repository (the name is included in the file ``.gitignore``).

That said, you can create a build directory wherever you want. This is just a suggestion we find useful in daily work.

|FOURC| uses ``cmake`` for configuration and creation of a build. We strongly recommend to use preset files for ``cmake``.
The command to run is

::

    cmake --preset=<name-of-preset> <4C_sourceDir> | tee config$(date +%y%m%d%H%M%N).log

Thus, a preset name needs to be passed to cmake via the command line argument ``--preset``.
Use

::

    cmake <4C_sourceDir> --list-presets

to get a list of all available presets.

In general, it is highly recommended to create your own preset, which is stored in ``<4C_home>/CMakeUserPresets.txt``.
To create your own preset you may start with the presets used in our CI, see ``presets/docker/CMakePresets.json``.
In a preset within this file, you should define a few options that are important to your specific build:

- the build type. This is given by the variable ``CMAKE_BUILD_TYPE`` and can be ``DEBUG`` or ``RELEASE``.
- the build directory. It is good practice to indicate the value for ``CMAKE_BUILD_TYPE`` in the folder name, e.g. by
  ``"binaryDir": "<4C-basedir>/builds/release-build"`` (the folder name is completely up to you).

More information about the cmake presets can be found :ref:`here <developer_cmake>`.
This section also contains an up-to-date :ref:`reference list<reference_cmake_variables>` of all variables used to configure |FOURC|.


.. note::

    When you see ``command |& tee something$(date +%y%m%d%H%M%N).log``,
    that is just a means of running a command and sending the output both to the screen and to a timestamped log file.
    This is by no means necessary, but if you run into problems, having these timestamped log files can be quite useful in debugging what's gone wrong.

Build
~~~~~

Now you may run the compile command within the build folder.

::

    ninja -j <numProcs> [full] |& tee build$(date +%y%m%d%H%M%N).log


where ``<numProcs>`` is the number of processors you want to use.
The optional parameter ``full`` also provides some utility executables, see :ref:`below<custom_target_specifiers>`.
The |FOURC| executable and unittests are also created, if this parameter is omitted.

.. note::

    After the first build, it is rarely necessary to reconfigure |FOURC| ; only the build-command is required.
    `cmake` is invoked *automatically* during the build process if something changed within ``CMakeLists.txt``.

To verify that the build was successful, it is highly recommended to run the test suite,
at least the minimal version of it.
You will find the information about it in the :ref:`testing <4Ctesting>` section below.

.. _set-up-your-ide:

Set-up your IDE
----------------

We recommend to use an Integrated Development Environment (IDE) for code development
because it eases various aspects of code development when working on large projects.
Current popular choices in the |FOURC| community are:

- :ref:`CLion <clion>`
- :ref:`Visual Studio Code <visualstudiocode>`

.. _clion:

CLion
~~~~~~

**Setting up CLion**

Let's assume that you already have cloned the repository and created a build directory as outlined above.
Open CLion and open the 4C source directory. CLion understands CMake preset files, so configuration is easy.
Consult the CLion documentation for more information on how to set up a project.
If you want to include a YAML schema for easier writing of input files within the project directory tree, see :ref:`below<clion_yaml_schema>`.

**Enable debugging with CLion**

The prerequisite is that you already have set up a debug configuration as explained above.
Make sure you have enabled a debug profile in your cmake settings.

- Select Edit Configurations... from the dropdown list right to the "green hammer".
- Click + to Add a new configuration and select CMAKE Application:

    - Enter a descriptive Name

        - serial debugging:

            - Select 4C from the dropdown menu for both Target and Executable

        - parallel debugging:

            - Select 4C from the dropdown menu for Target
            - Enter ``<PathToMpirun>/mpirun`` to Executable (find with ``which mpirun`` in console)
            - Add the arguments for mpirun:
              ``-np <NumberOfProcesses> <PathTo4C-debug>/4C <PathToTest/TestFile> <OutputPreFix>``

    - Add any other parameters you need for the program to run (for example, the input file name and the output basename) to the arguments.
    - Enter the path you want to run the program in (maybe the one where your input file is located) to Working directory
    - Remove everything in Before launch and click ok

- Select the created configuration from the dropdown list
- Click on the green beetle in the toolbar to start a debug run.

The program will run until it reaches the end, a breakpoint, or a segmentation fault.

.. _clion_yaml_schema:

**Adding yaml Schema to CLion**

In order to use the Yaml schema in CLion, which simplifies editing |FOURC| input files significantly,
add a JSON schema file, which is automatically created during the build process, to your configuration.

1. In File :math:`\to` Settings, search for "JSON schema", which will bring you to **JSON Schema Mappings**.
2. Add a new entry by clicking on the "+" sign.
3. Give it a descriptive name.
4. Select the Schema file ``4C_schema.json`` from your local build directory.
5. Select Schema version *JSON Schema v4*.
6. Add the file name pattern ``*.4C.yaml``, to which this schema is applied.

Note that this Schema is only valid for the current project, that is, only files in this directory tree are recognized.


.. _visualstudiocode:

Visual Studio Code
~~~~~~~~~~~~~~~~~~~

`Visual Studio Code <https://code.visualstudio.com/>`_ is a code editor optimized for building and debugging modern web and cloud applications.
It can also be used for developing |FOURC|.
Visual Studio Code can connect to a remote computer so you can work on your home computer via SSH, see `here <https://code.visualstudio.com/docs/remote/remote-overview>`_.
If you want to include a YAML schema for easier writing of input files within the project directory tree, see :ref:`below<vscode_yaml_schema>`.


**Setting up VS Code**

Let's assume that you already have cloned the repository, created a build directory and created your own CMakeUserPreset.json as outlined above.
To include the |FOURC| source code into VS Code and enable VS Code to build |FOURC|, follow these steps:

#. Install C/C++ extension for VS Code
#. Install cmake extension for VS Code
#. Open folder with source code of |FOURC|
#. Select cmake preset of your choice

**Setting up VS Code for Remote Development**

Start from scratch without doing the instructions from above. Do not clone your repository on your local machine, all files remain on the remote machine)
Steps to do on your (remote) workstation:

#. Install VS Code

Steps to do on your local machine:

#. Install VS Code
#. Install Remote development pack plugin on your local machine: <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack>
#. Add your remote workstation over the ssh connection via the Remote Explorer (one icon on the left side)
#. connect to your remote workstation
#. Install C/C++ extension via GUI (will install it on your local and remote computer)
#. Open |FOURC| source directory and start coding

**Clangd Language Server (Clang-tidy)**

To profit from clang-tidy (and many more features like cross-references, refactorings, code completion, navigation, find unused includes),
there is an vs code extension that enables clangd for VS Code.
For a full list of features see here: <https://clangd.llvm.org/features.html>

.. figure:: /_assets/vscode-clangd.png
   :alt: Screenshot of the clangd capability for VS Code
   :width: 100%

*Setup*

#. Install extension clangd from the marketspace: <https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd>
#. add a .clangd configuration file in your source directory. An example could look like this:

::

    CompileFlags:
        CompilationDatabase: /path/to/build_directory  # update this to your configuration
        Compiler: /usr/bin/mpic++
        Add: [-I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi, -I/usr/lib/x86_64-linux-gnu/openmpi/include, -pthread, -L/usr/lib/x86_64-linux-gnu/openmpi/lib, -lmpi_cxx, -lmpi] # take this from mpic++ --showme
    Index:
        Background: Build
    Diagnostics:
        UnusedIncludes: Strict

**Debugging with VS Code**

If you want to use VS Code for debugging, you need to add debugging configurations in .vscode/launch.jsonand a debug version of |FOURC|.
In the following, the following folder structure is assumed:

- `<4C-sourcedir>`: Path to the source files
- `<4C-debug-execdir>`: Path to the debug build version
- `<4C-problemdir>`: Path to the run directory

In the following, different configuration examples are given.
They have to be placed in .vscode/launch.json in the configurations-list.

*Debugging a serial run*

::

    {
        "name": "Debug input file",
        "type": "cppdbg",
        "request": "launch",
        "program": "<4C-debug-execdir>/4C",
        "args": ["/path/to/inputfile", "<4C-problemdir>/xxx"],
        "cwd": "<4C-problemdir>",
        "setupCommands": [
            { "text": "handle SIGPIPE nostop noprint pass", "description": "ignore SIGPIPE", "ignoreFailures": true }
        ]
    }


*Debugging a serial restart*

::

    {
        "name": "Debug input file from restart",
        "type": "cppdbg",
        "request": "launch",
        "program": "<4C-debug-execdir>/build_debug/4C",
        "args": [
            "/path/to/inputfile",
            "<4C-problemdir>/xxxx"
            "restart=1",
            "restartfrom=<4C-problemdir>/xxx"
        ],
        "cwd": "<4C-problemdir>",
        "setupCommands": [
            { "text": "handle SIGPIPE nostop noprint pass", "description": "ignore SIGPIPE", "ignoreFailures": true }
        ]
    }

*Debugging a unit test*

::

    {
        "name": "Unit test",
        "type": "cppdbg",
        "request": "launch",
        "program": "<4C-debug-execdir>/Unittests/unittests",
        "args": [
            "Mat::Elastic::CoupAnisoExpoAnisotropyExtension_TestSuite"
        ],
        "cwd": "<4C-problemdir>",
    }

*Debugging a MPI application*

All-Stop mode

This mode is the "normal" mode. On a breakpoint, all processes make a pause.

::

    {
        "name": "Debug MPI input file (all stop)",
        "type": "cppdbg",
        "request": "launch",
        "program": "/usr/lib64/openmpi/bin/mpirun",
        "args": [
            "-np",
            "3", // specify number of mpi ranks here
            "<4C-debug-execdir>/4C",
            "/path/to/inputfile",
            "<4C-problemdir>/xxx",
        ],
        "cwd": "<4C-problemdir>",
        "setupCommands": [
            {
                "description": "On a fork, keep gdb attached to both processes.",
                "text": "-gdb-set detach-on-fork off",
                "ignoreFailures": false
            },
            {
                "text": "-gdb-set schedule-multiple on",
                "ignoreFailures": false
            }
        ]
    }

Tracking down race conditions

With this method, you have control to each processor during the execution.
However, you have to attach each processor manually.
Start |FOURC| with the following command in an extra terminal:

::

    ~/build_debug$ mpirun -np 2 ./4C <input> <output> --interactive
    Global rank 0 with PID 17235 on helmholtz.lnm.mw.tum.de is ready for attach
    Global rank 1 with PID 17236 on helmholtz.lnm.mw.tum.de is ready for attach

    ** Enter a character to continue >

In the output, you see for each mpi rank the respective process id.
Now you can attach gdb to each process with the following configuration:

::

    {
        "name": "Attach gdb",
        "type": "cppdbg",
        "request": "attach",
        "program": "<4C-debug-execdir>/4C",
        "processId": "${command:pickProcess}",
        "MIMode": "gdb"
    }

Start it two times and choose in the prompt the respective process id.
Wait until both instances are connected and then start the computation by pressing any key in the 4C terminal.

.. _vscode_yaml_schema:

**Adding yaml Schema to VS Code**

In order to use the Yaml schema in VS Code, which simplifies editing |FOURC|  input files significantly, is done in two steps.configure

1. Install the official YAML by Red Hat extension within VS Code

2. Goto file :math:`\to` Preferences :math:`\to` Settings

   Search for "schemas".
   You'll find the entry **JSON: Schemas** containing the link *Edit in settings.json*; click on it.

   In the ``setting.json`` file, you might have entries already; add the following entry to the main dictionary
   (adjust the directory of the json file to your build directory, where this file is located):

   ::

       "yaml.schemas": {
         "<path_to>/4C_schema.json": "*.4C.yaml",
       },

The schema is now automatically applied on Yaml files that end with ``.4C.yaml``.

Since the ``settings.json`` file is by default stored in ``$HOME/.config/Code/User``,
it affects all files with the respective suffix that you open with VS Code, not only those in the |FOURC| project directory.

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

