Configure, build, and use an IDE
================================

This workflow applies after external dependencies have been installed, regardless of whether they were installed
automatically with Spack or manually.

.. _installation_configure:

Configure
---------

|FOURC| enforces an out-of-source build: the build directory must not be the source directory. This keeps generated
files separate from the checkout. Many development tools work well when the build directory is a subdirectory of the
source directory. A possible layout is::

    <4C_sourceDir>/
        build/
            debug/
            release/
            other_configuration/

The directory name ``build`` is excluded from Git by ``.gitignore``. You can place build directories elsewhere, but
use a separate directory for each build configuration.

|FOURC| uses CMake to configure builds. We recommend CMake presets and a local ``CMakeUserPresets.json`` file for
your build choices. Configure a build from the source directory with::

    cmake --preset <name-of-preset>

List the available presets with::

    cmake --list-presets

A local preset should define at least ``CMAKE_BUILD_TYPE`` and ``binaryDir``. More information about CMake presets
is available in the :ref:`developer CMake guide <developer_cmake>`, including an up-to-date
:ref:`reference list <reference_cmake_variables>` of configuration variables.

.. note::

    When you see ``command |& tee something$(date +%y%m%d%H%M%N).log``, the command output is sent both to the
    terminal and to a timestamped log file. This is optional, but logs can help diagnose configuration problems.

Build
-----

Run the compile command from the configured build directory::

    ninja -j <numProcs> [full] |& tee build$(date +%y%m%d%H%M%N).log

Here, ``<numProcs>`` is the number of processors to use. The optional ``full`` target also builds utility
executables; see :ref:`custom build targets <custom_target_specifiers>`. Without it, the |FOURC| executable and
unit tests are built.

.. note::

    After the first build, reconfiguration is rarely necessary. CMake is invoked automatically during a build when
    a ``CMakeLists.txt`` file changes.

To verify the build, run at least the minimal test suite; see :ref:`testing <4Ctesting>`.

.. _set-up-your-ide:

Set up your IDE
---------------

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
4. Select the Schema file ``4C_schema_completion.json`` from your local build directory.
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

Let's assume that you already have cloned the repository, created a build directory and created your own CMakeUserPresets.json as outlined above.
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
            "<4C-problemdir>/xxxx",
            "--restart=1",
            "--restartfrom=<4C-problemdir>/xxx"
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

In order to use the Yaml schema in VS Code, which simplifies editing |FOURC|  input files significantly, is done in two steps.

1. Install the official YAML by Red Hat extension within VS Code

2. Go to file :math:`\to` Preferences :math:`\to` Settings

   Search for "schemas".
   You'll find the entry **Yaml: Schemas** containing the link *Edit in settings.json*; click on it.

   In the ``settings.json`` file, you might have entries already; add the following entry to the main dictionary
   (adjust the directory of the json file to your build directory, where this file is located):

   ::

       "yaml.schemas": {
         "<path_to>/4C_schema_completion.json": "*.4C.yaml",
       },

The schema is now automatically applied on Yaml files that end with ``.4C.yaml``.

Since the ``settings.json`` file is by default stored in ``$HOME/.config/Code/User``,
it affects all files with the respective suffix that you open with VS Code, not only those in the |FOURC| project directory.

