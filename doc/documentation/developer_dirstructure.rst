Directory structure
--------------------
The |FOURC| code comes with documentation, example input files and
support scripts. The important subdirectories are the following:


:``apps``: Contains executables built on top of the |FOURC| library code.

:``cmake``: Contains the CMake configuration files to build |FOURC|.

:``dependencies``: Contains installation scripts for the external dependencies of |FOURC|.

:``doc``:   Contains this documentation and also the setup for Doxygen.

:``doc_removed_code``: Contains a changelog of major deletions in the code base.

:``docker``: Contains the setup files for docker images for running |FOURC| inside of it.

:``presets``:   Contains preset files for CMake.

:``src``: Contains the bulk of the |FOURC| code organized into several modules. See below for more details.

:``tests``:   Contains end-to-end tests that run |FOURC| as a whole with optional pre- and post-processing steps. The
    tests are not directly related to one specific feature (these reside next to the sources).

:``tests/input_files``:   Contains various valid and running input files.
    These input files are also used for automated testing.

:``unittests``:  [Legacy] Contains the source files for stand alone tests aka unittests.
    These tests will move closer to the source code into the respective modules.

:``utilities``:  Contains useful scripts to develop and test |FOURC|.

The top-level directory should only contain files that are commonly expected in this location,
such as the README, LICENSE, and configuration files for tools like clang-format and clang-tidy.

Details of the ``src`` directory
--------------------------------

The ``src`` directory contains the modules that make up the |FOURC| library. Each module is
split into a ``src`` and ``tests`` directory. The ``src`` directory contains the production code
of the module, while the ``tests`` directory contains the module-related tests.

Types of source files used within |FOURC|
"""""""""""""""""""""""""""""""""""""""""

The code base consists almost exclusively of C++ source files. To better indicate the intended usage
of a file, both to the build system and developers, the following file extensions are used:

:``.cpp``:  C++ source files. These files are compiled.

:``.hpp``:  C++ header files. These files are included in other source files.

:``.fwd.hpp``:  C++ forward declaration header files. These files contain forward declarations
    of classes and functions that appear over and over again. They are especially useful for
    external dependencies to isolate the exposed surface area of the dependency.
    This type of file is usually included in other header files.

:``.templates.hpp``:  C++ header files which contain additional template definitions. The reason
    why you might want to separate certain expensive template definitions from the main header
    file is to speed up compilation times. Sometimes a template definition requires additional
    expensive includes, which are not necessary for the main header file. In this case, the
    template definition is best moved to a ``.templates.hpp`` file.

:``.inst.hpp``: C++ header files which contain explicit template instantiations. These files are included
    in the corresponding ``.cpp`` file to instantiate the templates. These files are not strictly necessary
    as you can also instantiate the templates directly in the ``.cpp`` file. However, by moving
    the instantiations to a separate file, you can reuse the same instantiations in multiple
    ``.cpp`` files which contain parts of the implementation of the same classes.

:``.<ext>.in``: These files are templates (not in the C++ sense!) requiring additional configuration
    via CMake. The CMake script will generate the actual file by replacing placeholders in the
    template file. The generated file will be placed in the build directory with the extension ``.<ext>``.




