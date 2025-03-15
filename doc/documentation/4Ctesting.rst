.. _4Ctesting:

Testing
=======

Clean code goes hand in hand with code testing, that is why in |FOURC| we set great value upon the latter.
This page is concerned with everything related to testing in |FOURC|.

Overview on testing mechanisms
------------------------------

|FOURC| uses the ``ctest`` environment for testing (refer to `ctest <https://cmake.org/cmake/help/latest/manual/ctest.1.html>`_)
and relies on a variety of test mechanism to guarantee and maintain its intended behaviour.
Tests fall into these categories:

- Regression tests: test dat-file based simulations only (no pre-/post-processing)
- Unit tests: isolate and exercise specific units of source code independently from other parts
- Framework tests: tests the complete |FOURC| workflow (mesh generation, pre-processing, simulation, post-processing)

|FOURC| tests can be triggered through various mechanisms:

- On Github, we use Github actions on every pull request and run additional tests every night.
- Locally, one can trigger ctest to run all or some tests.

    - filter tests by adding ``-R <regex>``. Only tests including ``<regex>`` in their names are performed
    - exclude tests by adding ``-E <regex>``. Tests with ``<regex>`` in their names are not performed
    - filter tests by adding ``-L <regex>``. Only tests with label ``<regex>`` in their names are performed
    - skip the clean up test by adding ``-FC test_cleanup``

- Particularly, if you want to run ctest on a single input file, simply run ``ctest -R <input_file>``,
  where ``<input_file>`` is the input filename without the ``.dat`` suffix. A ctest with that name should exist.


Guidelines for |FOURC| input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``*.dat``-file of a CI test (residing in ``<4C_sourcedir>/tests/input_files``) should include:

- A short but informative summary of what is tested, maybe also which results are compared,
  in the header
- No unused sections
- No unused parameters (as far as possible)
- No alternative input parameters/values as comment
- No empty lines

In general a "clean" format of the file is demanded (alignment of parameters/values).

Accompanying ``*.xml``-files will be formatted by ``pre-commit``. Check ``.pre-commit-config.yaml`` for details.

Executing |FOURC| unit tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure and build |FOURC| as described in `README <https://github.com/4C-multiphysics/4C/blob/main/README.md>`_.
In the |FOURC| build directory ``<builddir>`` a subfolder ``unittests`` with executable unittests inside is generated.

    Note: in order to execute the following commands, change to build directory <builddir>

The |FOURC| unit tests are included in ctest as part of the minimal tests and also in the full test suite:

::

    ctest -L minimal
    ctest -R unittests

The test executables are located in the ``tests/`` directory in the build folder. They support the ``--help`` argument,
which can be used to get a list of available options, e.g. to filter for specific tests.

Many IDEs also come with plugins or support for GoogleTest allowing to run tests directly from the IDE.

How to add unit tests to |FOURC|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|FOURC| uses `GoogleTest <https://github.com/google/googletest>`_ for unit testing. If you are new to this framework,
read the `primer <https://google.github.io/googletest/primer.html>`_.
In general, we recommended to look through existing unit tests directories first to get an idea on how the tests are organized and how GoogleTest can be used.

Unit tests reside close to the module containing the tested functionality, namely in the ``tests`` directory next to a module's ``src`` directory.
It can be a good idea to have one unit test file per source file. Conventionally, this test file is named as the source file with the suffix ``_test``.
Unit tests are picked up automatically by CMake through ``four_c_auto_define_tests()``.