.. _ccache:

Speed up recompilation using ``ccache``
----------------------------------------

Working with many different branches in the same local repository is a key feature of Git providing an efficient work flow for code development.

After switching to another branch in the repository, a lot of files must potentially be recompiled.
Although ``make`` and ``ninja`` allow for incremental builds that only recompile
what is necessary, they cannot remember the build result for multiple versions of files.

Here, ``ccache`` can speed up recompilation time (refer to `ccache <https://ccache.samba.org/manual/latest.html>`__).
``ccache`` is a compiler cache.
It speeds up recompilation by caching the result of previous compilations and detecting when the same compilation is done again.

The recommended way to use ``ccache`` with |FOURC| is to set the
``CMAKE_CXX_COMPILER_LAUNCHER`` variable to the path of ``ccache`` in your CMake preset file.

Enforce a complete recompilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you ever want to clear the cache of ``ccache``, you can run

::

    ccache -C -z


