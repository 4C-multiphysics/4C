.. _compiletimetracereport:

Clang Time Tracing
==================

Overview
--------

Clang's *time tracing* feature records detailed timing information during the
compilation process. This helps identify slow compilation units and
bottlenecks in the build system.

In our project, a full time tracing run is executed **once per night** as part
of nightly testing. The latest nightly Clang time tracing report is published
automatically and can be viewed here:

`View Nightly Time Trace Report <https://4c-multiphysics.github.io/4C/clang18_fourc_assertions_trilinos_assertions_compile_time_report.html>`_
