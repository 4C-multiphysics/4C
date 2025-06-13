========
|FOURC|
========

Mission statement
=================

|FOURC| is a parallel multiphysics research code
to analyze and solve a plethora of physical problems
described by ordinary or partial differential equations.
Its development is driven by challenging research questions and real-world problems,
for which existing tools do not suffice,
either due to the lack of capabilities or due to falling short of accuracy or performance.

|FOURC| not only provides ready-to-use simulation capabilities for a variety of physical models,
including single fields such as solids and structures, fluids, or scalar transport,
and multiphysics coupling and interactions between several fields,
but also a modular software environment for research in mathematical modeling and numerical methods.
Pre- and post-processing tools facilitate the use of |FOURC| within streamlined application workflows in science and engineering.
For spatial discretization, |FOURC| mostly relies on finite element methods (FEM, CutFEM).
It leverages the `Trilinos project <https://trilinos.github.io>`_ for sparse linear algebra, nonlinear solvers, and linear solvers and preconditioners
to be executed on MPI-parallel computing clusters.
Through its comprehensive set of physics modules available to all users without coding effort,
|FOURC| facilitates the advancement of research in all areas of science, engineering, and biomedicine.

Content
=======

This guide to |FOURC| is structured as follows:

:ref:`About 4C<about>`
   Learn about the capabilities and history of |FOURC|.

:ref:`The 4C Community<4Ccommunity>`
   A brief summary of the roles and responsibilities within the |FOURC| community.

:ref:`Installation<Installation>`
   A summary of all requirements of |FOURC| and detailed steps how to build |FOURC|.

:ref:`Tutorials<tutorials>`
   A series of beginner-level tutorials showcases the setup procedure for specific application scenarios.

:ref:`Analysis guide<analysisguide>`
   Detailed explanations on the whole tool chain from model generation (pre-processing)
   over running a simulation to the evaluation of results (post-processing) offers deep insight into using |FOURC|
   for advanced simulation scenarios.
   This guide includes background information and detailed descriptions
   for the specification of elements, boundary conditions, constitutive laws
   as well as options for linear and nonlinear solvers.

:ref:`Developer guide<developerguide>`
   This guide gets you started on actively developing and contributing to |FOURC|.
   It covers our CI/CD testing infrastructure, coding guidelines, and useful tools for the daily development of |FOURC|.

:ref:`Input Parameter Reference<inputparameterreference>`
   A comprehensive list of all input parameters, elements, materials, and boundary conditions
   with short descriptions for each option

:ref:`Tools and Scripts<toolsAndScripts>`
   A collection of useful scripts for working with |FOURC|

:ref:`Appendix<appendix>`
   Information on contributing to this documentation as well as selected topics of interest

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   about/about
   community/4Ccommunity
   installation/installation
   parsed_files/tutorials
   analysis_guide/analysisguide
   developer_guide/developmentguide
   input_parameter_reference/parameterreference
   tools/tools
   appendix/appendix
