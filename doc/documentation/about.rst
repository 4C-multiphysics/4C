.. _about:

=============
About |FOURC|
=============

Overview
--------

|FOURC| provides both C++ implementations of various single-field as well as coupled multi-field solvers
for one-, two- and three-dimensional nonlinear problems in science and engineering
and a variety of advanced coupling utilities
to quickly create new multi-field solvers if required by the research question at hand.
At its core,
|FOURC| employs the Finite Element Method (FEM) for spatial discretization of the governing field equations.
Ready-to-use single-field solvers range from solid and structural mechanics over scalar transport all the way to fluid flow.
Through partitioned or monolithic coupling of single-field solvers,
either through volume-, surface-, or line-coupling on matching, non-matching or embedded meshes,
multiphysics phenomena such as fluid-solid interaction (FSI), thermo-mechanics, or coupled scalar-solid-thermo systems can be tackled.
Through its versatile library of constitutive laws,
|FOURC| is well positioned for the analysis of biological tissue, all-solid-state batteries, or plasticity to just name a few.
Besides classical FEM, cut finite element methods (CutFEMs) allow to treat embedded interfaces.
Furthermore, particle methods such as the discrete element method (DEM) and smoothed particle hydrodynamics (SPH) are available
and can be combined with FEM discretizations.
|FOURC|'s embedding into customizable pre- and postprocessing routines
as well as its interoperability with the `QUEENS library <https://www.queens-py.org/>`_,
an open-source Python framework for conducting intricate multi-query analyses of arbitrary computational models
including parameter studies, uncertainty quantification, and inverse problems,
elevates |FOURC| to a powerful and comprehensive tool in modern computational engineering.
Through the underlying discretization and linear algebra framework,
all simulations can run on parallel computing clusters following a distributed memory paradigm.
While |FOURC| delivers different physics modules, distributed sparse linear algebra and parallel computing out-of-the-box,
all of its components, e.g., single- and multi-field solvers, coupling schemes, numerical algorithms,
can be tailored to the research problem at hand
to also allow the solution of unseen challenges in computational mechanics
while still standing on the solid foundation of a powerful and mature multiphysics simulation framework.
This approach enables the user to exert fine-grained control over *every* aspect and component of the simulation tool chain
and, thus, allows for research in advanced numerical methods within a code base
that is capable to answer real-world problems accurately and at scale.

Capabilities
------------

|FOURC| offers a variety of discretization techniques and numerical tools to its users:
FEM serves as |FOURC|'s main approach for spatial discretization,
while extensions such as CutFEM (for fluid, solid, or FSI problems for example)
or IGA using NURBS are available for selected problem types and applications.
Time-dependent problems can be tackled with various implicit and explicit single- and multi-step methods.
To enable meshtying of non-matching grids, a tool box for mortar coupling is available
(with example applications in different physics,
e.g., solid and contact mechanics, fluid flow, FSI, or lithium-ion cells.
To support the development of physical models,
constitutive laws are collected in a unified framework to enable interoperability with abstract FEM evaluation routines.

|FOURC| implements a series of single-physics modules:
While problems in solid mechanics can also encompass contact conditions and multi-point constraints,
structural models range from trusses over geometrically exact beam models
(and beam interaction effects) to shell elements.
For flow problems, |FOURC| comes with an incompressible Navier--Stokes solver
(including different turbulence models),
low-Mach number flows, or two-phase flows.
For biomedical applications,
reduced order models for flow in arteries or airways are available alongside suitable Windkessel models.
Finally, solvers for transport of scalar fields such as heat or chemical concentrations are available.

With the intent to study multiphysics phenomena,
|FOURC| builds upon its single-field solvers
to implement partitioned and monolithic multiphysics solvers for surface- and volume-coupled problems.
In surface coupling,
existing capabilities from the solid and structural mechanics module are coupled to incompressible fluid flow,
resulting in partitioned and monolithic FSI solvers
using an arbitrary Lagrangean-Eulerian (ALE) description for deforming fluid domains
with scalable multi-level solvers,
monolithic solvers for fixed-grid FSI based on CutFEM,
or partitioned approaches for fluid-beam interaction.
In volume-coupling,
multi-field problems such as thermo-solid interaction (TSI),
scatra-thermo interaction (STI),
solid-scatra interaction (SSI),
or porous media are available.

History
-------

The development of |FOURC| arose from the need to tackle challenging research questions
in and with numerical methods for ordinary and partial differential equations
and to advance complex models for real-world applications,
all based on a proper theoretical foundation and with verified and state-of-the-art methods and software implementations.
Since suitable tools are often not available either in commercial or in other (academic) research codes,
we want to close this gap by developing |FOURC|, a comprehensive multiphysics simulation framework.
|FOURC| originated in the 2000s at the `Institute for Computational Mechanics <https://www.epc.ed.tum.de/lnm/home/>`_ of the Technical University of Munich [#f1]_.

.. rubric:: Footnotes

.. [#f1] A predecessor has been developed under the name BACI, which has been built upon the extensive experience of
   individual developers of the software ccarat of the Institute for Structural Mechanics of the University of Stuttgart.
