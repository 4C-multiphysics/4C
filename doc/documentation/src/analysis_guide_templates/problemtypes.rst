.. _problemtypes:

Problem types
==============

.. ToDo:

   Here, we have to describe the different types that can be calculated with |FOURC|. It's not complete yet.

.. _singlefieldproblems:

Single field problems:
----------------------

Ale
~~~~

**General**

Ale stands for *arbitrary Lagrange-Euler*.
Thus, the mesh may have a particular movement, which is defined by the ``ALE_TYPE``.

**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`.
The number of degrees of freedom depends on the dimensionality of the structure (2 for 2D structures, 3 for 3D)


ArterialNetwork
~~~~~~~~~~~~~~~

**General**

This problemtype considers the blood flow in networks of arteries, which are modelled as 1D line elements in three-dimensional space.


**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>` (only 1D line elements are available).
The number of degrees of freedom


Cardiac_Monodomain
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

This problemtype considers .

**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`.


Fluid
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This problemtype considers the flow of fluids through a domain.

**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`.
The number of degrees of freedom depends on the dimension of the problem;
One may have 2 or 3 degrees for the velocity plus one degree for the fluid pressure.


Electrochemistry
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This problemtype is kind of a scalar transport simulation, which considers xxx as the primary variable,
and xxx is the flow.

**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`, and the domain may be 1D, 2D, or 3D.


Fluid_Top_Opt
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

**Elements and Degrees of freedom**

The element types used are given in xxx. The number of degrees of freedom


Fluid_XFEM
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.

**Elements and Degrees of freedom**

The element types used are given in xxx. The number of degrees of freedom


Level_Set
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

.. ToDo::

   No information about this special problem type.


**Elements and Degrees of freedom**

The element types used are given in xxx. The number of degrees of freedom


Particle
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This is a completely different discretization type that is not based on a mesh consisting of elements and nodes but instead uses particles as spatial discretization points.

Currently, two different particle interaction methods are implemented.
The *discrete element method (DEM)* assumes spherical shaped particles that physically interact with each other.
With *smoothed particle hydrodynamics (SPH)* particles are interpreted as discretization points that along with a smoothing kernel discretize a continuum.

**Particles and Degrees of freedom**

Generally, particles are defined by their spatial coordinates in 3D space.
Additionally, further states may be defined for the particles as degrees of freedom, e.g., temperature.


Polymer_Network
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

The polymer network is, similarly to the Arterial Network, a network of 1D constituents in 3D space, namely here a network of polymer fibers.

**Elements and Degrees of freedom**

The element types are solely BEAM3R elements used, which can be found in the :ref:`Spatial Discretization reference<discretization_reference>`.



ReducedDimensionalAirWays
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

Similar to other network models, the *Reduced dimensional Airways* simulation considers a network of 1D elements in 3D space,
namely here 1D pipe models of human airways.

**Elements and Degrees of freedom**

The element types used here are 1D elements given in the :ref:`Spatial Discretization reference<discretization_reference>`.


Scalar_Transport
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

This problemtype considers the transport of a scalar variable through a domain.
Such a scalar could be the temperature (which is usually treated by PROBLEMTYPE Thermo),
but also any other field quantity.

**Elements and Degrees of freedom**

Since the problemtype is scalar transport,
the value of this scalar is the only active degree of freedom for this type.

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`.

**Results**

The main output consists of the scalar quantity itself and its flux.

Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

The problemtype *Structure* considers deformations and stresses in solid mechanics.

**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`. The number of degrees of freedom depends on the element type.
In general, all elements have displacements in spatial directions (2 or 3, depending on the dimensionality).
In the case of C1-steady elements like beams and shells, the rotations (again 2 or 3) are added to the degrees of freedom,
so there are up to 6 DoFs.

**Results**

The result variables are stresses and strains within the elements,
and reaction forces at the nodes, where a Dirichlet boundary condition has been applied.
Other internal variables may be calculated as necessary (and desired).

Thermo
~~~~~~~~~~~~~~~~~~~~~~~~~

**General**

The problemtype *Thermo* considers heat transfer in arbitrary structures.

**Elements and Degrees of freedom**

The element types used are given in the :ref:`Spatial Discretization reference<discretization_reference>`. There is only one degree of freedom, that is the temperature.

**Results**

Internally: Heat flux per area
At Dirichlet boundary conditions: heat flux



.. _multifieldproblems:

Multi field problems:
----------------------

These problems combine a number of single field problems, and are therefore sometimes called *Coupled Problems*.

Biofilm_Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Elastohydrodynamic_Lubrication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: STRUCTURAL | LUBRICATION | ELASTO HYDRO

Fluid_Ale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:  FLUID | ALE | FSI

Fluid_Beam_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: FSI | FLUID | STRUCTURAL

Fluid_Freesurface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: FLUID | FSI | ALE


Fluid_Poro_Structure_Interaction_XFEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: STRUCTURAL | POROELASTICITY | FSI | FLUID

Fluid_Porous_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:



Fluid_Porous_Structure_Scalar_Scalar_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_RedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction_RedModels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_Structure_Interaction_XFEM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Fluid_XFEM_LevelSet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Gas_Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Low_Mach_Number_Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Lubrication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Multiphase_Poroelasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Multiphase_Poroelasticity_ScaTra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Multiphase_Porous_Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

NP_Supporting_Procs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Particle_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Poroelastic_scalar_transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Poroelasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Scalar_Thermo_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Structure_Ale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Structure_Scalar_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Structure_Scalar_Thermo_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Thermo_Fluid_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Thermo_Structure_Interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics:

Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One has to define solvers for the following dynamics: