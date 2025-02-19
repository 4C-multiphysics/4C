.. _materials:

Materials
===========

General information
--------------------

The definition of materials happens in the section ``MATERIALS``.
A material model defines its constitutive behavior, which may consist of several terms;
however, there is always *one* material assigned to each element.
This material can be an explicit material definition on its own,
or a collection of behaviours consisting of a number of material models to be defined subsequently.
The explicit definition of a material model is in general given by a single line of the form

::

   MAT <id> <materialname> <parameters...>

One may also define a material by summation of several potentials.

A collection of material behaviors, on the other hand, looks like this:

::

   MAT <id> <collectionname> NUMMAT <nmaterials> MATIDS <id_1> ... <id_nmaterials> <possibly further parameters>

Here, the terms of the ``<nmaterial>`` constitutive behaviors have to follow with their own number,
which must correspond to ``<id_1> ... <id_nmaterials>``.




Structural Material Models
--------------------------


A material model in structural mechanics defines the connection between deformation (usually strain) and stress.
Many material models are available, including (hyper-)elastic, elasto-plastic, visco-elastic, visco-plastic, and even damage models.
If you wish to implement a new material model, this is of course also possible, it is an in-house code after all.
You'll find more information on implementing material models in the :ref:`material section<materialdevelopment>` of the developer guide.



Fluid Material Models
---------------------



Other Material Models
---------------------

Coupling Material Models for Various Physics on a Single Discretization
-----------------------------------------------------------------------

One may use a single (multiphysics) element type for a multiphysics simulation with matching discretizations.
However, since each discretization belongs to a single physics representation and can thus only be connected to a single material,
the other material has to be connected to the same discretization by cloning the discretization to the other physics.

In |FOURC|, we use a material mapping section, which connects two material models to a single representation.
One can see it here in for a simple Thermo-Structure-Interaction problem:

::

   ----------------------------------------------------------MATERIALS
   MAT 1 MAT_Struct_ThermoStVenantK YOUNGNUM 1 YOUNG 2.1e08 NUE 0.3 DENS 7.85e-06 THEXPANS 1.1e-05 CAPA 6.445 CONDUCT 1031.2 INITTEMP 273.15
   MAT 2 MAT_Fourier CONDUCT_PARA_NUM 1 CAPA 6.445 CONDUCT 1031.2
   -----------------------------------------------CLONING MATERIAL MAP
   SRC_FIELD structure SRC_MAT 1 TAR_FIELD thermo TAR_MAT 2
   -------------------------------------------------STRUCTURE ELEMENTS
   1 SOLIDSCALAR HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM nonlinear TECH fbar


.. ToDo::

    Here we should have a list of the discretizations that can be coupled


