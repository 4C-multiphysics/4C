FSI Tutorial 3d with *pre_exodus* and Cubit
==============================================

Introduction
------------

As example, we consider a 3d driven cavity example as sketched in the
:ref:`figure below <tut_fsi_preexo>` with a depth of 0.05. Hint: In case you
want or need to see a sample solution for this tutorial (the FSI part)
you will find corresponding files in the |FOURC| subfolder
*/tests/framework-tests/*! However, it is highly recommended to look
at these files only in case you encounter severe problems while
stepping through the tutorial.
For further details and references we refer the reader to [Wall99]_.

.. figure:: figures/Angabeskizze.jpg
   :alt: The driven cavity example
   :width: 60%
   :align: center
   :name: tut_fsi_preexo

   The driven cavity example

In the following, we first create the finite element mesh for the complete structure,
but then we'll split the simulation into three sections:

- Structure part
- Fluid part
- Fluid-Structure interaction part

Creating the Geometry with Cubit
--------------------------------

Besides meshing cubit also has several geometry creation methods. We
refer to the provided manual and tutorials. It supports scripting (also
Python), therefore we provide the following *Journal*-file containing
the necessary geometry commands as well as mesh and definitions for
elements and boundary conditions, respectively.


.. literalinclude:: tutorial_fsi_3d.jou
   :linenos:


Within Cubit, open the Journal-Editor (*Tools* :math:`\to` *Journal
Editor*), paste the text above and press *play*. For later usage it is
convenient to save the current content of the Journal-Editor into a *\*.jou* file.
Export now the created geometry and mesh to an exodus-file of your choice,
let say, ``<yourmesh>.e`` via *File* :math:`\to` *Export...*.
During export, set the dimension explicitly to 3d.

.. _workingWithPreExodus:

Working with *pre_exodus*
-------------------------

*pre_exodus* is a C++ code embedded into the |FOURC| environment. It is
meant to transfer a given mesh into a |FOURC|-readable input file.
Information about this tool can be found in the :ref:`Analysis Guide <pre_exodus>`.

Besides a given mesh as the one we just created using CUBIT (see above), we need two more files:

- the bc-file, which contains the specific element declaration and the particular boundary conditions, and
- the header file, which contains the general parameters such as solvers, algorithmic parameters, etc..

In the following sections, we'll learn how to create the these two files.

After creating the header and bc file, we may start the solver use the call

``./4C <inputdirectory>/your_example.dat <outputdirectory>/outputprefix``

(in the |FOURC|-directory). The results are then written to the result directory with the prefix you chose.
Note that a number of files are created with this prefix depending on the output you requested.

The Structure Part
------------------

In the following we will always refer to the same exodus-file of the
whole geometry and mesh. For the different problems we differentiate
between *header-file* and *bc-file*. We start with the pure
structural simulation.

Header-file
~~~~~~~~~~~~~~

Find the following sections in ’default.head’ and edit as given:

   ::

      -----PROBLEM TYPE

   set ``PROBLEMTYPE Structure``

   ::

      -----STRUCTURAL DYNAMIC

   set ``DYNAMICTYPE     GenAlpha``

   set ``LINEAR_SOLVER 1``

   set ``NUMSTEP 10``

   set ``TIMESTEP 0.5``

   ::

      -----SOLVER 1

   set ``NAME Solver``

   set ``SOLVER UMFPACK``

   ::

      -----MATERIALS

   insert ``MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 250.0 NUE 0.3 DENS 500``

Safe the file under a different name, e.g. ``dc_struct.head``.

bc-file
~~~~~~~~~~~

As mentioned above we create our *bc-file* from the ’default.bc’.
This consists of a introduction part where some global mesh statistics
are given, a couple of example entries for definitions, a list of all
defined mesh entities in your mesh-file, and finally a list of all valid
|FOURC| conditions to choose from.

For our structure problem we will only define the corresponding mesh
entities. They are easily identified by the names you have given in
Cubit, e.g. “flexible bottom”, or “inflow”. You may notice that we do
not assign any ’E’-entity-numbering. They are automatically determined
in the order they appear in the *bc-file*. Therefore, if you want to
change the numbering in the |FOURC| input file, e.g. for correct hierarchy,
you have to change the order in the *bc-file*.

Therefore we assign:

- The structure elements

   .. literalinclude:: tutorial_fsi_3d.bc
      :lines: 2-8
      :linenos:
      :lineno-start: 2
      :emphasize-lines: 4

-  Surface clamping conditions and fixtures (including hierarchical line conditions)

   .. literalinclude:: tutorial_fsi_3d.bc
      :lines: 18-44
      :linenos:
      :lineno-start: 18
      :emphasize-lines: 5, 12, 19, 26

   .. literalinclude:: tutorial_fsi_3d.bc
      :lines: 53-79
      :linenos:
      :lineno-start: 53
      :emphasize-lines: 5, 12, 19,26


-  The later coupling surface is first loaded with
   constant vertical surface pressure

   .. literalinclude:: tutorial_fsi_3d.bc
      :lines: 46-51
      :linenos:
      :lineno-start: 46
      :emphasize-lines: 5

   Thus, ``sectionname`` and ``description`` have to be changed for a pure solid simulation to

   ::

      sectionname="DESIGN SURF NEUMANN CONDITIONS"
      description="NUMDOF 3 ONOFF 0 1 0 VAL 0.0 -0.01 0.0 FUNCT 0 0 0 TYPE Live"

- Delete the remaining items ``eb2`` and ``ns10`` to ``ns39``.

For Dirichlet conditions the first entry specifies the global degree of
freedom of the problem, denoted as :math:`n`. The :math:`n` following
entries are the flags specifying which degrees of freedom are to be
prescribed in detail, where ``1`` means Dirichlet condition is applied
and ``0`` corresponds to a free degree of freedom. In structure
computations the flags two to four correspond to displacements in
:math:`x`, :math:`y` and :math:`z` direction, respectively. Flags five
to seven are ignored here. The next six entries of the ``description``
specify the values of the prescribed displacements. Subsequently six
time dependent curve ids can be given, corresponding to the ``CURVE``
section in the *header-file*. Finally six spatial function ids can be
defined, corresponding to the ``FUNCT`` section in the *header-file*.
``CURVE`` and ``FUNCT`` are evaluated and their values are multiplied
with the values defined before.

Safe the file under a different name, e.g. ’dc_struct.bc’.

Creating |FOURC| Input File and Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a terminal, run the following command:
``./pre_exodus --exo=dc.e --head=dc_struct.head --bc=dc_struct.bc --dat=dc_struct.dat``
where the filenames might have to be replaced accordingly. This will
result in the specified dat-file which is already validated to be
accepted by |FOURC|. However, if the file is meaningful cannot be assured.

Run the simulation by providing this dat-file and an output file to |FOURC|
and postprocess the results (refer to :ref:`Working with PreExodus<workingWithPreExodus>` for the simulation,
and to :ref:`Post Processing<fsi3dtutorialpostprocessing>` for the post processing).

The Fluid Part
--------------

Again, we rely on the same mesh-file an edit ’default.head’ and
’default.bc’. We simulate a driven cavity with rigid bottom.

.. _header-file-1:

Header-file
~~~~~~~~~~~~~~~

Find the following sections in ’default.head’ and edit as given:

   ::

      -----PROBLEM TYPE

   set ``PROBLEMTYPE Fluid``

   ::

      -----FLUID DYNAMIC

   set ``LINEAR_SOLVER 1``

   set ``NUMSTEP 100``

   set ``TIMESTEP 0.05``

   ::

      -----SOLVER 1

   set ``NAME Fluid solver``

   set ``SOLVER Belos``

   ::

      -----MATERIALS

   insert ``MAT 1 MAT_fluid  DYNVISCOSITY 0.01 DENSITY 1.0``

   ::

      -----FUNCT 1

   insert ``SYMBOLIC_FUNCTION_OF_SPACE_TIME (1-cos(2*t*pi/5))``
   defining time-dependent inflow and lid movement

   ::

      -----FUNCT 2

   insert ``SYMBOLIC_FUNCTION_OF_SPACE_TIME 10*(y-1)*(1-cos(2*t*pi/5))``
   representing the spatial inflow distribution

Safe the file under a different name, e.g. ’dc_fluid.head’.


bc-file
~~~~~~~~

For the pure fluid simulation we assign:



   .. literalinclude:: tutorial_fsi_3d.bc
      :lines: 10-16
      :linenos:
      :lineno-start: 10

The boundary conditions are assigned as follows:

::

   *ns10="CONDITION"    *No-Slip-Condition*
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns11="CONDITION"   *No-Slip-Condition*
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns12="CONDITION"`` Inplane Free-Slip-Condition
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 0 0 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns13="CONDITION"`` Inplane Free-Slip-Condition
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 0 0 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns14="CONDITION"`` No-Slip-Condition for pure fluid simulation
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns15="CONDITION"`` Driving lid, time dependent
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 1 0 0 0 "

   *ns16="CONDITION"`` Inflow, time and space dependent
   sectionname="DESIGN SURF DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 1 0 0 0"

   *ns18="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns19="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns20="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns21="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns22="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns23="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns24="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns25="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns26="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns27="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 1 0 0 0"

   *ns28="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 1 0 0 0"

   *ns29="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns30="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns31="CONDITION"`` hierarchically corresponding line condition
   sectionname="DESIGN LINE DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns32="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns33="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns34="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns35="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns36="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns37="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 0.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

   *ns38="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 1 0 0 0"

   *ns39="CONDITION"`` hierarchically corresponding point condition
   sectionname="DESIGN POINT DIRICH CONDITIONS"
   description="NUMDOF 4 ONOFF 1 1 1 0 VAL 1.0 0.0 0.0 0.0 FUNCT 0 0 0 0"

Delete the remaining items ``eb1``, ``ns17`` and ``ns1`` to ``ns9``.

Fluid Dirichlet conditions differ from structure Dirichlet conditions
only in the meaning of the first values. Instead of displacements, here
velocities (two in 2d, three in 3d) and pressure values can be
described.

Safe the file under a different name, e.g. ``dc_fluid.bc``.


Creating |FOURC| Input File and Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, we create the |FOURC| input file by the command line tool
``./pre_exodus --exo=dc.e --head=dc_fluid.head --bc=dc_fluid.bc --dat=dc_fluid.dat``
where the filenames might have to be replaced accordingly.
This will result in the specified dat-file which is already validated to be accepted by |FOURC|.
However, if the file is meaningful cannot be assured.

Run the simulation by providing this dat-file and an output file to |FOURC|
and postprocess the results (refer to :ref:`FSI 3D Tutorial Postprocessing <fsi3dtutorialpostprocessing>`).

The FSI Part
------------

Again, edit ``default.head`` as outlined below.
However to create the ``bc-file`` we copy
together the existing parts from ``dc_struct.bc`` and ``dc_fluid.bc`` and
change only the necessary coulping conditions as shown below.

.. _header-file-2:

header-file
~~~~~~~~~~~~~~~

Find the following sections in ’default.head’ and edit as given:

   ::

         -----ALE DYNAMIC

   set ``LINEAR_SOLVER 1``

   ::

         -----FLUID DYNAMIC

   set ``LINEAR_SOLVER 2``

   ::

      -----FSI DYNAMIC

   set ``NUMSTEP = 50``

   set ``TIMESTEP = 0.1``

   ::

         -----STRUCTURAL DYNAMIC

   set ``DYNAMICTYPE     GenAlpha``

   set ``LINEAR_SOLVER 3``

   ::

      -----MATERIALS

   insert
   ``MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 1000.0 NUE 0.3 DENS 500``

   insert ``MAT 2 MAT_fluid  DYNVISCOSITY 0.01 DENSITY 1.0``

   insert
   ``MAT 3 MAT_Struct_StVenantKirchhoff YOUNG 500.0 NUE 0.3 DENS 500``

   ::

      -----CLONING MATERIAL MAP

   insert ``SRC_FIELD fluid SRC_MAT 2 TAR_FIELD ale TAR_MAT 3``

   ::

      -----FUNCT1

   insert ``SYMBOLIC_FUNCTION_OF_SPACE_TIME (1-cos(2*t*pi/5))``
   defining time-dependent inflow and lid movement

   ::

      -----FUNCT2

   insert ``SYMBOLIC_FUNCTION_OF_SPACE_TIME 10*(y-1)*(1-cos(2*t*pi/5))``
   representing the spatial inflow distribution

   ::

      -----SOLVER 1

   set ``NAME Ale solver``

   set ``SOLVER UMFPACK``

   ::

      -----SOLVER 2

   set ``NAME Fluid solver``

   set ``SOLVER Belos``

   ::

      -----SOLVER 3

   set ``NAME Structural solver``

   set ``SOLVER UMFPACK``

Safe the file under a different name, e.g. ’dc_fsi.head’.


bc-file
~~~~~~~~~~~

We assume that you merged the fluid und structure bc-files, so the following entities have
to be changed:

-  ``*eb1="ELEMENT"`` the structure elements with their material

   .. container:: small

      ::

          sectionname="STRUCTURE"
          description="MAT 1 KINEM nonlinear TECH eas_full"
          elementname="SOLID"

-  ``*eb2="ELEMENT"`` the fluid elements with ALE and their material

   .. container:: small

      ::

         sectionname="FLUID"
         description="MAT 2 NA ALE"
         elementname="FLUID"

-  ``*ns5="CONDITION"`` the coupling surface from the structure side

   .. container:: small

      ::

         sectionname="DESIGN FSI COUPLING SURF CONDITIONS"
         description="coupling_id 1 "

-  ``*ns14="CONDITION"`` the coupling surface from the fluid side

   .. container:: small

      ::

         sectionname="DESIGN FSI COUPLING SURF CONDITIONS"
         description="coupling_id 1 "

-  ``*ns22="CONDITION"`` release the ’coupling-line’ for inplane
   directions

   .. container:: small

      ::

         sectionname="DESIGN LINE DIRICH CONDITIONS"
         description="NUMDOF 4 ONOFF 0 0 1 0 VAL 0.0 0.0 0.0 0.0  FUNCT 0 0 0 0"

-  ``*ns23="CONDITION"`` release the ’coupling-line’ for inplane
   directions

   .. container:: small

      ::

         sectionname="DESIGN LINE DIRICH CONDITIONS"
         description="NUMDOF 4 ONOFF 0 0 1 0 VAL 0.0 0.0 0.0 0.0  FUNCT 0 0 0 0"

Further, you need to provide Dirichlet conditions to the automatically
created ALE field. The displacement of the ALE field is restricted to
zero at the left, right and top of the computational domain. At the
front and the back plane, zero displacement in z-direction is demanded.
Important: There are no Dirichlet conditions for ALE at the bottom of
the cavity, since this is the FSI coupling interface. Thus, please add
the following condition definitions to your new ’bc-file’:

-  ``*ns10="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns11="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns12="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 0 0 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns13="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 0 0 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns15="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns16="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns17="CONDITION"``

   .. container:: small

      ::

          sectionname="DESIGN SURF ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns18="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns19="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns20="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns21="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns24="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns25="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns26="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns27="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns28="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns29="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns30="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns31="CONDITION"`` hierarchically corresponding line condition

   .. container:: small

      ::

          sectionname="DESIGN LINE ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns36="CONDITION"`` hierarchically corresponding point condition

   .. container:: small

      ::

          sectionname="DESIGN POINT ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns37="CONDITION"`` hierarchically corresponding point condition

   .. container:: small

      ::

          sectionname="DESIGN POINT ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns38="CONDITION"`` hierarchically corresponding point condition

   .. container:: small

      ::

          sectionname="DESIGN POINT ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

-  ``*ns39="CONDITION"`` hierarchically corresponding point condition

   .. container:: small

      ::

          sectionname="DESIGN POINT ALE DIRICH CONDITIONS"
         description="NUMDOF 3 ONOFF 1 1 1 VAL 0.0 0.0 0.0  FUNCT 0 0 0 "

As any of these conditions matches an already defined NodeSet it will
also match the corresponding ’E-id’ in the later |FOURC| input file.
Finally save the file under a different name, e.g. ’dc_fsi.bc’.

Creating |FOURC| Input File and Running the Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a third time, run the command
``./pre_exodus --exo=dc.e --head=dc_fsi.head --bc=dc_fsi.bc --dat=dc_fsi.dat``
where the filenames might have to be replaced accordingly. This will
result in the specified dat-file which is already validated to be
accepted by |FOURC|.

Run the simulation by providing the created dat-file and an output file
to |FOURC| and postprocess the results.


.. _fsi3dtutorialpostprocessing:

Postprocessing
--------------

You can postprocess your results with any visualization software you
like. In this tutorial, we choose *Paraview*.

Before you can open the results, you have to generate a filter again.
Call *make post_drt_ensight* in the |FOURC|-directory. Filter your results
in the output directory with the call

.. container:: center

   ``./post_drt_ensight --file=[outputdirectory]/outputprefix``

After this open *paraview*, go to

-  *File* :math:`\to` *Open Data* and select the filtered *\*.case
   file*.

-  Only for older versions of *Paraview*:

   -  Select the time step in the *Select Time Value* window on the left
      and

   -  shift *Byte order* to *little endian*

-  Click on *accept* (or *apply*) to activate the display.

-  In the *Display tab* (section *Color*) you can choose now between
   *Point pressure* and *Point velocity*, whatever you want to display.

-  For the scale, activate the *Scalar bar* button in the *View
   section*.

