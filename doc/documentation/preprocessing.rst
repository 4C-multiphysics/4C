.. _preprocessing:

Preprocessing
---------------

|FOURC| reads the mesh, boundary conditions, materials and simulation parameters from an ASCII file in a proprietary format,
which usually has the suffix ``.dat``, but this suffix is not necessary, it can be anything.

There are not so many means to create a valid input file. At this point, we know of the following
different ways to create an input file. In general, you'll have two options:

#. Either you create the input file in |FOURC|'s native format directly,
#. or you create an input file in a general binary format for finite element information, called ``Exodus II``, develeloped by `Sandia National Laboratories
   <https://www.sandia.gov/files/cubit/15.8/help_manual/WebHelp/finite_element_model/exodus/exodus2_file_specification.htm>`_.
   This can be converted into |FOURC| s native format by an internal converter, :ref:`pre_exodus <pre_exodus>`.

Since the conversion from the ``Exodus II`` format is the most versatile way to generate a |FOURC| input file, this method is explained first.

.. _pre_exodus:

Exodus II to |FOURC| file conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main procedure to generate a valid |FOURC| input file is from a binary mesh file in ``Exodus II`` format, see above,
which includes nodes, elements, node sets, element sets, and side sets, as can be created by Cubit.
Since this file does not contain boundary conditions, element type information, materials and solver details,
additional information is needed, which is given by two additional files:

#. for the global system parameters (solver, material, step information, etc.), called the *headerfile*, and

#. a file for the correlation between element sets and type declarations, as well as boundary conditions definitions.
   This is the so-called the *bcfile*.

These three files are merged into an input file for |FOURC| by the program ``pre_exodus``.
The created |FOURC| input file is then *automatically* validated using all available |FOURC| validation and is therefore likely to run.

The program is created together with |FOURC| executable, if ``make full`` has been invoked,
but it can also be compiled solely by ``make pre_exodus``.

::

   $> pre_exodus --exo=<exodusfile> --bc=<bcfile> --head=<headerfile> --dat=<4Cinput> \
               [ --d2 | --d3  ]           \
               [ --quadtri  ]             \
               [ --gmesh=<startelement> ]


In general one might not have already a proper *header-file* and matching *bc-file*. By typing

``./pre_exodus --exo=<yourmesh>.e``

two preliminary files ``default.head`` and ``default.bc`` are created.
The first contains the currently valid header parameters with default values and commented options
which you can edit to adapt it to your means.
Similarly, ``default.bc`` consists of all your mesh entities and a list of all currently valid conditions.
See next section for details how to work with it and how to get valid input files.

.. note::
   When you have an already existing input file, you can always validate it by simply executing ``./pre_exodus --dat=inputfile.dat``,
   before(!) you start a parallel |FOURC| computation on a cluster, for example.

*Optional parameters*

The optional parameter ``--quadtri`` reads the exodus file and converts all quad elements in two triangular elements.
It does not write a dat file, but writes a new exodus file instead named ``tri_<problemname>.e``.
NOTE: This feature is only for 2D elements, it does **not** modify 3D elements.


Generating ``Exodus II`` files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even though the generation of ``Exodus II`` files might be out of scope of a |FOURC| manual,
users are informed on how to generate these files conveniently, so options are given in the following:

.. _cubit:

**CUBIT**


CUBIT `<http://cubit.sandia.gov/>`_ is a powerful pre- postprocessing
tool. (The commercial version of the software was called *Trelis*,
but has been renamed into CUBIT now as well, so we may stick to the name CUBIT).

CUBIT can create ``Exodus II`` files which can be converted into a
valid |FOURC| inpufile using the pre_exodus filter, so the preprocessing is a two step process:

#. Cubit
   - create geometry, mesh, and necessary node sets
   - export to exodus file format (\*.e)
#. :ref:`pre_exodus <pre_exodus>`
   - define appropriate boundary conditions and element types
   - convert into a |FOURC| \*.dat file.

Note that

- it is not necessary to define boundary conditions in Cubit, since they are not converted
  to the dat file later on.

- you should only define node sets, but not sidesets (surface sets). The node sets are
  converted into surface sets if the surface definition is given in the boundary condition
  control file (<problem>.bc) anyway.


**Other Software**

Geometry as well as element and node sets can be created in any finite element preprocessor.
However, the preprocessor should be capable of exporting a file format, which can be converted
by the python toolset meshio (see <https://pypi.org/project/meshio/>) into an exodus file, with
which the input can be converted into a |FOURC| .dat file.

Also, the exported input file can probably be imported in Cubit, then further edited and
eventually exported as an exodus (.e) file.

So the steps are

#. Create finite element model and sets in your favorite preprocessor

#. Export to some format, like ``Exodus II`` or the Gmesh format ``.msh`` file.

#. **Optional:** Read in the model to Cubit for further editing

#. **Optional:** If you are not able to write in ``Exodus II`` format,
   use the python module meshio (packed in pip) to convert the mesh to an exodus (.e) file
   (<https://pypi.org/project/meshio/>)

#. Run ``pre_exodus`` from your |FOURC| build to convert the data (see above).


.. _create4Cinput:

Other ways to create a |FOURC| input directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _abaqus:

**ABAQUS**

There is an in-house Python module ``abaqus_meshio`` for the conversion from an ABAQUS input file (``.inp``) to dat file.
This python module is available in the `scripts gitlab <https://gitlab.lrz.de/baci/scripts>`_.
Since the ``.inp`` file can also be generated using Cubit, this submodule can be used in conjunction with Cubit as well, see above.
The usage of this submodule starts firstly by importing it providing the path where it is located.

.. code-block:: python

   import sys

   abaqus_meshio_path = "path_to_abaqus_meshio"
   sys.path.append(abaqus_meshio_path)

Subsequently, the inp shall be read using the command

.. code-block:: python

   model = abaqus_meshio.read("path_to_inp.dat")

Unlike ``meshio.read``, the command ``abaqus_meshio.read`` will return a model, which is instance of ``BModel``, where:

- ``model.rootAssembly.instances[instance_name].mesh`` is a ``BMesh. ``BMesh`` is a subclass of ``meshio.Mesh``
  with additional attributes sections (for material assignment) and surfaces (for distributed load).
- ``model`` has attributes materials (from MATERIAL), parts (from PART/END PART) and steps (from STEP)
- ``model.parts[part_name].mesh`` is again a ``BMesh``, ``model.rootAssembly.instances[instance_name].mesh`` is a transformation of this mesh.

``BModel`` is designed to mimic the way Abaqus systematically stores its data. To access the original ``meshio.Mesh`` one has to use ``model.parts[part_name].mesh``.

Proving that the information from inp is properly stored, the transformation to dat file is done by a simple command

.. code-block:: python

   fourc_io = abaqus_meshio.Inp2Baci(model, [params_step_1])
   fourc_io.write("prefix")

If the inp has many steps defined by STEP/END STEP keywords, the list of parameters for each step has to be provided,
e.g. ``[params_step_1, params_step_2, ...]``.
Default parameters for a structural analysis can be obtained using

.. code-block:: python

   params_step_1 = abaqus_meshio.GenerateDefaultParams()


Modify |FOURC| input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|FOURC| input files are text files so you can modify them using your
favorite text editor. You can see all possible parameters and keywords in the
:ref:`reference part <inputparameterreference>`.

.. However, sometimes you might want some more
.. modifications (e.g. modifying many nodes coordinates) that might be better
.. done by a script. And indeed there is a python script that can help you editing input files.


