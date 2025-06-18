.. _preprocessing:

Preprocessing
---------------

|FOURC| reads the mesh, boundary conditions, materials and simulation parameters from a central
input file.

.. admonition:: Under development

    A large refactoring effort is currently in progress to improve |FOURC|'s input files.
    The information in this section is not fully updated to reflect these changes.

There are not so many means to create a valid input file. At this point, we know of the following
different ways to create an input file. In general, you'll have two options:

#. Either you create the mesh in |FOURC|'s native format directly,
#. or you create a mesh file in a general binary format for finite element information, called EXODUS II, develeloped by `Sandia National Laboratories
   <https://www.sandia.gov/files/cubit/15.8/help_manual/WebHelp/finite_element_model/exodus/exodus2_file_specification.htm>`_.
   This can be read into |FOURC| via its input file.

Generating ``EXODUS II`` files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even though the generation of ``EXODUS II`` files might be out of scope of a |FOURC| manual,
users are informed on how to generate these files conveniently, so options are given in the following:

.. _cubit:

**CUBIT**


CUBIT `<http://cubit.sandia.gov/>`_ is a powerful pre- postprocessing
tool. (The commercial version of the software was called *Trelis*,
but has been renamed into CUBIT now as well, so we may stick to the name CUBIT).

Cubit allows you to create the geometry, mesh, and necessary node sets and export them to
the EXODUS file format.

Note that

- it is not necessary to define boundary conditions in Cubit. This can be done directly in the |FOURC| input file.

- you should only define node sets, but not sidesets (surface sets). Side sets are not yet
  supported in |FOURC|.


**Other Software**

Geometry as well as element and node sets can be created in any finite element preprocessor.
However, the preprocessor should be capable of exporting a file format, which can be converted
by the python toolset meshio (see <https://pypi.org/project/meshio/>) into an EXODUS file.

.. admonition:: Warning

    The EXODUS file that is exported by meshio can currently not read in by |FOURC| directly!

Also, the exported input file can probably be imported in Cubit, then further edited and
eventually exported as an EXODUS (.e) file.

So the steps are

#. Create finite element model and sets in your favorite preprocessor

#. Export to ``EXODUS II`` format if possible.

#. If you can only export to some other format than EXODUS II:
   - **Option 1** Read in the model to Cubit for further editing and write out an EXODUS II file.
   - **Option 2** Currently you should write a python script to convert your format to the |FOURC| input file format.


.. _create4Cinput:

Other ways to create a |FOURC| input directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _abaqus:

**ABAQUS**

.. admonition:: Outdated content

    The abaqus converter produces only the old proprietary input file format.
    If an update is desired, an issue should be opened.

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


