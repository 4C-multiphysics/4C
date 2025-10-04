.. _preprocessing:

Preprocessing
---------------

|FOURC| reads the mesh, boundary conditions, materials and simulation parameters from a central
input file.


Working with |FOURC| input files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|FOURC| input files are text files in the YAML format. Since YAML is a superset of JSON,
you may also use JSON syntax although we tend to prefer YAML for human-readability.
By convention, |FOURC| input files have the extension ``.4C.yaml`` or ``.4C.json``.

We build a schema called ``4C_schema.json`` which is located in the root build directory. This
schema describes the structure of a valid |FOURC| input file.
If you want to write and edit |FOURC| input files, we recommend to set up schema validation in your editor
(see :ref:`installation` for details). Doing so provides documentation and autocompletion
for all parameters.


Creating meshes for |FOURC|
~~~~~~~~~~~~~~~~~~~~~~~~~~~

|FOURC| can read meshes in two different ways:

#. Either you create the mesh in |FOURC|'s native format directly. Refer to the :ref:`tools` section
   to find tools that can help you to create |FOURC| input files.
#. or you directly read the mesh from standardized mesh formats. |FOURC| supports reading vtu-files,
   or EXODUS II.

Generating ``VTU`` files
~~~~~~~~~~~~~~~~~~~~~~~~

VTU is a standard file format for unstructured grids and is part of the VTK library. VTU files
can be generated from various mesh formats using the Python library `meshio <https://pypi.org/project/meshio/>`_,
or they can often be exported directly from meshing software.

|FOURC| uses element blocks to assign physical fields and their properties to subsets of elements
in the mesh. The VTU file must contain a scalar cell data array called ``block_id``, which assigns
an integer ID to each element.

.. admonition:: Warning

   Each block may only contain elements of the same type (e.g., all hexahedra or all tetrahedra).
   For mixed meshes, multiple block IDs must be defined. Multiple blocks can be assigned to the same
   physical field in the |FOURC| input file.

To define boundary conditions, |FOURC| relies on the definition of point sets. |FOURC| creates
point sets from integer-type point data arrays named ``point_set_<id>``. A point is interpreted as
belonging to a set if the corresponding entry in the point data array is nonzero.


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