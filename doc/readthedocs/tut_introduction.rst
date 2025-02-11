
Creating a Geometry with Cubit
===============================

Cubit is a finite element preprocessor that may not only do the meshing on an existing geometry,
but aldo create geometries by many different procedures, and it can also read in and modify CAD files for further treatment.
It supports scripting (also Python), therefore we provide a *Journal*-file containing the necessary geometry commands as well as mesh and definitions for elements and boundary conditions, respectively.

For all the tutorials, a so-called journal file is available, which makes it possible to reproduce the mesh generation step by step.
Within Cubit, open the Journal-Editor (*Tools*\ :math:`\to`\ *Journal Editor*), paste the text from the journal file and press *play*.
For later usage it is convenient to save the current content of the Journal-Editor into a *\*.jou* file.
Sometimes the export to an exodus mesh file is already contained in the journal file;
if not, export the created mesh to an exodus-file via *File*\ :math:`\to`\ *Export...*.
During export, one has to set the dimension explicitly to 2D, if the structure is 2D.
