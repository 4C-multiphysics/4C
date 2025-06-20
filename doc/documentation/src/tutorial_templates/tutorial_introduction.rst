
Creating a Geometry with Coreform Cubit\ |reg|
===================================================

Coreform Cubit\ |reg|  is a finite element preprocessor that may not only do the meshing on an existing geometry,
but also create geometries by many different procedures, and it can also read in and modify CAD files for further treatment.
It supports scripting (in its own language, but also Python), therefore we provide a *Journal*-file for all the tutorials containing the necessary geometry commands
as well as mesh and definitions for elements and boundary conditions, respectively.
These journal files, located in ``<4C-source>/tests/tutorials``, make it possible to reproduce the mesh generation step by step and even modify the example.

Within Cubit, you may paste single lines into the command line window (at the bottom of the Cubit window) to run only short sequences,
or open the Journal-Editor (*Tools*\ :math:`\to`\ *Journal Editor*), load the file from the `tests/tutorials` directory
(or paste the text from the journal file) and press *play* to run the whole file at once.

Sometimes the export to an EXODUS mesh file is already contained in the journal file;
if not, export the created mesh to an EXODUS file via *File*\ :math:`\to`\ *Export...*.

If you changed some lines/commands from the journal file, it is convenient to save the current content of the Journal-Editor into a *\*.jou* file for later usage.