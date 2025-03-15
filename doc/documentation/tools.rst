.. _toolsAndScripts:

=================
Tools and Scripts
=================

A number of tools and scripts are available for |FOURC|, some of which are located in the scripts repository.
They are explained briefly in this section.

abaqus to |FOURC| input converter
=================================

The converter for abaqus input files is based on meshio, but contrary to the common meshio strategy
not only the mesh is converted to the .dat file, but also the boundary conditions and step data.

|FOURC| converter to other formats
===================================

- **dattoensight:** Converts a dat file into ensight control and data files, so that the input (including all node and side sets)
  can be viewed, e.g., in paraview.

- **dat2mesh:** Converts a .dat file into a ``gmesh`` input file, so that it can be read by other pre processors.



