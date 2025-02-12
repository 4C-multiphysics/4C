.. _tutorials:

==========
Tutorials
==========


.. note::

    All corresponding files for the tutorials are located in the |FOURC| subfolder `<4C-sourcedir>/tests/framework-tests/*`.

Most of the tutorials use Cubit to generate the finite element geometry, so we will introduce one possible pre-processing method here for the general case.
The output format used here is Exodus II, which is a binary format. This cannot be read by |FOURC| directly,
but a converter within the |FOURC| framework, ``pre_exodus`` is used to generate a valid input file.
You'll find more information about this converter in the :ref:`Preprocessing section <preprocessing>`.

The following tutorials are available to show different features of |FOURC|.

.. toctree::
   :maxdepth: 2

   tut_introduction
   tut_fluid_preexo
   tut_fsi_preexo
   tut_fsi_preexo_2d
   tut_solid_preexo
   tutorial_contact_3d
   tut_battery
