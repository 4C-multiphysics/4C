.. _tutorials:

==========
Tutorials
==========


.. note::

    All corresponding files for the tutorials are located in the |FOURC| subfolder `<4C-sourcedir>/tests/tutorials/*`.

Most of the tutorials use Cubit to generate the finite element geometry,
so we will introduce one possible pre-processing method here for the general case.
The output format used here is the binary `EXODUS <https://sandialabs.github.io/seacas-docs/html/index.html>`_
format.

The following tutorials are available to show different features of |FOURC|.
If you are new to |FOURC|, we recommend to start with a tutorial that fits to your application:

- Scalar transport or thermal analysis: start with the :ref:`Poisson tutorial <poissontutorial>`.
- Solid mechanics: start with the :ref:`solid mechanics tutorial <3d_solidtutorial>`.
- Contact mechanics: start with the :ref:`solid mechanics tutorial <3d_solidtutorial>`
  and proceed to the :ref:`3D contact tutorial <contacttutorial>`.
- Fluid dynamics: start with the :ref:`fluid mechanics tutorial <fluidtutorial>`.

The other tutorials are more advanced and show some of the multiphysics capabilities of |FOURC|.

- FSI: Connection between fluid and solid mechanics in 2D and 3D as well as using a monolithic approach
  (:ref:`FSI 2D tutorial <2d_fsitutorial>`, :ref:`FSI 3D tutorial <3d_fsitutorial>`,
  :ref:`FSI tutorial using a monolithic approach <monolithic_fsitutorial>`).
- Battery simulation: Coupled thermal, mechanical and electrochemical simulation of a battery
  (:ref:`battery tutorial <batterytutorial>`).

Preconditioning and iterative solvers:
learn how to efficiently solve large linear systems in |FOURC| by comparing different preconditioner options
(:ref:`preconditioning tutorial <preconditioningtutorial>`).

Here is the outline of all available tutorials:

.. toctree::
   :maxdepth: 2

   tutorial_introduction
   tutorial_poisson
   tutorial_fluid
   tutorial_solid
   tutorial_contact_3d
   tutorial_fsi_2d
   tutorial_fsi_3d
   tutorial_fsi_monolithic
   tutorial_battery
   tutorial_preconditioning