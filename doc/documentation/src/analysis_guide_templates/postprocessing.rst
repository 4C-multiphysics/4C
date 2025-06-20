.. _postprocessing:

Postprocessing
----------------

|FOURC| supports three types of output for simulation results:

- :ref:`direct vtk/vtu output <directvtkoutput>`
- :ref:`Conversion to common formats  <conversiontoreadableformats>`: |FOURC| provides binary output in a proprietary format, which can be converted into other formats for general post processing
  by converters provided as extra tools within the software suite.
- :ref:`post_monitor <postmonitor>`: traditional text file output of nodal result data.

Commonly the results are visualized with the open-source post processing tool :ref:`**Paraview** <paraview>`.

.. _directvtkoutput:

Direct VTK output
~~~~~~~~~~~~~~~~~~~

Direct VTK output can be enforced by the associated sections in the input file.
The general information that VTK output is to be written, is given in the section
:ref:`IO/RUNTIME VTK OUTPUT <SECio_runtimevtkoutput>`.

For the different discretizations (Fluids, Solids, Beams), specific subsections define output quantities
(click on the links to see the respective reference information):

- :ref:`IO/RUNTIME VTK OUTPUT/STRUCTURE <SECio_runtimevtkoutput_structure>`.
- :ref:`IO/RUNTIME VTK OUTPUT/BEAMS <SECio_runtimevtkoutput_beams>`.
- :ref:`IO/RUNTIME VTK OUTPUT/FLUID <SECio_runtimevtkoutput_fluid>`.


.. _conversiontoreadableformats:

Conversion to readable formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some fields, the direct VTK output is not yet supported,
but |FOURC| can write result data in a proprietary binary format.
In order to make the simulation results accessible,
they have to be converted using the ``post_processor`` script located in the |FOURC| build directory.
As per default, running this script on the result (.control) file of the simulation, converts the results to the Ensight format.
**Ensight** is a post processor provided by *Ansys*, mainly invented for fluid mechanics visualizations.

The ``post_processor`` script has a lot of options to specify the post processing details.
A complete list of these options is printed to the screen when executing ``./post_processor --help`` .
Most of these options are self-explanatory. Other output format filters are available via the ``--filter`` option.
Here the vtu filter is of particular interest when you want to convert the results to VTK file format,
and did not provide the direct vtk output (see :ref:`above <directvtkoutput>`).

.. Note::
    When using the |FOURC| post processing script ``post_processor``,
    be aware that derived quantities like e.g. stresses and strains are not automatically extracted from the simulation results!
    Presumed you set the proper flags in the input file (i.e. the quantities were actually calculated),
    you still have to set the ``--stress`` etc. options of the post processing script to extract these quantities from the simulation results.

You always have to provide the control file,
which is provided from the simulation and stores information about all available time steps and result quantities,
by the parameter ``--file=<problemname>`` without the suffix ``.control``.
Additional post-processing options (output from ``post_processor --help``) are given in the following table:

+----------------+-----------+---------+-------------------------------------------------+
| Parameter      |value type |default  |Explanation                                      |
+================+===========+=========+=================================================+
|--file          |string     |"xxx"    | basename of the control file (suffix: .control) |
+----------------+-----------+---------+-------------------------------------------------+
|--filter        |string     |"ensight"|"ensight" | "vti" | "vtu"                        |
+----------------+-----------+---------+-------------------------------------------------+
|--start         |int        |0        |                                                 |
+----------------+-----------+---------+-------------------------------------------------+
|--end           |int        |-1       |Note that other negative numbers are not allowed!|
+----------------+-----------+---------+-------------------------------------------------+
|--step          |int        |1        |                                                 |
+----------------+-----------+---------+-------------------------------------------------+
|--output        |string     |"xxx"    |output file name                                 |
+----------------+-----------+---------+-------------------------------------------------+
|--stress        |string     |""       |cxyz, ndxyz, cxyz_ndxyz, c123, nd123, c123\_nd123|
+----------------+-----------+---------+-------------------------------------------------+
|--strain        |string     |""       |see stress                                       |
+----------------+-----------+---------+-------------------------------------------------+
|--mortar        |string     |"no"     |"no" \|"yes"                                     |
+----------------+-----------+---------+-------------------------------------------------+
|--optquantity   |string     |""       |cxyz, ndxyz, cxyz\_ndxyz                         |
+----------------+-----------+---------+-------------------------------------------------+
|--heatflux      |string     |""       |see stress                                       |
+----------------+-----------+---------+-------------------------------------------------+
|--tempgrad      |string     |""       |see stress                                       |
+----------------+-----------+---------+-------------------------------------------------+
|--structvelacc  |string     |"no"     |"no" \| "yes"                                    |
+----------------+-----------+---------+-------------------------------------------------+
|--rotation      |string     |"no"     |"no" \| "yes"                                    |
+----------------+-----------+---------+-------------------------------------------------+
|--structmatdisp |string     |"no"     |"no" \| "yes"                                    |
+----------------+-----------+---------+-------------------------------------------------+
|--outputtype    |string     |         |bin  \| ascii (only for vtu)                     |
+----------------+-----------+---------+-------------------------------------------------+

For the respective filters, there are wrappers available:

+-------------+--------------------------------+
|wrapper      |is equal to                     |
+=============+================================+
|post_ensight |post_processor --filter=ensight |
+-------------+--------------------------------+
|post_vti     |post_processor --filter=vti     |
+-------------+--------------------------------+
|post_vtu     |post_processor --filter=vtu     |
+-------------+--------------------------------+


Process output steps from ``start`` to ``end`` every ``step``. Works on
real time steps, steps not written by |FOURC| are counted, too. Both
``start`` and ``end`` can be empty, in which case the filter will
process from the first and to the last step, respectively.

.. _postmonitor:

post_monitor
~~~~~~~~~~~~

This tiny tool generates files containing an ascii table with tab separators for simple gnuplot output of nodal results:

::

   ./post_monitor [options]

The options are very similar to the ones of ``post_processor```, however, the output is very limited.
Only those variables can be requested, which are originally nodal variables;
integration point variables are not used, even if ``--stress=ndxyz`` is given on the command line
(an error message is not given anyway).

Actually, the only important options are::

    --file=<controlfilename>              # can be given without .control suffix
    --field=<fieldname>                   # can be any field type like structure, fluid, scatra, etc.
    --node=<nodenumber>
    --output=<filename>                   # if not given, the control file basename with suffix .mon is used

The options ``--start``, ``--step`` and ``--end`` may also be useful here.

 .. _paraview:

ParaView
~~~~~~~~~~

Paraview <https://www.paraview.org> can read various post processing data formats.
Particularly, it can read the directly written *vtk/vtu* format, and also the *ensight* format,
which one can extract by converting the default output with ``post_ensight``, see above.

The capabilities of paraview are to diverse to even mention them here.
A number of ParaView tutorials are available in the internet::

   http://www.paraview.org/Wiki/The_ParaView_Tutorial


Animations
~~~~~~~~~~

.. The ultimate goal of scientific research is a beautiful movie!

There are several way to create animations using |FOURC| output files.
Movies should be playable across platforms (at least Linux/windows/macOS)
and embeddable inside MS Powerpoint presentations
without the need of having different movie versions in different formats.

Since the tools to create videos is vast, here we will simply give a rough overview of a few tools that are used frequently.

**Recoding videos**

If you have stored a video and just want to have it in a different format, or with different framerate, resolution, etc.,
you may use one of the following tools.

Command line based:

- ffmpeg
- mencoder

With GUI (freeware):

- Blender (all OS)
- Shortcut (Windows)

**Video editing**

If you want to join, cut or render videos in a fancy way, add subtitles, or provide picture-in-picture, etc.,
you might want to use a tool with more features, which then comes with a graphical user interface:

- Blender (all OS)
- Shortcut (Windows)

**Videos from Pictures**

Of course, it's again Blender which can do it all, but if it's only about combining pictures,
``ffmpeg`` does the same more easily (note that a number of pictures can be added by format specifiers)::

    ffmpeg -i resultframe%03d.png -r 25 resultframes.mpg

Here, the framerate is set to 25 per second, and all pictures should be ordered by its number,
which then of course must be given by 3 digits.