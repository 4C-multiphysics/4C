========
|FOURC|
========

Vision
======

We aim to advance the frontiers of computational science and engineering by providing a versatile, extensible and open-source research software framework for the systematic development, analysis, and application of advanced numerical methods for modeling and simulation of complex multiphysics phenomena across scales and disciplines.

Mission
=======

|FOURC| Multiphysics is a modular, parallel and open-source simulation environment tailored to the needs of researchers and computational scientists to enable and accelerate research in computational science and engineering. Our mission is to:

- support the formulation and enable the rigorous study of complex single- and multiphysics models across spatial and temporal scales through a variety of physical models, numerical methods, and coupling algorithms with a strong focus on finite element methods and particle methods accompanied by comprehensive documentation, tutorials and a welcoming culture to ensure a low entry barrier;
- curate and advance a modular and extensible framework to develop mathematical models for challenging real-world problems in science, engineering and biomedicine described by differential equations and to devise and implement novel numerical methods with a clear focus on methodological innovation and practical usability;
- offer a platform for both simulation practitioners, studying real-world problems through numerical simulation, as well as researchers in numerical modeling and computational methods, aiming at the development of accurate models and innovative numerical methods and their efficient software implementation in the support of complex real-world scenarios;
- enable parallel and scalable computations on workstations and clusters to increase efficiency and utilization of available hardware resources with a strong focus on medium- and large-scale practical applications;
- foster a growing international research community in which engineers, scientists, and domain experts can cooperate, contribute, accelerate scientific discovery, and share advances in computational modeling and numerical method development and are committed to open scientific exchange, collaborative development, and sustainable software practices.

Content
=======

This guide to |FOURC| is structured as follows:

:ref:`About 4C<about>`
   Learn about the capabilities and history of |FOURC|.

:ref:`The 4C Community<4Ccommunity>`
   A brief summary of the roles and responsibilities within the |FOURC| community.

:ref:`Installation<Installation>`
   A summary of all requirements of |FOURC| and detailed steps how to build |FOURC|.

:ref:`Tutorials<tutorials>`
   A series of beginner-level tutorials showcases the setup procedure for specific application scenarios.

:ref:`Analysis guide<analysisguide>`
   Detailed explanations on the whole tool chain from model generation (pre-processing)
   over running a simulation to the evaluation of results (post-processing) offers deep insight into using |FOURC|
   for advanced simulation scenarios.
   This guide includes background information and detailed descriptions
   for the specification of elements, boundary conditions, constitutive laws
   as well as options for linear and nonlinear solvers.

:ref:`Developer guide<developerguide>`
   This guide gets you started on actively developing and contributing to |FOURC|.
   It covers our CI/CD testing infrastructure, coding guidelines, and useful tools for the daily development of |FOURC|.

:ref:`Input Parameter Reference<inputparameterreference>`
   A comprehensive list of all input parameters, elements, materials, and boundary conditions
   with short descriptions for each option

:ref:`Tools<tools>`
   A collection of useful tools for working with |FOURC|

:ref:`Appendix<appendix>`
   Information on contributing to this documentation as well as selected topics of interest

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   about/about
   community/4Ccommunity
   installation/installation
   tutorials/tutorials
   analysis_guide/analysisguide
   developer_guide/developmentguide
   reference_guide/parameterreference
   tools/tools
   appendix/appendix
