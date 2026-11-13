.. _Installation:

Installation
============

|FOURC| is developed and used on Linux. It can also be compiled under Windows through the Windows Subsystem for Linux
(WSL), which provides a complete Linux environment within Windows.

Choose the path that matches your goal:

For users
---------

:doc:`Install 4C for simulations <installation_user>`
   Run a ready-to-use 4C Docker container or install a released version and its dependencies natively with Spack.

.. toctree::
   :hidden:

   installation_user

For developers
--------------

:doc:`Set up 4C with automatic TPL installation <installation_developer_auto_tpl>`
   Create a contributor checkout and use Spack to automatically install the external dependency stack.

:doc:`Set up 4C with manual TPL installation <installation_developer_manual_tpl>`
   Create a contributor checkout and install each external dependency yourself.

.. toctree::
   :hidden:

   installation_developer_auto_tpl
   installation_developer_manual_tpl
   installation_developer_workflow