.. _teuchos-time-monitor:

Teuchos Time Monitor
--------------------------

The ``TimeMonitor`` from ``Trilinos`` package ``Teuchos`` provides MPI collective timer statistics.
Refer to the ``Teuchos::TimeMonitor`` Class Reference https://trilinos.org/docs/dev/packages/teuchos/doc/html/classTeuchos_1_1TimeMonitor.html for a detailed documentation.

Add a timer for a method in |FOURC|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to get parallel timing statistics of a method in |FOURC| include the following header

::

    #include <Teuchos_TimeMonitor.hpp>


in the ``.cpp``-file that contains the implementation of the method and add the macro ``TEUCHOS_FUNC_TIME_MONITOR``
with the name of the method in the implementation of the method:

.. code-block:: cpp

    void <NAMESPACE>::<FunctionName>(...)
    {
      TEUCHOS_FUNC_TIME_MONITOR("<NAMESPACE>::<FunctionName>");

      /* implementation */
    }


Running a simulation on 3 processors for example yields the following ``TimeMonitor`` output in the terminal:

::

    ============================================================================================================

                                       TimeMonitor results over 3 processors

    Timer Name                   MinOverProcs       MeanOverProcs      MaxOverProcs       MeanOverCallCounts
    ------------------------------------------------------------------------------------------------------------
    <NAMESPACE>::<FunctionName>  0.1282 (1000)      0.2134 (1000)      0.2562 (1000)      0.0002132 (1001)
    ============================================================================================================


The output gives the minimum, maximum, and mean ``execution time (number of counts)`` for all processors and also the mean execution time over all counts.

**Note:** The ``TimeMonitor`` output of a |FOURC| simulation in general already contains a variety of methods that are monitored, meaning there is a line with timings for each method in the output.

How to interpret the output of the ``TimeMonitor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examining the output of the ``TimeMonitor`` is probably one of the easiest steps in profiling the behaviour of a program. The information given in the output of the `TimeMonitor` may server for

- getting an idea of the execution time of a method
- knowing how often a method is called during the runtime of the program
- investigating the parallel load balancing of a method (compare ``MinOverProcs`` with ``MaxOverProcs``)

and thereby helps identifying bottlenecks in the overall program.