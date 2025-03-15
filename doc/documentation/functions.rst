.. _functiondefinitions:

Function definitions
====================

Function are particularly useful for the application of boundary conditions, but they can also
be used for other definitions. The definition of a function is extremely versatile and may be
accomplished in various manners. Up to 15 different function definitions are allowed, and the
respective section always starts with

::

   ----------------FUNCT1
   to
   ----------------FUNCT15

These functions allow to define spatial and temporal conditions in terms of mathematical
functions, e.g., in sine form etc.

The respective input line for defining the function is of the following form:

::

   [COMPONENT <num>] SYMBOLIC-FUNCTION_OF_SPACE_TIME|SYMBOLIC_FUNCTION_OF_TIME|VARFUNCTION <functiondefinition>

As the name says, the definition can be a function :math:`f`

#. of space and time, that is, :math:`f(x,y,z,t)`
#. only of time, :math:`f(t)`
#. of a solution dependent variable, :math:`f(s)`.

|FOURC| has its own function parser, which recognises variables, algebraic terms, brackets and a number of functions, namely the following: :math:`\mathrm{acos}, \mathrm{asin}, \mathrm{atan}`, :math:`\cos, \sin, \tan`, :math:`\cosh, \sinh, \tanh`, :math:`\exp, \log, \log10`, :math:`\mathrm{sqrt}`, :math:`\mathrm{heaviside}`, :math:`\mathrm{fabs}, \mathrm{atan2}`. Additionally, the constant :math:`pi` is known.

If the function cannot easily be given as a symbolic expression, the function may depend on additional user defined variables, which are given in the following way:

::

   VARIABLE <num> NAME <varname> TYPE expression DESCRIPTION
   VARIABLE <num> NAME <varname> TYPE linearinterpolation NUMPOINTS <num> TIMES num*{time} VALUES num*{value}
   VARIABLE <num> NAME <varname> TYPE multifunction NUMPOINTS <num> TIMES num*{time} DESCRIPTION (num-1)*{function}
   VARIABLE <num> NAME <varname> TYPE fourierinterpolation NUMPOINTS <num> [BYNUM] TIMERANGE {starttime} {endtime} VALUES num*{value} [PERIODIC] T1 <time1> T2 <time2>

where

- **expression** is simply a symbolic expression similar to the function definition itself.
  That is,

  ::

     SYMBOLIC-FUNCTION_OF_SPACE_TIME 10*myvar
     VARIABLE 0 NAME myvar TYPE expression DESCRIPTION 5*t

  is completely equivalent to

  ::

     SYMBOLIC-FUNCTION_OF_SPACE_TIME 50*t

- **linearinterpolation** defines a table with a given number of sampling points leading to
  multilinear expression. For example, one can define a trapezoidal amplitude::

     SYMBOLIC-FUNCTION_OF_SPACE_TIME myvar
     VARIABLE 0 NAME myvar TYPE linearinterpolation NUMPOINTS 5 TIMES 0 1 2 3 10 VALUES 0.0 1.0 1.0 0.0 0.0

- **multifunction** is similar to expression, but a number of symbolic functions can be entered,
  which are  only valid in a specific time range. therefore, one has to define one point in time more than functions. As an example, see::

     SYMBOLIC-FUNCTION_OF_SPACE_TIME myvar
     VARIABLE 0 NAME myvar TYPE multifunction NUMPOINTS 3 TIMES 0.0 1.0 10.0 DESCRIPTION 5*t  5*exp(-(t-1.0))

- **fourierinterpolation** defines a Fourier series with a number of sampling point




**Fluid**

For fluids some other keywords are available beside the ones given above:

::

   BELTRAMI c1 <value>
   KIMMOIN-RHS MAT <num> ISSTAT [0|1] ISSTOKES [0|1]
   KIMMOIN-UP MAT <num> ISSTAT [0|1]
   KIMMOIN-STRESS MAT 1 ISSTAT [0|1] AMPLITUDE <value>
   CHANNELWEAKLYCOMPRESSIBLE
   WEAKLYCOMPRESSIBLE_ETIENNE_FSI_FLUID MAT_FLUID <num> MAT_STRUCT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_FSI_FLUID_FORCE MAT_FLUID <num> MAT_STRUCT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_FSI_FLUID_VISCOSITY MAT_FLUID <num> MAT_STRUCT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE MAT_STRUCT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_FSI_STRUCTURE_FORCE MAT_STRUCT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_CFD MAT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_CFD_FORCE MAT <num>
   WEAKLYCOMPRESSIBLE_ETIENNE_CFD_VISCOSITY MAT <num>
   WEAKLYCOMPRESSIBLE_MANUFACTUREDFLOW MAT <num>
   WEAKLYCOMPRESSIBLE_MANUFACTUREDFLOW_FORCE MAT <num>
   WEAKLYCOMPRESSIBLE_POISEUILLE MAT <num>
   WEAKLYCOMPRESSIBLE_POISEUILLE_FORCE MAT <num>
   ZALESAKSDISK

   FORWARDFACINGSTEP
   MOVINGLEVELSETCYLINDER
   MOVINGLEVELSETTORUS
   MOVINGLEVELSETTORUSVELOCITY
   MOVINGLEVELSETTORUSSLIPLENGTH
   URQUIZABOXFLOW
   URQUIZABOXFLOW_TRACTION
   URQUIZABOXFLOW_FORCE
   TAYLORCOUETTEFLOW
   COLLAPSINGWATERCOLUMN
   CORRECTIONTERMCHANNELWEAKLYCOMPRESSIBLE

**Porous materials**

Here, we can also consider the following keyword

::

   POROMULTIPHASESCATRA_FUNCTION
