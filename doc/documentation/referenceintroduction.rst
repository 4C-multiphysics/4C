General
========

Sections
--------

The total input of |FOURC| is read from formatted input files that
contain all data to define a problem. The file is divided in single
blocks by horizontal lines which end with a key word specifying the
following content. There must not be spaces in front of a keyword. An
example is the ``PROBLEM TYPE`` introductory line

::

   -------------------------------------PROBLEM TYPE

Plenty of keyword lines and blocks refer to special applications of the
program (such as ``OPTIMIZATION`` , ``ALE DYNAMIC`` etc.). These
often can be omitted when not required, which shortens the file and
eases its use. Other blocks however are of general character and must
not be left out in order to obtain a valid input file.

The order of the blocks can be chosen arbitrary while a convention
certainly increases comfort. The single blocks are explained in more
detail subsequently. Emphasis is here put on those blocks which are of
general meaning and are used within all kinds of different problem
types. Specific problem dependent blocks such as ``ALE SOLVER``,
``FLUID DYNAMIC`` or ``OPTIMISATION`` can be added to this manually
as required.

**Notes**

- The parameter names and the (string) values are case sensitive. In many cases, different version
  of a parameter value can be entered. All of the possibilities for string values are given in the
  following sections.
- All parameters have a default value, which is given in the respective section.
  If you don't want to change it, you don't have to give it here (but it does not hurt as well).
- The parameters in each section are single words,
  and they are separated from the value by a whitespace.
- However, there is one block, namely the `STRUCT NOX` section (including subsections),
  where the parameters are made up of more than one word.
  There, one has to separate the parameter name and its value by an equal sign (=).
  This section is the only one, which uses lower case letters in the parameter names.
  (It will be changed in the future, so that the equal signs will be obsolete)

Comments
--------

Comments start with ``//``. These can appear in the first rows or
somewhere else. Everything on their right side is a comment and
neglected.


Extended Backus–Naur formalism
------------------------------

An extended Backus–Naur Formalism (EBNF, cf.
[Reiser94]_), is used to describe the input
lines.

-  | Several listed construct are regarded as concatenated:
   | ``C = A B`` means ``C`` consists of ``A``
     followed by ``B``.

-  | Alternatives are separated by ``|``:
   | ``C = A | B`` means ``C`` :math:`=`
     ``A`` or ``C`` :math:`=` ``B`` , but not ``C`` :math:`=` ``A B``
     or ``C =``  :math:`\emptyset`.

-  | Brackets ``[`` and ``]`` denote optionality of the enclosed
     construct:
   | ``B = [A]`` results in ``B = A`` or ``B =`` :math:`\emptyset`.

-  | A variable that has to be replaced by something else is often by triangular brackets:
   | ``B = <x>`` means that x is not used, but has to be replaced by something appropriate.

-  | Braces ``{`` and ``}`` denote a repetition of a
     construct, which includes zero repetitions:
   | ``B = {A}`` is equivalent to
     ``B =`` :math:`\emptyset,` ``A``, ``AA``, ``AAA`` :math:`,\ldots`

-  Parentheses ``(`` and ``)`` group expressions.

-  An ellipsis :math:`\ldots` represents reasonable continuation.


