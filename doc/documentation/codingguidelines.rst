.. _coding-guidelines:

Coding Guidelines
==================

General Guidelines regarding Coding in C++
--------------------------------------------

**Preamble:** |FOURC| has a long history. Large parts were written pre-C++11 and are not modern C++.
In addition, code review was not widely practiced back in these days.
Thus, it is recommended to critically examine the code and design while working on the code base and refer to
general guidelines on C++ coding.
We like to follow the `C++ Core Guidelines <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines>`_.

The current C++ standard in |FOURC| is C++20 and use of C++20 features is encouraged.

|FOURC|-specific coding guidelines
------------------------------------

In addition, we have a few guidelines that are specific to |FOURC| and are often a product of its long history.
Some of these guidelines are partly covered by the C++ Core Guidelines but are repeated here to emphasize their
importance for the future development of |FOURC|. In these cases, we want to make clear that we are aware that the
code base is not perfect, that we agree with the guidelines, and that we believe the guidelines should be followed
in the future.

**New code should always follow the following guidelines. Touching old code is always a great opportunity to improve it with respect to these guidelines.**

Avoid define flags
^^^^^^^^^^^^^^^^^^

Do not introduce new define flags. They complicate testing and lead to untested code.

- If you have additional debug code, put it behind ``FOUR_C_ENABLE_ASSERTIONS``.
- If you want to selectively print information during execution of an algorithm, make this a configurable parameter of
  the class or function.

**Exception:** A few flags are defined at configuration time to tell which features are enabled or disabled. These
flags are all managed via CMake.

Avoid header-in-header inclusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Avoid and actively resolve header-in-header inclusion to speed up compilation time. Use forward declarations instead.

**Attention:** Do not forward declare classes from external libraries such as Trilinos all over the code base. If you
want to avoid including an (expensive) header file via a forward declaration, put that forward declaration in a header
file and include this header. This way, the forward declaration is only in one place.

Use of smart pointers
^^^^^^^^^^^^^^^^^^^^^

If necessary, use smart pointers for their memory management capabilities, e.g., ``std::shared_ptr``, ``std::unique_ptr``
(you may find more information on passing smart pointers
`here <https://www.modernescpp.com/index.php/c-core-guidelines-passing-smart-pointer/>`_

By default, pass parameters by const reference and only expose smart pointers if memory management is necessary.

See also `<https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rr-owner>`_.

Use of ``Teuchos::ParameterList``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prefer parameter container classes over ``Teuchos::ParameterList`` for passing parameters to functions. Often,
a simple struct with all necessary data as public, non-const members is sufficient.

Reason: ``Teuchos::ParameterList`` is a flexible container for input(!) parameters, but it is also very slow and heavy.
In addition, the string-based lookup is not compile-time checked.

Const-correctness
^^^^^^^^^^^^^^^^^

Make new code const-correct and fix old code in that regard when working on it. Const-correctness is mainly relevant
for function parameters and member functions.

**Note:** The member fields of a struct or class should often not be ``const``, as this would prevent the struct or
class from being moved or copied. Instead, use ``const`` member functions to access the fields in a ``const`` context.

Enums
^^^^^

- By default, use scoped ``enum class`` instead of unscoped ``enum``.
- Use enums instead of strings for parameters that take one from a fixed set of values.
- You do not have to write conversion functions between enum constants and strings.
  Use ``magic_enum::enum_name`` to get the name of an enum constant and ``magic_enum::enum_cast`` to convert a string to
  an enum constant.

See also `Enum.3 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#enum3-prefer-class-enums-over-plain-enums>`_.


|FOURC|-specific Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Namespaces** use CamelCase: ``LinAlg::``
- **Class names** / **function** names / **enum types** use camel case, starting with a capital letter:
  ``ClassToDoSomething / FunctionToPerformAnOperation()``
- **Variables** and **enum values** use

    - snake_case, i.e. all lower-case letters separated by underscores, e.g. ``variable_with_descriptive_name``
    - camelCase starting with a lower-case letter, e.g. ``variableWithDescriptiveName``

- **Class members** end with an underscore: ``variable_``
- **Define flags** are in all caps: ``DEBUG``

Variable names must not be just a single letter, because they are impossible to find in a global search operation.
(Exception: loop indices such as i, j, but remember that even loop indices could/should have descriptive names.)
