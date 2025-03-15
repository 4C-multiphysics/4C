.. _doxygen:

Documenting the code with Doxygen
---------------------------------


|FOURC| uses `Doxygen <http://www.doxygen.nl>`__ to generate documentation from annotated source code.
You can find a current documentation `here <https://4c-multiphysics.github.io/4C/doxygen/index.html>`_.

What is Doxygen and why does |FOURC| rely on it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Doxygen is a documentation generator, i.e. a tool for writing software reference documentation.
The documentation is written within the code, so documentation lives close by to the code that it is documenting.
This makes it relatively easy to keep up to date.
Doxygen can cross reference documentation and code, so that the reader of a document can easily refer to the actual code.

|FOURC|'s Policy
~~~~~~~~~~~~~~~~

|FOURC| requires a set of minimum Doxygen items for each code entity.

At a bare minimum, |FOURC| requires that

- any source code file has a file header consisting of the tags
    - ``@file``
    - ``@brief``
    - ``@level``
- any newly added source code entity has a ``@brief`` description, and functions additionally require at least the ``@param`` and ``@return`` information.

It is the responsibility of the developer adding the new code to include the relevant documentation.
It is the responsibility of the merge request reviewer to not allow a request to be merged
until this minimum documentation is in place and is understandable to someone other than the author.

It is also highly recommended and encouraged that any time you come across undocumented entities,
you either add the documentation yourself if you are knowledgeable enough to do so,
or contact the file author and ask them to add the documentation.
Our goal is to have a fully documented code base, and the more proactive every developer is about that, the easier it will be to achieve.

> **Note:** Your IDE might offer supportive capabilities to ease Doxygen-style documentation.
Checkout the :ref:`IDE setup instructions <Set-up-your-ide>` for details.

Building the Doxygen Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local build using CMake
"""""""""""""""""""""""""""

|FOURC| has defined a custom build target ``documentation`` (see also our :ref:`list of custom build targets <build4Cwithcustomtargets>`.
In order to create the Doxygen HTML webpage locally, just issue the command

::

    cd <buildDirectory>
    make documentation


This will build the Doxygen documentation in the directory ``<buildDirectory>/doc/html/``.
It can be viewed by accessing ``<buildDirectory>/doc/html/index.html`` in any browser.

To trigger a *re-build* of the documentation, just call ``make documentation`` again.
The documentation target is always considered out of date and therefore always built.
It is unfortunately not possible to only rebuild the documentation for changed code.
However, Doxygen builds seem to benefit from using :ref:`ccache <ccache>`.

Pipeline build
""""""""""""""""

We have a Doxygen pipeline in place that creates the Doxygen documentation every night.
We check for errors in Latex processing. Other than that, no checks are performed right now.

General Remarks on documenting the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Documenting source code already starts when writing the code (even before writing its comments and documentation).
Here are some general remarks and guidelines taken from Robert C. Martin's book
"Clean Code" to facilitate writing easy-to-understand and well-documented code [Martin08]_.

Use Intention-Revealing Names
"""""""""""""""""""""""""""""""

Writing easy-to-understand and well-documented code starts at selecting descriptive and intention-revealing names for software entities.
Ideally, the name of a variable, function, or class should answer all the big questions.
It should tell you why it exists, what it does, and how it is used.

Use Searchable Names
"""""""""""""""""""""""""

Single-letter names and numeric constants have a particular problem in that they are not easy to locate across a body of text.
Global search will deliver more accurate results, if the variable names are unique and searchable.

Use Pronounceable Names
"""""""""""""""""""""""""

Using pronouncable names makes it easier to discuss code with fellow users and developers.
Don't be afraid of using non-abbreviated names!
Typing on a keyboard is quick, and we use a code formatter to take care of formatting and line breaks.

Documenting a File
~~~~~~~~~~~~~~~~~~~

Each source code file has to start with a file header that exhibits these tags:

- ``@file``: keyword to tell Doxygen, that this comment block is related to the file description.
  The file name **must not** to be given.
- ``@brief``: summarize the file's content in a brief description

Here's an example:

.. code-block:: cpp

    /*---------------------------------------------------------------------*/
    /*! @file
    @brief This header provides the interface for all FE simulations

    @level 3
    */
    /*---------------------------------------------------------------------*/

**Important:** Do not add personal contact information (like email, phone number, webpage)
since they become outdated very quickly and make it more difficult to update maintainer information in the future.

If GitHooks have been configured properly, the ``pre-commit`` hook will enforce this policy and provide hints on how to fix policy violations.
Furthermore, this policy is checked during the code checks in every pipeline run.

Documenting a Class or Struct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

> **Note:**  The documentation of a ``struct`` is the same as that required for a ``class`` as they are essentially the same thing,
differing only in the default visibility of their members.

**Important note for derived classes:**
For derived classes doxygen will automatically copy the documentation of members from the base class if you don't add any documentation.
This should be the default for a well-designed class hierarchy.
If you feel the need to add additional information to a member of a derived class
but also want to keep the base class documentation you can use `\copydoc <http://www.doxygen.nl/manual/commands.html#cmdcopydoc>`_.
Do not just copy the whole documentation block from the base class if there is no difference.

The Class Itself
"""""""""""""""""""""""

Classes should be preceded by a comment block along these lines:

.. code-block:: cpp

    /*!
     *  @brief A brief description of the class goes here.  The brief description
     *         is terminated by a blank line.
     *
     *  The detailed description of the class follows the blank line.  This should
     *  give an unfamiliar developer enough information to understand the purpose
     *  of the class and how to interact with instantiations of it via the methods
     *  that will be defined below.
     *
     *  If a detailed description continues over multiple paragraphs, separate
     *  paragraphs with a blank line.
     *
     * @tparam stuff That‘s the documentation of a template parameter.
     */
    template<stuff>
    class ClassName
      :
      public BaseClass<stuff>
    {
      // Insert class definition here.
    } // end of class ClassName

Methods
""""""""""

Within the class definition, methods should be preceded by a comment block along these lines:

**Note:** It's good practice to use indicators like ``[in]``, ``[out]``, and ``[in/out]`` to indicate whether a function arguments is input, output, or both.

.. code-block:: cpp

    /*!
     *  @brief A brief description of the method goes here.  The brief
     *         description is terminated by a blank line.
     *
     *  The detailed description of the method follows the blank line.  This
     *  should give an unfamiliar developer an understanding of what the method
     *  is doing and why you would use it.
     *
     *  \note If anything is noteworthy, feel free to include that here.  Perhaps
     *        you might mention how this compares to another method in the class,
     *        if it is to be preferred over another method, if it has been
     *        deprecated and should no longer be used, etc.  Similar commands you
     *        can use in this manner are `\remark` and `\warning`.
     *
     *  @pre List any prerequisites for this method
     *
     *  @post Provide information on an post-conditions, e.g. how the state of this
     *        class is affected when running this method.
     *
     *  @param[in,out] arg1 This is a description of `arg1`.  This variable is used
     *                      as both input and output; that is, its value changes
     *                      and persists after the function call.
     *  @param[out]    arg2 This is a description of `arg2`.  This variable is used
     *                      as output; its value when the function begins is
     *                      irrelevant.
     *  @param[in]     arg3 This is a description of `arg3`.  Be sure to mention
     *                      that it's inclusion is optional, and if omitted, what
     *                      the default value is.
     *
     *  \throws ExceptionType This is a description of why `ExceptionType` would be
     *                        thrown in the midst of the function execution.
     *
     *  @returns This is a description of what the function returns on completion,
     *           if anything.
     */
    returnType
    functionName(someType arg1, anotherType arg2, yetAnotherType arg3 = someDefaultValue);


If any of the lines in the comment block above (``\note``, ``\warning``, ``\remark`, ``@pre``, ``@post``, ``@param``, ``\throws``, ``@returns``)
are not applicable to the function you're documenting, simply omit them.

Member Data
""""""""""""""

Member data should be preceded by a comment block along these lines:

.. code-block:: cpp

    /*!
     *  @brief A brief description of the data goes here.  The brief description
     *         is terminated by a blank line.
     *
     *  The detailed description of the data follows the blank line.  If the brief
     *  description gives enough information to understand the variable, its use
     *  and purpose, then a detailed description may not be necessary.
     */
    DataType member_data_;


Enumerations
""""""""""""""

When documenting an ``enum``, use something along the lines of the following:

**Note:** Place detailed and elongated descriptions preferably in front of the documented item to avoid strange formatting by our code formatter ``clang_format``.

.. code-block:: cpp

    /*!
     *  @brief A brief description of the enum goes here.  The brief
     *         description is terminated by a blank line.
     *
     *  The detailed description of the enum follows the blank line.  This should
     *  give an unfamiliar developer an understanding of what the enum represents
     *  and how it is used.
     */
    enum EnumName
    {
      /*!< This is a description of the SOMETHING value of the enum.
           It can be as detailed as you like. */
      SOMETHING,
      SOMETHING_ELSE //!< This is a brief description of SOMETHING_ELSE.
    }; // end of enum EnumName

Typedefs and Usings
""""""""""""""""""""""

When documenting a ``typedef``, the syntax is essentially the same as that used for member data:

.. code-block:: cpp

    /*!
     *  @brief A brief description of the typedef goes here.  The brief
     *         description is terminated by a blank line.
     *
     *  The detailed description of the typedef follows the blank line.  If the
     *  brief description gives enough information to understand the typedef, its
     *  use and purpose, then a detailed description may not be necessary.
     */
    typedef OriginalType NewName;

However, as of C++11, we should really be phasing out ``typedef``s in favor of ``using`` statements.
Unfortunately, Doxygen has not been updated to support those out of the box, so it needs a little help.
We can use the ``\var`` command to tell Doxygen to document a ``using`` statement as if it were a variable:

.. code-block:: cpp

    /*!
     *  \var NewName
     *
     *  @brief A brief description of the using statement goes here.  The brief
     *         description is terminated by a blank line.
     *
     *  The detailed description of the using statement follows the blank line.
     *  If the brief description gives enough information to understand the
     *  using statement, then a detailed description may not be necessary.
     */
    using NewName = OriginalType;

Grouping Entities
""""""""""""""""""""""

You may find it useful to group certain functions, variables, etc., together into named sections, particularly if your class contains a great many members.
This can aid in understanding the design and intended use of the class.
To group entities together in the documentation, use the following:

.. code-block:: cpp

    //! @name Group 1 Name
    //! @{

    // Insert functions, variables, typedefs, etc., here, along with their
    // documentation.

    //! @}


Be sure not to forget the ``//! @{`` and ``//! @}``, which open and close the group.

Or a little longer:

.. code-block:: cpp

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** @name Group 1 Name                                                       */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    // Insert functions, variables, typedefs, etc., here, along with their
    // documentation.

    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of Group 1 Name                                                      */
    /*                                                                           */
    //-----------------------------------------------------------------------------


**Note:**  These grouping characters must appear on their own lines.
If they're on a line with other non-comment characters, Doxygen won't process them correctly.

*Documenting the Group Itself*

If you like, you can include documentation that pertains to all members of a group.  To do so, use something along the lines of:

.. code-block:: cpp

    /*! @name Group Name
     *  @{
     *
     *  This is a detailed description that pertains to all the members of this
     *  group.
     *
     *  @param[in] x - This is an input that pertains to every member of the group.
     *
     *  @returns This is something every member of the group returns, at least in
     *           some general sense.
     */

    // Insert the members of the group, along with any corresponding documentation.
    // You can either document just the first member (see below) or document each
    // member separately.

    //! @}                                                                       */


This documentation will appear between the group name and its members in the brief description section of the automatically generated HTML page,
but it will not appear with the detailed documentation of any of the members.

*Only Documenting the First Member*

The documentation on the first member of a group will get spread across all members of the group in the generated HTML.
This can be useful, for instance, if you have a handful of functions that all do more or less the same thing &mdash;
you can document them once in the code,
but someone using the Doxygen as a reference manual will be able to see that documentation regardless of which function in the group they happen to be looking at.

For instance, if you have

.. code-block:: cpp

    //! @name Random Generators
    //! @{

    /**
     *  @brief Get a random variable.
     *
     *  Generate a random ``int``/``double``/``char``/``string``.
     */
    int rand();
    double rand();
    char rand();
    std::string rand();

    //! @}


the generated HTML will be such that it'll look like you copied and pasted the comment before ``int rand()`` in front of the other three routines.

**Note:**  If you do not wish all members of a group to share the same documentation, *each and every member must be documented separately*.

Documenting a Namespace
~~~~~~~~~~~~~~~~~~~~~~~~

Namespaces appear throughout our code, but if documentation were to show up before each occurrence of a namespace,
those various bits of documentation will get bundled together by Doxygen in a manner depending on in what order it processes the files.
To avoid the potential confusion there, namespaces should be documented in ``src/headers/namespaces.H``.
Each namespace should be preceded by a comment block along these lines:

.. code-block:: cpp

    /*!
     *  @brief A brief description of the namespace goes here.  The brief
     *         description is terminated by a blank line.
     *
     *  The detailed description of the namespace follows the blank line.  This
     *  should give an unfamiliar developer enough information to understand the
     *  purpose of the namespace, what it contains, why it was organized in such a
     *  way, etc.
     *
     *  If the detailed description continues over multiple paragraphs, separate
     *  paragraphs with a blank line.
     */
    namespace something
    {
    } // end of namespace something


``namespaces.H`` must only contain namespace (and nested namespaces) documentation, in particular no actual source code.

General Doxygen Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~


The `Doxygen manual <http://www.doxygen.nl/manual/index.html>`_ will tell you everything you need to know about using Doxygen to document your code.
Here are some highlights:

Brief Descriptions
"""""""""""""""""""""""

If at any point a brief description is all that is needed to fully document some member, you can use one of the following Doxygen shortcut syntaxes.

.. code-block:: cpp

    /// This is the brief description.
    int someFunction()

    //! This is also a brief description.
    bool someOtherFunction()


Todos
"""""""""""

If, for whatever reason, you are unable to complete the documentation of a class, function, variable, etc.,
- perhaps you need to consult with a coworker to ensure you have an accurate description of what you're documenting -
be sure to use the ``@todo`` command to flag this as documentation that still needs work.
For instance,

.. code-block:: cpp

    /*!
     *  @brief This function does something really cool.
     *
     *  But I'm not entirely sure what it is yet.
     *
     *  @todo Finish documenting this function.
     */
    returnType
    awesomeFunction();


``@todo`` items populate the "Todo List" page under the "Related Pages" tab of our Doxygen site,
so it's easy to see what still needs work.

Undocumented Entities
"""""""""""""""""""""""""

When you run Doxygen, for instance,

::

    cd <4C-execdir>
    ninja doxygen | tee doxygen.log


it will warn you about any undocumented entities.
You can search through the output for warnings associated with files you've touched to ensure you haven't missed documenting anything.
For instance,

::

    grep warning doxygen.log | grep FileIModified.hpp

Comment Blocks
""""""""""""""""

Note that in the midst of Doxygen-style comment blocks

.. code-block:: cpp

    /*!
     *  Text
     *  goes
     *  here.
     */


any leading ``\*`` marks are stripped out by Doxygen before any other processing is done.

Automatic Link Generation
"""""""""""""""""""""""""""""""

Doxygen will automatically `generate hyperlinks <http://www.doxygen.nl/manual/autolink.html>`_ in a handful of scenarios.
When mentioning a function, be sure to include ``()`` at the end to tell Doxygen to generate a link to that function's documentation, as in

.. code-block:: cpp

    /*!
     *  Check out `functionName()`.  It's pretty great.
     */


Markdown Syntax
""""""""""""""""""

Doxygen does have support for rendering `Markdown syntax <http://www.doxygen.nl/manual/markdown.html>`_.

When it comes to documenting classes, functions, and data as specified above,
be sure to use `` `backticks` `` around class names, function names, and snippets of code that appear inline to have them rendered in a monospaced font.

If creating a bulleted list, use non-asterisk bullet markers, as leading asterisks will be stripped away.
That is:

.. code-block:: cpp

    /*!
     *  Here's a list:
     *  - Item 1
     *    - Subitem 1
     *    - Subitem 2
     *  - Item 2
     */


For enumerations, use the ``-#`` marker:

.. code-block:: cpp

    /*!
     *  Here's a list:
     *  -# Item 1
     *    -# Subitem 1
     *    -# Subitem 2
     *  -# Item 2
     */


Including Math
""""""""""""""""""""

Doxygen allows you to include mathematical formulas and the like by surrounding LaTeX with certain delimiters:

- ``\f$`` for inline math, as in ``\f$ E = m c^2 \f$``.
- ``\f[`` and ``\f]`` for unnumbered displayed equations, analogous to ``\begin{equation*}`` and ``\end{equation*}``; and
- ``\f{environment}{`` and ``\f}`` for other LaTeX math environments, such as ``eqnarray``.

See `this page of the Doxygen documentation <http://www.doxygen.nl/manual/formulas.html>`_ for examples.

Including Code
"""""""""""""""""""

There are a handful of different ways to include code blocks in Doxygen, but the one that seems to work in the widest variety of cases is the following:

.. code-block:: cpp

    /*!
     *  Here's a bit of code
     *  \code{.cpp}
        int
        main(
          int   argc,
          char* argv[])
        {
          std::cout << "Hello World!" << std::endl;
        } // end of main()
        \endcode
     *  that prints "Hello World!".
     */

The argument to the ``\code{}`` command is a file extension that'll tell Doxygen what kind of syntax highlighting it should use.
Unfortunately, between the ``\code{}`` and ``\endcode`` keywords, there is no stripping of leading asterisks as would normally take place.
It will, however, strip out whatever leading indentation exists.

