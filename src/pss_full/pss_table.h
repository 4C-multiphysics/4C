/*!
\file
\brief A very simple symbol table implementation.

<pre>
Maintainer: Ulrich Kuettler
            kuettler@lnm.mw.tum.de
            http://www.lnm.mw.tum.de/Members/kuettler
            089 - 289-15238
</pre>

ccarat writes control files that describe its binary output. Those
files are meant to be human readable and very flexible but still must
be read back. To accomplish this we need a small parser and a way to
represent these files internally. And here it is.

A control file consists of definitions. Simple definitions look like
this:

<pre>
attr_name = value
</pre>

where value might be a string, an integer or a double constant. (It's
easily possible to enhance the parser to understand mathematical
expressions. Speak up if you need this.) Additionally you can group
your definitions:

<pre>
group_name:
    attr_name_1 = value
    attr_name_2 = value
</pre>

This looks a little like python and indeed indentions do matter
here. But on the other hand you can have many groups using the same
name. That's convenient because restart information is written every
some steps into identical groups.

Files that follow this scheme can be read into a simple symbol
table. This table can be queried for those values quite easily.

*/

#ifndef PSS_TABLE_H
#define PSS_TABLE_H

#include "../headers/standardtypes.h"
#include "../headers/am.h"
#ifdef PARALLEL
#include <mpi.h>
#endif


/*----------------------------------------------------------------------*/
/*!
  \brief Types of the symbols we store in a map.

  We can have strings, integer and doubles as well as a subgroup.

  \author u.kue
  \date 08/04
*/
/*----------------------------------------------------------------------*/
typedef enum _SYMBOL_TYPES {
  sym_undefined,
  sym_string,
  sym_int,
  sym_real,
  sym_map
} SYMBOL_TYPES;


/*----------------------------------------------------------------------*/
/*!
  \brief Data struct of a map's symbol.

  Depending on its type a symbol contains different values. You don't
  need to deal with these explicitly, there are access
  functions. However, you need to access the ``next`` pointer.

  Inside a map all symbols with the same key are linked by their next
  pointers. Thus a map node gives the first symbol and from there you
  can follow the next pointer until you reach a NULL.

  \warning String and map values are assumed to be allocated
  dynamically and are owned by the symbol. That is when a symbol is
  destroyed its string or map value will be free'd, too.

  \author u.kue
  \date 08/04
*/
/*----------------------------------------------------------------------*/
typedef struct _SYMBOL {
  /* put the union first to have the double value properly aligned */
  union {
    CHAR* string;
    INT integer;
    DOUBLE real;
    struct _MAP* dir;
  } s;
  SYMBOL_TYPES type;
  struct _SYMBOL* next;
} SYMBOL;


/*----------------------------------------------------------------------*/
/*!
  \brief A map node that contains a key and a symbol.

  The backbone structure of a map is the map node. Map nodes
  constitute a binary tree (unbalanced by now), so there are left and
  right subnodes. The symbol is the start of a list of symbols that
  share one key.

  The node knows how many symbols there are.

  \warning The map node owns its key and symbols as well as any
  subnodes. It'll destroy them when it is destroyed itself.

  \author u.kue
  \date 08/04
*/
/*----------------------------------------------------------------------*/
typedef struct _MAP_NODE {
  CHAR* key;
  SYMBOL* symbol;
  INT count;
  struct _MAP_NODE* lhs;
  struct _MAP_NODE* rhs;
} MAP_NODE;


/*----------------------------------------------------------------------*/
/*!
  \brief The central map structure.

  You need to initialize an object of this structure in order to have
  a map. Internally it's nothing but a map node and a total count.

  \author u.kue
  \date 08/04
*/
/*----------------------------------------------------------------------*/
typedef struct _MAP {
  MAP_NODE root;
  INT count;
} MAP;


/*----------------------------------------------------------------------*/
/*!
  \brief Bring a map variable up to a clean state.

  That's needed before anything can be done with a map.

  \author u.kue
  \date 08/04
*/
/*----------------------------------------------------------------------*/
void init_map(MAP* map);


/*----------------------------------------------------------------------*/
/*!
  \brief Clean up.

  \author u.kue
  \date 08/04
*/
/*----------------------------------------------------------------------*/
void destroy_map(MAP* map);


/* Find the first symbol with the given key. Use this if you have to
 * travel all symbols with that key. */
SYMBOL* map_find_symbol(MAP* map, const CHAR* key);


/* Find the last symbols value. The value has to be of the given
 * type. Returns false on failure. */
INT map_find_string(MAP* map, const CHAR* key, CHAR** string);
INT map_find_int(MAP* map, const CHAR* key, INT* integer);
INT map_find_real(MAP* map, const CHAR* key, DOUBLE* real);
INT map_find_map(MAP* map, const CHAR* key, MAP** dir);


/* Find the last symbols value. The value has to be of the given
 * type. Calls dserror on failture. */
CHAR* map_read_string(MAP* map, const CHAR* key);
INT map_read_int(MAP* map, const CHAR* key);
DOUBLE map_read_real(MAP* map, const CHAR* key);
MAP* map_read_map(MAP* map, const CHAR* key);


/* Tell whether there is a symbol with given key and value. Only the
 * last symbol with that key is checked. */
INT map_has_string(MAP* map, const CHAR* key, const CHAR* value);
INT map_has_int(MAP* map, const CHAR* key, const INT value);
INT map_has_real(MAP* map, const CHAR* key, const DOUBLE value);
INT map_has_map(MAP* map, const CHAR* key);


/* Insert a new symbol. */
void map_insert_string(MAP* map, CHAR* string, CHAR* key);
void map_insert_int(MAP* map, INT integer, CHAR* key);
void map_insert_real(MAP* map, DOUBLE real, CHAR* key);
void map_insert_map(MAP* map, MAP* dir, CHAR* key);


/* Insert a new symbol. Make a copy of all strings. */
void map_insert_string_cpy(MAP* map, CHAR* string, CHAR* key);
void map_insert_int_cpy(MAP* map, INT integer, CHAR* key);
void map_insert_real_cpy(MAP* map, DOUBLE real, CHAR* key);
void map_insert_map_cpy(MAP* map, MAP* dir, CHAR* key);


/* Tell the number of symbols under this key. */
INT map_symbol_count(MAP* map, const CHAR* key);


/* Take a symbol chain out of the map. Leave the symbol alive. */
void map_disconnect_symbols(MAP* map, const CHAR* key);


/* Prepend the symbol chain to one under the given key. */
void map_prepend_symbols(MAP* map, const CHAR* key, SYMBOL* symbol, INT count);


/* Tell whether this symbol has the given type. */
INT symbol_is_string(const SYMBOL* symbol);
INT symbol_is_int(const SYMBOL* symbol);
INT symbol_is_real(const SYMBOL* symbol);
INT symbol_is_map(const SYMBOL* symbol);


/* Extract the value of this symbol. Returns false on failture. */
INT symbol_get_string(const SYMBOL* symbol, CHAR** string);
INT symbol_get_int(const SYMBOL* symbol, INT* integer);
INT symbol_get_real(const SYMBOL* symbol, DOUBLE* real);
INT symbol_get_real_as_float(const SYMBOL* symbol, float* real);
INT symbol_get_map(const SYMBOL* symbol, MAP** map);


/* Extract the value of this symbol. Call dserror on failure. */
CHAR* symbol_string(const SYMBOL* symbol);
INT symbol_int(const SYMBOL* symbol);
DOUBLE symbol_real(const SYMBOL* symbol);
MAP* symbol_map(const SYMBOL* symbol);


/* Write the map to file f in a readable way. */
void map_print(FILE* f, MAP* map, INT indent);
void map_node_print(FILE* f, MAP_NODE* node, INT indent);
void symbol_print(FILE* f, CHAR* key, SYMBOL* symbol, INT indent);


/* Read the control file given by name. Put its contents into the map. */
void parse_control_file(MAP* map, const CHAR* filename, MPI_Comm comm);


/* Read the control file given by name. Put its contents into the map.
 * (serial only!)*/
void parse_control_file_serial(MAP* map, const CHAR* filename);


#endif
