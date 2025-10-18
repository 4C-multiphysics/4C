// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_LEGACY_TABLE_HPP
#define FOUR_C_IO_LEGACY_TABLE_HPP

#include "4C_config.hpp"

#include "4C_io_legacy_types.hpp"

#include <mpi.h>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*!
  \brief Bring a map variable up to a clean state.

  That's needed before anything can be done with a map.

*/
/*----------------------------------------------------------------------*/
void init_map(MAP* map);


/*----------------------------------------------------------------------*/
/*!
  \brief Clean up.

*/
/*----------------------------------------------------------------------*/
void destroy_map(MAP* map);


/* Find the first symbol with the given key. Use this if you have to
 * travel all symbols with that key. */
SYMBOL* map_find_symbol(MAP* map, const char* key);


/* Find the last symbols value. The value has to be of the given
 * type. Returns false on failure. */
int map_find_string(MAP* map, const char* key, const char** string);
int map_find_int(MAP* map, const char* key, int* integer);
int map_find_real(MAP* map, const char* key, double* real);
int map_find_map(MAP* map, const char* key, MAP** dir);


/* Find the last symbols value. The value has to be of the given
 * type. Calls FOUR_C_THROW on failure. */
const char* map_read_string(MAP* map, const char* key);
int map_read_int(MAP* map, const char* key);
double map_read_real(MAP* map, const char* key);
MAP* map_read_map(MAP* map, const char* key);


/* Tell whether there is a symbol with given key and value. Only the
 * last symbol with that key is checked. */
int map_has_string(MAP* map, const char* key, const char* value);
int map_has_int(MAP* map, const char* key, const int value);
int map_has_real(MAP* map, const char* key, const double value);
int map_has_map(MAP* map, const char* key);


/* Insert a new symbol. */
void map_insert_string(MAP* map, char* string, char* key);
void map_insert_int(MAP* map, int integer, char* key);
void map_insert_real(MAP* map, double real, char* key);
void map_insert_map(MAP* map, MAP* dir, char* key);


/* Tell the number of symbols under this key. */
int map_symbol_count(MAP* map, const char* key);


/* Take a symbol chain out of the map. Leave the symbol alive. */
void map_disconnect_symbols(MAP* map, const char* key);


/* Prepend the symbol chain to one under the given key. */
void map_prepend_symbols(MAP* map, const char* key, SYMBOL* symbol, int count);


/* Tell whether this symbol has the given type. */
int symbol_is_map(const SYMBOL* symbol);


/* Extract the value of this symbol. Returns false on failure. */
int symbol_get_string(const SYMBOL* symbol, const char** string);
int symbol_get_int(const SYMBOL* symbol, int* integer);
int symbol_get_real(const SYMBOL* symbol, double* real);
int symbol_get_real_as_float(const SYMBOL* symbol, float* real);
int symbol_get_map(const SYMBOL* symbol, MAP** map);


/* Extract the value of this symbol. Call FOUR_C_THROW on failure. */
MAP* symbol_map(const SYMBOL* symbol);


/* Read the control file given by name. Put its contents into the map. */
void parse_control_file(MAP* map, const char* filename, MPI_Comm comm);

FOUR_C_NAMESPACE_CLOSE

#endif
