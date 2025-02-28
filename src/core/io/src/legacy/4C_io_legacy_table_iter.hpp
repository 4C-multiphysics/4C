// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_LEGACY_TABLE_ITER_HPP
#define FOUR_C_IO_LEGACY_TABLE_ITER_HPP

#include "4C_config.hpp"

#include "4C_io_legacy_table.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*!
  \brief map node stack element
  \date 08/06
 */
/*----------------------------------------------------------------------*/
typedef struct StackElement
{
  struct StackElement* snext;
  MapNode* map_node;
} STACK_ELEMENT;


/*----------------------------------------------------------------------*/
/*!
  \brief stack of map nodes
  \date 08/06
 */
/*----------------------------------------------------------------------*/
typedef struct Stack
{
  int count;
  STACK_ELEMENT head;
} STACK;


/*----------------------------------------------------------------------*/
/*!
  \brief map iterator

  Visit all maps inside a map. This is a tree iterator. The map is
  implemented as a tree. Hence there is a stack inside this iterator.
  \date 08/06
 */
/*----------------------------------------------------------------------*/
typedef struct MapIterator
{
  MAP* map;
  STACK stack;
} MAP_ITERATOR;


void init_map_iterator(MAP_ITERATOR* iterator, MAP* map);

int next_map_node(MAP_ITERATOR* iterator);

MapNode* iterator_get_node(MAP_ITERATOR* iterator);

FOUR_C_NAMESPACE_CLOSE

#endif
