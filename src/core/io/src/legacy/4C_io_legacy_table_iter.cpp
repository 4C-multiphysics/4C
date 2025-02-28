// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_legacy_table_iter.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*!
  \brief map iterator constructor

 */
/*----------------------------------------------------------------------*/
void init_map_iterator(MAP_ITERATOR* iterator, MAP* map)
{
  iterator->stack.count = 0;
  iterator->map = map;
  iterator->stack.head.map_node = nullptr;
  iterator->stack.head.snext = nullptr;
}

/*----------------------------------------------------------------------*/
/*!
  \brief map iterator push

 */
/*----------------------------------------------------------------------*/
static void push_map_node(MAP_ITERATOR* iterator, MapNode* map_node)
{
  STACK_ELEMENT* new_element;

  new_element = new STACK_ELEMENT;
  new_element->map_node = map_node;
  new_element->snext = iterator->stack.head.snext;
  iterator->stack.head.snext = new_element;
  iterator->stack.count++;
}

/*----------------------------------------------------------------------*/
/*!
  \brief map iterator pop

 */
/*----------------------------------------------------------------------*/
static void pop_map_node(MAP_ITERATOR* iterator)
{
  STACK_ELEMENT* tmp_free;

  if (iterator->stack.count == 0)
  {
    FOUR_C_THROW("map iterator stack empty");
  }
  else
  {
    tmp_free = iterator->stack.head.snext;
    iterator->stack.head.snext = iterator->stack.head.snext->snext;
    iterator->stack.count--;
    delete tmp_free;
  }
}

/*----------------------------------------------------------------------*/
/*!
  \brief map iterator

  \param iterator (i/o) the map iterator to be advanced
  \return true if a new node was found

 */
/*----------------------------------------------------------------------*/
int next_map_node(MAP_ITERATOR* iterator)
{
  int result = 0;

  /* if the map is empty there is nothing to iterate */
  if (iterator->map != nullptr)
  {
    /*first call of this iterator*/
    if (iterator->stack.head.map_node == nullptr)
    {
      /* we actually dont need the map->root information, we just use it
       * to show that the iterator is finally initialized*/
      iterator->stack.head.map_node = &iterator->map->root;

      if (iterator->map->root.rhs != nullptr) push_map_node(iterator, iterator->map->root.rhs);
      if (iterator->map->root.lhs != nullptr) push_map_node(iterator, iterator->map->root.lhs);

      /*if iterator is still empty return 0*/
      result = iterator->stack.head.snext != nullptr;
    }
    else
    {
      if (iterator->stack.head.snext != nullptr)
      {
        MapNode* tmp;
        MapNode* lhs;
        MapNode* rhs;

        /* we remove the first member of the stack and add his rhs and lhs */
        tmp = iterator->stack.head.snext->map_node;
        lhs = tmp->lhs;
        rhs = tmp->rhs;
        tmp = nullptr;

        /* caution! tmp is freed at this point! */
        pop_map_node(iterator);

        if (rhs != nullptr) push_map_node(iterator, rhs);
        if (lhs != nullptr) push_map_node(iterator, lhs);

        /*if iterator is empty now return 0*/
        result = iterator->stack.head.snext != nullptr;
      }
    }
  }
  return result;
}


/*----------------------------------------------------------------------*/
/*!
  \brief map iterator current node

 */
/*----------------------------------------------------------------------*/
MapNode* iterator_get_node(MAP_ITERATOR* iterator) { return iterator->stack.head.snext->map_node; }

FOUR_C_NAMESPACE_CLOSE
