#ifndef FOUR_C_FLUID_ELE_NULLSPACE_HPP
#define FOUR_C_FLUID_ELE_NULLSPACE_HPP

#include "4C_config.hpp"

#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Nodes
{
  class Node;
}

namespace FLD
{
  /*!
  \brief Helper function for the nodal nullspace of fluid elements

  \param node (in):    node to calculate the nullspace on
    \param numdof (in):  number of degrees of freedom
    \param dimnsp (in):  dimension of the nullspace
                         */
  Core::LinAlg::SerialDenseMatrix compute_fluid_null_space(
      const Core::Nodes::Node& node, const int numdof, const int dimnsp);
}  // namespace FLD

FOUR_C_NAMESPACE_CLOSE

#endif
