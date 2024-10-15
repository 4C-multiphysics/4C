/*! \file
\brief Nullspace computation for a discretization
\level 0
*/

#ifndef FOUR_C_FEM_DISCRETIZATION_NULLSPACE_HPP
#define FOUR_C_FEM_DISCRETIZATION_NULLSPACE_HPP

#include "4C_config.hpp"

#include "4C_linalg_multi_vector.hpp"

#include <Teuchos_RCPDecl.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;

  /*!
   \brief Calculate the nullspace based on a given discretization

  The nullspace is build by looping over all nodes of a discretization and stored
          in the respective variable.

     \param dis (in): discretization
     \param numdf (in): number of degrees of freedom
     \param dimns (in): nullspace dimension
     \param map (in): nullspace map
      */
  Teuchos::RCP<Core::LinAlg::MultiVector<double>> compute_null_space(
      const Core::FE::Discretization& dis, const int numdf, const int dimns,
      const Epetra_Map& dofmap);
}  // namespace Core::FE

FOUR_C_NAMESPACE_CLOSE

#endif
