// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CARDIOVASCULAR0D_DOFSET_HPP
#define FOUR_C_CARDIOVASCULAR0D_DOFSET_HPP

#include "4C_config.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_sparse_pod.hpp"

#include <list>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Utils
{
  /*!
  \brief A set of degrees of freedom

  \note This is an internal class of the Cardiovascular0D manager that one
  should not need to touch on an ordinary day. It is here to support the
  Cardiovascular0D manager class. And does all the degree of freedom assignmets
  for the Cardiovascular0Ds.

  <h3>Purpose</h3>

  This class represents one set of degrees of freedom for the
  Cardiovascular0Ds in the usual parallel fashion. That is there is a
  dof_row_map() and a DofColMap() that return the maps of the global FE
  system of equation in row and column setting respectively. These maps
  are used by the algorithm's Core::LinAlg::Vector<double> classes among others.

  It is not connected to elements or nodes.
  <h3>Invariants</h3>

  There are two possible states in this class: Reset and setup. To
  change back and forth use assign_degrees_of_freedom() and reset().


  */
  class Cardiovascular0DDofSet : public Core::DOFSets::DofSet
  {
   public:
    /*!
    \brief Standard Constructor

    */
    Cardiovascular0DDofSet();



    //! @name Access methods

    virtual int first_gid()
    {
      int lmin = dofrowmap_->min_my_gid();
      if (dofrowmap_->num_my_elements() == 0) lmin = std::numeric_limits<int>::max();
      int gmin = std::numeric_limits<int>::max();
      Core::Communication::min_all(&lmin, &gmin, 1, dofrowmap_->get_comm());
      return gmin;
    };

    //@}

    //! @name Construction

    /// Assign dof numbers using all elements and nodes of the discretization.
    virtual int assign_degrees_of_freedom(const std::shared_ptr<Core::FE::Discretization> dis,
        const int ndofs, const int start,
        const std::shared_ptr<Core::LinAlg::ProperOrthogonalDecomposition> mor);

    /// reset all internal variables
    void reset() override;

    //@}

   protected:
  };  // class Cardiovascular0DDofSet
}  // namespace Utils


// << operator
std::ostream& operator<<(std::ostream& os, const Utils::Cardiovascular0DDofSet& dofset);


FOUR_C_NAMESPACE_CLOSE

#endif
