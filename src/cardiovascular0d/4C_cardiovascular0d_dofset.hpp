/*----------------------------------------------------------------------*/
/*! \file

\brief A set of degrees of freedom

\level 2


*----------------------------------------------------------------------*/
#ifndef FOUR_C_CARDIOVASCULAR0D_DOFSET_HPP
#define FOUR_C_CARDIOVASCULAR0D_DOFSET_HPP

#include "4C_config.hpp"

#include "4C_cardiovascular0d_mor_pod.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset.hpp"

#include <Epetra_IntVector.h>
#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>

#include <list>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace UTILS
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
  are used by the algorithm's Core::LinAlg::Vector classes amoung others.

  It is not connected to elements or nodes.
  <h3>Invariants</h3>

  There are two possible states in this class: Reset and setup. To
  change back and forth use assign_degrees_of_freedom() and reset().


  \author tk     */
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
      int lmin = dofrowmap_->MinMyGID();
      if (dofrowmap_->NumMyElements() == 0) lmin = std::numeric_limits<int>::max();
      int gmin = std::numeric_limits<int>::max();
      dofrowmap_->Comm().MinAll(&lmin, &gmin, 1);
      return gmin;
    };

    //@}

    //! @name Construction

    /// Assign dof numbers using all elements and nodes of the discretization.
    virtual int assign_degrees_of_freedom(const Teuchos::RCP<Core::FE::Discretization> dis,
        const int ndofs, const int start,
        const Teuchos::RCP<FourC::Cardiovascular0D::ProperOrthogonalDecomposition> mor);

    /// reset all internal variables
    void reset() override;

    //@}

   protected:
  };  // class Cardiovascular0DDofSet
}  // namespace UTILS


// << operator
std::ostream& operator<<(std::ostream& os, const UTILS::Cardiovascular0DDofSet& dofset);


FOUR_C_NAMESPACE_CLOSE

#endif
