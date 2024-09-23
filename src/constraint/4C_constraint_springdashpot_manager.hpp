/*----------------------------------------------------------------------*/
/*! \file

\brief Methods for spring and dashpot constraints / boundary conditions:

\level 2


*----------------------------------------------------------------------*/

#ifndef FOUR_C_CONSTRAINT_SPRINGDASHPOT_MANAGER_HPP
#define FOUR_C_CONSTRAINT_SPRINGDASHPOT_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_Operator.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace CONSTRAINTS
{
  class SpringDashpot;

  class SpringDashpotManager
  {
   public:
    /*!
      \brief Constructor
    */
    SpringDashpotManager(Teuchos::RCP<Core::FE::Discretization> dis);

    /*!
     \brief Return if there are spring dashpots
    */
    bool have_spring_dashpot() const { return havespringdashpot_; };

    //! add contribution of spring dashpot BC to residual vector and stiffness matrix
    void stiffness_and_internal_forces(Teuchos::RCP<Core::LinAlg::SparseMatrix> stiff,
        Teuchos::RCP<Epetra_Vector> fint, Teuchos::RCP<Epetra_Vector> disn,
        Teuchos::RCP<Epetra_Vector> veln, Teuchos::ParameterList parlist);

    //! update for each new time step
    void update();

    //! output of gap, normal, and nodal stiffness
    void output(Teuchos::RCP<Core::IO::DiscretizationWriter> output,
        Teuchos::RCP<Core::FE::Discretization> discret, Teuchos::RCP<Epetra_Vector> disp);

    //! output of prestressing offset for restart
    void output_restart(Teuchos::RCP<Core::IO::DiscretizationWriter> output_restart,
        Teuchos::RCP<Core::FE::Discretization> discret, Teuchos::RCP<Epetra_Vector> disp);

    /*!
     \brief Read restart information
    */
    void read_restart(Core::IO::DiscretizationReader& reader, const double& time);

    //! reset spring after having done a MULF prestressing update (mhv 12/2015)
    void reset_prestress(Teuchos::RCP<Epetra_Vector> disold);

   private:
    Teuchos::RCP<Core::FE::Discretization> actdisc_;    ///< standard discretization
    std::vector<Teuchos::RCP<SpringDashpot>> springs_;  ///< all spring dashpot instances

    bool havespringdashpot_;  ///< are there any spring dashpot BCs at all?
    int n_conds_;             ///< number of spring dashpot conditions
  };                          // class
}  // namespace CONSTRAINTS
FOUR_C_NAMESPACE_CLOSE

#endif
