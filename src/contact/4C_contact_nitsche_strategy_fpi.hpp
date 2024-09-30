/*---------------------------------------------------------------------*/
/*! \file

\brief Nitsche contact solving strategy for problems with FPI

\level 3


*/
/*---------------------------------------------------------------------*/
#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_FPI_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_FPI_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_strategy_poro.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  class Element;

  /*!
   \brief Contact solving strategy with Nitsche's method.

   This is a specialization of the abstract contact algorithm as defined in AbstractStrategy.
   For a more general documentation of the involved functions refer to CONTACT::AbstractStrategy.

   */
  class NitscheStrategyFpi : public NitscheStrategyPoro
  {
   public:
    //! Standard constructor
    NitscheStrategyFpi(const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<Teuchos::RCP<CONTACT::Interface>> interface,
        int dim, Teuchos::RCP<Epetra_Comm> comm, double alphaf, int maxdof)
        : NitscheStrategyPoro(
              dof_row_map, NodeRowMap, params, std::move(interface), dim, comm, alphaf, maxdof),
          pen_n_(params.get<double>("PENALTYPARAM")),
          weighting_(Teuchos::getIntegralValue<Inpar::CONTACT::NitscheWeighting>(
              params, "NITSCHE_WEIGHTING"))
    {
      if (Teuchos::getIntegralValue<Inpar::CONTACT::FrictionType>(params, "FRICTION") !=
          Inpar::CONTACT::friction_none)
        FOUR_C_THROW("NitscheStrategyFpi: No frictional contact implemented for Nitsche FPSCI!");
    }

    //! Shared data constructor
    NitscheStrategyFpi(const Teuchos::RCP<CONTACT::AbstractStratDataContainer>& data_ptr,
        const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap, Teuchos::ParameterList params,
        std::vector<Teuchos::RCP<CONTACT::Interface>> interface, int dim,
        Teuchos::RCP<const Epetra_Comm> comm, double alphaf, int maxdof)
        : NitscheStrategyPoro(data_ptr, dof_row_map, NodeRowMap, params, std::move(interface), dim,
              comm, alphaf, maxdof),
          pen_n_(params.get<double>("PENALTYPARAM")),
          weighting_(Teuchos::getIntegralValue<Inpar::CONTACT::NitscheWeighting>(
              params, "NITSCHE_WEIGHTING"))
    {
      if (Teuchos::getIntegralValue<Inpar::CONTACT::FrictionType>(params, "FRICTION") !=
          Inpar::CONTACT::friction_none)
        FOUR_C_THROW("NitscheStrategyFpi: No frictional contact implemented for Nitsche FPSCI!");
    }
    //! Set Contact State and update search tree and normals
    void set_state(
        const enum Mortar::StateType& statename, const Core::LinAlg::Vector& vec) override;

    //! The the contact state at local coord of Element cele and compare to the fsi_traction,
    //! return true if contact is evaluated, reture false if FSI is evaluated
    bool check_nitsche_contact_state(CONTACT::Element* cele,
        const Core::LinAlg::Matrix<2, 1>& xsi,  // local coord on the ele element
        const double& full_fsi_traction,        // stressfluid + penalty
        double& gap                             // gap
    );

   protected:
    //! Update search tree and normals
    void do_contact_search();

   private:
    //! Nitsche normal penalty parameter
    double pen_n_;
    //! Nitsche weighting strategy
    Inpar::CONTACT::NitscheWeighting weighting_;
  };
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
