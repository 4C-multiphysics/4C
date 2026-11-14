// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_STRATEGY_FSI_HPP
#define FOUR_C_CONTACT_NITSCHE_STRATEGY_FSI_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_strategy.hpp"
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
  class NitscheStrategyFsi : public NitscheStrategy
  {
   public:
    //! Standard constructor
    NitscheStrategyFsi(const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
        int dim, MPI_Comm comm, double alphaf, int maxdof)
        : NitscheStrategy(
              dof_row_map, NodeRowMap, params, std::move(interface), dim, comm, alphaf, maxdof),
          pen_n_(params.get<double>("PENALTYPARAM")),
          weighting_(
              Teuchos::getIntegralValue<CONTACT::NitscheWeighting>(params, "NITSCHE_WEIGHTING"))
    {
      if (Teuchos::getIntegralValue<CONTACT::FrictionType>(params, "FRICTION") !=
          CONTACT::FrictionType::none)
        FOUR_C_THROW("NitscheStrategyFsi: No frictional contact implemented for Nitsche FSCI!");
    }

    //! Shared data constructor
    NitscheStrategyFsi(const std::shared_ptr<CONTACT::AbstractStrategyDataContainer>& data_ptr,
        const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<CONTACT::Interface>> interface,
        int dim, MPI_Comm comm, double alphaf, int maxdof)
        : NitscheStrategy(data_ptr, dof_row_map, NodeRowMap, params, std::move(interface), dim,
              comm, alphaf, maxdof),
          pen_n_(params.get<double>("PENALTYPARAM")),
          weighting_(
              Teuchos::getIntegralValue<CONTACT::NitscheWeighting>(params, "NITSCHE_WEIGHTING"))
    {
      if (Teuchos::getIntegralValue<CONTACT::FrictionType>(params, "FRICTION") !=
          CONTACT::FrictionType::none)
        FOUR_C_THROW("NitscheStrategyFsi: No frictional contact implemented for Nitsche FSCI!");
    }
    //! Evaluate and apply RHS and Stiffness Matrix for Contact
    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor) override;

    //! Set Contact State and update search tree and normals
    void set_state(
        const Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec) override;

    //! The the contact state at local coord of Element cele and compare to the fsi_traction,
    //! return true if contact is evaluated, return false if FSI is evaluated
    bool check_nitsche_contact_state(CONTACT::Element* cele,
        const Core::LinAlg::Matrix<2, 1>& xsi,  ///< local coord on the ele element
        const double& full_fsi_traction,        ///< stressfluid + penalty
        double& gap                             ///< gap
    );

   protected:
    //! Update search tree and normals
    void do_contact_search();

   private:
    //! Nitsche normal penalty parameter
    double pen_n_;
    //! Nitsche weighting strategy
    CONTACT::NitscheWeighting weighting_;
  };

  namespace Utils
  {
    //! The the contact state at local coord of Element cele and compare to the fsi_traction,
    //! return true if contact is evaluated, return false if FSI is evaluated
    bool check_nitsche_contact_state(CONTACT::Interface& contactinterface,  ///< Contact interface
        const double& pen_n,                    ///< Nitsche normal penalty parameter
        CONTACT::NitscheWeighting weighting,    ///< Nitsche interface stress weighting
        CONTACT::Element* cele,                 ///< the contact element
        const Core::LinAlg::Matrix<2, 1>& xsi,  ///< local coord on the ele element
        const double& full_fsi_traction,        ///< stressfluid + penalty
        double& gap                             ///< gap
    );
  }  // namespace Utils
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
