// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_EHL_INTEGRATOR_HPP
#define FOUR_C_CONTACT_EHL_INTEGRATOR_HPP

#include "4C_config.hpp"

#include "4C_contact_integrator.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_utils_pairedvector.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
  class SerialDenseMatrix;
}  // namespace Core::LinAlg

namespace CONTACT
{
  class IntegratorEhl : public CONTACT::Integrator
  {
   public:
    /*!
     \brief Constructor  with shape function specification

     Constructs an instance of this class using a specific type of shape functions.<br>
     Note that this is \b not a collective call as overlaps are
     integrated in parallel by individual processes.<br>
     Note also that this constructor relies heavily on the
     Core::FE::IntegrationPoints structs to get Gauss points
     and corresponding weights.

     */
    IntegratorEhl(Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
        : Integrator(params, eletype, comm)
    {
    }


   protected:
    /*!
     \brief Perform integration at GP
     */
    void integrate_gp_2d(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
        double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
        Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi) override;

    /*!
     \brief Perform integration at GP
     */
    void integrate_gp_3d(Mortar::Element& sele, Mortar::Element& mele,
        Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
        Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
        Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
        double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
        Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi) override;

   private:
    // integrate surface gradient
    void gp_weighted_surf_grad_and_deriv(Mortar::Element& sele, const double* xi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxigp,
        const Core::LinAlg::SerialDenseVector& lmval,
        const Core::LinAlg::SerialDenseMatrix& lmderiv,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
        const Core::LinAlg::SerialDenseMatrix& sderiv2, const double& wgt, const double& jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap);

    // integrate relative and average tangential velocity
    void gp_weighted_av_rel_vel(Mortar::Element& sele, Mortar::Element& mele,
        const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseVector& lmval,
        const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& sderiv,
        const Core::LinAlg::SerialDenseMatrix& mderiv,
        const Core::LinAlg::SerialDenseMatrix& lmderiv,
        const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
        const double& wgt, const double& jac, const Core::Gen::Pairedvector<int, double>& derivjac,
        const double* normal, const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit,
        const double& gap, const Core::Gen::Pairedvector<int, double>& deriv_gap, const double* sxi,
        const double* mxi, const std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi);
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
