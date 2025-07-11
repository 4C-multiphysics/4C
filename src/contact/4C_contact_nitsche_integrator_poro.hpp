// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_INTEGRATOR_PORO_HPP
#define FOUR_C_CONTACT_NITSCHE_INTEGRATOR_PORO_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_integrator.hpp"
#include "4C_utils_pairedvector.hpp"


FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
}

namespace CONTACT
{
  class IntegratorNitschePoro : public IntegratorNitsche
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
    IntegratorNitschePoro(
        Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm);

   protected:
    /*!
     \brief Perform integration at GP
            This is where the distinction between methods should be,
            i.e. mortar, augmented, gpts,...
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
            This is where the distinction between methods should be,
            i.e. mortar, augmented, gpts,...
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

    /*!
    \brief Evaluate cauchy stress component and its derivatives
    */
    template <int dim>
    void so_ele_cauchy(Mortar::Element& moEle, double* boundary_gpcoord,
        std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
        const Core::LinAlg::Matrix<dim, 1>& normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
        const Core::LinAlg::Matrix<dim, 1>& direction,
        std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, const double w,
        double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt,
        Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_p);

   private:
    /*!
    \brief evaluate GPTS forces and linearization at this gp
    */
    template <int dim>
    void gpts_forces(Mortar::Element& sele, Mortar::Element& mele,
        const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxi,
        const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, const double jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
        const double gap, const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
        std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double* sxi, double* mxi);

   protected:
    template <int dim>
    void integrate_test(const double fac, Mortar::Element& ele,
        const Core::LinAlg::SerialDenseVector& shape, const Core::LinAlg::SerialDenseMatrix& deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, const double jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
        const double test_val, const Core::Gen::Pairedvector<int, double>& test_deriv_d,
        const Core::Gen::Pairedvector<int, double>& test_deriv_p,
        const Core::LinAlg::Matrix<dim, 1>& test_dir,
        const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);

    template <int dim>
    void integrate_poro_no_out_flow(const double fac, Mortar::Element& ele, double* xi,
        const Core::LinAlg::SerialDenseVector& shape, const Core::LinAlg::SerialDenseMatrix& deriv,
        const double jac, const Core::Gen::Pairedvector<int, double>& jacintcellmap,
        const double wgt, const Core::LinAlg::Matrix<dim, 1>& normal,
        const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
        Mortar::Element& otherele, const Core::LinAlg::SerialDenseVector& othershape);

    bool get_poro_pressure(Mortar::Element& ele, const Core::LinAlg::SerialDenseVector& shape,
        Mortar::Element& otherele, const Core::LinAlg::SerialDenseVector& othershape,
        double& poropressure);

    void get_poro_quantitiesat_gp(Mortar::Element& ele, double* xi, double& spresgp,  //(in)
        double& sJ, std::map<int, double>& sJLin, double& sporosity, double& sdphi_dp,
        double& sdphi_dJ);  // out

   private:
    bool no_penetration_;  // no outflow in contact zone ...
    double dv_dd_;         // 1/(theta*dt) for OST
  };
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
