// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_INTEGRATOR_PORO_SCATRA_HPP
#define FOUR_C_CONTACT_NITSCHE_INTEGRATOR_PORO_SCATRA_HPP

#include "4C_config.hpp"

#include "4C_contact_nitsche_integrator.hpp"
#include "4C_utils_pairedvector.hpp"

#include <Epetra_CrsMatrix.h>
#include <Epetra_FEVector.h>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
}

namespace Discret
{
  namespace Elements
  {
    class ScaTraEleParameterTimInt;
    class ScaTraEleParameterBoundary;
  }  // namespace Elements
}  // namespace Discret

namespace CONTACT
{
  class IntegratorNitschePoroScatra : public IntegratorNitsche
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
    IntegratorNitschePoroScatra(
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
        Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_p,
        Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_s);

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

    /*!
     * @brief  integrate the ScaTra residual and linearizations
     *
     * @tparam dim  dimension of the problem
     * @param[in] fac  pre-factor to correct sign dependent on integration of master or slave side
     *                 terms
     * @param[in,out] ele      mortar contact element or integration cell mortar element
     * @param[in] shape_func   shape function evaluated at current Gauss point
     * @param[in] shape_deriv  shape function derivative at current Gauss point
     * @param[in] d_xi_dd      directional derivative of Gauss point coordinates
     * @param[in] jac          Jacobian determinant of integration cell
     * @param[in] d_jac_dd     directional derivative of cell Jacobian
     * @param[in] wgt          Gauss point weight
     * @param[in] test_val     quantity to be integrated
     * @param[in] d_test_val_dd  directional derivative of test_val
     * @param[in] d_test_val_ds  derivative of test_val w.r.t. scalar
     */
    template <int dim>
    void integrate_scatra_test(double fac, Mortar::Element& ele,
        const Core::LinAlg::SerialDenseVector& shape_func,
        const Core::LinAlg::SerialDenseMatrix& shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd, double jac,
        const Core::Gen::Pairedvector<int, double>& d_jac_dd, double wgt, double test_val,
        const Core::Gen::Pairedvector<int, double>& d_test_val_dd,
        const Core::Gen::Pairedvector<int, double>& d_test_val_ds);

    /*!
     * @brief integrate the scatra-structure interaction interface condition
     *
     * @tparam dim  dimension of the problem
     * @param[in,out] slave_ele       slave-side mortar contact element
     * @param[in] slave_shape         slave-side shape function evaluated at current Gauss point
     * @param[in] slave_shape_deriv   slave-side shape function derivative at current Gauss point
     * @param[in] d_slave_xi_dd       slave-side directional derivative of Gauss point coordinates
     * @param[in,out] master_ele      master-side mortar contact element
     * @param[in] master_shape        master-side shape function evaluated at current Gauss point
     * @param[in] master_shape_deriv  master-side shape function derivative at current Gauss point
     * @param[in] d_master_xi_dd      master-side directional derivative of Gauss point coordinates
     * @param[in] cauchy_nn_average_pen_gap         normal contact stress
     * @param[in] d_cauchy_nn_weighted_average_dd   directional derivatives of normal contact stress
     * w.r.t displacement
     * @param[in] d_cauchy_nn_weighted_average_dc   directional derivatives of normal contact stress
     * w.r.t scalar/concentration
     * @param[in] jac                 Jacobian determinant of integration cell
     * @param[in] d_jac_dd            directional derivative of cell Jacobian
     * @param[in] wgt                 Gauss point weight
     */
    template <int dim>
    void integrate_ssi_interface_condition(Mortar::Element& slave_ele,
        const Core::LinAlg::SerialDenseVector& slave_shape,
        const Core::LinAlg::SerialDenseMatrix& slave_shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_slave_xi_dd,
        Mortar::Element& master_ele, const Core::LinAlg::SerialDenseVector& master_shape,
        const Core::LinAlg::SerialDenseMatrix& master_shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_master_xi_dd,
        const double cauchy_nn_average_pen_gap,
        const Core::Gen::Pairedvector<int, double>& d_cauchy_nn_weighted_average_dd,
        const Core::Gen::Pairedvector<int, double>& d_cauchy_nn_weighted_average_dc, double jac,
        const Core::Gen::Pairedvector<int, double>& d_jac_dd, double wgt);

    /*!
     * @brief calculate the concentrations and derivatives at the current Gauss point
     *
     * @tparam dim  dimension of the problem
     * @param[in] ele   mortar contact element or integration cell mortar element
     * @param[in] shape_func   shape function evaluated at current Gauss point
     * @param[in] shape_deriv  shape function derivative at current Gauss point
     * @param[in] d_xi_dd     directional derivative of Gauss point coordinates
     * @param[out] gp_conc    concentration at current Gauss point
     * @param[out] d_conc_dc  derivative of concentration w.r.t. concentration
     * @param[out] d_conc_dd  directional derivative of concentration
     */
    template <int dim>
    void setup_gp_concentrations(Mortar::Element& ele,
        const Core::LinAlg::SerialDenseVector& shape_func,
        const Core::LinAlg::SerialDenseMatrix& shape_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd, double& gp_conc,
        Core::Gen::Pairedvector<int, double>& d_conc_dc,
        Core::Gen::Pairedvector<int, double>& d_conc_dd);

    //! access method to scatra time integration factors
    const Discret::Elements::ScaTraEleParameterTimInt* get_scatra_ele_parameter_tim_int() const
    {
      return scatraparamstimint_;
    }

    //! access method to scatra-scatra coupling specific parameters
    const Discret::Elements::ScaTraEleParameterBoundary* get_scatra_ele_parameter_boundary() const
    {
      return scatraparamsboundary_;
    }

   private:
    bool no_penetration_;  // no outflow in contact zone ...
    double dv_dd_;         // 1/(theta*dt) for OST

    //! scatra time integration factors
    const Discret::Elements::ScaTraEleParameterTimInt* scatraparamstimint_;
    //! scatra-scatra coupling specific parameters
    const Discret::Elements::ScaTraEleParameterBoundary* scatraparamsboundary_;
  };
}  // namespace CONTACT

FOUR_C_NAMESPACE_CLOSE

#endif
