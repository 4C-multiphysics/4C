// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_INTEGRATOR_HPP
#define FOUR_C_CONTACT_NITSCHE_INTEGRATOR_HPP

#include "4C_config.hpp"

#include "4C_contact_integrator.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_general_utils_gausspoints.hpp"
#include "4C_mortar_element.hpp"
#include "4C_utils_pairedvector.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseVector;
}

namespace CONTACT
{
  class IntegratorNitsche : public CONTACT::Integrator
  {
   public:
    /*!
     * \brief Constructor with shape function specification
     *
     * Constructs an instance of this class using a specific type of shape functions.<br> Note that
     * this is \b not a collective call as overlaps are integrated in parallel by individual
     * processes.<br> Note also that this constructor relies heavily on the
     * Core::FE::IntegrationPoints structs to get Gauss points and corresponding weights.
     *
     * @param [in] params  interface contact parameter list
     * @param [in] eletype shape of integration cell for segment based integration or slave side
     *                     mortar contact element for element based integration
     * @param [in] comm    contact interface communicator
     */
    IntegratorNitsche(Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
        : Integrator(params, eletype, comm),
          theta_(params.get<double>("NITSCHE_THETA")),
          theta_2_(params.get<double>("NITSCHE_THETA_2")),
          nit_wgt_(
              Teuchos::getIntegralValue<CONTACT::NitscheWeighting>(params, "NITSCHE_WEIGHTING")),
          ppn_(imortar_.get<double>("PENALTYPARAM")),
          ppt_(imortar_.get<double>("PENALTYPARAMTAN")),
          frcoeff_(imortar_.get<double>("FRCOEFF", -1.)),
          frbound_(imortar_.get<double>("FRBOUND", -1.)),
          frtype_(Teuchos::getIntegralValue<CONTACT::FrictionType>(imortar_, "FRICTION")),
          dt_(imortar_.get<double>("TIMESTEP")),
          two_half_pass_(imortar_.get<bool>("Two_half_pass"))
    {
    }

   protected:
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
     * @brief Evaluate cauchy stress component and its derivatives
     *
     * @tparam dim   dimension of the problem
     * @param[in] moEle  mortar element
     * @param[in] boundary_gpcoord      Gauss point coordinates
     * @param[in] boundary_gpcoord_lin  directional derivative of Gauss point coordinates
     * @param[in] gp_wgt        Gauss point weight
     * @param[in] normal        normal
     * @param[in] normal_deriv  directional derivative of normal
     * @param[in] direction        direction of evaluation (e.g. normal or tangential direction)
     * @param[in] direction_deriv  directional derivative of direction of evaluation
     * @param[in] w                Nitsche weight
     * @param[out] cauchy_nt        Cauchy stress tensor contracted with normal vector and direction
     *                              vector \f[ \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot
     *                              \boldsymbol{t} \f]
     * @param[out] deriv_sigma_nt   directional derivative of cauchy_nt \f[ \frac{ \mathrm{d}
     *                              \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot
     *                              \boldsymbol{t}}{\mathrm{d} \boldsymbol{d}} \f]
     * @param[out] adjoint_test     directional derivative of cauchy_nt \f[ \frac{ \mathrm{d}
     *                              \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot
     *                              \boldsymbol{t}}{\mathrm{d} \boldsymbol{d}} \f]
     * @param[out] deriv_adjoint_test   second directional derivative of cauchy_nt \f[ \frac{
     *                                  \mathrm{d}^2 \boldsymbol{\sigma} \cdot \boldsymbol{n} \cdot
     *                                  \boldsymbol{t}}{\mathrm{d} \boldsymbol{d}^2} \f]
     */
    template <int dim>
    void so_ele_cauchy(Mortar::Element& moEle, double* boundary_gpcoord,
        std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, double gp_wgt,
        const Core::LinAlg::Matrix<dim, 1>& normal,
        std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
        const Core::LinAlg::Matrix<dim, 1>& direction,
        std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, double w,
        double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt,
        Core::LinAlg::SerialDenseVector& adjoint_test,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>& deriv_adjoint_test);

   private:
    /*!
     * @brief evaluate GPTS forces and linearization at this gp
     *
     * @tparam dim  dimension of the problem
     * @param[in] sele  slave side mortar element
     * @param[in] mele  master side mortar element
     * @param[in] sval  slave side shape function evaluated at current Gauss point
     * @param[in] sderiv  slave side shape function derivative at current Gauss point
     * @param[in] dsxi    directional derivative of slave side Gauss point coordinates
     * @param[in] mval    master side shape function evaluated at current Gauss point
     * @param[in] mderiv  master side shape function derivative at current Gauss point
     * @param[in] dmxi    directional derivative of master side Gauss point coordinates
     * @param[in] jac            Jacobian determinant of integration cell
     * @param[in] jacintcellmap  directional derivative of cell Jacobian
     * @param[in] wgt     Gauss point weight
     * @param[in] gap     gap
     * @param[in] dgapgp  directional derivative of gap
     * @param[in] gpn         Gauss point normal
     * @param[in] dnmap_unit  directional derivative of Gauss point normal
     * @param[in] sxi         slave side Gauss point coordinates
     * @param[in] mxi         master side Gauss point coordinates
     */
    template <int dim>
    void gpts_forces(Mortar::Element& sele, Mortar::Element& mele,
        const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dsxi,
        const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, double jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap, double wgt, double gap,
        const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
        std::vector<Core::Gen::Pairedvector<int, double>>& deriv_contact_normal, double* sxi,
        double* mxi);

   protected:
    template <int dim>
    void build_adjoint_test(Mortar::Element& moEle, double fac,
        const Core::LinAlg::SerialDenseMatrix& dsntdd,
        const Core::LinAlg::SerialDenseMatrix& d2sntdd2,
        const Core::LinAlg::SerialDenseMatrix& d2sntDdDn,
        const Core::LinAlg::SerialDenseMatrix& d2sntDdDt,
        const Core::LinAlg::SerialDenseMatrix& d2sntDdDpxi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& boundary_gpcoord_lin,
        Core::LinAlg::Matrix<dim, dim> derivtravo_slave,
        const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv,
        Core::LinAlg::SerialDenseVector& adjoint_test,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>& deriv_adjoint_test);

    template <int dim>
    void integrate_test(double fac, Mortar::Element& ele,
        const Core::LinAlg::SerialDenseVector& shape, const Core::LinAlg::SerialDenseMatrix& deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, double jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap, double wgt, double test_val,
        const Core::Gen::Pairedvector<int, double>& test_deriv,
        const Core::LinAlg::Matrix<dim, 1>& test_dir,
        const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);

    template <int dim>
    void integrate_adjoint_test(double fac, double jac,
        const Core::Gen::Pairedvector<int, double>& jacintcellmap, double wgt, double test,
        const Core::Gen::Pairedvector<int, double>& deriv_test, Mortar::Element& moEle,
        Core::LinAlg::SerialDenseVector& adjoint_test,
        Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>& deriv_adjoint_test);

    //! nitsche theta
    double theta_;
    //! nitsche theta 2
    double theta_2_;
    //! type of nitsche weighting
    CONTACT::NitscheWeighting nit_wgt_;
    //! nitsche penalty parameter in normal direction
    double ppn_;
    //! nitsche penalty parameter in tangential direction
    double ppt_;
    //! coulomb friction coefficient
    double frcoeff_;
    //! tresca friction bound
    double frbound_;
    //! type of contact friction law
    CONTACT::FrictionType frtype_;
    //! time step size
    double dt_;
    //! flag indicating if unbiased two half pass algorithm is activated or not
    bool two_half_pass_;
  };

  namespace Utils
  {
    //! map local surface coordinate to local parent coordinate
    template <int dim>
    void map_gp_to_parent(Mortar::Element& moEle, double* boundary_gpcoord, double wgt,
        Core::LinAlg::Matrix<dim, 1>& pxsi, Core::LinAlg::Matrix<dim, dim>& derivtravo_slave);

    //! templated version of MapGPtoParent
    template <Core::FE::CellType parentdistype, int dim>
    void inline so_ele_gp(Mortar::Element& sele, double wgt, const double* gpcoord,
        Core::LinAlg::Matrix<dim, 1>& pxsi, Core::LinAlg::Matrix<dim, dim>& derivtrafo);

    //! actually not the velocity but the displacement increment
    template <int dim>
    void rel_vel_invariant(Mortar::Element& sele, const double* sxi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
        const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
        Mortar::Element& mele, const double* mxi,
        const std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi,
        const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
        const double& gap, const Core::Gen::Pairedvector<int, double>& deriv_gap,
        Core::LinAlg::Matrix<dim, 1>& relVel,
        std::vector<Core::Gen::Pairedvector<int, double>>& relVel_deriv, double fac = 1.0);

    template <int dim>
    void rel_vel(Mortar::Element& ele, const Core::LinAlg::SerialDenseVector& shape,
        const Core::LinAlg::SerialDenseMatrix& deriv,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, double fac,
        Core::LinAlg::Matrix<dim, 1>& relVel,
        std::vector<Core::Gen::Pairedvector<int, double>>& relVel_deriv);

    template <int dim>
    void vector_scalar_product(const Core::LinAlg::Matrix<dim, 1>& v1,
        const std::vector<Core::Gen::Pairedvector<int, double>>& v1d,
        const Core::LinAlg::Matrix<dim, 1>& v2,
        const std::vector<Core::Gen::Pairedvector<int, double>>& v2d, double& val,
        Core::Gen::Pairedvector<int, double>& val_deriv);

    void build_tangent_vectors3_d(const double* np,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dn, double* t1p,
        std::vector<Core::Gen::Pairedvector<int, double>>& dt1, double* t2p,
        std::vector<Core::Gen::Pairedvector<int, double>>& dt2);

    template <int dim>
    void build_tangent_vectors(const double* np,
        const std::vector<Core::Gen::Pairedvector<int, double>>& dn, double* t1p,
        std::vector<Core::Gen::Pairedvector<int, double>>& dt1, double* t2p,
        std::vector<Core::Gen::Pairedvector<int, double>>& dt2);

    /*!
     * @brief determine weights (harmonic/slave/master) and scale penalty
     *
     * @param[in] sele     slave-side nitsche weight
     * @param[in] mele     master-side nitsche weight
     * @param[in] nit_wgt  weighting methood for nitsche contact
     * @param[in] dt       time step size
     * @param[out] ws      slave-side nitsche weight
     * @param[out] wm      master-side nitsche weight
     * @param[out] pen     scaled nitsche penalty parameter in normal direction
     * @param[out] pet     scaled nitsche penalty parameter in tangential direction
     */
    void nitsche_weights_and_scaling(Mortar::Element& sele, Mortar::Element& mele,
        CONTACT::NitscheWeighting nit_wgt, double dt, double& ws, double& wm, double& pen,
        double& pet);


    // --- template and inline functions --- //

    template <Core::FE::CellType parentdistype, int dim>
    void inline so_ele_gp(Mortar::Element& sele, const double wgt, const double* gpcoord,
        Core::LinAlg::Matrix<dim, 1>& pxsi, Core::LinAlg::Matrix<dim, dim>& derivtrafo)
    {
      Core::FE::CollectedGaussPoints intpoints =
          Core::FE::CollectedGaussPoints(1);  // reserve just for 1 entry ...
      intpoints.append(gpcoord[0], gpcoord[1], 0.0, wgt);

      // get coordinates of gauss point w.r.t. local parent coordinate system
      Core::LinAlg::SerialDenseMatrix pqxg(1, dim);
      derivtrafo.clear();

      Core::FE::boundary_gp_to_parent_gp<dim>(pqxg, derivtrafo, intpoints,
          sele.parent_element()->shape(), sele.shape(), sele.face_parent_number());

      // coordinates of the current integration point in parent coordinate system
      for (int idim = 0; idim < dim; idim++) pxsi(idim) = pqxg(0, idim);
    }

  }  // namespace Utils
}  // namespace CONTACT
FOUR_C_NAMESPACE_CLOSE

#endif
