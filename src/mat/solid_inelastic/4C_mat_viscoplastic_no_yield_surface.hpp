// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOPLASTIC_NO_YIELD_SURFACE_HPP
#define FOUR_C_MAT_VISCOPLASTIC_NO_YIELD_SURFACE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for the ViscoPlasticNoYieldSurface material
    class ViscoPlasticNoYieldSurface : public Core::Mat::PAR::Parameter
    {
     public:
      explicit ViscoPlasticNoYieldSurface(const Core::Mat::PAR::Parameter::Data& matdata);
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! @name return methods of material parameters
      //! @{

      //! return Density  \f$ \rho \f$
      double density() const { return density_; };
      //! return Poisson's ratio \f$ \nu \f$
      double nue() const { return nue_; };
      //! return Young's modulus  \f$ E \f$
      double young() const { return young_; };
      //! return temperature \f$ T \f$
      double temperature() const { return temperature_; };
      //! return plastic shear strain rate pre-exp-factor \f$ A \f$
      double pre_exp_fac() const { return pre_exp_fac_; };
      //! return activation energy \f$ Q \f$
      double activation_energy() const { return activation_energy_; };
      //! return gas constant \f$ R \f$
      double gas_constant() const { return gas_constant_; };
      //! return strain-rate-sensitivity \f$ m \f$
      double strain_rate_sensitivity() const { return strain_rate_sensitivity_; };
      //! return flow resistance pre-factor \f$ H_0 \f$
      double flow_res_pre_fac() const { return flow_res_pre_fac_; };
      //! return initial flow resistance \f$ S^0 \f$
      double init_flow_res() const { return init_flow_res_; };
      //! return flow resistance exponent \f$ a \f$
      double flow_res_exp() const { return flow_res_exp_; };
      //! return flow resistance saturation factor \f$ S_* \f$
      double flow_res_sat_fac() const { return flow_res_sat_fac_; };
      //! return flow resistance saturation exponent \f$ b \f$
      double flow_res_sat_exp() const { return flow_res_sat_exp_; };

      //! @}

     private:
      //! @name material parameters
      //! @{

      //! Density \f$ \rho \f$
      const double density_;
      //! Possion's ratio \f$ \nu \f$
      const double nue_;
      //! Young's modulus \f$ E \f$
      const double young_;
      //! temperature \f$ T \f$
      const double temperature_;
      //! plastic shear strain rate pre-exp-factor \f$ A \f$
      const double pre_exp_fac_;
      //! activation energy \f$ Q \f$
      const double activation_energy_;
      //! gas constant \f$ R \f$
      const double gas_constant_;
      //! strain-rate-sensitivity \f$ m \f$
      const double strain_rate_sensitivity_;
      //! initial flow resistance \f$ S^0 \f$
      const double init_flow_res_;
      //! flow resistance exponent \f$ a \f$
      const double flow_res_exp_;
      //! flow resistance pre-factor \f$ H_0 \f$
      const double flow_res_pre_fac_;
      //! flow resistance saturation factor \f$ S_* \f$
      const double flow_res_sat_fac_;
      //! flow resistance saturation exponent \f$ b \f$
      const double flow_res_sat_exp_;

      //! @}
    };
  }  // namespace PAR

  class ViscoPlasticNoYieldSurfaceType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ViscoPlasticNoYieldSurfaceType"; }

    static ViscoPlasticNoYieldSurfaceType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ViscoPlasticNoYieldSurfaceType instance_;
  };

  /*----------------------------------------------------------------------*/
  // forward declaration
  struct PreCalculatedTerms;

  /*!
   * \brief Implementation of a visco-plastic finite strain material with no yield surface.
   *
   * This class implements the visco-plastic finite strain material with no yield surface based on
   * the following papers / student work:
   *   -# L. Anand et al., An Elastic-Viscoplastic Model for Lithium", Journal of The
   *      Electrochemical Society, 2019.
   *   -# G. Weber et al., Finite deformation constitutive equations and a time integration
   *      procedure for isotropic, hyperelastic-viscoplastic solids.", Computer Methods in Applied
   *      Mechanics and Engineering, 1990.
   *   -# Details on the implementation in master's thesis of Philipp Bofinger, Supervisor:
   *      Christoph Schmidt.
   *
   * \note The time integration of the internal evolution equations is a backward one-step time
   *       integration algorithm.
   */
  class ViscoPlasticNoYieldSurface : public So3Material
  {
   public:
    //! construct empty material object
    ViscoPlasticNoYieldSurface();

    //! construct the material object given material parameters
    explicit ViscoPlasticNoYieldSurface(Mat::PAR::ViscoPlasticNoYieldSurface* params);

    int unique_par_object_id() const override
    {
      return ViscoPlasticNoYieldSurfaceType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

   private:
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_vp_no_yield_surface;
    }

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ViscoPlasticNoYieldSurface>(*this);
    }

    //! @name Evaluation methods
    //! @{

    /*!
     * @param[in] Me_trial_Vstress  trial stresses in stress-like Voigt notation
     * @param[in] p                 mean normal pressure
     * @return deviatoric trial stresses
     */
    Core::LinAlg::Matrix<3, 3>& calculate_deviatoric_trial_stresses(
        const Core::LinAlg::Matrix<6, 1>& Me_trial_Vstress, const double p) const;

    /*!
     * @brief elastic stiffness tensor in intermediate configuration is calculated
     *
     * @param[in] eigen_vectors  eigenvectors of \f$ F^{*}_\text{e} \f$
     * @param[in] eigen_values   eigenvalues of \f$ F^{*}_\text{e} \f$
     * @return elastic stiffness tensor in intermediate configuration
     */
    Core::LinAlg::Matrix<Mat::NUM_STRESS_3D, Mat::NUM_STRESS_3D>& calculate_elastic_stiffness(
        const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
        const Core::LinAlg::Matrix<3, 1>& eigen_values) const;

    /*!
     * @brief calculate linearization for local newton loop of internal evolution equations
     *
     * @param[in] equ_tens_stress_np  current equivalent tensile stress
     * @param[in] flow_resistance_np  current flow resistance
     * @param[in] terms               terms that can be precalculated and reused several times
     * @return Linearization matrix
     */
    Core::LinAlg::Matrix<2, 2>& calculate_linearization(const double equ_tens_stress_np,
        const double flow_resistance_np, const Mat::PreCalculatedTerms& terms);

    /*!
     * @brief calculate the residual for the equations solved within the local newton loop of
     * internal evolution equations
     *
     * @param[in] equ_tens_stress_np     current equivalent tensile stress
     * @param[in] equ_tens_trial_stress  last converged equivalent tensile stress
     * @param[in] flow_resistance_np     current flow resistance
     * @param[in] flow_resistance_n      last converged flow resistance
     * @param[in] terms                  terms that can be precalculated and reused several times
     * @return residual vector
     */
    Core::LinAlg::Matrix<2, 1>& calculate_residual(const double equ_tens_stress_np,
        const double equ_tens_trial_stress, const double flow_resistance_np,
        const double flow_resistance_n, const PreCalculatedTerms& terms);

    /*!
     * @param[in] defgrd    deformation gradient
     * @param[in] Re_trial  trial elastic rotation tensor
     * @param[in] Me        energy conjugated stress tensor
     * @return second Piola--Kirchhoff stresses
     */
    Core::LinAlg::Matrix<3, 3>& calculate_second_piola_kirchhoff_stresses(
        const Core::LinAlg::Matrix<3, 3>* defgrd, const Core::LinAlg::Matrix<3, 3>& Re_trial,
        const Core::LinAlg::Matrix<3, 3>& Me) const;

    /*!
     * @param[in] Fe_trial        trial elastic deformation gradient \f$ F^{*}_\text{e} \f$
     * @param[out] eigen_values   eigenvalues of \f$ F^{*}_\text{e} \f$
     * @param[out] eigen_vectors  eigenvectors of \f$ F^{*}_\text{e} \f$
     */
    void calculate_trial_elastic_defgrad_eigenvalues_and_eigenvectors(
        const Core::LinAlg::Matrix<3, 3>& Fe_trial, Core::LinAlg::Matrix<3, 1>& eigen_values,
        Core::LinAlg::Matrix<3, 3>& eigen_vectors) const;

    /*!
     * @param[in] Fe_trial       trial elastic deformation gradient \f$ F^{*}_\text{e} \f$
     * @param[in] eigen_vectors  eigenvectors of \f$ F^{*}_\text{e} \f$
     * @param[in] eigen_values   eigenvalues of \f$ F^{*}_\text{e} \f$
     * @return trial elastic rotation tensor \f$ R^{*}_\text{e} \f$
     */
    Core::LinAlg::Matrix<3, 3>& calculate_trial_elastic_rotation(
        const Core::LinAlg::Matrix<3, 3>& Fe_trial, const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
        const Core::LinAlg::Matrix<3, 1>& eigen_values) const;

    /*!
     * @param[in] eigen_vectors  eigenvectors of \f$ F^{*}_\text{e} \f$
     * @param[in] eigen_values   eigenvalues of \f$ F^{*}_\text{e} \f$
     * @return logarithmic elastic strains in strain-like Voigt notation
     */
    Core::LinAlg::Matrix<6, 1>& calculate_log_elastic_strain_in_strain_like_voigt_notation(
        const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
        const Core::LinAlg::Matrix<3, 1>& eigen_values) const;

    /*!
     * @param[in] Me_trial_dev  deviatoric part of trial elastic stresses
     * @return trial elastic equivalent stress [sqrt(3/2 * trace(Me_trial_dev . Me_trial_dev))]
     */
    double calculate_trial_equivalent_stress(const Core::LinAlg::Matrix<3, 3>& Me_trial_dev) const;

    /*!
     * @brief inverse viscous deformation gradient is updated and returned
     *
     * @param[in] last_iFv       last converged inverse viscous deformation gradient
     * @param[in] eigen_vectors  eigenvectors of \f$ F^{*}_\text{e} \f$
     * @param[in] eigen_values   eigenvalues of \f$ F^{*}_\text{e} \f$
     * @param[in] eta            deviatoric elastic deformation share
     * @return inverse viscous deformation gradient
     */
    Core::LinAlg::Matrix<3, 3>& calculate_updated_inverse_viscous_defgrad(
        const Core::LinAlg::Matrix<3, 3>& last_iFv, const Core::LinAlg::Matrix<3, 3>& eigen_vectors,
        const Core::LinAlg::Matrix<3, 1>& eigen_values, const double eta) const;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /*!
     * @brief Local Newton-loop to solve for the internal evolution equations
     *
     * @param[in/out] x  solution vector
     * @param[in]     dt time step
     */
    void local_newton_loop(Core::LinAlg::Matrix<2, 1>& x, double dt);

    /*!
     * @brief Calculate and return terms of the formulation that can be reused several times
     *
     * @param[in] equ_tens_stress_np  current equivalent tensile stress
     * @param[in] flow_resistance_np  current flow resistance
     * @param[in] dt                  time step
     * @return pre-calculated terms
     */
    PreCalculatedTerms pre_calculate_terms(
        const double equ_tens_stress_np, const double flow_resistance_np, const double dt);

    //! @}

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /*!
     * @brief computes isotropic elasticity tensor in matrix notion for 3d
     *
     * @param[out] cmat elasticity tensor in intermediate configuration
     */
    void setup_cmat(Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>& cmat);

    void update() override;

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    double density() const override { return params_->density(); }

    //! material parameters
    Mat::PAR::ViscoPlasticNoYieldSurface* params_;

    //! inverse plastic deformation gradient for each Gauss point at last converged state
    std::vector<Core::LinAlg::Matrix<3, 3>> last_plastic_defgrd_inverse_;

    //! current inverse plastic deformation gradient for each Gauss point
    std::vector<Core::LinAlg::Matrix<3, 3>> current_plastic_defgrd_inverse_;

    //! flow resistance 'S' for each Gauss point at last converged state
    std::vector<double> last_flowres_isotropic_;

    //! current flow resistance 'S' for each Gauss point
    std::vector<double> current_flowres_isotropic_;
  };
}  // namespace Mat

/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
