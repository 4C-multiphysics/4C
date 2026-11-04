// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PLASTICGTN_HPP
#define FOUR_C_MAT_PLASTICGTN_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_structure_new_input.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace Utils
  {
    class FunctionOfAnything;
  }
}  // namespace Core

namespace Mat
{
  namespace PAR
  {
    /**
     * \brief Elasto-plastic GTN Material Model
     *
     * This constitutive law accounts for the void coalescence within the plastic deformation. It is
     * useful to study the damage mechanism of metal-based (ductile) material.
     *
     * Reference:
     *   [1] V. Tvergaard, A. Needleman, Analysis of the cup-cone fracture in a round tensile bar,
     * Acta Metallurgica 32 (1) (1984) 157-169.
     *   [2] C. C. Chu, A. Needleman, Void Nucleation Effects
     * in Biaxially Stretched Sheets, Journal of Engineering Materials and Technology 102 (3) (1980)
     * 249-256.
     */
    class PlasticGTN : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      PlasticGTN(const Core::Mat::PAR::Parameter::Data& matdata);
      //! @name material parameters
      //@{
      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! Density
      const double density_;
      //! tolerance for local Newton iteration
      const double tol_;
      //! maximum number of local iteration
      const int itermax_;
      //! initial yield stress (constant)
      const double yield_;
      //! linear isotropic hardening modulus
      const double isohard_;
      //! function ID for evaluation of hardening law
      const int functionID_hardening_;
      //! damage threshold
      const double fc_;
      //! slope of damage function when damage exceeds threshold
      const double kappa_;
      //! stabilization parameter for damage function. This parameter will provide a round-off at
      //! the corner of fstar function, which improves continuity and convergence.
      const double ef_;
      //! initial damage
      const double f0_;
      //! parameter for damage nucleation
      const double fn_;
      //! parameter for damage nucleation
      const double sn_;
      //! parameter for damage nucleation
      const double en_;
      //! parameter of GTN
      const double k1_;
      //! parameter of GTN
      const double k2_;
      //! parameter of GTN
      const double k3_;
      //@}
      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };
  }  // namespace PAR

  class PlasticGTNType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "PlasticGTNType"; }
    static PlasticGTNType& instance() { return instance_; };
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static PlasticGTNType instance_;
  };

  class PlasticGTN : public So3Material
  {
   public:
    PlasticGTN();

    explicit PlasticGTN(Mat::PAR::PlasticGTN* params);

    int unique_par_object_id() const override
    {
      return PlasticGTNType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;
    void unpack(Core::Communication::UnpackBuffer& buffer) override;
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_plgtn;
    }

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::linear)
        FOUR_C_THROW(
            "The plastic GTN material model is only working with linear kinematics. A nonlinear "
            "kinematics version is planned.");
    }

    std::shared_ptr<Material> clone() const override { return std::make_shared<PlasticGTN>(*this); }

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;
    void update() override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }
    double density() const override { return params_->density_; }

    bool initialized() const { return (isinit_ and !elastic_strain_n_.empty()); }

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    Mat::PAR::PlasticGTN* params_;
    // declare the stress and strain variables.
    std::vector<Core::LinAlg::Matrix<3, 3>> elastic_strain_n_;
    std::vector<Core::LinAlg::Matrix<3, 3>> elastic_strain_n1_;
    std::vector<Core::LinAlg::Matrix<3, 3>> strain_n_;
    std::vector<Core::LinAlg::Matrix<3, 3>> strain_n1_;
    std::vector<Core::LinAlg::Matrix<3, 3>> stress_n_;
    std::vector<Core::LinAlg::Matrix<3, 3>> stress_n1_;
    std::vector<double> f_n_;
    std::vector<double> f_n1_;
    std::vector<double> epbar_n_;
    std::vector<double> epbar_n1_;
    const Core::Utils::FunctionOfAnything* hardening_function_{nullptr};
    bool isinit_;

    /**************************************************
                PLASTIC CALCULATION ROUTINES
     **************************************************/

    /*!
     * @brief Compute the local system
     *
     * @param[out] h the residual vector of the local system
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] p_trial trial hydrostatic pressure
     * @param[in] q_trial trial deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] f current void (damage)
     * @param[in] f_n previous void (damage)
     * @param[in] dlambda current plastic multiplier
     * @param[in] epbar_n previous accumulated plastic strain
     */
    void compute_local_system(Core::LinAlg::Matrix<5, 1>& h, const double p, const double q,
        const double p_trial, const double q_trial, const double sigmastar, const double f,
        const double f_n, const double dlambda, const double epbar_n) const;

    /*!
     * @brief Compute the Jacobian of the local system
     *
     * @param[out] jac Jacobian of the local system
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] f current void (damage)
     * @param[in] dlambda current plastic multiplier
     * @param[in] epbar_n previous accumulated plastic strain
     */
    void compute_jacobian(Core::LinAlg::Matrix<5, 5>& jac, const double p, const double q,
        const double sigmastar, const double f, const double dlambda, const double epbar_n) const;

    /*!
     * @brief Returns the value of yield function
     *
     * @param[in] alpha   The value of accumulated plastic strain
     */
    double yield_value(const double alpha) const;

    /*!
     * @brief Returns the derivative of yield function
     *
     * @param[in] alpha   The value of accumulated plastic strain
     */
    double yield_derivative(const double alpha) const;

    /*!
     * @brief Returns the modified damage variable
     *
     * @param[in] f  The value of real damage
     */
    double compute_fstar(const double f) const;

    /*!
     * @brief Returns the derivative modified damage variable w.r.t damage
     *
     * @param[in] f  The value of real damage
     */
    double compute_dfstar_df(const double f) const;

    /*!
     * @brief Returns the second derivative modified damage variable w.r.t damage
     *
     * @param[in] f  The value of real damage
     */
    double compute_d2fstar_df2(const double f) const;

    /*!
     * @brief Returns the value of damage nucleation function
     *
     * @param[in] alpha   The value of accumulated plastic strain
     */
    double compute_damage_nucleation(const double alpha) const;

    /*!
     * @brief Returns the derivative of damage nucleation function
     *
     * @param[in] alpha   The value of accumulated plastic strain
     */
    double compute_damage_nucleation_derivative(const double alpha) const;

    /*!
     * @brief Returns the value of yield function
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_phi(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial derivative of yield function w.r.t hydrostatic pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_dphi_dp(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial derivative of yield function w.r.t deviatoric pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_dphi_dq(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial derivative of yield function w.r.t void (damage)
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_dphi_df(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial derivative of yield function w.r.t effective scalar stress
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_dphi_dsigmastar(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t effective scalar stress
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dsigmastar2(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t effective scalar stress
     * and hydrostatic pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dsigmastar_dp(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t effective scalar stress
     * and deviatoric pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dsigmastar_dq(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t effective scalar stress
     * and damage
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dsigmastar_df(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t hydrostatic pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dp2(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t hydrostatic and
     * deviatoric pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dp_dq(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t hydrostatic pressure and
     * damage
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dp_df(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t deviatoric pressure
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dq2(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t deviatoric pressure and
     * damage
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_dq_df(
        const double p, const double q, const double sigmastar, const double fstar) const;

    /*!
     * @brief Returns the partial second derivative of yield function w.r.t damage
     *
     * @param[in] p current hydrostatic pressure
     * @param[in] q current deviatoric pressure
     * @param[in] sigmastar current effective scalar stress
     * @param[in] fstar current effective void (damage)
     */
    double compute_d2phi_df2(
        const double p, const double q, const double sigmastar, const double fstar) const;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
