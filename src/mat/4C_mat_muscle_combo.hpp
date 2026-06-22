// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MUSCLE_COMBO_HPP
#define FOUR_C_MAT_MUSCLE_COMBO_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_field.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_function.hpp"

#include <memory>
#include <variant>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    class MuscleCombo : public Core::Mat::PAR::Parameter
    {
     public:
      //! valid types for prescription of time-/space-dependent muscle activation
      enum class ActivationType
      {
        function_of_space_time,  ///< analytical activation prescription via a symbolic function of
                                 ///< space and time
        map  ///< discrete elementwise-defined activation prescription via an input json input file
      };

      /// constructor
      MuscleCombo(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{
      /// Passive material parameters
      struct PassiveParams
      {
        double alpha;   ///< material parameter, >0
        double beta;    ///< material parameter, >0
        double gamma;   ///< material parameter, >0
        double kappa;   ///< material parameter for coupled volumetric contribution
        double omega0;  ///< weighting factor for isotropic tissue constituents, governs ratio
                        ///< between muscle matrix material (omega0) and muscle fibers (omegap)
                        ///< with omega0 + omegap = 1
      };

      /// passive material parameters
      const PassiveParams passive_;

      /// Function-based activation prescription
      struct FunctionActivation
      {
        int function_id;
      };

      /// Field-based activation prescription
      struct FieldActivation
      {
        Core::IO::InputField<std::vector<std::pair<double, double>>> field;
      };

      /**
       * @brief Type-dependent activation source
       *
       * Depending on the type of activation prescription this is one of the options below:
       * - Analytical function in the input file prescribing the activation as a function of space
       * and time.
       * - Map retrieved from the json input file path in the input file specifying a discrete
       * values. The integer key refers to the element ids, the vector bundles time-activation
       * pairs.
       */
      using ActivationSource = std::variant<std::monostate, FunctionActivation, FieldActivation>;

      /**
       * @brief Active microstructural parameters
       */
      struct ActiveParams
      {
        double Popt;       ///< optimal (maximal) active tetanised stress
        double lambdaMin;  ///< minimal active fiber stretch
        double
            lambdaOpt;  ///< optimal active fiber stretch related to active nominal stress maximum

        /// activation prescription
        ActivationSource source = std::monostate{};
      };

      /// optional active parameters
      std::optional<ActiveParams> active_;

      const double density_;  ///< density
      //@}

      const Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>
          fiber_orientation_;  ///< fiber orientation field
    };  // end class MuscleCombo
  }  // end namespace PAR


  class MuscleComboType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "Muscle_ComboType"; }

    static MuscleComboType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MuscleComboType instance_;
  };


  /*!
   * \brief Combo muscle material
   *
   * This constituent represents an active hyperelastic muscle material using a generalized active
   * strain approach. Stress and material tangent are consistently derived from the strain-energy
   * function.
   *
   * The general material formulation is equal to Weickenmeier et al. [1] with the following
   * modifications:
   * 1. The derivative of omegaa w.r.t. C is included as described for the active stress
   * approach in Giantesio et al. [2]. This leads to an additional term in the stress and material
   * tangent computation and an updated equation for the activation level omegaa.
   * 2. The twitch superposition is neglected and the time and space dependent optimal nominal
   * stress is computed through a user-prescribed function.
   * 3. A velocity dependence is not considered.
   *
   * References:
   * [1] J. Weickenmeier, M. Itskov, E Mazza and M. Jabareen, 'A
   * physically motivated constitutive model for 3D numerical simulation of skeletal muscles',
   * International journal for numerical methods in biomedical engineering, vol. 30, no. 5, pp.
   * 545-562, 2014, doi: 10.1002/cnm.2618.
   * [2] G. Giantesio, A. Musesti, 'Strain-dependent internal
   * parameters in hyperelastic biological materials', International Journal of Non-Linear
   * Mechanics, vol. 95, pp. 162-167, 2017, doi:10.1016/j.ijnonlinmec.2017.06.012.
   */
  class MuscleCombo : public So3Material
  {
   public:
    // Constructor for empty material object
    MuscleCombo();

    // Constructor for the material given the material parameters
    explicit MuscleCombo(Mat::PAR::MuscleCombo* params);

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MuscleCombo>(*this);
    }

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_muscle_combo;
    };

    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (kinem != Inpar::Solid::KinemType::linear &&
          kinem != Inpar::Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    [[nodiscard]] double density() const override { return params_->density_; }

    [[nodiscard]] int unique_par_object_id() const override
    {
      return MuscleComboType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    bool uses_extended_update() override { return true; };

    void update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, int const gp,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        int const eleGID) override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    using ActivationEvaluator =
        std::variant<std::monostate, const Core::Utils::FunctionOfSpaceTime*,
            const Core::IO::InputField<std::vector<std::pair<double, double>>>*>;

   private:
    /// Combo material parameters
    Mat::PAR::MuscleCombo* params_{};

    /// Activation evaluator, either analytical symbolic function of space and time or discrete
    /// activation map
    ActivationEvaluator activation_evaluator_;

    /// Struct to bundle activation level and its derivatives for stress and material tangent
    /// computation
    struct ActivationTerms
    {
      double omegaa;
      double derivOmegaa;
      double derivDerivOmegaa;
      double eta;
      double dEta;
    };

    /// Evaluate activation level and its derivatives for stress and material tangent computation
    ActivationTerms evaluate_activation_level(const Teuchos::ParameterList& params,
        const EvaluationContext<3>& context, const int eleGID, const double lambdaM);
  };  // end class Muscle_Combo

}  // end namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif