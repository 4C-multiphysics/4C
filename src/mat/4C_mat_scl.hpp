// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCL_HPP
#define FOUR_C_MAT_SCL_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_elchsinglemat.hpp"

namespace FourC::Core::Utils
{
  class FunctionOfScalar;
}
FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for electrolytes including space-charge-layer formation
    class Scl : public ElchSingleMat
    {
     public:
      explicit Scl(const Core::Mat::PAR::Parameter::Data& matdata);

      //! gets the function defining the concentration scaling of the transference number from the
      //! global problem for the first call and returns it for all later calls
      [[nodiscard]] const Core::Utils::FunctionOfScalar&
      transference_number_concentration_scaling_funct();

      [[nodiscard]] bool has_transference_number_concentration_scaling() const
      {
        return transference_number_concentration_scaling_funct_num_.has_value();
      }

      /// @name material parameters
      ///@{
      /// valence (= charge number)
      const double valence_;

      //! transference number
      const double transference_number_;

      //! function number defining the optional concentration scaling of the transference number
      const std::optional<int> transference_number_concentration_scaling_funct_num_;
      //! function defining the optional concentration scaling of the transference number
      std::optional<std::reference_wrapper<const Core::Utils::FunctionOfScalar>>
          transference_number_concentration_scaling_funct_{std::nullopt};

      //! maximum concentration of species
      const double cmax_;

      //! strategy for extrapolation of diffusion coefficient
      const int extrapolation_diffusion_coeff_strategy_;

      //! limit concentration for extrapolation strategy
      const double clim_;

      //! bulk concentration i.e. anion concentration for equal transference numbers
      const double cbulk_;

      //! dielectric susceptibility of electrolyte material
      const double susceptibility_;

      //! difference in partial molar volumes (vacancy <=> interstitial)
      const double delta_nu_;

      //! Faraday constant
      const double faraday_;

      //! universal gas constant
      const double R_;

      //! vacuum Permittivity
      const double epsilon_0_;
      ///@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;
    };

  }  // namespace PAR

  class SclType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "SclType"; }

    static SclType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static SclType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for the material properties of an ion species in an electrolyte solution
  class Scl : public ElchSingleMat
  {
   public:
    Scl() = default;
    /// construct the material object given material parameters
    explicit Scl(Mat::PAR::Scl* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return SclType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_scl;
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Scl>(*this);
    }

    /// valence (= charge number)
    [[nodiscard]] double valence() const { return params_->valence_; }

    /// computation of the transference number
    [[nodiscard]] double compute_transference_number(double concentration) const;
    /// computation of the first derivative of the transference number w.r.t. concentration
    [[nodiscard]] double compute_concentration_derivative_of_transference_number(
        double concentration) const;

    [[nodiscard]] double compute_diffusion_coefficient(
        double concentration, double temperature) const override;

    [[nodiscard]] double compute_concentration_derivative_of_diffusion_coefficient(
        double concentration, double temperature) const override;

    /// computation of dielectric susceptibility (currently a constant)
    [[nodiscard]] double compute_susceptibility() const { return params_->susceptibility_; }

    /// computation of 1/(z^2F^2) with valence of cations
    [[nodiscard]] double inv_val_valence_faraday_squared() const;

    /// Computation of dielectric permittivity based on dielectric susceptibility
    [[nodiscard]] double compute_permittivity() const;

    /// Returns Value of cation concentration in the neutral bulk (= anion concentration)
    [[nodiscard]] double bulk_concentration() const { return params_->cbulk_; }

    /// computation of mobility factor in linear onsager ansatz
    [[nodiscard]] double compute_onsager_coefficient(
        double concentration, double temperature) const;

    /// computation of the derivative of the mobility factor w.r.t to cation concentration
    [[nodiscard]] double compute_concentration_derivative_of_onsager_coefficient(
        double concentration, double temperature) const;

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::Scl* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
