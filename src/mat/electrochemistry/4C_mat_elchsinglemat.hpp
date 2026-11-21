// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELCHSINGLEMAT_HPP
#define FOUR_C_MAT_ELCHSINGLEMAT_HPP

#include "4C_config.hpp"

#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

namespace FourC::Core::Utils
{
  class FunctionOfTime;
}
FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! parameters for abstract battery material
    class ElchSingleMat : public Core::Mat::PAR::Parameter
    {
     public:
      //! constructor
      explicit ElchSingleMat(const Core::Mat::PAR::Parameter::Data& matdata);

      [[nodiscard]] bool has_diffusion_coefficient_concentration_scaling() const
      {
        return diffusion_coefficient_concentration_scaling_funct_num_.has_value();
      }

      [[nodiscard]] bool has_diffusion_coefficient_temperature_scaling() const
      {
        return diffusion_coefficient_temperature_scaling_funct_num_.has_value();
      }

      [[nodiscard]] bool has_conductivity_concentration_scaling() const
      {
        return conductivity_concentration_scaling_funct_num_.has_value();
      }

      [[nodiscard]] bool has_conductivity_temperature_scaling() const
      {
        return conductivity_temperature_scaling_funct_num_.has_value();
      }

      //! gets the function defining the concentration scaling of the diffusion coefficient from the
      //! global problem for the first call and returns it for all later calls
      [[nodiscard]] const Core::Utils::FunctionOfTime&
      diffusion_coefficient_concentration_scaling_funct();

      //! gets the function defining the temperature scaling of the diffusion coefficient from the
      //! global problem for the first call and returns it for all later calls
      [[nodiscard]] const Core::Utils::FunctionOfTime&
      diffusion_coefficient_temperature_scaling_funct();

      //! gets the function defining the concentration scaling of the conductivity from the global
      //! problem for the first call and returns it for all later calls
      [[nodiscard]] const Core::Utils::FunctionOfTime& conductivity_concentration_scaling_funct();

      //! gets the function defining the temperature scaling of the conductivity from the global
      //! problem for the first call and returns it for all later calls
      [[nodiscard]] const Core::Utils::FunctionOfTime& conductivity_temperature_scaling_funct();

      //! @name parameters for abstract battery material
      ///@{
      //! diffusion coefficient
      const double diffusion_coefficient_;

      //! function number defining the optional concentration scaling of the diffusion coefficient
      const std::optional<int> diffusion_coefficient_concentration_scaling_funct_num_;
      //! function defining the optional concentration scaling of the diffusion coefficient
      std::optional<std::reference_wrapper<const Core::Utils::FunctionOfTime>>
          diffusion_coefficient_concentration_scaling_funct_{std::nullopt};

      //! function number defining the optional temperature scaling of the diffusion coefficient
      const std::optional<int> diffusion_coefficient_temperature_scaling_funct_num_;
      //! function defining the optional temperature scaling of the diffusion coefficient
      std::optional<std::reference_wrapper<const Core::Utils::FunctionOfTime>>
          diffusion_coefficient_temperature_scaling_funct_{std::nullopt};

      //! conductivity
      const double conductivity_;

      //! function number defining the optional concentration scaling of the conductivity
      const std::optional<int> conductivity_concentration_scaling_funct_num_;
      //! function defining the optional concentration scaling of the conductivity
      std::optional<std::reference_wrapper<const Core::Utils::FunctionOfTime>>
          conductivity_concentration_scaling_funct_{std::nullopt};

      //! function number defining the optional temperature scaling of conductivity
      const std::optional<int> conductivity_temperature_scaling_funct_num_;
      //! function defining the optional temperature scaling of the conductivity
      std::optional<std::reference_wrapper<const Core::Utils::FunctionOfTime>>
          conductivity_temperature_scaling_funct_{std::nullopt};
      ///@}
    };  // class Mat::PAR::ElchSingleMat
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  //! wrapper for abstract battery material
  class ElchSingleMat : public Core::Mat::Material
  {
   public:
    ElchSingleMat() = default;

    //! construct electrode material with specific material parameters
    explicit ElchSingleMat(Mat::PAR::ElchSingleMat* params);

    //! @name packing and unpacking
    ///@{
    [[nodiscard]] int unique_par_object_id() const override = 0;

    void pack(Core::Communication::PackBuffer& data) const override = 0;

    void unpack(Core::Communication::UnpackBuffer& buffer) override = 0;
    ///@}

    //! compute diffusion coefficient accounting for concentration and temperature dependence
    [[nodiscard]] virtual double compute_diffusion_coefficient(
        double concentration, double temperature) const;

    //! compute diffusion coefficient accounting for concentration dependence
    [[nodiscard]] double compute_diffusion_coefficient_concentration_dependent(
        double concentration) const;

    //! compute first derivative of diffusion coefficient w.r.t. concentration
    [[nodiscard]] virtual double compute_concentration_derivative_of_diffusion_coefficient(
        double concentration, double temperature) const;

    //! compute first derivative of diffusion coefficient w.r.t. temperature
    [[nodiscard]] double compute_temperature_derivative_of_diffusion_coefficient(
        double concentration, double temperature) const;

    //! compute conductivity accounting for concentration and temperature dependence
    [[nodiscard]] double compute_conductivity(double concentration, double temperature) const;

    //! compute the first derivative of conductivity w.r.t. concentration
    [[nodiscard]] double compute_concentration_derivative_of_conductivity(
        double concentration, double temperature) const;

    //! compute the first derivative of conductivity w.r.t. temperature
    [[nodiscard]] double compute_temperature_derivative_of_conductivity(
        double concentration, double temperature) const;

   protected:
    //! synchronize base-class parameter pointer after (de-)serialization in derived materials
    void set_elch_single_mat_params(Mat::PAR::ElchSingleMat* params) { params_ = params; }

   private:
    //! my material parameters
    Mat::PAR::ElchSingleMat* params_{nullptr};
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
