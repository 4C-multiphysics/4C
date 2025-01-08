// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELCHSINGLEMAT_HPP
#define FOUR_C_MAT_ELCHSINGLEMAT_HPP

#include "4C_config.hpp"

#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! parameters for abstract battery material
    class ElchSingleMat : public Core::Mat::PAR::Parameter
    {
     public:
      //! @name parameters for abstract battery material
      //@{
      //! function number to describe concentration dependence of diffusion coefficient
      const int diffusion_coefficient_concentration_dependence_funct_num_;

      //! function number defining the temperature scaling of the diffusion coefficient
      const int diffusion_coefficient_temperature_scaling_funct_num_;

      //! number of parameters for diffusion coefficient
      const int number_diffusion_coefficent_params_;

      //! parameters for diffusion coefficient
      const std::vector<double> diffusion_coefficent_params_;

      //! number of parameters for scaling function describing temperature dependence of diffusion
      //! coefficient
      const int number_diffusion_temp_scale_funct_params_;

      //! parameters for scaling function describing temperature dependence of diffusion coefficient
      const std::vector<double> diffusion_temp_scale_funct_params_;

      //! function number to describe concentration dependence of conductivity
      const int conductivity_concentration_dependence_funct_num_;

      //! function number defining the temperature scaling of conductivity
      const int conductivity_temperature_scaling_funct_num_;

      //! number of parameters for conductivity
      const int number_conductivity_params_;

      //! parameters for conductivity
      const std::vector<double> conductivity_params_;

      //! number of parameters for scaling function describing temperature dependence of
      //! conductivity
      const int number_conductivity_temp_scale_funct_params_;

      //! parameters for scaling function describing temperature dependence conductivity
      const std::vector<double> conductivity_temp_scale_funct_params_;

      //! universal gas constant for evaluation of diffusion coefficient by means of
      //! Arrhenius-ansatz
      const double R_;
      //@}

     protected:
      //! constructor
      explicit ElchSingleMat(const Core::Mat::PAR::Parameter::Data& matdata);

      //! check whether number of parameters is consistent with curve number
      void check_provided_params(int functnr, const std::vector<double>& functparams);
    };  // class Mat::PAR::ElchSingleMat
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  //! wrapper for abstract battery material
  class ElchSingleMat : public Core::Mat::Material
  {
   public:
    //! @name packing and unpacking
    /*!
      \brief Return unique ParObject id

      Every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override = 0;

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique ParObject ID delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override = 0;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      unique_par_object_id().

      \param data (in) : vector storing all data to be unpacked into this instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override = 0;
    //@}

    //! compute diffusion coefficient accounting for concentration and temperature dependence
    virtual double compute_diffusion_coefficient(double concentration, double temperature) const;

    //! compute concentration dependent diffusion coefficient according to function number
    double compute_diffusion_coefficient_concentration_dependent(double concentration) const;

    //! compute first derivative of diffusion coefficient w.r.t. concentration
    virtual double compute_concentration_derivative_of_diffusion_coefficient(
        double concentration, double temperature) const;

    //! compute first derivative of diffusion coefficient w.r.t. temperature
    double compute_temperature_derivative_of_diffusion_coefficient(
        double concentration, double temperature) const;

    //! compute conductivity accounting for concentration and temperature dependence
    double compute_conductivity(double concentration, double temperature) const;

    //! compute concentration dependent conductivity according to function number
    double compute_conductivity_concentration_dependent(double concentration) const;

    //! compute first derivative of conductivity w.r.t. concentration
    double compute_concentration_derivative_of_conductivity(
        double concentration, double temperature) const;

    //! compute first derivative of conductivity w.r.t. temperature
    double compute_temperature_derivative_of_conductivity(
        double concentration, double temperature) const;

    //! abbreviations for pre-defined functions
    //@{
    static constexpr int CONSTANT_FUNCTION = -1;
    static constexpr int LINEAR_FUNCTION = -2;
    static constexpr int QUADRATIC_FUNCTION = -3;
    static constexpr int POWER_FUNCTION = -4;
    static constexpr int CONDUCT = -5;
    static constexpr int MOD_CUBIC_FUNCTION = -6;
    static constexpr int CUBIC_FUNCTION = -7;
    static constexpr int NYMAN = -8;
    static constexpr int DEBYE_HUECKEL = -9;
    static constexpr int KOHLRAUSCH_SQUAREROOT = -10;
    static constexpr int GOLDIN = -11;
    static constexpr int STEWART_NEWMAN = -12;
    static constexpr int TDF = -13;
    static constexpr int ARRHENIUS = -14;
    static constexpr int INVERSE_LINEAR = -15;
    //@}

   protected:
    //! compute temperature dependent scale factor
    double compute_temperature_dependent_scale_factor(
        double temperature, int functionNumber, const std::vector<double>& functionParams) const;

    //! compute derivative of temperature dependent scale factor w.r.t. temperature
    double compute_temperature_dependent_scale_factor_deriv(
        double temperature, int functionNumber, const std::vector<double>& functionParams) const;

    //! return function number describing concentration dependence of the diffusion coefficient
    int diffusion_coefficient_concentration_dependence_funct_num() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())
          ->diffusion_coefficient_concentration_dependence_funct_num_;
    };

    //! return the function number describing the temperature scaling of the diffusion coefficient
    int diffusion_coefficient_temperature_scaling_funct_num() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())
          ->diffusion_coefficient_temperature_scaling_funct_num_;
    };

    //! return function number describing concentration dependence of the conductivity
    int conductivity_concentration_dependence_funct_num() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())
          ->conductivity_concentration_dependence_funct_num_;
    };

    //! return the function number describing the temperature scaling of the conductivity
    int conductivity_temperature_scaling_funct_num() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())
          ->conductivity_temperature_scaling_funct_num_;
    };

    //! return parameters for diffusion coefficient
    const std::vector<double>& diffusion_coefficient_params() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())->diffusion_coefficent_params_;
    };

    //! return parameters for temperature scaling function for diffusion coefficient
    const std::vector<double>& temp_scale_function_params_diff() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())
          ->diffusion_temp_scale_funct_params_;
    };

    //! return parameters for conductivity
    const std::vector<double>& conductivity_params() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())->conductivity_params_;
    };

    //! return parameters for temperature scaling function for conductivity
    const std::vector<double>& temp_scale_function_params_cond() const
    {
      return dynamic_cast<Mat::PAR::ElchSingleMat*>(parameter())
          ->conductivity_temp_scale_funct_params_;
    };

    //! evaluate value as predefined function of any scalar (e.g. concentration, temperature)
    //!
    //! \param functnr      negative function number to be evaluated
    //! \param scalar       scalar value to insert into function
    //! \param functparams  constants that define the functions
    //! \return             function evaluated at value of scalar
    double eval_pre_defined_funct(
        int functnr, double scalar, const std::vector<double>& functparams) const;

    //! evaluate first derivative of predefined function of any scalar (e.g. concentration,
    //! temperature)
    double eval_first_deriv_pre_defined_funct(
        int functnr, double scalar, const std::vector<double>& functparams) const;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
