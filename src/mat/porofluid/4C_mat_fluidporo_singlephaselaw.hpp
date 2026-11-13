// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_SINGLEPHASELAW_HPP
#define FOUR_C_MAT_FLUIDPORO_SINGLEPHASELAW_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Utils
  {
    class FunctionOfAnything;
  }  // namespace Utils
}  // namespace Discret

namespace Mat
{
  namespace PAR
  {
    //! interface class for generic phase (pressure-saturation) law
    class FluidPoroPhaseLaw : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      explicit FluidPoroPhaseLaw(const Core::Mat::PAR::Parameter::Data& matdata);

      //! build the phase law
      static FluidPoroPhaseLaw* create_phase_law(int phaselawId);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      /// initialize
      virtual void initialize() = 0;

      /// pressure ids involved in phase law
      virtual const std::vector<int>* pres_ids() = 0;

      //! evaluate saturation of phase
      virtual double evaluate_saturation(const std::span<const double> pressure) = 0;

      //! evaluate derivative of saturation w.r.t. pressure
      virtual double evaluate_deriv_of_saturation_wrt_pressure(
          int doftoderive, const std::span<const double> pressure) = 0;

      //! evaluate second derivative of saturation w.r.t. pressure
      virtual double evaluate_second_deriv_of_saturation_wrt_pressure(
          int firstdoftoderive, int seconddoftoderive, const std::vector<double>& pressure) = 0;

      //! evaluate generalized pressure related to phase law
      virtual double evaluate_gen_pressure(double saturation) = 0;

      //! evaluate derivative of pressure w.r.t. saturation
      virtual double evaluate_deriv_of_pressure_wrt_saturation(
          int doftoderive, double saturation) = 0;
    };

    /*----------------------------------------------------------------------*/
    //! a linear pressure-saturation relation. Only for testing, no physical meaning.
    class FluidPoroPhaseLawLinear : public FluidPoroPhaseLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroPhaseLawLinear(const Core::Mat::PAR::Parameter::Data& matdata);

      /// initialize
      void initialize() override { return; };

      /// @name material parameters
      //@{
      //! number of degrees of freedom
      const int numdof_;
      //! coefficients defining the pressures involved int the pressure-saturation law
      const std::vector<int> presids_;
      //! 'relative tension' parameter (proportionality coefficient)
      const double reltensions_;
      //! saturation value at zero pressure
      const double sat0_;
      //@}

      /// pressure ids involved in phase law
      const std::vector<int>* pres_ids() override { return &presids_; };

      //! evaluate saturation of phase
      double evaluate_saturation(const std::span<const double> pressure) override;

      //! evaluate generalized pressure related to phase law
      double evaluate_gen_pressure(double saturation) override;

      //! evaluate derivative of saturation w.r.t. pressure
      double evaluate_deriv_of_saturation_wrt_pressure(
          int doftoderive, const std::span<const double> pressure) override;

      //! evaluate second derivative of saturation w.r.t. pressure
      double evaluate_second_deriv_of_saturation_wrt_pressure(int firstdoftoderive,
          int seconddoftoderive, const std::vector<double>& pressure) override;

      //! evaluate derivative of pressure w.r.t. saturtion
      double evaluate_deriv_of_pressure_wrt_saturation(int doftoderive, double saturation) override;
    };
    // class FluidPoroPhaseLawLinear

    /*----------------------------------------------------------------------*/
    //! tangent pressure-saturation relationship
    class FluidPoroPhaseLawTangent : public FluidPoroPhaseLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroPhaseLawTangent(const Core::Mat::PAR::Parameter::Data& matdata);

      /// initialize
      void initialize() override { return; };

      /// @name material parameters
      //@{
      //! number of degrees of freedom
      const int numdof_;
      //! coefficients defining the pressures involved int the pressure-saturation law
      const std::vector<int> presids_;
      //! relative tension coefficient
      const double reltensions_;
      //! exponent within phase law
      const double exp_;
      //! saturation value at zero pressure
      const double sat0_;
      //@}

      /// pressure ids involved in phase law
      const std::vector<int>* pres_ids() override { return &presids_; };

      //! evaluate saturation of phase
      double evaluate_saturation(const std::span<const double> pressure) override;

      //! evaluate derivative of saturation w.r.t. pressure
      double evaluate_deriv_of_saturation_wrt_pressure(
          int doftoderive, const std::span<const double> pressure) override;

      //! evaluate second derivative of saturation w.r.t. pressure
      double evaluate_second_deriv_of_saturation_wrt_pressure(int firstdoftoderive,
          int seconddoftoderive, const std::vector<double>& pressure) override;

      //! evaluate generalized pressure related to phase law
      double evaluate_gen_pressure(double saturation) override;

      //! evaluate derivative of pressure w.r.t. saturation
      double evaluate_deriv_of_pressure_wrt_saturation(int doftoderive, double saturation) override;
    };
    // class FluidPoroPhaseLawTangent

    /*----------------------------------------------------------------------*/
    //! a phase law indicating that the saturation is calculated from
    //! saturation constraint (i.e. the sum of all saturations is equal to one)
    class FluidPoroPhaseLawConstraint : public FluidPoroPhaseLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroPhaseLawConstraint(const Core::Mat::PAR::Parameter::Data& matdata)
          : FluidPoroPhaseLaw(matdata) {};

      /// initialize
      void initialize() override { return; };

      /// pressure ids involved in phase law
      const std::vector<int>* pres_ids() override
      {
        FOUR_C_THROW(
            "The constraint phase law does not have pressure coupling! \n "
            "Combining Saturation DOF and constraint phase law is invalid!");
        return nullptr;
      };


      //! evaluate saturation of phase
      double evaluate_saturation(const std::span<const double> pressure) override
      {
        FOUR_C_THROW("The constraint phase law does not implement evaluation routines!");
        return 0.0;
      };

      //! evaluate derivative of saturation w.r.t. pressure
      double evaluate_deriv_of_saturation_wrt_pressure(
          int doftoderive, const std::span<const double> pressure) override
      {
        FOUR_C_THROW("The constraint phase law does not implement evaluation routines!");
        return 0.0;
      };

      //! evaluate second derivative of saturation w.r.t. pressure
      double evaluate_second_deriv_of_saturation_wrt_pressure(
          int firstdoftoderive, int seconddoftoderive, const std::vector<double>& pressure) override
      {
        FOUR_C_THROW("The constraint phase law does not implement evaluation routines!");
        return 0.0;
      };

      //! evaluate generalized pressure related to phase law
      double evaluate_gen_pressure(double saturation) override
      {
        FOUR_C_THROW("The constraint phase law does not implement evaluation routines!");
        return 0.0;
      };

      //! evaluate derivative of pressure w.r.t. saturation
      double evaluate_deriv_of_pressure_wrt_saturation(int doftoderive, double saturation) override
      {
        FOUR_C_THROW("The constraint phase law does not implement evaluation routines!");
        return 0.0;
      };
    };
    // class FluidPoroPhaseLawTangent

    /*----------------------------------------------------------------------*/
    //! pressure-saturation relationship defined by function
    class FluidPoroPhaseLawByFunction : public FluidPoroPhaseLaw
    {
     public:
      /// standard constructor
      explicit FluidPoroPhaseLawByFunction(const Core::Mat::PAR::Parameter::Data& matdata);

      /// initialize
      void initialize() override;

      /// @name material parameters
      //@{
      //! number of degrees of freedom
      const int numdof_;
      //! coefficients defining the pressures involved int the pressure-saturation law
      const std::vector<int> presids_;
      //! function ID for evaluation of saturation
      const int functionID_saturation_;
      //! function ID for evaluation of pressure
      const int functionID_pressure_;
      //@}

      /// pressure ids involved in phase law
      const std::vector<int>* pres_ids() override { return &presids_; };

      //! evaluate saturation of phase
      double evaluate_saturation(const std::span<const double> pressure) override;

      //! evaluate derivative of saturation w.r.t. pressure
      double evaluate_deriv_of_saturation_wrt_pressure(
          int doftoderive, const std::span<const double> pressure) override;

      //! evaluate second derivative of saturation w.r.t. pressure
      double evaluate_second_deriv_of_saturation_wrt_pressure(int firstdoftoderive,
          int seconddoftoderive, const std::vector<double>& pressure) override;

      //! evaluate generalized pressure related to phase law
      double evaluate_gen_pressure(double saturation) override;

      //! evaluate derivative of pressure w.r.t. saturation
      double evaluate_deriv_of_pressure_wrt_saturation(int doftoderive, double saturation) override;

     private:
      //! templated internal Initialize implementation
      template <int dim>
      void initialize_internal();

      //! templated internal EvaluateSaturation implementation
      template <int dim>
      double evaluate_saturation_internal(const std::span<const double> pressure);

      //! templated internal evaluate_deriv_of_saturation_wrt_pressure implementation
      template <int dim>
      double evaluate_deriv_of_saturation_wrt_pressure_internal(
          int doftoderive, const std::span<const double> pressure);

      //! templated internal evaluate_deriv_of_pressure_wrt_saturation implementation
      template <int dim>
      double evaluate_deriv_of_pressure_wrt_saturation_internal(int doftoderive, double saturation);

      //! templated internal EvaluateGenPressure implementation
      template <int dim>
      double evaluate_gen_pressure_internal(double saturation);


      //! vector for input of differential pressure to function
      std::vector<std::pair<std::string, double>> dp_;
      //! vector for input of saturation to function
      std::vector<std::pair<std::string, double>> s_;
    };
    // class FluidPoroPhaseLawTangent

  }  // namespace PAR
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
