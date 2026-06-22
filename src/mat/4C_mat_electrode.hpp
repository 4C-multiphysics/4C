// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELECTRODE_HPP
#define FOUR_C_MAT_ELECTRODE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_elchsinglemat.hpp"
#include "4C_utils_function_of_scalar.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    //! models for open circuit potential of electrode
    enum class OCPModels
    {
      function,
      redlichkister,
      taralov
    };

    //! struct storing the minimum and maximum lithiation values for which the prescribed open
    //! circuit potential calculation model is valid
    struct LithiationBounds
    {
      double x_min;
      double x_max;
    };

    //! parameters for electrode material
    class Electrode : public ElchSingleMat
    {
     public:
      //! constructor
      explicit Electrode(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create an instance of electrode material
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! gets the ocp function from the global problem for the first call and returns it for all
      //! later calls
      [[nodiscard]] const Core::Utils::FunctionOfScalar& ocp_function();

      //! returns the lower bound of the range of validity for the ocp calculation model
      //! throws if no lithiation bounds have been set in the input
      [[nodiscard]] double x_min() const;

      //! returns the upper bound of the range of validity for the ocp calculation model
      //! throws if no lithiation bounds have been set in the input
      [[nodiscard]] double x_max() const;

      //! @name parameters for electrode material
      //! @{
      //! saturation value of intercalated Lithium concentration
      const double cmax_;

      //! lithiation value corresponding to saturation value of intercalated Lithium concentration
      //! #cmax_
      const double chimax_;

      //! model for half cell open circuit potential
      const OCPModels ocpmodel_;

      //! number of the function defining the open circuit potential
      int ocp_function_num_;

      //! function defining the open circuit potential
      std::optional<std::reference_wrapper<const Core::Utils::FunctionOfScalar>> ocp_function_{
          std::nullopt};

      //! parameters underlying half cell open circuit potential model
      std::vector<double> ocppara_;

      //! lithiation bounds for which the prescribed open circuit potential calculation model is
      //! valid
      std::optional<LithiationBounds> lithiation_bounds_;
      //! @}
    };  // class Mat::PAR::Electrode
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  class ElectrodeType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "ElectrodeType"; }

    static ElectrodeType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ElectrodeType instance_;
  };


  /*----------------------------------------------------------------------*/
  //! wrapper for electrode material
  class Electrode : public ElchSingleMat
  {
   public:
    //! construct empty electrode material
    Electrode() = default;

    //! construct electrode material with specific material parameters
    explicit Electrode(Mat::PAR::Electrode* params);

    //! @name packing and unpacking
    ///@{
    [[nodiscard]] int unique_par_object_id() const override
    {
      return ElectrodeType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;
    ///@}

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_electrode;
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<Electrode>(*this);
    }

    //! return lithiation value corresponding to the saturation value of intercalated Lithium
    //! concentration
    [[nodiscard]] double chi_max() const { return params_->chimax_; }

    //! return saturation value of intercalated Lithium concentration
    [[nodiscard]] double c_max() const { return params_->cmax_; }

    /*!
     * @brief compute the current lithiation
     *
     * @param[in] concentration  concentration
     * @param[in] chi_max        lithiation value of saturation value of intercalated lithium
     *                           concentration
     * @param[in] c_max          saturation value of intercalated lithium concentration
     * @param[in] detjacobian    determinant of the deformation gradient
     * @return the current lithiation
     */
    static double compute_intercalation_fraction(const double concentration, const double chi_max,
        const double c_max, const double detjacobian)
    {
      return (concentration * chi_max * detjacobian) / c_max;
    }

    /*!
     * @brief compute the derivative of the lithiation w.r.t. the concentration
     *
     * @param[in] chi_max      lithiation value of saturation value of intercalated lithium
     *                         concentration
     * @param[in] c_max        saturation value of intercalated lithium concentration
     * @param[in] detjacobian  determinant of the deformation gradient
     * @return the derivative of current lithiation w.r.t. concentration
     */
    static double compute_d_intercalation_fraction_d_concentration(
        const double chi_max, const double c_max, const double detjacobian)
    {
      return chi_max * detjacobian / c_max;
    }

    /*!
     * @brief compute the derivative of the lithiation w.r.t. the determinant of the deformation
     * gradient
     *
     * @param[in] concentration  concentration
     * @param[in] chi_max        lithiation value of saturation value of intercalated lithium
     *                           concentration
     * @param[in] c_max          saturation value of intercalated lithium concentration
     * @return  the derivative of the lithiation w.r.t. the determinant of the deformation gradient
     */
    static double compute_d_intercalation_fraction_d_det_f(
        const double concentration, const double chi_max, const double c_max)
    {
      return concentration * chi_max / c_max;
    }

    /*!
     * @brief compute the first derivative of the open circuit potential w.r.t. concentration
     *
     * @param[in] concentration  concentration
     * @param[in] faraday        Faraday constant
     * @param[in] frt            factor F/RT
     * @param[in] detF           determinant of Jacobian from deformation at Gauss point
     * @return derivative of open circuit potential w.r.t. concentration
     */
    [[nodiscard]] double compute_d_open_circuit_potential_d_concentration(
        double concentration, double faraday, double frt, double detF) const;

    /*!
     * @brief calculate the first derivative of the open circuit potential w.r.t. the determinant of
     * the deformation gradient
     *
     * @param[in] concentration  concentration
     * @param[in] faraday        Faraday constant
     * @param[in] frt            factor F/RT
     * @param[in] detF           determinant of Jacobian from deformation at Gauss point
     * @return derivative of open circuit potential w.r.t. the determinant of the deformation
     * gradient
     */
    [[nodiscard]] double compute_d_open_circuit_potential_d_det_f(
        double concentration, double faraday, double frt, double detF) const;

    /*!
     * @brief calculate the first derivative of the open circuit potential w.r.t. the intercalation
     * fraction
     *
     * @param[in] X         intercalation fraction
     * @param[in] faraday   Faraday constant
     * @param[in] frt       factor F/RT
     * @return derivative of the open circuit potential w.r.t. the intercalation fraction
     */
    [[nodiscard]] double compute_d_open_circuit_potential_d_intercalation_fraction(
        double X, double faraday, double frt) const;

    /*!
     * @brief compute the first derivative of the open circuit potential w.r.t. temperature
     *
     * @param[in] concentration  concentration
     * @param[in] faraday        Faraday constant
     * @param[in] gasconstant    General gas constant
     * @return derivative of open circuit potential w.r.t. temperature
     */
    [[nodiscard]] double compute_d_open_circuit_potential_d_temperature(
        const double concentration, const double faraday, const double gasconstant) const;

    /*!
     * @brief compute the open circuit potential
     *
     * @param[in] concentration  concentration
     * @param[in] faraday        Faraday constant
     * @param[in] frt            factor F/RT
     * @param[in] detF           determinant of jacobian at gauss point
     * @return open circuit potential
     */
    [[nodiscard]] double compute_open_circuit_potential(
        double concentration, double faraday, double frt, double detF) const;

    /*!
     * @brief compute the second derivative of the open circuit potential w.r.t. concentration
     *
     * @param[in] concentration  concentration
     * @param[in] faraday        Faraday constant
     * @param[in] frt            factor F/RT
     * @param[in] detF           determinant of jacobian at gauss point
     * @return 2nd derivative of open circuit potential w.r.t. concentration
     */
    [[nodiscard]] double compute_d2_open_circuit_potential_d_concentration_d_concentration(
        double concentration, double faraday, double frt, double detF) const;

    //! return material parameters
    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    //! my material parameters
    Mat::PAR::Electrode* params_;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
