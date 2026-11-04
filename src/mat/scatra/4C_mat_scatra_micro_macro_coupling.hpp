// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_MICRO_MACRO_COUPLING_HPP
#define FOUR_C_MAT_SCATRA_MICRO_MACRO_COUPLING_HPP

#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"

#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  // forward declaration
  class ScatraMultiScaleGP;

  namespace PAR
  {
    //! material parameters
    class ScatraMicroMacroCoupling
    {
     public:
      //! constructor
      ScatraMicroMacroCoupling(const Core::Mat::PAR::Parameter::Data& matdata);

      //! return name of micro-scale input file
      std::string micro_input_file_name() const { return microfile_; }

      //! return number of micro-scale discretization
      int micro_dis_num() const { return microdisnum_; };

      //! return specific micro-scale surface area A_s
      double specific_micro_scale_surface_area() const { return A_s_; }

     protected:
      //! @name material parameters
      //@{
      //! name of micro-scale input file
      const std::string microfile_;

      //! number of micro-scale discretization
      const int microdisnum_;

      //! specific micro-scale surface area
      const double A_s_;
      //@}
    };  // class Mat::PAR::ScatraMicroMacroCoupling
  }  // namespace PAR

  /*----------------------------------------------------------------------*/
  //! material wrapper
  class ScatraMicroMacroCoupling
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ScatraMicroMacroCoupling() = default;

    //! initialize multi-scale scalar transport material
    //! \param[in] ele_id         macro-scale element ID
    //! \param[in] gp_id          macro-scale Gauss point ID
    //! \param[in] is_ale         true, if the underlying macro dis deforms
    void initialize(const int ele_id, const int gp_id, bool is_ale);

    /*!
     * @brief prepare time step on micro scale
     *
     * @param[in] gp_id        macro-scale Gauss point ID
     * @param[in] phinp_macro  macro-scale state variables
     */
    void prepare_time_step(const int gp_id, const std::vector<double>& phinp_macro) const;

    /*!
     * @brief collect the output data on micro scale
     *
     * @param[in] gp_id  macro-scale Gauss point ID
     */
    void collect_output_data(const int gp_id) const;

    /*!
     * @brief evaluate multi-scale scalar transport material
     *
     * @param[in] gp_id        macro-scale Gauss point ID
     * @param[in] phinp_macro  macro-scale state variables
     * @param[out] q_micro        micro-scale flux
     * @param[out] dq_dphi_micro  derivatives of micro-scale flux w.r.t. macro-scale state variables
     * @param[in] detF       determinant of deformation gradient of macro dis at current Gauss point
     * @param[in] solve      flag indicating whether micro-scale problem should be solved
     */
    void evaluate(const int gp_id, const std::vector<double>& phinp_macro, double& q_micro,
        std::vector<double>& dq_dphi_micro, double detF, const bool solve = true) const;

    /*!
     * @brief evaluate mean concentration on micro scale
     *
     * @param[in] gp_id macro-scale Gauss point ID
     * @return mean concentration
     */
    double evaluate_mean_concentration(const int gp_id) const;

    /*!
     * @brief evaluate mean concentration time derivative on micro scale
     *
     * @param[in] gp_id  macro-scale Gauss point ID
     * @return time derivative od mean concentration
     */
    double evaluate_mean_concentration_time_derivative(const int gp_id) const;

    /*!
     * @brief update multi-scale scalar transport material
     *
     * @param[in] gp_id  macro-scale Gauss point ID
     */
    void update(const int gp_id) const;

    /*!
     * @brief create output on micro scale
     *
     * @param[in] gp_id  macro-scale Gauss point ID
     */
    void output(const int gp_id) const;

    /*!
     * @brief  read restart on micro scale
     *
     * @param[in] gp_id  macro-scale Gauss point ID
     */
    void read_restart(const int gp_id) const;

    //! return name of micro-scale input file
    std::string micro_input_file_name() const { return params()->micro_input_file_name(); }

    //! return number of micro-scale discretization
    int micro_dis_num() const { return params()->micro_dis_num(); }

    //! return specific micro-scale surface area A_s
    //!
    //! \param detF determinant of deformation gradient of macro dis at current Gauss point
    //! \return  specific area between micro and macro dis
    double specific_micro_scale_surface_area(const double detF) const
    {
      return params()->specific_micro_scale_surface_area() * std::pow(detF, -1.0 / 3.0);
    }

    //! set time stepping data: time step size @p dt, current time @p time, and number of time step
    //! @p step on current Gauss point @p gp_id
    void set_time_stepping(int gp_id, double dt, double time, int step);

   protected:
    //! construct empty material
    ScatraMicroMacroCoupling();

   private:
    //! material parameters
    virtual const Mat::PAR::ScatraMicroMacroCoupling* params() const = 0;

    //! map between Gauss point ID and Gauss point submaterial
    std::map<int, std::shared_ptr<ScatraMultiScaleGP>> matgp_;
  };  // material wrapper
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
