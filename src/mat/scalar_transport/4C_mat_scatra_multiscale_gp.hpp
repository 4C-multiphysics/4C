// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_SCATRA_MULTISCALE_GP_HPP
#define FOUR_C_MAT_SCATRA_MULTISCALE_GP_HPP

#include "4C_config.hpp"

#include "4C_io_control.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linear_solver_method_linalg.hpp"

#include <memory>
#include <vector>

// forward declarations

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  class TimIntOneStepTheta;
}

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationVisualizationWriterMesh;
}  // namespace Core::IO

namespace Mat
{
  //! class implementation
  class ScatraMultiScaleGP
  {
   public:
    //!
    //! \param ele_id        macro-scale element ID
    //! \param gp_id         macro-scale Gauss point ID
    //! \param microdisnum   number of micro-scale discretization
    //! \param is_ale        true, if the underlying macro dis deforms
    ScatraMultiScaleGP(const int ele_id, const int gp_id, const int microdisnum, bool is_ale);

    //! destructor
    ~ScatraMultiScaleGP();

    //! perform initializations
    void init();

    //! prepare time step
    void prepare_time_step(const std::vector<double>& phinp_macro  //!< macro-scale state variables
    );

    //! evaluate micro scale
    //! \param phinp_macro     macro-scale state variables
    //! \param q_micro         micro-scale coupling flux
    //! \param dq_dphi_micro   derivatives of micro-scale coupling flux w.r.t. macro-scale state
    //!                        variables
    //! \param detF            determinant of deformation gradient of macro dis at current
    //!                        Gauss point
    //! \param solve           flag indicating whether micro-scale problem should be
    //!                        solved
    void evaluate(const std::vector<double>& phinp_macro, double& q_micro,
        std::vector<double>& dq_dphi_micro, double detF, const bool solve = true);

    //! evaluate mean concentration on micro scale
    double evaluate_mean_concentration() const;

    //! evaluate mean concentration time derivative on micro scale
    double evaluate_mean_concentration_time_derivative() const;

    //! update micro-scale time integrator at the end of each time step
    void update();

    //! output micro-scale quantities
    void output();

    //! collect the micro scale output data
    void collect_and_write_output_data();

    //! read restart on micro scale
    void read_restart();

    //! calculate derivative of determinate w.r.t. time according to macro time int scheme
    void calculate_ddet_f_dt(ScaTra::TimIntOneStepTheta& microtimint);

    //! set time stepping data: time step size @p dt, current time @p time, and number of time step
    //! @p step
    void set_time_stepping(double dt, double time, int step);

   private:
    //! create new result file
    void new_result_file();

    //! create path of new result file
    std::string new_result_file_path(const std::string& newprefix);

    //! macro-scale Gauss point ID
    const int gp_id_;

    //! macro-scale element ID
    const int ele_id_;

    //! flag indicating whether macro-scale element is ghosted or not
    const bool eleowner_;

    //! number of micro-scale discretization
    const int microdisnum_;

    //! time step
    int step_;

    //! micro-scale state vector at old time step
    std::shared_ptr<Core::LinAlg::Vector<double>> phin_;

    //! micro-scale state vector at new time step
    std::shared_ptr<Core::LinAlg::Vector<double>> phinp_;

    //! time derivative of micro-scale state vector at old time step
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtn_;

    //! time derivative of micro-scale state vector at new time step
    std::shared_ptr<Core::LinAlg::Vector<double>> phidtnp_;

    //! micro-scale history vector
    std::shared_ptr<Core::LinAlg::Vector<double>> hist_;

    //! micro-scale discretization writer
    std::shared_ptr<Core::IO::DiscretizationWriter> micro_output_;

    //! micro-scale visualization writer
    std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> micro_visualization_writer_;

    std::optional<Core::IO::OutputControl> micro_output_control_;

    //! file name prefix for restart
    std::string restartname_;

    //! determinate of deformation gradient of macro dis at last time step
    double det_fn_;

    //! determinate of deformation gradient of macro dis at current step
    double det_fnp_;

    //! derivative of determinate of deformation gradient of macro dis w.r.t. time at last time step
    double ddet_fdtn_;

    //! derivative of determinate of deformation gradient of macro dis w.r.t. time at current time
    //! step
    double ddet_fdtnp_;

    //! indicates if macro dis deforms
    const bool is_ale_;
  };  // class Mat::ScatraMultiScaleGP

}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
