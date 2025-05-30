// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_OST_HPP
#define FOUR_C_SCATRA_TIMINT_OST_HPP

#include "4C_config.hpp"

#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_scatra_timint_implicit.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN

namespace ScaTra
{
  class TimIntOneStepTheta : public virtual ScaTraTimIntImpl
  {
   public:
    /// Standard Constructor
    TimIntOneStepTheta(std::shared_ptr<Core::FE::Discretization> actdis,  //!< discretization
        std::shared_ptr<Core::LinAlg::Solver> solver,                     //!< linear solver
        std::shared_ptr<Teuchos::ParameterList> params,                   //!< parameter list
        std::shared_ptr<Teuchos::ParameterList> extraparams,     //!< supplementary parameter list
        std::shared_ptr<Core::IO::DiscretizationWriter> output,  //!< output writer
        const int probnum = 0                                    //!< global problem number
    );

    /// don't want = operator and cctor
    TimIntOneStepTheta operator=(const TimIntOneStepTheta& old) = delete;

    /// copy constructor
    TimIntOneStepTheta(const TimIntOneStepTheta& old) = delete;

    void init() override;

    void setup() override;

    void pre_solve() override {};

    void post_solve() override {};

    void print_time_step_info() override;

    void compute_intermediate_values() override {};

    void compute_interior_values() override {};

    void compute_time_derivative() override;

    void compute_time_deriv_pot0(const bool init) override {};

    void update() override;

    void read_restart(
        const int step, std::shared_ptr<Core::IO::InputControl> input = nullptr) override;

    std::shared_ptr<Core::LinAlg::Vector<double>> phiaf() override { return nullptr; }

    std::shared_ptr<Core::LinAlg::Vector<double>> phiam() override { return nullptr; }

    std::shared_ptr<Core::LinAlg::Vector<double>> phidtam() override { return nullptr; }

    std::shared_ptr<Core::LinAlg::Vector<double>> fs_phi() override
    {
      if (Sep_ != nullptr) Sep_->multiply(false, *phinp_, *fsphinp_);
      return fsphinp_;
    }

    std::shared_ptr<Teuchos::ParameterList> scatra_time_parameter_list() override
    {
      std::shared_ptr<Teuchos::ParameterList> timeparams;
      timeparams = std::make_shared<Teuchos::ParameterList>();
      timeparams->set("using stationary formulation", false);
      timeparams->set("using generalized-alpha time integration", false);
      timeparams->set("total time", time_);
      timeparams->set("time factor", theta_ * dta_);
      timeparams->set("alpha_F", 1.0);
      return timeparams;
    }

    //! set state on micro scale in multi-scale simulations
    void set_state(std::shared_ptr<Core::LinAlg::Vector<double>>
                       phin,  //!< micro-scale state vector at old time step
        std::shared_ptr<Core::LinAlg::Vector<double>>
            phinp,  //!< micro-scale state vector at new time step
        std::shared_ptr<Core::LinAlg::Vector<double>>
            phidtn,  //!< time derivative of micro-scale state vector at old time step
        std::shared_ptr<Core::LinAlg::Vector<double>>
            phidtnp,  //!< time derivative of micro-scale state vector at new time step
        std::shared_ptr<Core::LinAlg::Vector<double>> hist,  //!< micro-scale history vector
        std::shared_ptr<Core::IO::DiscretizationWriter>
            output,  //!< micro-scale discretization writer
        std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> visualization_writer,
        const std::vector<double>&
            phinp_macro,   //!< values of state variables at macro-scale Gauss point
        const int step,    //!< time step
        const double time  //!< time
    );

    //! clear state on micro scale in multi-scale simulations
    void clear_state();

    void pre_calc_initial_time_derivative() override;

    void post_calc_initial_time_derivative() override;

    void write_restart() const override;

   protected:
    void set_element_time_parameter(bool forcedincrementalsolver = false) const override;

    void set_time_for_neumann_evaluation(Teuchos::ParameterList& params) override;

    void calc_initial_time_derivative() override;

    void set_old_part_of_righthandside() override;

    void explicit_predictor() const override;

    void add_neumann_to_residual() override;

    void avm3_separation() override;

    void dynamic_computation_of_cs() override;

    void dynamic_computation_of_cv() override;

    void add_time_integration_specific_vectors(bool forcedincrementalsolver = false) override;

    double residual_scaling() const override { return 1.0 / (dta_ * theta_); }

    /// time factor for one-step-theta/BDF2 time integration
    double theta_;

    /// fine-scale solution vector at time n+1
    std::shared_ptr<Core::LinAlg::Vector<double>> fsphinp_;
  };

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
