// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SSTI_MONOLITHIC_HPP
#define FOUR_C_SSTI_MONOLITHIC_HPP

#include "4C_config.hpp"

#include "4C_ssti_algorithm.hpp"

#include <Teuchos_Time.hpp>

FOUR_C_NAMESPACE_OPEN

// forward declarations

namespace Core::LinAlg
{
  class Equilibration;
  enum class EquilibrationMethod;
  enum class MatrixType;
  class Solver;
  class SparseMatrix;
  class SparseOperator;
}  // namespace Core::LinAlg

namespace ScaTra
{
  class meshtying_strategy_s2_i;
}

namespace STI
{
  class ScatraThermoOffDiagCoupling;
}

namespace SSI
{
  class ScatraStructureOffDiagCoupling;
}  // namespace SSI

namespace SSTI
{
  class AssembleStrategyBase;
  class ConvCheckMono;
  class SSTIMapsMono;
  class SSTIMatrices;
  class ThermoStructureOffDiagCoupling;

  //! equilibration methods applied to system matrix
  struct SSTIMonoEquilibrationMethod
  {
    const Core::LinAlg::EquilibrationMethod global;     //! unique equilibration
    const Core::LinAlg::EquilibrationMethod scatra;     //! equilibration for scatra block
    const Core::LinAlg::EquilibrationMethod structure;  //! equilibration for structure block
    const Core::LinAlg::EquilibrationMethod thermo;     //! equilibration for thermo block
  };

  enum class Subproblem
  {
    structure,
    scalar_transport,
    thermo
  };

  class SSTIMono : public SSTIAlgorithm
  {
   public:
    explicit SSTIMono(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);
    //! get vector containing positions within system matrix for specific subproblem
    std::vector<int> get_block_positions(Subproblem subproblem) const;

    //! get position within global dof map for specific subproblem
    int get_problem_position(Subproblem subproblem) const;

    //! Setup of algorithm
    //@{
    void init(MPI_Comm comm, const Teuchos::ParameterList& sstitimeparams,
        const Teuchos::ParameterList& scatraparams, const Teuchos::ParameterList& thermoparams,
        const Teuchos::ParameterList& structparams) override;
    void setup() override;
    void setup_system() override;
    //@}

    //! Loop over all time steps
    void timeloop() override;

    //! return all maps
    std::shared_ptr<SSTI::SSTIMapsMono> all_maps() const { return ssti_maps_mono_; };

    //! number of current Newton Iteration
    unsigned int newton_iteration() const { return iter(); };

    //! state vectors
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> increment() const { return increment_; };
    std::shared_ptr<Core::LinAlg::Vector<double>> residual() const { return residual_; };
    //}

    //! statistics for evaluation and solving
    std::vector<double> time_statistics() const
    {
      return {dtevaluate_ + dtassemble_, dtsolve_, dtnewton_};
    };

   private:
    //! assemble global system of equations
    void assemble_mat_and_rhs();

    //! build null spaces associated with blocks of global system matrix
    void build_null_spaces() const;

    //! Get Matrix and Right-Hand-Side for all subproblems incl. coupling
    void evaluate_subproblems();

    //! get solution increment for given subproblem
    std::shared_ptr<Core::LinAlg::Vector<double>> extract_sub_increment(Subproblem sub);

    // build and return vector of equilibration methods for each block of system matrix
    std::vector<Core::LinAlg::EquilibrationMethod> get_block_equilibration() const;

    //! evaluate time step using Newton-Raphson iteration
    void newton_loop();

    //! output solution to screen and files
    void output() override;

    void prepare_newton_step();

    //! prepare time step
    void prepare_time_step() override;

    //! solve linear system of equations
    void linear_solve();

    //! update scalar transport and structure fields after time step evaluation
    void update() override;

    //! update routine after newton iteration
    void update_iter_states();

    //! Newton Raphson loop
    //@{
    std::shared_ptr<Core::LinAlg::Vector<double>> increment_;
    std::shared_ptr<Core::LinAlg::Vector<double>> residual_;
    std::shared_ptr<Core::LinAlg::Solver> solver_;
    //@}

    //! evaluation of off-diagonal blocks
    //@{
    std::shared_ptr<SSI::ScatraStructureOffDiagCoupling> scatrastructureoffdiagcoupling_;
    std::shared_ptr<STI::ScatraThermoOffDiagCoupling> scatrathermooffdiagcoupling_;
    std::shared_ptr<SSTI::ThermoStructureOffDiagCoupling> thermostructureoffdiagcoupling_;
    //@}

    //! time monitor
    //@{
    double dtassemble_;
    double dtevaluate_;
    double dtnewton_;
    double dtsolve_;
    std::shared_ptr<Teuchos::Time> timer_;
    //@}

    //! control parameters
    //@{
    //! equilibration method applied to system matrix
    const struct SSTIMonoEquilibrationMethod equilibration_method_;
    const Core::LinAlg::MatrixType matrixtype_;
    //@}

    //! convergence check of Newton iteration
    std::shared_ptr<SSTI::ConvCheckMono> convcheck_;

    //! all maps
    std::shared_ptr<SSTI::SSTIMapsMono> ssti_maps_mono_;

    //! system matrix and submatrices
    std::shared_ptr<SSTI::SSTIMatrices> ssti_matrices_;

    //! strategy how to assembly system matrix and rhs
    std::shared_ptr<SSTI::AssembleStrategyBase> strategy_assemble_;

    //! all equilibration of global system matrix and RHS is done in here
    std::shared_ptr<Core::LinAlg::Equilibration> strategy_equilibration_;
  };
}  // namespace SSTI
FOUR_C_NAMESPACE_CLOSE

#endif
