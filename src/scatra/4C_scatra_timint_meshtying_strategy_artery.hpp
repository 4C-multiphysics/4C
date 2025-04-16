// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_ARTERY_HPP
#define FOUR_C_SCATRA_TIMINT_MESHTYING_STRATEGY_ARTERY_HPP

#include "4C_config.hpp"

#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_print.hpp"
#include "4C_scatra_timint_meshtying_strategy_base.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Adapter
{
  class ArtNet;
}  // namespace Adapter
namespace FSI
{
  class Monolithic;

  namespace Utils
  {
    class MatrixRowTransform;
    class MatrixColTransform;
    class MatrixRowColTransform;
  }  // namespace Utils
}  // namespace FSI

namespace PoroPressureBased
{
  class PoroMultiPhaseScaTraArtCouplBase;
}

namespace ScaTra
{
  class MeshtyingStrategyArtery : public MeshtyingStrategyBase
  {
   public:
    //! constructor
    explicit MeshtyingStrategyArtery(
        ScaTra::ScaTraTimIntImpl* scatratimint  //!< scalar transport time integrator
    );

    //! return global map of degrees of freedom
    const Core::LinAlg::Map& dof_row_map() const override;

    //! return global map of degrees of freedom
    std::shared_ptr<const Core::LinAlg::Map> art_scatra_dof_row_map() const;

    //! evaluate mesh-tying
    //! \note  nothing is done here
    //!        actual coupling (meshtying) is evaluated in Solve
    //!        reason for that is that we need the system matrix of the continuous scatra
    //!        problem with DBCs applied which is performed directly before calling solve
    void evaluate_meshtying() override {};

    //! init
    void init_meshtying() override;

    bool system_matrix_initialization_needed() const override { return false; }

    std::shared_ptr<Core::LinAlg::SparseOperator> init_system_matrix() const override
    {
      FOUR_C_THROW(
          "This meshtying strategy does not need to initialize the system matrix, but relies "
          "instead on the initialization of the field. If this changes, you also need to change "
          "'system_matrix_initialization_needed()' to return true");
      // dummy return
      return nullptr;
    }

    std::shared_ptr<Core::LinAlg::MultiMapExtractor> interface_maps() const override
    {
      FOUR_C_THROW("InterfaceMaps() is not implemented in MeshtyingStrategyArtery.");
      return nullptr;
    }

    //! setup
    void setup_meshtying() override;

    //! solver
    const Core::LinAlg::Solver& solver() const override;

    //! init the convergence check
    void init_conv_check_strategy() override;

    //! solve resulting linear system of equations
    void solve(const std::shared_ptr<Core::LinAlg::Solver>& solver,         //!< solver
        const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,  //!< system matrix
        const std::shared_ptr<Core::LinAlg::Vector<double>>& increment,     //!< increment vector
        const std::shared_ptr<Core::LinAlg::Vector<double>>& residual,      //!< residual vector
        const std::shared_ptr<Core::LinAlg::Vector<double>>& phinp,  //!< state vector at time n+1
        const int iteration,  //!< number of current Newton-Raphson iteration
        Core::LinAlg::SolverParams& solver_params) const override;

    void setup_system(
        const std::shared_ptr<Core::LinAlg::SparseOperator>& systemmatrix,  //!< system matrix
        const std::shared_ptr<Core::LinAlg::Vector<double>>& residual       //!< residual vector
    ) const;

    //! init the convergence check
    void set_artery_scatra_time_integrator(
        std::shared_ptr<ScaTra::ScaTraTimIntImpl> artscatratimint);

    //! set the artery time integrator
    void set_artery_time_integrator(std::shared_ptr<Adapter::ArtNet> arttimint);

    //! set the element pairs that are close as found by search algorithm
    void set_nearby_ele_pairs(const std::map<int, std::set<int>>* nearbyelepairs);

    //! prepare a time step
    void prepare_time_step() const;

    //! set the artery pressure
    void set_artery_pressure() const;

    //! apply mesh movement
    void apply_mesh_movement();

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> combined_system_matrix()
    {
      return comb_systemmatrix_;
    }

    //! get the combined rhs
    std::shared_ptr<Core::LinAlg::Vector<double>> combined_rhs() const { return rhs_; }

    //! get the combined increment
    std::shared_ptr<Core::LinAlg::Vector<double>> combined_increment() const
    {
      return comb_increment_;
    }

    //! access to time integrator
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> art_scatra_field() { return artscatratimint_; }

    //! check if initial fields match
    void check_initial_fields() const;

    //! update increment of 1D discretization
    void update_art_scatra_iter(std::shared_ptr<const Core::LinAlg::Vector<double>> combined_inc);

    /*!
     * extract single field vectors
     * @param[i] globalvec combined 1D-3D vector
     * @param[o] vec_cont  3D vector
     * @param[o] vec_art   1D vector
     */
    void extract_single_field_vectors(std::shared_ptr<const Core::LinAlg::Vector<double>> globalvec,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& vec_cont,
        std::shared_ptr<const Core::LinAlg::Vector<double>>& vec_art) const;

   private:
    //! initialize the linear solver
    void initialize_linear_solver(const Teuchos::ParameterList& scatraparams);
    //! time integrators
    std::shared_ptr<ScaTra::ScaTraTimIntImpl> artscatratimint_;
    std::shared_ptr<Adapter::ArtNet> arttimint_;

    //! mesh tying object
    std::shared_ptr<PoroPressureBased::PoroMultiPhaseScaTraArtCouplBase> arttoscatracoupling_;

    //! the two discretizations
    std::shared_ptr<Core::FE::Discretization> artscatradis_;
    std::shared_ptr<Core::FE::Discretization> scatradis_;

    //! block systemmatrix
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> comb_systemmatrix_;

    //! combined rhs
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs_;

    //! combined rhs
    std::shared_ptr<Core::LinAlg::Vector<double>> comb_increment_;

  };  // class MeshtyingStrategyArtery

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
