// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_PARTITIONED_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_ELAST_SCATRA_PARTITIONED_HPP


#include "4C_config.hpp"

#include "4C_porofluid_pressure_based_elast_scatra_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Global
{
  class Problem;
}  // namespace Global

namespace PoroPressureBased
{
  //! Partitioned solution scheme of porofluid-elasticity problems with scalar transport
  class PorofluidElastScatraPartitionedAlgorithm : public PorofluidElastScatraBaseAlgorithm
  {
   public:
    PorofluidElastScatraPartitionedAlgorithm(
        Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& poroparams,
        const Teuchos::ParameterList& structparams, const Teuchos::ParameterList& fluidparams,
        const Teuchos::ParameterList& scatraparams, const std::string& struct_disname,
        const std::string& fluid_disname, const std::string& scatra_disname, bool isale,
        int nds_disp, int nds_vel, int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    /// setup
    void setup_system() override;

    /// setup solver (only needed for poromultiphase monolithic coupling)
    void setup_solver() override;

    /// time step of coupled problem
    void time_step() override { return solve(); };

    /// print header
    void print_header_partitioned();

    /// print header
    void iter_update_states();

    //! perform iteration loop between fields
    virtual void solve() = 0;


   protected:
    //! perform iteration step of structure-fluid field
    void do_poro_step();

    //! perform iteration step of scatra field
    void do_scatra_step();

    //! convergence check of outer loop
    bool convergence_check(int itnum);

    //! scalar increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> scatra_inc_np_;
    //! structure increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> structure_inc_np_;
    //! fluid increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> porofluid_inc_np_;
    //! artery scatra increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> artery_scatra_inc_np_;
    //! artery pressure increment of the outer loop
    std::shared_ptr<Core::LinAlg::Vector<double>> artery_pressure_inc_np_;

    //! maximum iteration steps
    int iter_max_;
    //! convergence tolerance
    double iter_tol_;
    //! is artery coupling active
    bool artery_coupling_active_;
  };

  //! Nested partitioned solution algorithm
  // +--------------------------+           +----------+
  // |         ---->            |  ------>  |          |
  // |  fluid        structure  |           | ScaTra   |
  // |         <----            |  <------  |          |
  // +--------------------------+           +----------+

  class PorofluidElastScatraNestedPartitionedAlgorithm
      : public PorofluidElastScatraPartitionedAlgorithm
  {
   public:
    PorofluidElastScatraNestedPartitionedAlgorithm(
        Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& poroparams,
        const Teuchos::ParameterList& structparams, const Teuchos::ParameterList& fluidparams,
        const Teuchos::ParameterList& scatraparams, const std::string& struct_disname,
        const std::string& fluid_disname, const std::string& scatra_disname, bool isale,
        int nds_disp, int nds_vel, int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    //! perform iteration loop between fields
    void solve() override;
  };

  //! Sequential partitioned solution algorithm
  // +-----------+          +-----------+           +-----------+
  // |           |  ----->  |           | --------> |           |
  // |   fluid   |          | structure |           |  ScaTra   |
  // |           |          |           |           |           |
  // +-----------+          +-----------+           +-----------+
  //      ^                                              |
  //      |----------------------------------------------+

  class PorofluidElastScatraSequentialPartitionedAlgorithm
      : public PorofluidElastScatraPartitionedAlgorithm
  {
   public:
    PorofluidElastScatraSequentialPartitionedAlgorithm(
        Global::Problem& problem, MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams);

    /// initialization
    void init(const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& algoparams, const Teuchos::ParameterList& poroparams,
        const Teuchos::ParameterList& structparams, const Teuchos::ParameterList& fluidparams,
        const Teuchos::ParameterList& scatraparams, const std::string& struct_disname,
        const std::string& fluid_disname, const std::string& scatra_disname, bool isale,
        int nds_disp, int nds_vel, int nds_solidpressure, int ndsporofluid_scatra,
        const std::map<int, std::set<int>>* nearby_ele_pairs) override;

    //! perform iteration loop between fields
    void solve() override;
  };


}  // namespace PoroPressureBased



FOUR_C_NAMESPACE_CLOSE

#endif
