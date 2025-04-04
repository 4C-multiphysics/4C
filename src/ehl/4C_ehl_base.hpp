// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_EHL_BASE_HPP
#define FOUR_C_EHL_BASE_HPP


#include "4C_config.hpp"

#include "4C_adapter_algorithmbase.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_ehl_input.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_lubrication_adapter.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Adapter
{
  class Structure;
  class CouplingEhlMortar;
}  // namespace Adapter

namespace Core::LinAlg
{
  class MapExtractor;
}

namespace EHL
{
  class Base : public Adapter::AlgorithmBase
  {
   public:
    explicit Base(MPI_Comm comm, const Teuchos::ParameterList& globaltimeparams,
        const Teuchos::ParameterList& lubricationparams, const Teuchos::ParameterList& structparams,
        const std::string struct_disname,
        const std::string lubrication_disname);  // Problem builder

    /// setup
    virtual void setup_system() = 0;

    /*!
     * @brief Perform all necessary tasks after setting up the object.
     * Currently, this only calls the post setup routine of the
     * structure field.
     */
    void post_setup() { structure_->post_setup(); }

    /// timeloop of coupled problem
    virtual void timeloop() = 0;

    /// test results (if necessary)
    void test_results(MPI_Comm comm);

    /// read restart
    void read_restart(int restart) override;

    //! access to structural field
    const std::shared_ptr<Adapter::Structure>& structure_field() { return structure_; }

    /// set structure solution on lubrication field
    void set_struct_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> disp);

    /// set lubrication solution on structure field
    void set_lubrication_solution(std::shared_ptr<const Core::LinAlg::Vector<double>> pressure);

    /// evaluate fluid forces on structure
    std::shared_ptr<Core::LinAlg::Vector<double>> evaluate_fluid_force(
        const Core::LinAlg::Vector<double>& pressure);

   protected:
    void add_pressure_force(
        Core::LinAlg::Vector<double>& slaveiforce, Core::LinAlg::Vector<double>& masteriforce);

    void add_poiseuille_force(
        Core::LinAlg::Vector<double>& slaveiforce, Core::LinAlg::Vector<double>& masteriforce);

    void add_couette_force(
        Core::LinAlg::Vector<double>& slaveiforce, Core::LinAlg::Vector<double>& masteriforce);

    /// underlying structure of the EHL problem
    std::shared_ptr<Adapter::Structure> structure_;

    /// underlying lubrication problem of the EHL problem
    std::shared_ptr<Lubrication::LubricationBaseAlgorithm> lubrication_;

    //! Type of coupling strategy between the two fields of the EHL problems
    const EHL::FieldCoupling fieldcoupling_;

    //! adapter for coupling the nodes of the lubrication field with the nodes from the master side
    //! of the structure
    std::shared_ptr<Adapter::CouplingEhlMortar> mortaradapter_;

    //! Interface traction vector in the slave str dof map
    std::shared_ptr<Core::LinAlg::Vector<double>> stritraction_D_;
    std::shared_ptr<Core::LinAlg::Vector<double>> stritraction_M_;

    //! Transformation matrix for lubrication pre dof map <-> lubrication disp dof map
    std::shared_ptr<Core::LinAlg::SparseMatrix> lubrimaptransform_;

    //! Mapextractors for dealing with interface vectors of the structure field
    std::shared_ptr<Core::LinAlg::MapExtractor> slaverowmapextr_;
    std::shared_ptr<Core::LinAlg::MapExtractor> masterrowmapextr_;
    std::shared_ptr<Core::LinAlg::MapExtractor> mergedrowmapextr_;

    //! Transformation matrix for slave side node map <-> slave side disp dof map
    std::shared_ptr<Core::LinAlg::SparseMatrix> slavemaptransform_;

    //! several adapters to transform maps
    std::shared_ptr<Coupling::Adapter::Coupling> ada_strDisp_to_lubDisp_;
    std::shared_ptr<Coupling::Adapter::Coupling> ada_strDisp_to_lubPres_;
    std::shared_ptr<Coupling::Adapter::Coupling> ada_lubPres_to_lubDisp_;

    //! height old vector to calculate the time derivative of height (Squeeze term)
    std::shared_ptr<const Core::LinAlg::Vector<double>> heightold_;

    //! use of a dry contact model
    bool dry_contact_;

    /// setup adapters for EHL on boundary
    virtual void setup_field_coupling(
        const std::string struct_disname, const std::string lubrication_disname);

    //! take current results for converged and save for next time step
    void update() override;

    //! write output
    virtual void output(bool forced_writerestart = false);
    std::shared_ptr<Core::LinAlg::Vector<double>> inf_gap_toggle_lub_;

    /// velocity calculation given the displacements
    std::shared_ptr<Core::LinAlg::Vector<double>> calc_velocity(
        const Core::LinAlg::Vector<double>& dispnp);

   private:
    /// setup discretizations and dofsets
    void setup_discretizations(
        MPI_Comm comm, const std::string struct_disname, const std::string lubrication_disname);

    /// set structure mesh displacement on lubrication field
    void set_mesh_disp(const Core::LinAlg::Vector<double>& disp);

    /// set average tangential interface velocity (ie structural velocities
    /// this is invariant w.r.t. rigid body rotations
    void set_average_velocity_field();

    /// set relative tangential interface velocity (ie structural velocities
    /// this is invariant w.r.t. rigid body rotations
    void set_relative_velocity_field();

    /// set film height of the lubrication field
    void set_height_field();

    /// set Time derivative of height (squeeze term)
    void set_height_dot();

    /// Create DBC map for unprojectable nodes
    void setup_unprojectable_dbc();
  };
}  // namespace EHL


FOUR_C_NAMESPACE_CLOSE

#endif
