// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LEVELSET_ALGORITHM_HPP
#define FOUR_C_LEVELSET_ALGORITHM_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_geometry_geo_utils.hpp"
#include "4C_inpar_levelset.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_scatra_timint_implicit.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_MpiComm.h>

#include <memory>

#define USE_PHIN_FOR_VEL  // TODO

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SerialDenseMatrix;
}


namespace ScaTra
{
  class LevelSetAlgorithm : public virtual ScaTraTimIntImpl
  {
   public:
    /// Standard Constructor
    LevelSetAlgorithm(std::shared_ptr<Core::FE::Discretization> dis,
        std::shared_ptr<Core::LinAlg::Solver> solver,
        std::shared_ptr<Teuchos::ParameterList> params,
        std::shared_ptr<Teuchos::ParameterList> sctratimintparams,
        std::shared_ptr<Teuchos::ParameterList> extraparams,
        std::shared_ptr<Core::IO::DiscretizationWriter> output);


    // -----------------------------------------------------------------
    // general methods
    // -----------------------------------------------------------------

    /// initialize level-set algorithm
    void init() override;

    /// setup level-set algorithm
    void setup() override;

    /// time loop
    void time_loop() override;

    void check_and_write_output_and_restart() override;

    /// read restart data
    void read_restart(
        const int step, std::shared_ptr<Core::IO::InputControl> input = nullptr) override = 0;

    //! set the velocity field (zero or field by function) (pure level-set problems)
    void set_velocity_field(bool init = false);

    /// set convective velocity field (+ pressure and acceleration field as
    /// well as fine-scale velocity field, if required) (function for coupled fluid problems)
    void set_velocity_field(std::shared_ptr<const Core::LinAlg::Vector<double>> convvel,
        std::shared_ptr<const Core::LinAlg::Vector<double>> acc,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel,
        std::shared_ptr<const Core::LinAlg::Vector<double>> fsvel, bool setpressure = false,
        bool init = false);


    // output position of center of mass assuming a smoothed interfaces
    void mass_center_using_smoothing();

    /// redistribute the scatra discretization and vectors according to nodegraph
    void redistribute(Core::LinAlg::Graph& nodegraph);

    void test_results() override;

    /// set time and step value
    void set_time_step(const double time, const int step) override;

    // -----------------------------------------------------------------
    // general methods
    // -----------------------------------------------------------------

    /// setup the variables to do a new time step
    void prepare_time_step() override;

    /// solve level-set equation
    void solve() override;

    /// calculate error compared to analytical solution
    void evaluate_error_compared_to_analytical_sol() override;

   protected:
    virtual void get_initial_volume_of_minus_domain(
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& phinp,
        const std::shared_ptr<const Core::FE::Discretization>& scatradis,
        double& volumedomainminus) const;

    //! identify interface side due to phivalue value
    inline bool plus_domain(double phiValue)
    {
      double TOL = 1.0e-14;
      if (phiValue < -TOL)
        return false;
      else
        return true;
    };

    /// update state vectors
    virtual void update_state() = 0;

    // -----------------------------------------------------------------
    // reinitialization
    // -----------------------------------------------------------------

    /// reinitialize level-set
    virtual void reinitialization();

    /** \brief calculation of nodal velocity field via L2-projection for reinitialization
     *
     * (helper function for reinit_eq()) */
    virtual void calc_node_based_reinit_vel();

    /// execute the elliptic reinitialization procedure
    void reinitialize_with_elliptic_equation();

    /// set element parameters for reinitialization equation
    void set_reinitialization_element_parameters(bool calcinitialtimederivative = false) const;

    /** \brief access nodal gradient-based values for reinitialization
     *
     * (reinit_eq() only; Sussman and Elliptic) */
    inline std::shared_ptr<Core::LinAlg::MultiVector<double>>& nodal_grad_based_value()
    {
      return nb_grad_val_;
    }

    inline std::shared_ptr<const Core::LinAlg::MultiVector<double>> nodal_grad_based_value() const
    {
      return nb_grad_val_;
    }

    /// access the interval for reinitialization (every 'reinitinterval_' time steps)
    inline const int& reinit_interval() const { return reinitinterval_; }

    /// use projection for grad phi and related quantities
    inline const bool& use_projection() const { return projection_; }

    // -----------------------------------------------------------------
    // Reconstructing nodal curvature
    // -----------------------------------------------------------------
    void reconstructed_nodal_curvature(std::shared_ptr<Core::LinAlg::Vector<double>> curvature,
        const std::shared_ptr<const Core::LinAlg::Vector<double>> phi,
        const std::shared_ptr<const Core::LinAlg::MultiVector<double>> gradphi);

    // -----------------------------------------------------------------
    // members
    // -----------------------------------------------------------------

    /// the parameter list for level-set problems
    std::shared_ptr<Teuchos::ParameterList> levelsetparams_;

    /// options for reinitialization of G-function;
    Inpar::ScaTra::ReInitialAction reinitaction_;

    /// flag to switch between standard integration and sub-time loop for reinitialization
    bool switchreinit_;

    /// maximal number of pseudo time steps (reinit_eq() only)
    int pseudostepmax_;

    /// pseudo time step counter (reinit_eq() only)
    int pseudostep_;

    /// pseudo time step length (reinit_eq() only)
    double dtau_;

    /// pseudo theata (reinit_eq() only)
    double thetareinit_;

   private:
    /// add parameters depending of the problem, i.e., loma, level-set, ...
    void add_problem_specific_parameters_and_vectors(Teuchos::ParameterList& params) override;

    /// manipulate velocity field away from the interface
    void manipulate_fluid_field_for_gfunc();

    /// modification of convective velocity at contact points
    void apply_contact_point_boundary_condition();

    // -----------------------------------------------------------------
    // reinitialization
    // -----------------------------------------------------------------

    /// algebraic reinitialization via solution of equation to steady state
    void reinit_eq();

    /// set time parameters for reinitialization equation
    void set_reinitialization_element_time_parameters();

    /// preparations to solve reinitialization equation within existing framework (helper function
    /// for reinit_eq())
    void prepare_time_loop_reinit();

    /// time loop for reinitialization equation (helper function for reinit_eq())
    void time_loop_reinit();

    /// clean necessary modifications to solve reinitialization equation within existing framework
    /// (helper function for reinit_eq())
    void finish_time_loop_reinit();

    /// setup the variables to do a new reinitialization time step (helper function for reinit_eq())
    void prepare_time_step_reinit();

    /// nonlinear solver for reinitialization equation (helper function for reinit_eq())
    void solve_reinit();

    /// correction step according to Sussman & Fatemi 1999 (helper function for reinit_eq())
    void correction_reinit();

    /// convergence check for reinit equation according to Sussman et al. 1994 (helper function for
    /// reinit_eq())
    bool convergence_check_reinit();

    /// update phi within the reinitialization loop
    virtual void update_reinit() = 0;

    /// geometric reinitialization via computation of distance of node to interface
    void reinit_geo(const std::map<int, Core::Geo::BoundaryIntCells>& interface);

    /// compute normal vector of interface patch (helper function for reinit_geo())
    void compute_normal_vector_to_interface(const Core::Geo::BoundaryIntCell& patch,
        const Core::LinAlg::SerialDenseMatrix& patchcoord, Core::LinAlg::Matrix<3, 1>& normal);

    /// compute distance to vertex of patch (helper function for reinit_geo())
    void compute_distance_to_patch(const Core::LinAlg::Matrix<3, 1>& node,
        const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
        double& vertexdist);

    /// compute distance to edge of patch (helper function for reinit_geo())
    void compute_distance_to_edge(const Core::LinAlg::Matrix<3, 1>& node,
        const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
        double& edgedist);

    /// find a facing interface patch by projection of node into boundary cell space (helper
    /// function for reinit_geo())
    void find_facing_patch_proj_cell_space(const Core::LinAlg::Matrix<3, 1>& node,
        const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
        const Core::LinAlg::Matrix<3, 1>& normal, bool& facenode, double& patchdist);

    /// compares the second entry of a pair<int,double>. To be passed to the sorting algo (helper
    /// function for reinit_geo())
    static bool my_compare_pairs(
        const std::pair<int, double>& first, const std::pair<int, double>& second)
    {
      if (fabs(first.second) < fabs(second.second))
        return true;
      else
        return false;
    };

    /// project node into the boundary cell space (2D) (helper function for reinit_geo())
    template <Core::FE::CellType distype>
    bool project_node_on_patch(const Core::LinAlg::Matrix<3, 1>& node,
        const Core::Geo::BoundaryIntCell& patch, const Core::LinAlg::SerialDenseMatrix& patchcoord,
        const Core::LinAlg::Matrix<3, 1>& normal, Core::LinAlg::Matrix<2, 1>& eta, double& alpha);

    /// correct the volume of the minus domain after reinitialization
    void correct_volume();

    /// elliptic reinitialization
    void reinit_elliptic(std::map<int, Core::Geo::BoundaryIntCells>& interface);

    // -----------------------------------------------------------------
    // additional post-processing and evaluation methods
    // -----------------------------------------------------------------

    virtual void output_of_level_set_specific_values();

    // check for mass conservation before and after reinitialization as well as
    // at the end of the time step
    void mass_conservation_check(const double actvolminus, const bool writetofile = false);

    // reconstruction of interface and output of domains
    void capture_interface(
        std::map<int, Core::Geo::BoundaryIntCells>& interface, const bool writetofile = false);

    // -----------------------------------------------------------------
    // members
    // -----------------------------------------------------------------

    // --------------
    // members related to reinitialization and mass conservation

    /// initial volume of minus domain
    double initvolminus_;

    /// phinp before reinitialization (reinit_eq() only)
    std::shared_ptr<Core::LinAlg::Vector<double>> initialphireinit_;

    /// nodal gradient-based values for reinitialization (reinit_eq() only; Sussman and Elliptic)
    std::shared_ptr<Core::LinAlg::MultiVector<double>> nb_grad_val_;

    /// interval for reinitialization (every 'reinitinterval_' time steps)
    int reinitinterval_;

    /// switch for reinitialization only within a band around the interface (reinit_geo() only)
    bool reinitband_;

    /// band width for reinitialization (maximum level-set value)
    double reinitbandwidth_;

    /// flag to activate corrector step (reinit_eq() only)
    bool reinitcorrector_;

    /// evaluation of velocity field for reinitialization (reinit_eq() only)
    Inpar::ScaTra::VelReinit useprojectedreinitvel_;

    /// 2D flag
    Inpar::ScaTra::LSDim lsdim_;

    /// use projection for grad phi and related quantities
    bool projection_;

    // TODO:
    //    /// vector containing denominator of penalty parameter for each element (reinit_eq() only)
    //    std::shared_ptr<Core::LinAlg::Vector<double>> lambda_ele_denominator_;
    //
    //    /// vector containing smoothed haevyside function for each dof (reinit_eq() only)
    //    std::shared_ptr<Core::LinAlg::Vector<double>> node_deriv_smoothfunct_;

    /// tolerance for convergence check according to Sussman et al. 1994 (turned off negative)
    /// (reinit_eq() only)
    double reinit_tol_;

    /// flag to correct volume after reinitialization
    bool reinitvolcorrection_;

    /// interface for elliptic reinitialization
    std::shared_ptr<std::map<int, Core::Geo::BoundaryIntCells>> interface_eleq_;

    // --------------
    // members related to transport and velocity fields from Navier-Stokes

    /// flag for replacing the velocity of nodes at a certain distance of the interface by an
    /// approximate interface velocity
    bool extract_interface_vel_;

    /// number of element layers around the interface with unmodified velocity
    int convel_layers_;

    /// flag for modification of convective velocity at contact points
    bool cpbc_;
  };

}  // namespace ScaTra

FOUR_C_NAMESPACE_CLOSE

#endif
