/*----------------------------------------------------------------------*/
/*! \file

\brief Control routine for fluid-fluid (in)stationary solvers with XFEM

\level 2

 */
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_FLUID_XFLUID_FLUID_HPP
#define FOUR_C_FLUID_XFLUID_FLUID_HPP


#include "4C_config.hpp"

#include "4C_fluid_xfluid.hpp"
#include "4C_fluid_xfluid_fluid_state.hpp"

FOUR_C_NAMESPACE_OPEN


namespace XFEM
{
  class MeshProjector;
  class MeshCouplingFluidFluid;
}  // namespace XFEM

namespace FLD
{
  namespace UTILS
  {
    class XFluidFluidMapExtractor;
  }

  /*!
    Class can handle a fluid described on a XFEM discretization and an embedded
    standard FEM discretization

    \author kruse
    \date 01/15
  */
  class XFluidFluid : public XFluid
  {
    friend class XFluidResultTest;

   public:
    //! ctor
    XFluidFluid(const Teuchos::RCP<FLD::FluidImplicitTimeInt>& embedded_fluid,  ///< embedded fluid
        const Teuchos::RCP<Core::FE::Discretization>&
            xfluiddis,                                       ///< background fluid discretization
        const Teuchos::RCP<Core::LinAlg::Solver>& solver,    ///< fluid solver
        const Teuchos::RCP<Teuchos::ParameterList>& params,  ///< xfluid params
        bool ale_xfluid = false,  ///< background (XFEM) fluid in ALE-formulation
        bool ale_fluid = false    ///< embedded fluid in ALE-formulation
    );

    //! ctor for multiple mesh coupling
    XFluidFluid(const Teuchos::RCP<FLD::FluidImplicitTimeInt>& embedded_fluid,  ///< embedded fluid
        const Teuchos::RCP<Core::FE::Discretization>&
            xfluiddis,  ///< background fluid discretization
        const Teuchos::RCP<Core::FE::Discretization>&
            soliddis,  ///< structure discretization to couple with
        const Teuchos::RCP<Core::LinAlg::Solver>& solver,    ///< fluid solver
        const Teuchos::RCP<Teuchos::ParameterList>& params,  ///< xfluid params
        bool ale_xfluid = false,  ///< background (XFEM) fluid in ALE-formulation
        bool ale_fluid = false    ///< embedded fluid in ALE-formulation
    );

    /// initialization
    void init() override { init(true); }
    void init(bool createinitialstate) override;

    void create_initial_state() override;

    /// don't keep a sparse matrix underneath
    void use_block_matrix(bool splitmatrix = true) override;

    /// set fluid-fluid coupling specific parameters
    void set_x_fluid_fluid_params();

    /// set initial flow field for fluid domains
    void set_initial_flow_field(
        const Inpar::FLUID::InitialField initfield, const int startfuncno) override;

    /// set fluid-fluid interface fixed for current time step
    void set_interface_fixed();

    /// free fluid-fluid interface
    void set_interface_free();

    /// setup the variables to do a new time step
    void prepare_time_step() override;

    /// get initial guess for both fluids
    Teuchos::RCP<const Core::LinAlg::Vector> initial_guess() override;

    /// prepare solution (cut happens here)
    void prepare_xfem_solve() override;

    /// Monolithic FSI needs to access the linear fluid problem.
    void evaluate(Teuchos::RCP<const Core::LinAlg::Vector>
            stepinc  ///< solution increment between time step n and n+1
        ) override;

    /// Update the solution after convergence of the nonlinear
    /// iteration. Current solution becomes old solution of next timestep.
    void time_update() override;

    /*!
     * \brief set underlying dof-maps for new shape derivatives matrix
     * (required in the context of monolithic XFFSI)
     * \param (in) fsiextractor map extractor to distinguish between FSI/non-FSI dof
     * in the merged fluid-fluid-dof-map
     * \param (in) condelements map of conditioned elements
     */
    void prepare_shape_derivatives(
        const Teuchos::RCP<const Core::LinAlg::MultiMapExtractor> fsiextractor,
        const Teuchos::RCP<std::set<int>> condelements);

    /*!
     * \brief return fluid-fluid system matrix as block matrix
     * (required in the context of monolithic fluidsplit-XFFSI)
     * \param (in) innermap map of inner embedded and background fluid dof
     * \param (in) condmap map of fsi interface dof
     * \return coupled fluid-fluid block system matrix
     */
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix(
        Teuchos::RCP<Epetra_Map> innermap, Teuchos::RCP<Epetra_Map> condmap);

    Teuchos::RCP<const Core::LinAlg::Vector> rhs() override { return xff_state_->xffluidresidual_; }
    Teuchos::RCP<const Core::LinAlg::Vector> velnp() override { return xff_state_->xffluidvelnp_; }
    Teuchos::RCP<Core::LinAlg::Vector> write_access_velnp() override
    {
      return xff_state_->xffluidvelnp_;
    }
    Teuchos::RCP<const Core::LinAlg::Vector> veln() override { return xff_state_->xffluidveln_; }
    Teuchos::RCP<const Core::LinAlg::Vector> stepinc() override { return stepinc_; }

    Teuchos::RCP<Core::LinAlg::Vector> write_access_disp_old_state() { return dispnpoldstate_; }

    /// get merged fluid dofmap
    Teuchos::RCP<const Epetra_Map> dof_row_map() override
    {
      //    return Teuchos::rcp((xff_state_->xffluiddofrowmap_).get(), false); // TODO: does not
      //    work at the moment, no nox structure split gives segfault!
      return xff_state_->xffluiddofrowmap_;
    }

    /// get merged vel-pres-splitter
    Teuchos::RCP<Core::LinAlg::MapExtractor> vel_pres_splitter() override
    {
      return xff_state_->xffluidvelpressplitter_;
    }

    // get merged pressure dof-map
    Teuchos::RCP<const Epetra_Map> pressure_row_map() override;
    // get merged velocity dof-map
    Teuchos::RCP<const Epetra_Map> velocity_row_map() override;

    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> extended_shape_derivatives()
    {
      return extended_shapederivatives_;
    }

    /// get the combined fluid-fluid system matrix
    Teuchos::RCP<Core::LinAlg::SparseMatrix> system_matrix() override
    {
      return Teuchos::rcp_dynamic_cast<Core::LinAlg::SparseMatrix>(xff_state_->xffluidsysmat_);
    }

    /// get the combined fluid-fluid system matrix in block form (embedded and background fluid
    /// blocks)
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> block_system_matrix() override
    {
      return Teuchos::rcp_dynamic_cast<Core::LinAlg::BlockSparseMatrixBase>(
          xff_state_->xffluidsysmat_);
    }

    /// get map extractor for background/embedded fluid
    Teuchos::RCP<FLD::UTILS::XFluidFluidMapExtractor> const& x_fluid_fluid_map_extractor()
    {
      return xff_state_->xffluidsplitter_;
    }

    //! get combined background and embedded fluid dirichlet map extractor
    Teuchos::RCP<const Core::LinAlg::MapExtractor> get_dbc_map_extractor() override
    {
      return xff_state_->xffluiddbcmaps_;
    }

    //! access to new merged fluid state container
    Teuchos::RCP<FLD::XFluidState> get_new_state() override;

    /// solve fluid field after ALE-mesh relaxation
    void update_monolithic_fluid_solution(const Teuchos::RCP<const Epetra_Map>& fsidofmap);

    /// interpolate embedded fluid state vectors from previous mesh displacement state
    void interpolate_embedded_state_vectors();

    /// create a result test
    Teuchos::RCP<Core::UTILS::ResultTest> create_field_test() override;

    /// write output for both fluid discretizations
    void output() override;

    /// evaluate errors compared to implemented analytical solutions
    Teuchos::RCP<std::vector<double>> evaluate_error_compared_to_analytical_sol() override;

   private:
    /// projection of history from other discretization - returns true if projection was successful
    /// for all nodes
    bool x_timint_project_from_embedded_discretization(
        const Teuchos::RCP<XFEM::XFluidTimeInt>& xfluid_timeint,  ///< xfluid time integration class
        std::vector<Teuchos::RCP<Core::LinAlg::Vector>>&
            newRowStateVectors,  ///< vectors to be reconstructed
        Teuchos::RCP<const Core::LinAlg::Vector>
            target_dispnp,     ///< displacement col - vector timestep n+1
        const bool screen_out  ///< screen output?
        ) override;

    //! transfer dofs between Newton steps
    bool x_timint_do_increment_step_transfer(
        const bool screen_out, const bool firstcall_in_timestep) override;

    //! create a new merged state container
    void create_state() override;

    /// call the loop over elements to assemble volume and interface integrals
    void assemble_mat_and_rhs(int itnum  ///< iteration number
        ) override;

    /// Update velocity and pressure by increment
    void update_by_increment() override;

    /// add additional EOS terms to interface-contributing embedded fluid elements
    void add_eos_pres_stab_to_emb_layer();

    //! @name XFF time integration

    /// embedded fluid field
    Teuchos::RCP<FLD::FluidImplicitTimeInt> embedded_fluid_;

    /// pointer to fluid-fluid mesh coupling object
    Teuchos::RCP<XFEM::MeshCouplingFluidFluid> mc_xff_;

    /// for XFEM time integration by projection from embedded mesh
    Teuchos::RCP<XFEM::MeshProjector> projector_;

    /// whether the embedded fluid is an ALE-fluid
    bool ale_embfluid_;

    /// merged fluid state (cut-dependent) at time n+1
    Teuchos::RCP<FLD::XFluidFluidState> xff_state_;

    /// vector with Newton increments (used for monolithic fully newton fsi approach)
    Teuchos::RCP<Core::LinAlg::Vector> stepinc_;

    /// ALE-displacements of previous step
    Teuchos::RCP<Core::LinAlg::Vector> dispnpoldstate_;

    /// shape derivatives matrix (linearization with respect to mesh motion),
    /// including background fluid dof (set to zero)
    Teuchos::RCP<Core::LinAlg::BlockSparseMatrixBase> extended_shapederivatives_;

    /// flag, that indicates active shape derivatives
    bool active_shapederivatives_;

    /// flag, that indicates evaluation of pressure EOS-terms on outer embedded elements
    bool xff_eos_pres_emb_layer_;

    /// if true, solve eigenvalue problem to determine characteristic length for Nitsche's parameter
    bool nitsche_evp_;

    //! @name Fluid-fluid specific parameters
    //@{
    enum Inpar::XFEM::MonolithicXffsiApproach
        monolithic_approach_;  ///< type of monolithic XFFSI-approach
    enum Inpar::XFEM::XFluidFluidTimeInt xfem_timeintapproach_;  ///< XFF time integration approach
    //@}

    //! name of fluid-fluid condition for access from condition manager
    const std::string cond_name_;
  };

} /* namespace FLD */
FOUR_C_NAMESPACE_CLOSE

#endif
