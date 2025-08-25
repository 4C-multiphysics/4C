// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_MESHTYING_ABSTRACT_STRATEGY_HPP
#define FOUR_C_CONTACT_MESHTYING_ABSTRACT_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_contact_utils.hpp"
#include "4C_inpar_mortar.hpp"
#include "4C_mortar_strategy_base.hpp"

#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Mortar
{
  class Interface;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}
namespace NOX
{
  namespace Nln
  {
    class Group;
  }  // namespace Nln
}  // namespace NOX

namespace CONTACT
{
  // forward declarations
  class MtNoxInterface;

  /*! \brief Main abstract class for meshtying solution strategies

  This is the templating abstract class for all meshtying solution algorithms.
  Every solution algorithm has to fit into the set of functions and calls defined herein
  and has to be specified in a corresponding subclass defining the concrete algorithmic steps.

  This class it itself derived from the Mortar::StrategyBase class, which is an even
  more abstract framework for any solution strategies involving mortar coupling.

  */
  class MtAbstractStrategy : public Mortar::StrategyBase
  {
   public:
    /*!
    \brief Standard Constructor

    Creates the strategy object and initializes all global variables, including
    all necessary Core::LinAlg::Maps and global vector and matrix quantities.

    \param[in] dof_row_map Dof row map of underlying problem
    \param[in] NodeRowMap Node row map of underlying problem
    \param[in] params List of contact/parameters
    \param[in] interface All contact interface objects
    \param[in] spatialDim Spatial dimension of the problem
    \param[in] comm Communicator
    \param[in] alphaf Mid-point for Generalized-alpha time integration
    \param[in] maxdof Highest DOF number in global problem
    */
    MtAbstractStrategy(const Core::LinAlg::Map* dof_row_map, const Core::LinAlg::Map* NodeRowMap,
        Teuchos::ParameterList params, std::vector<std::shared_ptr<Mortar::Interface>> interface,
        const int spatialDim, const MPI_Comm& comm, const double alphaf, const int maxdof);



    //! @name Access methods

    /*!
    \brief Return Lagrange multiplier vector (t_n+1)

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier() const override
    {
      return z_;
    }

    /*!
    \brief Return old Lagrange multiplier vector (t_n)

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_old() const override
    {
      return zold_;
    }

    /*!
    \brief Return Lagrange multiplier vector from last Uzawa step

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagr_mult_uzawa() const { return zuzawa_; }

    /*!
    \brief Return constraint rhs vector (only in saddle-point formulation

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> constraint_rhs() const override
    {
      return constrrhs_;
    }

    /*!
    \brief Returns increment of Lagrange multiplier solution vector in SaddlePointSolve routine

    */
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_increment()
        const override
    {
      return zincr_;
    }

    /*!
    \brief Gather maps needed for contact/meshtying specific multigrid preconditioners

    @param MasterDofMap Dof row map of master interface
    @param SlaveDofMap Dof row map of slave interface
    @param InnerDofMap Dof row map of interior volume
    @param ActiveDofMap Dof row map of active slave contact interface
    */
    void collect_maps_for_preconditioner(std::shared_ptr<Core::LinAlg::Map>& MasterDofMap,
        std::shared_ptr<Core::LinAlg::Map>& SlaveDofMap,
        std::shared_ptr<Core::LinAlg::Map>& InnerDofMap,
        std::shared_ptr<Core::LinAlg::Map>& ActiveDofMap) const override;

    /*!
    \brief Return mortar matrix D

    */
    std::shared_ptr<const Core::LinAlg::SparseMatrix> d_matrix() const override { return dmatrix_; }

    /*!
    \brief Return mortar matrix M

    */
    std::shared_ptr<const Core::LinAlg::SparseMatrix> m_matrix() const override { return mmatrix_; }

    /*!
    \brief Get dual quadratic 3d slave element flag

    Returns TRUE if at least one higher-order 3d slave element with
    dual Lagrange multiplier shape functions in any interface.

    */
    virtual const bool& dualquadslavetrafo() const { return dualquadslavetrafo_; };

    /*!
    \brief Return parallel redistribution status (yes or no)

    */
    bool par_redist() const
    {
      Inpar::Mortar::ParallelRedist partype =
          Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(
              params().sublist("PARALLEL REDISTRIBUTION"), "PARALLEL_REDIST");
      if (partype != Inpar::Mortar::ParallelRedist::redist_none)
        return true;
      else
        return false;
    }

    //@}

    //! @name Evaluation methods

    /*!
    \brief Redistribute all meshtying interfaces in parallel

    Here, we call each interface to perform redistribution for each interface individually. Since
    this changes maps and interface discretizations, we have to fill_complete() all interface
    discretizations and re-setup the strategy object afterwards by calling setup(bool).

    If parallel redistribution is disabled in the input file or if this is a serial computation,
    i.e. only one MPI rank, then we just print the current parallel distribution to the screen, but
    do not change it.

    \pre Meshtying interface discretizations are distributed to multiple processors, but maybe not
    in an optimal way, i.e. with sub-optimal load balancing.

    \post If desired by the user, all meshtying interface discretizations have been re-distributed,
    such that load balancing among processors is closer to optimal.
    */
    void redistribute_meshtying() final;

    /*!
    \brief Global evaluation method called from time integrator

    */
    void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor = false) override;

    /*! \brief Reset call at the beginning of the apply_force(), apply_stiff() and
     * apply_force_stiff() [derived]
     *

     *  */
    virtual void reset(const Core::LinAlg::Vector<double>& dis)
    {
      FOUR_C_THROW("Not yet considered for meshtying!");
    };

    /*! \brief Global evaluation method called from Solid::ModelEvaluator::Contact class [derived]
     *
     *  Evaluation of the right-hand-side only. Necessary and meaningful for line search strategies
     *  for example.
     *

     *  */
    virtual bool apply_force()
    {
      FOUR_C_THROW("Not yet considered for msht!");
      return false;
    };

    /*! \brief Global evaluation method called from Solid::ModelEvaluator::Contact class [derived]
     *
     *  Evaluation of the mesh-tying right-hand-side and the mesh-tying jacobian. We call this
     * method also, when we are only interested in the jacobian, since the created overhead is
     * negligible.
     *

     *  */
    virtual bool apply_force_stiff()
    {
      FOUR_C_THROW("Not yet considered for msht!");
      return false;
    };

    /*!
    \brief Set current deformation state

    All interfaces are called to set the current deformation state.

    \param statename (in): std::string defining which quantity to set (only "displacement"
    applicable) \param vec (in): current global state of the quantity defined by statename

    */
    void set_state(
        const enum Mortar::StateType& statetype, const Core::LinAlg::Vector<double>& vec) override;

    /*!
    \brief Do mortar coupling in reference configuration

    Only do this ONCE for meshtying upon initialization!
    This method calls initialize() on all contact interfaces, which
    resets all kind of nodal quantities. It then calls evaluate() on
    all meshtying interfaces, which does all the geometric coupling stuff.
    Concretely, this is an evaluation of all involved quantities at nodal
    level. It includes the nodal normal calculations, search, projection
    and overlap detection and integration of the Mortar terms D and M.

    Then - on global level - it resets the Mortar matrices D and M accordingly.
    The nodal quantities computed before are assembled to global matrices. No
    setup of the global system is to be done here yet, so there is no need to
    pass in the effective stiffness K or the effective load vector f.

    Note: Only quantities common to all subsequent solving strategies (Lagrange,
    Penalty) are computed here. In case they need additional mortar variables,
    use the overloaded function call in the derived class and refer back to this function.

    */
    void mortar_coupling(const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis) override;

    //@}

    //! @name Quantity control methods

    /*!
    \brief Get some nodal quantity globally and store into Mortar::Node(s)

    The enum input parameter defines, which quantity has to be updated.
    Currently, the possibilities "lmold", "lmcurrent", "lmupdate" and
    "lmuzawa" exist. Note that "lmold" means the converged value LM_n
    of the last time / load step, whereas "lmcurrent" addresses the current
    (not necessarily converged) value of the LM_n+1. "lmupdate" is a special
    option called only in recover() after the update of the Lagr. multipliers.
    It basically does the same as "lmcurrent", but also checks for D.B.C.
    problems. Finally, "lmuzawa" addresses the LM update within an Uzawa
    augmented Lagrangian scheme.

    \param type (in): enum defining which quantity to store into Mortar::Node(s)

    */
    void store_nodal_quantities(Mortar::StrategyBase::QuantityType type) override;

    /*!
    \brief Get dirichlet B.C. status and store into Mortar::Node(s)

    This is called once at the beginning of the simulation
    to set the D.B.C. status in each Mortar::Node.

    \param dbcmaps (in): MapExtractor carrying global dbc map

    */
    void store_dirichlet_status(std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmaps) override;

    /*!
    \brief Update meshtying at end of time step

    \param dis (in):  current displacements (-> old displacements)

    */
    void update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    /*!
    \brief Perform a write restart

    A write restart is initiated by the contact manager. However, the manager has no
    direct access to the nodal quantities. Different from writing a restart step, now
    all the restart action has to be performed on the level of the meshtying algorithm,
    for short: here's the right place.

    */
    void do_read_restart(Core::IO::DiscretizationReader& reader,
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override;

    //@}

    //! @name Output

    /*!
    \brief Print interfaces

    \param[in] os Output stream used for printing
    */
    void print(std::ostream& os) const override;

    /*!
    \brief Print current active set to screen for debugging purposes

    */
    void print_active_set() const override;

    /*!
    \brief Write results for visualization separately for each meshtying/contact interface

    Call each interface, such that each interface can handle its own output of results.

    \param[in] outputParams Parameter list with stuff required by interfaces to write output
    */
    void postprocess_quantities_per_interface(
        std::shared_ptr<Teuchos::ParameterList> outputParams) const final;

    //! @}

    //! @name Preconditioner methods
    //! @{

    /*! Derived method
     *
     * (see NOX::Nln::CONSTRAINT::Interface::Preconditioner for more information) */
    bool is_saddle_point_system() const override;

    /*! \brief Derived method
     *
     * (see NOX::Nln::CONSTRAINT::Interface::Preconditioner for more information) */
    bool is_condensed_system() const override;

    /*! Fill the maps vector for the linear solver preconditioner
     *
     * The following order is defined:
     * (0) masterDofMap
     * (1) slaveDofMap
     * (2) innerDofMap
     * (3) activeDofMap
     *
     * */
    void fill_maps_for_preconditioner(
        std::vector<Teuchos::RCP<Core::LinAlg::Map>>& maps) const override;

    //! @}

    /*! @name Purely virtual functions
     *
     * All these functions are defined in one or more specific derived classes,
     * i.e CONTACT::MeshtyingLagrangeStrategy or CONTACT::MeshtyingPenaltyStrategy.
     * As the base class Mortar::StrategyBase is always called from the control routine
     * (time integrator), these functions need to be defined purely virtual here.
     */

    double constraint_norm() const override = 0;
    void evaluate_meshtying(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) override = 0;
    void initialize_uzawa(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) override = 0;
    double initial_penalty() const override = 0;
    void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) override = 0;
    void reset_penalty() override = 0;
    void modify_penalty() override = 0;
    void build_saddle_point_system(std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
        std::shared_ptr<Core::LinAlg::Vector<double>> fd,
        std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
        std::shared_ptr<Core::LinAlg::SparseOperator>& blockMat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs) override = 0;
    void update_displacements_and_l_mincrements(std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol) override = 0;
    void update_uzawa_augmented_lagrange() override = 0;
    void update_constraint_norm(int uzawaiter = 0) override = 0;

    //! @}

    /*! @name Empty functions (contact)
     *
     * All these functions only have functionality in contact simulations, thus they
     * are defined as empty here in the case of meshtying. They can be called from the
     * control routine (time integrator), whenever you like.
     */

    bool active_set_converged() const override { return true; }
    bool is_friction() const override { return false; }
    bool wear_both_discrete() const override { return false; }
    bool is_in_contact() const override { return true; }
    bool was_in_contact() const override { return true; }
    bool was_in_contact_last_time_step() const override { return true; }
    std::shared_ptr<const Core::LinAlg::Vector<double>> contact_normal_stress() const override
    {
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> contact_tangential_stress() const override
    {
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> contact_normal_force() const override
    {
      return nullptr;
    }
    std::shared_ptr<const Core::LinAlg::Vector<double>> contact_tangential_force() const override
    {
      return nullptr;
    }
    void assemble_mortar() override {}
    void do_write_restart(
        std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>& restart_vectors,
        bool forcedrestart = false) const override
    {
    }
    void initialize_and_evaluate_interface() override {}
    void initialize_mortar() override {}
    void initialize() override {}
    double inttime() const override { return inttime_; };
    void inttime_init() override { inttime_ = 0.0; };
    int number_of_active_nodes() const override { return 0; }
    int number_of_slip_nodes() const override { return 0; }
    void compute_contact_stresses() final {};
    void aug_forces(Core::LinAlg::Vector<double>& augfs_lm, Core::LinAlg::Vector<double>& augfs_g,
        Core::LinAlg::Vector<double>& augfm_lm, Core::LinAlg::Vector<double>& augfm_g) {};
    bool redistribute_contact(std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) final
    {
      return false;
    }
    void reset_active_set() override {}
    void save_reference_state(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) override {}
    void update_active_set() override {}
    void update_active_set_semi_smooth(const bool firstStepPredictor = false) override {}
    std::shared_ptr<Core::LinAlg::SparseMatrix> evaluate_normals(
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) override
    {
      return nullptr;
    }
    void evaluate_reference_state() override {}
    void evaluate_relative_movement() override {}
    void predict_relative_movement() override {}
    std::shared_ptr<const Core::LinAlg::Map> slave_row_nodes_ptr() const override
    {
      return gsnoderowmap_;
    }
    std::shared_ptr<const Core::LinAlg::Map> active_row_nodes() const override { return nullptr; }
    std::shared_ptr<const Core::LinAlg::Map> active_row_dofs() const override { return nullptr; }
    std::shared_ptr<const Core::LinAlg::Map> non_redist_slave_row_dofs() const override
    {
      return non_redist_gsdofrowmap_;
    }
    std::shared_ptr<const Core::LinAlg::Map> non_redist_master_row_dofs() const override
    {
      return non_redist_gmdofrowmap_;
    }
    std::shared_ptr<const Core::LinAlg::Map> slip_row_nodes() const override { return nullptr; }
    std::shared_ptr<const Core::LinAlg::Map> slave_dof_row_map_ptr() const { return gsdofrowmap_; }
    std::shared_ptr<const Core::LinAlg::Map> master_dof_row_map_ptr() const { return gmdofrowmap_; }

    //! @}

    //! @name New time integration
    //! @{

    //! Return the NOX::Nln::CONSTRAINT::Interface::Required member object
    const std::shared_ptr<CONTACT::MtNoxInterface>& nox_interface_ptr()
    {
      return noxinterface_ptr_;
    };

    /*! \brief Return the desired right-hand-side block pointer (read-only)
     *
     *  \remark Please note, that a nullptr pointer is returned, if no active contact
     *  contributions are present.
     *
     *  \param bt (in): Desired vector block type, e.g. displ, constraint, ...
     *

     *  */
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> get_rhs_block_ptr(
        const enum CONTACT::VecBlockType& bt) const = 0;

    /*! \brief Return the desired matrix block pointer (read-only)
     *
     *  \remark Please note, that a nullptr pointer is returned, if no active contact
     *  contributions are present.
     *
     *  \param bt (in): Desired matrix block type, e.g. displ_displ, displ_lm, ...
     *

     *  */
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> get_matrix_block_ptr(
        const enum CONTACT::MatBlockType& bt) const = 0;

    /*! \brief Return the current (maybe redistributed) Lagrange multiplier dof row map
     *
     */
    virtual std::shared_ptr<const Core::LinAlg::Map> lm_dof_row_map_ptr() const
    {
      return glmdofrowmap_;
    };

    /*! \brief Return the non-redistributed Lagrange multiplier dof row map
     *
     */
    virtual std::shared_ptr<const Core::LinAlg::Map> non_redist_lm_dof_row_map_ptr() const
    {
      return non_redist_glmdofrowmap_;
    };

    //! Modify system before linear solve
    virtual void run_pre_apply_jacobian_inverse(
        std::shared_ptr<Core::LinAlg::SparseMatrix> kteff, Core::LinAlg::Vector<double>& rhs)
    { /* do nothing */
    }

    //! modify result after linear solve
    virtual void run_post_apply_jacobian_inverse(Core::LinAlg::Vector<double>& result)
    { /* do nothing */ }

    //! evaluate force terms
    virtual bool evaluate_force(const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) = 0;

    //! evaluate stiffness terms
    virtual bool evaluate_stiff(const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) = 0;

    //! evaluate force and stiffness terms
    virtual bool evaluate_force_stiff(
        const std::shared_ptr<const Core::LinAlg::Vector<double>> dis) = 0;

    //! after applying Newton increment
    virtual void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
        const Core::LinAlg::Vector<double>& dir, const Core::LinAlg::Vector<double>& xnew) {};

    /*! \brief Get the correct RHS for convergence check
     *
     * \todo Is this really about the right-hand side vector or the residual?
     *
     * @param[in/out] rhs Right-hand side vector
     */
    virtual void remove_condensed_contributions_from_rhs(
        Core::LinAlg::Vector<double>& rhs) const {};

    //!@}

   protected:
    /*!
    \brief Assemble global coordinate vector

    \param sidename (in): std::string indicating slave or master side
    \param ref (in): boolean indicating evaluation in reference configuration
    \param vec (in/out)):  empty global vetcor to be assembled to

    */
    void assemble_coords(const std::string& sidename, bool ref, Core::LinAlg::Vector<double>& vec);

    /*!
    \brief Do mesh initialization for rotational invariance

    Only do this ONCE for meshtying upon initialization!
    This method relocates the slave nodes such that the meshtying constraint
    is satisfied in the reference configuration, which is a prerequisite for
    ensuring both rotational invariance and absence of initial stresses at the
    same time. Basically the constraint equation needs to be solved for this,
    which is specific to the applied solving strategy (dual Lagrange or Penalty).
    In the dual LM, matrix D is diagonal, thus its inversion is trivial and no
    linear system needs to be solved. In the penalty case, matrix D is not diagonal
    and we apply a default Core::LinAlg::Solver to solve for the modified slave positions.
    Thus, this linear system solve is done in the derived method FIRST and then
    we refer back to this base class function.

    \param Xslavemod (in): modified slave reference configuration
    */
    virtual void mesh_initialization(std::shared_ptr<Core::LinAlg::Vector<double>> Xslavemod);

   private:
    /*!
    \brief Evaluate contact

    This is just a tiny control routine, deciding which Evaluate-routine
    of those listed below is to be called (based on input-file information)
    Note that into ALL derived evaluate() routines, a REFERENCE to the pointer
    on the effective stiffness matrix is handed in. This way, after building the
    new effective stiffness matrix with contact, we can simply let the pointer
    kteff point onto the new object. The same is true for the effective force
    vector feff. Be careful: kteff is of type std::shared_ptr<Core::LinAlg::SparseOperator>&.

    \param kteff (in/out): effective stiffness matrix (without -> with contact)
    \param feff (in/out): effective residual / force vector (without -> with contact)

    */
    void evaluate(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) override;

    /*!
    \brief Restrict slave boundary to actual meshtying zone

    Only do this ONCE for meshtying upon initialization!
    This method first detects for each interface the actually tied part
    of the slave surface (i.e. the nodes that carry a D/M contribution).
    Then all slave maps on interface level and on global level are
    re-initialized and re-setup according to the the above defined
    actual slave meshtying zone. This is necessary for problems in which
    the slave surface does not fully project onto the master surface
    and thus the actual meshtying zone cannot be defined within the
    input file. Thus, it is computed here.

    */
    void restrict_meshtying_zone() override;

    /*!
    \brief Setup this strategy object (maps, vectors, etc.)

    All global maps and vectors are initialized by collecting
    the necessary information from all interfaces. In the case
    of a parallel redistribution, this method is called again
    to re-setup the above mentioned quantities. In this case
    the input parameter is set to TRUE.

    */
    void setup(bool redistributed);

   protected:
    // don't want cctor (= operator impossible anyway for abstract class)
    MtAbstractStrategy(const MtAbstractStrategy& old) = delete;

    //! Vector with all meshtying interfaces
    std::vector<std::shared_ptr<Mortar::Interface>> interface_;

    //! Global Lagrange multiplier dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> glmdofrowmap_;

    //! Global slave dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> gsdofrowmap_;

    //! Global master dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> gmdofrowmap_;

    //! Global internal dof row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> gndofrowmap_;

    //! Global slave and master dof row map (slave+master map)
    std::shared_ptr<Core::LinAlg::Map> gsmdofrowmap_;

    //! Global displacement dof row map (s+m+n map)
    std::shared_ptr<Core::LinAlg::Map> gdisprowmap_;

    //! Global slave node row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> gsnoderowmap_;

    //! Global master node row map (of all interfaces)
    std::shared_ptr<Core::LinAlg::Map> gmnoderowmap_;

    //! @name Parallel redistribution
    //!@{

    //! Global Lagrange multiplier dof row map (before parallel redistribution)
    std::shared_ptr<Core::LinAlg::Map> non_redist_glmdofrowmap_;

    //! Global slave dof row map (before parallel redistribution)
    std::shared_ptr<Core::LinAlg::Map> non_redist_gsdofrowmap_;

    //! Global master dof row map (before parallel redistribution)
    std::shared_ptr<Core::LinAlg::Map> non_redist_gmdofrowmap_;

    //! Global slave and master dof row map (before parallel redistribution)
    std::shared_ptr<Core::LinAlg::Map> non_redist_gsmdofrowmap_;

    //! Global dirichlet toggle of all slave dofs (before parallel redistribution)
    std::shared_ptr<Core::LinAlg::Vector<double>> non_redist_gsdirichtoggle_;

    //!@}

    //! @name Binning strategy
    //!@{

    //! Initial element column map for binning strategy (slave and master)
    std::vector<std::shared_ptr<Core::LinAlg::Map>> initial_elecolmap_;

    //! Global Mortar matrix \f$D\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> dmatrix_;

    //! Global Mortar matrix \f$M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> mmatrix_;

    //! Global weighted gap vector \f$g\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> g_;

    //! Global constraint right-hand side vector (only for saddlepoint problems)
    std::shared_ptr<Core::LinAlg::Vector<double>> constrrhs_;

    //! Current vector of Lagrange multipliers at \f$t_{n+1}\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> z_;

    //! Old vector of Lagrange multipliers at \f$t_n\f$
    std::shared_ptr<Core::LinAlg::Vector<double>> zold_;

    /*! \brief Lagrange multiplier vector increment within SaddlePointSolve
     *
     *  \remark This is \em not the increment of #z_ between \f$t_{n+1}\f$ and \f$t_{n}\f$!)
     */
    std::shared_ptr<Core::LinAlg::Vector<double>> zincr_;

    //! Vector of Lagrange multipliers from last Uzawa step
    std::shared_ptr<Core::LinAlg::Vector<double>> zuzawa_;

    //! @name Status flags
    //!@{

    /*! \brief Flag indicating whether transformation should be applied
     *
     * \todo What transformation?
     */
    bool dualquadslavetrafo_;

    //!@}


    /*! \brief Transformation matrix \f$T\f$ for dual quad 3D case
     *
     * \todo What is matrix \f$T\f$?
     * \todo What is quad? Quadratic shape functions or quadrilateral elements?
     * \todo What is the difference to #systrafo_?
     */
    std::shared_ptr<Core::LinAlg::SparseMatrix> trafo_;

    /*! \brief Transformation matrix \f$T\f$ for dual quad 3D case
     *
     * \todo What is matrix \f$T\f$?
     * \todo What is quad? Quadratic shape functions or quadrilateral elements?
     * \todo What is the difference to #trafo_?
     */
    std::shared_ptr<Core::LinAlg::SparseMatrix> systrafo_;

    /*! \brief Inverse trafo matrix \f$T^{-1}\f$ for dual quad 3D case
     *
     * \todo What is matrix \f$T\f$?
     * \todo What is quad? Quadratic shape functions or quadrilateral elements?
     */
    std::shared_ptr<Core::LinAlg::SparseMatrix> invtrafo_;

    /*! \brief Integration time
     *
     * \todo Is this the wall clock time required to perform the mortar integration?
     */
    double inttime_;

    //! Structural force
    std::shared_ptr<Core::LinAlg::Vector<double>> f_;

    //! Structural force (slave)
    std::shared_ptr<Core::LinAlg::Vector<double>> fs_;

    //! Matrix containing \f$D\f$ and \f$-M\f$
    std::shared_ptr<Core::LinAlg::SparseMatrix> dm_matrix_;

    /*! \brief Matrix containing D and -M. transposed
     *
     * \todo Is it \f$D\f$ and \f$-M^T\f$ or (D and -M)^T? In latter case, why store it explicitly?
     */
    std::shared_ptr<Core::LinAlg::SparseMatrix> dm_matrix_t_;

    //! Lagrange multiplier diagonal block
    std::shared_ptr<Core::LinAlg::SparseMatrix> lm_diag_matrix_;

   private:
    ///< pointer to the NOX::Nln::CONSTRAINT::Interface::Required object
    std::shared_ptr<CONTACT::MtNoxInterface> noxinterface_ptr_;

  };  // class MtAbstractStrategy
}  // namespace CONTACT

// << operator
std::ostream& operator<<(std::ostream& os, const CONTACT::MtAbstractStrategy& strategy);

FOUR_C_NAMESPACE_CLOSE

#endif
