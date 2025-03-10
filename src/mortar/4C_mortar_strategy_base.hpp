// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_STRATEGY_BASE_HPP
#define FOUR_C_MORTAR_STRATEGY_BASE_HPP

#include "4C_config.hpp"

#include "4C_inpar_contact.hpp"  // for the Inpar::CONTACT enums
#include "4C_linalg_vector.hpp"
#include "4C_mortar_interface.hpp"                                       // for the enum state type
#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"  // interface specifications
#include "4C_utils_parameter_list.fwd.hpp"

#include <Epetra_Map.h>
#include <Epetra_Operator.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Inpar
{
  namespace Solid
  {
    enum DynamicType : int;
  }  // namespace Solid
}  // namespace Inpar
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE
namespace Core::LinAlg
{
  class MapExtractor;
  class Solver;
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Core::IO
{
  class DiscretizationWriter;
  class DiscretizationReader;
}  // namespace Core::IO

namespace Mortar
{
  /*! \brief Data container object for the strategy base
   *
   *  This object makes it possible to interchange and share the current state of the
   *  contact simulation between different strategy objects. By using this the
   *  actual strategy stays stateless!
   *
   */
  class StrategyDataContainer
  {
   public:
    //! constructor
    StrategyDataContainer();

    //! destructor
    virtual ~StrategyDataContainer() = default;

    //! Return underlying problem dof row map (not only interfaces)
    std::shared_ptr<Epetra_Map>& prob_dofs_ptr() { return probdofs_; }
    std::shared_ptr<const Epetra_Map> prob_dofs_ptr() const { return probdofs_; }

    //! Return underlying problem node row map (not only interfaces)
    std::shared_ptr<Epetra_Map>& prob_nodes_ptr() { return probnodes_; }
    std::shared_ptr<const Epetra_Map> prob_nodes_ptr() const { return probnodes_; }

    //! Return communicator
    MPI_Comm& comm_ptr() { return comm_; }
    MPI_Comm comm_ptr() const { return comm_; }

    //! Return containing contact input parameters
    Teuchos::ParameterList& s_contact() { return scontact_; }
    const Teuchos::ParameterList& s_contact() const { return scontact_; }

    //! Return dimension of problem (2D or 3D)
    int& n_dim() { return dim_; }
    const int& n_dim() const { return dim_; }

    //! Return generalized-alpha parameter (0.0 for statics)
    double& alpha_f() { return alphaf_; }
    const double& alpha_f() const { return alphaf_; }

    /// get the (dynamic) time integration type
    inline Inpar::Solid::DynamicType get_dyn_type() const { return dyntype_; }

    /// return dynamic time integration parameter
    inline double get_dyn_parameter_n() const { return dynparam_n_; }

    /// set dynamic time integration parameter
    inline void set_dyn_parameter_n(const double dynparamN) { dynparam_n_ = dynparamN; }

    /// set the (dynamic) time integration type
    inline void set_dyn_type(Inpar::Solid::DynamicType dyntype) { dyntype_ = dyntype; }

    //! Return flag indicating parallel redistribution status
    bool& is_par_redist() { return parredist_; }
    const bool& is_par_redist() const { return parredist_; }

    //! Return highest dof number in problem discretization
    int& max_dof() { return maxdof_; }
    const int& max_dof() const { return maxdof_; }

    //! Return current used system type
    Inpar::CONTACT::SystemType& sys_type() { return systype_; }
    const Inpar::CONTACT::SystemType& sys_type() const { return systype_; }

   private:
    //! Underlying problem dof row map (not only interfaces)
    std::shared_ptr<Epetra_Map> probdofs_;

    //! Underlying problem node row map (not only interfaces)
    std::shared_ptr<Epetra_Map> probnodes_;

    //! Communicator
    MPI_Comm comm_;

    //! Containing contact input parameters
    Teuchos::ParameterList scontact_;

    //! Dimension of problem (2D or 3D)
    int dim_;

    //! Generalized-alpha parameter (0.0 for statics)
    double alphaf_;

    //! Flag indicating parallel redistribution status
    bool parredist_;

    //! Highest dof number in problem discretization
    int maxdof_;

    //! Current used system type
    Inpar::CONTACT::SystemType systype_;

    //! time integration type
    Inpar::Solid::DynamicType dyntype_;

    //! time integration parameter for the contributions of the old/previous time step
    double dynparam_n_;
  };

  /*!
  \brief Abstract base class for mortar solution strategies

  Every specific solution algorithm (e.g. mortar contact with Lagrange multipliers or
  mortar meshtying with penalty method) has to be specified in a corresponding derived
  subclass defining the concrete algorithmic steps.

  */
  class StrategyBase : public NOX::Nln::CONSTRAINT::Interface::Preconditioner
  {
   public:
    //! @name Enums and Friends
    //! @{
    // can be called by store_nodal_quantities() or StoreDMtoNodes()
    enum QuantityType
    {
      lmcurrent,  //!< current lagr. mult.
      lmold,      //!< lagr. mult. for last converged state
      lmupdate,   //!< update current lagr. mult. (same as for lmcurrent + DBC check)
      lmuzawa,    //!< lagr. mutl. from last Uzawa step
      activeold,  //!< contact status of last converged state
      slipold,    //!< slip for last converged state
      dm,
      pentrac,
      weightedwear,  //!< weighted wear (internal state var. approach)
      wupdate,       //!< update current pv wear for current step (slave)
      wmupdate,      //!< update current pv wear for current step (master)
      wold,          //!< pv wear for last converged state (slave)
      wmold,         //!< pv wear for last converged state (master)
      wupdateT,      //!< accumulated pv wear for different time scales
      lmThermo,      //!< thermal Lagrange multiplier
      n_old          //!< old normal
    };
    //! @}

    /*!
    \brief Standard Constructor

    Creates the strategy base object and initializes all global variables.

    \param data_ptr (in): data container object
    \param dofrowmap (in): dofrowmap of underlying problem
    \param noderowmap (in): noderowmap of underlying problem
    \param elementrowmap (in): elementrowmap of underlying problem
    \param params (in): List of meshtying/contact parameters
    \param spatialDim (in): Global problem dimension
    \param comm (in): A communicator object
    \param alphaf (in): Midpoint for Gen-alpha time integration
    \param maxdof (in): Highest dof number in global problem

    */
    StrategyBase(const std::shared_ptr<Mortar::StrategyDataContainer>& data_ptr,
        const Epetra_Map* dof_row_map, const Epetra_Map* NodeRowMap,
        const Teuchos::ParameterList& params, const int spatialDim, const MPI_Comm& comm,
        const double alphaf, const int maxdof);

    //! @name Access methods
    //! @{
    //! Get parameter list
    Teuchos::ParameterList& params() { return scontact_; }
    const Teuchos::ParameterList& params() const { return scontact_; }

    //! return the current system type
    const Inpar::CONTACT::SystemType& system_type() const { return systype_; }

    //! Get problem dimension
    int n_dim() const { return dim_; }

    //! Get Epetra communicator
    MPI_Comm get_comm() const { return comm_; }

    //! Get the underlying problem dof row map
    const std::shared_ptr<Epetra_Map>& problem_dofs() { return probdofs_; }
    std::shared_ptr<const Epetra_Map> problem_dofs() const { return probdofs_; }

    //! Get the underlying problem node row map
    const std::shared_ptr<Epetra_Map>& problem_nodes() { return probnodes_; }
    std::shared_ptr<const Epetra_Map> problem_nodes() const { return probnodes_; }

    //@}

    /// Set the time integration information
    void set_time_integration_info(const double time_fac, const Inpar::Solid::DynamicType dyntype);

    //! @name Purely virtual functions

    // All these functions are defined in one or more specific derived classes,
    // such as CONTACT::ContactLagrangeStrategy or CONTACT::MeshtyingPenaltyStrategy.
    // As the base class Mortar::StrategyBase is always called from the control routine
    // (time integrator), these functions need to be defined purely virtual here.

    virtual std::shared_ptr<const Epetra_Map> slave_row_nodes_ptr() const = 0;
    virtual std::shared_ptr<const Epetra_Map> active_row_nodes() const = 0;
    virtual std::shared_ptr<const Epetra_Map> active_row_dofs() const = 0;
    virtual std::shared_ptr<const Epetra_Map> non_redist_slave_row_dofs() const = 0;
    virtual std::shared_ptr<const Epetra_Map> non_redist_master_row_dofs() const = 0;
    virtual bool active_set_converged() const = 0;
    virtual void apply_force_stiff_cmt(std::shared_ptr<Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<Core::LinAlg::SparseOperator>& kt,
        std::shared_ptr<Core::LinAlg::Vector<double>>& f, const int step, const int iter,
        bool predictor = false) = 0;
    virtual void assemble_mortar() = 0;
    virtual void collect_maps_for_preconditioner(std::shared_ptr<Epetra_Map>& MasterDofMap,
        std::shared_ptr<Epetra_Map>& SlaveDofMap, std::shared_ptr<Epetra_Map>& InnerDofMap,
        std::shared_ptr<Epetra_Map>& ActiveDofMap) const = 0;
    virtual double constraint_norm() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> contact_normal_stress() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> contact_tangential_stress()
        const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> contact_normal_force() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> contact_tangential_force()
        const = 0;
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> d_matrix() const = 0;
    virtual void do_read_restart(Core::IO::DiscretizationReader& reader,
        std::shared_ptr<const Core::LinAlg::Vector<double>> dis) = 0;
    virtual void do_write_restart(
        std::map<std::string, std::shared_ptr<Core::LinAlg::Vector<double>>>& restart_vectors,
        bool forcedrestart = false) const = 0;
    virtual void evaluate(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) = 0;
    virtual void evaluate_meshtying(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff,
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) = 0;
    virtual std::shared_ptr<Core::LinAlg::SparseMatrix> evaluate_normals(
        std::shared_ptr<Core::LinAlg::Vector<double>> dis) = 0;
    virtual void evaluate_reference_state() = 0;
    virtual void evaluate_relative_movement() = 0;
    virtual void predict_relative_movement() = 0;
    virtual bool is_friction() const = 0;
    virtual void initialize_and_evaluate_interface() = 0;
    virtual void initialize_mortar() = 0;
    virtual void initialize() = 0;
    virtual void initialize_uzawa(std::shared_ptr<Core::LinAlg::SparseOperator>& kteff,
        std::shared_ptr<Core::LinAlg::Vector<double>>& feff) = 0;
    virtual double initial_penalty() const = 0;
    virtual double inttime() const = 0;
    virtual void inttime_init() = 0;
    virtual bool is_in_contact() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_old() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> constraint_rhs() const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> lagrange_multiplier_increment()
        const = 0;
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> mesh_initialization() = 0;
    virtual std::shared_ptr<const Core::LinAlg::SparseMatrix> m_matrix() const = 0;
    virtual void mortar_coupling(
        const std::shared_ptr<const Core::LinAlg::Vector<double>>& dis) = 0;
    virtual int number_of_active_nodes() const = 0;
    virtual int number_of_slip_nodes() const = 0;
    virtual void compute_contact_stresses() = 0;

    /*!
    \brief Write results for visualization separately for each meshtying/contact interface

    Call each interface, such that each interface can handle its own output of results.

    \param[in] outputParams Parameter list with stuff required by interfaces to write output
    */
    virtual void postprocess_quantities_per_interface(
        std::shared_ptr<Teuchos::ParameterList> outputParams) const = 0;

    virtual void print(std::ostream& os) const = 0;
    virtual void print_active_set() const = 0;
    virtual void recover(std::shared_ptr<Core::LinAlg::Vector<double>> disi) = 0;
    virtual bool redistribute_contact(std::shared_ptr<const Core::LinAlg::Vector<double>> dis,
        std::shared_ptr<const Core::LinAlg::Vector<double>> vel) = 0;
    virtual void redistribute_meshtying() = 0;
    virtual void reset_active_set() = 0;
    virtual void reset_penalty() = 0;
    virtual void modify_penalty() = 0;
    virtual void restrict_meshtying_zone() = 0;
    virtual void build_saddle_point_system(std::shared_ptr<Core::LinAlg::SparseOperator> kdd,
        std::shared_ptr<Core::LinAlg::Vector<double>> fd,
        std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<Core::LinAlg::MapExtractor> dbcmaps,
        std::shared_ptr<Epetra_Operator>& blockMat,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blocksol,
        std::shared_ptr<Core::LinAlg::Vector<double>>& blockrhs) = 0;
    virtual void update_displacements_and_l_mincrements(
        std::shared_ptr<Core::LinAlg::Vector<double>> sold,
        std::shared_ptr<const Core::LinAlg::Vector<double>> blocksol) = 0;
    virtual void save_reference_state(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) = 0;
    virtual void set_state(
        const enum Mortar::StateType& statename, const Core::LinAlg::Vector<double>& vec) = 0;
    virtual std::shared_ptr<const Epetra_Map> slip_row_nodes() const = 0;
    virtual void store_dirichlet_status(
        std::shared_ptr<const Core::LinAlg::MapExtractor> dbcmaps) = 0;
    virtual void store_nodal_quantities(Mortar::StrategyBase::QuantityType type) = 0;
    virtual void update(std::shared_ptr<const Core::LinAlg::Vector<double>> dis) = 0;
    virtual void update_active_set() = 0;
    virtual void update_active_set_semi_smooth(const bool firstStepPredictor = false) = 0;
    virtual void update_uzawa_augmented_lagrange() = 0;
    virtual void update_constraint_norm(int uzawaiter = 0) = 0;
    virtual bool was_in_contact() const = 0;
    virtual bool was_in_contact_last_time_step() const = 0;

    // Flag for Poro No Penetration Condition (overloaded by LagrangeStrategyPoro)
    virtual bool has_poro_no_penetration() const { return false; }

    // Nitsche stuff
    virtual bool is_nitsche() const { return false; }

    // wear stuff
    virtual bool weighted_wear() const { return false; }
    virtual bool wear_both_discrete() const { return false; }
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> wear_rhs() const { return nullptr; }
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> wear_m_rhs() const
    {
      return nullptr;
    }
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> w_solve_incr() const
    {
      return nullptr;
    }
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> wm_solve_incr() const
    {
      return nullptr;
    }
    virtual std::shared_ptr<const Core::LinAlg::Vector<double>> contact_wear() const
    {
      return nullptr;
    }
    virtual void reset_wear() {}
    virtual void output_wear() {}
    virtual std::shared_ptr<const Epetra_Map> master_slip_nodes() const { return nullptr; }
    virtual std::shared_ptr<const Epetra_Map> master_active_nodes() const { return nullptr; }

    // constraint preconditioner functions
    bool is_saddle_point_system() const override = 0;
    bool is_condensed_system() const override = 0;
    virtual bool is_penalty() const = 0;
    void fill_maps_for_preconditioner(
        std::vector<Teuchos::RCP<Epetra_Map>>& maps) const override = 0;
    //@}

   private:
    /*! return the mutable mortar data container
     *
     * \remark This has to stay PRIVATE, otherwise this function becomes ambiguous.
     *
     */
    Mortar::StrategyDataContainer& data() { return *data_ptr_; }

   public:
    /*! return the read-only mortar data container
     *
     * \remark This has to stay PRIVATE, otherwise this function becomes ambiguous.
     *
     */
    const Mortar::StrategyDataContainer& data() const { return *data_ptr_; }

   protected:
    // don't want cctor (= operator impossible anyway for abstract class)
    StrategyBase(const StrategyBase& old);

    /*! @name References to the data container content
     *
     * \remark Please add no new member variables to the strategy base! Use
     *  the corresponding data container instead (--> Mortar::StrategyDataContainer).
     *  If you have any questions concerning this, do not hesitate and ask me.
     *                                                          hiermeier 05/16 */
    //! @{
    std::shared_ptr<Epetra_Map>&
        probdofs_;  //!< ref. to underlying problem dof row map (not only interfaces)
    std::shared_ptr<Epetra_Map>&
        probnodes_;  //!< ref. to underlying problem node row map (not only interfaces)

    MPI_Comm& comm_;                    //!< ref. to communicator
    Teuchos::ParameterList& scontact_;  //!< ref. to containing contact input parameters
    int& dim_;                          //!< ref. to dimension of problem (2D or 3D)
    double& alphaf_;                    //!< ref. to generalized-alpha parameter (0.0 for statics)
    bool& parredist_;                   //!< ref. to flag indicating parallel redistribution status
    int& maxdof_;                       //!< ref. to highest dof number in problem discretization
    Inpar::CONTACT::SystemType& systype_;  //!< ref. to current used system type
                                           //! @}

   private:
    //! pointer to the data container object
    std::shared_ptr<Mortar::StrategyDataContainer> data_ptr_;

  };  // class StrategyBase
}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
