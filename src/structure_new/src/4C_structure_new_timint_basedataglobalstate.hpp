// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAGLOBALSTATE_HPP
#define FOUR_C_STRUCTURE_NEW_TIMINT_BASEDATAGLOBALSTATE_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"
#include "4C_linalg_mapextractor.hpp"
#include "4C_linalg_sparseoperator.hpp"
#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"
#include "4C_solver_nonlin_nox_enum_lists.hpp"
#include "4C_structure_new_enum_lists.hpp"
#include "4C_timestepping_mstep.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

namespace Teuchos
{
  class Time;
}
namespace NOX
{
  namespace Epetra
  {
    class Vector;
  }  // namespace Epetra
}  // namespace NOX

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class Beam3Base;
  }  // namespace Elements
}  // namespace Discret

namespace Core::LinAlg
{
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Solid
{
  class ModelEvaluatorManager;
  namespace ModelEvaluator
  {
    class Generic;
  }  // namespace ModelEvaluator
  namespace TimeInt
  {
    class BaseDataSDyn;

    /** \brief Global state data container for the structural (time) integration
     *
     * This data container holds everything, which refers directly to the
     * structural problem state, e.g. current step counter, time, forces, displacements,
     * velocities, accelerations, mass matrix, damping matrix, and the entire
     * jacobian (incl. the constraint blocks, if a saddle point system should be
     * solved).
     *
     * */
    class BaseDataGlobalState
    {
     public:
      /// enum, which specifies the desired global vector initialization during creation
      enum class VecInitType
      {
        zero,               ///< fill the vector with zeros
        last_time_step,     ///< use the last converged time step state
        init_current_state  ///< use the current state
      };

      /// constructor
      BaseDataGlobalState();

      /// destructor
      virtual ~BaseDataGlobalState() = default;

      /** \brief copy the init information only and set the issetup flag to false
       *
       */
      virtual BaseDataGlobalState& operator=(const BaseDataGlobalState& source);

      /*! \brief Initialize class variables
       *
       * @param discret discretization object
       * @param sdynparams Parameter list for structural dynamics from input file
       * @param datasdyn Structural dynamics data container
       */
      void init(const std::shared_ptr<Core::FE::Discretization> discret,
          const Teuchos::ParameterList& sdynparams,
          const std::shared_ptr<const BaseDataSDyn> datasdyn);

      /// setup of the new class variables
      virtual void setup();

      /// read initial field conditions
      void set_initial_fields();

      /*! \brief Setup blocking of linear system & vectors
       *
       * Depending on the actual model, the linear system will exhibit a block structure,
       * e.g. when adding imposing constraints like in contact problems.
       * Here, we select and set a suitable blocking for each problem type by considering
       * input data related to model, discretization, and solution strategy.
       *
       * @param[in] me Model evaluator
       * @param[in] mt Model type
       *
       * @return Max GID in the entire problem
       */
      int setup_block_information(
          const Solid::ModelEvaluator::Generic& me, const Inpar::Solid::ModelType& mt);

      /// setup the multi map extractor for saddle point problems
      void setup_multi_map_extractor();

      /// setup the map extractors for all active element technologies
      void setup_element_technology_map_extractors();

      /*! \brief Return map extractor for element technology
       *
       * @param[in] etech Type of element technology that is queried
       *
       * @return MultiMapExtractor for the required type of element technology
       */
      const Core::LinAlg::MultiMapExtractor& get_element_technology_map_extractor(
          const Inpar::Solid::EleTech etech) const;

      /** setup the map extractor for translational <-> rotation pseudo-vector DoFs
       *                              (additive)    <->  (non-additive)      */
      void setup_rot_vec_map_extractor(Core::LinAlg::MultiMapExtractor& multimapext);

      /*! \brief Extract the part of a vector which belongs to the displacement dofs.
       *
       * \todo ToDo "displacement dofs" might be misleading, since this could also be applied to
       * extract velocities of those DOFs associated with translations.
       *
       * \param source (in) : full vector to extract from. */
      std::shared_ptr<Core::LinAlg::Vector<double>> extract_displ_entries(
          const Core::LinAlg::Vector<double>& source) const;

      /*! \brief Extract the part of a vector which belongs to the model dofs.
       *
       * \param mt (in)     : model type of the desired block.
       * \param source (in) : full vector to extract from. */
      std::shared_ptr<Core::LinAlg::Vector<double>> extract_model_entries(
          const Inpar::Solid::ModelType& mt, const Core::LinAlg::Vector<double>& source) const;

      /* \brief Extract the part of a vector which belongs to non-additive rotation
       * (pseudo-)vector dofs.
       *
       * \param source (in) : full vector to extract from. */
      std::shared_ptr<Core::LinAlg::Vector<double>> extract_rot_vec_entries(
          const Core::LinAlg::Vector<double>& source) const;

      /** \brief Read-only access of the desired block of the global jacobian
       *  matrix in the global state data container.
       *
       *  \param mt (in)  : Model type of the desired block.
       *  \param bt (in)  : Desired matrix block type.
       *
       *  */
      std::shared_ptr<const Core::LinAlg::SparseMatrix> get_jacobian_block(
          const Inpar::Solid::ModelType mt, const MatBlockType bt) const;

      /// Get the block of the stiffness matrix which belongs to the displacement dofs.
      std::shared_ptr<Core::LinAlg::SparseMatrix> extract_displ_block(
          Core::LinAlg::SparseOperator& jac) const;

      /* \brief Get the block of the desired model which belongs to the given block type.
       *
       * \param jac (in) : Full jacobian to extract from.
       * \param mt (in)  : Model type of the desired block.
       * \param bt (in)  : Desired matrix block type.  */
      std::shared_ptr<Core::LinAlg::SparseMatrix> extract_model_block(
          Core::LinAlg::SparseOperator& jac, const Inpar::Solid::ModelType& mt,
          const MatBlockType& bt) const;

      std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>> extract_displ_row_of_blocks(
          Core::LinAlg::SparseOperator& jac) const;

      std::shared_ptr<std::vector<Core::LinAlg::SparseMatrix*>> extract_row_of_blocks(
          Core::LinAlg::SparseOperator& jac, const Inpar::Solid::ModelType& mt) const;

      /** \brief Assign a Core::LinAlg::SparseMatrix to one of the blocks of the corresponding
       * model
       *
       *  You can choose between one of the following blocks
       *
       *          ===       ===
       *         | DD     DLm  |
       *         |             |
       *         | LmD    LmLm |
       *          ===       ===     */
      void assign_model_block(Core::LinAlg::SparseOperator& jac,
          const Core::LinAlg::SparseMatrix& matrix, const Inpar::Solid::ModelType& mt,
          const MatBlockType& bt) const
      {
        assign_model_block(jac, matrix, mt, bt, Core::LinAlg::DataAccess::View);
      };
      void assign_model_block(Core::LinAlg::SparseOperator& jac,
          const Core::LinAlg::SparseMatrix& matrix, const Inpar::Solid::ModelType& mt,
          const MatBlockType& bt, const Core::LinAlg::DataAccess& access) const;

      /// Get the displacement block of the global jacobian matrix in the global
      /// state data container.
      std::shared_ptr<const Core::LinAlg::SparseMatrix> get_jacobian_displ_block() const;

      /// Get the displacement block of the global jacobian matrix in the global
      /// state data container.
      std::shared_ptr<Core::LinAlg::SparseMatrix> jacobian_displ_block();

      /// Create the global solution vector
      std::shared_ptr<::NOX::Epetra::Vector> create_global_vector() const;
      std::shared_ptr<::NOX::Epetra::Vector> create_global_vector(
          const enum VecInitType& vecinittype,
          const std::shared_ptr<const Solid::ModelEvaluatorManager>& modeleval) const;

      /// Create the structural stiffness matrix block
      Core::LinAlg::SparseOperator* create_structural_stiffness_matrix_block();

      /// Create the jacobian matrix
      std::shared_ptr<Core::LinAlg::SparseOperator>& create_jacobian();

      std::shared_ptr<Core::LinAlg::SparseOperator> create_aux_jacobian() const;

     protected:
      inline const bool& is_init() const { return isinit_; };

      inline const bool& is_setup() const { return issetup_; };

      inline void check_init_setup() const
      {
        FOUR_C_ASSERT(
            is_init() and is_setup(), "Call Solid::BaseDataGlobalState::init() and setup() first!");
      }

      inline void check_init() const
      {
        FOUR_C_ASSERT(is_init(), "Solid::BaseDataGlobalState::init() has not been called, yet!");
      }

     public:
      /// @name Get general purpose algorithm members (read only access)
      ///@{

      //! return the dimension of the problem
      [[nodiscard]] unsigned int get_dim() const { return dim_; }

      /// attached discretisation
      std::shared_ptr<const Core::FE::Discretization> get_discret() const
      {
        check_init();
        return discret_;
      };

      /// communicator
      MPI_Comm get_comm_ptr() const
      {
        check_init();
        return comm_;
      };

      MPI_Comm get_comm() const
      {
        check_init();
        return comm_;
      };

      /// ID of actual processor in parallel
      const int& get_my_rank() const
      {
        check_init();
        return my_rank_;
      };

      ///@}

      /// @name Get discretization related stuff (read only access)
      ///@{

      /// dof map of vector of unknowns
      virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map() const;

      /// dof map of vector of unknowns
      /// method for multiple dofsets
      virtual std::shared_ptr<const Core::LinAlg::Map> dof_row_map(unsigned nds) const;

      /// view of dof map of vector of unknowns
      virtual const Core::LinAlg::Map* dof_row_map_view() const;

      /// view of dof map of vector of additive unknowns
      /* in case we have non-additve DoFs in the structure discretization
       * (e.g. rotation vector DoFs of beams), this method is overloaded */
      const Core::LinAlg::Map* additive_dof_row_map_view() const;

      /// view of dof map of vector of rotation vector unknowns
      /* (e.g. rotation vector DoFs of beams), this method is overloaded */
      const Core::LinAlg::Map* rot_vec_dof_row_map_view() const;

      ///@}

      /// @name Get general control parameters (read only access)
      ///@{

      /// Return target time \f$t_{n+1}\f$
      const double& get_time_np() const
      {
        check_init();
        return timenp_;
      };

      /// Return time \f$t_{n}\f$ of last converged step
      const double& get_time_n() const
      {
        check_init();
        return timen_[0];
      };

      /// Return time vector \f$t_{n}, t_{n-1}, ...\f$ of last converged steps
      const TimeStepping::TimIntMStep<double>& get_multi_time() const
      {
        check_init();
        return timen_;
      };

      /// Return time step index for \f$t_{n+1}\f$
      const int& get_step_np() const
      {
        check_init();
        return stepnp_;
      };

      /// Return time step index for \f$t_{n}\f$
      const int& get_step_n() const
      {
        check_init();
        return stepn_;
      };

      /// Return the restart step
      int get_restart_step() const
      {
        check_init();
        return restartstep_;
      }

      /// Get the last number of linear iterations of the %step
      int get_last_lin_iteration_number(const unsigned step) const;

      /// Get the number of non-linear iterations of the %step
      int get_nln_iteration_number(const unsigned step) const;

      /// Return time for lin solver
      double get_linear_solver_time() const
      {
        check_init_setup();
        return dtsolve_;
      };

      /// Return element evaluation time
      double get_element_evaluation_time() const
      {
        check_init_setup();
        return dtele_;
      };

      /// Return time step size \f$\Delta t\f$
      const TimeStepping::TimIntMStep<double>& get_delta_time() const
      {
        check_init();
        return dt_;
      };

      /// Return timer for solution technique
      std::shared_ptr<const Teuchos::Time> get_timer() const
      {
        check_init_setup();
        return timer_;
      };

      /// returns the prediction indicator
      const bool& is_predict() const
      {
        check_init_setup();
        return ispredict_;
      };
      ///@}

      /// @name Get state variables (read only access)
      ///@{

      /// Return displacements \f$D_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_dis_np() const
      {
        check_init_setup();
        return disnp_;
      }

      /// Return displacements \f$D_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_dis_n() const
      {
        check_init_setup();
        return dis_(0);
      }

      /// Return velocities \f$V_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_vel_np() const
      {
        check_init_setup();
        return velnp_;
      }

      /// Return velocities \f$V_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_vel_n() const
      {
        check_init_setup();
        return vel_(0);
      }

      /// Return velocities \f$V_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_vel_nm() const
      {
        check_init_setup();
        return vel_(-1);
      }

      /// Return accelerations \f$A_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_acc_np() const
      {
        check_init_setup();
        return accnp_;
      }

      /// Return accelerations \f$A_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_acc_n() const
      {
        check_init_setup();
        return acc_(0);
      }

      /// Return internal force \f$fint_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fint_n() const
      {
        check_init_setup();
        return fintn_;
      }

      /// Return internal force \f$fint_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fint_np() const
      {
        check_init_setup();
        return fintnp_;
      }

      /// Return external force \f$fext_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fext_n() const
      {
        check_init_setup();
        return fextn_;
      }

      /// Return external force \f$fext_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fext_np() const
      {
        check_init_setup();
        return fextnp_;
      }

      /// Return reaction force \f$freact_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_freact_n() const
      {
        check_init_setup();
        return freactn_;
      }

      /// Return reaction force \f$freact_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_freact_np() const
      {
        check_init_setup();
        return freactnp_;
      }

      /// Return inertia force \f$finertial_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_finertial_n() const
      {
        check_init_setup();
        return finertialn_;
      }

      /// Return inertial force \f$finertial_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_finertial_np() const
      {
        check_init_setup();
        return finertialnp_;
      }

      /// Return visco force \f$fvisco_{n}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fvisco_n() const
      {
        check_init_setup();
        return fviscon_;
      }

      /// Return visco force \f$fvisco_{n+1}\f$
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fvisco_np() const
      {
        check_init_setup();
        return fvisconp_;
      }


      /** \brief Return entire force \f$fstructure_{old}\f$
       *
       *  Please note that this old structural residual is already scaled by the
       *  different time integration factors! */
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_fstructure_old() const
      {
        check_init_setup();
        return fstructold_;
      }
      ///@}

      /// @name Get system matrices (read only access)
      ///@{
      /// returns the entire structural jacobian
      std::shared_ptr<const Core::LinAlg::SparseOperator> get_jacobian() const
      {
        check_init_setup();
        return jac_;
      }

      /// mass matrix (constant)
      std::shared_ptr<const Core::LinAlg::SparseOperator> get_mass_matrix() const
      {
        check_init_setup();
        return mass_;
      }

      /// damping matrix
      std::shared_ptr<const Core::LinAlg::SparseOperator> get_damp_matrix() const
      {
        check_init_setup();
        return damp_;
      }
      ///@}

      /// @name Get general purpose algorithm members (read only access)
      ///@{
      /// attached discretization
      std::shared_ptr<Core::FE::Discretization> get_discret()
      {
        check_init();
        return discret_;
      };

      ///@}

      /// @name Access saddle-point system information
      /// @{

      /** \brief Returns Core::LinAlg::Map pointer of the given model
       *
       *  If the given model is not found, nullptr is returned. */
      std::shared_ptr<const Core::LinAlg::Map> block_map_ptr(
          const Inpar::Solid::ModelType& mt) const
      {
        if (model_maps_.find(mt) != model_maps_.end()) return model_maps_.at(mt);

        return nullptr;
      };

      /// Returns Core::LinAlg::Map of the given model
      Core::LinAlg::Map block_map(const Inpar::Solid::ModelType& mt) const;

      /** \brief Returns the Block id of the given model type.
       *
       *  If the block is not found, -1 is returned. */
      int block_id(const enum Inpar::Solid::ModelType& mt) const
      {
        if (model_block_id_.find(mt) != model_block_id_.end()) return model_block_id_.at(mt);

        return -1;
      };

      /// Returns the maximal block number
      int max_block_number() const
      {
        check_init_setup();
        return max_block_num_;
      };

      /// Returns global problem map pointer
      std::shared_ptr<const Core::LinAlg::Map> global_problem_map_ptr() const
      {
        return gproblem_map_ptr_;
      };

      /// Returns global problem map
      const Core::LinAlg::Map& global_problem_map() const
      {
        FOUR_C_ASSERT(gproblem_map_ptr_, "The global problem map is not defined!");
        return *gproblem_map_ptr_;
      };

      const Core::LinAlg::MultiMapExtractor& block_extractor() const;

      /// @}

      /// @name Get mutable general control parameters (read and write access)
      ///@{

      /// Return target time \f$t_{n+1}\f$
      double& get_time_np()
      {
        check_init();
        return timenp_;
      };

      /// Return time \f$t_{n}\f$ of last converged step
      double& get_time_n()
      {
        check_init();
        return timen_[0];
      };

      /// Return time \f$t_{n}, t_{n-1}, ...\f$ of last converged steps
      TimeStepping::TimIntMStep<double>& get_multi_time()
      {
        check_init();
        return timen_;
      };

      /// Return time step index for \f$t_{n+1}\f$
      int& get_step_np()
      {
        check_init();
        return stepnp_;
      };

      /// Return time step index for \f$t_{n}\f$
      int& get_step_n()
      {
        check_init();
        return stepn_;
      };

      /// Set the number of non-linear iterations of the #stepn_
      void set_nln_iteration_number(const int nln_iter);

      /// Return time for linear solver
      double& get_linear_solver_time()
      {
        check_init_setup();
        return dtsolve_;
      };

      /// Return element evaluation time
      double& get_element_evaluation_time()
      {
        check_init_setup();
        return dtele_;
      };

      /// Return time step size \f$\Delta t\f$
      TimeStepping::TimIntMStep<double>& get_delta_time()
      {
        check_init();
        return dt_;
      };

      /// Return the prediction indicator
      bool& is_predict()
      {
        check_init_setup();
        return ispredict_;
      }
      ///@}

      /// @name Get mutable state variables (read and write access)
      ///@{

      /// Return displacements \f$D_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_dis_np()
      {
        check_init_setup();
        return disnp_;
      }

      /// Return displacements \f$D_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> get_dis_n()
      {
        check_init_setup();
        return dis_(0);
      }

      /// Return multi-displacement vector \f$D_{n}, D_{n-1}, ...\f$
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>& get_multi_dis()
      {
        check_init_setup();
        return dis_;
      }

      /// Return velocities \f$V_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_vel_np()
      {
        check_init_setup();
        return velnp_;
      }

      /// Return velocities \f$V_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> get_vel_n()
      {
        check_init_setup();
        return vel_(0);
      }

      /// Return multi-velocity vector \f$V_{n}, V_{n-1}, ...\f$
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>& get_multi_vel()
      {
        check_init_setup();
        return vel_;
      }

      /// Return multi-velocity vector \f$V_{n}, V_{n-1}, ...\f$
      const TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>& get_multi_vel() const
      {
        check_init_setup();
        return vel_;
      }

      /// Return accelerations \f$A_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_acc_np()
      {
        check_init_setup();
        return accnp_;
      }

      /// Return accelerations \f$A_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> get_acc_n()
      {
        check_init_setup();
        return acc_(0);
      }

      /// Return multi-acceleration vector \f$A_{n}, A_{n-1}, ...\f$
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>& get_multi_acc()
      {
        check_init_setup();
        return acc_;
      }

      /// Return multi-acceleration vector \f$A_{n}, A_{n-1}, ...\f$
      const TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>>& get_multi_acc() const
      {
        check_init_setup();
        return acc_;
      }

      /// Return internal force \f$fint_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fint_n()
      {
        check_init_setup();
        return fintn_;
      }

      /// Return internal force \f$fint_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fint_np()
      {
        check_init_setup();
        return fintnp_;
      }

      /// Return external force \f$fext_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fext_n()
      {
        check_init_setup();
        return fextn_;
      }

      /// Return external force \f$fext_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fext_np()
      {
        check_init_setup();
        return fextnp_;
      }

      /// Return reaction force \f$freact_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_freact_n()
      {
        check_init_setup();
        return freactn_;
      }

      /// Return reaction force \f$freact_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_freact_np()
      {
        check_init_setup();
        return freactnp_;
      }

      /// Return inertia force \f$finertial_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_finertial_n()
      {
        check_init_setup();
        return finertialn_;
      }

      /// Return inertial force \f$finertial_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_finertial_np()
      {
        check_init_setup();
        return finertialnp_;
      }

      /// Return viscous force \f$f_{viscous,n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fvisco_n()
      {
        check_init_setup();
        return fviscon_;
      }

      /// Return viscous force \f$fviscous_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fvisco_np()
      {
        check_init_setup();
        return fvisconp_;
      }

      /** \brief Return entire force \f$fstructure_{old}\f$
       *
       *  Please note that this old structural residual is already scaled by the
       *  different time integration factors! */
      std::shared_ptr<Core::LinAlg::Vector<double>>& get_fstructure_old()
      {
        check_init_setup();
        return fstructold_;
      }

      ///@}

      /// @name Get mutable system matrices
      ///@{
      /// returns the entire structural jacobian
      std::shared_ptr<Core::LinAlg::SparseOperator>& get_jacobian()
      {
        check_init_setup();
        return jac_;
      }

      /// mass matrix (constant)
      std::shared_ptr<Core::LinAlg::SparseOperator>& get_mass_matrix()
      {
        check_init_setup();
        return mass_;
      }

      /// damping matrix
      std::shared_ptr<Core::LinAlg::SparseOperator>& get_damp_matrix()
      {
        check_init_setup();
        return damp_;
      }
      ///@}

     protected:
      /// mutable access to the global problem map
      std::shared_ptr<Core::LinAlg::Map>& global_problem_map_ptr() { return gproblem_map_ptr_; }

      /** \brief mutable access to the structural stiffness member variable [PROTECTED ONLY]
       *
       *  Do NOT change this to PUBLIC! Use the ExtractMatrixBlock() function
       *  instead.
       *

       *  */
      std::shared_ptr<Core::LinAlg::SparseOperator>& stiff_ptr() { return stiff_; }

     protected:
      /// @name variables for internal use only
      ///@{
      /// flag indicating if init() has been called
      bool isinit_;

      /// flag indicating if setup() has been called
      bool issetup_;

      /// read only access
      std::shared_ptr<const BaseDataSDyn> datasdyn_;
      ///@}

     private:
      /// @name General purpose algorithm members
      ///@{

      //! dimension of the problem
      const unsigned int dim_;

      /// attached discretisation
      std::shared_ptr<Core::FE::Discretization> discret_;

      /// communicator
      MPI_Comm comm_;

      /// ID of actual processor in parallel
      int my_rank_;

      ///@}

      /// @name General control parameters
      ///@{
      /// target time \f$t_{n+1}\f$
      double timenp_;

      /// time \f$t_{n}\f$ of last converged step
      TimeStepping::TimIntMStep<double> timen_;

      /// time step size \f$\Delta t\f$
      TimeStepping::TimIntMStep<double> dt_;

      /// time step index \f$n\f$
      int stepn_;

      /// time step index \f$n+1\f$
      int stepnp_;

      /** step number from which the current simulation has been restarted. If
       *  no restart has been performed, zero is returned. */
      int restartstep_;

      /// pairs of (step ID, number of nonlinear iteration in this step)
      std::vector<std::pair<int, int>> nln_iter_numbers_;

      /// A new time step started and we predict the new solution
      bool ispredict_;
      ///@}

      /// @name Global state vectors
      ///@{

      /// global displacements \f${D}_{n}, D_{n-1}, ...\f$
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>> dis_;

      /// global velocities \f${V}_{n}, V_{n-1}, ...\f$
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>> vel_;

      /// global accelerations \f${A}_{n}, A_{n-1}, ...\f$
      TimeStepping::TimIntMStep<Core::LinAlg::Vector<double>> acc_;

      /// global displacements \f${D}_{n+1}\f$ at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> disnp_;

      /// global velocities \f${V}_{n+1}\f$ at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> velnp_;

      /// global accelerations \f${A}_{n+1}\f$ at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> accnp_;

      /// global internal force vector at \f$t_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fintn_;

      /// global internal force vector at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fintnp_;

      /// global external force vector at \f$t_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fextn_;

      /// global external force vector at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fextnp_;

      /// global reaction force vector at \f$t_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> freactn_;

      /// global reaction force vector at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> freactnp_;

      /// global inertial force vector at \f$t_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> finertialn_;

      /// global inertial force vector at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> finertialnp_;

      /// global viscous force vector at \f$t_{n}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fviscon_;

      /// global viscous force vector at \f$t_{n+1}\f$
      std::shared_ptr<Core::LinAlg::Vector<double>> fvisconp_;

      /** \brief dynamic structural right hand side of the previous time step
       *
       *  The vector fstructold holds the structural right hand side without dynamic mass and
       * viscous contributions at \f$t_{n + timefac_n}\f$:
       *
       *  f_{struct,n} = a_n * f_{int,n} - a_n * f_{ext,n} + b_n * f_{contact,n} + c_n *
       * f_{cardio,n} ..., where a_n, b_n, c_n represent different time integration factors.
       *
       *  */
      std::shared_ptr<Core::LinAlg::Vector<double>> fstructold_;
      ///@}
      /// @name System matrices
      ///@{
      /// supposed to hold the entire jacobian (saddle point system if desired)
      std::shared_ptr<Core::LinAlg::SparseOperator> jac_;

      /** \brief structural stiffness matrix block
       *
       *  This variable is not allowed to become directly accessible by any public
       *  member function! Only indirect access, e.g. via extract_model_block() or
       *  protected access is allowed!
       *

       *  */
      std::shared_ptr<Core::LinAlg::SparseOperator> stiff_;

      /// mass matrix (constant)
      std::shared_ptr<Core::LinAlg::SparseOperator> mass_;

      /// damping matrix
      std::shared_ptr<Core::LinAlg::SparseOperator> damp_;
      ///@}

      /// @name Time measurement
      ///@{

      /// timer for solution technique
      std::shared_ptr<Teuchos::Time> timer_;

      /// linear solver time
      double dtsolve_;

      /// element evaluation time
      double dtele_;
      ///@}

      /// @name variables to create a saddle-point system
      /// @{

      /// Core::LinAlg::Map s of the different models
      std::map<Inpar::Solid::ModelType, std::shared_ptr<const Core::LinAlg::Map>> model_maps_;

      /// block information for the different models
      std::map<Inpar::Solid::ModelType, int> model_block_id_;

      int max_block_num_;

      /// global problem map
      std::shared_ptr<Core::LinAlg::Map> gproblem_map_ptr_;

      /// multi map extractor
      Core::LinAlg::MultiMapExtractor blockextractor_;

      // all active element technology map extractors
      std::map<Inpar::Solid::EleTech, Core::LinAlg::MultiMapExtractor> mapextractors_;

      /// map extractor for split of translational <-> rotational pseudo-vector DoFs
      Core::LinAlg::MultiMapExtractor rotvecextractor_;

      /// map extractor for structure/pressure coupled problems
      std::shared_ptr<Core::LinAlg::MapExtractor> pressextractor_;
      /// @}
    };  // class BaseDataGlobalState
  }  // namespace TimeInt
}  // namespace Solid

namespace NOX
{
  namespace Nln
  {
    namespace GROUP
    {
      namespace PrePostOp
      {
        namespace TimeInt
        {
          /*! \brief helper class
           *
           *  This class is an implementation of the NOX::Nln::Abstract::PrePostOperator
           *  and is used to modify the computeX() routine of the given NOX::Nln::Group.
           *  It's called by the wrapper class NOX::Nln::GROUP::PrePostOperator. We use it
           *  to update the non-additive rotation (pseudo-)vector DOFs in a consistent
           *  (multiplicative) manner.
           *
           *  */
          class RotVecUpdater : public NOX::Nln::Abstract::PrePostOperator
          {
           public:
            //! constructor
            RotVecUpdater(const std::shared_ptr<const FourC::Solid::TimeInt::BaseDataGlobalState>&
                    gstate_ptr);

            /*! \brief Derived function, which is called before a call to
             * NOX::Nln::Group::computeX()
             *
             *  This method is used to update non-additive rotation vector DoFs */
            void run_pre_compute_x(const NOX::Nln::Group& input_grp,
                const Core::LinAlg::Vector<double>& dir, const double& step,
                const NOX::Nln::Group& curr_grp) override;

           private:
            //! pointer to the FourC::Solid::TimeInt::BaseDataGlobalState object (read-only)
            std::shared_ptr<const FourC::Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

          };  // class RotVecUpdater
        }  // namespace TimeInt
      }  // namespace PrePostOp
    }  // namespace GROUP
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
