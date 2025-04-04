// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_STRUCTURE_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_STRUCTURE_HPP

#include "4C_config.hpp"

#include "4C_inpar_structure.hpp"  // enumerators
#include "4C_io_visualization_parameters.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"  // interface to the element evaluation
#include "4C_structure_new_error_evaluator.hpp"
#include "4C_structure_new_model_evaluator_generic.hpp"  // base class

#include <memory>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::IO
{
  class DiscretizationVisualizationWriterMesh;
}
class BeamDiscretizationRuntimeOutputWriter;

namespace Core::LinAlg
{
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace Solid
{
  namespace ModelEvaluator
  {
    class Structure : public Generic
    {
     public:
      //! constructor
      Structure();


      void setup() override;

      //! @name Derived public Solid::ModelEvaluator::Generic methods
      //! @{

      //! derived
      Inpar::Solid::ModelType type() const override { return Inpar::Solid::model_structure; }

      //! derived
      void reset(const Core::LinAlg::Vector<double>& x) override;

      //! derived
      bool evaluate_force() override;

      //! derived
      bool evaluate_stiff() override;

      //! derived
      bool evaluate_force_stiff() override;

      //! derived
      void pre_evaluate() override {};

      //! derived
      void post_evaluate() override {};

      /*! \brief Initialize viscous and inertial matrices
       *
       *  This is the place where we calculate the default mass matrix and the
       *  Rayleigh damping matrix only once during the compute_mass_matrix_and_init_acc routine.
       *

       *  */
      bool initialize_inertia_and_damping();

      //! derived
      bool assemble_force(Core::LinAlg::Vector<double>& f, const double& timefac_np) const override;

      //! derived
      bool assemble_jacobian(
          Core::LinAlg::SparseOperator& jac, const double& timefac_np) const override;

      //! derived
      void write_restart(
          Core::IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! derived
      void read_restart(Core::IO::DiscretizationReader& ioreader) override;

      //! derived
      void run_pre_compute_x(const Core::LinAlg::Vector<double>& xold,
          Core::LinAlg::Vector<double>& dir_mutable, const NOX::Nln::Group& curr_grp) override;

      //! derived
      void run_recover() override;

      //! derived
      void run_post_compute_x(const Core::LinAlg::Vector<double>& xold,
          const Core::LinAlg::Vector<double>& dir,
          const Core::LinAlg::Vector<double>& xnew) override;

      //! derived
      void run_post_iterate(const ::NOX::Solver::Generic& solver) override;

      //! derived
      void predict(const Inpar::Solid::PredEnum& pred_type) override;

      //! derived
      void update_step_state(const double& timefac_n) override;

      //! derived
      void update_step_element() override;

      //! derived
      void update_residual() override;

      //! derived
      void determine_stress_strain() override;

      //! derived
      void determine_energy() override;

      /*! \brief Fill energy map in eval_data
       *
       *  \param disnp (in): Current displacement vector
       *  \param velnp (in): Current velocity vector
       *  \param global (in): If true, sum and share the result over all procs and
       *                      save the global result. */
      void determine_energy(const Core::LinAlg::Vector<double>& disnp,
          const Core::LinAlg::Vector<double>* velnp, const bool global);

      /*! \brief determine the strain energy
       *
       *  \param disnp (in): Current displacement vector
       *  \param global (in): If true, sum the result over all procs and
       *                      save the global result. */
      void determine_strain_energy(const Core::LinAlg::Vector<double>& disnp, const bool global);

      //! derived
      void determine_optional_quantity() override {}

      //! derived
      void reset_step_state() override;

      //! derived
      void output_step_state(Core::IO::DiscretizationWriter& iowriter) const override;

      //! derived
      void runtime_pre_output_step_state() override;

      //! derived
      void runtime_output_step_state() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Map> get_block_dof_row_map_ptr() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_current_solution_ptr() const override;

      //! derived
      std::shared_ptr<const Core::LinAlg::Vector<double>> get_last_time_step_solution_ptr()
          const override;

      //! [derived]
      void post_output() override;

      //! [derived]
      void evaluate_jacobian_contributions_from_element_level_for_ptc() override;

      //! calculate the L2 displacement error in comparison to the analytical solution.
      void evaluate_analytical_error();

      //! [derived]
      void assemble_jacobian_contributions_from_element_level_for_ptc(
          std::shared_ptr<Core::LinAlg::SparseMatrix>& modjac, const double& timefac_n) override;

      //! [derived]
      void create_backup_state(const Core::LinAlg::Vector<double>& dir) override;

      //! [derived]
      void recover_from_backup_state() override;

      //! @}

     protected:
      //! pre-operator for \ref evaluate_internal
      virtual void pre_evaluate_internal() { /* empty */ };

     private:
      //! apply the internal force contributions
      bool apply_force_internal();

      //! apply the external force contributions
      bool apply_force_external();

      //! apply the internal force contributions and the evaluate the structural stiffness terms
      bool apply_force_stiff_internal();

      //! apply the external force contributions and evaluate possible linearization contributions
      bool apply_force_stiff_external();

      /** \brief Run before apply_force_stiff_external is executed
       *
       *  \param(in) fextnp: external force vector
       *  \param(in) stiff : structural tangential stiffness block
       *
       *  \return TRUE, if the execution of apply_force_stiff_external shall be
       *  skipped. Otherwise FALSE will be returned.
       *
       *  */
      bool pre_apply_force_stiff_external(
          Core::LinAlg::Vector<double>& fextnp, Core::LinAlg::SparseMatrix& stiff) const;

      //! Set the params_interface in the parameter list and call the other evaluate_neumann routine
      void evaluate_neumann(Core::LinAlg::Vector<double>& eval_vec,
          const std::shared_ptr<Core::LinAlg::SparseOperator>& eval_mat);

      /*! \brief Check if the given parameter list is valid and call the
       *  evaluate_neumann routine of the discretization
       *
       *  \param eval_vec (out) : external force vector
       *  \param eval_mat (out) : linearization of the external force (optional)
       *

       *  */
      void evaluate_neumann(Teuchos::ParameterList& p, Core::LinAlg::Vector<double>& eval_vec,
          const std::shared_ptr<Core::LinAlg::SparseOperator>& eval_mat);

      //! Set the params_interface in the parameter list and call the other evaluate_internal
      //! routine
      void evaluate_internal(std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
          std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec);

      /*! \brief Check if the given parameter list is valid and call the
       *  Evaluate routine of the discretization
       *
       *  \param eval_vec (out) : array of different internal forces (f_int, f_inertial)
       *  \param eval_mat (out) : array of different matrices (stiffness, mass, damping)
       *

       *  */
      void evaluate_internal(Teuchos::ParameterList& p,
          std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
          std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec);

      /*! \brief  Set the params_interface in the parameter list and call the other
       * evaluate_internal_specified_elements routine */
      void evaluate_internal_specified_elements(
          std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
          std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec,
          const Core::LinAlg::Map* ele_map_to_be_evaluated);

      /*! \brief  Check if the given parameter list is valid and call the
       *  Evaluate routine for all elements specified in the element map
       *
       *  */
      void evaluate_internal_specified_elements(Teuchos::ParameterList& p,
          std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
          std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec,
          const Core::LinAlg::Map* ele_map_to_be_evaluated);

      /*! \brief Add static structural internal force and stiffness matrix to the
       *         evaluate call (default)
       *
       *  Set matrix, vector and default action type.
       *
       *  \param eval_mat (out): pointer to the evaluation matrix array, which is
       *                         changed.
       *  \param eval_vec (out): pointer to the evaluation vector array, which is
       *                         changed.
       *

       *  */
      void static_contributions(std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
          std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec);

      /*! \brief Add static structural internal force to the evaluate call (default)
       *
       *  Set vector and default action type.
       *
       *  \param eval_vec (out): pointer to the evaluation vector array, which is
       *                         changed.
       *

       *  */
      void static_contributions(std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec);

      /*! \brief Add material damping matrix  to the evaluate call (optional)
       *
       *  \param eval_mat (out): pointer to the evaluation matrix array,
       *                         which will be augmented with the damping matrix
       *                         if desired.
       *
       *  \warning Material damping and non-linear mass effects cannot be
       *  considered at the same time at the moment!
       *

       *  */
      void material_damping_contributions(std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat);

      /*! \brief Add mass matrix and inertial force to the evaluate call (optional)
       *
       *  Set vector, matrix and default mass or lumped mass action type.
       *
       *  \param eval_mat (out): pointer to the evaluation matrix array,
       *                         which will be augmented with the mass matrix
       *                         if desired.
       *
       *  \param eval_vec (out): pointer to the evaluation vector array,
       *                         which will be augmented with the inertial force
       *                         vector if desired.
       *
       *  \warning Material damping and non-linear mass effects cannot be
       *  considered at the same time at the moment!
       *

       *  */
      void inertial_contributions(std::shared_ptr<Core::LinAlg::SparseOperator>* eval_mat,
          std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec);

      /*! \brief Add inertial force to the evaluate call (optional)
       *
       *  Set vectors and internal inertial force action type.
       *
       *  \param eval_vec (out): pointer to the evaluation vector array,
       *                         which will be augmented with the inertial force
       *                         vector if desired.
       *
       *  \warning Material damping and non-linear mass effects cannot be
       *  considered at the same time at the moment!
       *

       *  */
      void inertial_contributions(std::shared_ptr<Core::LinAlg::Vector<double>>* eval_vec);

      /*! \brief Evaluate the inertial forces (for the standard case) and
       *         any viscous damping forces
       *

       *  */
      void inertial_and_viscous_forces();

      /*! Check the fill_complete status of the stiffness and mass matrix
       *  and complete them, if necessary */
      void fill_complete();

      /*! \brief Assemble the Rayleigh damping matrix
       *
       *  Please note, that this has to been done only once during the
       *  Solid::Integrator::compute_mass_matrix_and_init_acc routine!
       *

       *  */
      void rayleigh_damping_matrix();

      /*! \brief Returns the interial force vector for non-linear mass problems
       *
       *  This function zeros the inertial force vector and returns it,
       *  if a non-linear mass problem is solved. Otherwise, a nullptr
       *  pointer is returned.
       *

       *  */
      std::shared_ptr<Core::LinAlg::Vector<double>> get_inertial_force();

      /*! \brief writes output for discretization structure
       *

       *  */
      void init_output_runtime_structure();

      /*!
       * \brief Initialize the gauss point data output routine
       *
       * This method must be called once after the input of all data.
       */
      void init_output_runtime_structure_gauss_point_data();

      /*! \brief writes output for discretization structure at the end of a time step
       *

       *  */
      void write_time_step_output_runtime_structure() const;

      /*! \brief writes output for discretization structure
       *         at the end of a nonlinear iteration
       *

       *  */
      void write_iteration_output_runtime_structure() const;

      /*! \brief writes output for discretization structure
       *

       *  */
      void write_output_runtime_structure(
          const Core::LinAlg::Vector<double>& displacement_state_vector,
          const Core::LinAlg::Vector<double>& velocity_state_vector,
          const Core::LinAlg::Vector<double>& acceleration_state_vector, int timestep_number,
          double time) const;

      /**
       * \brief Calculate the stress and / or strains for runtime output.
       */
      void output_runtime_structure_postprocess_stress_strain();

      void output_runtime_structure_gauss_point_data();

      void output_runtime_structure_postprocess_optional_data();

      /*! \brief writes special output for beam elements
       *

       *  */
      void init_output_runtime_beams();

      /*! \brief writes special output for beam elements at the end of a time step
       *

       *  */
      void write_time_step_output_runtime_beams() const;

      /*! \brief writes special output for beam elements at the end of a nonlinear iteration
       *

       *  */
      void write_iteration_output_runtime_beams() const;

      /*! \brief writes special output for beam elements
       *

       *  */
      void write_output_runtime_beams(Core::LinAlg::Vector<double>& displacement_state_vector,
          int timestep_number, double time) const;

      /*! \brief Write the parameters from the Solid::ModelEvaluator::Data
       *         to the Teuchos::ParameterList
       *
       *  todo: This function is temporary! It converts back to the old
       *  format using ParameterList to communicate with the element/materials.
       *  We delete this function, as soon as the old structural time
       *  integration is no longer supported and all elements use the data
       *  interface class.
       *

       *  */
      void params_interface2_parameter_list(
          std::shared_ptr<Solid::ModelEvaluator::Data> interface_ptr,
          Teuchos::ParameterList& params);

     private:
      //! @name Accessors to the data container content
      //! @{

      //! global internal force at \f$t_{n+1}\f$
      Core::LinAlg::Vector<double>& fint_np();

      //! global internal force at \f$t_{n+1}\f$ (read-only)
      const Core::LinAlg::Vector<double>& fint_np() const;

      //! global internal force at \f$t_{n}\f$ (read-only)
      const Core::LinAlg::Vector<double>& fint_n() const;

      //! global external force at \f$t_{n+1}\f$
      Core::LinAlg::Vector<double>& fext_np();

      //! global external force at \f$t_{n+1}\f$ (read-only)
      const Core::LinAlg::Vector<double>& fext_np() const;

      //! global external force at \f$t_{n}\f$ (read-only)
      const Core::LinAlg::Vector<double>& fext_n() const;

      //! inertial force at \f$t_{n+1}\f$
      Core::LinAlg::Vector<double>& finertial_np();

      //! inertial force at \f$t_{n+1}\f$ (read-only)
      const Core::LinAlg::Vector<double>& finertial_np() const;

      //! viscous force at \f$t_{n+1}\f$
      Core::LinAlg::Vector<double>& fvisco_np();

      //! viscous force at \f$t_{n+1}\f$ (read-only)
      const Core::LinAlg::Vector<double>& fvisco_np() const;

      //! structural displacement at \f$t_{n+1}\f$
      Core::LinAlg::Vector<double>& dis_np();

      //! structural displacement at \f$t_{n+1}\f$ (read-only)
      const Core::LinAlg::Vector<double>& dis_np() const;

      //! structural stiffness block
      Core::LinAlg::SparseMatrix& stiff() const;

      //! modified stiffness block
      Core::LinAlg::SparseMatrix& stiff_ptc() const;

      //! structural mass matrix
      Core::LinAlg::SparseOperator& mass();

      //! structural mass matrix (read-only)
      const Core::LinAlg::SparseOperator& mass() const;

      //! structural damping matrix
      Core::LinAlg::SparseOperator& damp();

      //! structural damping matrix
      const Core::LinAlg::SparseOperator& damp() const;

      //! @}

     private:
      //! structural element evaluation time
      double* dt_ele_ptr_;

      //! mass linearization type
      enum Inpar::Solid::MassLin masslin_type_;

      //! @name class only variables
      //! @{

      //! structural stiffness matrix
      Core::LinAlg::SparseMatrix* stiff_ptr_;

      //! contains ptc stiffness contributions calculated on elements
      std::shared_ptr<Core::LinAlg::SparseMatrix> stiff_ptc_ptr_;

      /*! \brief displacement increment
       *  Necessary for the EAS reconstruction, incremental strain evaluation,
       *  etc.. */
      std::shared_ptr<Core::LinAlg::Vector<double>> dis_incr_ptr_;

      //! visualization parameters
      Core::IO::VisualizationParameters visualization_params_;

      //! error evaluator parameters
      ErrorEvaluator::Parameters error_evaluator_parameters_;

      std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> vtu_writer_ptr_;

      //! beam discretization runtime output writer
      std::shared_ptr<BeamDiscretizationRuntimeOutputWriter> beam_vtu_writer_ptr_;

      //! @}
    };

  }  // namespace ModelEvaluator
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
