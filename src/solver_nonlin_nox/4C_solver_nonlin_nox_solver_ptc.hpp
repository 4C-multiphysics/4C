#ifndef FOUR_C_SOLVER_NONLIN_NOX_SOLVER_PTC_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_SOLVER_PTC_HPP


#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"  // base class of NOX::Nln::LinSystem::PrePostOp::PseudoTransient
#include "4C_solver_nonlin_nox_solver_linesearchbased.hpp"  // base class of NOX::Nln::Solver::PseudoTransient

#include <NOX_Abstract_Vector.H>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace Core::LinAlg
{
  class SparseOperator;
  class SparseMatrix;
}  // namespace Core::LinAlg

namespace NOX
{
  namespace Nln
  {
    class LinearSystem;
    namespace Solver
    {
      /*! \brief Pseudo Transient Continuation (PTC) non-linear solver
       *
       * This implementation is based on the ::NOX::Solver::PseudoTransient class
       * and the former 4C internal implementation of the PTC method. In contrast
       * to the NOX internal implementation we try to avoid the need of the Thyra
       * interface package.
       *
       * During the line search routine we use always the transient residual [3] for the
       * current iteration point \f$x_{n}\f$
       * \f[
       *   r_{t}(x) = \delta_{n}^{-1} V_{n} (x - x_{n}) + F(x)
       * \f]
       * and a desired merit function (e.g. the sum of squares merit function).
       *
       * See also:
       *
       * [1] C. T. Kelley, D. E. Keyes, "Convergence analysis of pseudo-transient continuation",
       *     SIAM J. Numer. Anal., Vol. 35, No. 2, pp. 508-523, 1998.
       *
       * [2] M. W. Gee, C. T. Kelley, R. B. Lehoucq, "Pseudo-transient continuation for nonlinear
       * transient elasticity", Int. J. Numer. Meth. Engng., Vol. 78, pp. 1209-1219, 2009.
       *
       * [3] M. Ceze, K. J. Fidkowski, "Constrained pseudo-transient continuation",
       *     Int. J. Numer. Meth. Engng., Vol. 102, pp. 1683-1703, 2015.
       *
       * \author Michael Hiermeier */
      class PseudoTransient : public NOX::Nln::Solver::LineSearchBased
      {
       public:
        //! Different pseudo time step control types
        enum TSCType
        {
          tsc_ser,  //!< switched evolution relaxation
          tsc_tte,  //!< temporal truncation error
          tsc_mrr   //!< model reduction ratio
        };

        //! Map pseudo time step control type stl_string to enum
        inline enum TSCType string_to_tsc_type(const std::string& name)
        {
          TSCType type = tsc_ser;
          if (name == "SER" || name == "Switched Evolution Relaxation")
            type = tsc_ser;
          else if (name == "TTE" || name == "Temporal Truncation Error")
            type = tsc_tte;
          else if (name == "MRR" || name == "Model Reduction Ratio")
            type = tsc_mrr;
          else
          {
            std::ostringstream msg;
            msg << "Unknown conversion from STL_STRING to TSCType enum for " << name << "."
                << std::endl;
            throw_error("String2TSCType", msg.str());
          }
          return type;
        };

        //! Different types of scaling operator V
        enum ScaleOpType
        {
          scale_op_identity,      //!< Use the identity matrix (see [2])
          scale_op_cfl_diagonal,  //!< Use a diagonal matrix based on the local
                                  //!< Courant-Friedrichs-Lewy (CFL) number (see [1]).
          scale_op_element_based,
          scale_op_element_based_constant
        };

        enum BuildOpType
        {
          build_op_everyiter,
          build_op_everytimestep
        };

        //! Map build operator type stl_string to enum
        inline enum BuildOpType string_to_build_op_type(const std::string& name)
        {
          BuildOpType type = build_op_everyiter;
          if (name == "every iter")
            type = build_op_everyiter;
          else if (name == "every timestep")
            type = build_op_everytimestep;
          else
          {
            std::ostringstream msg;
            msg << "Unknown conversion from STL_STRING to BuildOperatorType enum for " << name
                << "." << std::endl;
            throw_error("String2BuildOpType", msg.str());
          }

          return type;
        };

        //! Map scaling operator type stl_string to enum
        inline enum ScaleOpType string_to_scale_op_type(const std::string& name)
        {
          ScaleOpType type = scale_op_identity;
          if (name == "Identity")
            type = scale_op_identity;
          else if (name == "CFL diagonal")
            type = scale_op_cfl_diagonal;
          else if (name == "Element based")
            type = scale_op_element_based;
          else
          {
            std::ostringstream msg;
            msg << "Unknown conversion from STL_STRING to ScaleOperatorType enum for " << name
                << "." << std::endl;
            throw_error("String2ScaleOpType", msg.str());
          }

          return type;
        };

       public:
        //! constructor
        PseudoTransient(const Teuchos::RCP<::NOX::Abstract::Group>& grp,
            const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
            const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests,
            const Teuchos::RCP<Teuchos::ParameterList>& params);


        //! reset the non-linear solver
        void reset(const ::NOX::Abstract::Vector& initialGuess,
            const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests,
            const Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic>& innerTests);
        void reset(const ::NOX::Abstract::Vector& initialGuess,
            const Teuchos::RCP<::NOX::StatusTest::Generic>& outerTests) override;

        void create_scaling_operator();

        void create_lin_system_pre_post_operator();

        void create_group_pre_post_operator();

        ::NOX::StatusTest::StatusType step() override;

        ::NOX::StatusTest::StatusType solve() override;

        //! Returns the inverse pseudo time step
        const double& get_inverse_pseudo_time_step() const;

        //! Returns the scaling factor
        const double& get_scaling_factor() const;

        //! Returns the scaling operator type
        const enum ScaleOpType& get_scaling_operator_type() const;

        //! Returns the scaling diagonal operator
        const Core::LinAlg::Vector<double>& get_scaling_diag_operator() const;

        //! Returns the use_pseudo_transient_residual status
        bool use_pseudo_transient_residual() const;

        //! Returns the pseudo transient continuation status
        bool is_ptc_solve() const;

       protected:
        //! Initialize the PTC specific variables and call the init function of the base class
        void init() override;

        //! Print the non-linear solver update
        void printUpdate() override;

        //! Compute the nodal pseudo velocity for the CFL update
        void compute_pseudo_velocity();

        /*! \brief Evaluate the model reduction ratio
         *
         *  We use the transient residual and the corresponding (quadratic) model
         * \f$m(\eta_{n}^{*})\f$. If we choose the Sum of Squares merit function, we get
         *
         *  \f[
         *    \rho = \frac{0.5 r_{t}^{T}(x_{n}) r_{t}^{T}(x_{n}) - 0.5 r_{t}^{T}(x_{n} +
         * \eta_{n}^{*} d) r_{t}^{T}(x_{n} + \eta_{n}^{*} d)}{0.5 r_{t}^{T}(x_{n}) r_{t}^{T}(x_{n})
         * - m(\eta_{n}^{*})}, \f] where \f{eqnarray*}{
         *    m(\eta) &=& 0.5 r_{t}^{T}(x_{n}) r_{t}(x_{n}) + \eta d_{n} \nabla_{x}
         * r_{t}(x)|_{x=x_{n}} r_{t}(x_{n}) + 0.5 \eta^{2} d_{n}^{T} \nabla_{x} r_{t}(x)|_{x=x_{n}}
         * \nabla_{x} r_{t}(x)|_{x=x_{n}}^{T} d_{n} \\
         *            &=& 0.5 F_{n}^{T} F_{n} + \eta d_{n} J_{t}^{T} F_{n} + 0.5 \eta^{2} d_{n}^{T}
         * J_{t}^{T} J_{t} d_{n}, \\ \nabla_{x} r_{t}(x)|_{x=x_{n}} &=& J_{t}^{T} = \delta_{n}^{-1}
         * V_{n} + \nabla_{x} F(x)|_{x=x_{n}}. \f}
         *
         *  Please note, that the transient or modified Jacobian is necessary for the evaluation.
         * Therefore, the order of the function calls has to be considered. We have to call the
         * eval_model_reduction_ratio() before we call the adjust_pseudo_time_step() function! */
        void eval_model_reduction_ratio();

        /*! \brief Adjust the pseudo time step if line search was active and the step length has
         * been changed
         *
         *  We use a least squares approximation based on the following idea.
         *  The pseudo time step is modified in such a way, that it represents in a least squares
         *  sense the actual used step, i.e. the direction vector scaled by the step length
         *  parameter \f$\eta_{n}^{*}\f$. The corresponding pseudo time step is called
         *  \f$\delta_{n}^{*}\f$:
         *
         *  \f[
         *    [(\delta_{n}^{*})^{-1} V_{n} + \nabla_{x} F_{n}^{T}] (x(t_{n}+\delta^{*}_{n}) - x_{n})
         * = -F_{n}, \f]
         *
         *  where we replace \f$(x(t_{n}+\delta^{*}_{n})\f$ by \f$\eta_{n}^{*} d_{n}\f$ and
         * reformulate the equation
         *
         *  \f[
         *    V_{n} d_{n} = - \delta_{n}^{*} ((\eta_{n}^{*})^{-1} F_{n} + \nabla_{x} F_{n}^{T}
         * d_{n}). \f]
         *
         *  Since we cannot find a suitable scalar \f$\delta_{n}^{*}\f$, which solves the problem
         * exactly, we formulate a least squares problem to minimize the error:
         *
         *  \f[
         *    \min\limits_{\delta_{n}} 0.5 \| V_{n} d_{n} - \delta_{n} ((\eta_{n}^{*})^{-1} F_{n} +
         * \nabla_{x} F_{n}^{T} d_{n}) \|^{2} \f]
         *
         *  The approximated pseudo time step is
         *  \f[
         *    \delta^{*}_{n} \approx - \frac{d_{n}^{T} V_{n} ((\eta_{n}^{*})^{-1} F_{n} + \nabla_{x}
         * F_{n}^{T} d_{n})}{\|(\eta_{n}^{*})^{-1} F-{n} + \nabla F_{n}^{T} d_{n} \|^{2}}. \f]
         *
         *  Please note, that we undo the changes to the jacobian, if the step length is not
         * equal 1.0! */
        void adjust_pseudo_time_step();

        /*! \brief Update the pseudo time step
         *
         *  There are three different options:
         *  - The switched evolution relaxation\n
         *    (in the most simple case)
         *    \f[
         *       \delta_{n} = \delta_{0}
         * \frac{\|\underline{F}(\underline{x}_{0})\|}{\|\underline{F}(\underline{x}_{n})\|} \f]
         *  - The temporal truncation error\n
         *
         *       MISSING AT THE MOMENT
         *
         *  - The model reduction ratio.\n
         *    See the eval_model_reduction_ratio() routine for more detail on how to evaluate the
         * ratio.
         *
         *    If \f$\rho\f$ is smaller than 0.2, the pseudo time step is decreased,
         *    else if \f$\rho\f$ is larger than 0.9, the pseudo time step is increased,
         *    otherwise we keep the current pseudo time step unchanged. */
        void update_pseudo_time_step();

        //! Returns true if the scaling operator has been evaluated.
        bool is_scaling_operator() const { return isScalingOperator_; };

       private:
        //! Throw class specific error
        void throw_error(const std::string& functionName, const std::string& errorMsg) const;

       protected:
        //! Inner Stopping test
        Teuchos::RCP<NOX::Nln::Inner::StatusTest::Generic> iTestPtr_;

        /*! Pointer to a linear system pre/post operator, which is used
         *  to modify the jacobian directly from the non-linear solver. */
        Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> prePostLinSysPtr_;

        /*! Pointer to a nln group pre/post operator, which is used
         *  to modify the right-hand-side directly from the non-linear solver. */
        Teuchos::RCP<NOX::Nln::Abstract::PrePostOperator> prePostGroupPtr_;

        //! Pointer to the scaling Operator (Identity scaling)
        Teuchos::RCP<Core::LinAlg::Vector<double>> scalingDiagOpPtr_;
        //! Pointer to the scaling Operator (Element based scaling)
        Teuchos::RCP<Core::LinAlg::SparseMatrix> scalingMatrixOpPtr_;

        //! @name Special pseudo transient continuation parameters
        //@{
        //! Pseudo step size for pseudo-transient stepping.
        double delta_;
        //! Inverse pseudo time step size for pseudo-transient stepping.
        double invDelta_;
        //! Initial pseudo time step size.
        double deltaInit_;
        //! Maximum pseudo time step size.
        double deltaMax_;
        //! Minimum pseudo time step size.
        double deltaMin_;
        //! Pseudo time step size from previous iteration.
        double deltaOld_;
        //! Pseudo-transient time.
        double pseudoTime_;
        //! Exponent for SER.
        double SER_alpha_;
        //! Scaling factor for modified Jacobian, i.e. the scaling operator.
        double scaleFactor_;

        //! solution time derivative used for scaling operator V in pseudo-transient paper.
        Teuchos::RCP<::NOX::Abstract::Vector> xDot_;

        //! Are we solving the pseudo transient continuation problem at the moment?
        bool isPtcSolve_;

        /** \brief If set to true, the computeF function is used to modify the residual during the
         * line search call.
         *
         *  The pseudo transient residual is defined as
         *  \f[
         *    r_{t}(\eta) = V_{n} \delta_{n}^{-1} [x_{n} + \eta d_{n} - x_{n}] + F(x_{n} + \eta
         * d_{n}) \f]
         */
        bool usePseudoTransientResidual_;

        //! if it's true, we compute automatically an initial pseudo time step.
        bool calcDeltaInit_;

        //! Has the scaling operator been evaluated, yet?
        bool isScalingOperator_;

        //! Maximum number of iterations before pseudo-transient is disabled and the algorithm
        //! switches to a line search-based direct to steady state solve.
        int maxPseudoTransientIterations_;

        //! time step control type
        enum TSCType tscType_;

        //! scaling operator type
        enum ScaleOpType scaleOpType_;

        //! scaling operator type
        enum BuildOpType build_scaling_op_;

        //! vector norm type (necessary for the time step control)
        enum ::NOX::Abstract::Vector::NormType normType_;

        //! The current model reduction ratio.
        double modelReductionRatio_;
        //@}
      };  // class PseudoTransient
    }     // namespace Solver

    namespace LinSystem
    {
      namespace PrePostOp
      {
        /*! \brief Pseudo Transient Continuation (PTC) non-linear solver helper class
         *
         * This class implementation is an example for the usage of the
         * NOX::Nln::Abstract::PrePostOperator and its wrapper the
         * NOX::Nln::LinSystem::PrePostOperator classes. The PTC solver uses them to modify the
         * linear system or to be even more precise the jacobian.
         *
         * \author Michael Hiermeier */
        class PseudoTransient : public NOX::Nln::Abstract::PrePostOperator
        {
         public:
          //! constructor
          PseudoTransient(Teuchos::RCP<Core::LinAlg::Vector<double>>& scalingDiagOp,
              Teuchos::RCP<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr,
              const NOX::Nln::Solver::PseudoTransient& ptcsolver);

          void run_post_compute_jacobian(Core::LinAlg::SparseOperator& jac,
              const Core::LinAlg::Vector<double>& x, const NOX::Nln::LinearSystem& linsys) override;

          void run_post_compute_fand_jacobian(Core::LinAlg::Vector<double>& rhs,
              Core::LinAlg::SparseOperator& jac, const Core::LinAlg::Vector<double>& x,
              const NOX::Nln::LinearSystem& linsys) override;

         protected:
          //! modify the jacobian as defined by the scaling operator type
          virtual void modify_jacobian(Core::LinAlg::SparseMatrix& jac);

         private:
          //! read-only access
          const NOX::Nln::Solver::PseudoTransient& ptcsolver_;

          //! reference to the scaling diagonal operator
          Teuchos::RCP<Core::LinAlg::Vector<double>>& scaling_diag_op_ptr_;

          //! reference to the scaling diagonal operator matrix
          Teuchos::RCP<Core::LinAlg::SparseMatrix>& scaling_matrix_op_ptr_;

        };  // class PseudoTransient
      }     // namespace PrePostOp
    }       // namespace LinSystem

    namespace GROUP
    {
      namespace PrePostOp
      {
        /*! \brief Pseudo Transient Continuation (PTC) non-linear solver helper class
         *
         * The PTC solver uses this helper class to modify the right hand side.
         *
         * \author Michael Hiermeier */
        class PseudoTransient : public NOX::Nln::Abstract::PrePostOperator
        {
         public:
          //! constructor
          PseudoTransient(Teuchos::RCP<Core::LinAlg::Vector<double>>& scalingDiagOpPtr,
              Teuchos::RCP<Core::LinAlg::SparseMatrix>& scalingMatrixOpPtr,
              const NOX::Nln::Solver::PseudoTransient& ptcsolver);


          void run_pre_compute_f(
              Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp) override;

          void run_post_compute_f(
              Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp) override;

         protected:
          Teuchos::RCP<::NOX::Epetra::Vector> eval_pseudo_transient_f_update(
              const NOX::Nln::Group& grp);

         private:
          //! read-only access
          const NOX::Nln::Solver::PseudoTransient& ptcsolver_;

          //! reference to the scaling diagonal operator
          Teuchos::RCP<Core::LinAlg::Vector<double>>& scaling_diag_op_ptr_;

          //! reference to the scaling diagonal operator matrix
          Teuchos::RCP<Core::LinAlg::SparseMatrix>& scaling_matrix_op_ptr_;

          //! reference to the flag which indicates, if the current right-hand-side is the transient
          //! right-hand-side
          bool is_pseudo_transient_residual_;

        };  // class PseudoTransient
      }     // namespace PrePostOp
    }       // namespace GROUP
  }         // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
