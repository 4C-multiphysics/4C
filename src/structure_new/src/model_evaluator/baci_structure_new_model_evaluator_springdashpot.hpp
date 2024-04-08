/*-----------------------------------------------------------*/
/*! \file

\brief Evaluation and assembly of all spring dashpot terms


\level 3

*/
/*-----------------------------------------------------------*/


#ifndef FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_SPRINGDASHPOT_HPP
#define FOUR_C_STRUCTURE_NEW_MODEL_EVALUATOR_SPRINGDASHPOT_HPP

#include "baci_config.hpp"

#include "baci_constraint_springdashpot.hpp"
#include "baci_structure_new_model_evaluator_generic.hpp"

#include <Teuchos_RCP.hpp>

BACI_NAMESPACE_OPEN

// forward declarations
namespace CORE::LINALG
{
  class SparseMatrix;
}  // namespace CORE::LINALG

namespace STR
{
  namespace MODELEVALUATOR
  {
    class SpringDashpot : public Generic
    {
     public:
      //! constructor
      SpringDashpot();

      void Setup() override;

      //! derived
      INPAR::STR::ModelType Type() const override { return INPAR::STR::model_springdashpot; }

      //! derived
      void Reset(const Epetra_Vector& x) override;

      //! derived
      bool EvaluateForce() override;

      //! derived
      bool EvaluateStiff() override;

      //! derived
      bool EvaluateForceStiff() override;

      //! derived
      void PreEvaluate() override{};

      //! derived
      void PostEvaluate() override{};

      //! derived
      bool AssembleForce(Epetra_Vector& f, const double& timefac_np) const override;

      //! Assemble the jacobian at \f$t_{n+1}\f$
      bool AssembleJacobian(
          CORE::LINALG::SparseOperator& jac, const double& timefac_np) const override;

      //! derived
      void WriteRestart(
          IO::DiscretizationWriter& iowriter, const bool& forced_writerestart) const override;

      //! derived
      void ReadRestart(IO::DiscretizationReader& ioreader) override;

      //! [derived]
      void Predict(const INPAR::STR::PredEnum& pred_type) override{};

      //! derived
      void RunPreComputeX(const Epetra_Vector& xold, Epetra_Vector& dir_mutable,
          const NOX::NLN::Group& curr_grp) override{};

      //! derived
      void RunPostComputeX(
          const Epetra_Vector& xold, const Epetra_Vector& dir, const Epetra_Vector& xnew) override
      {
      }

      //! derived
      void RunPostIterate(const ::NOX::Solver::Generic& solver) override{};

      //! derived
      void UpdateStepState(const double& timefac_n) override;

      //! derived
      void UpdateStepElement() override{};

      //! derived
      void DetermineStressStrain() override{};

      //! derived
      void DetermineEnergy() override{};

      //! derived
      void DetermineOptionalQuantity() override{};

      //! derived
      void OutputStepState(IO::DiscretizationWriter& iowriter) const override;

      //! derived
      void ResetStepState() override;

      //! derived
      Teuchos::RCP<const Epetra_Map> GetBlockDofRowMapPtr() const override;

      //! derived
      Teuchos::RCP<const Epetra_Vector> GetCurrentSolutionPtr() const override;

      //! derived
      Teuchos::RCP<const Epetra_Vector> GetLastTimeStepSolutionPtr() const override;

      //! [derived]
      void PostOutput() override;

     private:
      //! all spring dashpot instances
      std::vector<Teuchos::RCP<CONSTRAINTS::SpringDashpot>> springs_;

      //! structural displacement at \f$t_{n+1}\f$
      Teuchos::RCP<const Epetra_Vector> disnp_ptr_;

      //! structural velocity at \f$t_{n+1}\f$
      Teuchos::RCP<const Epetra_Vector> velnp_ptr_;

      //! structural stiffness matrix
      Teuchos::RCP<CORE::LINALG::SparseMatrix> stiff_spring_ptr_;

      //! spring forces at \f$t_{n+1}\f$
      Teuchos::RCP<Epetra_Vector> fspring_np_ptr_;
    };

  }  // namespace MODELEVALUATOR
}  // namespace STR

BACI_NAMESPACE_CLOSE

#endif  // STRUCTURE_NEW_MODEL_EVALUATOR_SPRINGDASHPOT_H
