// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINEAR_SOLVER_AMGNXN_PRECONDITIONER_HPP
#define FOUR_C_LINEAR_SOLVER_AMGNXN_PRECONDITIONER_HPP

#include "4C_config.hpp"

#include "4C_linalg_blocksparsematrix.hpp"
#include "4C_linear_solver_amgnxn_hierarchies.hpp"
#include "4C_linear_solver_amgnxn_smoothers.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_linear_solver_preconditioner_type.hpp"

#include <Epetra_Operator.h>
#include <MueLu.hpp>
#include <MueLu_BaseClass.hpp>
#include <MueLu_Level.hpp>
#include <MueLu_UseDefaultTypes.hpp>
#include <MueLu_Utilities.hpp>

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Core::LinearSolver
{
  class AmGnxnPreconditioner : public PreconditionerTypeBase
  {
   public:
    AmGnxnPreconditioner(Teuchos::ParameterList& params);

    void setup(Core::LinAlg::SparseOperator& matrix, const Core::LinAlg::MultiVector<double>& x,
        Core::LinAlg::MultiVector<double>& b) override;

    virtual void setup(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A);

    /// linear operator used for preconditioning
    std::shared_ptr<Epetra_Operator> prec_operator() const override;

   private:
    // Private variables
    std::shared_ptr<Epetra_Operator> p_;  // The underlying preconditioner object
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a_;  // A own copy of the system matrix
    Teuchos::ParameterList& params_;

  };  // AMGnxn_Preconditioner

  class AmGnxnInterface
  {
   public:
    AmGnxnInterface(Teuchos::ParameterList& params, int NumBlocks);

    std::vector<std::string> get_mue_lu_xml_files() { return xml_files_; }
    std::vector<int> get_num_pdes() { return num_pdes_; }
    std::vector<int> get_null_spaces_dim() { return null_spaces_dim_; }
    std::vector<std::shared_ptr<std::vector<double>>> get_null_spaces_data()
    {
      return null_spaces_data_;
    }
    // int GetNumLevelAMG(){return NumLevelAMG_;}
    Teuchos::ParameterList get_preconditioner_params() { return prec_params_; }
    Teuchos::ParameterList get_smoothers_params() { return smoo_params_; }
    std::string get_preconditioner_type() { return prec_type_; }

   private:
    std::vector<std::string> xml_files_;
    std::vector<int> num_pdes_;
    std::vector<int> null_spaces_dim_;
    std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data_;
    // int NumLevelAMG_;
    Teuchos::ParameterList prec_params_;
    Teuchos::ParameterList smoo_params_;
    std::string prec_type_;

    // Helper function to convert int to std::string
    std::string convert_int(int number)
    {
      std::stringstream ss;
      ss << number;
      return ss.str();
    }
  };

  class AmGnxnOperator : virtual public Epetra_Operator
  {
   public:
    AmGnxnOperator(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A,
        std::vector<int> num_pdes, std::vector<int> null_spaces_dim,
        std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data,
        const Teuchos::ParameterList& amgnxn_params, const Teuchos::ParameterList& smoothers_params,
        const Teuchos::ParameterList& muelu_params);

    // virtual functions given by Epetra_Operator. The only one to be used is ApplyInverse()
    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

    int SetUseTranspose(bool UseTranspose) override
    {
      // default to false
      return 0;
    }

    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override
    {
      FOUR_C_THROW("Function not implemented");
      return -1;
    }

    double NormInf() const override
    {
      FOUR_C_THROW("Function not implemented");
      return -1.0;
    }

    const char* Label() const override { return "AMG(BlockSmoother)"; }

    bool UseTranspose() const override
    {
      // default to false
      return false;
    }

    bool HasNormInf() const override
    {
      FOUR_C_THROW("Function not implemented");
      return false;
    }

    // Only required to properly define an Epetra_Operator, not should be used!
    const Epetra_Comm& Comm() const override { return a_->Comm(); }

    // Only required to properly define an Epetra_Operator, not should be used!
    const Epetra_Map& OperatorDomainMap() const override { return a_->OperatorDomainMap(); }

    // Only required to properly define an Epetra_Operator, not should be used!
    const Epetra_Map& OperatorRangeMap() const override { return a_->OperatorRangeMap(); }

    void setup();

   private:
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a_;
    std::vector<Teuchos::ParameterList> muelu_lists_;
    std::vector<int> num_pdes_;
    std::vector<int> null_spaces_dim_;
    std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data_;
    Teuchos::ParameterList amgnxn_params_;
    Teuchos::ParameterList smoothers_params_;
    Teuchos::ParameterList muelu_params_;

    bool is_setup_flag_;

    std::shared_ptr<AMGNxN::CoupledAmg> v_;

  };  // class AMGnxn_Operator

  class BlockSmootherOperator : virtual public Epetra_Operator
  {
   public:
    BlockSmootherOperator(std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> A,
        std::vector<int> num_pdes, std::vector<int> null_spaces_dim,
        std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data,
        const Teuchos::ParameterList& amgnxn_params,
        const Teuchos::ParameterList& smoothers_params);

    // virtual functions given by Epetra_Operator. The only one to be used is ApplyInverse()
    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override;

    int SetUseTranspose(bool UseTranspose) override
    {
      // default to false
      return 0;
    }

    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const override
    {
      FOUR_C_THROW("Function not implemented");
      return -1;
    }

    double NormInf() const override
    {
      FOUR_C_THROW("Function not implemented");
      return -1.0;
    }

    const char* Label() const override { return "BlockSmoother(X)"; }

    bool UseTranspose() const override
    {
      // default to false
      return false;
    }

    bool HasNormInf() const override
    {
      FOUR_C_THROW("Function not implemented");
      return false;
    }

    // Only required to properly define an Epetra_Operator, not should be used!
    const Epetra_Comm& Comm() const override { return a_->Comm(); }

    // Only required to properly define an Epetra_Operator, not should be used!
    const Epetra_Map& OperatorDomainMap() const override { return a_->OperatorDomainMap(); }

    // Only required to properly define an Epetra_Operator, not should be used!
    const Epetra_Map& OperatorRangeMap() const override { return a_->OperatorRangeMap(); }

    void setup();  // TODO

   private:
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> a_;
    std::vector<int> num_pdes_;
    std::vector<int> null_spaces_dim_;
    std::vector<std::shared_ptr<std::vector<double>>> null_spaces_data_;
    Teuchos::ParameterList amgnxn_params_;
    Teuchos::ParameterList smoothers_params_;

    bool is_setup_flag_;
    Teuchos::RCP<AMGNxN::BlockedSmoother> s_;
    Teuchos::RCP<AMGNxN::GenericSmoother> sbase_;

  };  // class BlockSmoother_Operator

}  // namespace Core::LinearSolver

FOUR_C_NAMESPACE_CLOSE

#endif
