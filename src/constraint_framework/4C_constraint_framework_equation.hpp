// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_FRAMEWORK_EQUATION_HPP
#define FOUR_C_CONSTRAINT_FRAMEWORK_EQUATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_sparsematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Constraints::SubmodelEvaluator
{
  /*! \brief The ConstraintEquationBase class serves as a base interface
   * for managing constraint equations within the constraint framework
   */
  class ConstraintEquationBase
  {
   public:
    virtual ~ConstraintEquationBase() = default;
    //! Constructor
    ConstraintEquationBase() = default;

    /*! \brief Assemble this equation's terms into the coupling matrix \f$Q_{Ld}\f$
     *
     * @param [in,out] Q_Ld coupling-stiffness matrix
     */
    virtual void evaluate_equation(Core::LinAlg::SparseMatrix& Q_Ld) = 0;

    /*! \brief Return the number of constraints the object contains
     *
     * @return [out] number of constraint equations the object contains
     */
    int get_number_of_constraint_equation_objects() const;

    /*! \brief Return the global id of the affected row of this equation
     *
     * @return [out] global id of the affected row of this equation
     */
    int get_first_row_id() const;

    /*! \brief Return the global id of the affected row of this equation
     *
     * @param [in] global_row_id global id of the affected row of this equation
     */
    void set_first_row_id(int global_row_id);

   private:
    //! Number of dof coupled per Object (= Number of equations per Obj.)
    int n_dof_coupled_ = 1;

    //! ID of the first constraint in the set
    int first_row_id_;
  };
  /*! \brief The class provides the method for evaluating linear coupled
   *  equations and manages associated coefficients and data.
   */
  class LinearCoupledEquation : public ConstraintEquationBase
  {
   public:
    ~LinearCoupledEquation() override = default;

    //! Default Constructor
    LinearCoupledEquation() = default;

    /*!
        \brief Standard Constructor
    */
    LinearCoupledEquation(int id, const std::vector<int>& dofs, std::vector<double> coefficients);

    //! derived
    void evaluate_equation(Core::LinAlg::SparseMatrix& Q_Ld) override;

   private:
    //! Struct with Term data: Coef, RowID, DofID
    struct TermData
    {
      double Coef;
      int RowId;  // is equal to the id of the MPC
      int DofId;  // dof the coefficient is multiplied with.
    };

    //! Vector with the data of the terms of a single equation
    std::vector<TermData> equation_data_;
  };
}  // namespace Constraints::SubmodelEvaluator

FOUR_C_NAMESPACE_CLOSE
#endif
