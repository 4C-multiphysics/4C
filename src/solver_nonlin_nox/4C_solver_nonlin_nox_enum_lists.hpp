// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLVER_NONLIN_NOX_ENUM_LISTS_HPP
#define FOUR_C_SOLVER_NONLIN_NOX_ENUM_LISTS_HPP

#include "4C_config.hpp"

#include <NOX_StatusTest_Generic.H>

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    //! Supported solution type names
    enum SolutionType : int
    {
      sol_unknown,
      sol_structure,           ///< structural problem
      sol_contact,             ///< contact problem
      sol_meshtying,           ///< meshtying problem
      sol_cardiovascular0d,    ///< 0D cardiovascular problem
      sol_lag_pen_constraint,  ///< Lagrange or/and penalty enforced constraint problem
      sol_scatra,              ///< scalar transport problem
      sol_beaminteraction_lm   ///< beaminteraction with Lagrange multipliers problem
    };

    //! Map quantity enum to std::string
    inline std::string solution_type_to_string(const SolutionType type)
    {
      switch (type)
      {
        case sol_structure:
          return "Structure";
        case sol_contact:
          return "Contact";
        case sol_meshtying:
          return "Meshtying";
        case sol_cardiovascular0d:
          return "Cardiovascular0D";
        case sol_lag_pen_constraint:
          return "Lag-Pen-Constraint";
        case sol_scatra:
          return "ScaTra";
        case sol_unknown:
        default:
          return "Unknown Solution Type";
      }
    };

    //! Map quantity std::string to enum
    inline SolutionType string_to_solution_type(const std::string& name)
    {
      SolutionType type = sol_unknown;
      if (name == "Structure")
        type = sol_structure;
      else if (name == "Contact")
        type = sol_contact;
      else if (name == "Meshtying")
        type = sol_meshtying;
      else if (name == "Cardiovascular0D")
        type = sol_cardiovascular0d;
      else if (name == "Lag-Pen-Constraint")
        type = sol_lag_pen_constraint;
      else if (name == "ScaTra")
        type = sol_scatra;

      return type;
    };

    //! type of the optimization problem
    enum OptimizationProblemType : int
    {
      opt_unconstrained,          ///< unconstrained optimization problem
      opt_equality_constrained,   ///< pure equality constrained optimization problem
      opt_inequality_constrained  ///< inequality or mixed optimization problem
    };

    /// Correction types
    enum class CorrectionType : int
    {
      vague,          ///< undefined correction type
      soc_automatic,  ///< let the algorithm choose which SOC strategy to follow
      soc_cheap,      ///< cheap SOC step, reuse the old system of equations
      soc_full        ///< do full SOC step in the flavor of the watch dog method
    };

    //! Map second order correction type enum to std::string
    inline std::string correction_type_to_string(const CorrectionType type)
    {
      switch (type)
      {
        case CorrectionType::vague:
          return "CorrectionType::vague";
        case CorrectionType::soc_cheap:
          return "CorrectionType::soc_cheap";
        case CorrectionType::soc_full:
          return "CorrectionType::soc_full";
        case CorrectionType::soc_automatic:
          return "CorrectionType::soc_automatic";
        default:
          return "Unknown second order correction Type";
      }
    };

    namespace LinSystem
    {
      enum class ConditionNumber
      {
        one_norm,         ///< 1-norm condition number
        inf_norm,         ///< inf-norm condition number
        max_min_ev_ratio  ///< condition number as ration of largest and smallest eigenvalue
      };

      inline std::string condition_number_to_string(const ConditionNumber condtype)
      {
        switch (condtype)
        {
          case ConditionNumber::one_norm:
            return "one_norm";
          case ConditionNumber::inf_norm:
            return "inf_norm";
          case ConditionNumber::max_min_ev_ratio:
            return "max_min_ev_ratio";
          default:
            return "unknown_condition_number_type";
        }
      }

      //! supported LinearSystem types
      enum LinearSystemType : int
      {
        linear_system_structure,
        linear_system_structure_meshtying,
        linear_system_structure_contact,
        linear_system_structure_cardiovascular0d,
        linear_system_structure_lag_pen_constraint,
        linear_system_scatra,
        linear_system_undefined
      };

      //! List of types that can be used for the Jacobian and/or Preconditioner.
      enum OperatorType : int
      {
        LinalgSparseOperator,     ///< A LINALG_SparseOperator object.
        LinalgSparseMatrixBase,   ///< A LINALG_SparseMatrixBase object.
        LinalgBlockSparseMatrix,  ///< A LINALG_BlockSparseMatrix object.
        LinalgSparseMatrix        ///< A LINALG_SparseMatrix object.
      };

      /// Map operator type to std::string
      inline std::string operator_type_to_string(const OperatorType type)
      {
        switch (type)
        {
          case LinalgSparseOperator:
            return "LINALG_SparseOperator";
          case LinalgBlockSparseMatrix:
            return "LINALG_BlockSparseMatrix";
          case LinalgSparseMatrixBase:
            return "LINALG_SparseMatrixBase";
          case LinalgSparseMatrix:
            return "LINALG_SparseMatrix";
          default:
            return "unknown operator type";
        }
      };
    }  // namespace LinSystem


    namespace MeritFunction
    {
      //! order of the linearization term
      enum LinOrder : int
      {
        //! all orders
        linorder_all,
        //! first order linearization
        linorder_first,
        //! second order linearization terms
        linorder_second
      };

      //! type of the linearization term, i.e. with respect to which quantity the linearization was
      //! performed
      enum LinType : int
      {
        //! linearization with respect to primary and Lagrange multiplier degrees of freedom
        lin_wrt_all_dofs,
        //! linearization with repsect to primary degrees of freedom
        lin_wrt_primary_dofs,
        //! linearization with respect to Lagrange multiplier degrees of freedom
        lin_wrt_lagrange_multiplier_dofs,
        //! linearization with respect to mixed degrees of freedom
        lin_wrt_mixed_dofs
      };

      //! merit function names
      enum MeritFctName : int
      {
        mrtfct_sum_of_squares,                //!< sum of squares merit function
        mrtfct_infeasibility_two_norm,        //!< infeasibility merit function
        mrtfct_infeasibility_two_norm_active, /*!< infeasibility merit function,
                                               *  considering only the active contributions */
        mrtfct_energy,                        //!< representative energy value
        mrtfct_vague                          //!< unspecified
      };

      inline std::string merit_func_name_to_string(const MeritFctName mrt_func)
      {
        switch (mrt_func)
        {
          case mrtfct_sum_of_squares:
            return "mrtfct_sum_of_squares";
          case mrtfct_infeasibility_two_norm:
            return "mrtfct_infeasibility_two_norm";
          case mrtfct_infeasibility_two_norm_active:
            return "mrtfct_infeasibility_two_norm_active";
          case mrtfct_energy:
            return "mrtfct_energy";
          case mrtfct_vague:
            return "mrtfct_vague";
          default:
            return "INVALID_merit_function_name";
        }
      }

    }  // namespace MeritFunction

    namespace StatusTest
    {
      //! map status type (outer status test) to std::string
      inline std::string status_type_to_string(const enum ::NOX::StatusTest::StatusType stype)
      {
        switch (stype)
        {
          case ::NOX::StatusTest::Unevaluated:
            return "Unevaluated";
          case ::NOX::StatusTest::Unconverged:
            return "Unconverged";
          case ::NOX::StatusTest::Converged:
            return "Converged";
          case ::NOX::StatusTest::Failed:
            return "Failed";
          default:
            return "Unknown StatusType";
        }
      };

      //! Supported quantity names for distinguished status tests
      enum QuantityType : int
      {
        quantity_unknown,         ///< unknown quantity (dummy)
        quantity_structure,       ///< check structural quantities
        quantity_contact_normal,  ///< check (semi-smooth) contact quantities (normal/frictionless)
        quantity_contact_friction,    ///< check (semi-smooth) contact quantities (frictionless)
        quantity_meshtying,           ///< check meshtying quantities
        quantity_cardiovascular0d,    ///< check 0d cardiovascular quantities
        quantity_lag_pen_constraint,  ///< check Lagrange/penalty enforced constraint quantities
        quantity_pressure,            ///< check pressure dofs
        quantity_eas,                 ///< check eas dofs
        quantity_levelset_reinit,     ///< check levelset reinitialization
        quantity_constraints,         ///< check constraint framework quantities
        quantity_beaminteraction_lm,  ///< check lagrange multiplier quantities for beaminteraction
      };

      /// Map quantity name to std::string
      inline std::string quantity_type_to_string(const QuantityType type)
      {
        switch (type)
        {
          case quantity_structure:
            return "Structure";
          case quantity_contact_normal:
            return "Contact-Normal";
          case quantity_contact_friction:
            return "Contact-Friction";
          case quantity_meshtying:
            return "Meshtying";
          case quantity_cardiovascular0d:
            return "Cardiovascular0D";
          case quantity_lag_pen_constraint:
            return "Lag-Pen-Constraint";
          case quantity_pressure:
            return "Pressure";
          case quantity_eas:
            return "EAS";
          case quantity_levelset_reinit:
            return "LevelSet-Reinit";
          case quantity_constraints:
            return "Constraints";
          case quantity_beaminteraction_lm:
            return "Beaminteraction-LM";
          case quantity_unknown:
          default:
            return "unknown quantity type";
        }
      };

      /// Map std::string to quantity type
      inline QuantityType string_to_quantity_type(const std::string& name)
      {
        QuantityType type = quantity_unknown;

        if (name == "Structure")
          type = quantity_structure;
        else if (name == "Contact-Normal")
          type = quantity_contact_normal;
        else if (name == "Contact-Friction")
          type = quantity_contact_friction;
        else if (name == "Meshtying")
          type = quantity_meshtying;
        else if (name == "Cardiovascular0D")
          type = quantity_cardiovascular0d;
        else if (name == "Lag-Pen-Constraint")
          type = quantity_lag_pen_constraint;
        else if (name == "Pressure")
          type = quantity_pressure;
        else if (name == "EAS")
          type = quantity_eas;
        else if (name == "LevelSet-Reinit")
          type = quantity_levelset_reinit;
        else if (name == "Constraints")
          type = quantity_constraints;
        else if (name == "Beaminteraction-LM")
          type = quantity_beaminteraction_lm;

        return type;
      };
    }  // namespace StatusTest
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
