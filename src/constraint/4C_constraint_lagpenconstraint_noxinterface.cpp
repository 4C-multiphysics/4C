// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_lagpenconstraint_noxinterface.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_aux.hpp"

#include <NOX_Epetra_Vector.H>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
LAGPENCONSTRAINT::NoxInterface::NoxInterface()
    : isinit_(false), issetup_(false), gstate_ptr_(nullptr)
{
  // should stay empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
LAGPENCONSTRAINT::NoxInterfacePrec::NoxInterfacePrec()
    : isinit_(false), issetup_(false), gstate_ptr_(nullptr)
{
  // should stay empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterface::init(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr)
{
  issetup_ = false;

  gstate_ptr_ = gstate_ptr;

  // set flag at the end
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterfacePrec::init(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr)
{
  issetup_ = false;

  gstate_ptr_ = gstate_ptr;

  // set flag at the end
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterface::setup()
{
  check_init();

  // set flag at the end
  issetup_ = true;
}
/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterfacePrec::setup()
{
  check_init();

  // set flag at the end
  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double LAGPENCONSTRAINT::NoxInterface::get_constraint_rhs_norms(
    const Core::LinAlg::Vector<double>& F, NOX::Nln::StatusTest::QuantityType chQ,
    ::NOX::Abstract::Vector::NormType type, bool isScaled) const
{
  if (chQ != NOX::Nln::StatusTest::quantity_lag_pen_constraint) return -1.0;


  Core::LinAlg::Vector<double> F_copy(F);
  std::shared_ptr<Core::LinAlg::Vector<double>> constrRhs =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, F_copy);

  // no constraint contributions present
  if (!constrRhs) return 0.0;

  return NOX::Nln::Aux::calc_vector_norm(*constrRhs, type, isScaled);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double LAGPENCONSTRAINT::NoxInterface::get_lagrange_multiplier_update_rms(
    const Core::LinAlg::Vector<double>& xNew, const Core::LinAlg::Vector<double>& xOld, double aTol,
    double rTol, NOX::Nln::StatusTest::QuantityType checkQuantity,
    bool disable_implicit_weighting) const
{
  if (checkQuantity != NOX::Nln::StatusTest::quantity_lag_pen_constraint) return -1.0;

  double rms = -1.0;

  Core::LinAlg::Vector<double> xOld_copy(xOld);
  Core::LinAlg::Vector<double> xNew_copy(xNew);
  // export the constraint solution
  std::shared_ptr<Core::LinAlg::Vector<double>> lagincr_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, xOld_copy);
  std::shared_ptr<const Core::LinAlg::Vector<double>> lagnew_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, xNew_copy);

  lagincr_ptr->update(1.0, *lagnew_ptr, -1.0);

  rms = NOX::Nln::Aux::root_mean_square_norm(
      aTol, rTol, *lagnew_ptr, *lagincr_ptr, disable_implicit_weighting);

  return rms;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double LAGPENCONSTRAINT::NoxInterface::get_lagrange_multiplier_update_norms(
    const Core::LinAlg::Vector<double>& xNew, const Core::LinAlg::Vector<double>& xOld,
    NOX::Nln::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
    bool isScaled) const
{
  if (checkQuantity != NOX::Nln::StatusTest::quantity_lag_pen_constraint) return -1.0;

  Core::LinAlg::Vector<double> xOld_copy(xOld);
  Core::LinAlg::Vector<double> xNew_copy(xNew);

  // export the constraint solution
  std::shared_ptr<Core::LinAlg::Vector<double>> lagincr_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, xOld_copy);
  std::shared_ptr<const Core::LinAlg::Vector<double>> lagnew_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, xNew_copy);

  lagincr_ptr->update(1.0, *lagnew_ptr, -1.0);

  return NOX::Nln::Aux::calc_vector_norm(*lagincr_ptr, type, isScaled);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double LAGPENCONSTRAINT::NoxInterface::get_previous_lagrange_multiplier_norms(
    const Core::LinAlg::Vector<double>& xOld, NOX::Nln::StatusTest::QuantityType checkQuantity,
    ::NOX::Abstract::Vector::NormType type, bool isScaled) const
{
  if (checkQuantity != NOX::Nln::StatusTest::quantity_lag_pen_constraint) return -1.0;

  Core::LinAlg::Vector<double> xOld_copy(xOld);

  // export the constraint solution
  std::shared_ptr<Core::LinAlg::Vector<double>> lagold_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, xOld_copy);

  return NOX::Nln::Aux::calc_vector_norm(*lagold_ptr, type, isScaled);
}



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool LAGPENCONSTRAINT::NoxInterfacePrec::is_saddle_point_system() const
{
  std::shared_ptr<const Core::FE::Discretization> dis = gstate_ptr_->get_discret();

  // ---------------------------------------------------------------------------
  // check type of constraint conditions (Lagrange multiplier vs. penalty)
  // ---------------------------------------------------------------------------
  bool have_lag_constraint = false;
  std::vector<const Core::Conditions::Condition*> lagcond_volconstr3d;
  std::vector<const Core::Conditions::Condition*> lagcond_areaconstr3d;
  std::vector<const Core::Conditions::Condition*> lagcond_areaconstr2d;
  std::vector<const Core::Conditions::Condition*> lagcond_mpconline2d;
  std::vector<const Core::Conditions::Condition*> lagcond_mpconplane3d;
  std::vector<const Core::Conditions::Condition*> lagcond_mpcnormcomp3d;
  dis->get_condition("VolumeConstraint_3D", lagcond_volconstr3d);
  dis->get_condition("AreaConstraint_3D", lagcond_areaconstr3d);
  dis->get_condition("AreaConstraint_2D", lagcond_areaconstr2d);
  dis->get_condition("MPC_NodeOnLine_2D", lagcond_mpconline2d);
  dis->get_condition("MPC_NodeOnPlane_3D", lagcond_mpconplane3d);
  dis->get_condition("MPC_NormalComponent_3D", lagcond_mpcnormcomp3d);
  if (lagcond_volconstr3d.size() or lagcond_areaconstr3d.size() or lagcond_areaconstr2d.size() or
      lagcond_mpconline2d.size() or lagcond_mpconplane3d.size() or lagcond_mpcnormcomp3d.size())
    have_lag_constraint = true;

  return have_lag_constraint;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool LAGPENCONSTRAINT::NoxInterfacePrec::is_condensed_system() const
{
  //  std::cout << "is_condensed_system" << std::endl;
  return false;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterfacePrec::fill_maps_for_preconditioner(
    std::vector<Teuchos::RCP<Core::LinAlg::Map>>& maps) const
{
}

FOUR_C_NAMESPACE_CLOSE
