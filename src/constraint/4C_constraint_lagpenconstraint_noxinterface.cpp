/*---------------------------------------------------------------------*/
/*! \file

\brief Concrete mplementation of all the %NOX::Nln::CONSTRAINT::Interface::Required
       (pure) virtual routines.

\level 3


\date July 29, 2016

*/
/*---------------------------------------------------------------------*/

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
    : isinit_(false), issetup_(false), gstate_ptr_(Teuchos::null)
{
  // should stay empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
LAGPENCONSTRAINT::NoxInterfacePrec::NoxInterfacePrec()
    : isinit_(false), issetup_(false), gstate_ptr_(Teuchos::null)
{
  // should stay empty
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterface::init(
    const Teuchos::RCP<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr)
{
  issetup_ = false;

  gstate_ptr_ = gstate_ptr;

  // set flag at the end
  isinit_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void LAGPENCONSTRAINT::NoxInterfacePrec::init(
    const Teuchos::RCP<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr)
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


  auto F_copy = Teuchos::RCP(new Core::LinAlg::Vector<double>(F));
  Teuchos::RCP<Core::LinAlg::Vector<double>> constrRhs =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, *F_copy);

  // no constraint contributions present
  if (constrRhs.is_null()) return 0.0;

  Teuchos::RCP<const ::NOX::Epetra::Vector> constrRhs_nox = Teuchos::RCP(new ::NOX::Epetra::Vector(
      constrRhs->get_ptr_of_Epetra_Vector(), ::NOX::Epetra::Vector::CreateView));


  double constrNorm = -1.0;
  constrNorm = constrRhs_nox->norm(type);
  if (isScaled) constrNorm /= static_cast<double>(constrRhs_nox->length());

  return constrNorm;
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

  auto xOld_copy = Teuchos::RCP(new Core::LinAlg::Vector<double>(xOld));
  auto xNew_copy = Teuchos::RCP(new Core::LinAlg::Vector<double>(xNew));
  // export the constraint solution
  Teuchos::RCP<Core::LinAlg::Vector<double>> lagincr_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, *xOld_copy);
  Teuchos::RCP<const Core::LinAlg::Vector<double>> lagnew_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, *xNew_copy);

  lagincr_ptr->Update(1.0, *lagnew_ptr, -1.0);
  Teuchos::RCP<const ::NOX::Epetra::Vector> lagincr_nox_ptr =
      Teuchos::RCP(new ::NOX::Epetra::Vector(
          lagincr_ptr->get_ptr_of_Epetra_Vector(), ::NOX::Epetra::Vector::CreateView));

  rms = NOX::Nln::Aux::root_mean_square_norm(
      aTol, rTol, lagnew_ptr, lagincr_ptr, disable_implicit_weighting);

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

  auto xOld_copy = Teuchos::RCP(new Core::LinAlg::Vector<double>(xOld));
  auto xNew_copy = Teuchos::RCP(new Core::LinAlg::Vector<double>(xNew));

  // export the constraint solution
  Teuchos::RCP<Core::LinAlg::Vector<double>> lagincr_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, *xOld_copy);
  Teuchos::RCP<const Core::LinAlg::Vector<double>> lagnew_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, *xNew_copy);

  lagincr_ptr->Update(1.0, *lagnew_ptr, -1.0);
  Teuchos::RCP<const ::NOX::Epetra::Vector> lagincr_nox_ptr =
      Teuchos::RCP(new ::NOX::Epetra::Vector(
          lagincr_ptr->get_ptr_of_Epetra_Vector(), ::NOX::Epetra::Vector::CreateView));

  double updatenorm = -1.0;

  updatenorm = lagincr_nox_ptr->norm(type);
  // do scaling if desired
  if (isScaled) updatenorm /= static_cast<double>(lagincr_nox_ptr->length());

  return updatenorm;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double LAGPENCONSTRAINT::NoxInterface::get_previous_lagrange_multiplier_norms(
    const Core::LinAlg::Vector<double>& xOld, NOX::Nln::StatusTest::QuantityType checkQuantity,
    ::NOX::Abstract::Vector::NormType type, bool isScaled) const
{
  if (checkQuantity != NOX::Nln::StatusTest::quantity_lag_pen_constraint) return -1.0;

  auto xOld_copy = Teuchos::RCP(new Core::LinAlg::Vector<double>(xOld));

  // export the constraint solution
  Teuchos::RCP<Core::LinAlg::Vector<double>> lagold_ptr =
      gstate_ptr_->extract_model_entries(Inpar::Solid::model_lag_pen_constraint, *xOld_copy);

  Teuchos::RCP<const ::NOX::Epetra::Vector> lagold_nox_ptr = Teuchos::RCP(new ::NOX::Epetra::Vector(
      lagold_ptr->get_ptr_of_Epetra_Vector(), ::NOX::Epetra::Vector::CreateView));

  double lagoldnorm = -1.0;

  lagoldnorm = lagold_nox_ptr->norm(type);
  // do scaling if desired
  if (isScaled) lagoldnorm /= static_cast<double>(lagold_nox_ptr->length());

  return lagoldnorm;
}



/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool LAGPENCONSTRAINT::NoxInterfacePrec::is_saddle_point_system() const
{
  Teuchos::RCP<const Core::FE::Discretization> dis = gstate_ptr_->get_discret();

  // ---------------------------------------------------------------------------
  // check type of constraint conditions (Lagrange multiplier vs. penalty)
  // ---------------------------------------------------------------------------
  bool have_lag_constraint = false;
  std::vector<Core::Conditions::Condition*> lagcond_volconstr3d(0);
  std::vector<Core::Conditions::Condition*> lagcond_areaconstr3d(0);
  std::vector<Core::Conditions::Condition*> lagcond_areaconstr2d(0);
  std::vector<Core::Conditions::Condition*> lagcond_mpconline2d(0);
  std::vector<Core::Conditions::Condition*> lagcond_mpconplane3d(0);
  std::vector<Core::Conditions::Condition*> lagcond_mpcnormcomp3d(0);
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
    std::vector<Teuchos::RCP<Epetra_Map>>& maps) const
{
  //  std::cout << "fill_maps_for_preconditioner" << std::endl;
  return;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool LAGPENCONSTRAINT::NoxInterfacePrec::computePreconditioner(
    const Epetra_Vector& x, Epetra_Operator& M, Teuchos::ParameterList* precParams)
{
  //  std::cout << "computePreconditioner" << std::endl;
  check_init_setup();
  // currently not supported
  // ToDo add the scaled thickness conditioning (STC) approach here
  return false;
}

FOUR_C_NAMESPACE_CLOSE
