/*----------------------------------------------------------------------*/
/*! \file

\brief Declaration

\level 1

*/
/*----------------------------------------------------------------------*/

#include "4C_linear_solver_amgnxn_vcycle.hpp"

#include "4C_linear_solver_amgnxn_smoothers.hpp"
#include "4C_utils_exceptions.hpp"

#include <MueLu_EpetraOperator.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Xpetra_MultiVectorFactory.hpp>

#include <iostream>

FOUR_C_NAMESPACE_OPEN


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::Vcycle::Vcycle(int NumLevels, int NumSweeps, int FirstLevel)
    : num_levels_(NumLevels),
      num_sweeps_(NumSweeps),
      first_level_(FirstLevel),
      avec_(NumLevels, Teuchos::null),
      pvec_(NumLevels - 1, Teuchos::null),
      rvec_(NumLevels - 1, Teuchos::null),
      svec_pre_(NumLevels, Teuchos::null),
      svec_pos_(NumLevels - 1, Teuchos::null),
      flag_set_up_a_(false),
      flag_set_up_p_(false),
      flag_set_up_r_(false),
      flag_set_up_pre_(false),
      flag_set_up_pos_(false)
{
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::Vcycle::set_operators(
    std::vector<Teuchos::RCP<BlockedMatrix>> Avec)
{
  if ((int)Avec.size() != num_levels_) FOUR_C_THROW("Error in Setting Avec_: Size dismatch.");
  for (int i = 0; i < num_levels_; i++)
  {
    if (Avec[i] == Teuchos::null) FOUR_C_THROW("Error in Setting Avec_: Null pointer.");
    avec_[i] = Avec[i];
  }
  flag_set_up_a_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::Vcycle::set_projectors(
    std::vector<Teuchos::RCP<BlockedMatrix>> Pvec)
{
  if ((int)Pvec.size() != num_levels_ - 1) FOUR_C_THROW("Error in Setting Pvec_: Size dismatch.");
  for (int i = 0; i < num_levels_ - 1; i++)
  {
    if (Pvec[i] == Teuchos::null) FOUR_C_THROW("Error in Setting Pvec_: Null pointer.");
    pvec_[i] = Pvec[i];
  }
  flag_set_up_p_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::Vcycle::set_restrictors(
    std::vector<Teuchos::RCP<BlockedMatrix>> Rvec)
{
  if ((int)Rvec.size() != num_levels_ - 1) FOUR_C_THROW("Error in Setting Rvec_: Size dismatch.");
  for (int i = 0; i < num_levels_ - 1; i++)
  {
    if (Rvec[i] == Teuchos::null) FOUR_C_THROW("Error in Setting Rvec_: Null pointer.");
    rvec_[i] = Rvec[i];
  }
  flag_set_up_r_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::Vcycle::set_pre_smoothers(
    std::vector<Teuchos::RCP<GenericSmoother>> SvecPre)
{
  if ((int)SvecPre.size() != num_levels_) FOUR_C_THROW("Error in Setting SvecPre: Size dismatch.");
  for (int i = 0; i < num_levels_; i++)
  {
    if (SvecPre[i] == Teuchos::null) FOUR_C_THROW("Error in Setting SvecPre: Null pointer.");
    svec_pre_[i] = SvecPre[i];
  }
  flag_set_up_pre_ = true;
  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::Vcycle::set_pos_smoothers(
    std::vector<Teuchos::RCP<GenericSmoother>> SvecPos)
{
  if ((int)SvecPos.size() != num_levels_ - 1)
    FOUR_C_THROW("Error in Setting SvecPos: Size dismatch.");
  for (int i = 0; i < num_levels_ - 1; i++)
  {
    if (SvecPos[i] == Teuchos::null) FOUR_C_THROW("Error in Setting SvecPos: Null pointer.");
    svec_pos_[i] = SvecPos[i];
  }
  flag_set_up_pos_ = true;
  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::Vcycle::do_vcycle(
    const BlockedVector& X, BlockedVector& Y, int level, bool InitialGuessIsZero) const
{
  if (level != num_levels_ - 1)  // Perform one iteration of the V-cycle
  {
    // Apply presmoother
    svec_pre_[level]->solve(X, Y, InitialGuessIsZero);

    // Compute residual
    BlockedVector DX = X.deep_copy();
    avec_[level]->apply(Y, DX);
    DX.update(1.0, X, -1.0);

    //  Create coarser representation of the residual
    int NV = X.get_vector(0)->NumVectors();
    Teuchos::RCP<BlockedVector> DXcoarse = rvec_[level]->new_range_blocked_vector(NV, false);
    rvec_[level]->apply(DX, *DXcoarse);

    // Damp error with coarser levels
    Teuchos::RCP<BlockedVector> DYcoarse = pvec_[level]->new_domain_blocked_vector(NV, false);
    do_vcycle(*DXcoarse, *DYcoarse, level + 1, true);

    // Compute correction
    BlockedVector DY = Y.deep_copy();
    pvec_[level]->apply(*DYcoarse, DY);
    Y.update(1.0, DY, 1.0);

    // Apply post smoother
    svec_pos_[level]->solve(X, Y, false);
  }
  else  // Apply presmoother
  {
    svec_pre_[level]->solve(X, Y, InitialGuessIsZero);
  }


  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::Vcycle::solve(
    const BlockedVector& X, BlockedVector& Y, bool InitialGuessIsZero) const
{
  // Check if everithing is set up
  if (!flag_set_up_a_) FOUR_C_THROW("Operators missing");
  if (!flag_set_up_p_) FOUR_C_THROW("Projectors missing");
  if (!flag_set_up_r_) FOUR_C_THROW("Restrictors missing");
  if (!flag_set_up_pre_) FOUR_C_THROW("Pre-smoothers missing");
  if (!flag_set_up_pos_) FOUR_C_THROW("Post-smoothers missing");

  // Work!
  for (int i = 0; i < num_sweeps_; i++)
    do_vcycle(X, Y, first_level_, InitialGuessIsZero and i == 0);
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
Core::LinearSolver::AMGNxN::VcycleSingle::VcycleSingle(int NumLevels, int NumSweeps, int FirstLevel)
    : num_levels_(NumLevels),
      num_sweeps_(NumSweeps),
      first_level_(FirstLevel),
      avec_(NumLevels, Teuchos::null),
      pvec_(NumLevels - 1, Teuchos::null),
      rvec_(NumLevels - 1, Teuchos::null),
      svec_pre_(NumLevels, Teuchos::null),
      svec_pos_(NumLevels - 1, Teuchos::null),
      flag_set_up_a_(false),
      flag_set_up_p_(false),
      flag_set_up_r_(false),
      flag_set_up_pre_(false),
      flag_set_up_pos_(false)
{
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::VcycleSingle::set_operators(
    std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Avec)
{
  if ((int)Avec.size() != num_levels_) FOUR_C_THROW("Error in Setting Avec_: Size dismatch.");
  for (int i = 0; i < num_levels_; i++)
  {
    if (Avec[i] == Teuchos::null) FOUR_C_THROW("Error in Setting Avec_: Null pointer.");
    avec_[i] = Avec[i];
  }
  flag_set_up_a_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::VcycleSingle::set_projectors(
    std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Pvec)
{
  if ((int)Pvec.size() != num_levels_ - 1) FOUR_C_THROW("Error in Setting Pvec_: Size dismatch.");
  for (int i = 0; i < num_levels_ - 1; i++)
  {
    if (Pvec[i] == Teuchos::null) FOUR_C_THROW("Error in Setting Pvec_: Null pointer.");
    pvec_[i] = Pvec[i];
  }
  flag_set_up_p_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::VcycleSingle::set_restrictors(
    std::vector<Teuchos::RCP<Core::LinAlg::SparseMatrix>> Rvec)
{
  if ((int)Rvec.size() != num_levels_ - 1) FOUR_C_THROW("Error in Setting Rvec_: Size dismatch.");
  for (int i = 0; i < num_levels_ - 1; i++)
  {
    if (Rvec[i] == Teuchos::null) FOUR_C_THROW("Error in Setting Rvec_: Null pointer.");
    rvec_[i] = Rvec[i];
  }
  flag_set_up_r_ = true;
  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::VcycleSingle::set_pre_smoothers(
    std::vector<Teuchos::RCP<SingleFieldSmoother>> SvecPre)
{
  if ((int)SvecPre.size() != num_levels_) FOUR_C_THROW("Error in Setting SvecPre: Size dismatch.");
  for (int i = 0; i < num_levels_; i++)
  {
    if (SvecPre[i] == Teuchos::null) FOUR_C_THROW("Error in Setting SvecPre: Null pointer.");
    svec_pre_[i] = SvecPre[i];
  }
  flag_set_up_pre_ = true;
  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/

void Core::LinearSolver::AMGNxN::VcycleSingle::set_pos_smoothers(
    std::vector<Teuchos::RCP<SingleFieldSmoother>> SvecPos)
{
  if ((int)SvecPos.size() != num_levels_ - 1)
    FOUR_C_THROW("Error in Setting SvecPos: Size dismatch.");
  for (int i = 0; i < num_levels_ - 1; i++)
  {
    if (SvecPos[i] == Teuchos::null) FOUR_C_THROW("Error in Setting SvecPos: Null pointer.");
    svec_pos_[i] = SvecPos[i];
  }
  flag_set_up_pos_ = true;
  return;
}

/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::VcycleSingle::do_vcycle(const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y, int level, bool InitialGuessIsZero) const
{
  if (level != num_levels_ - 1)  // Perform one iteration of the V-cycle
  {
    // Apply presmoother
    svec_pre_[level]->apply(X, Y, InitialGuessIsZero);

    // Compute residual
    int NV = X.NumVectors();
    Core::LinAlg::MultiVector<double> DX(X.Map(), NV, false);
    avec_[level]->Apply(Y, DX);
    DX.Update(1.0, X, -1.0);

    //  Create coarser representation of the residual
    const Epetra_Map& Map = rvec_[level]->range_map();
    Core::LinAlg::MultiVector<double> DXcoarse(Map, NV, false);
    rvec_[level]->Apply(DX, DXcoarse);

    // Damp error with coarser levels
    const Epetra_Map& Map2 = pvec_[level]->domain_map();
    Core::LinAlg::MultiVector<double> DYcoarse(Map2, NV, false);
    do_vcycle(DXcoarse, DYcoarse, level + 1, true);

    // Compute correction
    Core::LinAlg::MultiVector<double> DY(Y.Map(), NV, false);
    pvec_[level]->Apply(DYcoarse, DY);
    Y.Update(1.0, DY, 1.0);

    // Apply post smoother
    svec_pos_[level]->apply(X, Y, false);
  }
  else  // Apply presmoother
  {
    svec_pre_[level]->apply(X, Y, InitialGuessIsZero);
  }

  return;
}


/*------------------------------------------------------------------------------*/
/*------------------------------------------------------------------------------*/
void Core::LinearSolver::AMGNxN::VcycleSingle::apply(const Core::LinAlg::MultiVector<double>& X,
    Core::LinAlg::MultiVector<double>& Y, bool InitialGuessIsZero) const
{
  // Check if everithing is set up
  if (!flag_set_up_a_) FOUR_C_THROW("Operators missing");
  if (!flag_set_up_p_) FOUR_C_THROW("Projectors missing");
  if (!flag_set_up_r_) FOUR_C_THROW("Restrictors missing");
  if (!flag_set_up_pre_) FOUR_C_THROW("Pre-smoothers missing");
  if (!flag_set_up_pos_) FOUR_C_THROW("Post-smoothers missing");

  // Work!
  for (int i = 0; i < num_sweeps_; i++)
    do_vcycle(X, Y, first_level_, InitialGuessIsZero and i == 0);
  return;
}

FOUR_C_NAMESPACE_CLOSE
