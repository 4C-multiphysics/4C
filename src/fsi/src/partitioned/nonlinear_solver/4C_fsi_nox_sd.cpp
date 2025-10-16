// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_sd.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_group.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Common.H>
#include <NOX_Epetra_Interface_Required.H>
#include <NOX_GlobalData.H>
#include <NOX_Solver_Generic.H>
#include <NOX_Utils.H>
#include <Teuchos_ParameterList.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN


NOX::FSI::SDRelaxation::SDRelaxation(
    const Teuchos::RCP<::NOX::Utils>& utils, Teuchos::ParameterList& params)
    : utils_(utils)
{
}



bool NOX::FSI::SDRelaxation::reset(
    const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params)
{
  utils_ = gd->getUtils();
  // Teuchos::ParameterList& p = params.sublist("SDRelaxation");
  return true;
}


bool NOX::FSI::SDRelaxation::compute(::NOX::Abstract::Group& newgrp, double& step,
    const ::NOX::Abstract::Vector& dir, const ::NOX::Solver::Generic& s)
{
  if (utils_->isPrintType(::NOX::Utils::InnerIteration))
  {
    utils_->out() << "\n"
                  << ::NOX::Utils::fill(72) << "\n"
                  << "-- SDRelaxation Line Search -- \n";
  }

  const ::NOX::Abstract::Group& oldgrp = s.getPreviousSolutionGroup();
  auto& grp = dynamic_cast<NOX::Nln::GroupBase&>(newgrp);

  // Perform single-step linesearch

  // Note that the following could be wrapped with a while loop to allow
  // iterations to be attempted

  double numerator = oldgrp.getF().innerProduct(dir);
  double denominator =
      compute_directional_derivative(dir, *grp.get_required_interface()).innerProduct(dir);

  step = -numerator / denominator;
  utils_->out() << "          RELAX = " << std::setw(5) << step << "\n";

  newgrp.computeX(oldgrp, dir, step);
  newgrp.computeF();

  double checkOrthogonality = fabs(newgrp.getF().innerProduct(dir));

  if (utils_->isPrintType(::NOX::Utils::InnerIteration))
  {
    utils_->out() << std::setw(3) << "1" << ":";
    utils_->out() << " step = " << utils_->sciformat(step);
    utils_->out() << " orth = " << utils_->sciformat(checkOrthogonality);
    utils_->out() << "\n" << ::NOX::Utils::fill(72) << "\n" << std::endl;
  }

  // write omega
  double fnorm = oldgrp.getF().norm();
  if (Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(
          dynamic_cast<const NOX::Nln::Vector&>(oldgrp.getF()).getEpetraVector().Comm())) == 0)
  {
    static int count;
    static std::ofstream* out;
    if (out == nullptr)
    {
      std::string s = Global::Problem::instance()->output_control_file()->file_name();
      s.append(".omega");
      out = new std::ofstream(s.c_str());
    }
    (*out) << count << " " << step << " " << fnorm << "\n";
    count += 1;
    out->flush();
  }

  return true;
}


::NOX::Abstract::Vector& NOX::FSI::SDRelaxation::compute_directional_derivative(
    const ::NOX::Abstract::Vector& dir, ::NOX::Epetra::Interface::Required& interface)
{
  // Allocate space for vecPtr and grpPtr if necessary
  if (!(vec_ptr_)) vec_ptr_ = dir.clone(::NOX::ShapeCopy);

  const auto& edir = dynamic_cast<const NOX::Nln::Vector&>(dir);
  auto& evec = dynamic_cast<NOX::Nln::Vector&>(*vec_ptr_);

  // we do not want the group to remember this solution
  // and we want to set our own flag
  // this tells computeF to do a SD relaxation calculation
  interface.computeF(
      edir.getEpetraVector(), evec.getEpetraVector(), ::NOX::Epetra::Interface::Required::User);

  return *vec_ptr_;
}

FOUR_C_NAMESPACE_CLOSE
