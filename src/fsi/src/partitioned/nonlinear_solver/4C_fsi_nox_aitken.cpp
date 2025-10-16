// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_nox_aitken.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_solver_nonlin_nox_vector.hpp"

#include <NOX_Abstract_Group.H>
#include <NOX_Abstract_Vector.H>
#include <NOX_Common.H>
#include <NOX_GlobalData.H>
#include <NOX_Solver_Generic.H>
#include <Teuchos_ParameterList.hpp>

#include <fstream>

FOUR_C_NAMESPACE_OPEN

NOX::FSI::AitkenRelaxation::AitkenRelaxation(
    const Teuchos::RCP<::NOX::Utils>& utils, Teuchos::ParameterList& params)
    : utils_(utils)
{
  Teuchos::ParameterList& p = params.sublist("Aitken");
  nu_ = p.get("Start nu", 0.0);

  maxstep_ = p.get("max step size", 0.0);
  if (maxstep_ > 0.0) nu_ = 1.0 - maxstep_;

  minstep_ = p.get("min step size", -1.0);

  restart_ = p.get<int>("restart", 0);

  restart_omega_ = p.get<double>("restart_omega", 0.0);
}



bool NOX::FSI::AitkenRelaxation::reset(
    const Teuchos::RCP<::NOX::GlobalData>& gd, Teuchos::ParameterList& params)
{
  // We might want to constrain the step size of the first relaxation
  // in a new time step.
  if (maxstep_ > 0.0 && maxstep_ < 1.0 - nu_) nu_ = 1. - maxstep_;

  if (minstep_ > 1.0 - nu_) nu_ = 1.0 - minstep_;

  if (!is_null(del_))
  {
    del_->init(1e20);
  }
  utils_ = gd->getUtils();
  return true;
}


bool NOX::FSI::AitkenRelaxation::compute(::NOX::Abstract::Group& grp, double& step,
    const ::NOX::Abstract::Vector& dir, const ::NOX::Solver::Generic& s)
{
  if (utils_->isPrintType(::NOX::Utils::InnerIteration))
  {
    utils_->out() << "\n"
                  << ::NOX::Utils::fill(72) << "\n"
                  << "-- Aitken Line Search -- \n";
  }

  const ::NOX::Abstract::Group& oldGrp = s.getPreviousSolutionGroup();
  const ::NOX::Abstract::Vector& F = oldGrp.getF();


  if (is_null(del_))
  {
    del_ = F.clone(::NOX::ShapeCopy);
    del2_ = F.clone(::NOX::ShapeCopy);
    del_->init(1.0e20);
    del2_->init(0.0);
  }

  del2_->update(1, *del_, 1, F);
  del_->update(-1, F);

  const double top = del2_->innerProduct(*del_);
  const double den = del2_->innerProduct(*del2_);

  // in case of a restart, we use the omega that
  // was written as restart output
  if (not restart_)
    nu_ = nu_ + (nu_ - 1.) * top / den;
  else
    nu_ = 1.0 - restart_omega_;

  // check constraints for step size
  if (maxstep_ > 0.0 && maxstep_ < 1.0 - nu_) nu_ = 1. - maxstep_;

  if (minstep_ > 1.0 - nu_) nu_ = 1.0 - minstep_;

  // calc step
  step = 1. - nu_;

  utils_->out() << "          RELAX = " << std::setw(5) << step << "\n";

  grp.computeX(oldGrp, dir, step);

  // Calculate F anew here. This results in another FSI loop. However
  // the group will store the result, so it will be reused until the
  // group's x is changed again. We do not waste anything.
  grp.computeF();

  // is this reasonable at this point?
  double checkOrthogonality = fabs(grp.getF().innerProduct(dir));

  if (utils_->isPrintType(::NOX::Utils::InnerIteration))
  {
    utils_->out() << std::setw(3) << "1" << ":";
    utils_->out() << " step = " << utils_->sciformat(step);
    utils_->out() << " orth = " << utils_->sciformat(checkOrthogonality);
    utils_->out() << "\n" << ::NOX::Utils::fill(72) << "\n" << std::endl;
  }

  // write omega
  double fnorm = grp.getF().norm();
  if (Core::Communication::my_mpi_rank(Core::Communication::unpack_epetra_comm(
          dynamic_cast<const NOX::Nln::Vector&>(F).getEpetraVector().Comm())) == 0)
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

  // reset restart flag
  restart_ = false;

  return true;
}


double NOX::FSI::AitkenRelaxation::get_omega() { return 1. - nu_; }

FOUR_C_NAMESPACE_CLOSE
