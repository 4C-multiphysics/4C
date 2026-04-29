// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_visco_generalizedmaxwell.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::GeneralizedMaxwell::GeneralizedMaxwell(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      numbranch_(matdata.parameters.get<int>("NUMBRANCH")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
      solve_(matdata.parameters.get<std::string>("SOLVE"))
{
  if (solve_ != "OneStepTheta" && solve_ != "ExponentialTimeDiscretization")
    FOUR_C_THROW(
        "Invalid input for SOLVE='{}' in VISCO_GeneralizedMaxwell (MAT {}). Use "
        "OneStepTheta or ExponentialTimeDiscretization.",
        solve_, matdata.id);
}

Mat::Elastic::GeneralizedMaxwell::GeneralizedMaxwell(Mat::Elastic::PAR::GeneralizedMaxwell* params)
    : params_(params), branchespotsum_(0), branchtau_(0), internalpotsum_(0)
{
  // loop over materials of GeneralizedMaxwell (branches)
  std::vector<int>::const_iterator m;
  for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
  {
    // make sure the summands of the current branch is empty
    internalpotsum_.clear();
    // get parameters of each branch
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::Summand> visco_branch = Mat::Elastic::Summand::factory(matid);

    double tau = -1.0;
    int branchmatid = -1;

    visco_branch->read_material_parameters(tau, branchmatid);

    if (tau <= 0.0)
      FOUR_C_THROW(
          "Invalid branch relaxation time TAU={}. TAU has to be positive in "
          "VISCO_GeneralizedMaxwellBranch.",
          tau);

    if (branchmatid <= 0)
      FOUR_C_THROW(
          "Invalid branch elasticity material id MATID={}. MATID has to be positive in "
          "VISCO_GeneralizedMaxwellBranch.",
          branchmatid);

    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(branchmatid);
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");

    // write summand in the vector of summands of each branch
    internalpotsum_.push_back(sum);

    // write into vector of summands of the GeneralizedMaxwell material
    branchespotsum_.push_back(internalpotsum_);
    branchtau_.push_back(tau);

  }  // end for-loop over branches
}

void Mat::Elastic::GeneralizedMaxwell::read_material_parameters(
    int& numbranch, const std::vector<int>*& matids, std::string& solve)
{
  numbranch = params_->numbranch_;
  matids = &params_->matids_;
  solve = params_->solve_;
}

// Viscobranch
Mat::Elastic::PAR::ViscoBranch::ViscoBranch(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      tau_(matdata.parameters.get<double>("TAU")),
      matid_(matdata.parameters.get<int>("MATID"))
{
}

Mat::Elastic::ViscoBranch::ViscoBranch(Mat::Elastic::PAR::ViscoBranch* params) : params_(params) {}

void Mat::Elastic::ViscoBranch::read_material_parameters(double& tau, int& matid)
{
  tau = params_->tau_;
  matid = params_->matid_;
}

FOUR_C_NAMESPACE_CLOSE
