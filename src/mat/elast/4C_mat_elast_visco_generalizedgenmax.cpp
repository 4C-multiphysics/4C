// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_visco_generalizedgenmax.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::GeneralizedGenMax::GeneralizedGenMax(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      numbranch_(matdata.parameters.get<int>("NUMBRANCH")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
      solve_(matdata.parameters.get<std::string>("SOLVE"))
{
}

Mat::Elastic::GeneralizedGenMax::GeneralizedGenMax(Mat::Elastic::PAR::GeneralizedGenMax* params)
    : params_(params), branchespotsum_(0), internalpotsum_(0)
{
  // loop over materials of GeneralizedGenMax (branches)
  std::vector<int>::const_iterator m;
  for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
  {
    // make sure the summands of the current branch is empty
    internalpotsum_.clear();
    // get parameters of each branch
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::Summand> ViscoBranch = Mat::Elastic::Summand::factory(matid);

    double nummat = -1.0;
    const std::vector<int>* branchmatids = nullptr;

    ViscoBranch->read_material_parameters(nummat, branchmatids);

    // loop over materials of ViscoBranch (components of the viscoelastic branch)
    for (int i = 0; i < nummat; ++i)
    {
      // get parameters of each component
      int curmatid = branchmatids->at(i);
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(curmatid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      // write summand in the vector of summands of each branch
      internalpotsum_.push_back(sum);
    }

    // write into vector of summands of the GeneralizedGenMax material
    branchespotsum_.push_back(internalpotsum_);

  }  // end for-loop over branches
}

void Mat::Elastic::GeneralizedGenMax::read_material_parameters(
    int& numbranch, const std::vector<int>*& matids, std::string& solve)
{
  numbranch = params_->numbranch_;
  matids = &params_->matids_;
  solve = params_->solve_;
}

// Viscobranch
Mat::Elastic::PAR::ViscoBranch::ViscoBranch(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nummat_(matdata.parameters.get<int>("NUMMAT")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS"))
{
}

Mat::Elastic::ViscoBranch::ViscoBranch(Mat::Elastic::PAR::ViscoBranch* params) : params_(params) {}

void Mat::Elastic::ViscoBranch::read_material_parameters(
    double& nummat, const std::vector<int>*& matids)
{
  nummat = params_->nummat_;
  matids = &params_->matids_;
}

// Viscopart
Mat::Elastic::PAR::ViscoPart::ViscoPart(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), tau_(matdata.parameters.get<double>("TAU"))
{
}

Mat::Elastic::ViscoPart::ViscoPart(Mat::Elastic::PAR::ViscoPart* params) : params_(params) {}

void Mat::Elastic::ViscoPart::read_material_parameters(double& tau) { tau = params_->tau_; }

FOUR_C_NAMESPACE_CLOSE
