// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_elasthyperbase.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_mat_mixture.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_prestress_strategy.hpp"
#include "4C_utils_enum.hpp"



FOUR_C_NAMESPACE_OPEN

// Constructor for the parameter class
Mixture::PAR::MixtureConstituentElastHyperBase::MixtureConstituentElastHyperBase(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituent(matdata),
      matid_prestress_strategy_(matdata.parameters.get<int>("PRESTRESS_STRATEGY")),
      nummat_(matdata.parameters.get<int>("NUMMAT")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS"))
{
  // check, if size of summands fits to the number of summands
  if (nummat_ != (int)matids_.size())
  {
    FOUR_C_THROW(
        "number of summands {} does not fit to the size of the summands vector"
        " {}",
        nummat_, matids_.size());
  }
}

// Constructor of the constituent holding the material parameters
Mixture::MixtureConstituentElastHyperBase::MixtureConstituentElastHyperBase(
    Mixture::PAR::MixtureConstituentElastHyperBase* params, int id)
    : MixtureConstituent(params, id),
      summand_properties_(),
      params_(params),
      potsum_(0),
      cosy_anisotropy_extension_()
{
  // Create summands
  for (const auto& matid : params_->matids_)
  {
    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
    if (sum == nullptr) FOUR_C_THROW("Failed to read elastic summand.");
    potsum_.push_back(sum);
  }

  // Create Prestress strategy
  if (params->get_prestressing_mat_id() > 0)
  {
    prestress_strategy_ =
        Mixture::PAR::PrestressStrategy::factory(params->get_prestressing_mat_id())
            ->create_prestress_strategy();
  }
}

// Pack the constituent
void Mixture::MixtureConstituentElastHyperBase::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  MixtureConstituent::pack_constituent(data);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
  summand_properties_.pack(data);

  add_to_pack(data, prestretch_);

  cosy_anisotropy_extension_.pack_anisotropy(data);

  if (prestress_strategy_ != nullptr) prestress_strategy_->pack(data);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{data};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (const auto& p : potsum_) p->pack_summand(data);
  }
}

// Unpack the constituent
void Mixture::MixtureConstituentElastHyperBase::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  MixtureConstituent::unpack_constituent(buffer);

  // make sure we have a pristine material
  params_ = nullptr;
  potsum_.clear();

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);

  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const unsigned int probinst =
          Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = dynamic_cast<Mixture::PAR::MixtureConstituentElastHyperBase*>(mat);
      }
      else
      {
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
      }
    }
  }

  summand_properties_.unpack(buffer);

  extract_from_pack(buffer, prestretch_);

  cosy_anisotropy_extension_.unpack_anisotropy(buffer);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{buffer};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    if (params_->get_prestressing_mat_id() > 0)
    {
      prestress_strategy_ =
          Mixture::PAR::PrestressStrategy::factory(params_->get_prestressing_mat_id())
              ->create_prestress_strategy();

      prestress_strategy_->unpack(buffer);
    }

    // make sure the referenced materials in material list have quick access parameters
    std::vector<int>::const_iterator m;
    for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
    {
      const int summatid = *m;
      std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(summatid);
      if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
      potsum_.push_back(sum);
    }

    // loop map of associated potential summands
    for (auto& summand : potsum_) summand->unpack_summand(buffer);
  }
}

void Mixture::MixtureConstituentElastHyperBase::register_anisotropy_extensions(
    Mat::Anisotropy& anisotropy)
{
  // Setup summands
  for (const auto& summand : potsum_) summand->register_anisotropy_extensions(anisotropy);

  anisotropy.register_anisotropy_extension(cosy_anisotropy_extension_);
}

// Reads the element from the input file
void Mixture::MixtureConstituentElastHyperBase::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  MixtureConstituent::read_element(numgp, container);

  // Setup summands
  for (const auto& summand : potsum_) summand->setup(numgp, container);

  // find out which formulations are used
  Mat::elast_hyper_properties(potsum_, summand_properties_);

  if (summand_properties_.viscoGeneral)
  {
    FOUR_C_THROW("Never use viscoelastic materials in Elasthyper-Toolbox.");
  }
}

// Updates all summands
void Mixture::MixtureConstituentElastHyperBase::update(
    Core::LinAlg::Tensor<double, 3, 3> const& defgrd, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  MixtureConstituent::update(defgrd, params, gp, eleGID);

  // loop map of associated potential summands
  for (auto& summand : potsum_) summand->update();

  // do nothing in the default case
  if (params_->get_prestressing_mat_id() > 0)
  {
    prestress_strategy_->update(cosy_anisotropy_extension_.get_coordinate_system_provider(gp),
        *this, defgrd, prestretch_[gp], params, gp, eleGID);
  }
}

void Mixture::MixtureConstituentElastHyperBase::setup(
    const Teuchos::ParameterList& params, const int eleGID)
{
  MixtureConstituent::setup(params, eleGID);
  if (params_->get_prestressing_mat_id() > 0)
  {
    prestretch_.resize(num_gp());

    prestress_strategy_->setup(*this, params, num_gp(), eleGID);
  }
}

void Mixture::MixtureConstituentElastHyperBase::pre_evaluate(
    MixtureRule& mixtureRule, const Teuchos::ParameterList& params, int gp, int eleGID)
{
  // do nothing in the default case
  if (params_->get_prestressing_mat_id() > 0)
  {
    prestress_strategy_->evaluate_prestress(mixtureRule,
        cosy_anisotropy_extension_.get_coordinate_system_provider(gp), *this, prestretch_[gp],
        params, gp, eleGID);
  }
}

void Mixture::MixtureConstituentElastHyperBase::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  if (prestress_strategy_ != nullptr)
  {
    names_and_size["mixture_constituent_" + std::to_string(id()) + "_elasthyper_prestretch"] = 6;
  }
}

bool Mixture::MixtureConstituentElastHyperBase::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (prestress_strategy_ != nullptr &&
      name == "mixture_constituent_" + std::to_string(id()) + "_elasthyper_prestretch")
  {
    for (int gp = 0; gp < num_gp(); ++gp)
    {
      for (int i = 0; i < 6; ++i)
      {
        data(gp, i) = prestretch_[gp].container()[i];
      }
    }
    return true;
  }
  return false;
}
FOUR_C_NAMESPACE_CLOSE
