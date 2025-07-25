// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_mixture.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_service.hpp"
#include "4C_utils_enum.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

// constructor of the parameters
Mat::PAR::Mixture::Mixture(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), constituents_(0)
{
  const auto& constituent_matids = matdata.parameters.get<std::vector<int>>("MATIDSCONST");

  // Create constituent materials according to their material IDs
  for (const auto& matid : constituent_matids)
  {
    constituents_.emplace_back(FourC::Mixture::PAR::MixtureConstituent::factory(matid));
  }

  // Create mixture rule
  mixture_rule_ =
      FourC::Mixture::PAR::MixtureRule::factory(matdata.parameters.get<int>("MATIDMIXTURERULE"));
}

// Create a material instance from parameters
std::shared_ptr<Core::Mat::Material> Mat::PAR::Mixture::create_material()
{
  return std::make_shared<Mat::Mixture>(this);
}

Mat::MixtureType Mat::MixtureType::instance_;

// Create a material instance from packed data
Core::Communication::ParObject* Mat::MixtureType::create(Core::Communication::UnpackBuffer& buffer)
{
  auto* mix_elhy = new Mat::Mixture();
  mix_elhy->unpack(buffer);

  return mix_elhy;
}

// constructor
Mat::Mixture::Mixture()
    : params_(nullptr),
      constituents_(
          std::make_shared<std::vector<std::unique_ptr<FourC::Mixture::MixtureConstituent>>>(0)),
      setup_(false),
      anisotropy_()
{
}

// constructor
Mat::Mixture::Mixture(Mat::PAR::Mixture* params)
    : params_(params),
      constituents_(
          std::make_shared<std::vector<std::unique_ptr<FourC::Mixture::MixtureConstituent>>>(0)),
      setup_(false),
      anisotropy_()
{
  // create instances of constituents
  int id = 0;
  for (auto const& constituent : params_->constituents_)
  {
    constituents_->emplace_back(constituent->create_constituent(id));
    constituents_->back()->register_anisotropy_extensions(anisotropy_);

    ++id;
  }

  // create instance of mixture rule
  mixture_rule_ = params->mixture_rule_->create_rule();
  mixture_rule_->set_constituents(constituents_);
  mixture_rule_->register_anisotropy_extensions(anisotropy_);
}

// Pack data
void Mat::Mixture::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // Pack material id
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);

  // pack setup flag
  add_to_pack(data, setup_);

  // Pack isPreEvaluated flag
  std::vector<int> isPreEvaluatedInt;
  isPreEvaluatedInt.resize(is_pre_evaluated_.size());
  for (unsigned i = 0; i < is_pre_evaluated_.size(); ++i)
  {
    isPreEvaluatedInt[i] = static_cast<int>(is_pre_evaluated_[i]);
  }
  add_to_pack(data, isPreEvaluatedInt);

  anisotropy_.pack_anisotropy(data);

  // pack all constituents
  // constituents are not accessible during post processing
  Core::Communication::PotentiallyUnusedBufferScope consitutent_scope{data};
  if (params_ != nullptr)
  {
    for (const auto& constituent : *constituents_)
    {
      constituent->pack_constituent(data);
    }

    // pack mixturerule
    mixture_rule_->pack_mixture_rule(data);
  }
}

// Unpack data
void Mat::Mixture::unpack(Core::Communication::UnpackBuffer& buffer)
{
  params_ = nullptr;
  constituents_->clear();
  setup_ = false;



  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid and recover params_
  int matid;
  extract_from_pack(buffer, matid);
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
      {
        params_ = dynamic_cast<Mat::PAR::Mixture*>(mat);
      }
      else
      {
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
      }
    }

    // Extract setup flag
    extract_from_pack(buffer, setup_);


    // Extract is isPreEvaluated
    std::vector<int> isPreEvaluatedInt;
    extract_from_pack(buffer, isPreEvaluatedInt);
    is_pre_evaluated_.resize(isPreEvaluatedInt.size());
    for (unsigned i = 0; i < isPreEvaluatedInt.size(); ++i)
    {
      is_pre_evaluated_[i] = static_cast<bool>(isPreEvaluatedInt[i]);
    }

    anisotropy_.unpack_anisotropy(buffer);

    // extract constituents
    // constituents are not accessible during post processing
    Core::Communication::PotentiallyUnusedBufferScope consitutent_scope{buffer};
    if (params_ != nullptr)
    {
      // create instances of constituents
      int id = 0;
      for (auto const& constituent : params_->constituents_)
      {
        constituents_->emplace_back(constituent->create_constituent(id));

        ++id;
      }

      // create instance of mixture rule
      mixture_rule_ = params_->mixture_rule_->create_rule();

      // make sure the referenced materials in material list have quick access parameters
      for (const auto& constituent : *constituents_)
      {
        constituent->unpack_constituent(buffer);
        constituent->register_anisotropy_extensions(anisotropy_);
      }

      // unpack mixturerule
      mixture_rule_->unpack_mixture_rule(buffer);
      mixture_rule_->set_constituents(constituents_);
      mixture_rule_->register_anisotropy_extensions(anisotropy_);

      // position checking is not available in post processing mode
    }
  }
}

// Read element and create arrays for the quantities at the Gauss points
void Mat::Mixture::setup(const int numgp, const Core::IO::InputParameterContainer& container)
{
  So3Material::setup(numgp, container);

  // resize preevaluation flag
  is_pre_evaluated_.resize(numgp, false);

  // Setup anisotropy
  anisotropy_.set_number_of_gauss_points(numgp);
  anisotropy_.read_anisotropy_from_element(container);

  // Let all constituents read the element input parameter container
  for (const auto& constituent : *constituents_)
  {
    constituent->read_element(numgp, container);
  }

  mixture_rule_->read_element(numgp, container);
}

// Post setup routine -> Call Setup of constituents and mixture rule
void Mat::Mixture::post_setup(const Teuchos::ParameterList& params, const int eleGID)
{
  So3Material::post_setup(params, eleGID);
  anisotropy_.read_anisotropy_from_parameter_list(params);
  if (constituents_ != nullptr)
  {
    for (const auto& constituent : *constituents_)
    {
      constituent->setup(params, eleGID);
    }
  }

  if (mixture_rule_ != nullptr)
  {
    mixture_rule_->setup(params, eleGID);
  }

  setup_ = true;
}

void Mat::Mixture::update()
{
  // Update all constituents
  for (const auto& constituent : *constituents_)
  {
    constituent->update();
  }

  mixture_rule_->update();
}

// This method is called between two timesteps
void Mat::Mixture::update(Core::LinAlg::Tensor<double, 3, 3> const& defgrd, const int gp,
    const Teuchos::ParameterList& params, const int eleGID)
{
  // Update all constituents
  for (const auto& constituent : *constituents_)
  {
    constituent->update(defgrd, params, gp, eleGID);
  }

  mixture_rule_->update(defgrd, params, gp, eleGID);
}

// Evaluates the material
void Mat::Mixture::evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID)
{
  // check, whether the post_setup method was already called
  if (!setup_) FOUR_C_THROW("The material's post_setup() method has not been called yet.");

  if (!is_pre_evaluated_[gp])
  {
    for (const auto& constituent : *constituents_)
    {
      is_pre_evaluated_[gp] = true;
      constituent->pre_evaluate(*mixture_rule_, params, gp, eleGID);
    }

    mixture_rule_->pre_evaluate(params, gp, eleGID);
  }

  // Evaluate mixturerule
  mixture_rule_->evaluate(*defgrad, glstrain, params, stress, cmat, gp, eleGID);
}

void Mat::Mixture::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  mixture_rule_->register_output_data_names(names_and_size);
  for (const auto& constituent : *constituents_)
  {
    constituent->register_output_data_names(names_and_size);
  }
}

bool Mat::Mixture::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  bool out = mixture_rule_->evaluate_output_data(name, data);
  for (const auto& constituent : *constituents_)
  {
    out = out || constituent->evaluate_output_data(name, data);
  }

  return out;
}
FOUR_C_NAMESPACE_CLOSE
