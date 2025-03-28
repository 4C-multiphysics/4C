// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_constituent_elasthyper_elastin_membrane.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_mat_anisotropy_extension.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_elast_isoneohooke.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::ElastinMembraneAnisotropyExtension::ElastinMembraneAnisotropyExtension(
    const std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>& structuralTensorStrategy)
    : FiberAnisotropyExtension<1>(structuralTensorStrategy)
{
  register_needed_tensors(
      FiberAnisotropyExtension::FIBER_VECTORS | FiberAnisotropyExtension::STRUCTURAL_TENSOR);
}

void Mixture::ElastinMembraneAnisotropyExtension::on_global_data_initialized()
{
  if (get_anisotropy()->has_gp_cylinder_coordinate_system())
  {
    for (int gp = 0; gp < get_anisotropy()->get_number_of_gauss_points(); ++gp)
    {
      std::array<Core::LinAlg::Matrix<3, 1>, 1> fibers;

      fibers[0].update(get_anisotropy()->get_gp_cylinder_coordinate_system(gp).get_rad());
      Mat::FiberAnisotropyExtension<1>::set_fibers(gp, fibers);
    }

    orthogonal_structural_tensor_.resize(get_anisotropy()->get_number_of_gauss_points());

    set_fiber_location(Mat::FiberLocation::GPFibers);
  }
  else if (get_anisotropy()->has_element_cylinder_coordinate_system())
  {
    std::array<Core::LinAlg::Matrix<3, 1>, 1> fibers;

    fibers[0].update(get_anisotropy()->get_element_cylinder_coordinate_system().get_rad());
    FiberAnisotropyExtension<1>::set_fibers(BaseAnisotropyExtension::GPDEFAULT, fibers);

    orthogonal_structural_tensor_.resize(1);

    set_fiber_location(Mat::FiberLocation::ElementFibers);
  }
  else
  {
    FOUR_C_THROW(
        "Could not find either GP cylinder coordinates nor element cylinder coordinates. The "
        "elastin material needs at least one.");
  }

  for (unsigned gp = 0; gp < orthogonal_structural_tensor_.size(); ++gp)
  {
    orthogonal_structural_tensor_[gp] = Core::LinAlg::identity_matrix<3>();

    orthogonal_structural_tensor_[gp].update(-1.0, get_structural_tensor(gp, 0), 1.0);
  }
}

const Core::LinAlg::Matrix<3, 3>&
Mixture::ElastinMembraneAnisotropyExtension::get_orthogonal_structural_tensor(int gp)
{
  if (orthogonal_structural_tensor_.empty())
  {
    FOUR_C_THROW("The coordinate system hast not been initialized yet.");

    // avoid compiler error
    return orthogonal_structural_tensor_[0];
  }

  if (orthogonal_structural_tensor_.size() == 1)
  {
    // using element fibers
    return orthogonal_structural_tensor_[0];
  }


  // using Gauss-point fibers
  return orthogonal_structural_tensor_[gp];
}

// Constructor for the parameter class
Mixture::PAR::MixtureConstituentElastHyperElastinMembrane::
    MixtureConstituentElastHyperElastinMembrane(const Core::Mat::PAR::Parameter::Data& matdata)
    : MixtureConstituentElastHyperBase(matdata),
      damage_function_id_(matdata.parameters.get<int>("DAMAGE_FUNCT")),
      nummat_membrane_(matdata.parameters.get<int>("MEMBRANENUMMAT")),
      matids_membrane_(matdata.parameters.get<std::vector<int>>("MEMBRANEMATIDS"))
{
  if (nummat_membrane_ != (int)matids_membrane_.size())
  {
    FOUR_C_THROW(
        "number of membrane summands {} does not fit to the size of the membrane summands vector"
        " {}",
        nummat_membrane_, matids_membrane_.size());
  }
}

// Create an instance of Mixture::MixtureConstituentElastHyper from the parameters
std::unique_ptr<Mixture::MixtureConstituent>
Mixture::PAR::MixtureConstituentElastHyperElastinMembrane::create_constituent(int id)
{
  return std::unique_ptr<Mixture::MixtureConstituentElastHyperElastinMembrane>(
      new Mixture::MixtureConstituentElastHyperElastinMembrane(this, id));
}

// Constructor of the constituent holding the material parameters
Mixture::MixtureConstituentElastHyperElastinMembrane::MixtureConstituentElastHyperElastinMembrane(
    Mixture::PAR::MixtureConstituentElastHyperElastinMembrane* params, int id)
    : MixtureConstituentElastHyperBase(params, id),
      params_(params),
      anisotropy_extension_(std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase>(
          new Mat::Elastic::StructuralTensorStrategyStandard(nullptr)))
{
  // Create summands
  for (const auto& matid : params_->matids_membrane_)
  {
    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
    if (sum == nullptr) FOUR_C_THROW("Failed to read elastic summand.");

    std::shared_ptr<Mat::Elastic::IsoNeoHooke> neoHooke =
        std::dynamic_pointer_cast<Mat::Elastic::IsoNeoHooke>(sum);

    if (!neoHooke)
    {
      FOUR_C_THROW(
          "Currently, only the an IsoNeoHooke material law is possible for use as an elastin "
          "membrane material");
    }

    potsum_membrane_.push_back(neoHooke);
  }
}

// Returns the material type
Core::Materials::MaterialType Mixture::MixtureConstituentElastHyperElastinMembrane::material_type()
    const
{
  return Core::Materials::mix_elasthyper_elastin_membrane;
}

// Pack the constituent
void Mixture::MixtureConstituentElastHyperElastinMembrane::pack_constituent(
    Core::Communication::PackBuffer& data) const
{
  MixtureConstituentElastHyperBase::pack_constituent(data);

  add_to_pack(data, current_reference_growth_);

  add_to_pack(data, mue_frac_);

  anisotropy_extension_.pack_anisotropy(data);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope{data};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
  {
    // loop map of associated potential summands
    for (const auto& p : potsum_membrane_) p->pack_summand(data);
  }
}

// Unpack the constituent
void Mixture::MixtureConstituentElastHyperElastinMembrane::unpack_constituent(
    Core::Communication::UnpackBuffer& buffer)
{
  MixtureConstituentElastHyperBase::unpack_constituent(buffer);

  extract_from_pack(buffer, current_reference_growth_);

  extract_from_pack(buffer, mue_frac_);

  anisotropy_extension_.unpack_anisotropy(buffer);

  // loop map of associated potential summands
  Core::Communication::PotentiallyUnusedBufferScope summand_scope{buffer};
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
    for (auto& summand : potsum_membrane_) summand->unpack_summand(buffer);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::register_anisotropy_extensions(
    Mat::Anisotropy& anisotropy)
{
  MixtureConstituentElastHyperBase::register_anisotropy_extensions(anisotropy);

  anisotropy.register_anisotropy_extension(anisotropy_extension_);

  for (const auto& summand : potsum_membrane_) summand->register_anisotropy_extensions(anisotropy);
}

// Reads the element from the input file
void Mixture::MixtureConstituentElastHyperElastinMembrane::read_element(
    int numgp, const Core::IO::InputParameterContainer& container)
{
  MixtureConstituentElastHyperBase::read_element(numgp, container);

  // Setup summands
  for (const auto& summand : potsum_membrane_) summand->setup(numgp, container);

  current_reference_growth_.resize(numgp, 1.0);
  mue_frac_.resize(numgp, 1.0);
}

// Updates all summands
void Mixture::MixtureConstituentElastHyperElastinMembrane::update(
    Core::LinAlg::Matrix<3, 3> const& defgrd, Teuchos::ParameterList& params, const int gp,
    const int eleGID)
{
  const auto& reference_coordinates = params.get<Core::LinAlg::Matrix<3, 1>>("gp_coords_ref");

  double totaltime = params.get<double>("total time", -1);
  if (totaltime < 0.0)
  {
    FOUR_C_THROW("Parameter 'total time' could not be read!");
  }

  current_reference_growth_[gp] =
      Global::Problem::instance()
          ->function_by_id<Core::Utils::FunctionOfSpaceTime>(params_->damage_function_id_)
          .evaluate(reference_coordinates.data(), totaltime, 0);

  MixtureConstituentElastHyperBase::update(defgrd, params, gp, eleGID);

  // loop map of associated potential summands
  for (auto& summand : potsum_membrane_) summand->update();
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::pre_evaluate(
    MixtureRule& mixtureRule, Teuchos::ParameterList& params, int gp, int eleGID)
{
  Mixture::MixtureConstituentElastHyperBase::pre_evaluate(mixtureRule, params, gp, eleGID);

  // Evaluate mue frac
  std::shared_ptr<ElastinMembranePrestressStrategy> strategy =
      std::dynamic_pointer_cast<ElastinMembranePrestressStrategy>(prestress_strategy());

  if (strategy == nullptr)
  {
    FOUR_C_THROW(
        "The used prestretch strategy is not compatible with elastin materials. It has to "
        "implement Mixture::ElastinMembranePrestressStrategy.");
  }

  mue_frac_[gp] = strategy->evaluate_mue_frac(mixtureRule,
      cylinder_coordinate_system_anisotropy_extension().get_coordinate_system_provider(gp), *this,
      *this, params, gp, eleGID);
}

double Mixture::MixtureConstituentElastHyperElastinMembrane::get_growth_scalar(int gp) const
{
  return current_reference_growth_[gp];
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::evaluate(
    const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<6, 1>& E_strain,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  FOUR_C_THROW("This constituent does not support Evaluation without an elastic part.");
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::evaluate_elastic_part(
    const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<3, 3>& iFextin,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID)
{
  // Compute total inelastic deformation gradient
  static Core::LinAlg::Matrix<3, 3> iFin(Core::LinAlg::Initialization::uninitialized);
  iFin.multiply_nn(iFextin, prestretch_tensor(gp));

  // Evaluate 3D elastic part
  Mat::elast_hyper_evaluate_elastic_part(
      F, iFin, S_stress, cmat, summands(), summand_properties(), gp, eleGID);

  // Evaluate Membrane
  static Core::LinAlg::Matrix<6, 1> Smembrane_stress(Core::LinAlg::Initialization::uninitialized);
  static Core::LinAlg::Matrix<6, 6> cmatmembrane(Core::LinAlg::Initialization::uninitialized);
  evaluate_stress_c_mat_membrane(F, iFin, params, Smembrane_stress, cmatmembrane, gp, eleGID);

  S_stress.update(1.0, Smembrane_stress, 1.0);
  cmat.update(1.0, cmatmembrane, 1.0);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::evaluate_membrane_stress(
    Core::LinAlg::Matrix<6, 1>& S, Teuchos::ParameterList& params, int gp, int eleGID)
{
  Core::LinAlg::Matrix<6, 6> cmat(Core::LinAlg::Initialization::uninitialized);
  const Core::LinAlg::Matrix<3, 3> Id = Core::LinAlg::identity_matrix<3>();
  Core::LinAlg::Matrix<3, 3> iFin(Core::LinAlg::Initialization::uninitialized);

  iFin.multiply_nn(Id, prestretch_tensor(gp));

  evaluate_stress_c_mat_membrane(Id, iFin, params, S, cmat, gp, eleGID);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::evaluate_stress_c_mat_membrane(
    const Core::LinAlg::Matrix<3, 3>& F, const Core::LinAlg::Matrix<3, 3>& iFin,
    Teuchos::ParameterList& params, Core::LinAlg::Matrix<6, 1>& S_stress,
    Core::LinAlg::Matrix<6, 6>& cmat, int gp, int eleGID) const
{
  static Core::LinAlg::Matrix<3, 3> Ce(Core::LinAlg::Initialization::uninitialized);
  Mat::evaluate_ce(F, iFin, Ce);

  // Compute structural tensors in grown configuration
  static Core::LinAlg::Matrix<3, 3> Aradgr(Core::LinAlg::Initialization::uninitialized);
  static Core::LinAlg::Matrix<3, 3> Aorthgr(Core::LinAlg::Initialization::uninitialized);
  evaluate_structural_tensors_in_grown_configuration(Aradgr, Aorthgr, iFin, gp, eleGID);

  static Core::LinAlg::Matrix<3, 3> AorthgrCeAorthgrArad(
      Core::LinAlg::Initialization::uninitialized);
  evaluate_aorthgr_ce_aorthgr_arad(AorthgrCeAorthgrArad, Aradgr, Aorthgr, Ce);
  double detX = AorthgrCeAorthgrArad.determinant();

  static Core::LinAlg::Matrix<3, 3> iFinAorthgriFinT(Core::LinAlg::Initialization::uninitialized);
  evaluatei_fin_aorthgri_fin_t(iFinAorthgriFinT, iFin, Aorthgr);

  // Z = F_{in}^{-T}*A_{gr}^{T}*X^{-1}*A_{gr}^{T}*F_{in}^{-1}
  static Core::LinAlg::Matrix<3, 3> iFinTAorthgrTiXTAorthgriFin_sym(
      Core::LinAlg::Initialization::uninitialized);
  static Core::LinAlg::Matrix<6, 1> iFinTAorthgrTiXTAorthgriFin_sym_stress(
      Core::LinAlg::Initialization::uninitialized);

  evaluatei_fin_t_aorthgr_ti_xt_aorthgri_fin(
      iFinTAorthgrTiXTAorthgriFin_sym, AorthgrCeAorthgrArad, iFin, Aorthgr);

  Core::LinAlg::Voigt::Stresses::matrix_to_vector(
      iFinTAorthgrTiXTAorthgriFin_sym, iFinTAorthgrTiXTAorthgriFin_sym_stress);

  // Get material parameter
  double mue = 0.0;
  for (const auto& summand : potsum_membrane_)
  {
    mue += summand->mue();
  }

  // Compute membrane stress
  static Core::LinAlg::Matrix<3, 3> Smembrane;
  Smembrane.update(mue * mue_frac_[gp], iFinAorthgriFinT, 0.0);
  Smembrane.update(-mue * mue_frac_[gp] / detX, iFinTAorthgrTiXTAorthgriFin_sym, 1.0);

  Core::LinAlg::Voigt::Stresses::matrix_to_vector(Smembrane, S_stress);

  // Compute constitutive tensor
  static Core::LinAlg::Matrix<6, 6> dAradgriXAradgr_symdC(
      Core::LinAlg::Initialization::uninitialized);
  dAradgriXAradgr_symdC.clear();

  Core::LinAlg::Tensor::add_holzapfel_product(
      dAradgriXAradgr_symdC, iFinTAorthgrTiXTAorthgriFin_sym_stress, -2.0);

  cmat.multiply_nt(2.0 * mue * mue_frac_[gp] / detX, iFinTAorthgrTiXTAorthgriFin_sym_stress,
      iFinTAorthgrTiXTAorthgriFin_sym_stress, 0.0);
  cmat.update(-mue * mue_frac_[gp] / detX, dAradgriXAradgr_symdC, 1.0);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::
    evaluate_structural_tensors_in_grown_configuration(Core::LinAlg::Matrix<3, 3>& Aradgr,
        Core::LinAlg::Matrix<3, 3>& Aorthgr, const Core::LinAlg::Matrix<3, 3>& iFin, const int gp,
        const int eleGID) const
{
  // Compute inelastic right Cauchy-Green deformation gradient
  static Core::LinAlg::Matrix<3, 3> iCin(Core::LinAlg::Initialization::uninitialized);
  iCin.multiply_nt(iFin, iFin);

  static Core::LinAlg::Matrix<3, 3> Fin(Core::LinAlg::Initialization::uninitialized);
  Fin.invert(iFin);

  // Compute radial structural tensor in grown configuration
  static Core::LinAlg::Matrix<3, 3> FinArad(Core::LinAlg::Initialization::uninitialized);
  FinArad.multiply_nn(Fin, anisotropy_extension_.get_structural_tensor(gp, 0));
  Aradgr.multiply_nt(
      iCin.dot(anisotropy_extension_.get_structural_tensor(gp, 0)), FinArad, Fin, 0.0);

  // Compute orthogonal (to radial) structural tensor in grown configuration
  Aorthgr = Core::LinAlg::identity_matrix<3>();
  Aorthgr.update(-1.0, Aradgr, 1.0);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::evaluate_aorthgr_ce_aorthgr_arad(
    Core::LinAlg::Matrix<3, 3>& AorthgrCeAorthgrArad, const Core::LinAlg::Matrix<3, 3>& Aradgr,
    const Core::LinAlg::Matrix<3, 3>& Aorthgr, const Core::LinAlg::Matrix<3, 3>& Ce)
{
  static Core::LinAlg::Matrix<3, 3> AorthgrCe(Core::LinAlg::Initialization::uninitialized);
  AorthgrCe.multiply_nn(Aorthgr, Ce);
  AorthgrCeAorthgrArad.multiply_nn(AorthgrCe, Aorthgr);
  AorthgrCeAorthgrArad.update(1.0, Aradgr, 1.0);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::evaluatei_fin_aorthgri_fin_t(
    Core::LinAlg::Matrix<3, 3>& iFinAorthgriFinT, const Core::LinAlg::Matrix<3, 3>& iFin,
    const Core::LinAlg::Matrix<3, 3>& Aorthgr)
{
  static Core::LinAlg::Matrix<3, 3> iFinAorthgr(Core::LinAlg::Initialization::uninitialized);
  iFinAorthgr.multiply_nn(iFin, Aorthgr);
  iFinAorthgriFinT.multiply_nt(iFinAorthgr, iFin);
}

void Mixture::MixtureConstituentElastHyperElastinMembrane::
    evaluatei_fin_t_aorthgr_ti_xt_aorthgri_fin(
        Core::LinAlg::Matrix<3, 3>& iFinTAorthgrTiXTAorthgriFin_sym,
        const Core::LinAlg::Matrix<3, 3>& AorthgrCeAorthgrArad,
        const Core::LinAlg::Matrix<3, 3>& iFin, const Core::LinAlg::Matrix<3, 3>& Aorthgr)
{
  static Core::LinAlg::Matrix<3, 3> iAorthgrCeAorthgrArad(
      Core::LinAlg::Initialization::uninitialized);
  iAorthgrCeAorthgrArad.invert(AorthgrCeAorthgrArad);

  static Core::LinAlg::Matrix<3, 3> AorthgriFin(Core::LinAlg::Initialization::uninitialized);
  AorthgriFin.multiply_nn(Aorthgr, iFin);

  static Core::LinAlg::Matrix<3, 3> iFinTAorthgrTiXT(Core::LinAlg::Initialization::uninitialized);
  iFinTAorthgrTiXT.multiply_tt(AorthgriFin, iAorthgrCeAorthgrArad);

  static Core::LinAlg::Matrix<3, 3> iFinTAorthgrTiXTAorthgriFin(
      Core::LinAlg::Initialization::uninitialized);
  iFinTAorthgrTiXTAorthgriFin.multiply_nn(iFinTAorthgrTiXT, AorthgriFin);

  iFinTAorthgrTiXTAorthgriFin_sym.update(0.5, iFinTAorthgrTiXTAorthgriFin, 0.0);
  iFinTAorthgrTiXTAorthgriFin_sym.update_t(0.5, iFinTAorthgrTiXTAorthgriFin, 1.0);
}
FOUR_C_NAMESPACE_CLOSE
