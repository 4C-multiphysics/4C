// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_remodelfiber.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_FADmatrix_utils.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::RemodelFiber::RemodelFiber(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      nummat_(matdata.parameters.get<int>("NUMMAT")),
      matids_(matdata.parameters.get<std::vector<int>>("MATIDS")),
      t_decay_(matdata.parameters.get<double>("TDECAY")),
      k_growth_(matdata.parameters.get<double>("GROWTHFAC")),
      init_w_col_(matdata.parameters.get<std::vector<double>>("COLMASSFRAC")),
      G_(matdata.parameters.get<double>("DEPOSITIONSTRETCH"))
{
  // check if sizes fit
  if (nummat_ != (int)matids_.size())
    FOUR_C_THROW("number of materials {} does not fit to size of material vector {}", nummat_,
        matids_.size());

  // check decay time validity
  if (t_decay_ <= 0.) FOUR_C_THROW("decay time must be positive");
}

Mat::Elastic::RemodelFiber::RemodelFiber(Mat::Elastic::PAR::RemodelFiber* params)
    : params_(params), potsumfiber_(0)
{
  // make sure the referenced materials in material list have quick access parameters
  std::vector<int>::const_iterator m;
  for (m = params_->matids_.begin(); m != params_->matids_.end(); ++m)
  {
    const int matid = *m;
    std::shared_ptr<Mat::Elastic::Summand> sum = Mat::Elastic::Summand::factory(matid);
    if (sum == nullptr) FOUR_C_THROW("Failed to allocate");
    potsumfiber_.push_back(std::make_shared<FiberData>(sum));
  }
}

void Mat::Elastic::RemodelFiber::pack_summand(Core::Communication::PackBuffer& data) const
{
  int num_fiber = 0;
  num_fiber = potsumfiber_.size();
  int num_gp = 0;
  num_gp = potsumfiber_[0]->cur_lambr.size();
  add_to_pack(data, num_fiber);
  add_to_pack(data, num_gp);

  for (int i = 0; i < num_fiber; ++i)
  {
    add_to_pack(data, potsumfiber_[i]->cur_lambr);
    add_to_pack(data, potsumfiber_[i]->last_lambr);
    add_to_pack(data, potsumfiber_[i]->cur_rho);
    add_to_pack(data, potsumfiber_[i]->last_rho);
    add_to_pack(data, potsumfiber_[i]->A);
    add_to_pack(data, potsumfiber_[i]->A_orth);
    add_to_pack(data, potsumfiber_[i]->AM);
    add_to_pack(data, potsumfiber_[i]->AM_orth);
    add_to_pack(data, potsumfiber_[i]->FrnM);
    add_to_pack(data, potsumfiber_[i]->diFrdlambrM);
    add_to_pack(data, potsumfiber_[i]->dFrdlambrM);
    add_to_pack(data, potsumfiber_[i]->iFrM);
    add_to_pack(data, potsumfiber_[i]->FrdotM);
    add_to_pack(data, potsumfiber_[i]->dFrdotdlambrM);
    add_to_pack(data, potsumfiber_[i]->remodel->sig_h);
    add_to_pack(data, potsumfiber_[i]->remodel->k_sig);
    add_to_pack(data, potsumfiber_[i]->remodel->t_decay);
    add_to_pack(data, potsumfiber_[i]->G);
    add_to_pack(data, cauchystress_[i]);
  }

  add_to_pack(data, init_rho_col_);

  Core::Communication::PotentiallyUnusedBufferScope summand_scope(data);
  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
    for (const auto& k : potsumfiber_) k->fiber->pack_summand(data);
}

void Mat::Elastic::RemodelFiber::unpack_summand(Core::Communication::UnpackBuffer& buffer)
{
  //  // make sure we have a pristine material
  //  params_ = nullptr;
  //  potsumfiber_.clear();

  int num_fiber = 0;
  extract_from_pack(buffer, num_fiber);
  int num_gp = 0;
  extract_from_pack(buffer, num_gp);

  cauchystress_.resize(num_fiber);

  double sig_h = 0.0;
  double k_sig = 0.0;
  double t_decay = 0.0;
  for (int k = 0; k < num_fiber; ++k)
  {
    extract_from_pack(buffer, potsumfiber_[k]->cur_lambr);
    extract_from_pack(buffer, potsumfiber_[k]->last_lambr);
    extract_from_pack(buffer, potsumfiber_[k]->cur_rho);
    extract_from_pack(buffer, potsumfiber_[k]->last_rho);
    extract_from_pack(buffer, potsumfiber_[k]->A);
    extract_from_pack(buffer, potsumfiber_[k]->A_orth);
    extract_from_pack(buffer, potsumfiber_[k]->AM);
    extract_from_pack(buffer, potsumfiber_[k]->AM_orth);
    extract_from_pack(buffer, potsumfiber_[k]->FrnM);
    extract_from_pack(buffer, potsumfiber_[k]->diFrdlambrM);
    extract_from_pack(buffer, potsumfiber_[k]->dFrdlambrM);
    extract_from_pack(buffer, potsumfiber_[k]->iFrM);
    extract_from_pack(buffer, potsumfiber_[k]->FrdotM);
    extract_from_pack(buffer, potsumfiber_[k]->dFrdotdlambrM);
    extract_from_pack(buffer, sig_h);
    extract_from_pack(buffer, k_sig);
    extract_from_pack(buffer, t_decay);
    extract_from_pack(buffer, potsumfiber_[k]->G);
    extract_from_pack(buffer, cauchystress_[k]);

    potsumfiber_[k]->growth = std::make_shared<GrowthEvolution>(k_sig, sig_h);
    potsumfiber_[k]->remodel = std::make_shared<RemodelEvolution>(k_sig, sig_h, t_decay);
  }

  extract_from_pack(buffer, init_rho_col_);

  // loop map of associated potential summands
  Core::Communication::PotentiallyUnusedBufferScope summand_scope(buffer);

  if (params_ != nullptr)  // summands are not accessible in postprocessing mode
    for (auto& k : potsumfiber_) k->fiber->unpack_summand(buffer);
}

void Mat::Elastic::RemodelFiber::register_anisotropy_extensions(Anisotropy& anisotropy)
{
  for (auto& summand : potsumfiber_)
  {
    summand->fiber->register_anisotropy_extensions(anisotropy);
  }
}

void Mat::Elastic::RemodelFiber::setup(int numgp, double rho_tot,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  // setup fiber and inelastic history variable
  cauchystress_.resize(potsumfiber_.size());
  init_rho_col_.resize(potsumfiber_.size());
  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    init_rho_col_[k] = rho_tot * params_->init_w_col_.at(k);
    potsumfiber_[k]->FrnM.resize(numgp);
    potsumfiber_[k]->cur_lambr.resize(numgp, 1.0);
    potsumfiber_[k]->cur_rho.resize(numgp, init_rho_col_[k]);
    potsumfiber_[k]->diFrdlambrM.resize(numgp);
    potsumfiber_[k]->dFrdlambrM.resize(numgp);
    potsumfiber_[k]->iFrM.resize(numgp);
    potsumfiber_[k]->FrdotM.resize(numgp);
    potsumfiber_[k]->dFrdotdlambrM.resize(numgp);
    potsumfiber_[k]->last_lambr.resize(numgp, 1.0);
    potsumfiber_[k]->last_rho.resize(numgp, init_rho_col_[k]);
    potsumfiber_[k]->G = params_->G_;
    cauchystress_[k].resize(numgp, 1.0);

    potsumfiber_[k]->fiber->setup(numgp, fibers, coord_system);
  }


  // some variables
  Core::LinAlg::Matrix<2, 1> dPI(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ddPII(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> dddPIII(Core::LinAlg::Initialization::zero);
  Core::LinAlg::SymmetricTensor<double, 3, 3> stressactv{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmatactive{};
  Core::LinAlg::Matrix<3, 3> stressactM(Core::LinAlg::Initialization::zero);

  setup_structural_tensors_gr();

  // quadratic prestretch in tensor notation
  Core::LinAlg::SymmetricTensor<double, 3, 3> Cpre{};
  const Core::LinAlg::Matrix<3, 3> CpreM = Core::LinAlg::make_matrix(Core::LinAlg::get_full(Cpre));

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  // identity matrix
  Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

  double sig_pre = 0.0;
  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(potsumfiber_[k]->fiber)).get())
    {
      Cpre = potsumfiber_[k]->G * potsumfiber_[k]->G * potsumfiber_[k]->A;
      t1->get_derivatives_aniso(dPI, ddPII, dddPIII, Cpre, 0, 0);
      sig_pre = 2.0 * dPI(0) * potsumfiber_[k]->G * potsumfiber_[k]->G;
      for (int gp = 0; gp < numgp; ++gp) cauchystress_[k][gp] = sig_pre;
      potsumfiber_[k]->growth = std::make_shared<GrowthEvolution>(params_->k_growth_, sig_pre);
      potsumfiber_[k]->remodel =
          std::make_shared<RemodelEvolution>(params_->k_growth_, sig_pre, params_->t_decay_);
    }
    else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(
                  potsumfiber_[k]->fiber))
                 .get())
    {
      Cpre = potsumfiber_[k]->G * potsumfiber_[k]->G * potsumfiber_[k]->A;
      t2->get_derivatives_aniso(dPI, ddPII, dddPIII, Cpre, 0, 0);
      sig_pre = 2.0 * dPI(0) * potsumfiber_[k]->G * potsumfiber_[k]->G;
      t2->evaluate_active_stress_cmat_aniso(
          Core::LinAlg::TensorGenerators::identity<double, 3, 3>, cmatactive, stressactv, 0, 0);
      Core::LinAlg::Voigt::Stresses::vector_to_matrix(
          Core::LinAlg::make_stress_like_voigt_view(stressactv), stressactM);
      sig_pre += stressactM.dot(potsumfiber_[k]->AM);
      for (int gp = 0; gp < numgp; ++gp) cauchystress_[k][gp] = sig_pre;
      potsumfiber_[k]->growth = std::make_shared<GrowthEvolution>(params_->k_growth_, sig_pre);
      potsumfiber_[k]->remodel =
          std::make_shared<RemodelEvolution>(params_->k_growth_, sig_pre, params_->t_decay_);
    }
    else
      FOUR_C_THROW(
          "So far, you can only use Elast_CoupAnisoExpo and Elast_CoupAnisoExpoActive in "
          "Elast_Remodelfiber!");
  }

  // Initialize inelastic deformation gradients in FiberData (default time step size (does not have
  // to be the real one))
  for (auto& k : potsumfiber_)
    for (int gp = 0; gp < numgp; ++gp) k->update_newton(gp, 1.0);
}

void Mat::Elastic::RemodelFiber::setup_structural_tensors_gr()
{
  // identity tensor
  Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

  // fiber directions
  std::vector<Core::LinAlg::Tensor<double, 3>> fibervecs;

  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    // Get fiberdirection
    potsumfiber_[k]->fiber->get_fiber_vecs(fibervecs);

    // build structural tensor in matrix notation
    potsumfiber_[k]->A = Core::LinAlg::self_dyadic(fibervecs[k]);
    potsumfiber_[k]->A_orth =
        Core::LinAlg::TensorGenerators::identity<double, 3, 3> - potsumfiber_[k]->A;
    potsumfiber_[k]->AM.multiply_nt(1.0, Core::LinAlg::make_matrix_view<3, 1>(fibervecs[k]),
        Core::LinAlg::make_matrix_view<3, 1>(fibervecs[k]), 0.0);
    // orthogonal structural tensor ( 1_{ij} - A_{ij} )
    potsumfiber_[k]->AM_orth.update(1.0, potsumfiber_[k]->AM, 0.0);
    potsumfiber_[k]->AM_orth.update(1.0, id, -1.0);
  }
}

void Mat::Elastic::RemodelFiber::update()
{
  // update history variable
  for (auto& k : potsumfiber_)
    for (unsigned gp = 0; gp < k->cur_rho.size(); ++gp) k->update_history(gp);
}

void Mat::Elastic::RemodelFiber::update_fiber_dirs(
    Core::LinAlg::Tensor<double, 3, 3> const& locsys, const double& dt)
{
  constexpr Core::LinAlg::Tensor<double, 3, 3> id =
      Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);

  for (auto& k : potsumfiber_) k->fiber->set_fiber_vecs(-1.0, locsys, id);

  setup_structural_tensors_gr();

  for (auto& k : potsumfiber_)
    for (unsigned gp = 0; gp < potsumfiber_[0]->cur_lambr.size(); ++gp) k->update_newton(gp, dt);

  update_sig_h();
};

void Mat::Elastic::RemodelFiber::update_sig_h()
{
  // some variables
  Core::LinAlg::SymmetricTensor<double, 3, 3> Cpre{};
  Core::LinAlg::Matrix<2, 1> dPI(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> ddPII(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<4, 1> dddPIII(Core::LinAlg::Initialization::zero);
  Core::LinAlg::SymmetricTensor<double, 3, 3> stressactv{};
  Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmatactive;
  Core::LinAlg::Matrix<3, 3> stressactM(Core::LinAlg::Initialization::zero);

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  // identity matrix
  Core::LinAlg::Matrix<3, 3> id(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i) id(i, i) = 1.0;

  double sig = 0.0;
  for (auto& k : potsumfiber_)
  {
    if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(k->fiber)).get())
    {
      Cpre = k->G * k->G * k->A;
      t1->get_derivatives_aniso(dPI, ddPII, dddPIII, Cpre, 0, 0);
      sig = 2.0 * dPI(0) * k->G * k->G;
      k->growth->set_sig_h(sig);
      k->remodel->set_sig_h(sig);
    }
    else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(k->fiber)).get())
    {
      Cpre = k->G * k->G * k->A;
      t2->get_derivatives_aniso(dPI, ddPII, dddPIII, Cpre, 0, 0);
      sig = 2.0 * dPI(0) * k->G * k->G;
      t2->evaluate_active_stress_cmat_aniso(
          Core::LinAlg::TensorGenerators::identity<double, 3, 3>, cmatactive, stressactv, 0, 0);
      Core::LinAlg::Voigt::Stresses::vector_to_matrix(
          Core::LinAlg::make_stress_like_voigt_view(stressactv), stressactM);
      sig += stressactM.dot(k->AM);
      k->growth->set_sig_h(sig);
      k->remodel->set_sig_h(sig);
    }
  }
}

void Mat::Elastic::RemodelFiber::evaluate_anisotropic_stress_cmat(
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<6, 6>& cmat, Core::LinAlg::Matrix<6, 1>& stress, int const gp,
    double const& dt, int const eleGID)
{
  // clear some variables
  stress.clear();
  cmat.clear();

  for (auto& k : potsumfiber_)
  {
    k->update_newton(gp, dt);
    add_stress_cmat(CM, iFgM, *k, gp, eleGID, stress, cmat);
  }
}

void Mat::Elastic::RemodelFiber::evaluate_derivatives_internal_newton(
    Core::LinAlg::Matrix<3, 3> const* const defgrd, int const nr_grf_proc, int const nr_grf_tot,
    int const gp, double const& dt, int const eleGID, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<3, 3> const& dFgdrhoM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    std::vector<std::vector<double>>& dWdrho, std::vector<std::vector<double>>& dWdlambr,
    std::vector<double>& W, std::vector<std::vector<double>>& dEdrho,
    std::vector<std::vector<double>>& dEdlambr, std::vector<double>& E)
{
  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, *defgrd, *defgrd, 0.0);

  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    potsumfiber_[k]->update_newton(gp, dt);

    // Residual of evolution equations
    evaluate_evolution_equation(
        W[nr_grf_proc + k], E[nr_grf_proc + k], CM, iFgM, dt, *(potsumfiber_[k]), gp, eleGID);

    // Derivatives of evolution equations
    double dWidrhoi = 0.0;
    double dWidrhoj = 0.0;
    double dEidrho = 0.0;
    evaluate_derivative_evolution_equation(dWidrhoi, dWidrhoj,
        dWdlambr[nr_grf_proc + k][nr_grf_proc + k], dEidrho,
        dEdlambr[nr_grf_proc + k][nr_grf_proc + k], CM, iFgM, dFgdrhoM, diFgdrhoM, dt,
        *(potsumfiber_[k]), gp, eleGID);

    for (int l = 0; l < nr_grf_tot; ++l)
    {
      dEdrho[nr_grf_proc + k][l] = dEidrho;
      if (l == (int)(k + nr_grf_proc))
        dWdrho[nr_grf_proc + k][l] = dWidrhoi;
      else
        dWdrho[nr_grf_proc + k][l] = dWidrhoj;
    }
  }
}

void Mat::Elastic::RemodelFiber::evaluate_derivatives_cauchy_green(
    Core::LinAlg::Matrix<3, 3> const* const defgrd, int const nr_grf_proc, int const gp,
    double const& dt, Core::LinAlg::Matrix<3, 3> const& iFgM,
    std::vector<Core::LinAlg::Matrix<1, 6>>& dWdC, std::vector<Core::LinAlg::Matrix<1, 6>>& dEdC,
    int const eleGID)
{
  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, *defgrd, *defgrd, 0.0);

  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    dEdC[nr_grf_proc + k].clear();
    dWdC[nr_grf_proc + k].clear();
    potsumfiber_[k]->update_newton(gp, dt);

    evaluated_evolution_equationd_c(dWdC[nr_grf_proc + k], dEdC[nr_grf_proc + k], CM, iFgM, dt,
        *(potsumfiber_[k]), k, gp, eleGID);
  }
}

void Mat::Elastic::RemodelFiber::evaluate_additional_growth_remodel_cmat(
    Core::LinAlg::Matrix<3, 3> const* const defgrd, int const nr_grf_proc,
    Core::LinAlg::Matrix<3, 3> const& iFgM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    std::vector<Core::LinAlg::Matrix<1, 6>> const& drhodC,
    std::vector<Core::LinAlg::Matrix<1, 6>> const& dlambrdC, Core::LinAlg::Matrix<6, 6>& cmat,
    int const gp, int const eleGID) const
{
  // clear some variables
  cmat.clear();

  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, *defgrd, *defgrd, 0.0);

  static Core::LinAlg::Matrix<6, 1> dSidrhoi(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> dSidrhoj(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> dSdlambr(Core::LinAlg::Initialization::zero);
  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    evaluate_derivatives2nd_piola_kirchhoff_growth_remodel(
        dSidrhoi, dSidrhoj, dSdlambr, CM, iFgM, diFgdrhoM, *(potsumfiber_[k]), gp, eleGID);

    for (unsigned l = 0; l < drhodC.size(); ++l)
    {
      if (l == (nr_grf_proc + k))
        cmat.multiply_nn(2.0, dSidrhoi, drhodC[l], 1.0);
      else
        cmat.multiply_nn(2.0, dSidrhoj, drhodC[l], 1.0);
    }
    cmat.multiply_nn(2.0, dSdlambr, dlambrdC[nr_grf_proc + k], 1.0);
  }
}

void Mat::Elastic::RemodelFiber::evaluate_growth_and_remodeling_expl(
    Core::LinAlg::Matrix<3, 3> const& defgrd, double const& dt,
    Core::LinAlg::Matrix<3, 3> const& iFgM, const int gp, const int eleGID)
{
  static Core::LinAlg::Matrix<3, 3> CM(Core::LinAlg::Initialization::zero);
  CM.multiply_tn(1.0, defgrd, defgrd, 0.0);
  double drhodt = 0.0;
  double dlambrdt = 0.0;

  for (unsigned k = 0; k < potsumfiber_.size(); ++k)
  {
    potsumfiber_[k]->update_newton(gp, dt);

    evaluated_evolution_equationdt(drhodt, dlambrdt, CM, iFgM, *(potsumfiber_[k]), k, gp, eleGID);

    update_growth_remodel_parameter(drhodt * dt, dlambrdt * dt, k, gp);
  }
}

template <class FUNC, typename T, typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::derivd_c(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFinM, Core::LinAlg::Matrix<3, 3, T> const& AM,
    FUNC const& func, const int gp, ForceAnalytical const eleGID,
    Core::LinAlg::Matrix<3, 3, T>& dfuncdC) const
{
  // clear some variables
  dfuncdC.clear();

  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  iFinM_fad = iFinM;

  // Setup FAD
  // first derivative
  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  CM_fad = CM;
  CM_fad.diff(0, 9);

  static Core::LinAlg::FADMatrix<3, 3> CeM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> tmp_fad(Core::LinAlg::Initialization::zero);
  tmp_fad.multiply_nn(1.0, CM_fad, iFinM_fad, 0.0);
  CeM_fad.multiply_tn(1.0, iFinM_fad, tmp_fad, 0.0);

  FAD r_fad = 0.0;
  func.evaluate_func(r_fad, CeM_fad, gp, eleGID);

  Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  first_deriv_to_matrix(r_fad, tmp);
  dfuncdC.update(1.0, tmp, 0.0);
}

template <class FUNC, typename T>
void Mat::Elastic::RemodelFiber::derivd_c(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFinM, Core::LinAlg::Matrix<3, 3, T> const& AM,
    FUNC const& func, const int gp, int const eleGID, Core::LinAlg::Matrix<3, 3, T>& dfuncdC) const
{
  // clear some variables
  dfuncdC.clear();

  // elastic right Cauchy-Green in matrix notation
  Core::LinAlg::SymmetricTensor<T, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));
  static Core::LinAlg::Matrix<3, 3, T> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> CinM(Core::LinAlg::Initialization::zero);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);

  // get derivatives of strain energy function w.r.t. I4
  static Core::LinAlg::Matrix<2, 1, T> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1, T> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1, T> dddPIIIe(Core::LinAlg::Initialization::zero);
  func.get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);

  dfuncdC.update(dPIe(0) / CinM.dot(AM), AM, 0.0);
}

template <class FUNC, typename T, typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::derivd_cd_c(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFinM, Core::LinAlg::Matrix<3, 3, T> const& AM,
    FUNC const& func, const int gp, ForceAnalytical const eleGID,
    Core::LinAlg::Matrix<6, 6, T>& dfuncdCdC) const
{
  // clear some variables
  dfuncdCdC.clear();

  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  iFinM_fad = iFinM;
  static Core::LinAlg::FADMatrix<3, 3> AM_fad(Core::LinAlg::Initialization::zero);
  AM_fad = AM;

  // Setup FAD
  // first derivative
  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  CM_fad = CM;
  CM_fad.diff(0, 9);

  Core::LinAlg::FADMatrix<3, 3> R_fad(Core::LinAlg::Initialization::zero);
  derivd_c(CM_fad, iFinM_fad, AM_fad, func, gp, eleGID, R_fad);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 6; ++j) dfuncdCdC(i, j) = R_fad(i, i).dx(j);
  for (int j = 0; j < 6; ++j)
  {
    dfuncdCdC(3, j) = R_fad(0, 1).dx(j);
    dfuncdCdC(4, j) = R_fad(1, 2).dx(j);
    dfuncdCdC(5, j) = R_fad(0, 2).dx(j);
  }
}

template <class FUNC, typename T>
void Mat::Elastic::RemodelFiber::derivd_cd_c(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFinM, Core::LinAlg::Matrix<3, 3, T> const& AM,
    FUNC const& func, const int gp, int const eleGID,
    Core::LinAlg::Matrix<6, 6, T>& dfuncdCdC) const
{
  // clear some variables
  dfuncdCdC.clear();

  // elastic right Cauchy-Green in matrix notation
  Core::LinAlg::SymmetricTensor<T, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));
  static Core::LinAlg::Matrix<3, 3, T> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> CinM(Core::LinAlg::Initialization::zero);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);

  // get derivatives of strain energy function w.r.t. I4
  static Core::LinAlg::Matrix<2, 1, T> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1, T> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1, T> dddPIIIe(Core::LinAlg::Initialization::zero);
  func.get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);

  static Core::LinAlg::Matrix<6, 1, T> Av(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(AM, Av);
  dfuncdCdC.multiply_nt(ddPIIe(0) / (CinM.dot(AM) * CinM.dot(AM)), Av, Av, 0.0);
}

void Mat::Elastic::RemodelFiber::add_stress_cmat(Core::LinAlg::Matrix<3, 3> const& CM,
    Core::LinAlg::Matrix<3, 3> const& iFgM, FiberData const& fiberdat, int const gp,
    int const eleGID, Core::LinAlg::Matrix<6, 1>& stress, Core::LinAlg::Matrix<6, 6>& cmat) const
{
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::Matrix<3, 3> firstderivM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> firstderivv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 6> secderiv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::SymmetricTensor<double, 3, 3> stressactv{};
  static Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmatact{};
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiberdat.fiber)).get() !=
      nullptr)
  {
    derivd_c(CM, iFinM, fiberdat.AM, *t1, gp, eleGID, firstderivM);
    derivd_cd_c(CM, iFinM, fiberdat.AM, *t1, gp, eleGID, secderiv);
  }
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiberdat.fiber))
               .get() != nullptr)
  {
    derivd_c(CM, iFinM, fiberdat.AM, *t2, gp, eleGID, firstderivM);
    derivd_cd_c(CM, iFinM, fiberdat.AM, *t2, gp, eleGID, secderiv);
    t2->evaluate_active_stress_cmat_aniso(
        Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CM)), cmatact, stressactv, gp,
        eleGID);
    stress.update(fiberdat.cur_rho[gp], Core::LinAlg::make_stress_like_voigt_view(stressactv), 1.0);
    cmat.update(fiberdat.cur_rho[gp], Core::LinAlg::make_stress_like_voigt_view(cmatact), 1.0);
  }

  Core::LinAlg::Voigt::Stresses::matrix_to_vector(firstderivM, firstderivv);
  stress.update(2.0 * fiberdat.cur_rho[gp], firstderivv, 1.0);
  cmat.update(4.0 * fiberdat.cur_rho[gp], secderiv, 1.0);
}

template <typename T>
void Mat::Elastic::RemodelFiber::evaluate_local_cauchy_stress(
    Core::LinAlg::Matrix<3, 3, T> const& CM, Core::LinAlg::Matrix<3, 3, T> const& iFinM,
    Core::LinAlg::Matrix<3, 3, T> const& AM, std::shared_ptr<Mat::Elastic::Summand> const& fiber,
    const int gp, int const eleGID, T& sig) const
{
  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  // const Core::LinAlg::Matrix<3, 3, T> CeM =
  // Core::LinAlg::make_matrix(Core::LinAlg::get_full(Ce));
  static Core::LinAlg::Matrix<3, 3, T> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> CinM(Core::LinAlg::Initialization::zero);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  const Core::LinAlg::SymmetricTensor<T, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));
  static Core::LinAlg::Matrix<2, 1, T> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1, T> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1, T> dddPIIIe(Core::LinAlg::Initialization::zero);
  T dPIact = 0.0;
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiber)).get())
  {
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  }
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiber)).get())
  {
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
    t2->get_derivative_aniso_active(dPIact);
  }

  sig = 2.0 * dPIe(0) * CM.dot(AM) / CinM.dot(AM) + dPIact;
}

template <typename T>
void Mat::Elastic::RemodelFiber::evaluatedsigd_ce(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFgM, Core::LinAlg::Matrix<3, 3, T> const& iFrM,
    Core::LinAlg::Matrix<3, 3, T> const& AM, std::shared_ptr<Mat::Elastic::Summand> const& fiber,
    const int gp, int const eleGID, Core::LinAlg::Matrix<3, 3, T>& dsigdCe) const
{
  // clear some variables
  dsigdCe.clear();

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::Matrix<3, 3, T> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> CinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> iFinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3, T> AgrM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, iFrM, 0.0);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  tmp.multiply_nn(1.0, FinM, AM, 0.0);
  AgrM.multiply_nt(1.0 / CinM.dot(AM), tmp, FinM, 0.0);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);
  const Core::LinAlg::SymmetricTensor<T, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));
  static Core::LinAlg::Matrix<2, 1, T> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1, T> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1, T> dddPIIIe(Core::LinAlg::Initialization::zero);
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiber)).get())
  {
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  }
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiber)).get())
  {
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  }

  dsigdCe.update(
      2.0 * (ddPIIe(0) * Core::LinAlg::make_matrix(Core::LinAlg::get_full(Ce)).dot(AgrM) + dPIe(0)),
      AgrM, 0.0);
}

template <typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::evaluatedsigd_ced_c(Core::LinAlg::Matrix<3, 3> const& CM,
    Core::LinAlg::Matrix<3, 3> const& iFgM, Core::LinAlg::Matrix<3, 3> const& iFrM,
    Core::LinAlg::Matrix<3, 3> const& AM, std::shared_ptr<Mat::Elastic::Summand> const fiber,
    const int gp, ForceAnalytical const eleGID, Core::LinAlg::Matrix<6, 6>& dsigdCedC) const
{
  // clear some variables
  dsigdCedC.clear();

  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  CM_fad = CM;
  CM_fad.diff(0, 9);
  static Core::LinAlg::FADMatrix<3, 3> iFgM_fad(Core::LinAlg::Initialization::zero);
  iFgM_fad = iFgM;
  static Core::LinAlg::FADMatrix<3, 3> iFrM_fad(Core::LinAlg::Initialization::zero);
  iFrM_fad = iFrM;
  static Core::LinAlg::FADMatrix<3, 3> AM_fad(Core::LinAlg::Initialization::zero);
  AM_fad = AM;

  static Core::LinAlg::FADMatrix<3, 3> dsigdCeM_fad(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM_fad, iFgM_fad, iFrM_fad, AM_fad, fiber, gp, eleGID, dsigdCeM_fad);

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j) dsigdCedC(i, j) = dsigdCeM_fad(i, i).dx(j);
    for (int j = 3; j < 6; ++j)
      dsigdCedC(i, j) = 0.5 * (dsigdCeM_fad(i, i).dx(j) + dsigdCeM_fad(i, i).dx(j + 3));
  }
  for (int j = 0; j < 3; ++j)
  {
    dsigdCedC(3, j) = 0.5 * (dsigdCeM_fad(0, 1).dx(j) + dsigdCeM_fad(1, 0).dx(j));
    dsigdCedC(4, j) = 0.5 * (dsigdCeM_fad(1, 2).dx(j) + dsigdCeM_fad(2, 1).dx(j));
    dsigdCedC(5, j) = 0.5 * (dsigdCeM_fad(0, 2).dx(j) + dsigdCeM_fad(2, 0).dx(j));
  }
  for (int j = 3; j < 6; ++j)
  {
    dsigdCedC(3, j) = 0.25 * (dsigdCeM_fad(0, 1).dx(j) + dsigdCeM_fad(0, 1).dx(j + 3) +
                                 dsigdCeM_fad(1, 0).dx(j) + dsigdCeM_fad(1, 0).dx(j + 3));
    dsigdCedC(4, j) = 0.25 * (dsigdCeM_fad(1, 2).dx(j) + dsigdCeM_fad(1, 2).dx(j + 3) +
                                 dsigdCeM_fad(2, 1).dx(j) + dsigdCeM_fad(2, 1).dx(j + 3));
    dsigdCedC(5, j) = 0.25 * (dsigdCeM_fad(0, 2).dx(j) + dsigdCeM_fad(0, 2).dx(j + 3) +
                                 dsigdCeM_fad(2, 0).dx(j) + dsigdCeM_fad(2, 0).dx(j + 3));
  }
}

void Mat::Elastic::RemodelFiber::evaluatedsigd_ced_c(Core::LinAlg::Matrix<3, 3> const& CM,
    Core::LinAlg::Matrix<3, 3> const& iFgM, Core::LinAlg::Matrix<3, 3> const& iFrM,
    Core::LinAlg::Matrix<3, 3> const& AM, std::shared_ptr<Mat::Elastic::Summand> const& fiber,
    const int gp, int const eleGID, Core::LinAlg::Matrix<6, 6>& dsigdCedC) const
{
  // clear some variables
  dsigdCedC.clear();

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> AgrM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, iFrM, 0.0);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  tmp.multiply_nn(1.0, FinM, AM, 0.0);
  AgrM.multiply_nt(1.0 / CinM.dot(AM), tmp, FinM, 0.0);
  const Core::LinAlg::SymmetricTensor<double, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));
  static Core::LinAlg::Matrix<2, 1> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1> dddPIIIe(Core::LinAlg::Initialization::zero);
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiber)).get())
  {
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  }
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiber)).get())
  {
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  }

  static Core::LinAlg::Matrix<6, 1> Agrv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> Av(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(AgrM, Agrv);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(AM, Av);
  dsigdCedC.multiply_nt(
      2.0 / CinM.dot(AM) * (dddPIIIe(0) * CM.dot(AM) / CinM.dot(AM) + 2.0 * ddPIIe(0)), Agrv, Av,
      0.0);
}

template <typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::evaluate_derivatives_cauchy_growth(
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<3, 3> const& dFgdrhoM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    FiberData const& fiberdat, int const gp, ForceAnalytical const eleGID, double& dsigdrho,
    Core::LinAlg::Matrix<3, 3>& dsigdCedrhoM) const
{
  // clear some variables
  dsigdrho = 0.0;
  dsigdCedrhoM.clear();

  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> iFgM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> iFrM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> AM_fad(Core::LinAlg::Initialization::zero);
  iFgM_fad = iFgM;
  iFgM_fad.diff(0, 9);
  iFrM_fad = fiberdat.iFrM[gp];
  CM_fad = CM;
  AM_fad = fiberdat.AM;
  iFinM_fad.multiply_nn(1.0, iFgM_fad, iFrM_fad, 0.0);

  FAD sig_fad = 0.0;
  evaluate_local_cauchy_stress(CM_fad, iFinM_fad, AM_fad, fiberdat.fiber, gp, eleGID, sig_fad);

  Core::LinAlg::Matrix<3, 3> dsigdiFgM(Core::LinAlg::Initialization::zero);
  first_deriv_to_matrix(sig_fad, dsigdiFgM);
  dsigdrho = dsigdiFgM.dot(diFgdrhoM);


  static Core::LinAlg::FADMatrix<3, 3> dsigdCeM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 9> dsigdCediFg(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM_fad, iFgM_fad, iFrM_fad, AM_fad, fiberdat.fiber, gp, eleGID, dsigdCeM_fad);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 9; ++j) dsigdCediFg(i, j) = dsigdCeM_fad(i, i).dx(j);
  for (int j = 0; j < 9; ++j)
    dsigdCediFg(3, j) = 0.5 * (dsigdCeM_fad(0, 1).dx(j) + dsigdCeM_fad(1, 0).dx(j));
  for (int j = 0; j < 9; ++j)
    dsigdCediFg(4, j) = 0.5 * (dsigdCeM_fad(1, 2).dx(j) + dsigdCeM_fad(2, 1).dx(j));
  for (int j = 0; j < 9; ++j)
    dsigdCediFg(5, j) = 0.5 * (dsigdCeM_fad(0, 2).dx(j) + dsigdCeM_fad(2, 0).dx(j));

  static Core::LinAlg::Matrix<9, 1> diFgdrho9x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> tmp6x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFgdrhoM, diFgdrho9x1);
  tmp6x1.multiply_nn(1.0, dsigdCediFg, diFgdrho9x1, 0.0);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(tmp6x1, dsigdCedrhoM);
}

void Mat::Elastic::RemodelFiber::evaluate_derivatives_cauchy_growth(
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<3, 3> const& dFgdrhoM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    FiberData const& fiberdat, int const gp, int const eleGID, double& dsigdrho,
    Core::LinAlg::Matrix<3, 3>& dsigdCedrhoM) const
{
  // clear some variables
  dsigdrho = 0.0;
  dsigdCedrhoM.clear();

  static Core::LinAlg::Matrix<3, 3> CAM(Core::LinAlg::Initialization::zero);
  CAM.multiply_nn(1.0, CM, fiberdat.AM, 0.0);
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinAM(Core::LinAlg::Initialization::zero);
  FgM.invert(iFgM);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  CinAM.multiply_tn(1.0, CinM, fiberdat.AM, 0.0);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);

  const Core::LinAlg::SymmetricTensor<double, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));

  Core::LinAlg::Matrix<3, 3> CeM = Core::LinAlg::make_matrix(Core::LinAlg::get_full(Ce));

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::Matrix<2, 1> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1> dddPIIIe(Core::LinAlg::Initialization::zero);
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiberdat.fiber)).get())
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiberdat.fiber))
               .get())
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);

  static Core::LinAlg::Matrix<3, 3> dsigdiFgM(Core::LinAlg::Initialization::zero);
  dsigdiFgM.multiply_nt(
      4.0 * ddPIIe(0) * CM.dot(fiberdat.AM) / (CinM.dot(fiberdat.AM) * CinM.dot(fiberdat.AM)), CAM,
      FgM, 0.0);
  dsigdiFgM.multiply_nt(
      4.0 * dPIe(0) * CM.dot(fiberdat.AM) / (CinM.dot(fiberdat.AM) * CinM.dot(fiberdat.AM)), CinAM,
      FgM, 1.0);
  dsigdrho = dsigdiFgM.dot(diFgdrhoM);


  static Core::LinAlg::Matrix<3, 3> AgrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CAFgTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinAFgTM(Core::LinAlg::Initialization::zero);
  tmp.multiply_nn(1.0, FinM, fiberdat.AM, 0.0);
  AgrM.multiply_nt(1.0 / CinM.dot(fiberdat.AM), tmp, FinM, 0.0);
  CAFgTM.multiply_nt(1.0, CAM, FgM, 0.0);
  CinAFgTM.multiply_nt(1.0, CinAM, FgM, 0.0);

  dsigdCedrhoM.update(
      4.0 / CinM.dot(fiberdat.AM) *
          (dddPIIIe(0) * CAFgTM.dot(diFgdrhoM) * CeM.dot(AgrM) +
              ddPIIe(0) * CM.dot(fiberdat.AM) * CinAFgTM.dot(diFgdrhoM) / CinM.dot(fiberdat.AM) +
              ddPIIe(0) * CAFgTM.dot(diFgdrhoM)),
      AgrM, 0.0);

  static Core::LinAlg::Matrix<3, 3> FrM(Core::LinAlg::Initialization::zero);
  FrM.invert(fiberdat.iFrM[gp]);
  static Core::LinAlg::Matrix<3, 3> FrdFgdrhoM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> dAgrdrhoM(Core::LinAlg::Initialization::zero);
  FrdFgdrhoM.multiply_nn(1.0, FrM, dFgdrhoM, 0.0);
  tmp.multiply_nn(1.0, FrdFgdrhoM, fiberdat.AM, 0.0);
  dAgrdrhoM.multiply_nt(1.0 / CinM.dot(fiberdat.AM), tmp, FinM, 0.0);
  dAgrdrhoM.multiply_nt(1.0 / CinM.dot(fiberdat.AM), FinM, tmp, 1.0);
  dAgrdrhoM.update(2.0 * CinAFgTM.dot(diFgdrhoM) / CinM.dot(fiberdat.AM), AgrM, 1.0);

  dsigdCedrhoM.update(2.0 * (ddPIIe(0) * CeM.dot(AgrM) + dPIe(0)), dAgrdrhoM, 1.0);
}

template <typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::evaluate_derivatives_cauchy_remodel(
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    FiberData const& fiberdat, int const gp, ForceAnalytical const eleGID, double& dsigdlambr,
    Core::LinAlg::Matrix<3, 3>& dsigdCedlambrM) const
{
  // clear some variables
  dsigdlambr = 0.0;
  dsigdCedlambrM.clear();

  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> iFgM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> iFrM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> AM_fad(Core::LinAlg::Initialization::zero);
  iFgM_fad = iFgM;
  iFrM_fad = fiberdat.iFrM[gp];
  iFrM_fad.diff(0, 9);
  CM_fad = CM;
  AM_fad = fiberdat.AM;
  iFinM_fad.multiply_nn(1.0, iFgM_fad, iFrM_fad, 0.0);

  FAD sig_fad = 0.0;
  evaluate_local_cauchy_stress(CM_fad, iFinM_fad, AM_fad, fiberdat.fiber, gp, eleGID, sig_fad);

  Core::LinAlg::Matrix<3, 3> dsigdiFrM(Core::LinAlg::Initialization::zero);
  first_deriv_to_matrix(sig_fad, dsigdiFrM);
  dsigdlambr = dsigdiFrM.dot(fiberdat.diFrdlambrM[gp]);


  static Core::LinAlg::FADMatrix<3, 3> dsigdCeM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 9> dsigdCediFr(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM_fad, iFgM_fad, iFrM_fad, AM_fad, fiberdat.fiber, gp, eleGID, dsigdCeM_fad);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 9; ++j) dsigdCediFr(i, j) = dsigdCeM_fad(i, i).dx(j);
  for (int j = 0; j < 9; ++j)
    dsigdCediFr(3, j) = 0.5 * (dsigdCeM_fad(0, 1).dx(j) + dsigdCeM_fad(1, 0).dx(j));
  for (int j = 0; j < 9; ++j)
    dsigdCediFr(4, j) = 0.5 * (dsigdCeM_fad(1, 2).dx(j) + dsigdCeM_fad(2, 1).dx(j));
  for (int j = 0; j < 9; ++j)
    dsigdCediFr(5, j) = 0.5 * (dsigdCeM_fad(0, 2).dx(j) + dsigdCeM_fad(2, 0).dx(j));

  static Core::LinAlg::Matrix<9, 1> diFrdlambr9x1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> tmp6x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(fiberdat.diFrdlambrM[gp], diFrdlambr9x1);
  tmp6x1.multiply_nn(1.0, dsigdCediFr, diFrdlambr9x1, 0.0);
  Core::LinAlg::Voigt::Stresses::vector_to_matrix(tmp6x1, dsigdCedlambrM);
}

void Mat::Elastic::RemodelFiber::evaluate_derivatives_cauchy_remodel(
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    FiberData const& fiberdat, int const gp, int const eleGID, double& dsigdlambr,
    Core::LinAlg::Matrix<3, 3>& dsigdCedlambrM) const
{
  // clear some variables
  dsigdlambr = 0.0;
  dsigdCedlambrM.clear();

  static Core::LinAlg::Matrix<3, 3> CAFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> tmp1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> AFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FinAFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinAM(Core::LinAlg::Initialization::zero);
  FrM.invert(fiberdat.iFrM[gp]);
  FgM.invert(iFgM);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);
  FinM.invert(iFinM);
  AFinTM.multiply_nt(1.0, fiberdat.AM, FinM, 0.0);
  FinAFinTM.multiply_nn(1.0, FinM, AFinTM, 0.0);
  CAFinTM.multiply_nn(1.0, CM, AFinTM, 0.0);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  CinAM.multiply_tn(1.0, CinM, fiberdat.AM, 0.0);
  tmp1.multiply_nn(1.0, CM, iFinM, 0.0);

  const Core::LinAlg::SymmetricTensor<double, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));

  const Core::LinAlg::Matrix<3, 3> CeM = Core::LinAlg::make_matrix(Core::LinAlg::get_full(Ce));

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;
  static Core::LinAlg::Matrix<2, 1> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1> dddPIIIe(Core::LinAlg::Initialization::zero);
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiberdat.fiber)).get())
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiberdat.fiber))
               .get())
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);

  static Core::LinAlg::Matrix<3, 3> dsigdiFrM(Core::LinAlg::Initialization::zero);
  dsigdiFrM.multiply_tn(
      4.0 * ddPIIe(0) * CM.dot(fiberdat.AM) / (CinM.dot(fiberdat.AM) * CinM.dot(fiberdat.AM)), iFgM,
      CAFinTM, 0.0);
  dsigdiFrM.multiply_tn(
      4.0 * dPIe(0) * CM.dot(fiberdat.AM) / (CinM.dot(fiberdat.AM) * CinM.dot(fiberdat.AM)), FrM,
      FinAFinTM, 1.0);
  dsigdlambr = dsigdiFrM.dot(fiberdat.diFrdlambrM[gp]);


  static Core::LinAlg::Matrix<3, 3> AgrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrTFinAFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFgTCAFinTM(Core::LinAlg::Initialization::zero);
  tmp.multiply_nn(1.0, FinM, fiberdat.AM, 0.0);
  AgrM.multiply_nt(1.0 / CinM.dot(fiberdat.AM), tmp, FinM, 0.0);
  FrTFinAFinTM.multiply_tn(1.0, FrM, FinAFinTM, 0.0);
  iFgTCAFinTM.multiply_tn(1.0, iFgM, CAFinTM, 0.0);

  dsigdCedlambrM.update(
      4.0 / CinM.dot(fiberdat.AM) *
          (dddPIIIe(0) * iFgTCAFinTM.dot(fiberdat.diFrdlambrM[gp]) * CeM.dot(AgrM) +
              ddPIIe(0) * CM.dot(fiberdat.AM) * FrTFinAFinTM.dot(fiberdat.diFrdlambrM[gp]) /
                  CinM.dot(fiberdat.AM) +
              ddPIIe(0) * iFgTCAFinTM.dot(fiberdat.diFrdlambrM[gp])),
      AgrM, 0.0);

  static Core::LinAlg::Matrix<3, 3> dFrdlambrFgM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> dAgrdlambrM(Core::LinAlg::Initialization::zero);
  dFrdlambrFgM.multiply_nn(1.0, fiberdat.dFrdlambrM[gp], FgM, 0.0);
  tmp.multiply_nn(1.0, dFrdlambrFgM, fiberdat.AM, 0.0);
  dAgrdlambrM.multiply_nt(1.0 / CinM.dot(fiberdat.AM), tmp, FinM, 0.0);
  dAgrdlambrM.multiply_nt(1.0 / CinM.dot(fiberdat.AM), FinM, tmp, 1.0);
  dAgrdlambrM.update(
      2.0 * FrTFinAFinTM.dot(fiberdat.diFrdlambrM[gp]) / CinM.dot(fiberdat.AM), AgrM, 1.0);

  dsigdCedlambrM.update(2.0 * (ddPIIe(0) * CeM.dot(AgrM) + dPIe(0)), dAgrdlambrM, 1.0);
}

template <typename T, typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::evaluatedsigd_c(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFinM, Core::LinAlg::Matrix<3, 3, T> const& AM,
    std::shared_ptr<Mat::Elastic::Summand> const fiber, const int gp, ForceAnalytical const eleGID,
    Core::LinAlg::Matrix<3, 3, T>& dsigdC) const
{
  // clear some variables
  dsigdC.clear();

  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  iFinM_fad = iFinM;
  static Core::LinAlg::FADMatrix<3, 3> AM_fad(Core::LinAlg::Initialization::zero);
  AM_fad = AM;

  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  CM_fad = CM;
  CM_fad.diff(0, 9);

  FAD sig_fad = 0.0;
  evaluate_local_cauchy_stress(CM_fad, iFinM_fad, AM_fad, fiber, gp, eleGID, sig_fad);

  Core::LinAlg::Matrix<3, 3, T> tmp(Core::LinAlg::Initialization::zero);
  FirstDerivToMatrix(sig_fad, dsigdC);
}

template <typename T>
void Mat::Elastic::RemodelFiber::evaluatedsigd_c(Core::LinAlg::Matrix<3, 3, T> const& CM,
    Core::LinAlg::Matrix<3, 3, T> const& iFinM, Core::LinAlg::Matrix<3, 3, T> const& AM,
    std::shared_ptr<Mat::Elastic::Summand> const& fiber, const int gp, int const eleGID,
    Core::LinAlg::Matrix<3, 3, T>& dsigdC) const
{
  // clear some variables
  dsigdC.clear();

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::Matrix<3, 3, T> FinM(Core::LinAlg::Initialization::zero);
  FinM.invert(iFinM);
  static Core::LinAlg::Matrix<3, 3, T> CinM(Core::LinAlg::Initialization::zero);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  const Core::LinAlg::SymmetricTensor<double, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));

  const Core::LinAlg::Matrix<3, 3> CeM = Core::LinAlg::make_matrix(Core::LinAlg::get_full(Ce));

  static Core::LinAlg::Matrix<2, 1> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1> dddPIIIe(Core::LinAlg::Initialization::zero);
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiber)).get())
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiber)).get())
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);

  dsigdC.update(2.0 / CinM.dot(AM) * (ddPIIe(0) * CM.dot(AM) / CinM.dot(AM) + dPIe(0)), AM, 0.0);
}

void Mat::Elastic::RemodelFiber::evaluate_evolution_equation(double& rg, double& rr,
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM, double const& dt,
    FiberData const& fiberdat, int const gp, int const eleGID) const
{
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);
  double sig = 0.0;
  evaluate_local_cauchy_stress(CM, iFinM, fiberdat.AM, fiberdat.fiber, gp, eleGID, sig);

  // Growth evolution equation
  fiberdat.growth->evaluate_func(rg, sig, fiberdat.cur_rho[gp], fiberdat.last_rho[gp], dt, eleGID);

  static Core::LinAlg::Matrix<3, 3> dsigdCe(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM, iFgM, fiberdat.iFrM[gp], fiberdat.AM, fiberdat.fiber, gp, eleGID, dsigdCe);

  static Core::LinAlg::Matrix<3, 3> YM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrdotiFrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);
  CeM.multiply_tn(1.0, iFinM, tmp, 0.0);
  FrdotiFrM.multiply_nn(1.0, fiberdat.FrdotM[gp], fiberdat.iFrM[gp], 0.0);
  YM.multiply_nn(1.0, CeM, FrdotiFrM, 0.0);

  // Remodel evolution equation
  fiberdat.remodel->evaluate_func(rr, sig, YM, dsigdCe, eleGID);
}

void Mat::Elastic::RemodelFiber::evaluate_derivative_evolution_equation(double& dWidrhoi,
    double& dWidrhoj, double& dWdlambr, double& dEdrho, double& dEdlambr,
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    Core::LinAlg::Matrix<3, 3> const& dFgdrhoM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    double const& dt, FiberData const& fiberdat, int const gp, int const eleGID) const
{
  // Derivative of growth evolution eq. w.r.t. the mass density
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);
  double sig = 0.0;
  evaluate_local_cauchy_stress(CM, iFinM, fiberdat.AM, fiberdat.fiber, gp, eleGID, sig);

  double dsigdrho = 0.0;
  static Core::LinAlg::Matrix<3, 3> dsigdCedrhoM(Core::LinAlg::Initialization::zero);
  evaluate_derivatives_cauchy_growth(
      CM, iFgM, dFgdrhoM, diFgdrhoM, fiberdat, gp, eleGID, dsigdrho, dsigdCedrhoM);

  fiberdat.growth->evaluated_funcidrhoi(dWidrhoi, sig, dsigdrho, fiberdat.cur_rho[gp], dt, eleGID);
  fiberdat.growth->evaluated_funcidrhoj(dWidrhoj, sig, dsigdrho, fiberdat.cur_rho[gp], dt, eleGID);

  // Derivative of growth evolution eq. w.r.t. the inelastic remodel stretch
  static Core::LinAlg::Matrix<3, 3> dsigdCedlambrM(Core::LinAlg::Initialization::zero);
  double dsigdlambr = 0.0;
  evaluate_derivatives_cauchy_remodel(CM, iFgM, fiberdat, gp, eleGID, dsigdlambr, dsigdCedlambrM);

  fiberdat.growth->evaluated_funcidlambr(dWdlambr, dsigdlambr, fiberdat.cur_rho[gp], dt, eleGID);

  // Derivative of remodel evolution eq. w.r.t. the inelastic remodel stretch
  static Core::LinAlg::Matrix<3, 3> tmp1(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> tmp2(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrdotiFrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> YM(Core::LinAlg::Initialization::zero);
  tmp1.multiply_nn(1.0, CM, iFinM, 0.0);
  CeM.multiply_tn(1.0, iFinM, tmp1, 0.0);
  FrdotiFrM.multiply_nn(1.0, fiberdat.FrdotM[gp], fiberdat.iFrM[gp], 0.0);
  YM.multiply_nn(1.0, CeM, FrdotiFrM, 0.0);

  static Core::LinAlg::Matrix<3, 3> dYdlambrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> dFrdotdlambriFrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrdotdiFrdlambrM(Core::LinAlg::Initialization::zero);
  tmp1.multiply_tn(1.0, iFinM, CM, 0.0);
  tmp2.multiply_nn(1.0, tmp1, iFgM, 0.0);
  tmp1.multiply_nn(1.0, tmp2, fiberdat.diFrdlambrM[gp], 0.0);
  dYdlambrM.multiply_nn(1.0, tmp1, FrdotiFrM, 0.0);
  dYdlambrM.multiply_tn(1.0, tmp1, FrdotiFrM, 1.0);
  dFrdotdlambriFrM.multiply_nn(1.0, fiberdat.dFrdotdlambrM[gp], fiberdat.iFrM[gp], 0.0);
  dYdlambrM.multiply_nn(1.0, CeM, dFrdotdlambriFrM, 1.0);
  FrdotdiFrdlambrM.multiply_nn(1.0, fiberdat.FrdotM[gp], fiberdat.diFrdlambrM[gp], 0.0);
  dYdlambrM.multiply_nn(1.0, CeM, FrdotdiFrdlambrM, 1.0);

  static Core::LinAlg::Matrix<3, 3> dsigdCeM(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM, iFgM, fiberdat.iFrM[gp], fiberdat.AM, fiberdat.fiber, gp, eleGID, dsigdCeM);

  fiberdat.remodel->evaluated_funcidlambri(
      dEdlambr, sig, dsigdlambr, YM, dYdlambrM, dsigdCeM, dsigdCedlambrM, eleGID);

  // Derivative of remodel evolution eq. w.r.t. the mass density
  static Core::LinAlg::Matrix<3, 3> dYdrhoM(Core::LinAlg::Initialization::zero);
  tmp1.multiply_tt(1.0, fiberdat.iFrM[gp], diFgdrhoM, 0.0);
  tmp2.multiply_nn(1.0, tmp1, CM, 0.0);
  tmp1.multiply_nn(1.0, tmp2, iFinM, 0.0);
  dYdrhoM.multiply_nn(1.0, tmp1, FrdotiFrM, 0.0);
  dYdrhoM.multiply_tn(1.0, tmp1, FrdotiFrM, 1.0);

  fiberdat.remodel->evaluated_funcidrho(
      dEdrho, sig, dsigdrho, YM, dYdrhoM, dsigdCeM, dsigdCedrhoM, eleGID);
}

void Mat::Elastic::RemodelFiber::evaluated_evolution_equationd_c(Core::LinAlg::Matrix<1, 6>& dWdC,
    Core::LinAlg::Matrix<1, 6>& dEdC, Core::LinAlg::Matrix<3, 3> const& CM,
    Core::LinAlg::Matrix<3, 3> const& iFgM, double const& dt, FiberData const& fiberdat,
    int const k, int const gp, int const eleGID)
{
  // Growth law
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);

  static Core::LinAlg::Matrix<3, 3> dsigdC(Core::LinAlg::Initialization::zero);
  evaluatedsigd_c(CM, iFinM, fiberdat.AM, fiberdat.fiber, gp, eleGID, dsigdC);
  static Core::LinAlg::Matrix<6, 1> dsigdCv(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(dsigdC, dsigdCv);

  fiberdat.growth->evaluated_funcid_c(dWdC, dsigdCv, fiberdat.cur_rho[gp], dt, eleGID);


  // Remodel law
  double sig = 0.0;
  evaluate_local_cauchy_stress(CM, iFinM, fiberdat.AM, fiberdat.fiber, gp, eleGID, sig);
  cauchystress_[k][gp] = sig;
  static Core::LinAlg::Matrix<6, 6> dsigdCedC(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ced_c(
      CM, iFgM, fiberdat.iFrM[gp], fiberdat.AM, fiberdat.fiber, gp, eleGID, dsigdCedC);
  static Core::LinAlg::Matrix<3, 3> dsigdCeM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<9, 1> dsigdCe9x1(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM, iFgM, fiberdat.iFrM[gp], fiberdat.AM, fiberdat.fiber, gp, eleGID, dsigdCeM);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(dsigdCeM, dsigdCe9x1);

  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrdotiFrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> YM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> Y_strain(Core::LinAlg::Initialization::zero);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);
  CeM.multiply_tn(1.0, iFinM, tmp, 0.0);
  FrdotiFrM.multiply_nn(1.0, fiberdat.FrdotM[gp], fiberdat.iFrM[gp], 0.0);
  YM.multiply_nn(1.0, CeM, FrdotiFrM, 0.0);
  Core::LinAlg::Voigt::Strains::matrix_to_vector(YM, Y_strain);

  Core::LinAlg::Matrix<9, 6> dYdC(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFrTFrdotTiFinTM(Core::LinAlg::Initialization::zero);
  iFinTM.update_t(1.0, iFinM, 0.0);
  iFrTFrdotTiFinTM.multiply_tt(1.0, FrdotiFrM, iFinM, 0.0);
  Core::LinAlg::FourTensorOperations::add_left_non_symmetric_holzapfel_product(
      dYdC, iFinTM, iFrTFrdotTiFinTM, 0.5);

  fiberdat.remodel->evaluated_funcid_c(
      dEdC, sig, dsigdCv, Y_strain, dYdC, dsigdCe9x1, dsigdCedC, eleGID);
}

void Mat::Elastic::RemodelFiber::evaluated_evolution_equationdt(double& drhodt, double& dlambrdt,
    Core::LinAlg::Matrix<3, 3> const& CM, Core::LinAlg::Matrix<3, 3> const& iFgM,
    FiberData const& fiberdat, int const k, int const gp, int const eleGID)
{
  // Time derivative of the mass density
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);
  double sig = 0.0;
  evaluate_local_cauchy_stress(CM, iFinM, fiberdat.AM, fiberdat.fiber, gp, eleGID, sig);
  cauchystress_[k][gp] = sig;

  fiberdat.growth->evaluatedrhodt(drhodt, sig, fiberdat.cur_rho[gp], eleGID);

  // Time derivative of the remodel stretch
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrdotredM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CeM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> YredM(Core::LinAlg::Initialization::zero);
  tmp.multiply_nn(1.0, CM, iFinM, 0.0);
  CeM.multiply_tn(1.0, iFinM, tmp, 0.0);
  static Core::LinAlg::Matrix<3, 3> dsigdCeM(Core::LinAlg::Initialization::zero);
  evaluatedsigd_ce(CM, iFgM, fiberdat.iFrM[gp], fiberdat.AM, fiberdat.fiber, gp, eleGID, dsigdCeM);

  FrdotredM.update(1.0 / fiberdat.G, fiberdat.AM, 0.0);
  FrdotredM.update(-0.5 * std::pow(fiberdat.cur_lambr[gp], -3. / 2.) * std::pow(fiberdat.G, 0.5),
      fiberdat.AM_orth, 1.0);
  tmp.multiply_nn(1.0, CeM, FrdotredM, 0.0);
  YredM.multiply_nn(1.0, tmp, fiberdat.iFrM[gp], 0.0);

  fiberdat.remodel->evaluatedlambrdt(dlambrdt, sig, YredM, dsigdCeM, eleGID);
}

template <typename ForceAnalytical>
void Mat::Elastic::RemodelFiber::evaluate_derivatives2nd_piola_kirchhoff_growth_remodel(
    Core::LinAlg::Matrix<6, 1>& dSidrhoi, Core::LinAlg::Matrix<6, 1>& dSidrhoj,
    Core::LinAlg::Matrix<6, 1>& dSdlambr, Core::LinAlg::Matrix<3, 3> const& CM,
    Core::LinAlg::Matrix<3, 3> const& iFgM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    FiberData const& fiberdat, int const gp, ForceAnalytical const eleGID) const
{
  // Derivative w.r.t. the mass density
  static Core::LinAlg::FADMatrix<3, 3> CM_fad(Core::LinAlg::Initialization::zero);
  CM_fad = CM;
  static Core::LinAlg::FADMatrix<3, 3> iFgM_fad(Core::LinAlg::Initialization::zero);
  iFgM_fad = iFgM;
  iFgM_fad.diff(0, 18);
  static Core::LinAlg::FADMatrix<3, 3> iFrM_fad(Core::LinAlg::Initialization::zero);
  iFrM_fad = fiberdat.iFrM[gp];
  iFrM_fad.diff(9, 18);
  static Core::LinAlg::FADMatrix<3, 3> iFinM_fad(Core::LinAlg::Initialization::zero);
  iFinM_fad.multiply_nn(1.0, iFgM_fad, iFrM_fad, 0.0);
  static Core::LinAlg::FADMatrix<3, 3> AM_fad(Core::LinAlg::Initialization::zero);
  AM_fad = fiberdat.AM;

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::FADMatrix<3, 3> firstderivM_fad(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> Sactv(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 6> cmatact(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::FADMatrix<3, 3> S_fad(Core::LinAlg::Initialization::zero);
  S_fad.clear();
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiberdat.fiber)).get())
  {
    derivd_c(CM_fad, iFinM_fad, AM_fad, *t1, eleGID, firstderivM_fad);
  }
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiberdat.fiber))
               .get())
  {
    derivd_c(CM_fad, iFinM_fad, AM_fad, *t2, eleGID, firstderivM_fad);
    t2->evaluate_active_stress_cmat_aniso(CM, cmatact, Sactv, eleGID);
  }

  S_fad.update(2.0 * fiberdat.cur_rho[gp], firstderivM_fad, 1.0);

  static Core::LinAlg::Matrix<6, 9> dSdiFg(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 9; ++j) dSdiFg(i, j) = S_fad(i, i).dx(j);
  for (int j = 0; j < 9; ++j) dSdiFg(3, j) = 0.5 * (S_fad(0, 1).dx(j) + S_fad(1, 0).dx(j));
  for (int j = 0; j < 9; ++j) dSdiFg(4, j) = 0.5 * (S_fad(1, 2).dx(j) + S_fad(2, 1).dx(j));
  for (int j = 0; j < 9; ++j) dSdiFg(5, j) = 0.5 * (S_fad(0, 2).dx(j) + S_fad(2, 0).dx(j));

  static Core::LinAlg::Matrix<9, 1> diFgdrho9x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(diFgdrhoM, diFgdrho9x1);
  static Core::LinAlg::Matrix<3, 3> firstderivM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<6, 1> firstderivv(Core::LinAlg::Initialization::zero);
  firstderivM = firstderivM_fad.convertto_double();
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(firstderivM, firstderivv);
  dSidrhoj.multiply_nn(1.0, dSdiFg, diFgdrho9x1, 0.0);
  dSidrhoi.update(1.0, dSidrhoj, 0.0);
  dSidrhoi.update(2.0, firstderivv, 1.0);
  dSidrhoi.update(1.0, Sactv, 1.0);


  // Derivative w.r.t. the inelastic remodel stretch
  static Core::LinAlg::Matrix<6, 9> dSdiFr(Core::LinAlg::Initialization::zero);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 9; ++j) dSdiFr(i, j) = S_fad(i, i).dx(j + 9);
  for (int j = 0; j < 9; ++j) dSdiFr(3, j) = 0.5 * (S_fad(0, 1).dx(j + 9) + S_fad(1, 0).dx(j + 9));
  for (int j = 0; j < 9; ++j) dSdiFr(4, j) = 0.5 * (S_fad(1, 2).dx(j + 9) + S_fad(2, 1).dx(j + 9));
  for (int j = 0; j < 9; ++j) dSdiFr(5, j) = 0.5 * (S_fad(0, 2).dx(j + 9) + S_fad(2, 0).dx(j + 9));

  static Core::LinAlg::Matrix<9, 1> diFrdlambr9x1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::matrix_3x3_to_9x1(fiberdat.diFrdlambrM[gp], diFrdlambr9x1);
  dSdlambr.multiply_nn(1.0, dSdiFr, diFrdlambr9x1, 0.0);
}

void Mat::Elastic::RemodelFiber::evaluate_derivatives2nd_piola_kirchhoff_growth_remodel(
    Core::LinAlg::Matrix<6, 1>& dSidrhoi, Core::LinAlg::Matrix<6, 1>& dSidrhoj,
    Core::LinAlg::Matrix<6, 1>& dSdlambr, Core::LinAlg::Matrix<3, 3> const& CM,
    Core::LinAlg::Matrix<3, 3> const& iFgM, Core::LinAlg::Matrix<3, 3> const& diFgdrhoM,
    FiberData const& fiberdat, int const gp, int const eleGID) const
{
  // Clear some variables
  dSidrhoi.clear();
  dSidrhoj.clear();
  dSdlambr.clear();

  // Derivative w.r.t. the mass density
  static Core::LinAlg::Matrix<3, 3> tmp(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> iFinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FgM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CAFgTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> CinAFgTM(Core::LinAlg::Initialization::zero);
  FgM.invert(iFgM);
  iFinM.multiply_nn(1.0, iFgM, fiberdat.iFrM[gp], 0.0);
  FinM.invert(iFinM);
  CinM.multiply_tn(1.0, FinM, FinM, 0.0);
  tmp.multiply_tn(1.0, CinM, fiberdat.AM, 0.0);
  CinAFgTM.multiply_nt(1.0, tmp, FgM, 0.0);
  tmp.multiply_tn(1.0, CM, fiberdat.AM, 0.0);
  CAFgTM.multiply_nt(1.0, tmp, FgM, 0.0);

  const Core::LinAlg::SymmetricTensor<double, 3, 3> Ce = Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Core::LinAlg::make_tensor_view(iFinM)) *
      Core::LinAlg::make_tensor_view(CM) * Core::LinAlg::make_tensor_view(iFinM));

  // temporary pointers to check the type of the remodelfiber (active or passive)
  std::shared_ptr<Mat::Elastic::CoupAnisoExpo> t1;
  std::shared_ptr<Mat::Elastic::CoupAnisoExpoActive> t2;

  static Core::LinAlg::Matrix<2, 1> dPIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 1> ddPIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<4, 1> dddPIIIe(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::SymmetricTensor<double, 3, 3> stressactv{};
  static Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmatact{};
  if ((t1 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpo>(fiberdat.fiber)).get())
    t1->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
  else if ((t2 = std::dynamic_pointer_cast<Mat::Elastic::CoupAnisoExpoActive>(fiberdat.fiber))
               .get())
  {
    t2->get_derivatives_aniso(dPIe, ddPIIe, dddPIIIe, Ce, gp, eleGID);
    t2->evaluate_active_stress_cmat_aniso(
        Core::LinAlg::assume_symmetry(Core::LinAlg::make_tensor(CM)), cmatact, stressactv, gp,
        eleGID);
    dSidrhoi.update(1.0, Core::LinAlg::make_stress_like_voigt_view(stressactv), 0.0);
  }

  static Core::LinAlg::Matrix<6, 1> Av(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Voigt::Stresses::matrix_to_vector(fiberdat.AM, Av);
  dSidrhoj.update(4.0 * fiberdat.cur_rho[gp] / (CinM.dot(fiberdat.AM) * CinM.dot(fiberdat.AM)) *
                      (ddPIIe(0) * CAFgTM.dot(diFgdrhoM) + dPIe(0) * CinAFgTM.dot(diFgdrhoM)),
      Av, 0.0);
  dSidrhoi.update(1.0, dSidrhoj, 1.0);
  dSidrhoi.update(2.0 * dPIe(0) / CinM.dot(fiberdat.AM), Av, 1.0);


  // Derivative w.r.t. the inelastic remodel stretch
  static Core::LinAlg::Matrix<3, 3> iFgTCAFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> AFinTM(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<3, 3> FrTFinAFinTM(Core::LinAlg::Initialization::zero);
  FrM.invert(fiberdat.iFrM[gp]);
  AFinTM.multiply_nt(1.0, fiberdat.AM, FinM, 0.0);
  tmp.multiply_nn(1.0, FinM, AFinTM, 0.0);
  FrTFinAFinTM.multiply_tn(1.0, FrM, tmp, 0.0);
  tmp.multiply_nn(1.0, CM, AFinTM, 0.0);
  iFgTCAFinTM.multiply_tn(1.0, iFgM, tmp, 0.0);

  dSdlambr.update(4.0 * fiberdat.cur_rho[gp] / (CinM.dot(fiberdat.AM) * CinM.dot(fiberdat.AM)) *
                      (ddPIIe(0) * iFgTCAFinTM.dot(fiberdat.diFrdlambrM[gp]) +
                          dPIe(0) * FrTFinAFinTM.dot(fiberdat.diFrdlambrM[gp])),
      Av, 0.0);
}

/*---------------------------------------------------------------------*
 | return names of visualization data (public)                         |
 *---------------------------------------------------------------------*/
void Mat::Elastic::RemodelFiber::vis_names(std::map<std::string, int>& names, unsigned int p)
{
  std::string inelastic_defgrd = "lambda_r";
  std::string result_inelastic_defgrad;
  for (unsigned int k = 0; k < potsumfiber_.size(); ++k)
  {
    std::stringstream sstm;
    sstm << inelastic_defgrd << "_" << p << "_" << k;
    result_inelastic_defgrad = sstm.str();

    names[result_inelastic_defgrad] = 1;
  }


  std::string fiber_cauchy_stress = "fiber_cauchy_stress";
  std::string result_fiber_cauchy_stress;
  for (unsigned int k = 0; k < potsumfiber_.size(); ++k)
  {
    std::stringstream sstm;
    sstm << fiber_cauchy_stress << "_" << p << "_" << k;
    result_fiber_cauchy_stress = sstm.str();

    names[result_fiber_cauchy_stress] = 1;
  }


  std::string cur_rho_col = "cur_rho_col";
  std::string result_cur_rho_col;
  for (unsigned int k = 0; k < potsumfiber_.size(); ++k)
  {
    std::stringstream sstm;
    sstm << cur_rho_col << "_" << p << "_" << k;
    result_cur_rho_col = sstm.str();

    names[result_cur_rho_col] = 1;
  }
}  // vis_names()


/*---------------------------------------------------------------------*
 | return visualization data (public)                                  |
 *---------------------------------------------------------------------*/
bool Mat::Elastic::RemodelFiber::vis_data(
    const std::string& name, std::vector<double>& data, int numgp, int eleID)
{
  if ((name == "lambda_r_0_0") || (name == "lambda_r_1_0"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : potsumfiber_[0]->last_lambr)
    {
      data[0] += gp;
    }
    data[0] = data[0] / potsumfiber_[0]->last_lambr.size();

    return true;
  }
  if ((name == "lambda_r_0_1") || (name == "lambda_r_1_1"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : potsumfiber_[1]->last_lambr)
    {
      data[0] += gp;
    }
    data[0] = data[0] / potsumfiber_[1]->last_lambr.size();

    return true;
  }
  if ((name == "lambda_r_0_2") || (name == "lambda_r_1_2"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : potsumfiber_[2]->last_lambr)
    {
      data[0] += gp;
    }
    data[0] = data[0] / potsumfiber_[2]->last_lambr.size();

    return true;
  }
  if ((name == "lambda_r_0_3") || (name == "lambda_r_1_3"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : potsumfiber_[3]->last_lambr)
    {
      data[0] += gp;
    }
    data[0] = data[0] / potsumfiber_[3]->last_lambr.size();

    return true;
  }


  if ((name == "fiber_cauchy_stress_0_0") || (name == "fiber_cauchy_stress_1_0"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : cauchystress_[0])
    {
      data[0] += gp;
    }
    data[0] = data[0] / cauchystress_[0].size();

    return true;
  }
  if ((name == "fiber_cauchy_stress_0_1") || (name == "fiber_cauchy_stress_1_1"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : cauchystress_[1])
    {
      data[0] += gp;
    }
    data[0] = data[0] / cauchystress_[1].size();

    return true;
  }
  if ((name == "fiber_cauchy_stress_0_2") || (name == "fiber_cauchy_stress_1_2"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : cauchystress_[2])
    {
      data[0] += gp;
    }
    data[0] = data[0] / cauchystress_[2].size();

    return true;
  }
  if ((name == "fiber_cauchy_stress_0_3") || (name == "fiber_cauchy_stress_1_3"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double gp : cauchystress_[3])
    {
      data[0] += gp;
    }
    data[0] = data[0] / cauchystress_[3].size();

    return true;
  }


  if ((name == "cur_rho_col_0_0") || (name == "cur_rho_col_1_0"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double i : potsumfiber_[0]->cur_rho)
    {
      data[0] += i;
    }
    data[0] = data[0] / potsumfiber_[0]->cur_rho.size();


    return true;
  }
  if ((name == "cur_rho_col_0_1") || (name == "cur_rho_col_1_1"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double i : potsumfiber_[1]->cur_rho)
    {
      data[0] += i;
    }
    data[0] = data[0] / potsumfiber_[1]->cur_rho.size();


    return true;
  }
  if ((name == "cur_rho_col_0_2") || (name == "cur_rho_col_1_2"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double i : potsumfiber_[2]->cur_rho)
    {
      data[0] += i;
    }
    data[0] = data[0] / potsumfiber_[2]->cur_rho.size();


    return true;
  }
  if ((name == "cur_rho_col_0_3") || (name == "cur_rho_col_1_3"))
  {
    if (data.size() != 1) FOUR_C_THROW("size mismatch");
    for (double i : potsumfiber_[3]->cur_rho)
    {
      data[0] += i;
    }
    data[0] = data[0] / potsumfiber_[3]->cur_rho.size();


    return true;
  }

  FOUR_C_THROW("The output is only implemented for four different fiber directions!!!");
  return false;
}  // vis_data()

FOUR_C_NAMESPACE_CLOSE
