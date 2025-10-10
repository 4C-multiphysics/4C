// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_couptransverselyisotropic.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_service.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

#include <Teuchos_ParameterList.hpp>

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupTransverselyIsotropic::CoupTransverselyIsotropic(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : ParameterAniso(matdata),
      alpha_(matdata.parameters.get<double>("ALPHA")),
      beta_(matdata.parameters.get<double>("BETA")),
      gamma_(matdata.parameters.get<double>("GAMMA")),
      angle_(matdata.parameters.get<double>("ANGLE")),
      fiber_gid_(matdata.parameters.get<int>("FIBER")),
      init_(matdata.parameters.get<int>("INIT"))
{ /* empty */
}

void Mat::Elastic::PAR::CoupTransverselyIsotropic::print() const
{
  Core::IO::cout << "--- Material parameters of CoupTransverselyIsotropic\n";
  Core::IO::cout << "ALPHA           = " << alpha_ << Core::IO::endl;
  Core::IO::cout << "BETA            = " << beta_ << Core::IO::endl;
  Core::IO::cout << "GAMMA           = " << gamma_ << Core::IO::endl;
  Core::IO::cout << "ANGLE           = " << angle_ << Core::IO::endl;
  Core::IO::cout << "GLOBAL FIBER ID = " << fiber_gid_ << Core::IO::endl;
  Core::IO::cout << "INIT            = " << init_ << Core::IO::endl;
  Core::IO::cout << "----------------------------------------------------\n";
}

Mat::Elastic::CoupTransverselyIsotropic::CoupTransverselyIsotropic(my_params* params)
    : params_(params)
{ /* empty */
}

void Mat::Elastic::CoupTransverselyIsotropic::setup(int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  switch (params_->init_)
  {
    // path if fibers aren't given in input file
    case 0:
    {
      // fibers aligned in YZ-plane with gamma around Z in global cartesian cosy
      set_fiber_vecs(-1.0,
          Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>),
          Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>));

      break;
    }
    // path if fibers are given in input file
    case 1:
    {
      std::ostringstream ss;
      ss << params_->fiber_gid_;
      std::string fibername = "FIBER" + ss.str();  // FIBER Name
      // CIR-AXI-RAD nomenclature
      if (coord_system.has_value())
      {
        // Read in of data
        Core::LinAlg::Tensor<double, 3, 3> locsys{};
        read_rad_axi_cir(*coord_system, locsys);
        // final setup of fiber data
        set_fiber_vecs(0.0, locsys,
            Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>));
      }
      // FIBERi nomenclature
      else if (fibers.element_fibers.size() >= static_cast<unsigned>(params_->fiber_gid_))
      {
        a_ = fibers.element_fibers[params_->fiber_gid_ - 1];  // zero-based index
        params_->structural_tensor_strategy()->setup_structural_tensor(a_, aa_);
      }
      // error path
      else
      {
        FOUR_C_THROW("Reading of element local cosy for anisotropic materials failed");
      }

      break;
    }
    default:
    {
      FOUR_C_THROW("INIT mode not implemented");
    }
  }
}

void Mat::Elastic::CoupTransverselyIsotropic::pack_summand(
    Core::Communication::PackBuffer& data) const
{
  add_to_pack(data, a_);
  add_to_pack(data, aa_);
}

void Mat::Elastic::CoupTransverselyIsotropic::unpack_summand(
    Core::Communication::UnpackBuffer& buffer)
{
  extract_from_pack(buffer, a_);
  extract_from_pack(buffer, aa_);
}

void Mat::Elastic::CoupTransverselyIsotropic::get_fiber_vecs(
    std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs) const
{
  fibervecs.push_back(a_);
}

void Mat::Elastic::CoupTransverselyIsotropic::set_fiber_vecs(const double newangle,
    const Core::LinAlg::Tensor<double, 3, 3>& locsys,
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd)
{
  if ((params_->angle_ < -90) || (params_->angle_ > 90))
    FOUR_C_THROW("Fiber angle not in [-90,90]! Given angle = {}", params_->angle_);
  // convert
  const double angle = (params_->angle_ * std::numbers::pi) / 180.;

  Core::LinAlg::Tensor<double, 3> ca;
  for (int i = 0; i < 3; ++i)
  {
    // a = cos gamma e3 + sin gamma e2
    ca(i) = std::cos(angle) * locsys(i, 2) + std::sin(angle) * locsys(i, 1);
  }
  // pull back in reference configuration
  Core::LinAlg::Tensor<double, 3> A_0;
  Core::LinAlg::Matrix<3, 3> idefgrd(Core::LinAlg::Initialization::zero);
  idefgrd.invert(Core::LinAlg::make_matrix_view(defgrd));

  A_0 = Core::LinAlg::inv(defgrd) * ca;
  a_ = 1. / Core::LinAlg::norm2(A_0) * A_0;

  params_->structural_tensor_strategy()->setup_structural_tensor(a_, aa_);
}

void Mat::Elastic::CoupTransverselyIsotropic::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // convert Green-Lagrange strain to right Cauchy-Green Tensor
  // C_{AB} = 2 * E_{AB} + I_{AB} [ REMARK: strain-like 6-Voigt vector ]
  Core::LinAlg::SymmetricTensor<double, 3, 3> C =
      2 * glstrain + Core::LinAlg::TensorGenerators::identity<double, 3, 3>;

  reset_invariants(C);

  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;

  psi += (alpha + 0.5 * beta * std::log(prinv(2)) + gamma * (i4_ - 1.0)) * (i4_ - 1.0) -
         0.5 * alpha * (i5_ - 1.0);
}

void Mat::Elastic::CoupTransverselyIsotropic::add_stress_aniso_principal(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
    const int gp, const int eleGID)
{
  // direct return if an error occurred
  if (reset_invariants(rcg, &params)) return;

  // switch to stress notation
  Core::LinAlg::SymmetricTensor<double, 3, 3> rcg_inv_s;
  update_second_piola_kirchhoff_stress(stress, rcg, rcg_inv_s);

  update_elasticity_tensor(cmat, rcg_inv_s);
}

void Mat::Elastic::CoupTransverselyIsotropic::update_elasticity_tensor(
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg_inv_s) const
{
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;

  const std::array<double, 6> identity = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};

  // (0) contribution
  {
    const double delta = 8.0 * gamma;
    cmat += delta * Core::LinAlg::dyadic(aa_, aa_);
  }

  // (1) contribution
  {
    const double delta = 2.0 * beta;
    cmat += delta * (Core::LinAlg::dyadic(rcg_inv_s, aa_) + Core::LinAlg::dyadic(aa_, rcg_inv_s));
  }

  Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);
  using vmap = Core::LinAlg::Voigt::IndexMappings;
  // (2) contribution
  {
    const double delta = -alpha;
    for (unsigned a = 0; a < 6; ++a)
    {
      for (unsigned b = 0; b < 6; ++b)
      {
        const unsigned i = vmap::voigt_6x6_to_four_tensor_index(a, b, 0);
        const unsigned j = vmap::voigt_6x6_to_four_tensor_index(a, b, 1);
        const unsigned k = vmap::voigt_6x6_to_four_tensor_index(a, b, 2);
        const unsigned l = vmap::voigt_6x6_to_four_tensor_index(a, b, 3);

        cmat_view(a, b) +=
            delta * (a_(i) * a_(l) * identity[vmap::symmetric_tensor_to_voigt6_index(j, k)] +
                        a_(i) * a_(k) * identity[vmap::symmetric_tensor_to_voigt6_index(j, l)] +
                        a_(k) * a_(j) * identity[vmap::symmetric_tensor_to_voigt6_index(i, l)] +
                        a_(l) * a_(j) * identity[vmap::symmetric_tensor_to_voigt6_index(i, k)]);
      }
    }
  }

  // (3) contribution
  {
    const double delta = -beta * (i4_ - 1.0);
    for (unsigned a = 0; a < 6; ++a)
    {
      for (unsigned b = 0; b < 6; ++b)
      {
        const unsigned i = vmap::voigt_6x6_to_four_tensor_index(a, b, 0);
        const unsigned j = vmap::voigt_6x6_to_four_tensor_index(a, b, 1);
        const unsigned k = vmap::voigt_6x6_to_four_tensor_index(a, b, 2);
        const unsigned l = vmap::voigt_6x6_to_four_tensor_index(a, b, 3);

        cmat_view(a, b) +=
            delta * (rcg_inv_s(i, k) * rcg_inv_s(j, l) + rcg_inv_s(i, l) * rcg_inv_s(j, k));
      }
    }
  }
}

int Mat::Elastic::CoupTransverselyIsotropic::reset_invariants(
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, const Teuchos::ParameterList* params)
{
  // calculate the square root of the third invariant alias the determinant
  // of the deformation gradient
  const double I3 = Core::LinAlg::det(rcg);
  if (I3 < 0.0)
  {
    std::stringstream msg;
    msg << __LINE__ << " -- " << __PRETTY_FUNCTION__ << "I3 is negative!";
    error_handling(params, msg);
    return -1;
  }

  // jacobian determinant
  j_ = std::sqrt(I3);

  // calculate pseudo invariant I4 ( strain measure in fiber direction )
  i4_ = Core::LinAlg::ddot(aa_, rcg);

  // calculate pseudo invariant I5 ( quad. strain measure in fiber direction )
  Core::LinAlg::Matrix<6, 1> rcg_quad(Core::LinAlg::Initialization::uninitialized);
  Core::LinAlg::Voigt::Stresses::power_of_symmetric_tensor(
      2, Core::LinAlg::make_stress_like_voigt_view(rcg), rcg_quad);
  i5_ = aa_(0, 0) * (rcg_quad(0)) + aa_(1, 1) * (rcg_quad(1)) + aa_(2, 2) * (rcg_quad(2)) +
        2 * aa_(0, 1) * (rcg_quad(3)) + 2 * aa_(1, 2) * (rcg_quad(4)) +
        2 * aa_(0, 2) * (rcg_quad(5));

  return 0;
}

void Mat::Elastic::CoupTransverselyIsotropic::update_second_piola_kirchhoff_stress(
    Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg_s,
    Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg_inv_s) const
{
  const double alpha = params_->alpha_;
  const double beta = params_->beta_;
  const double gamma = params_->gamma_;

  // compute inverse right Cauchy Green tensor
  rcg_inv_s = Core::LinAlg::inv(rcg_s);

  // (0) contribution
  {
    const double fac = beta * (i4_ - 1.0);
    stress += fac * rcg_inv_s;
  }

  // (1) contribution
  {
    const double fac = 2.0 * (alpha + beta * std::log(j_) + 2.0 * gamma * (i4_ - 1.0));
    stress += fac * aa_;
  }

  // (2) contribution
  {
    const double fac = -alpha;
    stress += fac * Core::LinAlg::assume_symmetry(Core::LinAlg::dyadic(rcg_s * a_, a_) +
                                                  Core::LinAlg::dyadic(a_, rcg_s * a_));
  }
}

void Mat::Elastic::CoupTransverselyIsotropic::error_handling(
    const Teuchos::ParameterList* params, std::stringstream& msg) const
{
  if (params and params->isParameter("interface"))
  {
    std::shared_ptr<Core::Elements::ParamsInterface> interface_ptr = nullptr;
    interface_ptr = params->get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface");
    std::shared_ptr<Solid::Elements::ParamsInterface> solid_params =
        std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(interface_ptr);

    if (solid_params->is_tolerate_errors())
    {
      solid_params->set_ele_eval_error_flag(Solid::Elements::ele_error_material_failed);
      return;
    }
  }

  FOUR_C_THROW("Uncaught error detected:\n{}", msg.str());
}
FOUR_C_NAMESPACE_CLOSE
