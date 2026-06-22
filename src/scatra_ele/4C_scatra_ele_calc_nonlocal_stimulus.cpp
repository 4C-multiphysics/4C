// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_nonlocal_stimulus.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_mat_list.hpp"
#include "4C_mat_mixture.hpp"
#include "4C_mat_scatra_nonlocal_stimulus.hpp"
#include "4C_mixture_constituent_remodelfiber_ssi.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::ScaTraEleCalcNonlocalStimulus(
    const int numdofpernode, const int numscal, const std::string& disname)
    : Discret::Elements::ScaTraEleCalc<distype, probdim>::ScaTraEleCalc(
          numdofpernode, numscal, disname),
      source_at_gp_(numscal, 0.0)
{
}

template <Core::FE::CellType distype, int probdim>
Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>*
Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::pair<std::string, int>>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcNonlocalStimulus<distype, probdim>>(
            new ScaTraEleCalcNonlocalStimulus<distype, probdim>(numdofpernode, numscal, disname));
      });

  return singleton_map[std::make_pair(disname, numdofpernode)].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::get_material_params(
    const Core::Elements::Element* ele, std::vector<double>& densn, std::vector<double>& densnp,
    std::vector<double>& densam, double& visc, const int iquad)
{
  std::shared_ptr<Core::Mat::Material> material = ele->material();

  my::reamanager_->clear(my::numscal_);

  if (material->material_type() == Core::Materials::m_matlist)
  {
    const std::shared_ptr<Mat::MatList> actmat = std::dynamic_pointer_cast<Mat::MatList>(material);
    if (actmat->num_mat() < my::numscal_) FOUR_C_THROW("Not enough materials in MatList.");

    for (int k = 0; k < my::numscal_; ++k)
    {
      int matid = actmat->mat_id(k);
      const std::shared_ptr<Core::Mat::Material> singlemat = actmat->material_by_id(matid);
      materials(singlemat, k, densn[k], densnp[k], densam[k], visc, iquad);
    }
  }
  else
  {
    materials(material, 0, densn[0], densnp[0], densam[0], visc, iquad);
  }
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::materials(
    const std::shared_ptr<const Core::Mat::Material> material, const int k, double& densn,
    double& densnp, double& densam, double& visc, const int iquad)
{
  switch (material->material_type())
  {
    case Core::Materials::m_scatra_nl_stimulus:
    {
      mat_scatra_nls(material, k, densn, densnp, densam, visc, iquad);
      break;
    }
    default:
    {
      my::materials(material, k, densn, densnp, densam, visc, iquad);
      break;
    }
  }
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::mat_scatra_nls(
    const std::shared_ptr<const Core::Mat::Material> material, const int k, double& densn,
    double& densnp, double& densam, double& visc, const int iquad)
{
  const auto actmat = std::dynamic_pointer_cast<const Mat::ScatraNonlocalStimulusMat>(material);

  my::diffmanager_->set_isotropic_diff(actmat->characteristic_length_sq(), k);

  my::reamanager_->set_rea_coeff(1.0, k);

  FOUR_C_ASSERT(
      constituent_[k] != nullptr, "constituent_[{}] is null. Was setup_calc() called?", k);

  if (iquad == -1)
    FOUR_C_THROW("Element center evaluation is not supported for ScaTraEleCalcNonlocalStimulus.");

  source_at_gp_[k] = constituent_[k]->evaluate_local_stimulus(iquad);
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::get_rhs_int(
    double& rhsint, const double densnp, const int k)
{
  my::get_rhs_int(rhsint, densnp, k);
  rhsint += source_at_gp_[k];
}

template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::setup_calc(
    Core::Elements::Element* ele, Core::FE::Discretization& discretization)
{
  const int result = my::setup_calc(ele, discretization);
  if (result == -1) return result;

  const std::shared_ptr<Mat::Mixture> structmat = get_struct_material(ele);

  constituent_.assign(my::numscal_, nullptr);

  const std::shared_ptr<Core::Mat::Material> scatra_material = ele->material(0);

  auto resolve = [&](const std::shared_ptr<const Core::Mat::Material>& mat)
      -> const Mixture::MixtureConstituentRemodelFiberSsi*
  {
    if (mat->material_type() != Core::Materials::m_scatra_nl_stimulus) return nullptr;

    const auto actmat = std::dynamic_pointer_cast<const Mat::ScatraNonlocalStimulusMat>(mat);
    const int struct_mat_id = actmat->structure_material_id();

    const auto* constituent = structmat->ssi_constituent_by_material_id(struct_mat_id);

    FOUR_C_ASSERT_ALWAYS(constituent != nullptr,
        "Constituent with material id {} is not a MixtureConstituentRemodelFiberSsi. "
        "ScaTraEleCalcNonlocalStimulus requires SSI RemodelFiber constituents.",
        struct_mat_id);

    FOUR_C_ASSERT_ALWAYS(constituent->is_nonlocal_stimulus_mode(),
        "Constituent with material id {} does not have NONLOCAL_STIMULUS_SCALAR_ID set. "
        "Please set NONLOCAL_STIMULUS_SCALAR_ID >= 0 in the constituent parameters.",
        struct_mat_id);

    return constituent;
  };

  if (scatra_material->material_type() == Core::Materials::m_matlist)
  {
    const auto actmat = std::dynamic_pointer_cast<const Mat::MatList>(scatra_material);
    for (int k = 0; k < my::numscal_; ++k)
      constituent_[k] = resolve(actmat->material_by_id(actmat->mat_id(k)));
  }
  else
  {
    constituent_[0] = resolve(scatra_material);
  }

  return 0;
}

template <Core::FE::CellType distype, int probdim>
std::shared_ptr<Mat::Mixture>
Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::get_struct_material(
    const Core::Elements::Element* ele)
{
  if (ele->num_material() <= 1)
    FOUR_C_THROW(
        "ScaTra NLS coupling requires a second structural material, but this is not defined for "
        "the element");

  auto mixture = std::dynamic_pointer_cast<Mat::Mixture>(ele->material(1));
  if (!mixture)
    FOUR_C_THROW(
        "ScaTra NLS coupling expects material(1) to be a Mat::Mixture structural material, but "
        "material(1) is not Mat::Mixture.");

  return mixture;
}


template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::evaluate_action_od(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const ScaTra::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1,
    Core::LinAlg::SerialDenseMatrix& elemat2, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseVector& elevec2, Core::LinAlg::SerialDenseVector& elevec3)
{
  // base class call
  const int result = my::evaluate_action_od(
      ele, params, discretization, action, la, elemat1, elemat2, elevec1, elevec2, elevec3);

  // nonlocal stimulus specific contribution to the off-diagonal block
  if (action == ScaTra::Action::calc_scatra_mono_odblock_mesh)
    sysmat_od_mesh_nls(elemat1, my::nsd_);

  return result;
}

template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalcNonlocalStimulus<distype, probdim>::sysmat_od_mesh_nls(
    Core::LinAlg::SerialDenseMatrix& emat, const int ndofpernodemesh)
{
  if constexpr (probdim == 3 && Core::FE::dim<distype> == 3)
  {
    Core::FE::IntPointsAndWeights<Core::FE::dim<distype>> intpoints(
        ScaTra::DisTypeToOptGaussRule<distype>::rule);

    for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
    {
      const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);
      my::set_internal_variables_for_mat_and_rhs();

      // Reference coordinates: xyze_ - edispnp_
      Core::LinAlg::Matrix<my::nsd_, my::nen_> nodal_reference_coords(my::xyze_);
      nodal_reference_coords.update(-1.0, my::edispnp_, 1.0);

      // Reference Jacobian J = dX/dxi and its inverse
      Core::LinAlg::Matrix<my::nsd_, my::nsd_> xjm0;
      xjm0.multiply_nt(nodal_reference_coords, my::deriv_);
      Core::LinAlg::Matrix<my::nsd_, my::nsd_> xji0(Core::LinAlg::Initialization::zero);
      xji0.invert(xjm0);

      // Reference shape derivatives dN_i/dX_r: chain rule dN/dxi * dxi/dX
      Core::LinAlg::Matrix<my::nen_, my::nsd_> dN_dX;
      dN_dX.multiply_tn(my::deriv_, xji0);

      // Deformation gradient F = dx/dX
      Core::LinAlg::Matrix<my::nsd_, my::nsd_> defgrd;
      defgrd.multiply(my::xyze_, dN_dX);

      // loop over all scalars
      for (int k = 0; k < my::numscal_; ++k)
      {
        if (constituent_[k] == nullptr) continue;

        const double d_sigma_d_lf2 = constituent_[k]->evaluate_d_cauchy_stress_d_lambda_f_sq(iquad);

        if (d_sigma_d_lf2 == 0.0) continue;

        // Eulerian structural tensor: b_alpha = F * A_alpha * F^T
        const Core::LinAlg::Matrix<my::nsd_, my::nsd_> A_alpha_mat = Core::LinAlg::make_matrix(
            Core::LinAlg::get_full(constituent_[k]->get_structural_tensor(iquad)));
        Core::LinAlg::Matrix<my::nsd_, my::nsd_> FA;
        FA.multiply(defgrd, A_alpha_mat);
        Core::LinAlg::Matrix<my::nsd_, my::nsd_> b_alpha;
        b_alpha.multiply_nt(FA, defgrd);

        // K_psi_u^{vi*numdof+k, ui*ndofmesh+d}
        //   += -fac * d_sigma_d_lf2 * funct_(vi) * 2 * (b_alpha_{d,c} * derxy_(c,ui))

        // loop over the scatra nodes with usually one numdofpernode_
        for (int vi = 0; vi < static_cast<int>(my::nen_); ++vi)
        {
          const int fvi = vi * my::numdofpernode_ + k;
          const double val = -fac * d_sigma_d_lf2 * my::funct_(vi);

          // loop over the mesh nodes
          for (int ui = 0; ui < static_cast<int>(my::nen_); ++ui)
          {
            // loop over the spatial dimensions of the mesh node
            for (unsigned int d = 0; d < my::nsd_; ++d)
            {
              double b_dot_derxy = 0.0;
              // compute scalar product
              for (unsigned int c = 0; c < my::nsd_; ++c)
                b_dot_derxy += b_alpha(d, c) * my::derxy_(c, ui);

              emat(fvi, ui * ndofpernodemesh + d) += val * 2.0 * b_dot_derxy;
            }
          }
        }
      }
    }
  }
  else
  {
    FOUR_C_THROW("sysmat_od_mesh_nls is only implemented for 3D elements.");
  }
}

// template classes

// 1D elements
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::line2, 1>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::line2, 2>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::line2, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::line3, 1>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::tri3, 2>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::tri3, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::tri6, 2>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::quad4, 2>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::quad4, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::quad9, 2>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::nurbs9, 2>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::hex8, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::hex27, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::tet4, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::tet10, 3>;
template class Discret::Elements::ScaTraEleCalcNonlocalStimulus<Core::FE::CellType::pyramid5, 3>;

FOUR_C_NAMESPACE_CLOSE
