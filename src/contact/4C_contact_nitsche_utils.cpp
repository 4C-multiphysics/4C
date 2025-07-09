// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_utils.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType parent_distype>
template <int num_dof_per_node>
void Mortar::ElementNitscheData<parent_distype>::assemble_rhs(Mortar::Element* mele,
    const Core::LinAlg::Matrix<Core::FE::num_nodes(parent_distype) * num_dof_per_node, 1>& rhs,
    std::vector<int>& dofs, std::shared_ptr<Core::LinAlg::FEVector<double>> fc) const
{
  const int nen = Core::FE::num_nodes(parent_distype);

  if (num_dof_per_node * nen > dofs.size())
    FOUR_C_THROW("num_dof_per_node*nen>dofs.size() {} > {}", num_dof_per_node * nen, dofs.size());

  if (fc != nullptr)
  {
    for (int n = 0; n < nen; ++n)
      fc->sum_into_global_values(
          num_dof_per_node, &dofs.at(n * num_dof_per_node), &rhs.data()[n * num_dof_per_node]);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType parent_distype>
template <int num_dof_per_node>
void Mortar::ElementNitscheData<parent_distype>::assemble_matrix(Mortar::Element* mele,
    const std::unordered_map<int,
        Core::LinAlg::Matrix<Core::FE::num_nodes(parent_distype) * num_dof_per_node, 1>>& k,
    std::vector<int>& dofs, std::shared_ptr<Core::LinAlg::SparseMatrix> kc) const
{
  const int nen = Core::FE::num_nodes(parent_distype);

  if (kc != nullptr)
  {
    for (auto& p : k)
    {
      for (int n = 0; n < nen; ++n)
      {
        if (Core::LinAlg::Matrix<num_dof_per_node, 1>(
                &(p.second.data()[n * num_dof_per_node]), true)
                .norm_inf() < 1e-16)
          continue;
        for (int d = 0; d < num_dof_per_node; ++d)
          kc->fe_assemble(
              p.second(n * num_dof_per_node + d), dofs.at(n * num_dof_per_node + d), p.first);
      }
    }
  }
}


template <Core::FE::CellType parent_distype>
void Mortar::ElementNitscheData<parent_distype>::assemble_rhs(Mortar::Element* mele,
    CONTACT::VecBlockType row, std::shared_ptr<Core::LinAlg::FEVector<double>> fc) const
{
  switch (row)
  {
    case CONTACT::VecBlockType::displ:
      assemble_rhs<Core::FE::dim<parent_distype>>(mele, rhs_, mele->mo_data().parent_dof(), fc);
      break;
    case CONTACT::VecBlockType::temp:
      if (mele->mo_data().parent_temp_dof().size())
        assemble_rhs<1>(mele, tsi_data_.rhs_t_, mele->mo_data().parent_temp_dof(), fc);
      break;
    case CONTACT::VecBlockType::porofluid:
      if (mele->mo_data().parent_pf_dof().size())  // not if the parent is an impermeable element
        assemble_rhs<Core::FE::dim<parent_distype> + 1>(
            mele, poro_data_.rhs_p_, mele->mo_data().parent_pf_dof(), fc);
      break;
    case CONTACT::VecBlockType::scatra:
      if (mele->mo_data().parent_scalar_dof().size())
        assemble_rhs<1>(mele, ssi_data_.rhs_s_, mele->mo_data().parent_scalar_dof(), fc);
      break;
    case CONTACT::VecBlockType::elch:
      if (mele->mo_data().parent_scalar_dof().size())
        assemble_rhs<2>(mele, ssi_elch_data_.rhs_e_, mele->mo_data().parent_scalar_dof(), fc);
      break;
    default:
      FOUR_C_THROW("unknown row");
  }
}

template <Core::FE::CellType parent_distype>
void Mortar::ElementNitscheData<parent_distype>::assemble_matrix(Mortar::Element* mele,
    CONTACT::MatBlockType block, std::shared_ptr<Core::LinAlg::SparseMatrix> kc) const
{
  switch (block)
  {
    case CONTACT::MatBlockType::displ_displ:
      assemble_matrix<Core::FE::dim<parent_distype>>(mele, k_, mele->mo_data().parent_dof(), kc);
      break;
    case CONTACT::MatBlockType::displ_temp:
      assemble_matrix<Core::FE::dim<parent_distype>>(
          mele, tsi_data_.k_dt_, mele->mo_data().parent_dof(), kc);
      break;
    case CONTACT::MatBlockType::temp_displ:
      if (mele->mo_data().parent_temp_dof().size())
        assemble_matrix<1>(mele, tsi_data_.k_td_, mele->mo_data().parent_temp_dof(), kc);
      break;
    case CONTACT::MatBlockType::temp_temp:
      if (mele->mo_data().parent_temp_dof().size())
        assemble_matrix<1>(mele, tsi_data_.k_tt_, mele->mo_data().parent_temp_dof(), kc);
      break;
    case CONTACT::MatBlockType::displ_porofluid:
      assemble_matrix<Core::FE::dim<parent_distype>>(
          mele, poro_data_.k_dp_, mele->mo_data().parent_dof(), kc);
      break;
    case CONTACT::MatBlockType::porofluid_displ:
      if (mele->mo_data().parent_pf_dof().size())  // not if the parent is an impermeable element
        assemble_matrix<Core::FE::dim<parent_distype> + 1>(
            mele, poro_data_.k_pd_, mele->mo_data().parent_pf_dof(), kc);
      break;
    case CONTACT::MatBlockType::porofluid_porofluid:
      if (mele->mo_data().parent_pf_dof().size())  // not if the parent is an impermeable element
        assemble_matrix<Core::FE::dim<parent_distype> + 1>(
            mele, poro_data_.k_pp_, mele->mo_data().parent_pf_dof(), kc);
      break;
    case CONTACT::MatBlockType::displ_scatra:
      assemble_matrix<Core::FE::dim<parent_distype>>(
          mele, ssi_data_.k_ds_, mele->mo_data().parent_dof(), kc);
      break;
    case CONTACT::MatBlockType::scatra_displ:
      if (mele->mo_data().parent_scalar_dof().size())
        assemble_matrix<1>(mele, ssi_data_.k_sd_, mele->mo_data().parent_scalar_dof(), kc);
      break;
    case CONTACT::MatBlockType::scatra_scatra:
      if (mele->mo_data().parent_scalar_dof().size())
        assemble_matrix<1>(mele, ssi_data_.k_ss_, mele->mo_data().parent_scalar_dof(), kc);
      break;
    case CONTACT::MatBlockType::displ_elch:
      assemble_matrix<Core::FE::dim<parent_distype>>(
          mele, ssi_elch_data_.k_de_, mele->mo_data().parent_dof(), kc);
      break;
    case CONTACT::MatBlockType::elch_displ:
      if (mele->mo_data().parent_scalar_dof().size())
        assemble_matrix<2>(mele, ssi_elch_data_.k_ed_, mele->mo_data().parent_scalar_dof(), kc);
      break;
    case CONTACT::MatBlockType::elch_elch:
      if (mele->mo_data().parent_scalar_dof().size())
        assemble_matrix<2>(mele, ssi_elch_data_.k_ee_, mele->mo_data().parent_scalar_dof(), kc);
      break;
    default:
      FOUR_C_THROW("unknown matrix block");
      break;
  }
}

template class Mortar::ElementNitscheData<Core::FE::CellType::hex8>;
template class Mortar::ElementNitscheData<Core::FE::CellType::tet4>;
template class Mortar::ElementNitscheData<Core::FE::CellType::hex27>;
template class Mortar::ElementNitscheData<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
