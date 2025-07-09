// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_NITSCHE_UTILS_HPP
#define FOUR_C_CONTACT_NITSCHE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_contact_utils.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_mortar_element.hpp"

#include <memory>
#include <unordered_map>

FOUR_C_NAMESPACE_OPEN

namespace Mortar
{
  class ElementNitscheContainer
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~ElementNitscheContainer() = default;

    ElementNitscheContainer() = default;

    virtual void clear() = 0;

    virtual void assemble_rhs(Mortar::Element* mele, CONTACT::VecBlockType row,
        std::shared_ptr<Core::LinAlg::FEVector<double>> fc) const = 0;

    virtual void assemble_matrix(Mortar::Element* mele, CONTACT::MatBlockType block,
        std::shared_ptr<Core::LinAlg::SparseMatrix> kc) const = 0;

    virtual double* rhs(int dof) = 0;
    virtual double* rhs() = 0;
    virtual double* k(int col) = 0;
    virtual double* k(int col, int dof) = 0;

    virtual double* rhs_t(int dof) = 0;
    virtual double* rhs_t() = 0;
    virtual double* ktt(int col) = 0;
    virtual double* ktd(int col) = 0;
    virtual double* kdt(int col) = 0;

    virtual double* rhs_p(int dof) = 0;
    virtual double* kpp(int col) = 0;
    virtual double* kpd(int col) = 0;
    virtual double* kdp(int col) = 0;

    virtual double* rhs_s(int dof) = 0;
    virtual double* kss(int col) = 0;
    virtual double* ksd(int col) = 0;
    virtual double* kds(int col) = 0;

    virtual double* rhs_e(int dof) = 0;
    virtual double* kee(int col) = 0;
    virtual double* ked(int col) = 0;
    virtual double* ked(int col, int dof) = 0;
    virtual double* kde(int col) = 0;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataTsi
  {
   public:
    void clear()
    {
      rhs_t_.clear();
      k_tt_.clear();
      k_td_.clear();
      k_dt_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes(parent_distype) * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_thermo_dof = Core::FE::num_nodes(parent_distype);

    Core::LinAlg::Matrix<num_parent_thermo_dof, 1> rhs_t_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_thermo_dof, 1>> k_tt_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_thermo_dof, 1>> k_td_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_dt_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataPoro
  {
   public:
    void clear()
    {
      rhs_p_.clear();
      k_pp_.clear();
      k_pd_.clear();
      k_dp_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes(parent_distype) * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_pf_dof =
        Core::FE::num_nodes(parent_distype) * (Core::FE::dim<parent_distype> + 1);

    Core::LinAlg::Matrix<num_parent_pf_dof, 1> rhs_p_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_pf_dof, 1>> k_pp_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_pf_dof, 1>> k_pd_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_dp_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataSsi
  {
   public:
    void clear()
    {
      rhs_s_.clear();
      k_ss_.clear();
      k_sd_.clear();
      k_ds_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes(parent_distype) * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_scatra_dof = Core::FE::num_nodes(parent_distype);

    Core::LinAlg::Matrix<num_parent_scatra_dof, 1> rhs_s_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_scatra_dof, 1>> k_ss_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_scatra_dof, 1>> k_sd_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_ds_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheDataSsiElch
  {
   public:
    void clear()
    {
      rhs_e_.clear();
      k_ee_.clear();
      k_ed_.clear();
      k_de_.clear();
    }

    static constexpr int num_parent_disp_dof =
        Core::FE::num_nodes(parent_distype) * Core::FE::dim<parent_distype>;
    static constexpr int num_parent_elch_dof = Core::FE::num_nodes(parent_distype) * 2;

    Core::LinAlg::Matrix<num_parent_elch_dof, 1> rhs_e_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_elch_dof, 1>> k_ee_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_elch_dof, 1>> k_ed_;
    std::unordered_map<int, Core::LinAlg::Matrix<num_parent_disp_dof, 1>> k_de_;
  };

  template <Core::FE::CellType parent_distype>
  class ElementNitscheData : public ElementNitscheContainer
  {
    using VectorType =
        Core::LinAlg::Matrix<Core::FE::num_nodes(parent_distype) * Core::FE::dim<parent_distype>,
            1>;

   public:
    const VectorType& rhs_vec() { return rhs_; }
    double* rhs(int dof) override { return &rhs_(dof); }
    double* rhs() override { return rhs_.data(); }
    double* k(int col) override { return k_[col].data(); }
    double* k(int col, int dof) override { return &k_[col](dof); }

    double* rhs_t(int dof) override { return &tsi_data_.rhs_t_(dof); }
    double* rhs_t() override { return tsi_data_.rhs_t_.data(); }
    double* ktt(int col) override { return tsi_data_.k_tt_[col].data(); }
    double* ktd(int col) override { return tsi_data_.k_td_[col].data(); }
    double* kdt(int col) override { return tsi_data_.k_dt_[col].data(); }

    double* rhs_p(int dof) override { return &poro_data_.rhs_p_(dof); }
    double* kpp(int col) override { return poro_data_.k_pp_[col].data(); }
    double* kpd(int col) override { return poro_data_.k_pd_[col].data(); }
    double* kdp(int col) override { return poro_data_.k_dp_[col].data(); }

    double* rhs_s(int dof) override { return &ssi_data_.rhs_s_(dof); }
    double* kss(int col) override { return ssi_data_.k_ss_[col].data(); }
    double* ksd(int col) override { return ssi_data_.k_sd_[col].data(); }
    double* kds(int col) override { return ssi_data_.k_ds_[col].data(); }

    double* rhs_e(int dof) override { return &ssi_elch_data_.rhs_e_(dof); }
    double* kee(int col) override { return ssi_elch_data_.k_ee_[col].data(); }
    double* ked(int col) override { return ssi_elch_data_.k_ed_[col].data(); }
    double* ked(int col, int dof) override { return &ssi_elch_data_.k_ed_[col](dof); }
    double* kde(int col) override { return ssi_elch_data_.k_de_[col].data(); }

    void assemble_rhs(Mortar::Element* mele, CONTACT::VecBlockType row,
        std::shared_ptr<Core::LinAlg::FEVector<double>> fc) const override;

    void assemble_matrix(Mortar::Element* mele, CONTACT::MatBlockType block,
        std::shared_ptr<Core::LinAlg::SparseMatrix> kc) const override;

    template <int num_dof_per_node>
    void assemble_rhs(Mortar::Element* mele,
        const Core::LinAlg::Matrix<Core::FE::num_nodes(parent_distype) * num_dof_per_node, 1>& rhs,
        std::vector<int>& dofs, std::shared_ptr<Core::LinAlg::FEVector<double>> fc) const;

    template <int num_dof_per_node>
    void assemble_matrix(Mortar::Element* mele,
        const std::unordered_map<int,
            Core::LinAlg::Matrix<Core::FE::num_nodes(parent_distype) * num_dof_per_node, 1>>& k,
        std::vector<int>& dofs, std::shared_ptr<Core::LinAlg::SparseMatrix> kc) const;


    void clear() override
    {
      rhs_.clear();
      k_.clear();
      tsi_data_.clear();
      poro_data_.clear();
      ssi_data_.clear();
      ssi_elch_data_.clear();
    }

   private:
    VectorType rhs_;
    std::unordered_map<int, VectorType> k_;
    Mortar::ElementNitscheDataTsi<parent_distype> tsi_data_;
    Mortar::ElementNitscheDataPoro<parent_distype> poro_data_;
    Mortar::ElementNitscheDataSsi<parent_distype> ssi_data_;
    Mortar::ElementNitscheDataSsiElch<parent_distype> ssi_elch_data_;
  };

}  // namespace Mortar

FOUR_C_NAMESPACE_CLOSE

#endif
