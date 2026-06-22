// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ADAPTER_COUPLING_PORO_MORTAR_HPP
#define FOUR_C_ADAPTER_COUPLING_PORO_MORTAR_HPP

/*---------------------------------------------------------------------*
 | headers                                                  ager 10/15 |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_adapter_coupling_nonlin_mortar.hpp"
#include "4C_contact_lagrange_strategy_poro.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN
/*---------------------------------------------------------------------*
 | forward declarations                                     ager 10/15 |
 *---------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace CONTACT
{
  class Interface;
}

namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Adapter
{
  class CouplingPoroMortar : public CouplingNonLinMortar
  {
   public:
    /*!
    \brief Empty constructor

    */
    CouplingPoroMortar(Global::Problem& problem, int spatial_dimension,
        Teuchos::ParameterList mortar_coupling_params,
        Teuchos::ParameterList contact_dynamic_params,
        Core::FE::ShapeFunctionType shape_function_type);


    virtual void evaluate_poro_mt(std::shared_ptr<Core::LinAlg::Vector<double>> fvel,
        std::shared_ptr<Core::LinAlg::Vector<double>> svel,
        std::shared_ptr<Core::LinAlg::Vector<double>> fpres,
        std::shared_ptr<Core::LinAlg::Vector<double>> sdisp,
        const std::shared_ptr<Core::FE::Discretization> sdis,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& f,
        std::shared_ptr<Core::LinAlg::SparseMatrix>& k_fs,
        std::shared_ptr<Core::LinAlg::Vector<double>>& frhs, Coupling::Adapter::Coupling& coupfs,
        std::shared_ptr<const Core::LinAlg::Map> fdofrowmap);

    void update_poro_mt();

    void recover_fluid_lm_poro_mt(std::shared_ptr<Core::LinAlg::Vector<double>> disi,
        std::shared_ptr<Core::LinAlg::Vector<double>> veli);  // h.Willmann

    // return the used poro lagrange strategy
    std::shared_ptr<CONTACT::LagrangeStrategyPoro> get_poro_strategy()
    {
      if (porolagstrategy_ == nullptr) FOUR_C_THROW("GetPoroStrategy(): No strategy set!");
      return porolagstrategy_;
    };

   protected:
    /*!
    \brief Read Mortar Condition

    */
    void read_mortar_condition(std::shared_ptr<Core::FE::Discretization> target_dis,
        std::shared_ptr<Core::FE::Discretization> source_dis, std::vector<int> coupleddof,
        const std::string& couplingcond, Teuchos::ParameterList& input,
        std::map<int, Core::Nodes::Node*>& target_global_nodes,
        std::map<int, Core::Nodes::Node*>& source_global_nodes,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& target_elements,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& source_elements) override;

    /*!
    \brief Add Mortar Elements

    */
    void add_mortar_elements(std::shared_ptr<Core::FE::Discretization> target_dis,
        std::shared_ptr<Core::FE::Discretization> source_dis, Teuchos::ParameterList& input,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& target_elements,
        std::map<int, std::shared_ptr<Core::Elements::Element>>& source_elements,
        std::shared_ptr<CONTACT::Interface>& interface, int numcoupleddof) override;

    /*!
    \brief complete interface, store as internal variable
           store maps as internal variable and do parallel redist.

    */
    void complete_interface(std::shared_ptr<Core::FE::Discretization> target_dis,
        std::shared_ptr<CONTACT::Interface>& interface) override;

    /*!
    \brief create strategy object if required

    */
    void create_strategy(std::shared_ptr<Core::FE::Discretization> target_dis,
        std::shared_ptr<Core::FE::Discretization> source_dis, Teuchos::ParameterList& input,
        int numcoupleddof) override;

   private:
    // poro lagrange strategy
    std::shared_ptr<CONTACT::LagrangeStrategyPoro> porolagstrategy_;

    // firstinit
    bool firstinit_;

    int source_type_;  // 1 poro, 0 struct, -1 default
    int target_type_;  // 1 poro, 0 struct, -1 default
  };
}  // namespace Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
