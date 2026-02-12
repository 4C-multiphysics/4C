// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_REDUCED_LUNG_DISCRETIZATION_HELPERS_HPP
#define FOUR_C_REDUCED_LUNG_DISCRETIZATION_HELPERS_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_reduced_lung_input.hpp"

#include <map>
#include <memory>
#include <span>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}

namespace Core::Rebalance
{
  struct RebalanceParameters;
}

namespace Discret
{
  namespace Elements
  {
    /**
     * @brief Element type wrapper for the reduced lung dummy line element.
     *
     * Provides the minimal registration and null-space information required by the
     * base class so ReducedLungLine can live inside a 4C discretization.
     */
    class ReducedLungLineType : public Core::Elements::ElementType
    {
     public:
      std::string name() const override { return "ReducedLungLineType"; }

      static ReducedLungLineType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, std::span<const double> x0, const int numdof) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static ReducedLungLineType instance_;
    };

    /**
     * @brief Dummy line element used to carry reduced lung topology in a discretization.
     *
     * This element exists purely so the reduced lung framework can leverage 4C's
     * discretization infrastructure (distribution, rebalancing, and output). It does
     * not represent a physical FE formulation. Consequently, several virtual methods
     * are implemented only to satisfy the base-class interface and are intentionally
     * minimal or throw if called (e.g., evaluate_neumann()).
     */
    class ReducedLungLine : public Core::Elements::Element
    {
     public:
      ReducedLungLine(int id, int owner);
      ReducedLungLine(const ReducedLungLine& old);

      Core::Elements::Element* clone() const override;

      Core::FE::CellType shape() const override;

      int num_line() const override;
      int num_surface() const override { return -1; }
      int num_volume() const override { return -1; }

      std::vector<std::shared_ptr<Core::Elements::Element>> lines() override;

      int unique_par_object_id() const override
      {
        return ReducedLungLineType::instance().unique_par_object_id();
      }

      int num_dof_per_node(const Core::Nodes::Node& node) const override { return 1; }

      int num_dof_per_element() const override { return 0; }

      /**
       * @brief Required by the Element interface but unused for reduced lung.
       *
       * Throws when invoked because no Neumann evaluation is defined for the dummy element.
       */
      int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
          const Core::Conditions::Condition& condition, std::vector<int>& lm,
          Core::LinAlg::SerialDenseVector& elevec1,
          Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;

      void print(std::ostream& os) const override;

      ReducedLungLineType& element_type() const override { return ReducedLungLineType::instance(); }

      /**
       * @brief Required by the Element interface but unused for reduced lung.
       *
       * Reduced lung elements are built programmatically from topology input, so
       * this method performs no parsing and returns true.
       */
      bool read_element(const std::string& eletype, const std::string& distype,
          const Core::IO::InputParameterContainer& container) override;
    };
  }  // namespace Elements
}  // namespace Discret

namespace ReducedLung
{
  void build_discretization_from_topology(Core::FE::Discretization& discretization,
      const ReducedLungParameters::LungTree::Topology& topology,
      const Core::Rebalance::RebalanceParameters& rebalance_parameters);
}

FOUR_C_NAMESPACE_CLOSE

#endif
