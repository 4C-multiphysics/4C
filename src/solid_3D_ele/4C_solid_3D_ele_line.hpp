// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SOLID_3D_ELE_LINE_HPP
#define FOUR_C_SOLID_3D_ELE_LINE_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"

#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Discret::Elements
{
  template <unsigned dim>
  class SolidLineType : public Core::Elements::ElementType
  {
   public:
    [[nodiscard]] std::string name() const override
    {
      return "SolidLineType" + std::to_string(dim);
    }

    static SolidLineType<dim>& instance();

    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override {}

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
    {
      FOUR_C_THROW("Computation of nullspace is not possible for solid-line element.");
    }

   private:
    static SolidLineType<dim> instance_;
  };

  /*!
  \brief An element representing a line edge of solid element in dim dimensions
  */
  template <unsigned dim>
  class SolidLine : public Core::Elements::FaceElement
  {
   public:
    /*!
    \brief Standard Constructor

    \param id : A unique global id
    \param owner: Processor owning this line
    \param nnode: Number of nodes attached to this element
    \param nodeids: global ids of nodes attached to this element
    \param nodes: the discretizations map of nodes to build ptrs to nodes from
    \param parent: The parent shell element of this line
    \param lline: the local line number of this line w.r.t. the parent element
    */
    SolidLine(int id, int owner, int nnode, const int* nodeids, Core::Nodes::Node** nodes,
        Core::Elements::Element* parent, const int lline);

    [[nodiscard]] Core::Elements::Element* clone() const override;

    [[nodiscard]] inline int unique_par_object_id() const override
    {
      return SolidLineType<dim>::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override {};

    void unpack(Core::Communication::UnpackBuffer& buffer) override {};

    [[nodiscard]] Core::FE::CellType shape() const override;

    [[nodiscard]] inline int num_dof_per_node(const Core::Nodes::Node& node) const override
    {
      return dim;
    }

    [[nodiscard]] inline int num_dof_per_element() const override { return 0; }

    void print(std::ostream& os) const override;

    [[nodiscard]] Core::Elements::ElementType& element_type() const override
    {
      return SolidLineType<dim>::instance();
    }

    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override;
  };
}  // namespace Discret::Elements

FOUR_C_NAMESPACE_CLOSE

#endif
