// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_solid_ele_line.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_legacy_enum_definitions_element_actions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_solid_ele.hpp"
#include "4C_solid_ele_neumann_evaluator.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_ParameterList.hpp>

#include <functional>
#include <string>

FOUR_C_NAMESPACE_OPEN


namespace
{
  template <Core::FE::CellType celltype, unsigned dim>
  std::array<Core::LinAlg::Tensor<double, dim>, Core::FE::num_nodes(celltype)>
  evaluate_current_coordinates(const Core::Elements::Element& ele,
      const Core::FE::Discretization& discretization, const std::vector<int>& lm)
  {
    constexpr unsigned num_dofs = dim * Core::FE::num_nodes(celltype);
    std::array<double, num_dofs> mydisp =
        Core::FE::extract_values_as_array<num_dofs>(*discretization.get_state("displacement"), lm);

    std::array<Core::LinAlg::Tensor<double, dim>, Core::FE::num_nodes(celltype)>
        current_nodal_coordinates;

    for (int i = 0; i < Core::FE::num_nodes(celltype); ++i)
    {
      for (std::size_t d = 0; d < dim; ++d)
      {
        current_nodal_coordinates[i](d) = ele.nodes()[i]->x()[d] + mydisp[i * dim + d];
      }
    }

    return current_nodal_coordinates;
  }
}  // namespace

template <unsigned dim>
Discret::Elements::SolidLineType<dim> Discret::Elements::SolidLineType<dim>::instance_;

template <unsigned dim>
Discret::Elements::SolidLineType<dim>& Discret::Elements::SolidLineType<dim>::instance()
{
  return instance_;
}

template <unsigned dim>
std::shared_ptr<Core::Elements::Element> Discret::Elements::SolidLineType<dim>::create(
    const int id, const int owner)
{
  return nullptr;
}


template <unsigned dim>
Discret::Elements::SolidLine<dim>::SolidLine(const Discret::Elements::SolidLine<dim>& old) noexcept
    : Core::Elements::FaceElement(old), num_dof_per_node_(old.num_dof_per_node_)
{
}

template <unsigned dim>
Discret::Elements::SolidLine<dim>::SolidLine(int id, int owner, int nnode, const int* nodeids,
    Core::Nodes::Node** nodes, Core::Elements::Element* parent, const int lline)
    : Core::Elements::FaceElement(id, owner)
{
  set_node_ids(nnode, nodeids);
  build_nodal_pointers(nodes);
  set_parent_target_element(parent, lline);

  num_dof_per_node_ = parent_element()->num_dof_per_node(*SolidLine::nodes()[0]);
  // Safety check if all nodes have the same number of dofs!
  for (int nlid = 1; nlid < num_node(); ++nlid)
  {
    if (num_dof_per_node_ != parent_target_element()->num_dof_per_node(*SolidLine::nodes()[nlid]))
      FOUR_C_THROW("You need different NumDofPerNode for each node on this solid line? ({} != {})",
          num_dof_per_node_, parent_target_element()->num_dof_per_node(*SolidLine::nodes()[nlid]));
  }
}

template <unsigned dim>
Core::Elements::Element* Discret::Elements::SolidLine<dim>::clone() const
{
  auto* newelement = new Discret::Elements::SolidLine<dim>(*this);
  return newelement;
}

template <unsigned dim>
Core::FE::CellType Discret::Elements::SolidLine<dim>::shape() const
{
  return Core::FE::cell_type_switch(parent_element()->shape(),
      [&](auto celltype_t)
      {
        switch (num_node())
        {
          case 2:
            return Core::FE::is_nurbs<celltype_t()> ? Core::FE::CellType::nurbs2
                                                    : Core::FE::CellType::line2;
          case 3:
            return Core::FE::is_nurbs<celltype_t()> ? Core::FE::CellType::nurbs3
                                                    : Core::FE::CellType::line3;
          default:
            FOUR_C_THROW("unexpected number of nodes {}", num_node());
        }
      });
}

template <unsigned dim>
void Discret::Elements::SolidLine<dim>::pack(Core::Communication::PackBuffer& data) const
{
  data.add_to_pack(num_dof_per_node_);
}

template <unsigned dim>
void Discret::Elements::SolidLine<dim>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  buffer.extract_from_pack(num_dof_per_node_);
}

template <unsigned dim>
void Discret::Elements::SolidLine<dim>::print(std::ostream& os) const
{
  os << "SolidLine<" + std::to_string(dim) + "> ";
  Element::print(os);
}

template <unsigned dim>
int Discret::Elements::SolidLine<dim>::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elematrix1, Core::LinAlg::SerialDenseMatrix& elematrix2,
    Core::LinAlg::SerialDenseVector& elevector1, Core::LinAlg::SerialDenseVector& elevector2,
    Core::LinAlg::SerialDenseVector& elevector3)
{
  set_params_interface_ptr(params);

  Core::Elements::ActionType act =
      Core::Elements::string_to_action_type(params.get<std::string>("action"));

  switch (act)
  {
    case Core::Elements::ActionType::calc_struct_constrarea:
    {
      FOUR_C_ASSERT_ALWAYS(
          shape() == Core::FE::CellType::line2, "Area Constraint only works for line2 elements!");

      constexpr Core::FE::CellType celltype = Core::FE::CellType::line2;


      // We are not interested in volume of ghosted elements
      if (Core::Communication::my_mpi_rank(discretization.get_comm()) != owner()) return 0;

      auto current_nodal_coordinates =
          evaluate_current_coordinates<celltype, dim>(*this, discretization, lm);

      elevector3[0] = 0.5 * (current_nodal_coordinates[0](1) + current_nodal_coordinates[1](1)) *
                      (current_nodal_coordinates[1](0) - current_nodal_coordinates[0](0));
    }
    break;
    case Core::Elements::ActionType::calc_struct_areaconstrstiff:
    {
      FOUR_C_ASSERT_ALWAYS(
          shape() == Core::FE::CellType::line2, "Area Constraint only works for line2 elements!");

      constexpr Core::FE::CellType celltype = Core::FE::CellType::line2;

      // We are not interested in volume of ghosted elements
      if (Core::Communication::my_mpi_rank(discretization.get_comm()) != owner()) return 0;

      auto current_nodal_coordinates =
          evaluate_current_coordinates<celltype, dim>(*this, discretization, lm);

      FOUR_C_ASSERT(elevector1.length() == 4,
          "Expected elevector1 to have size 4 for line2 area constraint! Given vector has size {}",
          elevector1.length());
      FOUR_C_ASSERT(elematrix1.num_cols() == 4 && elematrix1.num_rows() == 4,
          "Expected elematrix1 to have 4 rows and 4 columns for line2 area constraint! Given "
          "matrix has {} rows and {} columns",
          elematrix1.num_rows(), elematrix1.num_cols());

      elevector1[0] = 0.5 * (current_nodal_coordinates[0](1) + current_nodal_coordinates[1](1));
      elevector1[1] = 0.5 * (current_nodal_coordinates[0](0) - current_nodal_coordinates[1](0));
      elevector1[2] = -0.5 * (current_nodal_coordinates[0](1) + current_nodal_coordinates[1](1));
      elevector1[3] = 0.5 * (current_nodal_coordinates[0](0) - current_nodal_coordinates[1](0));
      elevector2 = elevector1;

      elematrix1.put_scalar(0.0);
      elematrix1(0, 1) = 0.5;
      elematrix1(0, 3) = 0.5;
      elematrix1(1, 0) = 0.5;
      elematrix1(1, 2) = -0.5;
      elematrix1(2, 1) = -0.5;
      elematrix1(2, 3) = -0.5;
      elematrix1(3, 0) = 0.5;
      elematrix1(3, 2) = -0.5;

      elevector3[0] = 0.5 * (current_nodal_coordinates[0](1) + current_nodal_coordinates[1](1)) *
                      (current_nodal_coordinates[1](0) - current_nodal_coordinates[0](0));
    }
    break;
    default:
      FOUR_C_THROW("Unknown type of action {} for SolidLine element", action_type_to_string(act));
  }
  return 0;
}

template <unsigned dim>
int Discret::Elements::SolidLine<dim>::evaluate_neumann(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const Core::Conditions::Condition& condition,
    std::vector<int>& lm, Core::LinAlg::SerialDenseVector& elevec1,
    Core::LinAlg::SerialDenseMatrix* elemat1)
{
  static_assert(dim == 2 || dim == 3, "SolidLine element only implemented for 2D and 3D!");
  set_params_interface_ptr(params);

  const double total_time = std::invoke(
      [&]()
      {
        if (parent_element()->is_params_interface())
          return parent_element()->params_interface_ptr()->get_total_time();
        else
          return params.get("total time", -1.0);
      });

  const auto& load_type = condition.parameters().get<std::string>("TYPE");
  if (load_type == "pseudo_orthopressure" || load_type == "orthopressure")
  {
    if constexpr (dim == 2)
    {
      const auto* parent_solid = dynamic_cast<const Discret::Elements::Solid<2>*>(parent_element());
      FOUR_C_ASSERT_ALWAYS(
          parent_solid != nullptr, "Solid line has no two-dimensional solid parent.");
      Discret::Elements::evaluate_normal_pressure_by_element(*this, discretization, condition, lm,
          elevec1, elemat1, total_time, parent_solid->reference_thickness());
      return 0;
    }
    else
    {
      FOUR_C_THROW("Normal pressure on a solid line is only supported in two dimensions.");
    }
  }

  Discret::Elements::evaluate_neumann_by_element<dim>(
      *this, discretization, condition, elevec1, total_time);
  return 0;
}

template class Discret::Elements::SolidLine<3>;
template class Discret::Elements::SolidLine<2>;

FOUR_C_NAMESPACE_CLOSE
