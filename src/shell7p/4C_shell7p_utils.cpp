// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell7p_utils.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_shell7p_ele.hpp"
#include "4C_shell7p_ele_scatra.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  enum class ShellEasTypes
  {
    none,
    N_1,
    N_2,
    N_3,
    N_4,
    N_5,
    N_6,
    N_7,
    N_8,
    N_9,
    N_11,
    N_undefined
  };

  template <ShellEasTypes eastype>
  struct EasTypeToNumEas
  {
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_1>
  {
    static const int num_eas = 1;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_2>
  {
    static const int num_eas = 2;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_3>
  {
    static const int num_eas = 3;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_4>
  {
    static const int num_eas = 4;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_5>
  {
    static const int num_eas = 5;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_6>
  {
    static const int num_eas = 6;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_7>
  {
    static const int num_eas = 7;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_8>
  {
    static const int num_eas = 8;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_9>
  {
    static const int num_eas = 9;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_11>
  {
    static const int num_eas = 11;
  };
  template <>
  struct EasTypeToNumEas<ShellEasTypes::N_undefined>
  {
  };

  void set_membrane_locking_size_quad4(int& num_eas, const std::string& type)
  {
    if (type == "N_1")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_2")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_3")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_3>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_5")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_5>::num_eas;
    }
    else if (type == "N_7")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_7>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate membrane locking. Only none, N_1, N_2, "
          "N_3, N_4, N_5, N_7 are allowed. Given: {}",
          type.c_str());
  }

  void set_bending_locking_size_quad4(int& num_eas, const std::string& type)
  {
    if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_5")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_5>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "N_7")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_7>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate bending locking. Only none, N_4, N_5, N_6, "
          "N_7 are allowed. Given: {}",
          type.c_str());
  }

  void set_thickness_locking_size_quad4(int& num_eas, const std::string& type)
  {
    if (type == "N_1")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_3")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_3>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "N_8")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_8>::num_eas;
    }
    else if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate thickness locking. Only none, N_1, N_3, "
          "N_4, "
          "N_6, N_8, N_9 are allowed. Given: {}",
          type.c_str());
  }

  void set_shear_strain_locking_size_quad4(int& num_eas, const std::string& type)
  {
    if (type == "N_2")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_2>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad4 to alleviate transverse shear strain locking: Only "
          "none, N_2, N_4 are allowed. Given: {}",
          type.c_str());
  }

  void set_membrane_locking_size_quad9(int& num_eas, const std::string& type)
  {
    if (type == "N_7")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_7>::num_eas;
    }
    else if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate membrane locking. Only none, N_7, N_9 are "
          "allowed. Given: {}",
          type.c_str());
  }

  void set_bending_locking_size_quad9(int& num_eas, const std::string& type)
  {
    if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "N_11")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_11>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate bending locking. Only none, N_9, N_11 are "
          "allowed. Given: {}",
          type.c_str());
  }

  void set_thickness_locking_size_quad9(int& num_eas, const std::string& type)
  {
    if (type == "N_1")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_1>::num_eas;
    }
    else if (type == "N_3")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_3>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "N_8")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_8>::num_eas;
    }
    else if (type == "N_9")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_9>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate thickness locking. Only none, N_1, N_3, "
          "N_4, N_6, N_8, N_9 are allowed. Given: {}",
          type.c_str());
  }

  void set_shear_strain_locking_size_quad9(int& num_eas, const std::string& type)
  {
    if (type == "N_2")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_2>::num_eas;
    }
    else if (type == "N_4")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_4>::num_eas;
    }
    else if (type == "N_6")
    {
      num_eas = EasTypeToNumEas<ShellEasTypes::N_6>::num_eas;
    }
    else if (type == "none")
    {
      num_eas = 0;
    }
    else
      FOUR_C_THROW(
          "Unrecognized EAS type for quad9 to alleviate transverse shear strain locking. Only "
          "none, N_2, N_4, N_6 are allowed. Given: {}",
          type.c_str());
  }

  inline auto SquareValue = [](auto a) { return a * a; };

}  // namespace

Core::LinAlg::SerialDenseMatrix Solid::Utils::Shell::compute_shell_null_space(
    Core::Nodes::Node& node, std::span<const double> x0, const Core::LinAlg::Matrix<3, 1>& dir)
{
  const auto& x = node.x();

  Core::LinAlg::SerialDenseMatrix nullspace(6, 6);
  // x-modes
  nullspace(0, 0) = 1.0;
  nullspace(0, 1) = 0.0;
  nullspace(0, 2) = 0.0;
  nullspace(0, 3) = 0.0;
  nullspace(0, 4) = x[2] - x0[2];
  nullspace(0, 5) = -x[1] + x0[1];
  // y-modes
  nullspace(1, 0) = 0.0;
  nullspace(1, 1) = 1.0;
  nullspace(1, 2) = 0.0;
  nullspace(1, 3) = -x[2] + x0[2];
  nullspace(1, 4) = 0.0;
  nullspace(1, 5) = x[0] - x0[0];
  // z-modes
  nullspace(2, 0) = 0.0;
  nullspace(2, 1) = 0.0;
  nullspace(2, 2) = 1.0;
  nullspace(2, 3) = x[1] - x0[1];
  nullspace(2, 4) = -x[0] + x0[0];
  nullspace(2, 5) = 0.0;
  // dx-modes
  nullspace(3, 0) = 0.0;
  nullspace(3, 1) = 0.0;
  nullspace(3, 2) = 0.0;
  nullspace(3, 3) = 0.0;
  nullspace(3, 4) = dir(2, 0);
  nullspace(3, 5) = -dir(1, 0);
  // dy-modes
  nullspace(4, 0) = 0.0;
  nullspace(4, 1) = 0.0;
  nullspace(4, 2) = 0.0;
  nullspace(4, 3) = -dir(2, 0);
  nullspace(4, 4) = 0.0;
  nullspace(4, 5) = dir(0, 0);
  // dz-modes
  nullspace(5, 0) = 0.0;
  nullspace(5, 1) = 0.0;
  nullspace(5, 2) = 0.0;
  nullspace(5, 3) = dir(1, 0);
  nullspace(5, 4) = -dir(0, 0);
  nullspace(5, 5) = 0.0;

  return nullspace;
}

void Solid::Utils::Shell::nodal_block_information_shell(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  numdf = 6;
  dimns = 6;
}

void Solid::Utils::Shell::Director::setup_director_for_element(
    const Core::Elements::Element& ele, Core::LinAlg::SerialDenseMatrix& nodal_directors)
{
  constexpr auto num_dim = Discret::Elements::Shell::Internal::num_dim;
  const int num_node = ele.num_node();
  Core::LinAlg::SerialDenseMatrix xrefe(num_node, num_dim);
  for (int i = 0; i < num_node; ++i)
  {
    for (int dim = 0; dim < num_dim; ++dim) xrefe(i, dim) = ele.nodes()[i]->x()[dim];
  }
  // allocate matrix for covariant metric vectors
  Core::LinAlg::SerialDenseMatrix metrics_covariant(num_dim, num_dim);
  for (int i = 0; i < num_node; ++i)
  {
    // get shape functions and derivatives at nodes
    Core::LinAlg::Matrix<num_dim, 1> nodal_coordinates =
        Core::FE::get_node_coordinates(i, ele.shape());
    Core::LinAlg::SerialDenseMatrix derivatives(num_dim, num_node);
    Core::FE::shape_function_2d_deriv1(
        derivatives, nodal_coordinates(0), nodal_coordinates(1), ele.shape());

    // get a1, a2 direction derivatives in r and s direction
    Core::LinAlg::multiply(metrics_covariant, derivatives, xrefe);

    // get thickness direction derivative perpendicular to a1 and a2
    // -> a3 = (a1 x a2) / (|a1 x a2 |)
    Core::LinAlg::Matrix<num_dim, 1> a1a2crossprod(Core::LinAlg::Initialization::zero);
    a1a2crossprod(0) = metrics_covariant(0, 1) * metrics_covariant(1, 2) -
                       metrics_covariant(0, 2) * metrics_covariant(1, 1);
    a1a2crossprod(1) = metrics_covariant(0, 2) * metrics_covariant(1, 0) -
                       metrics_covariant(0, 0) * metrics_covariant(1, 2);
    a1a2crossprod(2) = metrics_covariant(0, 0) * metrics_covariant(1, 1) -
                       metrics_covariant(0, 1) * metrics_covariant(1, 0);
    double a1a2crossnorm = a1a2crossprod.norm2();
    if (a1a2crossnorm > 1.0e-14) a1a2crossprod.scale(1.0 / a1a2crossnorm);

    // set nodal director matrix for each node row vice
    for (int j = 0; j < num_dim; j++) nodal_directors(i, j) = a1a2crossprod(j);
  }
}

void Solid::Utils::Shell::Director::average_director(const Core::LinAlg::Matrix<3, 8>& dir_list,
    const int num_directors, Core::LinAlg::Matrix<3, 1>& nodal_director)
{
  Core::LinAlg::Matrix<3, 1> davn(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1> averdir(Core::LinAlg::Initialization::zero);
  for (int dim = 0; dim < Discret::Elements::Shell::Internal::num_dim; ++dim)
    averdir(dim) = dir_list(dim, 0);

  for (int i = 1; i < num_directors; ++i)
  {
    // make cross product of two directors
    Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);
    normal(0) = averdir(1) * dir_list(2, i) - averdir(2) * dir_list(1, i);
    normal(1) = averdir(2) * dir_list(0, i) - averdir(0) * dir_list(2, i);
    normal(2) = averdir(0) * dir_list(1, i) - averdir(1) * dir_list(0, i);
    const double length = normal.dot(normal);

    // if the length is small, both directors point nearly in the same direction
    if (length <= 1.e-12)
    {
      for (int dim = 0; dim < Discret::Elements::Shell::Internal::num_dim; ++dim)
        davn(dim) = 0.5 * (averdir(dim) + dir_list(dim, i));
    }
    // if not average the nodal directors
    else
    {
      const double denom =
          (SquareValue(dir_list(0, i)) + SquareValue(dir_list(2, i))) * SquareValue(averdir(1)) +
          (-2. * dir_list(0, i) * averdir(0) * dir_list(1, i) -
              2. * dir_list(2, i) * averdir(2) * dir_list(1, i)) *
              averdir(1) +
          (SquareValue(dir_list(2, i)) + SquareValue(dir_list(1, i))) * SquareValue(averdir(0)) -
          2. * averdir(2) * averdir(0) * dir_list(2, i) * dir_list(0, i) +
          (SquareValue(dir_list(0, i)) + SquareValue(dir_list(1, i))) * SquareValue(averdir(2));

      if (std::abs(denom) <= 1.e-13) FOUR_C_THROW("Making of modified directors failed");

      const double alpha = (averdir(2) * dir_list(2, i) - SquareValue(dir_list(0, i)) +
                               averdir(0) * dir_list(0, i) - SquareValue(dir_list(1, i)) +
                               dir_list(1, i) * averdir(1) - SquareValue(dir_list(2, i))) /
                           denom;

      davn(0, 0) = -alpha * SquareValue(averdir(1)) * dir_list(0, i) +
                   alpha * averdir(1) * averdir(0) * dir_list(1, i) + averdir(0) +
                   alpha * averdir(2) * averdir(0) * dir_list(2, i) -
                   alpha * SquareValue(averdir(2)) * dir_list(0, i);

      davn(1, 0) = alpha * averdir(0) * averdir(1) * dir_list(0, i) + averdir(1) +
                   alpha * averdir(2) * averdir(1) * dir_list(2, i) -
                   alpha * SquareValue(averdir(0)) * dir_list(1, i) -
                   alpha * SquareValue(averdir(2)) * dir_list(1, i);

      davn(2, 0) = -alpha * SquareValue(averdir(1)) * dir_list(2, i) +
                   alpha * averdir(1) * averdir(2) * dir_list(1, i) -
                   alpha * SquareValue(averdir(0)) * dir_list(2, i) +
                   alpha * averdir(0) * averdir(2) * dir_list(0, i) + averdir(2);
    }
    for (int dim = 0; dim < Discret::Elements::Shell::Internal::num_dim; ++dim)
    {
      averdir(dim) = davn(dim);
      nodal_director(dim) = davn(dim);
    }
  }
}

void Solid::Utils::Shell::Director::export_director_map_from_row_to_col_map(
    const Core::Elements::ElementType& eletype, Core::FE::Discretization& dis,
    std::map<int, std::vector<double>>& director_map)
{
  // export this map from nodal row map to nodal col map
  const Core::LinAlg::Map* noderowmap = dis.node_row_map();
  const Core::LinAlg::Map* nodecolmap = dis.node_col_map();
  Core::Communication::Exporter exporter(*noderowmap, *nodecolmap, dis.get_comm());
  exporter.do_export(director_map);

  // loop through column nodes and put directors back into discretization
  for (auto actnode : dis.my_col_node_range())
  {
    auto curr = director_map.find(actnode.global_id());
    FOUR_C_ASSERT(curr != director_map.end(), "Cannot find director map entry");
    for (auto ele : actnode.adjacent_elements())
    {
      auto* tmpele = ele.user_element();
      if (tmpele->element_type() != eletype) continue;
      if (auto* scatra_ele = dynamic_cast<Discret::Elements::Shell7pScatra*>(tmpele))
      {
        for (int k = 0; k < scatra_ele->num_node(); ++k)
        {
          if (scatra_ele->nodes()[k] == actnode.user_node())
          {
            scatra_ele->set_nodal_director(k, curr->second);
            break;
          }
        }
      }
      else if (auto* shell_ele = dynamic_cast<Discret::Elements::Shell7p*>(tmpele))
      {
        for (int k = 0; k < shell_ele->num_node(); ++k)
        {
          if (shell_ele->nodes()[k] == actnode.user_node())
          {
            shell_ele->set_nodal_director(k, curr->second);
            break;
          }
        }
      }
      else
        FOUR_C_THROW("Element is not a shell element");
    }
  }
}


void Solid::Utils::Shell::Director::average_directors_at_nodes(
    const Core::Elements::ElementType& eletype, const Core::FE::Discretization& dis,
    std::map<int, std::vector<double>>& director_map)
{
  const int max_ele = 8;
  static constexpr int num_dim = Discret::Elements::Shell::Internal::num_dim;
  Core::LinAlg::Matrix<num_dim, max_ele> collaverdir(Core::LinAlg::Initialization::zero);

  // loop through all row nodes and build director map
  for (const auto& act_node : dis.my_row_node_range())
  {
    int num_directors = 0;
    for (auto ele : act_node.adjacent_elements())
    {
      auto* tmpele = ele.user_element();
      if (tmpele->element_type() != eletype) continue;
      if (auto* scatra_ele = dynamic_cast<const Discret::Elements::Shell7pScatra*>(tmpele))
      {
        for (int k = 0; k < scatra_ele->num_node(); ++k)
        {
          if (scatra_ele->nodes()[k] == act_node.user_node())
          {
            const auto nodal_directors = scatra_ele->get_directors();
            for (int dim = 0; dim < num_dim; ++dim)
              collaverdir(dim, num_directors) = nodal_directors(k, dim);
            ++num_directors;
            FOUR_C_ASSERT(num_directors <= max_ele, "max_ele too small");
            break;
          }
        }
      }
      else if (auto* shell_ele = dynamic_cast<const Discret::Elements::Shell7p*>(tmpele))
      {
        for (int k = 0; k < shell_ele->num_node(); ++k)
        {
          if (shell_ele->nodes()[k] == act_node.user_node())
          {
            const auto nodal_directors = shell_ele->get_directors();
            for (int dim = 0; dim < num_dim; ++dim)
              collaverdir(dim, num_directors) = nodal_directors(k, dim);
            ++num_directors;
            FOUR_C_ASSERT(num_directors <= max_ele, "max_ele too small");
            break;
          }
        }
      }
      else
        FOUR_C_THROW("Element is not a shell element");
    }
    FOUR_C_ASSERT(num_directors, "Number of neighboring nodes to a node is zero");

    if (num_directors == 1)  // no averaging if number of neighboring elements to a node is one
    {
      director_map[act_node.global_id()].resize(num_dim);
      for (int dim = 0; dim < num_dim; ++dim)
        director_map[act_node.global_id()][dim] = collaverdir(dim, 0);
    }
    else  // average director at node actnode
    {
      Core::LinAlg::Matrix<num_dim, 1> nodal_director(Core::LinAlg::Initialization::zero);
      average_director(collaverdir, num_directors, nodal_director);
      director_map[act_node.global_id()].resize(num_dim);
      for (int dim = 0; dim < num_dim; ++dim)
        director_map[act_node.global_id()][dim] = nodal_director(dim);
    }
  }
}

void Solid::Utils::Shell::Director::setup_shell_element_directors(
    const Core::Elements::ElementType& eletype, Core::FE::Discretization& dis)
{
  for (auto ele : dis.my_col_element_range())
  {
    auto* actele = ele.user_element();
    if (actele->element_type() != eletype) return;
    if (auto* scatra_ele = dynamic_cast<Discret::Elements::Shell7pScatra*>(actele))
    {
      // create matrix nodal_directors for nodal basis vector in thickness direction in material
      // configuration
      const int num_node = ele.num_nodes();
      Core::LinAlg::SerialDenseMatrix nodal_directors(
          num_node, Discret::Elements::Shell::Internal::num_dim);
      setup_director_for_element(*scatra_ele, nodal_directors);
      scatra_ele->set_all_nodal_directors(nodal_directors);
    }
    else if (auto* shell_ele = dynamic_cast<Discret::Elements::Shell7p*>(actele))
    {
      // create matrix nodal_directors for nodal basis vector in thickness direction in material
      // configuration
      const int num_node = ele.num_nodes();
      Core::LinAlg::SerialDenseMatrix nodal_directors(
          num_node, Discret::Elements::Shell::Internal::num_dim);
      setup_director_for_element(*shell_ele, nodal_directors);
      shell_ele->set_all_nodal_directors(nodal_directors);
    }
    else
      FOUR_C_THROW("Element is not a shell element");
  }

  std::map<int, std::vector<double>> director_map;
  average_directors_at_nodes(eletype, dis, director_map);

  export_director_map_from_row_to_col_map(eletype, dis, director_map);
}



void Solid::Utils::Shell::lump_mass_matrix(Core::LinAlg::SerialDenseMatrix& mass_matrix)
{
  // lump mass matrix
  FOUR_C_ASSERT(mass_matrix.numRows() == mass_matrix.numCols(),
      "The provided mass matrix is not a square matrix!");

  // we assume mass is a square matrix
  for (int c = 0; c < mass_matrix.numCols(); ++c)  // parse columns
  {
    double d = 0.0;
    for (int r = 0; r < mass_matrix.numRows(); ++r)  // parse rows
    {
      d += mass_matrix(r, c);  // accumulate row entries
      mass_matrix(r, c) = 0.0;
    }
    mass_matrix(c, c) = d;  // apply sum of row entries on diagonal
  }
}


void Solid::Utils::Shell::ReadElement::read_and_set_locking_types(const Core::FE::CellType& distype,
    const Core::IO::InputParameterContainer& container,
    Solid::Elements::ShellLockingTypes& locking_types)
{
  std::string type;
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    {
      const auto& eas = container.get<std::vector<std::string>>("EAS");
      set_membrane_locking_size_quad4(locking_types.membrane, eas[0]);
      set_bending_locking_size_quad4(locking_types.bending, eas[1]);
      set_thickness_locking_size_quad4(locking_types.thickness, eas[2]);
      set_shear_strain_locking_size_quad4(locking_types.transverse_shear_strain_const, eas[3]);
      set_shear_strain_locking_size_quad4(locking_types.transverse_shear_strain_lin, eas[4]);
      break;
    }
    case Core::FE::CellType::quad9:
    {
      const auto& eas = container.get<std::vector<std::string>>("EAS");
      set_membrane_locking_size_quad9(locking_types.membrane, eas[0]);
      set_bending_locking_size_quad9(locking_types.bending, eas[1]);
      set_thickness_locking_size_quad9(locking_types.thickness, eas[2]);
      set_shear_strain_locking_size_quad9(locking_types.transverse_shear_strain_const, eas[3]);
      set_shear_strain_locking_size_quad9(locking_types.transverse_shear_strain_lin, eas[4]);
      break;
    }
    default:
      FOUR_C_THROW("EAS is not supported with {}", distype);
  }
  locking_types.total = locking_types.membrane + locking_types.bending + locking_types.thickness +
                        locking_types.transverse_shear_strain_const +
                        locking_types.transverse_shear_strain_lin;
}

int Solid::Utils::Shell::ReadElement::read_and_set_element_material(
    const Core::IO::InputParameterContainer& container)
{
  int material = container.get<int>("MAT");
  return material;
}

int Solid::Utils::Shell::ReadElement::read_and_set_num_ans(const Core::FE::CellType& distype)
{
  switch (distype)
  {
    case Core::FE::CellType::quad4:
    {
      return 2;
    }
    case Core::FE::CellType::quad9:
    {
      return 6;
    }
    default:
      FOUR_C_THROW("ANS is not supported with {}", distype);
  }
}
FOUR_C_NAMESPACE_CLOSE
