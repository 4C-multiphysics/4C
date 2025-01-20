// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_so3_poro.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_so3_line.hpp"
#include "4C_so3_poro_eletypes.hpp"
#include "4C_so3_surface.hpp"
#include "4C_solid_3D_ele_calc_lib_integration.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

template <class So3Ele, Core::FE::CellType distype>
Discret::Elements::So3Poro<So3Ele, distype>::So3Poro(int id, int owner)
    : So3Ele(id, owner),
      intpoints_(Discret::Elements::create_gauss_integration<distype>(
          Discret::Elements::DisTypeToOptGaussRule<distype>::rule)),
      init_(false),
      isNurbs_(false),
      weights_(true),
      myknots_(numdim_),
      fluid_mat_(nullptr),
      fluidmulti_mat_(nullptr),
      struct_mat_(nullptr)
{
  numgpt_ = intpoints_.num_points();

  invJ_.resize(numgpt_, Core::LinAlg::Matrix<numdim_, numdim_>(true));
  detJ_.resize(numgpt_, 0.0);
  xsi_.resize(numgpt_, Core::LinAlg::Matrix<numdim_, 1>(true));
  anisotropic_permeability_directions_.resize(3, std::vector<double>(3, 0.0));
  anisotropic_permeability_nodal_coeffs_.resize(3, std::vector<double>(numnod_, 0.0));
}

template <class So3Ele, Core::FE::CellType distype>
Discret::Elements::So3Poro<So3Ele, distype>::So3Poro(
    const Discret::Elements::So3Poro<So3Ele, distype>& old)
    : So3Ele(old),
      invJ_(old.invJ_),
      detJ_(old.detJ_),
      xsi_(old.xsi_),
      intpoints_(Discret::Elements::create_gauss_integration<distype>(
          Discret::Elements::DisTypeToOptGaussRule<distype>::rule)),
      init_(old.init_),
      isNurbs_(old.isNurbs_),
      weights_(old.weights_),
      myknots_(old.myknots_),
      fluid_mat_(old.fluid_mat_),
      fluidmulti_mat_(old.fluidmulti_mat_),
      struct_mat_(old.struct_mat_),
      anisotropic_permeability_directions_(old.anisotropic_permeability_directions_),
      anisotropic_permeability_nodal_coeffs_(old.anisotropic_permeability_nodal_coeffs_)
{
  numgpt_ = intpoints_.num_points();
}

template <class So3Ele, Core::FE::CellType distype>
Core::Elements::Element* Discret::Elements::So3Poro<So3Ele, distype>::clone() const
{
  auto* newelement = new Discret::Elements::So3Poro<So3Ele, distype>(*this);
  return newelement;
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // detJ_
  add_to_pack(data, detJ_);

  // invJ_
  auto size = static_cast<int>(invJ_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, invJ_[i]);

  // xsi_
  size = static_cast<int>(xsi_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, xsi_[i]);

  // isNurbs_
  add_to_pack(data, isNurbs_);

  // anisotropic_permeability_directions_
  size = static_cast<int>(anisotropic_permeability_directions_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, anisotropic_permeability_directions_[i]);

  // anisotropic_permeability_nodal_coeffs_
  size = static_cast<int>(anisotropic_permeability_nodal_coeffs_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, anisotropic_permeability_nodal_coeffs_[i]);

  // add base class Element
  So3Ele::pack(data);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // detJ_
  extract_from_pack(buffer, detJ_);

  // invJ_
  int size = 0;
  extract_from_pack(buffer, size);
  invJ_.resize(size, Core::LinAlg::Matrix<numdim_, numdim_>(true));
  for (int i = 0; i < size; ++i) extract_from_pack(buffer, invJ_[i]);

  // xsi_
  size = 0;
  extract_from_pack(buffer, size);
  xsi_.resize(size, Core::LinAlg::Matrix<numdim_, 1>(true));
  for (int i = 0; i < size; ++i) extract_from_pack(buffer, xsi_[i]);

  // isNurbs_
  extract_from_pack(buffer, isNurbs_);

  // anisotropic_permeability_directions_
  size = 0;
  extract_from_pack(buffer, size);
  anisotropic_permeability_directions_.resize(size, std::vector<double>(3, 0.0));
  for (int i = 0; i < size; ++i) extract_from_pack(buffer, anisotropic_permeability_directions_[i]);

  // anisotropic_permeability_nodal_coeffs_
  size = 0;
  extract_from_pack(buffer, size);
  anisotropic_permeability_nodal_coeffs_.resize(size, std::vector<double>(numnod_, 0.0));
  for (int i = 0; i < size; ++i)
    extract_from_pack(buffer, anisotropic_permeability_nodal_coeffs_[i]);

  // extract base class Element
  So3Ele::unpack(buffer);

  init_ = true;
}

template <class So3Ele, Core::FE::CellType distype>
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::So3Poro<So3Ele, distype>::surfaces()
{
  return Core::Communication::element_boundary_factory<StructuralSurface, Core::Elements::Element>(
      Core::Communication::buildSurfaces, *this);
}

template <class So3Ele, Core::FE::CellType distype>
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::So3Poro<So3Ele, distype>::lines()
{
  return Core::Communication::element_boundary_factory<StructuralLine, Core::Elements::Element>(
      Core::Communication::buildLines, *this);
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::print(std::ostream& os) const
{
  os << "So3_poro ";
  os << Core::FE::cell_type_to_string(distype).c_str() << " ";
  Core::Elements::Element::print(os);
}

template <class So3Ele, Core::FE::CellType distype>
bool Discret::Elements::So3Poro<So3Ele, distype>::read_element(const std::string& eletype,
    const std::string& eledistype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  So3Ele::read_element(eletype, eledistype, container);

  // setup poro material
  std::shared_ptr<Mat::StructPoro> poromat = std::dynamic_pointer_cast<Mat::StructPoro>(material());
  if (poromat == nullptr) FOUR_C_THROW("no poro material assigned to poro element!");
  poromat->poro_setup(numgpt_, container);

  read_anisotropic_permeability_directions_from_element_line_definition(container);
  read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(container);

  return true;
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::
    read_anisotropic_permeability_directions_from_element_line_definition(
        const Core::IO::InputParameterContainer& container)
{
  for (int dim = 0; dim < 3; ++dim)
  {
    std::string definition_name = "POROANISODIR" + std::to_string(dim + 1);
    if (container.get_if<std::vector<double>>(definition_name) != nullptr)
      anisotropic_permeability_directions_[dim] =
          container.get<std::vector<double>>(definition_name);
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::
    read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(
        const Core::IO::InputParameterContainer& container)
{
  for (int dim = 0; dim < 3; ++dim)
  {
    std::string definition_name = "POROANISONODALCOEFFS" + std::to_string(dim + 1);
    if (container.get_if<std::vector<double>>(definition_name) != nullptr)
      anisotropic_permeability_nodal_coeffs_[dim] =
          container.get<std::vector<double>>(definition_name);
  }
}

template <class So3Ele, Core::FE::CellType distype>
void Discret::Elements::So3Poro<So3Ele, distype>::vis_names(std::map<std::string, int>& names)
{
  So3Ele::vis_names(names);
}

template <class So3Ele, Core::FE::CellType distype>
bool Discret::Elements::So3Poro<So3Ele, distype>::vis_data(
    const std::string& name, std::vector<double>& data)
{
  return So3Ele::vis_data(name, data);
}

template <class So3Ele, Core::FE::CellType distype>
int Discret::Elements::So3Poro<So3Ele, distype>::unique_par_object_id() const
{
  switch (distype)
  {
    case Core::FE::CellType::tet4:
      return SoTet4PoroType::instance().unique_par_object_id();
    case Core::FE::CellType::tet10:
      return SoTet10PoroType::instance().unique_par_object_id();
    case Core::FE::CellType::hex8:
      return SoHex8PoroType::instance().unique_par_object_id();
    case Core::FE::CellType::hex27:
      return SoHex27PoroType::instance().unique_par_object_id();
    case Core::FE::CellType::nurbs27:
      return SoNurbs27PoroType::instance().unique_par_object_id();
    default:
      FOUR_C_THROW("unknown element type!");
      break;
  }
  return -1;
}

template <class So3Ele, Core::FE::CellType distype>
Core::Elements::ElementType& Discret::Elements::So3Poro<So3Ele, distype>::element_type() const
{
  switch (distype)
  {
    case Core::FE::CellType::tet4:
      return SoTet4PoroType::instance();
    case Core::FE::CellType::tet10:
      return SoTet10PoroType::instance();
    case Core::FE::CellType::hex8:
      return SoHex8PoroType::instance();
    case Core::FE::CellType::hex27:
      return SoHex27PoroType::instance();
    case Core::FE::CellType::nurbs27:
      return SoNurbs27PoroType::instance();
    default:
      FOUR_C_THROW("unknown element type!");
  }
}

template <class So3Ele, Core::FE::CellType distype>
inline Core::Nodes::Node** Discret::Elements::So3Poro<So3Ele, distype>::nodes()
{
  return So3Ele::nodes();
}

template <class So3Ele, Core::FE::CellType distype>
inline std::shared_ptr<Core::Mat::Material> Discret::Elements::So3Poro<So3Ele, distype>::material()
    const
{
  return So3Ele::material();
}

template <class So3Ele, Core::FE::CellType distype>
inline int Discret::Elements::So3Poro<So3Ele, distype>::id() const
{
  return So3Ele::id();
}

FOUR_C_NAMESPACE_CLOSE

#include "4C_so3_poro.inst.hpp"