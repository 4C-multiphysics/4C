// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_w1_poro.hpp"

#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_mat_fluidporo.hpp"
#include "4C_mat_fluidporo_multiphase.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_poroelast_utils.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

template <Core::FE::CellType distype>
Discret::Elements::Wall1Poro<distype>::Wall1Poro(int id, int owner)
    : Discret::Elements::Wall1(id, owner), intpoints_(distype), weights_(true), myknots_(numdim_)
{
  numgpt_ = intpoints_.num_points();

  invJ_.resize(numgpt_, Core::LinAlg::Matrix<numdim_, numdim_>(true));
  detJ_.resize(numgpt_, 0.0);
  xsi_.resize(numgpt_, Core::LinAlg::Matrix<numdim_, 1>(true));
  anisotropic_permeability_directions_.resize(2, std::vector<double>(2, 0.0));
  anisotropic_permeability_nodal_coeffs_.resize(2, std::vector<double>(numnod_, 0.0));

  init_ = false;

  scatra_coupling_ = false;
}

template <Core::FE::CellType distype>
Discret::Elements::Wall1Poro<distype>::Wall1Poro(const Discret::Elements::Wall1Poro<distype>& old)
    : Discret::Elements::Wall1(old),
      invJ_(old.invJ_),
      detJ_(old.detJ_),
      xsi_(old.xsi_),
      intpoints_(distype),
      init_(old.init_),
      scatra_coupling_(old.scatra_coupling_),
      weights_(old.weights_),
      myknots_(old.myknots_),
      anisotropic_permeability_directions_(old.anisotropic_permeability_directions_),
      anisotropic_permeability_nodal_coeffs_(old.anisotropic_permeability_nodal_coeffs_)
{
  numgpt_ = intpoints_.num_points();
}

template <Core::FE::CellType distype>
Core::Elements::Element* Discret::Elements::Wall1Poro<distype>::clone() const
{
  auto* newelement = new Discret::Elements::Wall1Poro<distype>(*this);
  return newelement;
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // detJ_
  add_to_pack(data, detJ_);

  // invJ_
  int size = static_cast<int>(invJ_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, invJ_[i]);

  // xsi_
  size = static_cast<int>(xsi_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, xsi_[i]);

  // scatra_coupling_
  add_to_pack(data, scatra_coupling_);

  // anisotropic_permeability_directions_
  size = static_cast<int>(anisotropic_permeability_directions_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, anisotropic_permeability_directions_[i]);

  // anisotropic_permeability_nodal_coeffs_
  size = static_cast<int>(anisotropic_permeability_nodal_coeffs_.size());
  add_to_pack(data, size);
  for (int i = 0; i < size; ++i) add_to_pack(data, anisotropic_permeability_nodal_coeffs_[i]);

  // add base class Element
  Discret::Elements::Wall1::pack(data);
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::unpack(Core::Communication::UnpackBuffer& buffer)
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

  // scatra_coupling_
  extract_from_pack(buffer, scatra_coupling_);

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
  Discret::Elements::Wall1::unpack(buffer);

  init_ = true;
}

template <Core::FE::CellType distype>
std::vector<std::shared_ptr<Core::Elements::Element>> Discret::Elements::Wall1Poro<distype>::lines()
{
  return Core::Communication::element_boundary_factory<Wall1Line, Wall1Poro>(
      Core::Communication::buildLines, *this);
}

template <Core::FE::CellType distype>
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::Wall1Poro<distype>::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::print(std::ostream& os) const
{
  os << "Wall1_Poro ";
  Element::print(os);
  std::cout << std::endl;
}

template <Core::FE::CellType distype>
bool Discret::Elements::Wall1Poro<distype>::read_element(const std::string& eletype,
    const std::string& eledistype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  Wall1::read_element(eletype, eledistype, container);

  // setup poro material
  std::shared_ptr<Mat::StructPoro> poromat = std::dynamic_pointer_cast<Mat::StructPoro>(material());
  if (poromat == nullptr) FOUR_C_THROW("material assigned to poro element is not a poro material!");
  poromat->poro_setup(numgpt_, container);

  read_anisotropic_permeability_directions_from_element_line_definition(container);
  read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(container);

  return true;
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::
    read_anisotropic_permeability_directions_from_element_line_definition(
        const Core::IO::InputParameterContainer& container)
{
  for (int dim = 0; dim < 2; ++dim)
  {
    std::string definition_name = "POROANISODIR" + std::to_string(dim + 1);
    if (container.get_if<std::vector<double>>(definition_name) != nullptr)
      anisotropic_permeability_directions_[dim] =
          container.get<std::vector<double>>(definition_name);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::
    read_anisotropic_permeability_nodal_coeffs_from_element_line_definition(
        const Core::IO::InputParameterContainer& container)
{
  for (int dim = 0; dim < 2; ++dim)
  {
    std::string definition_name = "POROANISONODALCOEFFS" + std::to_string(dim + 1);
    if (container.get_if<std::vector<double>>(definition_name) != nullptr)
      anisotropic_permeability_nodal_coeffs_[dim] =
          container.get<std::vector<double>>(definition_name);
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::get_materials()
{
  // get structure material
  if (struct_mat_ == nullptr)
  {
    struct_mat_ = std::dynamic_pointer_cast<Mat::StructPoro>(material());
    if (struct_mat_ == nullptr) FOUR_C_THROW("cast to poro material failed");

    if (struct_mat_->material_type() != Core::Materials::m_structporo and
        struct_mat_->material_type() != Core::Materials::m_structpororeaction and
        struct_mat_->material_type() != Core::Materials::m_structpororeactionECM)
      FOUR_C_THROW("invalid structure material for poroelasticity");
  }

  // get fluid material
  if (fluid_mat_ == nullptr)
  {
    // access second material in structure element
    if (num_material() > 1)
    {
      fluid_mat_ = std::dynamic_pointer_cast<Mat::FluidPoro>(material(1));
      if (fluid_mat_ == nullptr) return;
      // FOUR_C_THROW("cast to fluid poro material failed");
      if (fluid_mat_->material_type() != Core::Materials::m_fluidporo)
        FOUR_C_THROW("invalid fluid material for poroelasticity");
    }
    else
      FOUR_C_THROW("no second material defined for element %i", id());
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::get_materials_pressure_based()
{
  // get structure material
  if (struct_mat_ == nullptr)
  {
    struct_mat_ = std::dynamic_pointer_cast<Mat::StructPoro>(material());
    if (struct_mat_ == nullptr) FOUR_C_THROW("cast to poro material failed");

    if (struct_mat_->material_type() != Core::Materials::m_structporo and
        struct_mat_->material_type() != Core::Materials::m_structpororeaction and
        struct_mat_->material_type() != Core::Materials::m_structpororeactionECM)
      FOUR_C_THROW("invalid structure material for poroelasticity");
  }

  // Get Fluid-multiphase-Material
  if (fluidmulti_mat_ == nullptr)
  {
    // access second material in structure element
    if (num_material() > 1)
    {
      fluidmulti_mat_ = std::dynamic_pointer_cast<Mat::FluidPoroMultiPhase>(material(1));
      if (fluidmulti_mat_ == nullptr) FOUR_C_THROW("cast to multiphase fluid poro material failed");
      if (fluidmulti_mat_->material_type() != Core::Materials::m_fluidporo_multiphase and
          fluidmulti_mat_->material_type() != Core::Materials::m_fluidporo_multiphase_reactions)
        FOUR_C_THROW("invalid fluid material for poro-multiphase-elasticity");
      if (fluidmulti_mat_->num_fluid_phases() == 0)
      {
        FOUR_C_THROW(
            "NUMFLUIDPHASES_IN_MULTIPHASEPORESPACE = 0 currently not supported since this requires "
            "an adaption of the definition of the solid pressure");
      }
    }
    else
      FOUR_C_THROW("no second material defined for element %i", id());
  }
}

template <Core::FE::CellType distype>
void Discret::Elements::Wall1Poro<distype>::vis_names(std::map<std::string, int>& names)
{
  solid_material()->vis_names(names);
}

template <Core::FE::CellType distype>
bool Discret::Elements::Wall1Poro<distype>::vis_data(
    const std::string& name, std::vector<double>& data)
{
  // Put the owner of this element into the file (use base class method for this)
  if (Wall1::vis_data(name, data)) return true;

  return solid_material()->vis_data(name, data, numgpt_, this->id());
}

template class Discret::Elements::Wall1Poro<Core::FE::CellType::tri3>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::quad4>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::quad9>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::nurbs4>;
template class Discret::Elements::Wall1Poro<Core::FE::CellType::nurbs9>;

FOUR_C_NAMESPACE_CLOSE
