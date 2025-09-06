// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_shell_kl_nurbs.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_comm_utils_factory.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/**
 *
 */
Discret::Elements::KirchhoffLoveShellNurbsType
    Discret::Elements::KirchhoffLoveShellNurbsType::instance_;


/**
 *
 */
Discret::Elements::KirchhoffLoveShellNurbsType&
Discret::Elements::KirchhoffLoveShellNurbsType::instance()
{
  return instance_;
}


/**
 *
 */
Core::Communication::ParObject* Discret::Elements::KirchhoffLoveShellNurbsType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::KirchhoffLoveShellNurbs* object =
      new Discret::Elements::KirchhoffLoveShellNurbs(-1, -1);
  object->unpack(buffer);
  return object;
}


/**
 *
 */
std::shared_ptr<Core::Elements::Element> Discret::Elements::KirchhoffLoveShellNurbsType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "SHELL_KIRCHHOFF_LOVE_NURBS" and eledistype == "NURBS9")
  {
    return std::make_shared<Discret::Elements::KirchhoffLoveShellNurbs>(id, owner);
  }
  return nullptr;
}

/**
 *
 */
std::shared_ptr<Core::Elements::Element> Discret::Elements::KirchhoffLoveShellNurbsType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::KirchhoffLoveShellNurbs>(id, owner);
}

/**
 *
 */
void Discret::Elements::KirchhoffLoveShellNurbsType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
  FOUR_C_THROW("NodalBlockInformation not implemented");
}

/**
 *
 */
Core::LinAlg::SerialDenseMatrix Discret::Elements::KirchhoffLoveShellNurbsType::compute_null_space(
    Core::Nodes::Node& node, const double* x0, int const numdof, int const dimnsp)
{
  FOUR_C_THROW("ComputeNullSpace not implemented");
}

/**
 *
 */
void Discret::Elements::KirchhoffLoveShellNurbsType::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["SHELL_KIRCHHOFF_LOVE_NURBS"];

  using namespace Core::IO::InputSpecBuilders;

  defs[Core::FE::CellType::nurbs9] = all_of({
      parameter<int>("MAT"),
      parameter<std::vector<int>>("GP", {.size = 2}),
  });
}


/**
 *
 */
Discret::Elements::KirchhoffLoveShellNurbs::KirchhoffLoveShellNurbs(int id, int owner)
    : Core::Elements::Element(id, owner),
      material_(0),
      gaussrule_({Core::FE::GaussRule1D::undefined, Core::FE::GaussRule1D::undefined})
{
}

/**
 *
 */
Discret::Elements::KirchhoffLoveShellNurbs::KirchhoffLoveShellNurbs(
    const Discret::Elements::KirchhoffLoveShellNurbs& old)
    : Core::Elements::Element(old), material_(old.material_), gaussrule_(old.gaussrule_)
{
}

/**
 *
 */
Core::Elements::Element* Discret::Elements::KirchhoffLoveShellNurbs::clone() const
{
  return new Discret::Elements::KirchhoffLoveShellNurbs(*this);
}

/**
 *
 */
void Discret::Elements::KirchhoffLoveShellNurbs::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Core::Elements::Element::pack(data);
  // material_
  add_to_pack(data, material_);
  // gaussrule_
  add_to_pack(data, gaussrule_[0]);
  add_to_pack(data, gaussrule_[1]);
}

/**
 *
 */
void Discret::Elements::KirchhoffLoveShellNurbs::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Core::Elements::Element::unpack(buffer);
  // material_
  extract_from_pack(buffer, material_);
  // gaussrule_
  extract_from_pack(buffer, gaussrule_[0]);
  extract_from_pack(buffer, gaussrule_[1]);
}

/**
 *
 */
void Discret::Elements::KirchhoffLoveShellNurbs::set_params_interface_ptr(
    const Teuchos::ParameterList& p)
{
  if (p.isParameter("interface"))
    interface_ptr_ = std::dynamic_pointer_cast<Solid::Elements::ParamsInterface>(
        p.get<std::shared_ptr<Core::Elements::ParamsInterface>>("interface"));
  else
    interface_ptr_ = nullptr;
}

/**
 *
 */
std::vector<std::shared_ptr<Core::Elements::Element>>
Discret::Elements::KirchhoffLoveShellNurbs::surfaces()
{
  return {Core::Utils::shared_ptr_from_ref(*this)};
}

FOUR_C_NAMESPACE_CLOSE
