// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fluid_ele_hdg_weak_comp.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fluid_ele_action.hpp"
#include "4C_fluid_ele_factory.hpp"
#include "4C_fluid_ele_interface.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_parameter_list.hpp"

FOUR_C_NAMESPACE_OPEN


// initialize static variable
Discret::Elements::FluidHDGWeakCompType Discret::Elements::FluidHDGWeakCompType::instance_;

Discret::Elements::FluidHDGWeakCompType& Discret::Elements::FluidHDGWeakCompType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::FluidHDGWeakCompType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  Discret::Elements::FluidHDGWeakComp* object = new Discret::Elements::FluidHDGWeakComp(-1, -1);
  object->unpack(buffer);
  return object;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::FluidHDGWeakCompType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "FLUIDHDGWEAKCOMP")
  {
    return std::make_shared<Discret::Elements::FluidHDGWeakComp>(id, owner);
  }
  return nullptr;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::FluidHDGWeakCompType::create(
    const int id, const int owner)
{
  return std::make_shared<Discret::Elements::FluidHDGWeakComp>(id, owner);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidHDGWeakCompType::nodal_block_information(
    Core::Elements::Element* dwele, int& numdf, int& dimns)
{
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidHDGWeakCompType::compute_null_space(
    Core::FE::Discretization& dis, std::vector<double>& ns, const double* x0, int numdf, int dimns)
{
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidHDGWeakCompType ::setup_element_definition(
    std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
{
  std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>> definitions_fluid;
  FluidType::setup_element_definition(definitions_fluid);

  auto& defs_fluid = definitions_fluid["FLUID"];

  auto& defs_hdg = definitions["FLUIDHDGWEAKCOMP"];

  using namespace Core::IO::InputSpecBuilders;

  for (const auto& [key, fluid_line_def] : defs_fluid)
  {
    defs_hdg[key] = all_of({
        fluid_line_def,
        parameter<int>("DEG"),
        parameter<std::optional<bool>>("SPC"),
    });
  }
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::FluidHDGWeakComp::FluidHDGWeakComp(int id, int owner)
    : Fluid(id, owner), degree_(1), completepol_(true)
{
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::FluidHDGWeakComp::FluidHDGWeakComp(
    const Discret::Elements::FluidHDGWeakComp& old)
    : Fluid(old), degree_(old.degree_), completepol_(old.completepol_)
{
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::FluidHDGWeakComp::clone() const
{
  Discret::Elements::FluidHDGWeakComp* newelement = new Discret::Elements::FluidHDGWeakComp(*this);
  return newelement;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidHDGWeakComp::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // add base class Element
  Fluid::pack(data);

  int degree = degree_;
  add_to_pack(data, degree);
  degree = completepol_;
  add_to_pack(data, degree);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidHDGWeakComp::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Fluid::unpack(buffer);

  int val = 0;
  extract_from_pack(buffer, val);
  FOUR_C_ASSERT(val >= 0 && val < 255, "Degree out of range");
  degree_ = val;
  extract_from_pack(buffer, val);
  completepol_ = val;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::Elements::FluidHDGWeakComp::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  bool success = Fluid::read_element(eletype, distype, container);
  degree_ = container.get<int>("DEG");

  completepol_ = container.get<std::optional<bool>>("SPC").value_or(false);

  return success;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
int Discret::Elements::FluidHDGWeakComp::evaluate(Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, std::vector<int>& lm,
    Core::LinAlg::SerialDenseMatrix& elemat1, Core::LinAlg::SerialDenseMatrix& elemat2,
    Core::LinAlg::SerialDenseVector& elevec1, Core::LinAlg::SerialDenseVector& elevec2,
    Core::LinAlg::SerialDenseVector& elevec3)
{
  // get the action required
  const auto act = Teuchos::getIntegralValue<FLD::Action>(params, "action");

  // get material
  std::shared_ptr<Core::Mat::Material> mat = material();

  // switch between different physical types as used below
  std::string impltype = "hdgweakcomp";

  switch (act)
  {
    //-----------------------------------------------------------------------
    // standard implementation enabling time-integration schemes such as
    // one-step-theta, BDF2, and generalized-alpha (n+alpha_F and n+1)
    //-----------------------------------------------------------------------
    case FLD::calc_fluid_systemmat_and_residual:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate(
              this, discretization, lm, params, mat, elemat1, elemat2, elevec1, elevec2, elevec3);
    }
    break;

    case FLD::calc_mass_matrix:
    case FLD::calc_fluid_error:
    case FLD::integrate_shape:
    case FLD::interpolate_hdg_to_node:
    case FLD::project_fluid_field:
    case FLD::update_local_solution:
    {
      return Discret::Elements::FluidFactory::provide_impl(shape(), impltype)
          ->evaluate_service(
              this, params, mat, discretization, lm, elemat1, elemat2, elevec1, elevec2, elevec3);
      break;
    }

    default:
      FOUR_C_THROW("Unknown type of action '{}' for FluidHDGWeakComp", act);
      break;
  }

  return 0;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::FluidHDGWeakComp::print(std::ostream& os) const
{
  os << "FluidHDGWeakComp ";
  Element::print(os);
}

FOUR_C_NAMESPACE_CLOSE
