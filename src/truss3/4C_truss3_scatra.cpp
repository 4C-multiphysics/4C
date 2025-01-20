// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_truss3_scatra.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_mat_lin_elast_1D.hpp"
#include "4C_structure_new_elements_paramsinterface.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Truss3ScatraType Discret::Elements::Truss3ScatraType::instance_;

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Truss3ScatraType& Discret::Elements::Truss3ScatraType::instance()
{
  return instance_;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Communication::ParObject* Discret::Elements::Truss3ScatraType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* object = new Discret::Elements::Truss3Scatra(-1, -1);
  object->unpack(buffer);
  return object;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Truss3ScatraType::create(
    const std::string eletype, const std::string eledistype, const int id, const int owner)
{
  if (eletype == "TRUSS3SCATRA")
  {
    std::shared_ptr<Core::Elements::Element> ele =
        std::make_shared<Discret::Elements::Truss3Scatra>(id, owner);
    return ele;
  }
  // return base class
  else
    return Discret::Elements::Truss3Type::create(eletype, eledistype, id, owner);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Elements::Element> Discret::Elements::Truss3ScatraType::create(
    const int id, const int owner)
{
  std::shared_ptr<Core::Elements::Element> ele =
      std::make_shared<Discret::Elements::Truss3Scatra>(id, owner);
  return ele;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3ScatraType::setup_element_definition(
    std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions)
{
  auto& defs = definitions["TRUSS3SCATRA"];

  using namespace Core::IO::InputSpecBuilders;

  // get definitions from standard truss element
  std::map<std::string, std::map<std::string, Core::IO::InputSpec>> definitions_truss;
  Truss3Type::setup_element_definition(definitions_truss);
  auto& defs_truss = definitions_truss["TRUSS3"];

  // copy definitions of standard truss element to truss element for scalar transport coupling
  defs["LINE2"] = anonymous_group({
      defs_truss["LINE2"],
      entry<std::string>("TYPE"),
  });
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Truss3Scatra::Truss3Scatra(int id, int owner)
    : Truss3(id, owner), impltype_(Inpar::ScaTra::impltype_undefined)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Discret::Elements::Truss3Scatra::Truss3Scatra(const Discret::Elements::Truss3Scatra& old)
    : Truss3(static_cast<Truss3>(old)), impltype_(old.impltype_)
{
}
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Core::Elements::Element* Discret::Elements::Truss3Scatra::clone() const
{
  auto* newelement = new Discret::Elements::Truss3Scatra(*this);
  return newelement;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);
  // add base class Element
  Truss3::pack(data);
  add_to_pack(data, impltype_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // extract base class Element
  Truss3::unpack(buffer);

  extract_from_pack(buffer, impltype_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
bool Discret::Elements::Truss3Scatra::read_element(const std::string& eletype,
    const std::string& distype, const Core::IO::InputParameterContainer& container)
{
  // read base element
  Truss3::read_element(eletype, distype, container);

  // read scalar transport implementation type
  auto impltype = container.get<std::string>("TYPE");

  if (impltype == "ElchDiffCond")
    impltype_ = Inpar::ScaTra::impltype_elch_diffcond;
  else if (impltype == "ElchDiffCondMultiScale")
    impltype_ = Inpar::ScaTra::impltype_elch_diffcond_multiscale;
  else if (impltype == "ElchElectrode")
    impltype_ = Inpar::ScaTra::impltype_elch_electrode;
  else
    FOUR_C_THROW("Invalid implementation type for Truss3Scatra elements!");

  return true;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::calc_internal_force_stiff_tot_lag(
    const std::map<std::string, std::vector<double>>& ele_state,
    Core::LinAlg::SerialDenseVector& forcevec, Core::LinAlg::SerialDenseMatrix& stiffmat)
{
  // safety check
  if (material()->material_type() != Core::Materials::m_linelast1D_growth and
      material()->material_type() != Core::Materials::m_linelast1D)
    FOUR_C_THROW("only linear elastic growth material supported for truss element");

  switch (material()->material_type())
  {
    case Core::Materials::m_linelast1D:
    {
      Truss3::calc_internal_force_stiff_tot_lag(ele_state, forcevec, stiffmat);
      break;
    }
    case Core::Materials::m_linelast1D_growth:
    {
      Core::LinAlg::Matrix<6, 1> curr_nodal_coords;
      Core::LinAlg::Matrix<6, 6> dtruss_disp_du;
      Core::LinAlg::Matrix<6, 1> dN_dx;
      Core::LinAlg::Matrix<2, 1> nodal_concentration;
      const int ndof = 6;

      prep_calc_internal_force_stiff_tot_lag_scatra(
          ele_state, curr_nodal_coords, dtruss_disp_du, dN_dx, nodal_concentration);

      // get data from input
      const auto* growth_mat = static_cast<const Mat::LinElast1DGrowth*>(material().get());

      // get Gauss rule
      auto intpoints = Core::FE::IntegrationPoints1D(gaussrule_);

      // computing forcevec and stiffmat
      forcevec.putScalar(0.0);
      stiffmat.putScalar(0.0);
      for (int gp = 0; gp < intpoints.nquad; ++gp)
      {
        const double dx_dxi = lrefe_ / 2.0;
        const double int_fac = dx_dxi * intpoints.qwgt[gp] * crosssec_;

        // get concentration at Gauss point
        const double c_GP =
            project_scalar_to_gauss_point(intpoints.qxg[gp][0], nodal_concentration);

        // calculate stress
        const double PK2_1D =
            growth_mat->evaluate_p_k2(curr_length(curr_nodal_coords) / lrefe_, c_GP);
        const double stiffness =
            growth_mat->evaluate_stiffness(curr_length(curr_nodal_coords) / lrefe_, c_GP);

        // calculate residual (force.vec) and linearisation (stiffmat)
        for (int row = 0; row < ndof; ++row)
        {
          const double def_grad = curr_nodal_coords(row) / lrefe_;
          const double scalar_R = int_fac * def_grad * PK2_1D;
          forcevec(row) += dN_dx(row) * scalar_R;
          for (int col = 0; col < ndof; ++col)
          {
            const double ddef_grad_du = dtruss_disp_du(row, col) / lrefe_;
            const double sign = (col < 3 ? 1.0 : -1.0);
            const double dPK2_1D_du =
                2.0 * stiffness * d_curr_lengthdu(curr_nodal_coords, col) / lrefe_ * sign;
            const double first_part = dN_dx(row) * ddef_grad_du * PK2_1D;
            const double second_part = dN_dx(row) * def_grad * dPK2_1D_du;
            stiffmat(row, col) += (first_part + second_part) * int_fac;
          }
        }
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Material type is not supported");
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::calc_gp_stresses(
    Teuchos::ParameterList& params, const std::map<std::string, std::vector<double>>& ele_state)
{
  // safety check
  if (material()->material_type() != Core::Materials::m_linelast1D_growth and
      material()->material_type() != Core::Materials::m_linelast1D)
    FOUR_C_THROW("only linear elastic growth material supported for truss element");

  switch (material()->material_type())
  {
    case Core::Materials::m_linelast1D:
    {
      Truss3::calc_gp_stresses(params, ele_state);
      break;
    }
    case Core::Materials::m_linelast1D_growth:
    {
      std::shared_ptr<std::vector<char>> stressdata = nullptr;
      Inpar::Solid::StressType iostress;
      if (is_params_interface())
      {
        stressdata = params_interface().stress_data_ptr();
        iostress = params_interface().get_stress_output_type();
      }
      else
      {
        stressdata = params.get<std::shared_ptr<std::vector<char>>>("stress", nullptr);
        iostress = params.get<Inpar::Solid::StressType>("iostress", Inpar::Solid::stress_none);
      }

      const Core::FE::IntegrationPoints1D intpoints(gaussrule_);

      Core::LinAlg::SerialDenseMatrix stress(intpoints.nquad, 1);

      Core::LinAlg::Matrix<6, 1> curr_nodal_coords;
      Core::LinAlg::Matrix<6, 6> dtruss_disp_du;
      Core::LinAlg::Matrix<6, 1> dN_dx;
      Core::LinAlg::Matrix<2, 1> nodal_concentration;

      prep_calc_internal_force_stiff_tot_lag_scatra(
          ele_state, curr_nodal_coords, dtruss_disp_du, dN_dx, nodal_concentration);

      // get data from input
      const auto* growth_mat = static_cast<const Mat::LinElast1DGrowth*>(material().get());

      const double def_grad = curr_length(curr_nodal_coords) / lrefe_;
      for (int gp = 0; gp < intpoints.nquad; ++gp)
      {
        // get concentration at Gauss point
        const double c_GP =
            project_scalar_to_gauss_point(intpoints.qxg[gp][0], nodal_concentration);

        const double PK2 = growth_mat->evaluate_p_k2(def_grad, c_GP);

        switch (iostress)
        {
          case Inpar::Solid::stress_2pk:
          {
            stress(gp, 0) = PK2;
            break;
          }
          case Inpar::Solid::stress_cauchy:
          {
            stress(gp, 0) = PK2 * def_grad;
            break;
          }
          case Inpar::Solid::stress_none:
            break;
          default:
            FOUR_C_THROW("Requested stress type not available");
        }
      }
      {
        Core::Communication::PackBuffer data;
        add_to_pack(data, stress);
        std::copy(data().begin(), data().end(), std::back_inserter(*stressdata));
      }
    }
    break;
    default:
    {
      FOUR_C_THROW("Material type is not supported");
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
double Discret::Elements::Truss3Scatra::project_scalar_to_gauss_point(
    const double xi, const Core::LinAlg::Matrix<2, 1>& c) const
{
  return (c(1) - c(0)) / 2.0 * xi + (c(1) + c(0)) / 2.0;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::extract_elemental_variables(Core::Elements::LocationArray& la,
    const Core::FE::Discretization& discretization, const Teuchos::ParameterList& params,
    std::map<std::string, std::vector<double>>& ele_state)
{
  // add displacements
  Truss3::extract_elemental_variables(la, discretization, params, ele_state);

  // first: check, if micro state is set; if not -> take macro state
  // get nodal phi from micro state
  std::vector<double> phi_ele;
  if (discretization.num_dof_sets() == 3 and discretization.has_state(2, "MicroCon"))
  {
    phi_ele.resize(la[2].lm_.size());
    phi_ele.clear();
    auto phi = discretization.get_state(2, "MicroCon");
    if (phi == nullptr) FOUR_C_THROW("Cannot get state vector 'MicroCon'");
    Core::FE::extract_my_values(*phi, phi_ele, la[2].lm_);
  }
  // get nodal phi from micro state
  else if (discretization.has_state(1, "scalarfield"))
  {
    phi_ele.resize(la[1].lm_.size());
    phi_ele.clear();
    auto phi = discretization.get_state(1, "scalarfield");
    if (phi == nullptr) FOUR_C_THROW("Cannot get state vectors 'scalar'");
    Core::FE::extract_my_values(*phi, phi_ele, la[1].lm_);
  }
  else
    FOUR_C_THROW("Cannot find state vector");

  if (ele_state.find("phi") == ele_state.end())
    ele_state.emplace(std::make_pair("phi", phi_ele));
  else
    ele_state["phi"] = phi_ele;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::prep_calc_internal_force_stiff_tot_lag_scatra(
    const std::map<std::string, std::vector<double>>& ele_state,
    Core::LinAlg::Matrix<6, 1>& curr_nodal_coords,
    Core::LinAlg::Matrix<6, 6>& dcurr_nodal_coords_du, Core::LinAlg::Matrix<6, 1>& dN_dx,
    Core::LinAlg::Matrix<2, 1>& nodal_concentration)
{
  prep_calc_internal_force_stiff_tot_lag(
      ele_state, curr_nodal_coords, dcurr_nodal_coords_du, dN_dx);

  const std::vector<double>& phi_ele = ele_state.at("phi");

  nodal_concentration(0) = phi_ele[0];
  switch (phi_ele.size())
  {
    case 2:
      nodal_concentration(1) = phi_ele[1];
      break;
    case 4:
      nodal_concentration(1) = phi_ele[2];
      break;
    case 6:
      nodal_concentration(1) = phi_ele[3];
      break;
    default:
      FOUR_C_THROW("Vector has size other than 2,4, or 6. Please use different mapping strategy!");
  }
}

/*--------------------------------------------------------------------------------------*
 *--------------------------------------------------------------------------------------*/
void Discret::Elements::Truss3Scatra::energy(
    const std::map<std::string, std::vector<double>>& ele_state, Teuchos::ParameterList& params,
    Core::LinAlg::SerialDenseVector& intenergy)
{
  // safety check
  if (material()->material_type() != Core::Materials::m_linelast1D_growth and
      material()->material_type() != Core::Materials::m_linelast1D)
    FOUR_C_THROW("only linear elastic growth material supported for truss element");

  switch (material()->material_type())
  {
    case Core::Materials::m_linelast1D:
    {
      Truss3::energy(ele_state, params, intenergy);
      break;
    }
    case Core::Materials::m_linelast1D_growth:
    {
      Core::LinAlg::Matrix<6, 1> curr_nodal_coords;
      Core::LinAlg::Matrix<6, 6> dtruss_disp_du;
      Core::LinAlg::Matrix<6, 1> dN_dx;
      Core::LinAlg::Matrix<2, 1> nodal_concentration;

      prep_calc_internal_force_stiff_tot_lag_scatra(
          ele_state, curr_nodal_coords, dtruss_disp_du, dN_dx, nodal_concentration);

      // get data from input
      const auto* growth_mat = static_cast<const Mat::LinElast1DGrowth*>(material().get());

      // get Gauss rule
      auto gauss_points = Core::FE::IntegrationPoints1D(my_gauss_rule(2, gaussexactintegration));

      // internal energy
      for (int j = 0; j < gauss_points.nquad; ++j)
      {
        const double dx_dxi = lrefe_ / 2.0;
        const double int_fac = dx_dxi * gauss_points.qwgt[j] * crosssec_;

        const double c_GP =
            project_scalar_to_gauss_point(gauss_points.qxg[j][0], nodal_concentration);

        eint_ = growth_mat->evaluate_elastic_energy(curr_length(curr_nodal_coords) / lrefe_, c_GP) *
                int_fac;
      }
      break;
    }
    default:
    {
      FOUR_C_THROW("Material type is not supported");
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
