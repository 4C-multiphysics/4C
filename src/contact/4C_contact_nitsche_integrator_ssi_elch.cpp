// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_integrator_ssi_elch.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_cell_type_traits.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_mat_electrode.hpp"
#include "4C_scatra_ele_boundary_calc_elch_electrode_utils.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_elch.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_solid_3D_ele_calc_lib.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


namespace
{
  template <Core::FE::CellType celltype, int prob_dim>
  Core::LinAlg::Matrix<Core::FE::num_nodes<celltype>, prob_dim> evaluate_nodal_coordinates(
      Core::Nodes::Node** nodes)
  {
    Core::LinAlg::Matrix<Core::FE::num_nodes<celltype>, prob_dim> xrefe(true);
    for (auto i = 0; i < Core::FE::num_nodes<celltype>; ++i)
    {
      const auto& x = nodes[i]->x();
      for (auto dim = 0; dim < prob_dim; ++dim) xrefe(i, dim) = x[dim];
    }

    return xrefe;
  }
}  // namespace

template <int d>
struct CONTACT::IntegratorNitscheSsiElch::ElementDataBundle
{
  Mortar::Element* element;
  double* xi;
  const Core::LinAlg::Matrix<d, 1>* gp_normal;
  const Core::LinAlg::SerialDenseVector* shape_funct;
  const Core::LinAlg::SerialDenseMatrix* shape_deriv;
  const std::vector<Core::Gen::Pairedvector<int, double>>* d_xi_dd;
};

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::IntegratorNitscheSsiElch::IntegratorNitscheSsiElch(
    Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
    : IntegratorNitscheSsi(params, eletype, comm)
{
  if (std::abs(theta_) > 1.0e-16)
    FOUR_C_THROW("SSI Elch Contact just implemented Adjoint free ...");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitscheSsiElch::integrate_gp_3d(Mortar::Element& sele,
    Mortar::Element& mele, Core::LinAlg::SerialDenseVector& sval,
    Core::LinAlg::SerialDenseVector& lmval, Core::LinAlg::SerialDenseVector& mval,
    Core::LinAlg::SerialDenseMatrix& sderiv, Core::LinAlg::SerialDenseMatrix& mderiv,
    Core::LinAlg::SerialDenseMatrix& lmderiv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi)
{
  gpts_forces<3>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt,
      gap, deriv_gap, normal, dnmap_unit, sxi, mxi);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::gpts_forces(Mortar::Element& slave_ele,
    Mortar::Element& master_ele, const Core::LinAlg::SerialDenseVector& slave_shape,
    const Core::LinAlg::SerialDenseMatrix& slave_shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_slave_xi_dd,
    const Core::LinAlg::SerialDenseVector& master_shape,
    const Core::LinAlg::SerialDenseMatrix& master_shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_master_xi_dd, const double jac,
    const Core::Gen::Pairedvector<int, double>& d_jac_dd, const double gp_wgt, const double gap,
    const Core::Gen::Pairedvector<int, double>& d_gap_dd, const double* gp_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd, double* slave_xi,
    double* master_xi)
{
  if (slave_ele.owner() != Core::Communication::my_mpi_rank(Comm_)) return;

  static const bool do_fast_checks = true;
  // first rough check
  if (do_fast_checks)
  {
    if ((std::abs(theta_) < 1.0e-16) and
        (gap > std::max(slave_ele.max_edge_size(), master_ele.max_edge_size())))
      return;
  }

  FOUR_C_ASSERT(dim == IntegratorNitscheSsiElch::n_dim(), "dimension inconsistency");

  // calculate normals and derivatives
  const Core::LinAlg::Matrix<dim, 1> normal(gp_normal, true);
  Core::LinAlg::Matrix<dim, 1> slave_normal, master_normal;
  std::vector<Core::Gen::Pairedvector<int, double>> d_slave_normal_dd(0, 0);
  std::vector<Core::Gen::Pairedvector<int, double>> d_master_normal_dd(0, 0);
  slave_ele.compute_unit_normal_at_xi(slave_xi, slave_normal.data());
  master_ele.compute_unit_normal_at_xi(master_xi, master_normal.data());
  slave_ele.deriv_unit_normal_at_xi(slave_xi, d_slave_normal_dd);
  master_ele.deriv_unit_normal_at_xi(master_xi, d_master_normal_dd);

  double pen = ppn_;
  double pet = ppt_;
  double nitsche_wgt_slave(0.0), nitsche_wgt_master(0.0);

  CONTACT::Utils::nitsche_weights_and_scaling(
      slave_ele, master_ele, nit_wgt_, dt_, nitsche_wgt_slave, nitsche_wgt_master, pen, pet);

  double cauchy_nn_weighted_average(0.0);
  Core::Gen::Pairedvector<int, double> d_cauchy_nn_weighted_average_dd(
      slave_ele.num_node() * 3 * 12 + slave_ele.mo_data().parent_disp().size() +
      master_ele.mo_data().parent_disp().size());
  Core::Gen::Pairedvector<int, double> d_cauchy_nn_weighted_average_ds(
      slave_ele.mo_data().parent_scalar_dof().size() +
      master_ele.mo_data().parent_scalar_dof().size());

  // evaluate cauchy stress components and derivatives
  so_ele_cauchy<dim>(slave_ele, slave_xi, d_slave_xi_dd, gp_wgt, slave_normal, d_slave_normal_dd,
      normal, d_gp_normal_dd, nitsche_wgt_slave, cauchy_nn_weighted_average,
      d_cauchy_nn_weighted_average_dd, d_cauchy_nn_weighted_average_ds);
  so_ele_cauchy<dim>(master_ele, master_xi, d_master_xi_dd, gp_wgt, master_normal,
      d_master_normal_dd, normal, d_gp_normal_dd, -nitsche_wgt_master, cauchy_nn_weighted_average,
      d_cauchy_nn_weighted_average_dd, d_cauchy_nn_weighted_average_ds);

  const double cauchy_nn_average_pen_gap = cauchy_nn_weighted_average + pen * gap;
  Core::Gen::Pairedvector<int, double> d_cauchy_nn_average_pen_gap_dd(
      d_cauchy_nn_weighted_average_dd.size() + d_gap_dd.size());
  for (const auto& p : d_cauchy_nn_weighted_average_dd)
    d_cauchy_nn_average_pen_gap_dd[p.first] += p.second;
  for (const auto& p : d_gap_dd) d_cauchy_nn_average_pen_gap_dd[p.first] += pen * p.second;

  if (cauchy_nn_average_pen_gap < 0.0)
  {
    // test in normal contact direction
    integrate_test<dim>(-1.0, slave_ele, slave_shape, slave_shape_deriv, d_slave_xi_dd, jac,
        d_jac_dd, gp_wgt, cauchy_nn_average_pen_gap, d_cauchy_nn_average_pen_gap_dd,
        d_cauchy_nn_weighted_average_ds, normal, d_gp_normal_dd);
    if (!two_half_pass_)
    {
      integrate_test<dim>(+1.0, master_ele, master_shape, master_shape_deriv, d_master_xi_dd, jac,
          d_jac_dd, gp_wgt, cauchy_nn_average_pen_gap, d_cauchy_nn_average_pen_gap_dd,
          d_cauchy_nn_weighted_average_ds, normal, d_gp_normal_dd);
    }

    ElementDataBundle<dim> electrode_quantities, electrolyte_quantities;
    bool slave_is_electrode(true);
    assign_electrode_and_electrolyte_quantities<dim>(slave_ele, slave_xi, slave_shape,
        slave_shape_deriv, slave_normal, d_slave_xi_dd, master_ele, master_xi, master_shape,
        master_shape_deriv, master_normal, d_master_xi_dd, slave_is_electrode, electrode_quantities,
        electrolyte_quantities);

    // integrate the scatra-scatra interface condition
    integrate_ssi_interface_condition<dim>(
        slave_is_electrode, jac, d_jac_dd, gp_wgt, electrode_quantities, electrolyte_quantities);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::integrate_test(const double fac, Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape,
    const Core::LinAlg::SerialDenseMatrix& shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd, const double jac,
    const Core::Gen::Pairedvector<int, double>& d_jac_dd, const double wgt, const double test_val,
    const Core::Gen::Pairedvector<int, double>& d_test_val_dd,
    const Core::Gen::Pairedvector<int, double>& d_test_val_ds,
    const Core::LinAlg::Matrix<dim, 1>& normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_normal_dd)
{
  if (std::abs(fac) < 1.0e-16) return;

  CONTACT::IntegratorNitsche::integrate_test<dim>(fac, ele, shape, shape_deriv, d_xi_dd, jac,
      d_jac_dd, wgt, test_val, d_test_val_dd, normal, d_normal_dd);

  for (const auto& d_testval_ds : d_test_val_ds)
  {
    double* row = ele.get_nitsche_container().kde(d_testval_ds.first);
    for (int s = 0; s < ele.num_node(); ++s)
    {
      for (int d = 0; d < dim; ++d)
      {
        row[Core::FE::get_parent_node_number_from_face_node_number(
                ele.parent_element()->shape(), ele.face_parent_number(), s) *
                dim +
            d] -= fac * jac * wgt * d_testval_ds.second * normal(d) * shape(s);
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double CONTACT::IntegratorNitscheSsiElch::calculate_det_f_of_parent_element(
    const ElementDataBundle<dim>& electrode_quantities)
{
  auto* electrode_ele = electrode_quantities.element;
  const auto xi_parent = Core::FE::calculate_parent_gp_from_face_element_data<dim>(
      electrode_quantities.xi, electrode_ele);

  // calculate defgrad based on element discretization type
  const Core::LinAlg::Matrix<dim, dim> defgrd = Core::FE::cell_type_switch<
      Core::FE::CelltypeSequence<Core::FE::CellType::hex8, Core::FE::CellType::tet4>>(
      electrode_ele->parent_element()->shape(),
      [&](auto celltype_t)
      {
        constexpr Core::FE::CellType celltype = celltype_t();

        const Discret::Elements::ElementNodes<celltype> element_nodes =
            Discret::Elements::evaluate_element_nodes<celltype>(
                *electrode_ele->parent_element(), electrode_ele->mo_data().parent_disp());

        const Discret::Elements::ShapeFunctionsAndDerivatives<celltype> shape_functions =
            evaluate_shape_functions_and_derivs<celltype>(xi_parent, element_nodes);

        const Discret::Elements::JacobianMapping<celltype> jacobian_mapping =
            Discret::Elements::evaluate_jacobian_mapping(shape_functions, element_nodes);

        return Discret::Elements::evaluate_deformation_gradient(jacobian_mapping, element_nodes);
      });

  return defgrd.determinant();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::calculate_spatial_derivative_of_det_f(const double detF,
    const ElementDataBundle<dim>& electrode_quantities,
    Core::Gen::Pairedvector<int, double>& d_detF_dd)
{
  auto* electrode_ele = electrode_quantities.element;
  switch (electrode_ele->shape())
  {
    case Core::FE::CellType::quad4:
    {
      calculate_spatial_derivative_of_det_f<Core::FE::CellType::quad4, dim>(
          detF, electrode_quantities, d_detF_dd);
      break;
    }
    case Core::FE::CellType::tri3:
    {
      calculate_spatial_derivative_of_det_f<Core::FE::CellType::tri3, dim>(
          detF, electrode_quantities, d_detF_dd);
      break;
    }
    default:
    {
      FOUR_C_THROW("Not implemented for discretization type: {}!", electrode_ele->shape());
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int dim>
void CONTACT::IntegratorNitscheSsiElch::calculate_spatial_derivative_of_det_f(const double detF,
    const ElementDataBundle<dim>& electrode_quantities,
    Core::Gen::Pairedvector<int, double>& d_detF_dd)
{
  auto electrode_ele = electrode_quantities.element;

  const int num_ele_nodes = Core::FE::num_nodes<distype>;
  const int ele_dim = Core::FE::dim<distype>;

  FOUR_C_ASSERT(num_ele_nodes == electrode_ele->num_node(),
      "Number of nodes is not matching discretization type");

  Core::LinAlg::Matrix<num_ele_nodes, dim> xyze =
      evaluate_nodal_coordinates<distype, dim>(electrode_ele->nodes());

  static Core::LinAlg::Matrix<ele_dim, num_ele_nodes> deriv;
  for (auto i = 0; i < num_ele_nodes; ++i)
  {
    const auto parent_nodeid = Core::FE::get_parent_node_number_from_face_node_number(
        electrode_ele->parent_element()->shape(), electrode_ele->face_parent_number(), i);
    for (auto j = 0; j < dim; ++j)
      xyze(i, j) += electrode_ele->mo_data().parent_disp()[parent_nodeid * dim + j];

    for (auto k = 0; k < ele_dim; ++k) deriv(k, i) = (*electrode_quantities.shape_deriv)(i, k);
  }

  static Core::LinAlg::Matrix<dim, num_ele_nodes> derxy;
  Core::FE::evaluate_shape_function_spatial_derivative_in_prob_dim<distype, dim>(
      derxy, deriv, xyze, *electrode_quantities.gp_normal);

  d_detF_dd.resize(electrode_ele->num_node() * dim);
  d_detF_dd.clear();

  // d_detF_dd = d_detF_dF * d_F_dd = detF F^{-T} * d_F_dd  = detF * d_N_dx
  for (auto i = 0; i < electrode_ele->num_node(); ++i)
  {
    for (auto j = 0; j < dim; ++j)
    {
      d_detF_dd[i * dim + j] = detF * derxy(j, i);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::integrate_ssi_interface_condition(
    const bool slave_is_electrode, const double jac,
    const Core::Gen::Pairedvector<int, double>& d_jac_dd, const double wgt,
    const ElementDataBundle<dim>& electrode_quantities,
    const ElementDataBundle<dim>& electrolyte_quantities)
{
  if (slave_is_electrode)
  {
    if (electrode_quantities.element->mo_data().parent_scalar_dof().empty()) return;
    if (electrolyte_quantities.element->mo_data().parent_scalar_dof().empty())
      FOUR_C_THROW("Something went wrong!");
  }
  else
  {
    if (electrolyte_quantities.element->mo_data().parent_scalar_dof().empty()) return;
    if (electrode_quantities.element->mo_data().parent_scalar_dof().empty())
      FOUR_C_THROW("Something went wrong!");
  }

  // get the scatra-scatra interface kinetic model
  const int kinetic_model = get_scatra_ele_parameter_boundary()->kinetic_model();

  // perform integration dependent on scatra-scatra interface kinetic model
  switch (kinetic_model)
  {
    case Inpar::S2I::kinetics_butlervolmer:
    case Inpar::S2I::kinetics_butlervolmerreduced:
    {
      // access material of parent element for ELCH simulations
      std::shared_ptr<const Mat::Electrode> electrode_material =
          std::dynamic_pointer_cast<const Mat::Electrode>(
              electrode_quantities.element->parent_element()->material(1));

      // get the relevant parameter
      const double faraday =
          Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
      const double frt = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->frt();
      const double kr = get_scatra_ele_parameter_boundary()->charge_transfer_constant();
      const double alphaa = get_scatra_ele_parameter_boundary()->alphadata();
      const double alphac = get_scatra_ele_parameter_boundary()->alpha_c();

      // calculate the electrode side concentration, potential and their derivatives at the current
      // gauss point
      double electrode_conc(0.0), electrode_pot(0.0);
      Core::Gen::Pairedvector<int, double> d_electrode_conc_dc(0), d_electrode_pot_dpot(0),
          d_electrode_conc_dd(0), d_electrode_pot_dd(0), d_detF_dd(0);
      setup_gp_elch_properties<dim>(electrode_quantities, electrode_conc, electrode_pot,
          d_electrode_conc_dc, d_electrode_conc_dd, d_electrode_pot_dpot, d_electrode_pot_dd);

      // calculate the electrolyte side concentration, potential and their derivatives at the
      // current gauss point
      double electrolyte_conc(0.0), electrolyte_pot(0.0);
      Core::Gen::Pairedvector<int, double> d_electrolyte_conc_dc(0), d_electrolyte_pot_dpot(0),
          d_electrolyte_conc_dd(0), d_electrolyte_pot_dd(0);
      setup_gp_elch_properties<dim>(electrolyte_quantities, electrolyte_conc, electrolyte_pot,
          d_electrolyte_conc_dc, d_electrolyte_conc_dd, d_electrolyte_pot_dpot,
          d_electrolyte_pot_dd);

      // extract saturation value of intercalated lithium concentration from electrode material
      const double cmax = electrode_material->c_max();
      if (cmax < 1.0e-12)
        FOUR_C_THROW("Saturation value c_max of intercalated lithium concentration is too small!");

      const double detF = calculate_det_f_of_parent_element(electrode_quantities);

      calculate_spatial_derivative_of_det_f(detF, electrode_quantities, d_detF_dd);

      const double epd =
          electrode_material->compute_open_circuit_potential(electrode_conc, faraday, frt, detF);

      // skip further computation in case equilibrium electric potential difference is outside
      // physically meaningful range
      if (not std::isinf(epd))
      {
        const double d_epd_dc =
            electrode_material->compute_d_open_circuit_potential_d_concentration(
                electrode_conc, faraday, frt, detF);

        // Butler-Volmer exchange mass flux density
        const double j0(kinetic_model == Inpar::S2I::kinetics_butlervolmerreduced
                            ? kr
                            : kr * std::pow(electrolyte_conc, alphaa) *
                                  std::pow(cmax - electrode_conc, alphaa) *
                                  std::pow(electrode_conc, alphac));

        // electrode-electrolyte overpotential at integration point
        const double eta = electrode_pot - electrolyte_pot - epd;

        // exponential Butler-Volmer terms
        const double expterm1 = std::exp(alphaa * frt * eta);
        const double expterm2 = std::exp(-alphac * frt * eta);
        const double expterm = expterm1 - expterm2;

        // calculate Butler-Volmer mass flux density
        const double j = j0 * expterm;

        // initialize a dummy resistance as the method below requires a resistance which is not
        // relevant in this case
        const double dummyresistance(0.0);
        // define flux linearization terms
        double dj_dc_electrode(0.0), dj_dc_electrolyte(0.0), dj_dpot_electrode(0.0),
            dj_dpot_electrolyte(0.0);
        // calculate flux linearizations
        Discret::Elements::calculate_butler_volmer_elch_linearizations(kinetic_model, j0, frt,
            d_epd_dc, alphaa, alphac, dummyresistance, expterm1, expterm2, kr, faraday,
            electrolyte_conc, electrode_conc, cmax, eta, dj_dc_electrode, dj_dc_electrolyte,
            dj_dpot_electrode, dj_dpot_electrolyte);

        // derivative of flux w.r.t. OCP is the same value as w.r.t. electrolyte potential
        const double dj_depd = dj_dpot_electrolyte;

        const double depd_ddetF = electrode_material->compute_d_open_circuit_potential_d_det_f(
            electrode_conc, faraday, frt, detF);
        const double dj_ddetF = dj_depd * depd_ddetF;

        // initialize derivatives of flux w.r.t. electrochemistry dofs
        Core::Gen::Pairedvector<int, double> dj_delch(
            d_electrode_conc_dc.size() + d_electrolyte_conc_dc.size() +
            d_electrode_pot_dpot.size() + d_electrolyte_pot_dpot.size());
        for (const auto& [d_electrodeconc_dc_dof, d_electrodeconc_dc_val] : d_electrode_conc_dc)
          dj_delch[d_electrodeconc_dc_dof] += dj_dc_electrode * d_electrodeconc_dc_val;
        for (const auto& [d_electrolyteconc_dc_dof, d_electrolyteconc_dc_val] :
            d_electrolyte_conc_dc)
          dj_delch[d_electrolyteconc_dc_dof] += dj_dc_electrolyte * d_electrolyteconc_dc_val;
        for (const auto& [d_electrodepot_dpot_dof, d_electrodepot_dpot_val] : d_electrode_pot_dpot)
          dj_delch[d_electrodepot_dpot_dof] += dj_dpot_electrode * d_electrodepot_dpot_val;
        for (const auto& [d_electrolytepot_dpot_dof, d_electrolytepot_dpot_val] :
            d_electrolyte_pot_dpot)
          dj_delch[d_electrolytepot_dpot_dof] += dj_dpot_electrolyte * d_electrolytepot_dpot_val;

        // initialize derivatives of flux w.r.t. displacements
        Core::Gen::Pairedvector<int, double> dj_dd(
            d_electrode_conc_dd.size() + d_electrode_pot_dd.size() + d_detF_dd.size() +
            d_electrolyte_conc_dd.size() + d_electrolyte_pot_dd.size());

        for (const auto& [d_electrodeconc_dd_dof, d_electrodeconc_dd_val] : d_electrode_conc_dd)
          dj_dd[d_electrodeconc_dd_dof] += dj_dc_electrode * d_electrodeconc_dd_val;
        for (const auto& [d_detF_dd_dof, d_detF_dd_val] : d_detF_dd)
          dj_dd[d_detF_dd_dof] += dj_ddetF * d_detF_dd_val;
        for (const auto& [d_electrodepot_dd_dof, d_electrodepot_dd_val] : d_electrode_pot_dd)
          dj_dd[d_electrodepot_dd_dof] += dj_dpot_electrode * d_electrodepot_dd_val;
        for (const auto& [d_electrolyteconc_dd_dof, d_electrolyteconc_dd_val] :
            d_electrolyte_conc_dd)
          dj_dd[d_electrolyteconc_dd_dof] += dj_dc_electrolyte * d_electrolyteconc_dd_val;
        for (const auto& [d_electrolytepot_dd_dof, d_electrolytepot_dd_val] : d_electrolyte_pot_dd)
          dj_dd[d_electrolytepot_dd_dof] += dj_dpot_electrolyte * d_electrolytepot_dd_val;

        if (!two_half_pass_ or slave_is_electrode)
          integrate_elch_test<dim>(
              1.0, electrode_quantities, jac, d_jac_dd, wgt, j, dj_dd, dj_delch);
        if (!two_half_pass_ or !slave_is_electrode)
          integrate_elch_test<dim>(
              -1.0, electrolyte_quantities, jac, d_jac_dd, wgt, j, dj_dd, dj_delch);
      }

      break;
    }
    case Inpar::S2I::kinetics_nointerfaceflux:
      break;
    default:
    {
      FOUR_C_THROW(
          "Evaluation is not implemented for this scatra-scatra interface kinetic model: {}",
          kinetic_model);
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::integrate_elch_test(const double fac,
    const ElementDataBundle<dim>& ele_data_bundle, const double jac,
    const Core::Gen::Pairedvector<int, double>& d_jac_dd, const double wgt, const double test_val,
    const Core::Gen::Pairedvector<int, double>& d_test_val_dd,
    const Core::Gen::Pairedvector<int, double>& d_test_val_ds)
{
  Mortar::Element& ele = *ele_data_bundle.element;
  const Core::LinAlg::SerialDenseVector& shape_func = *ele_data_bundle.shape_funct;
  const Core::LinAlg::SerialDenseMatrix& shape_deriv = *ele_data_bundle.shape_deriv;
  const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd = *ele_data_bundle.d_xi_dd;

  // get time integration factors
  const double time_fac = get_scatra_ele_parameter_tim_int()->time_fac();
  const double time_fac_rhs = get_scatra_ele_parameter_tim_int()->time_fac_rhs();

  // get number of electrons per charge transfer reaction
  const int num_electrons = get_scatra_ele_parameter_boundary()->num_electrons();

  // prepare the RHS integration value
  const double val = fac * jac * wgt * test_val;

  // setup derivative of RHS integration value w.r.t. the displacement dofs
  Core::Gen::Pairedvector<int, double> d_val_dd(d_jac_dd.size() + d_test_val_dd.size());
  for (const auto& djac_dd : d_jac_dd)
    d_val_dd[djac_dd.first] += fac * djac_dd.second * wgt * test_val;
  for (const auto& d_testval_dd : d_test_val_dd)
    d_val_dd[d_testval_dd.first] += fac * jac * wgt * d_testval_dd.second;

  for (int s = 0; s < ele.num_node(); ++s)
  {
    const int slave_parent = Core::FE::get_parent_node_number_from_face_node_number(
        ele.parent_element()->shape(), ele.face_parent_number(), s);
    const int slave_parent_conc = slave_parent * numdofpernode_;
    const int slave_parent_pot = slave_parent_conc + 1;

    *ele.get_nitsche_container().rhs_e(slave_parent_conc) -= shape_func(s) * val * time_fac_rhs;
    *ele.get_nitsche_container().rhs_e(slave_parent_pot) -=
        num_electrons * shape_func(s) * val * time_fac_rhs;

    for (const auto& d_testval_ds : d_test_val_ds)
    {
      double* row = ele.get_nitsche_container().kee(d_testval_ds.first);

      row[slave_parent_conc] += shape_func(s) * fac * jac * wgt * d_testval_ds.second * time_fac;
      row[slave_parent_pot] +=
          num_electrons * shape_func(s) * fac * jac * wgt * d_testval_ds.second * time_fac;
    }

    for (const auto& dval_dd : d_val_dd)
    {
      double* row = ele.get_nitsche_container().ked(dval_dd.first);

      row[slave_parent_conc] += shape_func(s) * dval_dd.second * time_fac;
      row[slave_parent_pot] += num_electrons * shape_func(s) * dval_dd.second * time_fac;
    }

    for (int e = 0; e < dim - 1; ++e)
    {
      for (const auto& d_xi_dd_e : d_xi_dd[e])
      {
        double* row = ele.get_nitsche_container().ked(d_xi_dd_e.first);

        row[slave_parent_conc] += shape_deriv(s, e) * d_xi_dd_e.second * val * time_fac;
        row[slave_parent_pot] +=
            num_electrons * shape_deriv(s, e) * d_xi_dd_e.second * val * time_fac;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::setup_gp_elch_properties(
    const ElementDataBundle<dim>& ele_data_bundle, double& gp_conc, double& gp_pot,
    Core::Gen::Pairedvector<int, double>& d_conc_dc,
    Core::Gen::Pairedvector<int, double>& d_conc_dd,
    Core::Gen::Pairedvector<int, double>& d_pot_dpot,
    Core::Gen::Pairedvector<int, double>& d_pot_dd)
{
  Mortar::Element& ele = *ele_data_bundle.element;
  const Core::LinAlg::SerialDenseVector& shape_func = *ele_data_bundle.shape_funct;
  const Core::LinAlg::SerialDenseMatrix& shape_deriv = *ele_data_bundle.shape_deriv;
  const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd = *ele_data_bundle.d_xi_dd;

  // resize and clear derivative vectors
  d_conc_dc.resize(shape_func.length());
  d_pot_dpot.resize(shape_func.length());
  d_conc_dc.clear();
  d_pot_dpot.clear();
  std::size_t deriv_size = 0;
  for (int i = 0; i < dim - 1; ++i) deriv_size += d_xi_dd.at(i).size();
  d_conc_dd.resize(deriv_size);
  d_pot_dd.resize(deriv_size);
  d_conc_dd.clear();
  d_pot_dd.clear();

  // calculate the nodal concentrations, potentials and derivatives w.r.t electrochemistry dofs
  Core::LinAlg::SerialDenseVector ele_conc(shape_func.length());
  Core::LinAlg::SerialDenseVector ele_pot(shape_func.length());
  for (int i = 0; i < ele.num_node(); ++i)
  {
    const int iparent = Core::FE::get_parent_node_number_from_face_node_number(
        ele.parent_element()->shape(), ele.face_parent_number(), i);
    const int iparent_conc = iparent * numdofpernode_;
    const int iparent_pot = iparent_conc + 1;

    ele_conc(i) = ele.mo_data().parent_scalar().at(iparent_conc);
    ele_pot(i) = ele.mo_data().parent_scalar().at(iparent_pot);

    d_conc_dc[ele.mo_data().parent_scalar_dof().at(iparent_conc)] = shape_func(i);
    d_pot_dpot[ele.mo_data().parent_scalar_dof().at(iparent_pot)] = shape_func(i);
  }

  // calculate the Gauss point concentration and potential
  gp_conc = shape_func.dot(ele_conc);
  gp_pot = shape_func.dot(ele_pot);

  // calculate the nodal concentrations, potentials and derivatives w.r.t displacement dofs
  for (int i = 0; i < dim - 1; ++i)
  {
    for (const auto& d_xi_dd_i : d_xi_dd.at(i))
    {
      double& dc_dd = d_conc_dd[d_xi_dd_i.first];
      double& dpot_dd = d_pot_dd[d_xi_dd_i.first];
      for (int n = 0; n < ele.num_node(); ++n)
      {
        dc_dd += ele_conc(n) * shape_deriv(n, i) * d_xi_dd_i.second;
        dpot_dd += ele_pot(n) * shape_deriv(n, i) * d_xi_dd_i.second;
      }
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::so_ele_cauchy(Mortar::Element& mortar_ele, double* gp_coord,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_coord_dd, const double gp_wgt,
    const Core::LinAlg::Matrix<dim, 1>& gp_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd,
    const Core::LinAlg::Matrix<dim, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_test_dir_dd,
    const double nitsche_wgt, double& cauchy_nt_wgt,
    Core::Gen::Pairedvector<int, double>& d_cauchy_nt_dd,
    Core::Gen::Pairedvector<int, double>& d_cauchy_nt_de)
{
  Core::LinAlg::SerialDenseMatrix d_sigma_nt_de;

  so_ele_cauchy_struct<dim>(mortar_ele, gp_coord, d_gp_coord_dd, gp_wgt, gp_normal, d_gp_normal_dd,
      test_dir, d_test_dir_dd, nitsche_wgt, cauchy_nt_wgt, d_cauchy_nt_dd, &d_sigma_nt_de);

  if (!mortar_ele.mo_data().parent_scalar().empty())
  {
    for (int i = 0; i < mortar_ele.parent_element()->num_node(); ++i)
      d_cauchy_nt_de[mortar_ele.mo_data().parent_scalar_dof().at(i * numdofpernode_)] +=
          nitsche_wgt * d_sigma_nt_de(i, 0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsiElch::assign_electrode_and_electrolyte_quantities(
    Mortar::Element& slave_ele, double* slave_xi,
    const Core::LinAlg::SerialDenseVector& slave_shape,
    const Core::LinAlg::SerialDenseMatrix& slave_shape_deriv,
    const Core::LinAlg::Matrix<dim, 1>& slave_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_slave_xi_dd,
    Mortar::Element& master_ele, double* master_xi,
    const Core::LinAlg::SerialDenseVector& master_shape,
    const Core::LinAlg::SerialDenseMatrix& master_shape_deriv,
    const Core::LinAlg::Matrix<dim, 1>& master_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_master_xi_dd,
    bool& slave_is_electrode, ElementDataBundle<dim>& electrode_quantities,
    ElementDataBundle<dim>& electrolyte_quantities)
{
  std::shared_ptr<const Mat::Electrode> electrode_material =
      std::dynamic_pointer_cast<const Mat::Electrode>(slave_ele.parent_element()->material(1));
  if (electrode_material == nullptr)
  {
    slave_is_electrode = false;

    electrode_material =
        std::dynamic_pointer_cast<const Mat::Electrode>(master_ele.parent_element()->material(1));

    // safety check
    FOUR_C_ASSERT_ALWAYS(electrode_material != nullptr,
        "Something went wrong, neither slave nor master side is electrode material. This is a "
        "fatal error!");
  }
  else
  {
    auto master_also_electrode =
        std::dynamic_pointer_cast<const Mat::Electrode>(master_ele.parent_element()->material(1));

    FOUR_C_ASSERT_ALWAYS(master_also_electrode == nullptr,
        "Both, slave and master side are electrode materials, this should not be the case!");
  }

  if (slave_is_electrode)
  {
    electrode_quantities.element = &slave_ele;
    electrode_quantities.xi = slave_xi;
    electrode_quantities.gp_normal = &slave_normal;
    electrode_quantities.shape_funct = &slave_shape;
    electrode_quantities.shape_deriv = &slave_shape_deriv;
    electrode_quantities.d_xi_dd = &d_slave_xi_dd;

    electrolyte_quantities.element = &master_ele;
    electrolyte_quantities.xi = master_xi;
    electrolyte_quantities.gp_normal = &master_normal;
    electrolyte_quantities.shape_funct = &master_shape;
    electrolyte_quantities.shape_deriv = &master_shape_deriv;
    electrolyte_quantities.d_xi_dd = &d_master_xi_dd;
  }
  else
  {
    electrolyte_quantities.element = &slave_ele;
    electrolyte_quantities.xi = slave_xi;
    electrolyte_quantities.gp_normal = &slave_normal;
    electrolyte_quantities.shape_funct = &slave_shape;
    electrolyte_quantities.shape_deriv = &slave_shape_deriv;
    electrolyte_quantities.d_xi_dd = &d_slave_xi_dd;

    electrode_quantities.element = &master_ele;
    electrode_quantities.xi = master_xi;
    electrode_quantities.gp_normal = &master_normal;
    electrode_quantities.shape_funct = &master_shape;
    electrode_quantities.shape_deriv = &master_shape_deriv;
    electrode_quantities.d_xi_dd = &d_master_xi_dd;
  }
}
FOUR_C_NAMESPACE_CLOSE
