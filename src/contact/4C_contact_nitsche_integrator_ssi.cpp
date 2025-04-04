// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_integrator_ssi.hpp"

#include "4C_contact_nitsche_utils.hpp"
#include "4C_fem_general_cell_type.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_solid_scatra_3D_ele.hpp"
#include "4C_solid_scatra_3D_ele_calc_lib_nitsche.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::IntegratorNitscheSsi::IntegratorNitscheSsi(
    Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
    : IntegratorNitsche(params, eletype, comm),
      scatraparamstimint_(Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")),
      scatraparamsboundary_(Discret::Elements::ScaTraEleParameterBoundary::instance("scatra"))
{
  if (std::abs(theta_) > 1.0e-16) FOUR_C_THROW("SSI Contact just implemented Adjoint free ...");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitscheSsi::integrate_gp_3d(Mortar::Element& sele, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
    Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
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
void CONTACT::IntegratorNitscheSsi::integrate_gp_2d(Mortar::Element& sele, Mortar::Element& mele,
    Core::LinAlg::SerialDenseVector& sval, Core::LinAlg::SerialDenseVector& lmval,
    Core::LinAlg::SerialDenseVector& mval, Core::LinAlg::SerialDenseMatrix& sderiv,
    Core::LinAlg::SerialDenseMatrix& mderiv, Core::LinAlg::SerialDenseMatrix& lmderiv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* sxi, double* mxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
    std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi)
{
  FOUR_C_THROW("2D is not implemented!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsi::gpts_forces(Mortar::Element& slave_ele,
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

  FOUR_C_ASSERT(dim == n_dim(), "dimension inconsistency");

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

    // integrate the scatra-scatra interface condition
    integrate_ssi_interface_condition<dim>(slave_ele, slave_shape, slave_shape_deriv, d_slave_xi_dd,
        master_ele, master_shape, master_shape_deriv, d_master_xi_dd, cauchy_nn_average_pen_gap,
        d_cauchy_nn_weighted_average_dd, d_cauchy_nn_weighted_average_ds, jac, d_jac_dd, gp_wgt);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsi::so_ele_cauchy(Mortar::Element& mortar_ele, double* gp_coord,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_coord_dd, const double gp_wgt,
    const Core::LinAlg::Matrix<dim, 1>& gp_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd,
    const Core::LinAlg::Matrix<dim, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_test_dir_dd,
    const double nitsche_wgt, double& cauchy_nt_wgt,
    Core::Gen::Pairedvector<int, double>& d_cauchy_nt_dd,
    Core::Gen::Pairedvector<int, double>& d_cauchy_nt_ds)
{
  Core::LinAlg::SerialDenseMatrix d_sigma_nt_ds;

  so_ele_cauchy_struct<dim>(mortar_ele, gp_coord, d_gp_coord_dd, gp_wgt, gp_normal, d_gp_normal_dd,
      test_dir, d_test_dir_dd, nitsche_wgt, cauchy_nt_wgt, d_cauchy_nt_dd, &d_sigma_nt_ds);

  if (!mortar_ele.mo_data().parent_scalar().empty())
  {
    for (int i = 0; i < mortar_ele.parent_element()->num_node(); ++i)
      d_cauchy_nt_ds[mortar_ele.mo_data().parent_scalar_dof().at(i)] +=
          nitsche_wgt * d_sigma_nt_ds(i, 0);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsi::so_ele_cauchy_struct(Mortar::Element& mortar_ele,
    double* gp_coord, const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_coord_dd,
    const double gp_wgt, const Core::LinAlg::Matrix<dim, 1>& gp_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd,
    const Core::LinAlg::Matrix<dim, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_test_dir_dd, double nitsche_wgt,
    double& cauchy_nt_wgt, Core::Gen::Pairedvector<int, double>& d_cauchy_nt_dd,
    Core::LinAlg::SerialDenseMatrix* d_sigma_nt_ds)
{
  static Core::LinAlg::Matrix<dim, 1> parent_xi(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<dim, dim> local_to_parent_trafo(Core::LinAlg::Initialization::zero);
  CONTACT::Utils::map_gp_to_parent<dim>(
      mortar_ele, gp_coord, gp_wgt, parent_xi, local_to_parent_trafo);

  // cauchy stress tensor contracted with normal and test direction
  double sigma_nt(0.0);
  Core::LinAlg::SerialDenseMatrix d_sigma_nt_dd;
  static Core::LinAlg::Matrix<dim, 1> d_sigma_nt_dn(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<dim, 1> d_sigma_nt_dt(Core::LinAlg::Initialization::zero);
  static Core::LinAlg::Matrix<dim, 1> d_sigma_nt_dxi(Core::LinAlg::Initialization::zero);

  Discret::Elements::SolidScatraCauchyNDirLinearizations<3> linearizations{};
  linearizations.solid.d_cauchyndir_dd = &d_sigma_nt_dd;
  linearizations.solid.d_cauchyndir_dn = &d_sigma_nt_dn;
  linearizations.solid.d_cauchyndir_ddir = &d_sigma_nt_dt;
  linearizations.solid.d_cauchyndir_dxi = &d_sigma_nt_dxi;

  linearizations.d_cauchyndir_ds = d_sigma_nt_ds;
  sigma_nt = std::invoke(
      [&]()
      {
        auto* solid_scatra_ele =
            dynamic_cast<Discret::Elements::SolidScatra*>(mortar_ele.parent_element());

        FOUR_C_ASSERT_ALWAYS(solid_scatra_ele,
            "Nitsche contact is not implemented for this element (expecting SOLIDSCATRA "
            "element)!");

        return solid_scatra_ele->get_normal_cauchy_stress_at_xi(mortar_ele.mo_data().parent_disp(),
            mortar_ele.mo_data().parent_scalar(), parent_xi, gp_normal, test_dir, linearizations);
      });

  cauchy_nt_wgt += nitsche_wgt * sigma_nt;

  for (int i = 0; i < mortar_ele.parent_element()->num_node() * dim; ++i)
    d_cauchy_nt_dd[mortar_ele.mo_data().parent_dof().at(i)] += nitsche_wgt * d_sigma_nt_dd(i, 0);

  for (int i = 0; i < dim - 1; ++i)
  {
    for (const auto& d_gp_coord_dd_i : d_gp_coord_dd[i])
    {
      for (int k = 0; k < dim; ++k)
      {
        d_cauchy_nt_dd[d_gp_coord_dd_i.first] +=
            nitsche_wgt * d_sigma_nt_dxi(k) * local_to_parent_trafo(k, i) * d_gp_coord_dd_i.second;
      }
    }
  }

  for (int i = 0; i < dim; ++i)
  {
    for (const auto& dn_dd_i : d_gp_normal_dd[i])
      d_cauchy_nt_dd[dn_dd_i.first] += nitsche_wgt * d_sigma_nt_dn(i) * dn_dd_i.second;

    for (const auto& dt_dd_i : d_test_dir_dd[i])
      d_cauchy_nt_dd[dt_dd_i.first] += nitsche_wgt * d_sigma_nt_dt(i) * dt_dd_i.second;
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsi::integrate_test(const double fac, Mortar::Element& ele,
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
    double* row = ele.get_nitsche_container().kds(d_testval_ds.first);
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
void CONTACT::IntegratorNitscheSsi::setup_gp_concentrations(Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape_func,
    const Core::LinAlg::SerialDenseMatrix& shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd, double& gp_conc,
    Core::Gen::Pairedvector<int, double>& d_conc_dc,
    Core::Gen::Pairedvector<int, double>& d_conc_dd)
{
  Core::LinAlg::SerialDenseVector ele_conc(shape_func.length());
  for (int i = 0; i < ele.num_node(); ++i)
    ele_conc(i) =
        ele.mo_data().parent_scalar().at(Core::FE::get_parent_node_number_from_face_node_number(
            ele.parent_element()->shape(), ele.face_parent_number(), i));

  // calculate gp concentration
  gp_conc = shape_func.dot(ele_conc);

  // calculate derivative of concentration w.r.t. concentration
  d_conc_dc.resize(shape_func.length());
  d_conc_dc.clear();
  for (int i = 0; i < ele.num_node(); ++i)
    d_conc_dc[ele.mo_data().parent_scalar_dof().at(
        Core::FE::get_parent_node_number_from_face_node_number(
            ele.parent_element()->shape(), ele.face_parent_number(), i))] = shape_func(i);

  // calculate derivative of concentration w.r.t. displacements
  std::size_t deriv_size = 0;
  for (int i = 0; i < dim - 1; ++i) deriv_size += d_xi_dd.at(i).size();
  d_conc_dd.resize(deriv_size);
  d_conc_dd.clear();
  for (int i = 0; i < dim - 1; ++i)
  {
    for (const auto& d_xi_dd_i : d_xi_dd.at(i))
    {
      for (int n = 0; n < ele.num_node(); ++n)
        d_conc_dd[d_xi_dd_i.first] += ele_conc(n) * shape_deriv(n, i) * d_xi_dd_i.second;
    }
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsi::integrate_ssi_interface_condition(Mortar::Element& slave_ele,
    const Core::LinAlg::SerialDenseVector& slave_shape,
    const Core::LinAlg::SerialDenseMatrix& slave_shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_slave_xi_dd,
    Mortar::Element& master_ele, const Core::LinAlg::SerialDenseVector& master_shape,
    const Core::LinAlg::SerialDenseMatrix& master_shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_master_xi_dd,
    const double cauchy_nn_average_pen_gap,
    const Core::Gen::Pairedvector<int, double>& d_cauchy_nn_weighted_average_dd,
    const Core::Gen::Pairedvector<int, double>& d_cauchy_nn_weighted_average_dc, const double jac,
    const Core::Gen::Pairedvector<int, double>& d_jac_dd, const double wgt)
{
  // do only integrate if there is something to integrate!
  if (slave_ele.mo_data().parent_scalar_dof().empty()) return;
  if (master_ele.mo_data().parent_scalar_dof().empty()) FOUR_C_THROW("This is not allowed!");

  // prepare the slave and master side gauss point concentrations and derivatives w.r.t. the
  // concentration and the displacement
  double slave_conc(0.0), master_conc(0.0);
  Core::Gen::Pairedvector<int, double> d_slave_conc_dc(0), d_master_conc_dc(0), d_slave_conc_dd(0),
      d_master_conc_dd(0);
  setup_gp_concentrations<dim>(slave_ele, slave_shape, slave_shape_deriv, d_slave_xi_dd, slave_conc,
      d_slave_conc_dc, d_slave_conc_dd);
  setup_gp_concentrations<dim>(master_ele, master_shape, master_shape_deriv, d_master_xi_dd,
      master_conc, d_master_conc_dc, d_master_conc_dd);

  // get the scatra-scatra interface condition kinetic model
  const int kinetic_model = get_scatra_ele_parameter_boundary()->kinetic_model();

  double flux;
  Core::Gen::Pairedvector<int, double> dflux_dd;
  Core::Gen::Pairedvector<int, double> dflux_dc;

  // perform integration according to kinetic model
  switch (kinetic_model)
  {
    case Inpar::S2I::kinetics_constperm:
    {
      const double permeability = (*get_scatra_ele_parameter_boundary()->permeabilities())[0];

      // calculate the interface flux
      flux = permeability * (slave_conc - master_conc);

      // initialize derivatives of flux w.r.t. concentrations
      dflux_dc.resize(d_slave_conc_dc.size() + d_master_conc_dc.size());
      for (const auto& p : d_slave_conc_dc) dflux_dc[p.first] += permeability * p.second;
      for (const auto& p : d_master_conc_dc) dflux_dc[p.first] -= permeability * p.second;

      // initialize derivatives of flux w.r.t. displacements
      dflux_dd.resize(d_slave_conc_dd.size() + d_master_conc_dd.size());
      for (const auto& p : d_slave_conc_dd) dflux_dd[p.first] += permeability * p.second;
      for (const auto& p : d_master_conc_dd) dflux_dd[p.first] -= permeability * p.second;

      break;
    }
    case Inpar::S2I::kinetics_linearperm:
    {
      const double permeability = (*get_scatra_ele_parameter_boundary()->permeabilities())[0];

      // calculate the interface flux
      // the minus sign is to obtain the absolute value of the contact forces
      flux = -permeability * cauchy_nn_average_pen_gap * (slave_conc - master_conc);

      // initialize derivatives of flux w.r.t. concentrations
      dflux_dc.resize(d_slave_conc_dc.size() + d_master_conc_dc.size() +
                      d_cauchy_nn_weighted_average_dc.size());

      for (const auto& p : d_slave_conc_dc)
        dflux_dc[p.first] -= permeability * cauchy_nn_average_pen_gap * p.second;
      for (const auto& p : d_master_conc_dc)
        dflux_dc[p.first] += permeability * cauchy_nn_average_pen_gap * p.second;

      for (const auto& p : d_cauchy_nn_weighted_average_dc)
        dflux_dc[p.first] -= permeability * (slave_conc - master_conc) * p.second;

      // initialize derivatives of flux w.r.t. displacements
      dflux_dd.resize(d_slave_conc_dd.size() + d_master_conc_dd.size() +
                      d_cauchy_nn_weighted_average_dd.size());

      for (const auto& p : d_slave_conc_dd)
        dflux_dd[p.first] -= permeability * cauchy_nn_average_pen_gap * p.second;
      for (const auto& p : d_master_conc_dd)
        dflux_dd[p.first] += permeability * cauchy_nn_average_pen_gap * p.second;

      for (const auto& p : d_cauchy_nn_weighted_average_dd)
        dflux_dd[p.first] -= permeability * (slave_conc - master_conc) * p.second;

      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Integration can not be performed as kinetic model of scatra-scatra interface condition "
          "is not recognized: {}",
          kinetic_model);

      break;
    }
  }

  integrate_scatra_test<dim>(-1.0, slave_ele, slave_shape, slave_shape_deriv, d_slave_xi_dd, jac,
      d_jac_dd, wgt, flux, dflux_dd, dflux_dc);
  if (!two_half_pass_)
  {
    integrate_scatra_test<dim>(1.0, master_ele, master_shape, master_shape_deriv, d_master_xi_dd,
        jac, d_jac_dd, wgt, flux, dflux_dd, dflux_dc);
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheSsi::integrate_scatra_test(const double fac, Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape_func,
    const Core::LinAlg::SerialDenseMatrix& shape_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_xi_dd, const double jac,
    const Core::Gen::Pairedvector<int, double>& d_jac_dd, const double wgt, const double test_val,
    const Core::Gen::Pairedvector<int, double>& d_test_val_dd,
    const Core::Gen::Pairedvector<int, double>& d_test_val_ds)
{
  // get time integration factors
  const double time_fac = get_scatra_ele_parameter_tim_int()->time_fac();
  const double time_fac_rhs = get_scatra_ele_parameter_tim_int()->time_fac_rhs();

  const double val = fac * jac * wgt * test_val;

  for (int s = 0; s < ele.num_node(); ++s)
  {
    *(ele.get_nitsche_container().rhs_s(Core::FE::get_parent_node_number_from_face_node_number(
        ele.parent_element()->shape(), ele.face_parent_number(), s))) +=
        time_fac_rhs * val * shape_func(s);
  }

  for (const auto& d_testval_ds : d_test_val_ds)
  {
    double* row = ele.get_nitsche_container().kss(d_testval_ds.first);
    for (int s = 0; s < ele.num_node(); ++s)
    {
      row[Core::FE::get_parent_node_number_from_face_node_number(
          ele.parent_element()->shape(), ele.face_parent_number(), s)] -=
          time_fac * fac * jac * wgt * d_testval_ds.second * shape_func(s);
    }
  }

  Core::Gen::Pairedvector<int, double> d_val_dd(d_jac_dd.size() + d_test_val_dd.size());
  for (const auto& djac_dd : d_jac_dd)
    d_val_dd[djac_dd.first] += fac * djac_dd.second * wgt * test_val;
  for (const auto& d_testval_dd : d_test_val_dd)
    d_val_dd[d_testval_dd.first] += fac * jac * wgt * d_testval_dd.second;

  for (const auto& dval_dd : d_val_dd)
  {
    double* row = ele.get_nitsche_container().ksd(dval_dd.first);
    for (int s = 0; s < ele.num_node(); ++s)
      row[Core::FE::get_parent_node_number_from_face_node_number(ele.parent_element()->shape(),
          ele.face_parent_number(), s)] -= time_fac * dval_dd.second * shape_func(s);
  }

  for (int e = 0; e < dim - 1; ++e)
  {
    for (const auto& d_xi_dd_e : d_xi_dd[e])
    {
      double* row = ele.get_nitsche_container().ksd(d_xi_dd_e.first);
      for (int s = 0; s < ele.num_node(); ++s)
        row[Core::FE::get_parent_node_number_from_face_node_number(ele.parent_element()->shape(),
            ele.face_parent_number(), s)] -= time_fac * val * shape_deriv(s, e) * d_xi_dd_e.second;
    }
  }
}

template void CONTACT::IntegratorNitscheSsi::so_ele_cauchy_struct<3>(Mortar::Element& mortar_ele,
    double* gp_coord, const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_coord_dd,
    const double gp_wgt, const Core::LinAlg::Matrix<3, 1>& gp_normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_gp_normal_dd,
    const Core::LinAlg::Matrix<3, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_test_dir_dd, double nitsche_wgt,
    double& cauchy_nt_wgt, Core::Gen::Pairedvector<int, double>& d_cauchy_nt_dd,
    Core::LinAlg::SerialDenseMatrix* d_sigma_nt_ds);

FOUR_C_NAMESPACE_CLOSE
