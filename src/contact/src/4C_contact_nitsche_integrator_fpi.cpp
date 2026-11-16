// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_integrator_fpi.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_contact_element.hpp"
#include "4C_contact_nitsche_integrator_fsi.hpp"
#include "4C_contact_node.hpp"
#include "4C_xfem_xfluid_contact_communicator.hpp"

FOUR_C_NAMESPACE_OPEN
/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::IntegratorNitscheFpi::IntegratorNitscheFpi(
    Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
    : IntegratorNitschePoro(params, eletype, comm), ele_contact_state_(-2)
{
  if (imortar_.isParameter("XFluidContactComm"))
    xf_c_comm_ = imortar_.get<std::shared_ptr<XFEM::XFluidContactComm>>("XFluidContactComm");
  else
    FOUR_C_THROW("Couldn't find XFluidContactComm!");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitscheFpi::integrate_deriv_ele_3d(Mortar::Element& source_elem,
    std::vector<Mortar::Element*> target_elems, bool* boundary_ele, bool* proj_, MPI_Comm comm,
    const std::shared_ptr<Mortar::ParamsInterface>& cparams_ptr)
{
  auto* contact_source_elem = dynamic_cast<CONTACT::Element*>(&source_elem);
  if (!contact_source_elem) FOUR_C_THROW("Could cast to Contact Element!");

  // do quick orientation check
  Core::LinAlg::Matrix<3, 1> s_n, t_n;
  double center[2] = {0., 0.};
  source_elem.compute_unit_normal_at_xi(center, s_n.data());
  for (auto t_it = target_elems.begin(); t_it != target_elems.end(); ++t_it)
  {
    (*t_it)->compute_unit_normal_at_xi(center, t_n.data());
    if (s_n.dot(t_n) > -1e-1)
    {
      target_elems.erase(t_it);
      --t_it;
    }
  }

  if (!target_elems.size()) return;

  if (xf_c_comm_->higher_integrationfor_contact_element(source_elem.id()))
    xf_c_comm_->get_cut_side_integration_points(source_elem.id(), coords_, weights_, ngp_);

  // Call Base Contact Integratederiv with potentially increased number of GPs!
  CONTACT::Integrator::integrate_deriv_ele_3d(
      source_elem, target_elems, boundary_ele, proj_, comm, cparams_ptr);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitscheFpi::integrate_gp_3d(Mortar::Element& source_elem,
    Mortar::Element& target_elem, Core::LinAlg::SerialDenseVector& source_val,
    Core::LinAlg::SerialDenseVector& lm_val, Core::LinAlg::SerialDenseVector& target_val,
    Core::LinAlg::SerialDenseMatrix& source_deriv, Core::LinAlg::SerialDenseMatrix& target_deriv,
    Core::LinAlg::SerialDenseMatrix& lm_deriv,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, double& wgt,
    double& jac, Core::Gen::Pairedvector<int, double>& derivjac, double* normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double& gap,
    Core::Gen::Pairedvector<int, double>& deriv_gap, double* source_xi, double* target_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi)
{
  // Here the consistent element normal is use to allow for a continuous transition between FSI and
  // Contact
  double n[3];
  source_elem.compute_unit_normal_at_xi(source_xi, n);
  std::vector<Core::Gen::Pairedvector<int, double>> dn(3, source_elem.num_node() * 3);
  dynamic_cast<CONTACT::Element&>(source_elem).deriv_unit_normal_at_xi(source_xi, dn);

  gpts_forces<3>(source_elem, target_elem, source_val, source_deriv, source_derivs_xi, target_val,
      target_deriv, target_derivs_xi, jac, derivjac, wgt, gap, deriv_gap, n, dn, source_xi,
      target_xi);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheFpi::gpts_forces(Mortar::Element& source_elem,
    Mortar::Element& target_elem, const Core::LinAlg::SerialDenseVector& source_val,
    const Core::LinAlg::SerialDenseMatrix& source_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi,
    const Core::LinAlg::SerialDenseVector& target_val,
    const Core::LinAlg::SerialDenseMatrix& target_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& d_target_xi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt, const double gap,
    const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double* source_xi,
    double* target_xi)
{
  // first rough check
  if (gap > 10 * std::max(source_elem.max_edge_size(), target_elem.max_edge_size())) return;

  const Core::LinAlg::Matrix<dim, 1> normal(gpn, true);

  if (dim != n_dim()) FOUR_C_THROW("dimension inconsistency");


  double pen = ppn_;
  double pet = ppt_;

  double w_source = 0.;
  double w_target = 0.;
  CONTACT::Utils::nitsche_weights_and_scaling(
      source_elem, target_elem, nit_wgt_, w_source, w_target, pen, pet);

  bool FSI_integrated = true;  // bool indicates if fsi condition is already evaluated ... --> if
                               // true no contribution here ...

  Core::LinAlg::Matrix<dim, 1> pxsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<dim, dim> derivtravo_source;
  CONTACT::Utils::map_gp_to_parent<dim>(source_elem, source_xi, wgt, pxsi, derivtravo_source);

  bool gp_on_this_proc;
  double normal_contact_transition = get_normal_contact_transition<dim>(source_elem, target_elem,
      source_val, target_val, source_xi, pxsi, normal, FSI_integrated, gp_on_this_proc);

  if (!gp_on_this_proc) return;

  static int processed_gps = 0;
  ++processed_gps;
  if (processed_gps == 100000)
  {
    std::cout << "==| Processed again 100000 C-Gps! (" << Core::Communication::my_mpi_rank(Comm_)
              << ") |==" << std::endl;
    processed_gps = 0;
  }


  // fast check
  const double snn_pengap =
      w_source * CONTACT::Utils::solid_cauchy_at_xi(dynamic_cast<CONTACT::Element*>(&source_elem),
                     Core::LinAlg::Matrix<dim - 1, 1>(source_xi, true), normal, normal) +
      w_target * CONTACT::Utils::solid_cauchy_at_xi(dynamic_cast<CONTACT::Element*>(&target_elem),
                     Core::LinAlg::Matrix<dim - 1, 1>(target_xi, true), normal, normal) +
      pen * gap;

  if (!FSI_integrated || (gap < (1 + xf_c_comm_->get_fpi_pcontact_fullfraction()) *
                                     xf_c_comm_->get_fpi_pcontact_exchange_dist() &&
                             xf_c_comm_->get_fpi_pcontact_exchange_dist() > 1e-16))
  {
    double ffac = 1;
    if (gap < (1 + xf_c_comm_->get_fpi_pcontact_fullfraction()) *
                  xf_c_comm_->get_fpi_pcontact_exchange_dist() &&
        FSI_integrated &&
        gap > xf_c_comm_->get_fpi_pcontact_fullfraction() *
                  xf_c_comm_->get_fpi_pcontact_exchange_dist())
      ffac = 1. - (gap / (xf_c_comm_->get_fpi_pcontact_exchange_dist()) -
                      xf_c_comm_->get_fpi_pcontact_fullfraction());

    integrate_poro_no_out_flow<dim>(-ffac, source_elem, source_xi, source_val, source_deriv, jac,
        jacintcellmap, wgt, normal, dnmap_unit, target_elem, target_val);
  }

  if (snn_pengap >= normal_contact_transition && !FSI_integrated)
  {
    Core::Gen::Pairedvector<int, double> lin_fluid_traction(0);
    integrate_test<dim>(-1., source_elem, source_val, source_deriv, d_source_xi, jac, jacintcellmap,
        wgt, normal_contact_transition, lin_fluid_traction, lin_fluid_traction, normal, dnmap_unit);

    update_ele_contact_state(source_elem, 0);
  }

  if (snn_pengap >= normal_contact_transition)
  {
    update_ele_contact_state(source_elem, -1);
    if (!FSI_integrated)
      xf_c_comm_->inc_gp(1);
    else
      xf_c_comm_->inc_gp(2);
    return;
  }

  double cauchy_nn_weighted_average = 0.;
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_d(
      source_elem.num_node() * 3 * 12 + source_elem.mo_data().parent_disp().size() +
      target_elem.mo_data().parent_disp().size());
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_p(
      source_elem.mo_data().parent_pf_pres().size() +
      target_elem.mo_data().parent_pf_pres().size());

  so_ele_cauchy<dim>(source_elem, source_xi, d_source_xi, wgt, normal, dnmap_unit, normal,
      dnmap_unit, w_source, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p);
  so_ele_cauchy<dim>(target_elem, target_xi, d_target_xi, wgt, normal, dnmap_unit, normal,
      dnmap_unit, w_target, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p);

  const double snn_av_pen_gap = cauchy_nn_weighted_average + pen * gap;
  Core::Gen::Pairedvector<int, double> d_snn_av_pen_gap(
      cauchy_nn_weighted_average_deriv_d.size() + dgapgp.size());
  for (const auto& p : cauchy_nn_weighted_average_deriv_d) d_snn_av_pen_gap[p.first] += p.second;
  for (const auto& p : dgapgp) d_snn_av_pen_gap[p.first] += pen * p.second;

  // test in normal contact direction
  integrate_test<dim>(-1., source_elem, source_val, source_deriv, d_source_xi, jac, jacintcellmap,
      wgt, snn_av_pen_gap, d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_p, normal,
      dnmap_unit);

  update_ele_contact_state(source_elem, 1);

  xf_c_comm_->inc_gp(0);
}

void CONTACT::IntegratorNitscheFpi::update_ele_contact_state(
    Mortar::Element& source_elem, int state)
{
  if (!state && ele_contact_state_)
  {
    ele_contact_state_ = state;
    xf_c_comm_->register_contact_elementfor_higher_integration(source_elem.id());
  }
  else if (ele_contact_state_ == -2)
    ele_contact_state_ = state;
  else if (ele_contact_state_ == -state)  // switch between contact and no contact
  {
    ele_contact_state_ = 0;
    xf_c_comm_->register_contact_elementfor_higher_integration(source_elem.id());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double CONTACT::IntegratorNitscheFpi::get_normal_contact_transition(Mortar::Element& source_elem,
    Mortar::Element& target_elem, const Core::LinAlg::SerialDenseVector& source_val,
    const Core::LinAlg::SerialDenseVector& target_val, const double* source_xi,
    const Core::LinAlg::Matrix<dim, 1>& pxsi, const Core::LinAlg::Matrix<dim, 1>& normal,
    bool& FSI_integrated, bool& gp_on_this_proc)
{
  double poropressure(0.0);
  if (get_poro_pressure(source_elem, source_val, target_elem, target_val, poropressure))
  {
    return xf_c_comm_->get_fsi_traction(&source_elem, pxsi,
        Core::LinAlg::Matrix<dim - 1, 1>(source_xi, false), normal, FSI_integrated, gp_on_this_proc,
        &poropressure);
  }
  else
    return xf_c_comm_->get_fsi_traction(&source_elem, pxsi,
        Core::LinAlg::Matrix<dim - 1, 1>(source_xi, false), normal, FSI_integrated,
        gp_on_this_proc);
}

FOUR_C_NAMESPACE_CLOSE
