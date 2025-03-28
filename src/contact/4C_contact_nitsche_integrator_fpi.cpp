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
void CONTACT::IntegratorNitscheFpi::integrate_deriv_ele_3d(Mortar::Element& sele,
    std::vector<Mortar::Element*> meles, bool* boundary_ele, bool* proj_, MPI_Comm comm,
    const std::shared_ptr<Mortar::ParamsInterface>& cparams_ptr)
{
  auto* csele = dynamic_cast<CONTACT::Element*>(&sele);
  if (!csele) FOUR_C_THROW("Could cast to Contact Element!");

  // do quick orientation check
  Core::LinAlg::Matrix<3, 1> sn, mn;
  double center[2] = {0., 0.};
  sele.compute_unit_normal_at_xi(center, sn.data());
  for (auto mit = meles.begin(); mit != meles.end(); ++mit)
  {
    (*mit)->compute_unit_normal_at_xi(center, mn.data());
    if (sn.dot(mn) > -1e-1)
    {
      meles.erase(mit);
      --mit;
    }
  }

  if (!meles.size()) return;

  if (xf_c_comm_->higher_integrationfor_contact_element(sele.id()))
    xf_c_comm_->get_cut_side_integration_points(sele.id(), coords_, weights_, ngp_);

  // Call Base Contact Integratederiv with potentially increased number of GPs!
  CONTACT::Integrator::integrate_deriv_ele_3d(sele, meles, boundary_ele, proj_, comm, cparams_ptr);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitscheFpi::integrate_gp_3d(Mortar::Element& sele, Mortar::Element& mele,
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
  // Here the consistent element normal is use to allow for a continuous transition between FSI and
  // Contact
  double n[3];
  sele.compute_unit_normal_at_xi(sxi, n);
  std::vector<Core::Gen::Pairedvector<int, double>> dn(3, sele.num_node() * 3);
  dynamic_cast<CONTACT::Element&>(sele).deriv_unit_normal_at_xi(sxi, dn);

  gpts_forces<3>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt,
      gap, deriv_gap, n, dn, sxi, mxi);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitscheFpi::gpts_forces(Mortar::Element& sele, Mortar::Element& mele,
    const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dsxi,
    const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt, const double gap,
    const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double* sxi, double* mxi)
{
  // first rough check
  if (gap > 10 * std::max(sele.max_edge_size(), mele.max_edge_size())) return;

  const Core::LinAlg::Matrix<dim, 1> normal(gpn, true);

  if (dim != n_dim()) FOUR_C_THROW("dimension inconsistency");


  double pen = ppn_;
  double pet = ppt_;

  double ws = 0.;
  double wm = 0.;
  CONTACT::Utils::nitsche_weights_and_scaling(sele, mele, nit_wgt_, dt_, ws, wm, pen, pet);

  bool FSI_integrated = true;  // bool indicates if fsi condition is already evaluated ... --> if
                               // true no contribution here ...

  Core::LinAlg::Matrix<dim, 1> pxsi(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<dim, dim> derivtravo_slave;
  CONTACT::Utils::map_gp_to_parent<dim>(sele, sxi, wgt, pxsi, derivtravo_slave);

  bool gp_on_this_proc;
  double normal_contact_transition = get_normal_contact_transition<dim>(
      sele, mele, sval, mval, sxi, pxsi, normal, FSI_integrated, gp_on_this_proc);

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
      ws * CONTACT::Utils::solid_cauchy_at_xi(dynamic_cast<CONTACT::Element*>(&sele),
               Core::LinAlg::Matrix<dim - 1, 1>(sxi, true), normal, normal) +
      wm * CONTACT::Utils::solid_cauchy_at_xi(dynamic_cast<CONTACT::Element*>(&mele),
               Core::LinAlg::Matrix<dim - 1, 1>(mxi, true), normal, normal) +
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

    integrate_poro_no_out_flow<dim>(
        -ffac, sele, sxi, sval, sderiv, jac, jacintcellmap, wgt, normal, dnmap_unit, mele, mval);
  }

  if (snn_pengap >= normal_contact_transition && !FSI_integrated)
  {
    Core::Gen::Pairedvector<int, double> lin_fluid_traction(0);
    integrate_test<dim>(-1., sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
        normal_contact_transition, lin_fluid_traction, lin_fluid_traction, normal, dnmap_unit);

    update_ele_contact_state(sele, 0);
  }

  if (snn_pengap >= normal_contact_transition)
  {
    update_ele_contact_state(sele, -1);
    if (!FSI_integrated)
      xf_c_comm_->inc_gp(1);
    else
      xf_c_comm_->inc_gp(2);
    return;
  }

  double cauchy_nn_weighted_average = 0.;
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_d(
      sele.num_node() * 3 * 12 + sele.mo_data().parent_disp().size() +
      mele.mo_data().parent_disp().size());
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_p(
      sele.mo_data().parent_pf_pres().size() + mele.mo_data().parent_pf_pres().size());

  so_ele_cauchy<dim>(sele, sxi, dsxi, wgt, normal, dnmap_unit, normal, dnmap_unit, ws,
      cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p);
  so_ele_cauchy<dim>(mele, mxi, dmxi, wgt, normal, dnmap_unit, normal, dnmap_unit, wm,
      cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p);

  const double snn_av_pen_gap = cauchy_nn_weighted_average + pen * gap;
  Core::Gen::Pairedvector<int, double> d_snn_av_pen_gap(
      cauchy_nn_weighted_average_deriv_d.size() + dgapgp.size());
  for (const auto& p : cauchy_nn_weighted_average_deriv_d) d_snn_av_pen_gap[p.first] += p.second;
  for (const auto& p : dgapgp) d_snn_av_pen_gap[p.first] += pen * p.second;

  // test in normal contact direction
  integrate_test<dim>(-1., sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, snn_av_pen_gap,
      d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_p, normal, dnmap_unit);

  update_ele_contact_state(sele, 1);

  xf_c_comm_->inc_gp(0);
}

void CONTACT::IntegratorNitscheFpi::update_ele_contact_state(Mortar::Element& sele, int state)
{
  if (!state && ele_contact_state_)
  {
    ele_contact_state_ = state;
    xf_c_comm_->register_contact_elementfor_higher_integration(sele.id());
  }
  else if (ele_contact_state_ == -2)
    ele_contact_state_ = state;
  else if (ele_contact_state_ == -state)  // switch between contact and no contact
  {
    ele_contact_state_ = 0;
    xf_c_comm_->register_contact_elementfor_higher_integration(sele.id());
  }
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
double CONTACT::IntegratorNitscheFpi::get_normal_contact_transition(Mortar::Element& sele,
    Mortar::Element& mele, const Core::LinAlg::SerialDenseVector& sval,
    const Core::LinAlg::SerialDenseVector& mval, const double* sxi,
    const Core::LinAlg::Matrix<dim, 1>& pxsi, const Core::LinAlg::Matrix<dim, 1>& normal,
    bool& FSI_integrated, bool& gp_on_this_proc)
{
  double poropressure(0.0);
  if (get_poro_pressure(sele, sval, mele, mval, poropressure))
  {
    return xf_c_comm_->get_fsi_traction(&sele, pxsi, Core::LinAlg::Matrix<dim - 1, 1>(sxi, false),
        normal, FSI_integrated, gp_on_this_proc, &poropressure);
  }
  else
    return xf_c_comm_->get_fsi_traction(&sele, pxsi, Core::LinAlg::Matrix<dim - 1, 1>(sxi, false),
        normal, FSI_integrated, gp_on_this_proc);
}

FOUR_C_NAMESPACE_CLOSE
