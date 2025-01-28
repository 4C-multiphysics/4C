// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_integrator_poro_scatra.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_integrator.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_s2i.hpp"
#include "4C_mat_structporo.hpp"
#include "4C_scatra_ele_parameter_boundary.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_so3_base.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_poro.hpp"
#include "4C_so3_poro_scatra.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_utils_exceptions.hpp"

#include <Epetra_FEVector.h>
#include <Teuchos_StandardParameterEntryValidators.hpp>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
CONTACT::IntegratorNitschePoroScatra::IntegratorNitschePoroScatra(
    Teuchos::ParameterList& params, Core::FE::CellType eletype, MPI_Comm comm)
    : IntegratorNitsche(params, eletype, comm),
      no_penetration_(params.get<bool>("CONTACTNOPEN")),
      dv_dd_(params.get<double>("porotimefac")),
      scatraparamstimint_(Discret::Elements::ScaTraEleParameterTimInt::instance("scatra")),
      scatraparamsboundary_(Discret::Elements::ScaTraEleParameterBoundary::instance("scatra"))
{
  if (fabs(theta_) > 1e-16) FOUR_C_THROW("Poro Contact just implemented Adjoint free ...");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitschePoroScatra::integrate_gp_3d(Mortar::Element& sele,
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
  // TEUCHOS_FUNC_TIME_MONITOR("CONTACT::IntegratorNitsche::integrate_gp_3d");
  // We use the consistent element normal for poro contact!
  // if (nit_normal_==Inpar::CONTACT::NitNor_ele)
  {
    gpts_forces<3>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt,
        gap, deriv_gap, normal, dnmap_unit, sxi, mxi);
  }
  //  else if (nit_normal_==Inpar::CONTACT::NitNor_sm)
  //    FOUR_C_THROW("Want to use the element normal!");
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitschePoroScatra::integrate_gp_2d(Mortar::Element& sele,
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
  FOUR_C_THROW("2D is not implemented!");
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitschePoroScatra::gpts_forces(Mortar::Element& sele, Mortar::Element& mele,
    const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dsxi,
    const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt, const double gap,
    const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
    std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, double* sxi, double* mxi)
{
  if (sele.owner() != Core::Communication::my_mpi_rank(Comm_)) return;

  static const bool do_fast_checks = true;
  // first rough check
  if (do_fast_checks)
    if (abs(theta_) < 1.e-12 && gap > std::max(sele.max_edge_size(), mele.max_edge_size())) return;

  const Core::LinAlg::Matrix<dim, 1> normal(gpn, true);

  if (dim != n_dim()) FOUR_C_THROW("dimension inconsistency");

  Core::LinAlg::Matrix<dim, 1> slave_normal, master_normal;
  std::vector<Core::Gen::Pairedvector<int, double>> deriv_slave_normal(0, 0);
  std::vector<Core::Gen::Pairedvector<int, double>> deriv_master_normal(0, 0);
  sele.compute_unit_normal_at_xi(sxi, slave_normal.data());
  mele.compute_unit_normal_at_xi(mxi, master_normal.data());
  sele.deriv_unit_normal_at_xi(sxi, deriv_slave_normal);
  mele.deriv_unit_normal_at_xi(mxi, deriv_master_normal);

  double pen = ppn_;
  double pet = ppt_;

  double ws = 0.;
  double wm = 0.;
  CONTACT::Utils::nitsche_weights_and_scaling(sele, mele, nit_wgt_, dt_, ws, wm, pen, pet);

  double cauchy_nn_weighted_average = 0.;
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_d(
      sele.num_node() * 3 * 12 + sele.mo_data().parent_disp().size() +
      mele.mo_data().parent_disp().size());
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_p(
      sele.mo_data().parent_pf_pres().size() + mele.mo_data().parent_pf_pres().size());
  Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv_s(
      sele.mo_data().parent_scalar_dof().size() + mele.mo_data().parent_scalar_dof().size());
  so_ele_cauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, normal, dnmap_unit, ws,
      cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p, cauchy_nn_weighted_average_deriv_s);
  so_ele_cauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, normal, dnmap_unit,
      -wm, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv_d,
      cauchy_nn_weighted_average_deriv_p, cauchy_nn_weighted_average_deriv_s);

  const double snn_av_pen_gap = cauchy_nn_weighted_average + pen * gap;
  Core::Gen::Pairedvector<int, double> d_snn_av_pen_gap(
      cauchy_nn_weighted_average_deriv_d.size() + dgapgp.size());
  for (const auto& p : cauchy_nn_weighted_average_deriv_d) d_snn_av_pen_gap[p.first] += p.second;
  for (const auto& p : dgapgp) d_snn_av_pen_gap[p.first] += pen * p.second;


  if (snn_av_pen_gap < 0.)
  {
    // test in normal contact direction
    integrate_test<dim>(-1., sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, snn_av_pen_gap,
        d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_p, normal, dnmap_unit);
    integrate_test<dim>(+1., mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt, snn_av_pen_gap,
        d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_p, normal, dnmap_unit);

    integrate_poro_no_out_flow<dim>(
        -1, sele, sxi, sval, sderiv, jac, jacintcellmap, wgt, normal, dnmap_unit, mele, mval);
    integrate_poro_no_out_flow<dim>(
        +1, mele, mxi, mval, mderiv, jac, jacintcellmap, wgt, normal, dnmap_unit, sele, sval);

    // integrate the scatra-scatra interface condition
    integrate_ssi_interface_condition<dim>(sele, sval, sderiv, dsxi, mele, mval, mderiv, dmxi,
        snn_av_pen_gap, d_snn_av_pen_gap, cauchy_nn_weighted_average_deriv_s, jac, jacintcellmap,
        wgt);
  }
}


template <int dim>
void CONTACT::IntegratorNitschePoroScatra::so_ele_cauchy(Mortar::Element& moEle,
    double* boundary_gpcoord,
    std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
    const Core::LinAlg::Matrix<dim, 1>& normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    const Core::LinAlg::Matrix<dim, 1>& direction,
    std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, const double w,
    double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_d,
    Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_p,
    Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_s)
{
  Core::LinAlg::Matrix<dim, 1> pxsi(true);
  Core::LinAlg::Matrix<dim, dim> derivtravo_slave;
  CONTACT::Utils::map_gp_to_parent<dim>(moEle, boundary_gpcoord, gp_wgt, pxsi, derivtravo_slave);

  double sigma_nt;
  Core::LinAlg::SerialDenseMatrix dsntdd, dsntdp, dsntds;
  Core::LinAlg::Matrix<dim, 1> dsntdn, dsntdt, dsntdpxi;

  if ((moEle.mo_data().parent_pf_pres().size() > 0) && (moEle.mo_data().parent_scalar().size() > 0))
  {
    if (auto* solid_ele = dynamic_cast<
            Discret::Elements::So3PoroScatra<Discret::Elements::SoHex8, Core::FE::CellType::hex8>*>(
            moEle.parent_element());
        solid_ele != nullptr)
    {
      solid_ele->get_cauchy_n_dir_and_derivatives_at_xi(pxsi, moEle.mo_data().parent_disp(),
          moEle.mo_data().parent_pf_pres(), moEle.mo_data().parent_scalar(), normal, direction,
          sigma_nt, &dsntdd, &dsntdp, &dsntds, &dsntdn, &dsntdt, &dsntdpxi);
    }
    else
    {
      FOUR_C_THROW("The element is not So3PoroScatra");
    }
  }
  else
  {
    switch (moEle.parent_element()->shape())
    {
      case Core::FE::CellType::hex8:
      {
        if (auto* solid_ele = dynamic_cast<Discret::Elements::SoBase*>(moEle.parent_element());
            solid_ele != nullptr)
        {
          solid_ele->get_cauchy_n_dir_and_derivatives_at_xi(pxsi, moEle.mo_data().parent_disp(),
              normal, direction, sigma_nt, &dsntdd, nullptr, nullptr, nullptr, nullptr, &dsntdn,
              &dsntdt, &dsntdpxi, nullptr, nullptr, nullptr, nullptr, nullptr);
        }
        else if (auto* solid_ele = dynamic_cast<Discret::Elements::Solid*>(moEle.parent_element());
                 solid_ele != nullptr)
        {
          Discret::Elements::CauchyNDirLinearizations<3> cauchy_linearizations{};
          cauchy_linearizations.d_cauchyndir_dd = &dsntdd;
          cauchy_linearizations.d_cauchyndir_dn = &dsntdn;
          cauchy_linearizations.d_cauchyndir_ddir = &dsntdt;
          cauchy_linearizations.d_cauchyndir_dxi = &dsntdpxi;

          sigma_nt = solid_ele->get_normal_cauchy_stress_at_xi<3>(
              moEle.mo_data().parent_disp(), pxsi, normal, direction, cauchy_linearizations);
        }
        else
        {
          FOUR_C_THROW("The element is neither SoBase nor Solid");
        }

        break;
      }
      default:
      {
        FOUR_C_THROW("Nitsche contact is not implemented for this shape of old solid element!");
      }
    }
  }

  cauchy_nt += w * sigma_nt;

  for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
    deriv_sigma_nt_d[moEle.mo_data().parent_dof().at(i)] += w * dsntdd(i, 0);

  for (int i = 0; i < dim - 1; ++i)
  {
    for (const auto& p : boundary_gpcoord_lin[i])
      for (int k = 0; k < dim; ++k)
        deriv_sigma_nt_d[p.first] += dsntdpxi(k) * derivtravo_slave(k, i) * p.second * w;
  }

  for (int d = 0; d < dim; ++d)
  {
    for (Core::Gen::Pairedvector<int, double>::const_iterator p = normal_deriv[d].begin();
         p != normal_deriv[d].end(); ++p)
      deriv_sigma_nt_d[p->first] += dsntdn(d) * p->second * w;
  }

  for (int d = 0; d < dim; ++d)
  {
    for (Core::Gen::Pairedvector<int, double>::const_iterator p = direction_deriv[d].begin();
         p != direction_deriv[d].end(); ++p)
      deriv_sigma_nt_d[p->first] += dsntdt(d) * p->second * w;
  }

  if (moEle.mo_data().parent_pf_pres().size() > 0)
  {
    for (int i = 0; i < moEle.parent_element()->num_node(); ++i)
      deriv_sigma_nt_p[moEle.mo_data().parent_pf_dof().at(i * (dim + 1) + 3)] += dsntdp(i, 0) * w;
  }

  if (moEle.mo_data().parent_scalar().size() > 0)
  {
    for (int i = 0; i < moEle.parent_element()->num_node(); ++i)
      deriv_sigma_nt_s[moEle.mo_data().parent_scalar_dof().at(i)] += dsntds(i, 0) * w;
  }
}

template <int dim>
void CONTACT::IntegratorNitschePoroScatra::integrate_test(const double fac, Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape, const Core::LinAlg::SerialDenseMatrix& deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const double test_val, const Core::Gen::Pairedvector<int, double>& test_deriv_d,
    const Core::Gen::Pairedvector<int, double>& test_deriv_p,
    const Core::LinAlg::Matrix<dim, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv)
{
  if (abs(fac) < 1.e-16) return;

  CONTACT::IntegratorNitsche::integrate_test<dim>(fac, ele, shape, deriv, dxi, jac, jacintcellmap,
      wgt, test_val, test_deriv_d, test_dir, test_dir_deriv);

  for (const auto& p : test_deriv_p)
  {
    double* row = ele.get_nitsche_container().kdp(p.first);
    for (int s = 0; s < ele.num_node(); ++s)
    {
      for (int d = 0; d < n_dim(); ++d)
      {
        row[Core::FE::get_parent_node_number_from_face_node_number(
                ele.parent_element()->shape(), ele.face_parent_number(), s) *
                dim +
            d] -= fac * jac * wgt * test_dir(d) * p.second * shape(s);
      }
    }
  }
}

template <int dim>
void CONTACT::IntegratorNitschePoroScatra::integrate_poro_no_out_flow(const double fac,
    Mortar::Element& ele, double* xi, const Core::LinAlg::SerialDenseVector& shape,
    const Core::LinAlg::SerialDenseMatrix& deriv, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const Core::LinAlg::Matrix<dim, 1>& normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    Mortar::Element& otherele, const Core::LinAlg::SerialDenseVector& othershape)
{
  if (abs(fac) < 1e-16) return;
  if (!no_penetration_) return;

  if (!ele.mo_data().parent_pf_dof().size()) return;

  // weighting for poro pressure depenent if two or onesided porocontact
  double sweight = 1;
  double oweight = 0;

  double spresgp = 0;
  double srelveln = 0;
  for (int j = 0; j < ele.num_node(); ++j)
  {
    int pj = Core::FE::get_parent_node_number_from_face_node_number(
        ele.parent_element()->shape(), ele.face_parent_number(), j);
    spresgp += ele.mo_data().parent_pf_pres()[pj] * shape(j);
    for (int d = 0; d < dim; ++d)
    {
      srelveln +=
          (ele.mo_data().parent_pf_vel()[pj * dim + d] - ele.mo_data().parent_vel()[pj * dim + d]) *
          shape(j) * normal(d);
    }
  }

  double sporosity = -1;
  double sJ = -1;
  std::map<int, double> sJLin;
  double sdphi_dp;
  double sdphi_dJ;
  get_poro_quantitiesat_gp(ele, xi, spresgp, sJ, sJLin, sporosity, sdphi_dp, sdphi_dJ);

  if (otherele.mo_data().parent_pf_dof().size())  // two sided poro contact case
  {
    sweight = 0.5;
    oweight = 1 - sweight;
  }

  double val = fac * jac * wgt / dv_dd_;  //*1./dv_dd_;
  for (int i = 0; i < ele.num_node(); ++i)
  {
    int pi = Core::FE::get_parent_node_number_from_face_node_number(
        ele.parent_element()->shape(), ele.face_parent_number(), i);
    for (int j = 0; j < ele.num_node(); ++j)
    {
      int pj = Core::FE::get_parent_node_number_from_face_node_number(
          ele.parent_element()->shape(), ele.face_parent_number(), j);
      for (int d = 0; d < dim; ++d)
      {
        (*ele.get_nitsche_container().rhs_p(pi * (dim + 1) + d)) +=
            shape(i) * ele.mo_data().parent_pf_pres()[pj] * shape(j) * normal(d) * val *
            sweight;  //(v,k1 p1 n)
        ele.get_nitsche_container().kpp(
            ele.mo_data().parent_pf_dof()[pj * (dim + 1) + dim])[(pi * (dim + 1) + d)] -=
            shape(i) * shape(j) * normal(d) * val * sweight;  //(v,k1 dp1/dp n)
        for (auto p = normal_deriv[d].begin(); p != normal_deriv[d].end(); ++p)
        {
          ele.get_nitsche_container().kpd(p->first)[(pi * (dim + 1) + d)] -=
              shape(i) * shape(j) * ele.mo_data().parent_pf_pres()[pj] * p->second * val *
              sweight;  //(v,k1 p1 dn/dd)

          ele.get_nitsche_container().kpd(p->first)[(pi * (dim + 1) + dim)] +=
              shape(i) * sporosity * p->second *
              (ele.mo_data().parent_pf_vel()[pj * dim + d] -
                  ele.mo_data().parent_vel()[pj * dim + d]) *
              shape(j) * val * sweight;  // (k1 q1, phi (vF-vS) dn/dd)
        }
        ele.get_nitsche_container().kpd(ele.mo_data().parent_dof()[pj * dim + d])[(
            pi * (dim + 1) + dim)] -=  // (k1 q1, phi (vF-dvS/dd) n)
            shape(i) * sporosity * normal(d) * shape(j) * val * dv_dd_ * sweight;

        ele.get_nitsche_container().kpp(ele.mo_data().parent_pf_dof()[pj * (dim + 1) + d])[(
            pi * (dim + 1) + dim)] +=  // (k1 q1, phi (dvF/dvF-vS) n)
            shape(i) * sporosity * normal(d) * shape(j) * val * sweight;

        (*ele.get_nitsche_container().rhs_p(pi * (dim + 1) + (dim))) -=
            shape(i) * sporosity * normal(d) *
            (ele.mo_data().parent_pf_vel()[pj * dim + d] -
                ele.mo_data().parent_vel()[pj * dim + d]) *
            shape(j) * val * sweight;  // (k1 q1, phi (vF-vS) n)
      }
      ele.get_nitsche_container().kpp(
          ele.mo_data().parent_pf_dof()[pj * (dim + 1) + dim])[(pi * (dim + 1) + dim)] +=
          val * srelveln * shape(i) * sdphi_dp * shape(j) * sweight;  // (k1 q1, phi/dp (vF-vS) n)
    }

    for (auto& dJit : sJLin)
    {
      ele.get_nitsche_container().kpd(dJit.first)[(pi * (dim + 1) + dim)] +=
          val * srelveln * shape(i) * sdphi_dJ * dJit.second *
          sweight;  // (k1 q1, phi/dd (vF-vS) n)
    }
  }

  if (oweight > 1e-16)
  {
    for (int i = 0; i < ele.num_node(); ++i)
    {
      int pi = Core::FE::get_parent_node_number_from_face_node_number(
          ele.parent_element()->shape(), ele.face_parent_number(), i);
      for (int j = 0; j < otherele.num_node(); ++j)
      {
        int pj = Core::FE::get_parent_node_number_from_face_node_number(
            otherele.parent_element()->shape(), otherele.face_parent_number(), j);
        for (int d = 0; d < dim; ++d)
        {
          (*ele.get_nitsche_container().rhs_p(pi * (dim + 1) + d)) +=
              shape(i) * otherele.mo_data().parent_pf_pres()[pj] * othershape(j) * normal(d) * val *
              oweight;  //(v,k2 p2 n)
          ele.get_nitsche_container().kpp(
              otherele.mo_data().parent_pf_dof()[pj * (dim + 1) + dim])[(pi * (dim + 1) + d)] -=
              shape(i) * othershape(j) * normal(d) * val * oweight;  //(v,k2 dp2/dp n)
          for (auto p = normal_deriv[d].begin(); p != normal_deriv[d].end(); ++p)
          {
            ele.get_nitsche_container().kpd(p->first)[(pi * (dim + 1) + d)] -=
                shape(i) * othershape(j) * otherele.mo_data().parent_pf_pres()[pj] * p->second *
                val * oweight;  //(v,k2 p2 dn/dd)

            otherele.get_nitsche_container().kpd(p->first)[(pj * (dim + 1) + dim)] +=
                othershape(j) * sporosity * p->second *
                (ele.mo_data().parent_pf_vel()[pi * dim + d] -
                    ele.mo_data().parent_vel()[pi * dim + d]) *
                shape(i) * val * oweight;  // (k2 q2, phi (vF-vS) dn/dd)
          }

          otherele.get_nitsche_container().kpd(ele.mo_data().parent_dof()[pi * dim + d])[(
              pj * (dim + 1) + dim)] -=  // (k2 q2, phi (vF-dvS/dd) n)
              othershape(j) * sporosity * normal(d) * shape(i) * val * dv_dd_ * oweight;

          otherele.get_nitsche_container().kpp(ele.mo_data().parent_pf_dof()[pi * (dim + 1) + d])[(
              pj * (dim + 1) + dim)] +=  // (k2 q2, phi (dvF/dvF-vS) n)
              othershape(j) * sporosity * normal(d) * shape(i) * val * oweight;

          (*otherele.get_nitsche_container().rhs_p(pj * (dim + 1) + (dim))) -=
              othershape(j) * sporosity * normal(d) *
              (ele.mo_data().parent_pf_vel()[pi * dim + d] -
                  ele.mo_data().parent_vel()[pi * dim + d]) *
              shape(i) * val * oweight;  // (k2 q2, phi (vF-vS) n)
        }
        otherele.get_nitsche_container().kpp(
            ele.mo_data().parent_pf_dof()[pi * (dim + 1) + dim])[(pj * (dim + 1) + dim)] +=
            val * srelveln * othershape(j) * sdphi_dp * shape(i) *
            oweight;  // (k2 q2, phi/dp (vF-vS) n)
      }
    }
    for (int j = 0; j < otherele.num_node(); ++j)
    {
      int pj = Core::FE::get_parent_node_number_from_face_node_number(
          otherele.parent_element()->shape(), otherele.face_parent_number(), j);
      for (auto& dJit : sJLin)
      {
        otherele.get_nitsche_container().kpd(dJit.first)[(pj * (dim + 1) + dim)] +=
            val * srelveln * othershape(j) * sdphi_dJ * dJit.second *
            oweight;  // (k2 q2, phi/dd (vF-vS) n)
      }
    }
  }
}

bool CONTACT::IntegratorNitschePoroScatra::get_poro_pressure(Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape, Mortar::Element& otherele,
    const Core::LinAlg::SerialDenseVector& othershape, double& poropressure)
{
  if (!ele.mo_data().parent_pf_dof().size() && !otherele.mo_data().parent_pf_dof().size())
    return false;

  double w1 = 1;
  if (ele.mo_data().parent_pf_dof().size() && otherele.mo_data().parent_pf_dof().size())
    w1 = 0.5;
  else if (ele.mo_data().parent_pf_dof().size())
    w1 = 1.0;
  else if (otherele.mo_data().parent_pf_dof().size())
    w1 = 0.0;
  else
    FOUR_C_THROW("Thats not exptected...!");
  double w2 = 1.0 - w1;

  poropressure = 0.0;
  if (ele.mo_data().parent_pf_dof().size())
  {
    for (int j = 0; j < ele.num_node(); ++j)
    {
      int pj = Core::FE::get_parent_node_number_from_face_node_number(
          ele.parent_element()->shape(), ele.face_parent_number(), j);
      poropressure += w1 * ele.mo_data().parent_pf_pres()[pj] * shape(j);
    }
  }

  if (otherele.mo_data().parent_pf_dof().size())
  {
    for (int j = 0; j < otherele.num_node(); ++j)
    {
      int pj = Core::FE::get_parent_node_number_from_face_node_number(
          otherele.parent_element()->shape(), otherele.face_parent_number(), j);
      poropressure += w2 * otherele.mo_data().parent_pf_pres()[pj] * othershape(j);
    }
  }
  return true;
}

void CONTACT::IntegratorNitschePoroScatra::get_poro_quantitiesat_gp(Mortar::Element& ele,
    double* xi,
    double& spresgp,  //(in)
    double& sJ, std::map<int, double>& sJLin, double& sporosity, double& sdphi_dp,
    double& sdphi_dJ)  // out
{
  static double dummy = 1.0;
  sJ = det_deformation_gradient(ele, dummy, xi, sJLin);
  Teuchos::ParameterList sparams;  // empty parameter list;

  std::shared_ptr<Mat::StructPoro> sstructmat =
      std::dynamic_pointer_cast<Mat::StructPoro>(ele.parent_element()->material(0));
  if (sstructmat == nullptr)
    sstructmat = std::dynamic_pointer_cast<Mat::StructPoro>(ele.parent_element()->material(1));
  if (sstructmat == nullptr) FOUR_C_THROW("Cast to StructPoro failed!");
  sstructmat->compute_surf_porosity(sparams, spresgp, sJ, ele.face_parent_number(), 1, sporosity,
      &sdphi_dp, &sdphi_dJ, nullptr, nullptr, nullptr, false);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitschePoroScatra::integrate_ssi_interface_condition(
    Mortar::Element& slave_ele, const Core::LinAlg::SerialDenseVector& slave_shape,
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
          "is not recognized: %i",
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
void CONTACT::IntegratorNitschePoroScatra::integrate_scatra_test(const double fac,
    Mortar::Element& ele, const Core::LinAlg::SerialDenseVector& shape_func,
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

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitschePoroScatra::setup_gp_concentrations(Mortar::Element& ele,
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

template void CONTACT::IntegratorNitschePoroScatra::integrate_test<2>(const double,
    Mortar::Element&, const Core::LinAlg::SerialDenseVector&,
    const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>& i, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const Core::Gen::Pairedvector<int, double>&,
    const Core::LinAlg::Matrix<2, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);
template void CONTACT::IntegratorNitschePoroScatra::integrate_test<3>(const double,
    Mortar::Element&, const Core::LinAlg::SerialDenseVector&,
    const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>& i, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const Core::Gen::Pairedvector<int, double>&,
    const Core::LinAlg::Matrix<3, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);

template void CONTACT::IntegratorNitschePoroScatra::integrate_poro_no_out_flow<3>(const double fac,
    Mortar::Element& ele, double* xi, const Core::LinAlg::SerialDenseVector& shape,
    const Core::LinAlg::SerialDenseMatrix& deriv, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const Core::LinAlg::Matrix<3, 1>& normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    Mortar::Element& otherele, const Core::LinAlg::SerialDenseVector& othershape);

template void CONTACT::IntegratorNitschePoroScatra::so_ele_cauchy<3>(Mortar::Element& moEle,
    double* boundary_gpcoord,
    std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
    const Core::LinAlg::Matrix<3, 1>& normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    const Core::LinAlg::Matrix<3, 1>& direction,
    std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, const double w,
    double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_d,
    Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_p,
    Core::Gen::Pairedvector<int, double>& deriv_sigma_nt_s);

FOUR_C_NAMESPACE_CLOSE
