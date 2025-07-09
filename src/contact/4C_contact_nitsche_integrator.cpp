// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_nitsche_integrator.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_input.hpp"
#include "4C_contact_nitsche_utils.hpp"
#include "4C_contact_node.hpp"
#include "4C_contact_paramsinterface.hpp"
#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_mat_elasthyper.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_solid_3D_ele_calc_lib_nitsche.hpp"
#include "4C_utils_exceptions.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorNitsche::integrate_gp_3d(Mortar::Element& sele, Mortar::Element& mele,
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
void CONTACT::IntegratorNitsche::integrate_gp_2d(Mortar::Element& sele, Mortar::Element& mele,
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
  gpts_forces<2>(sele, mele, sval, sderiv, derivsxi, mval, mderiv, derivmxi, jac, derivjac, wgt,
      gap, deriv_gap, normal, dnmap_unit, sxi, mxi);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <int dim>
void CONTACT::IntegratorNitsche::gpts_forces(Mortar::Element& sele, Mortar::Element& mele,
    const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dsxi,
    const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dmxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt, const double gap,
    const Core::Gen::Pairedvector<int, double>& dgapgp, const double* gpn,
    std::vector<Core::Gen::Pairedvector<int, double>>& deriv_contact_normal, double* sxi,
    double* mxi)
{
  if (sele.owner() != Core::Communication::my_mpi_rank(Comm_)) return;

  if (dim != n_dim()) FOUR_C_THROW("dimension inconsistency");

  if (frtype_ != CONTACT::FrictionType::none && dim != 3) FOUR_C_THROW("only 3D friction");
  if (frtype_ != CONTACT::FrictionType::none && frtype_ != CONTACT::FrictionType::coulomb &&
      frtype_ != CONTACT::FrictionType::tresca)
    FOUR_C_THROW("only coulomb or tresca friction");
  if (frtype_ == CONTACT::FrictionType::coulomb && frcoeff_ < 0.)
    FOUR_C_THROW("negative coulomb friction coefficient");
  if (frtype_ == CONTACT::FrictionType::tresca && frbound_ < 0.)
    FOUR_C_THROW("negative tresca friction bound");

  Core::LinAlg::Matrix<dim, 1> slave_normal, master_normal;
  std::vector<Core::Gen::Pairedvector<int, double>> deriv_slave_normal;
  std::vector<Core::Gen::Pairedvector<int, double>> deriv_master_normal;
  sele.compute_unit_normal_at_xi(sxi, slave_normal.data());
  mele.compute_unit_normal_at_xi(mxi, master_normal.data());
  sele.deriv_unit_normal_at_xi(sxi, deriv_slave_normal);
  mele.deriv_unit_normal_at_xi(mxi, deriv_master_normal);

  double pen = ppn_;
  double pet = ppt_;

  const Core::LinAlg::Matrix<dim, 1> contact_normal(gpn, true);

  if (stype_ == CONTACT::SolvingStrategy::nitsche)
  {
    double cauchy_nn_weighted_average = 0.;
    Core::Gen::Pairedvector<int, double> cauchy_nn_weighted_average_deriv(
        sele.num_node() * 3 * 12 + sele.mo_data().parent_disp().size() +
        mele.mo_data().parent_disp().size());

    Core::LinAlg::SerialDenseVector normal_adjoint_test_slave(sele.mo_data().parent_dof().size());
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector> deriv_normal_adjoint_test_slave(
        sele.mo_data().parent_dof().size() + deriv_contact_normal[0].size() + dsxi[0].size(), -1,
        Core::LinAlg::SerialDenseVector(sele.mo_data().parent_dof().size(), true));

    Core::LinAlg::SerialDenseVector normal_adjoint_test_master(mele.mo_data().parent_dof().size());
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector> deriv_normal_adjoint_test_master(
        mele.mo_data().parent_dof().size() + deriv_contact_normal[0].size() + dmxi[0].size(), -1,
        Core::LinAlg::SerialDenseVector(mele.mo_data().parent_dof().size(), true));

    double ws = 0.;
    double wm = 0.;
    CONTACT::Utils::nitsche_weights_and_scaling(sele, mele, nit_wgt_, dt_, ws, wm, pen, pet);

    // variables for friction (declaration only)
    Core::LinAlg::Matrix<dim, 1> t1, t2;
    std::vector<Core::Gen::Pairedvector<int, double>> dt1, dt2;
    Core::LinAlg::Matrix<dim, 1> relVel;
    std::vector<Core::Gen::Pairedvector<int, double>> relVel_deriv(
        dim, sele.num_node() * dim + mele.num_node() * dim + dsxi[0].size() + dmxi[0].size());
    double vt1(0.0), vt2(0.0);
    Core::Gen::Pairedvector<int, double> dvt1(0);
    Core::Gen::Pairedvector<int, double> dvt2(0);
    double cauchy_nt1_weighted_average = 0.;
    Core::Gen::Pairedvector<int, double> cauchy_nt1_weighted_average_deriv(
        sele.num_node() * 3 * 12 + sele.mo_data().parent_disp().size() +
        mele.mo_data().parent_disp().size());
    Core::LinAlg::SerialDenseVector t1_adjoint_test_slave(sele.mo_data().parent_dof().size());
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector> deriv_t1_adjoint_test_slave(
        sele.mo_data().parent_dof().size() + deriv_contact_normal[0].size() + dsxi[0].size(), -1,
        Core::LinAlg::SerialDenseVector(sele.mo_data().parent_dof().size(), true));
    Core::LinAlg::SerialDenseVector t1_adjoint_test_master(mele.mo_data().parent_dof().size());
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector> deriv_t1_adjoint_test_master(
        mele.mo_data().parent_dof().size() + deriv_contact_normal[0].size() + dmxi[0].size(), -1,
        Core::LinAlg::SerialDenseVector(mele.mo_data().parent_dof().size(), true));
    double cauchy_nt2_weighted_average = 0.;
    Core::Gen::Pairedvector<int, double> cauchy_nt2_weighted_average_deriv(
        sele.num_node() * 3 * 12 + sele.mo_data().parent_disp().size() +
        mele.mo_data().parent_disp().size());
    Core::LinAlg::SerialDenseVector t2_adjoint_test_slave(sele.mo_data().parent_dof().size());
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector> deriv_t2_adjoint_test_slave(
        sele.mo_data().parent_dof().size() + deriv_contact_normal[0].size() + dsxi[0].size(), -1,
        Core::LinAlg::SerialDenseVector(sele.mo_data().parent_dof().size(), true));
    Core::LinAlg::SerialDenseVector t2_adjoint_test_master(mele.mo_data().parent_dof().size());
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector> deriv_t2_adjoint_test_master(
        mele.mo_data().parent_dof().size() + deriv_contact_normal[0].size() + dmxi[0].size(), -1,
        Core::LinAlg::SerialDenseVector(mele.mo_data().parent_dof().size(), true));
    double sigma_nt1_pen_vt1(0.0), sigma_nt2_pen_vt2(0.0);
    Core::Gen::Pairedvector<int, double> d_sigma_nt1_pen_vt1(
        dgapgp.capacity() + cauchy_nn_weighted_average_deriv.capacity() +
            cauchy_nt1_weighted_average_deriv.capacity() + dvt1.capacity(),
        0, 0);
    Core::Gen::Pairedvector<int, double> d_sigma_nt2_pen_vt2(
        dgapgp.capacity() + cauchy_nn_weighted_average_deriv.capacity() +
            cauchy_nt2_weighted_average_deriv.capacity() + dvt2.capacity(),
        0, 0);
    // variables for friction (end)

    so_ele_cauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, contact_normal,
        deriv_contact_normal, ws, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv,
        normal_adjoint_test_slave, deriv_normal_adjoint_test_slave);
    so_ele_cauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, contact_normal,
        deriv_contact_normal, -wm, cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv,
        normal_adjoint_test_master, deriv_normal_adjoint_test_master);

    const double snn_av_pen_gap = cauchy_nn_weighted_average + pen * gap;
    Core::Gen::Pairedvector<int, double> d_snn_av_pen_gap(
        cauchy_nn_weighted_average_deriv.size() + dgapgp.size());
    for (const auto& p : cauchy_nn_weighted_average_deriv) d_snn_av_pen_gap[p.first] += p.second;
    for (const auto& p : dgapgp) d_snn_av_pen_gap[p.first] += pen * p.second;

    // evaluation of tangential stuff
    if (frtype_ != CONTACT::FrictionType::none)
    {
      CONTACT::Utils::build_tangent_vectors<dim>(
          contact_normal.data(), deriv_contact_normal, t1.data(), dt1, t2.data(), dt2);
      CONTACT::Utils::rel_vel_invariant<dim>(sele, sxi, dsxi, sval, sderiv, mele, mxi, dmxi, mval,
          mderiv, gap, dgapgp, relVel, relVel_deriv);
      CONTACT::Utils::vector_scalar_product<dim>(t1, dt1, relVel, relVel_deriv, vt1, dvt1);
      CONTACT::Utils::vector_scalar_product<dim>(t2, dt2, relVel, relVel_deriv, vt2, dvt2);

      so_ele_cauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, t1, dt1, ws,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1_adjoint_test_slave,
          deriv_t1_adjoint_test_slave);
      so_ele_cauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, t1, dt1, -wm,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1_adjoint_test_master,
          deriv_t1_adjoint_test_master);

      so_ele_cauchy<dim>(sele, sxi, dsxi, wgt, slave_normal, deriv_slave_normal, t2, dt2, ws,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2_adjoint_test_slave,
          deriv_t2_adjoint_test_slave);
      so_ele_cauchy<dim>(mele, mxi, dmxi, wgt, master_normal, deriv_master_normal, t2, dt2, -wm,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2_adjoint_test_master,
          deriv_t2_adjoint_test_master);
    }  // evaluation of tangential stuff

    if (frtype_ != CONTACT::FrictionType::none)
    {
      integrate_test<dim>(-1. + theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1, dt1);
      integrate_test<dim>(-1. + theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2, dt2);
      if (!two_half_pass_)
      {
        integrate_test<dim>(+1. - theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, t1, dt1);
        integrate_test<dim>(+1. - theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, t2, dt2);
      }

      integrate_adjoint_test<dim>(-theta_ / pet, jac, jacintcellmap, wgt,
          cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, sele,
          t1_adjoint_test_slave, deriv_t1_adjoint_test_slave);
      integrate_adjoint_test<dim>(-theta_ / pet, jac, jacintcellmap, wgt,
          cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, sele,
          t2_adjoint_test_slave, deriv_t2_adjoint_test_slave);
      if (!two_half_pass_)
      {
        integrate_adjoint_test<dim>(-theta_ / pet, jac, jacintcellmap, wgt,
            cauchy_nt1_weighted_average, cauchy_nt1_weighted_average_deriv, mele,
            t1_adjoint_test_master, deriv_t1_adjoint_test_master);
        integrate_adjoint_test<dim>(-theta_ / pet, jac, jacintcellmap, wgt,
            cauchy_nt2_weighted_average, cauchy_nt2_weighted_average_deriv, mele,
            t2_adjoint_test_master, deriv_t2_adjoint_test_master);
      }
    }

    if (snn_av_pen_gap >= 0.)
    {
      integrate_test<dim>(-1. + theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
          deriv_contact_normal);
      if (!two_half_pass_)
      {
        integrate_test<dim>(+1. - theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
            deriv_contact_normal);
      }

      integrate_adjoint_test<dim>(-theta_ / pen, jac, jacintcellmap, wgt,
          cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, sele,
          normal_adjoint_test_slave, deriv_normal_adjoint_test_slave);
      if (!two_half_pass_)
      {
        integrate_adjoint_test<dim>(-theta_ / pen, jac, jacintcellmap, wgt,
            cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, mele,
            normal_adjoint_test_master, deriv_normal_adjoint_test_master);
      }
    }
    else
    {
      // test in normal contact direction
      integrate_test<dim>(-1., sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
          cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
          deriv_contact_normal);
      if (!two_half_pass_)
      {
        integrate_test<dim>(+1., mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
            cauchy_nn_weighted_average, cauchy_nn_weighted_average_deriv, contact_normal,
            deriv_contact_normal);
      }

      integrate_test<dim>(-theta_2_ * pen, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, gap,
          dgapgp, contact_normal, deriv_contact_normal);
      if (!two_half_pass_)
      {
        integrate_test<dim>(+theta_2_ * pen, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt, gap,
            dgapgp, contact_normal, deriv_contact_normal);
      }

      integrate_adjoint_test<dim>(theta_, jac, jacintcellmap, wgt, gap, dgapgp, sele,
          normal_adjoint_test_slave, deriv_normal_adjoint_test_slave);
      if (!two_half_pass_)
      {
        integrate_adjoint_test<dim>(theta_, jac, jacintcellmap, wgt, gap, dgapgp, mele,
            normal_adjoint_test_master, deriv_normal_adjoint_test_master);
      }

      if (frtype_ != CONTACT::FrictionType::none)
      {
        double fr = 0.0;
        switch (frtype_)
        {
          case CONTACT::FrictionType::coulomb:
            fr = frcoeff_ * (-1.) * (snn_av_pen_gap);
            break;
          case CONTACT::FrictionType::tresca:
            fr = frbound_;
            break;
          default:
            FOUR_C_THROW("why are you here???");
            break;
        }

        double tan_tr = sqrt(
            (cauchy_nt1_weighted_average + pet * vt1) * (cauchy_nt1_weighted_average + pet * vt1) +
            (cauchy_nt2_weighted_average + pet * vt2) * (cauchy_nt2_weighted_average + pet * vt2));

        // stick
        if (tan_tr < fr)
        {
          sigma_nt1_pen_vt1 = cauchy_nt1_weighted_average + pet * vt1;
          for (const auto& p : dvt1) d_sigma_nt1_pen_vt1[p.first] += pet * p.second;
          for (const auto& p : cauchy_nt1_weighted_average_deriv)
            d_sigma_nt1_pen_vt1[p.first] += p.second;

          sigma_nt2_pen_vt2 = cauchy_nt2_weighted_average + pet * vt2;
          for (const auto& p : dvt2) d_sigma_nt2_pen_vt2[p.first] += pet * p.second;
          for (const auto& p : cauchy_nt2_weighted_average_deriv)
            d_sigma_nt2_pen_vt2[p.first] += p.second;
        }
        // slip
        else
        {
          Core::Gen::Pairedvector<int, double> tmp_d(
              dgapgp.size() + cauchy_nn_weighted_average_deriv.size() +
                  cauchy_nt1_weighted_average_deriv.size() + dvt1.size(),
              0, 0);
          if (frtype_ == CONTACT::FrictionType::coulomb)
            for (const auto& p : d_snn_av_pen_gap) tmp_d[p.first] += -frcoeff_ / tan_tr * p.second;

          for (const auto& p : cauchy_nt1_weighted_average_deriv)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt1_weighted_average + pet * vt1) * p.second;
          for (const auto& p : dvt1)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt1_weighted_average + pet * vt1) * (+pet) * p.second;

          for (const auto& p : cauchy_nt2_weighted_average_deriv)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt2_weighted_average + pet * vt2) * p.second;
          for (const auto& p : dvt2)
            tmp_d[p.first] += -fr / (tan_tr * tan_tr * tan_tr) *
                              (cauchy_nt2_weighted_average + pet * vt2) * (+pet) * p.second;

          sigma_nt1_pen_vt1 = fr / tan_tr * (cauchy_nt1_weighted_average + pet * vt1);
          for (const auto& p : tmp_d)
            d_sigma_nt1_pen_vt1[p.first] += p.second * (cauchy_nt1_weighted_average + pet * vt1);
          for (const auto& p : cauchy_nt1_weighted_average_deriv)
            d_sigma_nt1_pen_vt1[p.first] += fr / tan_tr * p.second;
          for (const auto& p : dvt1) d_sigma_nt1_pen_vt1[p.first] += fr / tan_tr * pet * p.second;

          sigma_nt2_pen_vt2 = fr / tan_tr * (cauchy_nt2_weighted_average + pet * vt2);
          for (const auto& p : tmp_d)
            d_sigma_nt2_pen_vt2[p.first] += p.second * (cauchy_nt2_weighted_average + pet * vt2);
          for (const auto& p : cauchy_nt2_weighted_average_deriv)
            d_sigma_nt2_pen_vt2[p.first] += fr / tan_tr * p.second;
          for (const auto& p : dvt2) d_sigma_nt2_pen_vt2[p.first] += fr / tan_tr * pet * p.second;
        }

        integrate_test<dim>(-theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
            sigma_nt1_pen_vt1, d_sigma_nt1_pen_vt1, t1, dt1);
        integrate_test<dim>(-theta_2_, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt,
            sigma_nt2_pen_vt2, d_sigma_nt2_pen_vt2, t2, dt2);
        if (!two_half_pass_)
        {
          integrate_test<dim>(+theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
              sigma_nt1_pen_vt1, d_sigma_nt1_pen_vt1, t1, dt1);
          integrate_test<dim>(+theta_2_, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt,
              sigma_nt2_pen_vt2, d_sigma_nt2_pen_vt2, t2, dt2);
        }

        integrate_adjoint_test<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt1_pen_vt1,
            d_sigma_nt1_pen_vt1, sele, t1_adjoint_test_slave, deriv_t1_adjoint_test_slave);
        integrate_adjoint_test<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt2_pen_vt2,
            d_sigma_nt2_pen_vt2, sele, t2_adjoint_test_slave, deriv_t2_adjoint_test_slave);
        if (!two_half_pass_)
        {
          integrate_adjoint_test<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt1_pen_vt1,
              d_sigma_nt1_pen_vt1, mele, t1_adjoint_test_master, deriv_t1_adjoint_test_master);
          integrate_adjoint_test<dim>(theta_ / pet, jac, jacintcellmap, wgt, sigma_nt2_pen_vt2,
              d_sigma_nt2_pen_vt2, mele, t2_adjoint_test_master, deriv_t2_adjoint_test_master);
        }
      }
    }
  }
  else if ((stype_ == CONTACT::SolvingStrategy::penalty) ||
           stype_ == CONTACT::SolvingStrategy::multiscale)
  {
    if (gap < 0.)
    {
      integrate_test<dim>(-pen, sele, sval, sderiv, dsxi, jac, jacintcellmap, wgt, gap, dgapgp,
          contact_normal, deriv_contact_normal);
      if (!two_half_pass_)
      {
        integrate_test<dim>(+pen, mele, mval, mderiv, dmxi, jac, jacintcellmap, wgt, gap, dgapgp,
            contact_normal, deriv_contact_normal);
      }
    }
  }
  else
    FOUR_C_THROW("unknown algorithm");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/

template <int dim>
void CONTACT::Utils::map_gp_to_parent(Mortar::Element& moEle, double* boundary_gpcoord,
    const double wgt, Core::LinAlg::Matrix<dim, 1>& pxsi,
    Core::LinAlg::Matrix<dim, dim>& derivtravo_slave)
{
  Core::FE::CellType distype = moEle.parent_element()->shape();
  switch (distype)
  {
    case Core::FE::CellType::hex8:
      CONTACT::Utils::so_ele_gp<Core::FE::CellType::hex8, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case Core::FE::CellType::tet4:
      CONTACT::Utils::so_ele_gp<Core::FE::CellType::tet4, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case Core::FE::CellType::quad4:
      CONTACT::Utils::so_ele_gp<Core::FE::CellType::quad4, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case Core::FE::CellType::quad9:
      CONTACT::Utils::so_ele_gp<Core::FE::CellType::quad9, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case Core::FE::CellType::tri3:
      CONTACT::Utils::so_ele_gp<Core::FE::CellType::tri3, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    case Core::FE::CellType::nurbs27:
      CONTACT::Utils::so_ele_gp<Core::FE::CellType::nurbs27, dim>(
          moEle, wgt, boundary_gpcoord, pxsi, derivtravo_slave);
      break;
    default:
      FOUR_C_THROW("Nitsche contact not implemented for used (bulk) elements");
  }
}


template <int dim>
void CONTACT::IntegratorNitsche::so_ele_cauchy(Mortar::Element& moEle, double* boundary_gpcoord,
    std::vector<Core::Gen::Pairedvector<int, double>> boundary_gpcoord_lin, const double gp_wgt,
    const Core::LinAlg::Matrix<dim, 1>& normal,
    std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    const Core::LinAlg::Matrix<dim, 1>& direction,
    std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv, const double w,
    double& cauchy_nt, Core::Gen::Pairedvector<int, double>& deriv_sigma_nt,
    Core::LinAlg::SerialDenseVector& adjoint_test,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>& deriv_adjoint_test)
{
  if constexpr (dim == 3)
  {
    Core::LinAlg::Matrix<dim, 1> pxsi(Core::LinAlg::Initialization::zero);
    Core::LinAlg::Matrix<dim, dim> derivtravo_slave;
    CONTACT::Utils::map_gp_to_parent<dim>(moEle, boundary_gpcoord, gp_wgt, pxsi, derivtravo_slave);

    // define which linearizations we need
    Core::LinAlg::SerialDenseMatrix d_cauchyndir_dd{};
    Core::LinAlg::SerialDenseMatrix d2_cauchyndir_dd2{};
    Core::LinAlg::SerialDenseMatrix d2_cauchyndir_dd_dn{};
    Core::LinAlg::SerialDenseMatrix d2_cauchyndir_dd_ddir{};
    Core::LinAlg::SerialDenseMatrix d2_cauchyndir_dd_dxi{};
    Core::LinAlg::Matrix<dim, 1> d_cauchyndir_dn{};
    Core::LinAlg::Matrix<dim, 1> d_cauchyndir_ddir{};
    Core::LinAlg::Matrix<dim, 1> d_cauchyndir_dxi{};

    Discret::Elements::CauchyNDirLinearizations<dim> linearizations{};
    linearizations.d_cauchyndir_dd = &d_cauchyndir_dd;
    linearizations.d2_cauchyndir_dd2 = &d2_cauchyndir_dd2;
    linearizations.d2_cauchyndir_dd_dn = &d2_cauchyndir_dd_dn;
    linearizations.d2_cauchyndir_dd_ddir = &d2_cauchyndir_dd_ddir;
    linearizations.d2_cauchyndir_dd_dxi = &d2_cauchyndir_dd_dxi;
    linearizations.d_cauchyndir_dn = &d_cauchyndir_dn;
    linearizations.d_cauchyndir_ddir = &d_cauchyndir_ddir;
    linearizations.d_cauchyndir_dxi = &d_cauchyndir_dxi;

    auto* solid_ele = dynamic_cast<Discret::Elements::Solid*>(moEle.parent_element());
    FOUR_C_ASSERT_ALWAYS(solid_ele, "Unknown solid element type");
    const double cauchy_n_dir = solid_ele->get_normal_cauchy_stress_at_xi(
        moEle.mo_data().parent_disp(), Core::LinAlg::reinterpret_as_tensor<3>(pxsi),
        Core::LinAlg::reinterpret_as_tensor<3>(normal),
        Core::LinAlg::reinterpret_as_tensor<3>(direction), linearizations);

    cauchy_nt += w * cauchy_n_dir;

    for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
      deriv_sigma_nt[moEle.mo_data().parent_dof().at(i)] += w * d_cauchyndir_dd(i, 0);

    for (int i = 0; i < dim - 1; ++i)
    {
      for (const auto& p : boundary_gpcoord_lin[i])
        for (int k = 0; k < dim; ++k)
          deriv_sigma_nt[p.first] += d_cauchyndir_dxi(k) * derivtravo_slave(k, i) * p.second * w;
    }


    for (int d = 0; d < dim; ++d)
      for (const auto& p : normal_deriv[d])
        deriv_sigma_nt[p.first] += d_cauchyndir_dn(d) * p.second * w;

    for (int d = 0; d < dim; ++d)
      for (const auto& p : direction_deriv[d])
        deriv_sigma_nt[p.first] += d_cauchyndir_ddir(d) * p.second * w;

    if (abs(theta_) > 1.e-12)
    {
      build_adjoint_test<dim>(moEle, w, d_cauchyndir_dd, d2_cauchyndir_dd2, d2_cauchyndir_dd_dn,
          d2_cauchyndir_dd_ddir, d2_cauchyndir_dd_dxi, boundary_gpcoord_lin, derivtravo_slave,
          normal_deriv, direction_deriv, adjoint_test, deriv_adjoint_test);
    }
  }
  else
  {
    FOUR_C_THROW("Only 3D elements are supported!");
  }
}

template <int dim>
void CONTACT::IntegratorNitsche::integrate_test(const double fac, Mortar::Element& ele,
    const Core::LinAlg::SerialDenseVector& shape, const Core::LinAlg::SerialDenseMatrix& deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt,
    const double test_val, const Core::Gen::Pairedvector<int, double>& test_deriv,
    const Core::LinAlg::Matrix<dim, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv)
{
  if (abs(fac) < 1.e-16) return;

  for (int d = 0; d < dim; ++d)
  {
    const double val = fac * jac * wgt * test_val * test_dir(d);

    for (int s = 0; s < ele.num_node(); ++s)
    {
      *(ele.get_nitsche_container().rhs(
          Core::FE::get_parent_node_number_from_face_node_number(
              ele.parent_element()->shape(), ele.face_parent_number(), s) *
              dim +
          d)) += val * shape(s);
    }

    std::unordered_map<int, double> val_deriv;

    for (const auto& p : jacintcellmap)
      val_deriv[p.first] += fac * p.second * wgt * test_val * test_dir(d);
    for (const auto& p : test_deriv) val_deriv[p.first] += fac * jac * wgt * test_dir(d) * p.second;
    for (const auto& p : test_dir_deriv[d])
      val_deriv[p.first] += fac * jac * wgt * test_val * p.second;

    for (const auto& p : val_deriv)
    {
      double* row = ele.get_nitsche_container().k(p.first);
      for (int s = 0; s < ele.num_node(); ++s)
      {
        row[Core::FE::get_parent_node_number_from_face_node_number(
                ele.parent_element()->shape(), ele.face_parent_number(), s) *
                dim +
            d] += p.second * shape(s);
      }
    }

    for (int e = 0; e < dim - 1; ++e)
    {
      for (const auto& p : dxi[e])
      {
        double* row = ele.get_nitsche_container().k(p.first);
        for (int s = 0; s < ele.num_node(); ++s)
        {
          row[Core::FE::get_parent_node_number_from_face_node_number(
                  ele.parent_element()->shape(), ele.face_parent_number(), s) *
                  dim +
              d] += val * deriv(s, e) * p.second;
        }
      }
    }
  }
}

template <int dim>
void CONTACT::IntegratorNitsche::build_adjoint_test(Mortar::Element& moEle, const double fac,
    const Core::LinAlg::SerialDenseMatrix& dsntdd, const Core::LinAlg::SerialDenseMatrix& d2sntdd2,
    const Core::LinAlg::SerialDenseMatrix& d2sntDdDn,
    const Core::LinAlg::SerialDenseMatrix& d2sntDdDt,
    const Core::LinAlg::SerialDenseMatrix& d2sntDdDpxi,
    const std::vector<Core::Gen::Pairedvector<int, double>>& boundary_gpcoord_lin,
    Core::LinAlg::Matrix<dim, dim> derivtravo_slave,
    const std::vector<Core::Gen::Pairedvector<int, double>>& normal_deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& direction_deriv,
    Core::LinAlg::SerialDenseVector& adjoint_test,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>& deriv_adjoint_test)
{
  for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
  {
    adjoint_test(i) = fac * dsntdd(i, 0);
    Core::LinAlg::SerialDenseVector& at = deriv_adjoint_test[moEle.mo_data().parent_dof().at(i)];
    for (int j = 0; j < moEle.parent_element()->num_node() * dim; ++j)
      at(j) += fac * d2sntdd2(i, j);
  }

  for (int d = 0; d < dim; ++d)
  {
    for (const auto& p : normal_deriv[d])
    {
      Core::LinAlg::SerialDenseVector& at = deriv_adjoint_test[p.first];
      for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
        at(i) += fac * d2sntDdDn(i, d) * p.second;
    }
  }

  for (int d = 0; d < dim; ++d)
  {
    for (const auto& p : direction_deriv[d])
    {
      Core::LinAlg::SerialDenseVector& at = deriv_adjoint_test[p.first];
      for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
        at(i) += fac * d2sntDdDt(i, d) * p.second;
    }
  }

  Core::LinAlg::SerialDenseMatrix tmp(moEle.parent_element()->num_node() * dim, dim, false);
  Core::LinAlg::SerialDenseMatrix deriv_trafo(Teuchos::View, derivtravo_slave.data(),
      derivtravo_slave.num_rows(), derivtravo_slave.num_rows(), derivtravo_slave.num_cols());
  if (Core::LinAlg::multiply(tmp, d2sntDdDpxi, deriv_trafo)) FOUR_C_THROW("multiply failed");
  for (int d = 0; d < dim - 1; ++d)
  {
    for (const auto& p : boundary_gpcoord_lin[d])
    {
      Core::LinAlg::SerialDenseVector& at = deriv_adjoint_test[p.first];
      for (int i = 0; i < moEle.parent_element()->num_node() * dim; ++i)
        at(i) += fac * tmp(i, d) * p.second;
    }
  }
}


template <int dim>
void CONTACT::IntegratorNitsche::integrate_adjoint_test(const double fac, const double jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap, const double wgt, const double test,
    const Core::Gen::Pairedvector<int, double>& deriv_test, Mortar::Element& moEle,
    Core::LinAlg::SerialDenseVector& adjoint_test,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>& deriv_adjoint_test)
{
  if (abs(fac) < 1.e-16) return;

  Core::LinAlg::SerialDenseVector Tmp(
      Teuchos::View, moEle.get_nitsche_container().rhs(), moEle.mo_data().parent_dof().size());
  Core::LinAlg::update(fac * jac * wgt * test, adjoint_test, 1.0, Tmp);

  for (const auto& p : deriv_adjoint_test)
  {
    Core::LinAlg::SerialDenseVector Tmp(Teuchos::View, moEle.get_nitsche_container().k(p.first),
        moEle.mo_data().parent_dof().size());
    Core::LinAlg::update(fac * jac * wgt * test, p.second, 1.0, Tmp);
  }

  for (const auto& p : jacintcellmap)
  {
    Core::LinAlg::SerialDenseVector Tmp(Teuchos::View, moEle.get_nitsche_container().k(p.first),
        moEle.mo_data().parent_dof().size());
    Core::LinAlg::update(fac * p.second * wgt * test, adjoint_test, 1.0, Tmp);
  }

  for (const auto& p : deriv_test)
  {
    Core::LinAlg::SerialDenseVector Tmp(Teuchos::View, moEle.get_nitsche_container().k(p.first),
        moEle.mo_data().parent_dof().size());
    Core::LinAlg::update(fac * jac * wgt * p.second, adjoint_test, 1.0, Tmp);
  }
}

void CONTACT::Utils::nitsche_weights_and_scaling(Mortar::Element& sele, Mortar::Element& mele,
    const CONTACT::NitscheWeighting nit_wgt, const double dt, double& ws, double& wm, double& pen,
    double& pet)
{
  const double he_slave = dynamic_cast<CONTACT::Element&>(sele).trace_he();
  const double he_master = dynamic_cast<CONTACT::Element&>(mele).trace_he();

  switch (nit_wgt)
  {
    case CONTACT::NitscheWeighting::slave:
    {
      ws = 1.;
      wm = 0.;
      pen /= he_slave;
      pet /= he_slave;
    }
    break;
    case CONTACT::NitscheWeighting::master:
    {
      wm = 1.;
      ws = 0.;
      pen /= he_master;
      pet /= he_master;
    }
    break;
    case CONTACT::NitscheWeighting::harmonic:
      ws = 1. / he_master;
      wm = 1. / he_slave;
      ws /= (ws + wm);
      wm = 1. - ws;
      pen = ws * pen / he_slave + wm * pen / he_master;
      pet = ws * pet / he_slave + wm * pet / he_master;

      break;
    default:
      FOUR_C_THROW("unknown Nitsche weighting");
      break;
  }
}

template <int dim>
void CONTACT::Utils::rel_vel(Mortar::Element& ele, const Core::LinAlg::SerialDenseVector& shape,
    const Core::LinAlg::SerialDenseMatrix& deriv,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dxi, const double fac,
    Core::LinAlg::Matrix<dim, 1>& relVel,
    std::vector<Core::Gen::Pairedvector<int, double>>& relVel_deriv)
{
  for (int n = 0; n < ele.num_node(); ++n)
  {
    for (int d = 0; d < dim; ++d)
    {
      relVel(d) += fac * shape(n) * (ele.get_nodal_coords(d, n) - ele.get_nodal_coords_old(d, n));
      relVel_deriv[d][dynamic_cast<Mortar::Node*>(ele.nodes()[n])->dofs()[d]] += fac * shape(n);

      for (int sd = 0; sd < dim - 1; ++sd)
      {
        for (const auto& p : dxi[sd])
        {
          relVel_deriv[d][p.first] +=
              fac * (ele.get_nodal_coords(d, n) - ele.get_nodal_coords_old(d, n)) * deriv(n, sd) *
              p.second;
        }
      }
    }
  }
}


template <int dim>
void CONTACT::Utils::rel_vel_invariant(Mortar::Element& sele, const double* sxi,
    const std::vector<Core::Gen::Pairedvector<int, double>>& derivsxi,
    const Core::LinAlg::SerialDenseVector& sval, const Core::LinAlg::SerialDenseMatrix& sderiv,
    Mortar::Element& mele, const double* mxi,
    const std::vector<Core::Gen::Pairedvector<int, double>>& derivmxi,
    const Core::LinAlg::SerialDenseVector& mval, const Core::LinAlg::SerialDenseMatrix& mderiv,
    const double& gap, const Core::Gen::Pairedvector<int, double>& deriv_gap,
    Core::LinAlg::Matrix<dim, 1>& relVel,
    std::vector<Core::Gen::Pairedvector<int, double>>& relVel_deriv, const double fac)
{
  Core::LinAlg::Matrix<3, 1> n_old;
  Core::LinAlg::Matrix<3, 2> d_n_old_dxi;
  dynamic_cast<CONTACT::Element&>(sele).old_unit_normal_at_xi(sxi, n_old, d_n_old_dxi);
  for (int i = 0; i < sele.num_node(); ++i)
  {
    for (int d = 0; d < dim; ++d)
    {
      relVel(d) += sele.get_nodal_coords_old(d, i) * sval(i) * fac;

      for (int e = 0; e < dim - 1; ++e)
        for (const auto& p : derivsxi[e])
          relVel_deriv[d][p.first] +=
              sele.get_nodal_coords_old(d, i) * sderiv(i, e) * p.second * fac;
    }
  }

  for (int i = 0; i < mele.num_node(); ++i)
  {
    for (int d = 0; d < dim; ++d)
    {
      relVel(d) -= mele.get_nodal_coords_old(d, i) * mval(i) * fac;

      for (int e = 0; e < dim - 1; ++e)
        for (const auto& p : derivmxi[e])
          relVel_deriv[d][p.first] -=
              mele.get_nodal_coords_old(d, i) * mderiv(i, e) * p.second * fac;
    }
  }
  for (int d = 0; d < dim; ++d)
  {
    relVel(d) += n_old(d) * gap * fac;

    for (int e = 0; e < dim - 1; ++e)
      for (const auto& p : derivsxi[e])
        relVel_deriv[d][p.first] += gap * d_n_old_dxi(d, e) * p.second * fac;

    for (const auto& p : deriv_gap) relVel_deriv[d][p.first] += n_old(d) * p.second * fac;
  }
}

template <int dim>
void CONTACT::Utils::vector_scalar_product(const Core::LinAlg::Matrix<dim, 1>& v1,
    const std::vector<Core::Gen::Pairedvector<int, double>>& v1d,
    const Core::LinAlg::Matrix<dim, 1>& v2,
    const std::vector<Core::Gen::Pairedvector<int, double>>& v2d, double& val,
    Core::Gen::Pairedvector<int, double>& val_deriv)
{
  val = v1.dot(v2);
  val_deriv.clear();
  val_deriv.resize(v1d[0].size() + v2d[0].size());
  for (int d = 0; d < dim; ++d)
  {
    for (const auto& p : v1d[d]) val_deriv[p.first] += v2(d) * p.second;
    for (const auto& p : v2d[d]) val_deriv[p.first] += v1(d) * p.second;
  }
}

void CONTACT::Utils::build_tangent_vectors3_d(const double* np,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dn, double* t1p,
    std::vector<Core::Gen::Pairedvector<int, double>>& dt1, double* t2p,
    std::vector<Core::Gen::Pairedvector<int, double>>& dt2)
{
  const Core::LinAlg::Matrix<3, 1> n(np, false);
  Core::LinAlg::Matrix<3, 1> t1(t1p, true);
  Core::LinAlg::Matrix<3, 1> t2(t2p, true);

  bool z = true;
  Core::LinAlg::Matrix<3, 1> tmp;
  tmp(2) = 1.;
  if (abs(tmp.dot(n)) > 1. - 1.e-4)
  {
    tmp(0) = 1.;
    tmp(2) = 0.;
    z = false;
  }

  t1.cross_product(tmp, n);
  dt1.resize(3, std::max(dn[0].size(), std::max(dn[1].size(), dn[2].size())));
  dt2.resize(3, std::max(dn[0].size(), std::max(dn[1].size(), dn[2].size())));

  const double lt1 = t1.norm2();
  t1.scale(1. / lt1);
  Core::LinAlg::Matrix<3, 3> p;
  for (int i = 0; i < 3; ++i) p(i, i) = 1.;
  p.multiply_nt(-1., t1, t1, 1.);
  p.scale(1. / lt1);
  if (z)
  {
    for (const auto& i : dn[1])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] -= p(d, 0) * i.second;

    for (const auto& i : dn[0])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] += p(d, 1) * i.second;
  }
  else
  {
    for (const auto& i : dn[2])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] -= p(d, 1) * i.second;

    for (const auto& i : dn[1])
      for (int d = 0; d < 3; ++d) dt1[d][i.first] += p(d, 2) * i.second;
  }

  t2.cross_product(n, t1);
  if (abs(t2.norm2() - 1.) > 1.e-10) FOUR_C_THROW("this should already form an orthonormal basis");

  for (const auto& i : dn[0])
  {
    dt2[1][i.first] -= t1(2) * (i.second);
    dt2[2][i.first] += t1(1) * (i.second);
  }
  for (const auto& i : dn[1])
  {
    dt2[0][i.first] += t1(2) * (i.second);
    dt2[2][i.first] -= t1(0) * (i.second);
  }
  for (const auto& i : dn[2])
  {
    dt2[0][i.first] -= t1(1) * (i.second);
    dt2[1][i.first] += t1(0) * (i.second);
  }
  for (const auto& i : dt1[0])
  {
    dt2[1][i.first] += n(2) * (i.second);
    dt2[2][i.first] -= n(1) * (i.second);
  }
  for (const auto& i : dt1[1])
  {
    dt2[0][i.first] -= n(2) * (i.second);
    dt2[2][i.first] += n(0) * (i.second);
  }
  for (const auto& i : dt1[2])
  {
    dt2[0][i.first] += n(1) * (i.second);
    dt2[1][i.first] -= n(0) * (i.second);
  }
}

template <int dim>
void CONTACT::Utils::build_tangent_vectors(const double* np,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dn, double* t1p,
    std::vector<Core::Gen::Pairedvector<int, double>>& dt1, double* t2p,
    std::vector<Core::Gen::Pairedvector<int, double>>& dt2)
{
  if (dim == 3)
    build_tangent_vectors3_d(np, dn, t1p, dt1, t2p, dt2);
  else
    FOUR_C_THROW("not implemented");
}

template void CONTACT::Utils::build_tangent_vectors<2>(const double*,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, double*,
    std::vector<Core::Gen::Pairedvector<int, double>>&, double*,
    std::vector<Core::Gen::Pairedvector<int, double>>&);

template void CONTACT::Utils::build_tangent_vectors<3>(const double*,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, double*,
    std::vector<Core::Gen::Pairedvector<int, double>>&, double*,
    std::vector<Core::Gen::Pairedvector<int, double>>&);



template void CONTACT::IntegratorNitsche::integrate_test<2>(const double, Mortar::Element&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>& i, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const Core::LinAlg::Matrix<2, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);
template void CONTACT::IntegratorNitsche::integrate_test<3>(const double, Mortar::Element&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>& i, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const Core::LinAlg::Matrix<3, 1>& test_dir,
    const std::vector<Core::Gen::Pairedvector<int, double>>& test_dir_deriv);

template void CONTACT::IntegratorNitsche::integrate_adjoint_test<2>(const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, Mortar::Element&, Core::LinAlg::SerialDenseVector&,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>&);

template void CONTACT::IntegratorNitsche::integrate_adjoint_test<3>(const double, const double,
    const Core::Gen::Pairedvector<int, double>&, const double, const double,
    const Core::Gen::Pairedvector<int, double>&, Mortar::Element&, Core::LinAlg::SerialDenseVector&,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>&);

template void CONTACT::IntegratorNitsche::build_adjoint_test<2>(Mortar::Element&, const double,
    const Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::SerialDenseMatrix&,
    const Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::SerialDenseMatrix&,
    const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, Core::LinAlg::Matrix<2, 2>,
    const std::vector<Core::Gen::Pairedvector<int, double>>&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, Core::LinAlg::SerialDenseVector&,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>&);

template void CONTACT::IntegratorNitsche::build_adjoint_test<3>(Mortar::Element&, const double,
    const Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::SerialDenseMatrix&,
    const Core::LinAlg::SerialDenseMatrix&, const Core::LinAlg::SerialDenseMatrix&,
    const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, Core::LinAlg::Matrix<3, 3>,
    const std::vector<Core::Gen::Pairedvector<int, double>>&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, Core::LinAlg::SerialDenseVector&,
    Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseVector>&);


template void CONTACT::Utils::rel_vel<2>(Mortar::Element&, const Core::LinAlg::SerialDenseVector&,
    const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, const double,
    Core::LinAlg::Matrix<2, 1>&, std::vector<Core::Gen::Pairedvector<int, double>>&);

template void CONTACT::Utils::rel_vel<3>(Mortar::Element&, const Core::LinAlg::SerialDenseVector&,
    const Core::LinAlg::SerialDenseMatrix&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, const double,
    Core::LinAlg::Matrix<3, 1>&, std::vector<Core::Gen::Pairedvector<int, double>>&);

template void CONTACT::Utils::vector_scalar_product<2>(const Core::LinAlg::Matrix<2, 1>&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, const Core::LinAlg::Matrix<2, 1>&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, double&,
    Core::Gen::Pairedvector<int, double>&);
template void CONTACT::Utils::vector_scalar_product<3>(const Core::LinAlg::Matrix<3, 1>&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, const Core::LinAlg::Matrix<3, 1>&,
    const std::vector<Core::Gen::Pairedvector<int, double>>&, double&,
    Core::Gen::Pairedvector<int, double>&);

template void CONTACT::Utils::rel_vel_invariant<2>(Mortar::Element&, const double*,
    const std::vector<Core::Gen::Pairedvector<int, double>>&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&,
    Mortar::Element&, const double*, const std::vector<Core::Gen::Pairedvector<int, double>>&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&, const double&,
    const Core::Gen::Pairedvector<int, double>&, Core::LinAlg::Matrix<2, 1>&,
    std::vector<Core::Gen::Pairedvector<int, double>>&, const double);

template void CONTACT::Utils::rel_vel_invariant<3>(Mortar::Element&, const double*,
    const std::vector<Core::Gen::Pairedvector<int, double>>&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&,
    Mortar::Element&, const double*, const std::vector<Core::Gen::Pairedvector<int, double>>&,
    const Core::LinAlg::SerialDenseVector&, const Core::LinAlg::SerialDenseMatrix&, const double&,
    const Core::Gen::Pairedvector<int, double>&, Core::LinAlg::Matrix<3, 1>&,
    std::vector<Core::Gen::Pairedvector<int, double>>&, const double);

template void CONTACT::Utils::map_gp_to_parent<2>(Mortar::Element&, double*, const double,
    Core::LinAlg::Matrix<2, 1>&, Core::LinAlg::Matrix<2, 2>&);

template void CONTACT::Utils::map_gp_to_parent<3>(Mortar::Element&, double*, const double,
    Core::LinAlg::Matrix<3, 1>&, Core::LinAlg::Matrix<3, 3>&);

FOUR_C_NAMESPACE_CLOSE
