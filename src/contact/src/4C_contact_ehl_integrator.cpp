// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_ehl_integrator.hpp"

#include "4C_contact_element.hpp"
#include "4C_contact_nitsche_integrator.hpp"  // for CONTACT::Utils:: functions
#include "4C_contact_node.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorEhl::integrate_gp_3d(Mortar::Element& source_elem,
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
  // check bound
  bool bound = false;
  for (int i = 0; i < source_elem.num_node(); ++i)
    if (dynamic_cast<CONTACT::Node*>(source_elem.nodes()[i])->is_on_boundor_ce())
      FOUR_C_THROW("no boundary modification for EHL implemented");

  // is quadratic case?
  bool quad = source_elem.is_quad();

  // weighted gap
  gp_3d_w_gap(source_elem, source_val, lm_val, &gap, jac, wgt, quad);
  for (int j = 0; j < source_elem.num_node(); ++j)
    gp_g_lin(j, source_elem, target_elem, source_val, target_val, lm_val, source_deriv, lm_deriv,
        gap, normal, jac, wgt, deriv_gap, derivjac, source_derivs_xi, dualmap);

  // integrate D and M matrix
  gp_dm(source_elem, target_elem, lm_val, source_val, target_val, jac, wgt, bound);
  gp_3d_dm_lin(source_elem, target_elem, source_val, target_val, lm_val, source_deriv, target_deriv,
      lm_deriv, wgt, jac, source_derivs_xi, target_derivs_xi, derivjac, dualmap);

  // get second derivative of shape function
  Core::LinAlg::SerialDenseMatrix ssecderiv(source_elem.num_node(), 3);
  source_elem.evaluate2nd_deriv_shape(source_xi, ssecderiv, source_elem.num_node());

  // weighted surface gradient
  gp_weighted_surf_grad_and_deriv(source_elem, source_xi, source_derivs_xi, lm_val, lm_deriv,
      dualmap, source_val, source_deriv, ssecderiv, wgt, jac, derivjac);

  //  // weighted tangential velocity (average and relative)
  gp_weighted_av_rel_vel(source_elem, target_elem, source_val, lm_val, target_val, source_deriv,
      target_deriv, lm_deriv, dualmap, wgt, jac, derivjac, normal, dnmap_unit, gap, deriv_gap,
      source_xi, target_xi, source_derivs_xi, target_derivs_xi);
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void CONTACT::IntegratorEhl::integrate_gp_2d(Mortar::Element& source_elem,
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
  FOUR_C_THROW("2D EHL integration not supported");
}


void CONTACT::IntegratorEhl::gp_weighted_surf_grad_and_deriv(Mortar::Element& source_elem,
    const double* xi, const std::vector<Core::Gen::Pairedvector<int, double>>& d_source_xi_gp,
    const Core::LinAlg::SerialDenseVector& lm_val, const Core::LinAlg::SerialDenseMatrix& lm_deriv,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap,
    const Core::LinAlg::SerialDenseVector& source_val,
    const Core::LinAlg::SerialDenseMatrix& source_deriv,
    const Core::LinAlg::SerialDenseMatrix& sderiv2, const double& wgt, const double& jac,
    const Core::Gen::Pairedvector<int, double>& jacintcellmap)
{
  // empty local basis vectors
  std::vector<std::vector<double>> gxi(2, std::vector<double>(3, 0));

  // metrics routine gives local basis vectors
  source_elem.metrics(xi, gxi.at(0).data(), gxi.at(1).data());

  if (n_dim() == 2)
  {
    gxi.at(1).at(0) = gxi.at(1).at(1) = 0.;
    gxi.at(1).at(2) = 1.;
  }

  Core::LinAlg::Matrix<3, 1> t1(gxi.at(0).data(), true);
  Core::LinAlg::Matrix<3, 1> t2(gxi.at(1).data(), true);
  Core::LinAlg::Matrix<3, 1> n;
  n.cross_product(t1, t2);
  n.scale(1. / n.norm2());
  Core::LinAlg::Matrix<3, 3> covariant_metric;
  for (int i = 0; i < 3; ++i)
  {
    covariant_metric(i, 0) = t1(i);
    covariant_metric(i, 1) = t2(i);
    covariant_metric(i, 2) = n(i);
  }
  Core::LinAlg::Matrix<3, 3> contravariant_metric;
  Core::LinAlg::invert(contravariant_metric, covariant_metric);

  std::vector<std::vector<double>> gxi_contra(2, std::vector<double>(3, 0));
  for (int i = 0; i < n_dim() - 1; ++i)
    for (int d = 0; d < n_dim(); ++d) gxi_contra.at(i).at(d) = contravariant_metric(i, d);

  for (int a = 0; a < source_elem.num_node(); ++a)
  {
    Core::Nodes::Node* node = source_elem.nodes()[a];
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(node);
    if (!cnode) FOUR_C_THROW("this is not a contact node");

    for (int c = 0; c < source_elem.num_node(); ++c)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data()
              .get_surf_grad()[dynamic_cast<CONTACT::Node*>(source_elem.nodes()[c])->dofs()[0]];
      for (int d = 0; d < n_dim(); ++d)
        for (int al = 0; al < n_dim() - 1; ++al)
          tmp(d) += wgt * jac * lm_val(a) * source_deriv(c, al) * gxi_contra.at(al).at(d);
    }

    for (int c = 0; c < source_elem.num_node(); ++c)
    {
      for (auto p = jacintcellmap.begin(); p != jacintcellmap.end(); ++p)
      {
        Core::LinAlg::Matrix<3, 1>& tmp = cnode->ehl_data().get_surf_grad_deriv()[p
                ->first][dynamic_cast<CONTACT::Node*>(source_elem.nodes()[c])->dofs()[0]];
        for (int d = 0; d < n_dim(); ++d)
          for (int al = 0; al < n_dim() - 1; ++al)
            tmp(d) += wgt * p->second * lm_val(a) * source_deriv(c, al) * gxi_contra.at(al).at(d);
      }
      for (int e = 0; e < n_dim() - 1; ++e)
        for (auto p = d_source_xi_gp.at(e).begin(); p != d_source_xi_gp.at(e).end(); ++p)
        {
          Core::LinAlg::Matrix<3, 1>& tmp = cnode->ehl_data().get_surf_grad_deriv()[p
                  ->first][dynamic_cast<CONTACT::Node*>(source_elem.nodes()[c])->dofs()[0]];
          for (int d = 0; d < n_dim(); ++d)
            for (int al = 0; al < n_dim() - 1; ++al)
              tmp(d) += wgt * jac * lm_deriv(a, e) * p->second * source_deriv(c, al) *
                        gxi_contra.at(al).at(d);
        }

      for (auto p = dualmap.begin(); p != dualmap.end(); ++p)
      {
        Core::LinAlg::Matrix<3, 1>& tmp = cnode->ehl_data().get_surf_grad_deriv()[p
                ->first][dynamic_cast<CONTACT::Node*>(source_elem.nodes()[c])->dofs()[0]];
        for (int d = 0; d < n_dim(); ++d)
          for (int al = 0; al < n_dim() - 1; ++al)
            for (int m = 0; m < source_elem.num_node(); ++m)
              tmp(d) += wgt * jac * p->second(a, m) * source_val(m) * source_deriv(c, al) *
                        gxi_contra.at(al).at(d);
      }
    }
  }
}

void CONTACT::IntegratorEhl::gp_weighted_av_rel_vel(Mortar::Element& source_elem,
    Mortar::Element& target_elem, const Core::LinAlg::SerialDenseVector& source_val,
    const Core::LinAlg::SerialDenseVector& lm_val,
    const Core::LinAlg::SerialDenseVector& target_val,
    const Core::LinAlg::SerialDenseMatrix& source_deriv,
    const Core::LinAlg::SerialDenseMatrix& target_deriv,
    const Core::LinAlg::SerialDenseMatrix& lm_deriv,
    const Core::Gen::Pairedvector<int, Core::LinAlg::SerialDenseMatrix>& dualmap, const double& wgt,
    const double& jac, const Core::Gen::Pairedvector<int, double>& derivjac, const double* normal,
    const std::vector<Core::Gen::Pairedvector<int, double>>& dnmap_unit, const double& gap,
    const Core::Gen::Pairedvector<int, double>& deriv_gap, const double* source_xi,
    const double* target_xi,
    const std::vector<Core::Gen::Pairedvector<int, double>>& source_derivs_xi,
    const std::vector<Core::Gen::Pairedvector<int, double>>& target_derivs_xi)
{
  constexpr int dim = 3;
  if (IntegratorEhl::n_dim() != 3)
    FOUR_C_THROW("dimension inconsistency, or is this not implemented for all spatial dimensions?");

  Core::LinAlg::Matrix<dim, 1> t1, t2;
  std::vector<Core::Gen::Pairedvector<int, double>> dt1, dt2;
  Core::LinAlg::Matrix<dim, 1> relVel;
  std::vector<Core::Gen::Pairedvector<int, double>> relVel_deriv(
      dim, source_elem.num_node() * dim + target_elem.num_node() * dim +
               source_derivs_xi[0].size() + target_derivs_xi[0].size());
  double vt1, vt2;
  Core::Gen::Pairedvector<int, double> dvt1(0);
  Core::Gen::Pairedvector<int, double> dvt2(0);

  CONTACT::Utils::build_tangent_vectors<dim>(normal, dnmap_unit, t1.data(), dt1, t2.data(), dt2);
  CONTACT::Utils::rel_vel_invariant<dim>(source_elem, source_xi, source_derivs_xi, source_val,
      source_deriv, target_elem, target_xi, target_derivs_xi, target_val, target_deriv, gap,
      deriv_gap, relVel, relVel_deriv, -.5);

  CONTACT::Utils::vector_scalar_product<dim>(t1, dt1, relVel, relVel_deriv, vt1, dvt1);
  CONTACT::Utils::vector_scalar_product<dim>(t2, dt2, relVel, relVel_deriv, vt2, dvt2);

  for (int i = 0; i < source_elem.num_node(); ++i)
  {
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_elem.nodes()[i]);
    for (int d = 0; d < dim; ++d)
      cnode->ehl_data().get_weighted_rel_tang_vel()(d) +=
          jac * wgt * lm_val(i) * (vt1 * t1(d) + vt2 * t2(d));

    for (auto p = derivjac.begin(); p != derivjac.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d)
        tmp(d) += p->second * wgt * lm_val(i) * (vt1 * t1(d) + vt2 * t2(d));
    }

    for (int e = 0; e < dim - 1; ++e)
      for (auto p = source_derivs_xi.at(e).begin(); p != source_derivs_xi.at(e).end(); ++p)
      {
        Core::LinAlg::Matrix<3, 1>& tmp =
            cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first];
        for (int d = 0; d < dim; ++d)
          tmp(d) += jac * wgt * lm_deriv(i, e) * p->second * (vt1 * t1(d) + vt2 * t2(d));
      }

    for (auto p = dualmap.begin(); p != dualmap.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d)
        for (int m = 0; m < source_elem.num_node(); ++m)
          tmp(d) += jac * wgt * p->second(i, m) * source_val(m) * (vt1 * t1(d) + vt2 * t2(d));
    }

    for (auto p = dvt1.begin(); p != dvt1.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d) tmp(d) += jac * wgt * lm_val(i) * p->second * t1(d);
    }

    for (int d = 0; d < dim; ++d)
      for (auto p = dt1.at(d).begin(); p != dt1.at(d).end(); ++p)
        cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first](d) +=
            jac * wgt * lm_val(i) * vt1 * p->second;

    for (auto p = dvt2.begin(); p != dvt2.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d) tmp(d) += jac * wgt * lm_val(i) * p->second * t2(d);
    }

    for (int d = 0; d < dim; ++d)
      for (auto p = dt2.at(d).begin(); p != dt2.at(d).end(); ++p)
        cnode->ehl_data().get_weighted_rel_tang_vel_deriv()[p->first](d) +=
            jac * wgt * lm_val(i) * vt2 * p->second;
  }

  CONTACT::Utils::rel_vel<dim>(
      source_elem, source_val, source_deriv, source_derivs_xi, -1., relVel, relVel_deriv);
  CONTACT::Utils::vector_scalar_product<dim>(t1, dt1, relVel, relVel_deriv, vt1, dvt1);
  CONTACT::Utils::vector_scalar_product<dim>(t2, dt2, relVel, relVel_deriv, vt2, dvt2);

  for (int i = 0; i < source_elem.num_node(); ++i)
  {
    CONTACT::Node* cnode = dynamic_cast<CONTACT::Node*>(source_elem.nodes()[i]);
    for (int d = 0; d < dim; ++d)
      cnode->ehl_data().get_weighted_av_tang_vel()(d) -=
          jac * wgt * lm_val(i) * (vt1 * t1(d) + vt2 * t2(d));

    for (auto p = derivjac.begin(); p != derivjac.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d)
        tmp(d) -= p->second * wgt * lm_val(i) * (vt1 * t1(d) + vt2 * t2(d));
    }

    for (int e = 0; e < dim - 1; ++e)
      for (auto p = source_derivs_xi.at(e).begin(); p != source_derivs_xi.at(e).end(); ++p)
      {
        Core::LinAlg::Matrix<3, 1>& tmp =
            cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first];
        for (int d = 0; d < dim; ++d)
          tmp(d) -= jac * wgt * lm_deriv(i, e) * p->second * (vt1 * t1(d) + vt2 * t2(d));
      }

    for (auto p = dualmap.begin(); p != dualmap.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d)
        for (int m = 0; m < source_elem.num_node(); ++m)
          tmp(d) -= jac * wgt * p->second(i, m) * source_val(m) * (vt1 * t1(d) + vt2 * t2(d));
    }

    for (auto p = dvt1.begin(); p != dvt1.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d) tmp(d) -= jac * wgt * lm_val(i) * p->second * t1(d);
    }

    for (int d = 0; d < dim; ++d)
      for (auto p = dt1.at(d).begin(); p != dt1.at(d).end(); ++p)
        cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first](d) -=
            jac * wgt * lm_val(i) * vt1 * p->second;

    for (auto p = dvt2.begin(); p != dvt2.end(); ++p)
    {
      Core::LinAlg::Matrix<3, 1>& tmp =
          cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first];
      for (int d = 0; d < dim; ++d) tmp(d) -= jac * wgt * lm_val(i) * p->second * t2(d);
    }

    for (int d = 0; d < dim; ++d)
      for (auto p = dt2.at(d).begin(); p != dt2.at(d).end(); ++p)
        cnode->ehl_data().get_weighted_av_tang_vel_deriv()[p->first](d) -=
            jac * wgt * lm_val(i) * vt2 * p->second;
  }
}

FOUR_C_NAMESPACE_CLOSE
