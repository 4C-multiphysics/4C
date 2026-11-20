// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_projector.hpp"

#include "4C_contact_node.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mortar_calc_utils.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  impl. for aux.-plane based projection                    farah 01/14|
 *----------------------------------------------------------------------*/
Mortar::Projector* Mortar::Projector::impl(Mortar::Element& ele)
{
  switch (ele.shape())
  {
    case Core::FE::CellType::quad4:
    {
      return ProjectorCalc<Core::FE::CellType::quad4>::instance();
    }
    case Core::FE::CellType::quad8:
    {
      return ProjectorCalc<Core::FE::CellType::quad8>::instance();
    }
    case Core::FE::CellType::quad9:
    {
      return ProjectorCalc<Core::FE::CellType::quad9>::instance();
    }
    case Core::FE::CellType::tri3:
    {
      return ProjectorCalc<Core::FE::CellType::tri3>::instance();
    }
    case Core::FE::CellType::tri6:
    {
      return ProjectorCalc<Core::FE::CellType::tri6>::instance();
    }
    case Core::FE::CellType::line2:
    {
      return ProjectorCalc<Core::FE::CellType::line2>::instance();
    }
    case Core::FE::CellType::line3:
    {
      return ProjectorCalc<Core::FE::CellType::line3>::instance();
    }
      //==================================================
      //                     NURBS
      //==================================================
    case Core::FE::CellType::nurbs2:
    {
      return ProjectorCalc<Core::FE::CellType::nurbs2>::instance();
    }
    case Core::FE::CellType::nurbs3:
    {
      return ProjectorCalc<Core::FE::CellType::nurbs3>::instance();
    }
    case Core::FE::CellType::nurbs4:
    {
      return ProjectorCalc<Core::FE::CellType::nurbs4>::instance();
    }
    case Core::FE::CellType::nurbs8:
    {
      return ProjectorCalc<Core::FE::CellType::nurbs8>::instance();
    }
    case Core::FE::CellType::nurbs9:
    {
      return ProjectorCalc<Core::FE::CellType::nurbs9>::instance();
    }
    default:
      FOUR_C_THROW(
          "Element shape {} ({} nodes) not activated. Just do it.", ele.shape(), ele.num_node());
      break;
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 |  impl. for element based projection                       farah 04/14|
 *----------------------------------------------------------------------*/
Mortar::Projector* Mortar::Projector::impl(const Mortar::Element& sele, const Mortar::Element& mele)
{
  switch (sele.shape())
  {
    case Core::FE::CellType::quad4:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad4,
              Core::FE::CellType::quad4>::instance();
        }
        case Core::FE::CellType::quad8:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad4,
              Core::FE::CellType::quad8>::instance();
        }
        case Core::FE::CellType::quad9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad4,
              Core::FE::CellType::quad9>::instance();
        }
        case Core::FE::CellType::tri3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad4,
              Core::FE::CellType::tri3>::instance();
        }
        case Core::FE::CellType::tri6:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad4,
              Core::FE::CellType::tri6>::instance();
        }
        case Core::FE::CellType::nurbs9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad4,
              Core::FE::CellType::nurbs9>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::quad8:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad8,
              Core::FE::CellType::quad4>::instance();
        }
        case Core::FE::CellType::quad8:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad8,
              Core::FE::CellType::quad8>::instance();
        }
        case Core::FE::CellType::quad9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad8,
              Core::FE::CellType::quad9>::instance();
        }
        case Core::FE::CellType::tri3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad8,
              Core::FE::CellType::tri3>::instance();
        }
        case Core::FE::CellType::tri6:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad8,
              Core::FE::CellType::tri6>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::quad9:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad9,
              Core::FE::CellType::quad4>::instance();
        }
        case Core::FE::CellType::quad8:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad9,
              Core::FE::CellType::quad8>::instance();
        }
        case Core::FE::CellType::quad9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad9,
              Core::FE::CellType::quad9>::instance();
        }
        case Core::FE::CellType::tri3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad9,
              Core::FE::CellType::tri3>::instance();
        }
        case Core::FE::CellType::tri6:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::quad9,
              Core::FE::CellType::tri6>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::tri3:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri3,
              Core::FE::CellType::quad4>::instance();
        }
        case Core::FE::CellType::quad8:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri3,
              Core::FE::CellType::quad8>::instance();
        }
        case Core::FE::CellType::quad9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri3,
              Core::FE::CellType::quad9>::instance();
        }
        case Core::FE::CellType::tri3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri3,
              Core::FE::CellType::tri3>::instance();
        }
        case Core::FE::CellType::tri6:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri3,
              Core::FE::CellType::tri6>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::tri6:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::quad4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri6,
              Core::FE::CellType::quad4>::instance();
        }
        case Core::FE::CellType::quad8:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri6,
              Core::FE::CellType::quad8>::instance();
        }
        case Core::FE::CellType::quad9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri6,
              Core::FE::CellType::quad9>::instance();
        }
        case Core::FE::CellType::tri3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri6,
              Core::FE::CellType::tri3>::instance();
        }
        case Core::FE::CellType::tri6:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::tri6,
              Core::FE::CellType::tri6>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::line2:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::line2:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::line2,
              Core::FE::CellType::line2>::instance();
        }
        case Core::FE::CellType::line3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::line2,
              Core::FE::CellType::line3>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::line3:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::line2:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::line3,
              Core::FE::CellType::line2>::instance();
        }
        case Core::FE::CellType::line3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::line3,
              Core::FE::CellType::line3>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
      //==================================================
      //                     NURBS
      //==================================================
    case Core::FE::CellType::nurbs2:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs2:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs2,
              Core::FE::CellType::nurbs2>::instance();
        }
        case Core::FE::CellType::nurbs3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs2,
              Core::FE::CellType::nurbs3>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::nurbs3:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs2:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs3,
              Core::FE::CellType::nurbs2>::instance();
        }
        case Core::FE::CellType::nurbs3:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs3,
              Core::FE::CellType::nurbs3>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::nurbs4:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs4,
              Core::FE::CellType::nurbs4>::instance();
        }
        case Core::FE::CellType::nurbs9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs4,
              Core::FE::CellType::nurbs9>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    case Core::FE::CellType::nurbs9:
    {
      switch (mele.shape())
      {
        case Core::FE::CellType::nurbs4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs9,
              Core::FE::CellType::nurbs4>::instance();
        }
        case Core::FE::CellType::nurbs9:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs9,
              Core::FE::CellType::nurbs9>::instance();
        }
        case Core::FE::CellType::quad4:
        {
          return ProjectorCalcEleBased<Core::FE::CellType::nurbs9,
              Core::FE::CellType::quad4>::instance();
        }
        default:
          FOUR_C_THROW("Element shape not supported!");
          break;
      }
      break;
    }
    default:
      FOUR_C_THROW(
          "Element shape {} ({} nodes) not activated. Just do it.", sele.shape(), sele.num_node());
      break;
  }
  return nullptr;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            farah 01/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Mortar::ProjectorCalc<distype>::ProjectorCalc()
{
  // nothing
}

/*----------------------------------------------------------------------*
 |  ctor ele-based (public)                                  farah 04/14|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
Mortar::ProjectorCalcEleBased<distype_s, distype_m>::ProjectorCalcEleBased()
{
  // nothing
}

template <Core::FE::CellType distype>
Mortar::ProjectorCalc<distype>* Mortar::ProjectorCalc<distype>::instance(
    Core::Utils::SingletonAction action)
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []()
      {
        return std::unique_ptr<Mortar::ProjectorCalc<distype>>(
            new Mortar::ProjectorCalc<distype>());
      });

  return singleton_owner.instance(action);
}

template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
Mortar::ProjectorCalcEleBased<distype_s, distype_m>*
Mortar::ProjectorCalcEleBased<distype_s, distype_m>::instance(Core::Utils::SingletonAction action)
{
  static Core::Utils::SingletonOwner<Mortar::ProjectorCalcEleBased<distype_s, distype_m>>
      singleton_owner(
          []()
          {
            return std::unique_ptr<Mortar::ProjectorCalcEleBased<distype_s, distype_m>>(
                new Mortar::ProjectorCalcEleBased<distype_s, distype_m>());
          });

  return singleton_owner.instance(action);
}


/*----------------------------------------------------------------------*
 |  Project a node along its nodal normal (public)            popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_nodal_normal(
    const Mortar::Node& node, const Mortar::Element& ele, double* xi) const
{
  bool ok = true;
  if (ndim_ == 2)
  {
    // local Newton iteration for xi, start in the element middle
    double eta[2] = {0.0, 0.0};
    double f = 0.0;
    double df = 0.0;
    int k = 0;

    for (k = 0; k < MORTARMAXITER; ++k)
    {
      f = evaluate_f_nodal_normal(node, ele, eta);
      if (abs(f) < MORTARCONVTOL) break;
      df = evaluate_grad_f_nodal_normal(node, ele, eta);
      if (abs(df) < 1.0e-12)
      {
        xi[0] = 1.0e12;
        return false;
        FOUR_C_THROW("Singular Jacobian for projection");
      }
      eta[0] += (-f) / df;
    }

    // get the result
    xi[0] = eta[0];

    // Newton iteration unconverged
    if (abs(f) > MORTARCONVTOL)
    {
      ok = false;
      xi[0] = 1.0e12;

      // Here (S->M projection) we only give a warning, no error!!!
      // This iteration sometimes diverges, when the projection is far off.
      // These cases are harmless, as these nodes then do not participate in
      // the overlap detection anyway!
      // std::cout << "***WARNING*** ProjectNodalNormal:" << " Newton unconverged for NodeID "
      //     << node.Id() << " and Mortar::ElementID " << ele.Id() << std::endl;
    }

    /*
     // Newton iteration converged
     else
     {
     std::cout << "Newton iteration converged in " << k << " step(s)!" << std::endl
     << "The result is: " << xi[0] << std::endl;
     }*/
  }

  else
    FOUR_C_THROW("ProjectNodalNormal: Called 2D version for 3D problem!");

  return ok;
}

/*----------------------------------------------------------------------*
 |  Project a node along element's normal field (public)      popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_element_normal(
    const Mortar::Node& node, const Mortar::Element& ele, double* xi) const
{
  bool ok = true;
  if (ndim_ == 2)
  {
    // local Newton iteration for xi, start in the element middle
    double eta[2] = {0.0, 0.0};
    double f = 0.0;
    double df = 0.0;
    int k = 0;

    for (k = 0; k < MORTARMAXITER; ++k)
    {
      f = evaluate_f_element_normal(node, ele, eta);
      if (abs(f) < MORTARCONVTOL) break;
      df = evaluate_grad_f_element_normal(node, ele, eta);
      if (abs(df) < 1.0e-12)
      {
        ok = false;
        xi[0] = 1.0e12;
        return ok;
        FOUR_C_THROW("Singular Jacobian for projection");
      }
      eta[0] += (-f) / df;
    }

    // get the result
    xi[0] = eta[0];

    // Newton iteration unconverged
    if (abs(f) > MORTARCONVTOL)
    {
      ok = false;
      xi[0] = 1.0e12;

      // Here (M->S projection) we only give a warning, no error!!!
      // This iteration sometimes diverges, when the projection is far off.
      // These cases are harmless, as these nodes then do not participate in
      // the overlap detection anyway!
      // std::cout << "***WARNING*** project_element_normal:" << " Newton unconverged for NodeID "
      //     << node.Id() << " and Mortar::ElementID " << ele.Id() << std::endl;
    }

    /*
     // Newton iteration converged
     else
     {
     std::cout << "Newton iteration converged in " << k << " step(s)!" << std::endl
     << "The result is: " << xi[0] << std::endl;
     }*/
  }

  else
    FOUR_C_THROW("project_element_normal: Called 2D version for 3D problem!");

  return ok;
}

/*----------------------------------------------------------------------*
 |  Project a Gauss point along its normal (public)           popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
bool Mortar::ProjectorCalcEleBased<distype_s, distype_m>::project_gauss_point_2d(
    const Mortar::Element& gpele, const double* gpeta, const Mortar::Element& ele, double* xi) const
{
  bool ok = true;
  if (ndim_ == 2)
  {
    Core::LinAlg::Matrix<ns_, 1> val;
    Core::LinAlg::Matrix<ndim_, ns_> coord;

    const Core::Nodes::Node* const* mynodes = gpele.nodes();
    if (!mynodes) FOUR_C_THROW("ProjectGaussPoint: Null pointer!");

    // get shape function values and derivatives at gpeta
    if (distype_s == Core::FE::CellType::nurbs2 || distype_s == Core::FE::CellType::nurbs3)
    {
      Core::LinAlg::SerialDenseVector auxval(ns_);
      Core::LinAlg::SerialDenseMatrix deriv(ns_, 1);
      gpele.evaluate_shape(gpeta, auxval, deriv, gpele.num_node());

      for (int i = 0; i < ns_; ++i) val(i) = auxval(i);
    }
    else
      Core::FE::shape_function_1d(val, gpeta[0], distype_s);

    // get interpolated GP normal and GP coordinates
    double gpn[ndim_];
    double gpx[ndim_];
    for (int i = 0; i < ndim_; ++i)
    {
      gpn[i] = 0.0;
      gpx[i] = 0.0;
    }

    for (int i = 0; i < ns_; ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);

      for (int j = 0; j < ndim_; ++j)
      {
        gpn[j] += val(i) * mymrtrnode->mo_data().n()[j];

        coord(j, i) = mymrtrnode->xspatial()[j];

        gpx[j] += val(i) * coord(j, i);
      }
    }

    // local Newton iteration for xi, start in the element middle
    double eta[2] = {0.0, 0.0};
    double f = 0.0;
    double df = 0.0;
    int k = 0;

    for (k = 0; k < MORTARMAXITER; ++k)
    {
      f = evaluate_f_gauss_point_2d(gpx, gpn, ele, eta);
      if (abs(f) < MORTARCONVTOL) break;
      df = evaluate_grad_f_gauss_point_2d(gpn, ele, eta);
      if (abs(df) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");
      eta[0] += (-f) / df;
    }

    // get the result
    xi[0] = eta[0];

    // Newton iteration unconverged
    if (abs(f) > MORTARCONVTOL)
    {
      ok = false;
      xi[0] = 1.0e12;

      return ok;

      //        FOUR_C_THROW("ProjectGaussPoint: Newton unconverged for GP at xi={}"
      //          " from Mortar::ElementID {}", gpeta[0], gpele.Id());
    }
  }

  else
    FOUR_C_THROW("ProjectGaussPoint: Called 2D version for 3D problem!");

  return ok;
}

/*----------------------------------------------------------------------*
 | Check projection for warped elements quad4 elements       farah 01/13|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
bool Mortar::ProjectorCalcEleBased<distype_s, distype_m>::check_projection4_auxplane(
    const Mortar::Element& ele, double* ngp, double* globgp) const
{
  if (ele.shape() == Core::FE::CellType::tri3) FOUR_C_THROW("ELEMENT SHAPE TRI3 -- NO WARPING");

  if (ele.shape() != Core::FE::CellType::quad4)
  {
    return true;
  }

  int nnode = ele.num_node();
  const Core::Nodes::Node* const* mynodes = ele.nodes();
  if (!mynodes) FOUR_C_THROW("Project: Null pointer!");

  // compute base-vectors
  std::vector<double> t0(3);
  std::vector<double> t1(3);
  std::vector<double> auxn(3);
  std::vector<double> auxc(3);
  std::vector<double> proj_gp(3);
  Core::LinAlg::Matrix<3, 3> P;
  Core::LinAlg::Matrix<3, 3> T;
  std::array<double, 3> n = {0.0, 0.0, 0.0};
  double length_t = 0.0;
  double length_n = 0.0;
  double a1 = 0.0;
  bool all_negative = true;
  const Node* mycnode_0 = nullptr;
  const Node* mycnode_1 = nullptr;
  const Node* mycnode_2 = nullptr;

  // project gp onto auxn
  for (int i = 0; i < nnode; ++i)  // loop over edges
  {
    if (i == 0)
    {
      mycnode_1 = dynamic_cast<const Node*>(mynodes[0]);
      if (!mycnode_1) FOUR_C_THROW("Project: Null pointer!");

      mycnode_0 = dynamic_cast<const Node*>(mynodes[3]);
      if (!mycnode_0) FOUR_C_THROW("Project: Null pointer!");

      mycnode_2 = dynamic_cast<const Node*>(mynodes[1]);
      if (!mycnode_2) FOUR_C_THROW("Project: Null pointer!");
    }
    if (i == 3)
    {
      mycnode_1 = dynamic_cast<const Node*>(mynodes[3]);
      if (!mycnode_1) FOUR_C_THROW("Project: Null pointer!");

      mycnode_0 = dynamic_cast<const Node*>(mynodes[2]);
      if (!mycnode_0) FOUR_C_THROW("Project: Null pointer!");

      mycnode_2 = dynamic_cast<const Node*>(mynodes[0]);
      if (!mycnode_2) FOUR_C_THROW("Project: Null pointer!");
    }
    if (i == 1 || i == 2)
    {
      mycnode_1 = dynamic_cast<const Node*>(mynodes[i]);
      if (!mycnode_1) FOUR_C_THROW("Project: Null pointer!");

      mycnode_0 = dynamic_cast<const Node*>(mynodes[i - 1]);
      if (!mycnode_0) FOUR_C_THROW("Project: Null pointer!");

      mycnode_2 = dynamic_cast<const Node*>(mynodes[i + 1]);
      if (!mycnode_2) FOUR_C_THROW("Project: Null pointer!");
    }

    // span vectors -- edges
    t0[0] = mycnode_0->xspatial()[0] - mycnode_1->xspatial()[0];
    t0[1] = mycnode_0->xspatial()[1] - mycnode_1->xspatial()[1];
    t0[2] = mycnode_0->xspatial()[2] - mycnode_1->xspatial()[2];

    t1[0] = mycnode_2->xspatial()[0] - mycnode_1->xspatial()[0];
    t1[1] = mycnode_2->xspatial()[1] - mycnode_1->xspatial()[1];
    t1[2] = mycnode_2->xspatial()[2] - mycnode_1->xspatial()[2];

    auxc[0] = mycnode_1->xspatial()[0];
    auxc[1] = mycnode_1->xspatial()[1];
    auxc[2] = mycnode_1->xspatial()[2];

    auxn[0] = t1[1] * t0[2] - t1[2] * t0[1];
    auxn[1] = t1[2] * t0[0] - t1[0] * t0[2];
    auxn[2] = t1[0] * t0[1] - t1[1] * t0[0];

    // fill Projection matrix P
    for (int j = 0; j < 3; ++j) P(j, 2) = -ngp[j];

    for (int j = 0; j < 3; ++j) P(j, 0) = t0[j];

    for (int j = 0; j < 3; ++j) P(j, 1) = t1[j];

    Core::LinAlg::invert(P);
    double lambda1 = P(0, 0) * (globgp[0] - auxc[0]) + P(0, 1) * (globgp[1] - auxc[1]) +
                     P(0, 2) * (globgp[2] - auxc[2]);
    double lambda2 = P(1, 0) * (globgp[0] - auxc[0]) + P(1, 1) * (globgp[1] - auxc[1]) +
                     P(1, 2) * (globgp[2] - auxc[2]);
    // double alpha= P(2,0)*(globgp[0]-auxc[0]) + P(2,1)*(globgp[1]-auxc[1]) +
    // P(2,2)*(globgp[2]-auxc[2]);

    proj_gp[0] = lambda1 * t0[0] + lambda2 * t1[0] + auxc[0];
    proj_gp[1] = lambda1 * t0[1] + lambda2 * t1[1] + auxc[1];
    proj_gp[2] = lambda1 * t0[2] + lambda2 * t1[2] + auxc[2];
    // std::cout << "proj_gp_AUX4PLANE= " << proj_gp[0] << " "  << proj_gp[1] << " " << proj_gp[2]
    // << std::endl;

    // check
    // edge 1
    n[0] = -(t0[1] * auxn[2] - t0[2] * auxn[1]);
    n[1] = -(t0[2] * auxn[0] - t0[0] * auxn[2]);
    n[2] = -(t0[0] * auxn[1] - t0[1] * auxn[0]);

    length_t = sqrt(t0[0] * t0[0] + t0[1] * t0[1] + t0[2] * t0[2]);
    length_n = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    // fill Projection matrix T
    for (int j = 0; j < 3; ++j) T(j, 0) = n[j] / length_n;

    for (int j = 0; j < 3; ++j) T(j, 1) = t0[j] / length_t;

    for (int j = 0; j < 3; ++j) T(j, 2) = auxc[j];

    Core::LinAlg::invert(T);
    a1 = T(0, 0) * proj_gp[0] + T(0, 1) * proj_gp[1] + T(0, 2) * proj_gp[2];

    if (a1 > 0.0) all_negative = false;

    // edge 2
    n[0] = (t1[1] * auxn[2] - t1[2] * auxn[1]);
    n[1] = (t1[2] * auxn[0] - t1[0] * auxn[2]);
    n[2] = (t1[0] * auxn[1] - t1[1] * auxn[0]);

    length_t = sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
    length_n = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    // fill Projection matrix T
    for (int j = 0; j < 3; ++j) T(j, 0) = n[j] / length_n;

    for (int j = 0; j < 3; ++j) T(j, 1) = t1[j] / length_t;

    for (int j = 0; j < 3; ++j) T(j, 2) = auxc[j];

    Core::LinAlg::invert(T);

    a1 = T(0, 0) * proj_gp[0] + T(0, 1) * proj_gp[1] + T(0, 2) * proj_gp[2];
    // a2=T(1,0)*proj_gp[0] + T(1,1)*proj_gp[1] + T(1,2)*proj_gp[2];

    if (a1 > 0.0) all_negative = false;
  }

  if (all_negative == false) return true;

  return false;  // creates error in ProjectGaussPoint3D()
}

/*----------------------------------------------------------------------*
 |  Project a Gauss point along its normal (3D)               popp 11/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
bool Mortar::ProjectorCalcEleBased<distype_s, distype_m>::project_gauss_point_3d(
    const Mortar::Element& gpele, const double* gpeta, const Mortar::Element& ele, double* xi,
    double& par) const
{
  if (ndim_ == 3)
  {
    Core::LinAlg::Matrix<ns_, 1> val;
    Core::LinAlg::Matrix<ndim_, ns_> coord;
    coord.clear();

    const Core::Nodes::Node* const* mypoints = gpele.nodes();
    const Core::Nodes::Node* const* mynodes = gpele.nodes();
    if (!mypoints) FOUR_C_THROW("ProjectGaussPoint: Null pointer!");

    // get shape function values and derivatives at gpeta
    if (distype_s == Core::FE::CellType::nurbs4 || distype_s == Core::FE::CellType::nurbs8 ||
        distype_s == Core::FE::CellType::nurbs9)
    {
      Core::LinAlg::SerialDenseVector auxval(ns_);
      Core::LinAlg::SerialDenseMatrix deriv(ns_, 2);
      gpele.evaluate_shape(gpeta, auxval, deriv, gpele.num_node());

      for (int i = 0; i < ns_; ++i) val(i) = auxval(i);
    }
    else
    {
      Core::FE::shape_function_2d(val, gpeta[0], gpeta[1], distype_s);
    }

    // get interpolated GP normal and GP coordinates
    double gpn[3];
    double gpx[3];
    for (int i = 0; i < ndim_; ++i)
    {
      gpn[i] = 0.0;
      gpx[i] = 0.0;
    }

    for (int i = 0; i < ns_; ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mypoints[i]);

      for (int j = 0; j < ndim_; ++j)
      {
        coord(j, i) = mymrtrnode->xspatial()[j];

        gpx[j] += val(i) * coord(j, i);
      }
    }

    for (int i = 0; i < gpele.num_node(); ++i)
    {
      const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);
      for (int j = 0; j < ndim_; ++j)
      {
        gpn[j] += val(i) * mymrtrnode->mo_data().n()[j];
      }
    }

    // start in the element center
    Core::FE::CellType dt = ele.shape();
    double eta[2] = {0.0, 0.0};

    if (dt == Core::FE::CellType::tri3 || dt == Core::FE::CellType::tri6)
    {
      eta[0] = 1.0 / 3.0;
      eta[1] = 1.0 / 3.0;
    }

    // auxiliary variable
    double alpha = 0.0;

    // function f (vector-valued)
    double f[3] = {0.0, 0.0, 0.0};

    // gradient of f (df/deta[0], df/deta[1], df/dalpha)
    Core::LinAlg::Matrix<3, 3> df;
    // start iteration
    int k = 0;
    double conv = 0.0;

    for (k = 0; k < MORTARMAXITER; ++k)
    {
      evaluate_f_gauss_point_3d(f, gpx, gpn, ele, eta, alpha);
      conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
      if (conv <= MORTARCONVTOL) break;
      evaluate_grad_f_gauss_point_3d(df, gpx, gpn, ele, eta, alpha);

      // safety check: if projection normal is parallel to the master element --> det can be zero
      double det = df.determinant();
      if (det > -1e-12 and det < 1e-12)
      {
        std::cout << "WARNING: GPProjection parallel to master element --> GP skipped for this "
                     "master element!"
                  << std::endl;
        // leave here
        return false;
      }

      // solve deta = - inv(df) * f
      double jacdet = Core::LinAlg::invert(df);
      if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

      // update eta and alpha
      eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
      eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
      alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];

      // Projection Check
      //      if (k==MORTARMAXITER-1)
      //      {
      //        bool check = check_projection4_auxplane(ele, gpn,gpx);
      //        if (check==false)
      //          FOUR_C_THROW("!!! STOP !!!   -->   Projection Error: Newton unconverged but GP on
      //          mele
      //          !!!");
      //      }
    }

    // Newton iteration unconverged
    if (conv > MORTARCONVTOL)
    {
      xi[0] = 1e12;
      xi[1] = 1e12;
      par = alpha;

      return false;
    }
    else
    {
      xi[0] = eta[0];
      xi[1] = eta[1];
      par = alpha;
    }
  }

  else
    FOUR_C_THROW("ProjectGaussPoint: Called 3D version for 2D problem!");

  return true;
}
/*----------------------------------------------------------------------*
 |  Project a Gauss point along AuxPlane normal (3D)          popp 11/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_gauss_point_auxn_3d(const double* globgp,
    const double* auxn, const Mortar::Element& ele, double* xi, double& par) const
{
  if (ndim_ == 3)
  {
    // start in the element center
    Core::FE::CellType dt = ele.shape();
    double eta[2] = {0.0, 0.0};

    switch (dt)
    {
      case Core::FE::CellType::tri3:
      case Core::FE::CellType::tri6:
      {
        eta[0] = 1.0 / 3.0;
        eta[1] = 1.0 / 3.0;

        break;
      }
      default:
        break;
    }

    // auxiliary variable
    double& alpha = par;
    alpha = 0.0;

    // function f (vector-valued)
    double f[3] = {0.0, 0.0, 0.0};

    // gradient of f (df/deta[0], df/deta[1], df/dalpha)
    Core::LinAlg::Matrix<3, 3> df;

    // start iteration
    int k = 0;
    double conv = 0.0;

    for (k = 0; k < MORTARMAXITER; ++k)
    {
      evaluate_f_gauss_point_auxn_3d(f, globgp, auxn, ele, eta, alpha);
      conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
      // std::cout << "Iteration " << k << ": -> |f|=" << conv << std::endl;
      if (conv <= MORTARCONVTOL) break;
      evaluate_grad_f_gauss_point_auxn_3d(df, globgp, auxn, ele, eta, alpha);

      // solve deta = - inv(df) * f
      double jacdet = Core::LinAlg::invert(df);
      if (abs(jacdet) < 1.0e-12)
      {
        FOUR_C_THROW("Singular Jacobian for projection");
      }

      // update eta and alpha
      eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
      eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
      alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];
    }

    // Newton iteration unconverged
    if (conv > MORTARCONVTOL)
    {
      //      std::cout << "res= " << conv << std::endl;
      //          FOUR_C_THROW("project_gauss_point_auxn_3d: Newton unconverged for GP"
      //          "at xi = ({},{},{}) onto Mortar::ElementID {}", globgp[0], globgp[1],
      //          globgp[2], ele.Id());
      xi[0] = 1e12;
      xi[1] = 1e12;
      return false;
    }

    // Newton iteration converged
    else
    {
      xi[0] = eta[0];
      xi[1] = eta[1];
    }

    // std::cout << "Newton iteration converged in " << k << " steps!" << std::endl;
  }

  else
    FOUR_C_THROW("ProjectGaussPoint: Called 3D version for 2D problem!");

  return true;
}


/*----------------------------------------------------------------------*
 |  Project snode onto melement with master normal           farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_normal_3d(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist) const
{
  if (ndim_ != 3) FOUR_C_THROW("project_s_node_by_m_normal_3d is only for 3D problems!");

  // start in the element center
  double eta[2] = {0.0, 0.0};
  switch (distype)
  {
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
    {
      eta[0] = 1.0 / 3.0;
      eta[1] = 1.0 / 3.0;
      break;
    }
    default:
      break;
  }

  // auxiliary variable for distance
  double alpha = 0.0;

  // function f (vector-valued)
  std::array<double, 3> f = {0.0, 0.0, 0.0};

  // gradient of f (df/deta[0], df/deta[1], df/dalpha)
  Core::LinAlg::Matrix<3, 3> df;

  // start iteration
  int k = 0;
  double conv = 0.0;

  for (k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************

    // build interpolation of master node coordinates for current eta
    double xm[3] = {0.0, 0.0, 0.0};
    double xs[3] = {0.0, 0.0, 0.0};
    double unormal[3] = {0.0, 0.0, 0.0};
    double normal[3] = {0.0, 0.0, 0.0};
    double normalpart[3] = {0.0, 0.0, 0.0};

    // calc xmaster
    Mortar::Utils::local_to_global<distype>(mele, eta, xm, 0);

    // calc normal part
    double length = mele.compute_unit_normal_at_xi(eta, unormal);
    for (int i = 0; i < 3; ++i) normalpart[i] = unormal[i] * alpha;

    for (int i = 0; i < 3; ++i) normal[i] = unormal[i] * length;

    // calc xslave
    for (int i = 0; i < 3; ++i) xs[i] = snode.xspatial()[i];

    // calculate F
    for (int i = 0; i < 3; ++i) f[i] = xm[i] + normalpart[i] - xs[i];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (conv <= MORTARCONVTOL) break;

    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************
    // master coordinate grad
    double meta0[3] = {0.0, 0.0, 0.0};  // x,xi_0
    double meta1[3] = {0.0, 0.0, 0.0};  // x,xi_1
    Mortar::Utils::local_to_global<distype>(mele, eta, meta0, 1);
    Mortar::Utils::local_to_global<distype>(mele, eta, meta1, 2);

    // normal grad
    Core::LinAlg::Matrix<3, n_> secderiv;
    std::array<double, 3> meta00 = {0.0, 0.0, 0.0};  // x , xi_0 xi_0
    std::array<double, 3> meta11 = {0.0, 0.0, 0.0};  // x , xi_1 xi_1
    std::array<double, 3> meta01 = {0.0, 0.0, 0.0};  // x , xi_0 xi_1

    Core::FE::shape_function_2d_deriv2(secderiv, eta[0], eta[1], distype);

    for (int i = 0; i < n_; ++i)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[i]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int d = 0; d < 3; ++d)
      {
        meta00[d] += secderiv(0, i) * mymnode->xspatial()[d];
        meta11[d] += secderiv(1, i) * mymnode->xspatial()[d];
        meta01[d] += secderiv(2, i) * mymnode->xspatial()[d];
      }
    }

    std::array<double, 3> naux_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> naux_1 = {0.0, 0.0, 0.0};

    // normal grad xi_0
    naux_0[0] = (meta00[1] * meta1[2] - meta00[2] * meta1[1]);
    naux_0[1] = (meta00[2] * meta1[0] - meta00[0] * meta1[2]);
    naux_0[2] = (meta00[0] * meta1[1] - meta00[1] * meta1[0]);

    naux_0[0] += (meta0[1] * meta01[2] - meta0[2] * meta01[1]);
    naux_0[1] += (meta0[2] * meta01[0] - meta0[0] * meta01[2]);
    naux_0[2] += (meta0[0] * meta01[1] - meta0[1] * meta01[0]);

    // normal grad xi_1
    naux_1[0] = (meta01[1] * meta1[2] - meta01[2] * meta1[1]);
    naux_1[1] = (meta01[2] * meta1[0] - meta01[0] * meta1[2]);
    naux_1[2] = (meta01[0] * meta1[1] - meta01[1] * meta1[0]);

    naux_1[0] += (meta0[1] * meta11[2] - meta0[2] * meta11[1]);
    naux_1[1] += (meta0[2] * meta11[0] - meta0[0] * meta11[2]);
    naux_1[2] += (meta0[0] * meta11[1] - meta0[1] * meta11[0]);

    std::array<double, 3> n_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> n_1 = {0.0, 0.0, 0.0};

    double fac0 = 0.0;
    double fac1 = 0.0;

    for (int i = 0; i < 3; ++i) fac0 += naux_0[i] * normal[i];

    for (int i = 0; i < 3; ++i)
      n_0[i] = naux_0[i] / length - fac0 * normal[i] / (length * length * length);

    for (int i = 0; i < 3; ++i) fac1 += naux_1[i] * normal[i];

    for (int i = 0; i < 3; ++i)
      n_1[i] = naux_1[i] / length - fac1 * normal[i] / (length * length * length);

    // evaluate function f gradient
    for (int i = 0; i < 3; ++i)
    {
      df(i, 0) = meta0[i] + n_0[i];
      df(i, 1) = meta1[i] + n_1[i];
      df(i, 2) = unormal[i];
    }

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = Core::LinAlg::invert(df);
    if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

    // update eta and alpha
    eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
    eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
    alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];
  }  // end newton loop

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL) FOUR_C_THROW("Projector not converged!");

  // Newton iteration converged
  xi[0] = eta[0];
  xi[1] = eta[1];
  mele.compute_unit_normal_at_xi(eta, normal);
  dist = alpha;

  return true;
}


/*----------------------------------------------------------------------*
 |  Project snode onto melement with master normal           farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_normal_3d_lin(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  if (ndim_ != 3) FOUR_C_THROW("project_s_node_by_m_normal_3d is only for 3D problems!");

  // start in the element center
  double eta[2] = {0.0, 0.0};
  if (distype == Core::FE::CellType::tri3 || distype == Core::FE::CellType::tri6)
  {
    eta[0] = 1.0 / 3.0;
    eta[1] = 1.0 / 3.0;
  }

  // auxiliary variable for distance
  double alpha = 0.0;

  // function f (vector-valued)
  std::array<double, 3> f = {0.0, 0.0, 0.0};

  // gradient of f (df/deta[0], df/deta[1], df/dalpha)
  Core::LinAlg::Matrix<3, 3> df;

  // start iteration
  int k = 0;
  double conv = 0.0;

  for (k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************

    // build interpolation of master node coordinates for current eta
    double xm[3] = {0.0, 0.0, 0.0};
    double xs[3] = {0.0, 0.0, 0.0};
    double unormal[3] = {0.0, 0.0, 0.0};
    double normal[3] = {0.0, 0.0, 0.0};
    double normalpart[3] = {0.0, 0.0, 0.0};

    // calc xmaster
    Mortar::Utils::local_to_global<distype>(mele, eta, xm, 0);

    // calc normal part
    double length = mele.compute_unit_normal_at_xi(eta, unormal);
    for (int i = 0; i < 3; ++i) normalpart[i] = unormal[i] * alpha;

    for (int i = 0; i < 3; ++i) normal[i] = unormal[i] * length;

    // calc xslave
    for (int i = 0; i < 3; ++i) xs[i] = snode.xspatial()[i];

    // calculate F
    for (int i = 0; i < 3; ++i) f[i] = xm[i] + normalpart[i] - xs[i];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (conv <= MORTARCONVTOL) break;

    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************
    // master coordinate grad
    std::array<double, 3> meta0 = {0.0, 0.0, 0.0};  // x , xi_0
    std::array<double, 3> meta1 = {0.0, 0.0, 0.0};  // x , xi_1
    Mortar::Utils::local_to_global<distype>(mele, eta, meta0, 1);
    Mortar::Utils::local_to_global<distype>(mele, eta, meta1, 2);

    // normal grad
    Core::LinAlg::Matrix<3, n_> secderiv;
    std::array<double, 3> meta00 = {0.0, 0.0, 0.0};  // x , xi_0 xi_0
    std::array<double, 3> meta11 = {0.0, 0.0, 0.0};  // x , xi_1 xi_1
    std::array<double, 3> meta01 = {0.0, 0.0, 0.0};  // x , xi_0 xi_1

    Core::FE::shape_function_2d_deriv2(secderiv, eta[0], eta[1], distype);

    for (int i = 0; i < n_; ++i)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[i]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int d = 0; d < 3; ++d)
      {
        meta00[d] += secderiv(0, i) * mymnode->xspatial()[d];
        meta11[d] += secderiv(1, i) * mymnode->xspatial()[d];
        meta01[d] += secderiv(2, i) * mymnode->xspatial()[d];
      }
    }

    std::array<double, 3> naux_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> naux_1 = {0.0, 0.0, 0.0};

    // normal grad xi_0
    naux_0[0] = (meta00[1] * meta1[2] - meta00[2] * meta1[1]);
    naux_0[1] = (meta00[2] * meta1[0] - meta00[0] * meta1[2]);
    naux_0[2] = (meta00[0] * meta1[1] - meta00[1] * meta1[0]);

    naux_0[0] += (meta0[1] * meta01[2] - meta0[2] * meta01[1]);
    naux_0[1] += (meta0[2] * meta01[0] - meta0[0] * meta01[2]);
    naux_0[2] += (meta0[0] * meta01[1] - meta0[1] * meta01[0]);

    // normal grad xi_1
    naux_1[0] = (meta01[1] * meta1[2] - meta01[2] * meta1[1]);
    naux_1[1] = (meta01[2] * meta1[0] - meta01[0] * meta1[2]);
    naux_1[2] = (meta01[0] * meta1[1] - meta01[1] * meta1[0]);

    naux_1[0] += (meta0[1] * meta11[2] - meta0[2] * meta11[1]);
    naux_1[1] += (meta0[2] * meta11[0] - meta0[0] * meta11[2]);
    naux_1[2] += (meta0[0] * meta11[1] - meta0[1] * meta11[0]);

    std::array<double, 3> n_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> n_1 = {0.0, 0.0, 0.0};

    double fac0 = 0.0;
    double fac1 = 0.0;

    for (int i = 0; i < 3; ++i) fac0 += naux_0[i] * normal[i];

    for (int i = 0; i < 3; ++i)
      n_0[i] = naux_0[i] / length - fac0 * normal[i] / (length * length * length);

    for (int i = 0; i < 3; ++i) fac1 += naux_1[i] * normal[i];

    for (int i = 0; i < 3; ++i)
      n_1[i] = naux_1[i] / length - fac1 * normal[i] / (length * length * length);

    // evaluate function f gradient
    for (int i = 0; i < 3; ++i)
    {
      df(i, 0) = meta0[i] + n_0[i];
      df(i, 1) = meta1[i] + n_1[i];
      df(i, 2) = unormal[i];
    }

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = Core::LinAlg::invert(df);
    if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

    // update eta and alpha
    eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
    eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
    alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];
  }  // end newton loop

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL) FOUR_C_THROW("Projector not converged!");

  // Newton iteration converged
  xi[0] = eta[0];
  xi[1] = eta[1];
  mele.compute_unit_normal_at_xi(eta, normal);
  dist = alpha;

  return true;
}


/*----------------------------------------------------------------------*
 |  Project snode onto melement with master normal           farah 08/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_nodal_normal_3d_lin(
    const Mortar::Node& snode, const Mortar::Element& mele, double* xi, double* normal,
    double& dist, std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  if (ndim_ != 3) FOUR_C_THROW("project_s_node_by_m_normal_3d_lin is only for 3D problems!");

  // start in the element center
  std::array<double, 2> eta = {0.0, 0.0};

  // auxiliary variable for distance
  double alpha = 0.0;

  // function f (vector-valued)
  std::array<double, 3> f = {0.0, 0.0, 0.0};

  // gradient of f (df/deta[0], df/deta[1], df/dalpha)
  Core::LinAlg::Matrix<3, 3> df;

  // start iteration
  int k = 0;
  double conv = 0.0;

  for (k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************
    // get shape function value
    Core::LinAlg::Matrix<n_, 1> mval;
    Core::LinAlg::Matrix<2, n_> mderiv;

    Core::FE::shape_function_2d(mval, eta[0], eta[1], distype);
    Core::FE::shape_function_2d_deriv1(mderiv, eta[0], eta[1], distype);

    // build interpolation of master node coordinates for current eta
    std::array<double, 3> xm = {0.0, 0.0, 0.0};
    std::array<double, 3> xs = {0.0, 0.0, 0.0};
    std::array<double, 3> normalnewton = {0.0, 0.0, 0.0};
    std::array<double, 3> normalpart = {0.0, 0.0, 0.0};

    // calc xmaster
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int i = 0; i < 3; ++i)
      {
        xm[i] += mval(j) * mymnode->xspatial()[i];
      }
    }

    // calc xslave
    for (int i = 0; i < 3; ++i) xs[i] = snode.xspatial()[i];

    // calc normal part
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int i = 0; i < 3; ++i)
      {
        normalnewton[i] += mval(j) * mymnode->mo_data().n()[i];
      }
    }

    for (int i = 0; i < 3; ++i) normalpart[i] = normalnewton[i] * alpha;

    // calculate F
    for (int i = 0; i < 3; ++i) f[i] = xm[i] + normalpart[i] - xs[i];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    //    std::cout << "k = " << k << "  tol= " << conv << std::endl;
    if (conv <= MORTARCONVTOL) break;
    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************

    // master coordinate grad
    std::array<double, 3> meta0 = {0.0, 0.0, 0.0};  // x , xi_0
    std::array<double, 3> meta1 = {0.0, 0.0, 0.0};  // x , xi_1

    // calc xmaster
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int i = 0; i < 3; ++i)
      {
        meta0[i] += mderiv(0, j) * mymnode->xspatial()[i];
        meta1[i] += mderiv(1, j) * mymnode->xspatial()[i];
      }
    }

    std::array<double, 3> n_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> n_1 = {0.0, 0.0, 0.0};

    // calc normal part
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");

      for (int i = 0; i < 3; ++i)
      {
        n_0[i] += mderiv(0, j) * mymnode->mo_data().n()[i];
        n_1[i] += mderiv(1, j) * mymnode->mo_data().n()[i];
      }
    }

    // evaluate function f gradient
    for (int i = 0; i < 3; ++i)
    {
      df(i, 0) = meta0[i] + alpha * n_0[i];
      df(i, 1) = meta1[i] + alpha * n_1[i];
      df(i, 2) = normalnewton[i];
    }

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = Core::LinAlg::invert(df);
    if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

    // update eta and alpha
    eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
    eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
    alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];
  }  // end newton loop

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL)
  {
    return false;
    FOUR_C_THROW("Projector not converged!");
  }

  //**********************************************
  //   Get stuff                                //
  //**********************************************
  // Newton iteration converged
  xi[0] = eta[0];
  xi[1] = eta[1];
  dist = alpha;

  // get shape function value
  Core::LinAlg::Matrix<n_, 1> mval;
  Core::LinAlg::Matrix<2, n_> mderiv;

  Core::FE::shape_function_2d(mval, eta[0], eta[1], distype);
  Core::FE::shape_function_2d_deriv1(mderiv, eta[0], eta[1], distype);

  // calc normal part
  normal[0] = 0.0;
  normal[1] = 0.0;
  normal[2] = 0.0;
  for (int j = 0; j < n_; ++j)
  {
    const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
    if (!mymnode) FOUR_C_THROW("Null pointer!");

    for (int i = 0; i < 3; ++i)
    {
      normal[i] += mval(j) * mymnode->mo_data().n()[i];
    }
  }

  //**********************************************
  //   Lin deta = - inv(dF) * Lin F             //
  //**********************************************
  // prepare linearizations
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  // get linsize
  int linsize = 0;
  for (int i = 0; i < n_; ++i)
  {
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const CONTACT::Node* mnode = dynamic_cast<const CONTACT::Node*>(node);
    linsize += mnode->get_linsize();
  }

  std::vector<Core::Gen::Pairedvector<int, double>> xmLin(3, n_);  // nnode entry per dimension
  std::vector<Core::Gen::Pairedvector<int, double>> xsLin(3, 1);   // one entry per dimension

  std::vector<Core::Gen::Pairedvector<int, double>> normalpartLin(
      3, linsize);  // linsize of all mnodes
  std::vector<Core::Gen::Pairedvector<int, double>> auxnormalLin(
      3, linsize);  // linsize of all mnodes

  std::vector<Core::Gen::Pairedvector<int, double>> etaLin(3, linsize + n_ + 1);  // added all sizes
  std::vector<Core::Gen::Pairedvector<int, double>> fLin(3, linsize + n_ + 1);    // added all sizes


  //--------------------------
  // master part:
  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const Node* mnode = dynamic_cast<const Node*>(node);

    for (int k = 0; k < 3; ++k) (xmLin[k])[mnode->dofs()[k]] += mval(i);
  }

  //--------------------------
  // normal part:
  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const CONTACT::Node* mnode = dynamic_cast<const CONTACT::Node*>(node);

    for (int k = 0; k < 3; ++k)
    {
      for (CI p = mnode->data().get_deriv_n()[k].begin(); p != mnode->data().get_deriv_n()[k].end();
          ++p)
      {
        (auxnormalLin[k])[p->first] += mval(i) * (p->second);
      }
    }
  }

  // *alpha
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = auxnormalLin[j].begin(); p != auxnormalLin[j].end(); ++p)
      (normalpartLin[j])[p->first] += (p->second) * alpha;
  }

  //--------------------------
  // slave part:
  for (int k = 0; k < 3; ++k) (xsLin[k])[snode.dofs()[k]] += 1.0;

  // All terms:
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = xmLin[j].begin(); p != xmLin[j].end(); ++p) (fLin[j])[p->first] += (p->second);

    for (CI p = normalpartLin[j].begin(); p != normalpartLin[j].end(); ++p)
      (fLin[j])[p->first] += (p->second);

    for (CI p = xsLin[j].begin(); p != xsLin[j].end(); ++p) (fLin[j])[p->first] -= (p->second);
  }

  for (int i = 0; i < 3; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      for (CI p = fLin[i].begin(); p != fLin[i].end(); ++p)
        (etaLin[k])[p->first] -= (p->second) * df(k, i);
    }
  }

  //**********************************************
  //   Lin N                                    //
  //**********************************************
  std::vector<Core::Gen::Pairedvector<int, double>> n_eta0_deriv(
      3, linsize + n_ + 1);  // added all sizes
  std::vector<Core::Gen::Pairedvector<int, double>> n_eta1_deriv(
      3, linsize + n_ + 1);                                                 // added all sizes
  std::vector<Core::Gen::Pairedvector<int, double>> n_n_deriv(3, linsize);  // linsize

  for (int k = 0; k < n_; ++k)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[k];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const CONTACT::Node* mnode = dynamic_cast<const CONTACT::Node*>(node);

    for (CI p = etaLin[0].begin(); p != etaLin[0].end(); ++p)
    {
      (n_eta0_deriv[0])[p->first] += mderiv(0, k) * (p->second) * mnode->mo_data().n()[0];
      (n_eta0_deriv[1])[p->first] += mderiv(0, k) * (p->second) * mnode->mo_data().n()[1];
      (n_eta0_deriv[2])[p->first] += mderiv(0, k) * (p->second) * mnode->mo_data().n()[2];
    }

    for (CI p = etaLin[1].begin(); p != etaLin[1].end(); ++p)
    {
      (n_eta1_deriv[0])[p->first] += mderiv(1, k) * (p->second) * mnode->mo_data().n()[0];
      (n_eta1_deriv[1])[p->first] += mderiv(1, k) * (p->second) * mnode->mo_data().n()[1];
      (n_eta1_deriv[2])[p->first] += mderiv(1, k) * (p->second) * mnode->mo_data().n()[2];
    }

    for (CI p = mnode->data().get_deriv_n()[0].begin(); p != mnode->data().get_deriv_n()[0].end();
        ++p)
      (n_n_deriv[0])[p->first] += mval(k) * (p->second);
    for (CI p = mnode->data().get_deriv_n()[1].begin(); p != mnode->data().get_deriv_n()[1].end();
        ++p)
      (n_n_deriv[1])[p->first] += mval(k) * (p->second);
    for (CI p = mnode->data().get_deriv_n()[2].begin(); p != mnode->data().get_deriv_n()[2].end();
        ++p)
      (n_n_deriv[2])[p->first] += mval(k) * (p->second);
  }


  for (int j = 0; j < 3; ++j)
  {
    for (CI p = n_eta1_deriv[j].begin(); p != n_eta1_deriv[j].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second);
    for (CI p = n_eta0_deriv[j].begin(); p != n_eta0_deriv[j].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second);
    for (CI p = n_n_deriv[j].begin(); p != n_n_deriv[j].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second);
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  Project snode onto melement with master normal           farah 05/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_nodal_normal_2d_lin(
    const Mortar::Node& snode, const Mortar::Element& mele, double* xi, double* normal,
    double& dist, std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  if (ndim_ != 2) FOUR_C_THROW("project_s_node_by_m_normal_2d_lin is only for 2D problems!");

  // start in the element center
  double eta[2] = {0.0, 0.0};

  // auxiliary variable for distance
  double alpha = 0.0;

  // function f (vector-valued)
  std::array<double, 2> f = {0.0, 0.0};

  // gradient of f (df/deta[0], df/dalpha)
  Core::LinAlg::Matrix<2, 2> df;

  // start iteration
  int k = 0;
  double conv = 0.0;

  for (k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************
    // get shape function value
    Core::LinAlg::Matrix<n_, 1> mval;
    Core::LinAlg::Matrix<1, n_> mderiv;

    Core::FE::shape_function_1d(mval, eta[0], distype);
    Core::FE::shape_function_1d_deriv1(mderiv, eta[0], distype);

    // build interpolation of master node coordinates for current eta
    std::array<double, 3> xm = {0.0, 0.0, 0.0};
    std::array<double, 3> xs = {0.0, 0.0, 0.0};
    std::array<double, 3> normalnewton = {0.0, 0.0, 0.0};
    std::array<double, 3> normalpart = {0.0, 0.0, 0.0};

    // calc xmaster
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int i = 0; i < 2; ++i)
      {
        xm[i] += mval(j) * mymnode->xspatial()[i];
      }
    }

    // calc xslave
    for (int i = 0; i < 2; ++i) xs[i] = snode.xspatial()[i];

    // calc normal part
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int i = 0; i < 2; ++i)
      {
        normalnewton[i] += mval(j) * mymnode->mo_data().n()[i];
      }
    }

    for (int i = 0; i < 3; ++i) normalpart[i] = normalnewton[i] * alpha;

    // calculate F
    for (int i = 0; i < 2; ++i) f[i] = xm[i] + normalpart[i] - xs[i];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1]);
    //    std::cout << "k = " << k << "  tol= " << conv << std::endl;
    if (conv <= MORTARCONVTOL) break;
    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************

    // master coordinate grad
    std::array<double, 3> meta0 = {0.0, 0.0, 0.0};  // x , xi_0

    // calc xmaster
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int i = 0; i < 2; ++i)
      {
        meta0[i] += mderiv(0, j) * mymnode->xspatial()[i];
      }
    }

    std::array<double, 3> n_0 = {0.0, 0.0, 0.0};

    // calc normal part
    for (int j = 0; j < n_; ++j)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");

      for (int i = 0; i < 2; ++i)
      {
        n_0[i] += mderiv(0, j) * mymnode->mo_data().n()[i];
      }
    }

    // evaluate function f gradient
    for (int i = 0; i < 2; ++i)
    {
      df(i, 0) = meta0[i] + alpha * n_0[i];
      df(i, 1) = normalnewton[i];
    }

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = Core::LinAlg::invert(df);
    if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

    // update eta and alpha
    eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1];
    alpha += -df(1, 0) * f[0] - df(1, 1) * f[1];
  }  // end newton loop

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL)
  {
    return false;
    FOUR_C_THROW("Projector not converged!");
  }

  //**********************************************
  //   Get stuff                                //
  //**********************************************
  // Newton iteration converged
  xi[0] = eta[0];
  xi[1] = eta[1];
  dist = alpha;

  // get shape function value
  Core::LinAlg::Matrix<n_, 1> mval;
  Core::LinAlg::Matrix<1, n_> mderiv;

  Core::FE::shape_function_1d(mval, eta[0], distype);
  Core::FE::shape_function_1d_deriv1(mderiv, eta[0], distype);

  // calc normal part
  normal[0] = 0.0;
  normal[1] = 0.0;
  normal[2] = 0.0;
  for (int j = 0; j < n_; ++j)
  {
    const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[j]);
    if (!mymnode) FOUR_C_THROW("Null pointer!");

    for (int i = 0; i < 2; ++i)
    {
      normal[i] += mval(j) * mymnode->mo_data().n()[i];
    }
  }

  //**********************************************
  //   Lin deta = - inv(dF) * Lin F             //
  //**********************************************
  // prepare linearizations
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  std::vector<Core::Gen::Pairedvector<int, double>> etaLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> fLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> xmLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> normalpartLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> xsLin(3, 1000);

  //--------------------------
  // master part:
  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const Node* mnode = dynamic_cast<const Node*>(node);

    for (int k = 0; k < 2; ++k) (xmLin[k])[mnode->dofs()[k]] += mval(i);
  }

  //--------------------------
  // normal part:
  std::vector<Core::Gen::Pairedvector<int, double>> x_0Lin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> auxnormalLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> auxnormalunitLin(3, 1000);

  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const CONTACT::Node* mnode = dynamic_cast<const CONTACT::Node*>(node);

    for (int k = 0; k < 3; ++k)
    {
      for (CI p = mnode->data().get_deriv_n()[k].begin(); p != mnode->data().get_deriv_n()[k].end();
          ++p)
      {
        (auxnormalLin[k])[p->first] += mval(i) * (p->second);
      }
    }
  }

  // *alpha
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = auxnormalLin[j].begin(); p != auxnormalLin[j].end(); ++p)
      (normalpartLin[j])[p->first] += (p->second) * alpha;
  }

  //--------------------------
  // slave part:
  for (int k = 0; k < 2; ++k) (xsLin[k])[snode.dofs()[k]] += 1.0;

  // All terms:
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = xmLin[j].begin(); p != xmLin[j].end(); ++p) (fLin[j])[p->first] += (p->second);

    for (CI p = normalpartLin[j].begin(); p != normalpartLin[j].end(); ++p)
      (fLin[j])[p->first] += (p->second);

    for (CI p = xsLin[j].begin(); p != xsLin[j].end(); ++p) (fLin[j])[p->first] -= (p->second);
  }

  for (int i = 0; i < 2; ++i)
  {
    for (int k = 0; k < 2; ++k)
    {
      for (CI p = fLin[i].begin(); p != fLin[i].end(); ++p)
        (etaLin[k])[p->first] -= (p->second) * df(k, i);
    }
  }

  //**********************************************
  //   Lin N                                    //
  //**********************************************
  std::vector<Core::Gen::Pairedvector<int, double>> n_eta0_deriv(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> n_eta1_deriv(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> n_n_deriv(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> normaltolineLinaux(3, 1000);

  for (int k = 0; k < n_; ++k)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[k];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const CONTACT::Node* mnode = dynamic_cast<const CONTACT::Node*>(node);

    for (CI p = etaLin[0].begin(); p != etaLin[0].end(); ++p)
    {
      (n_eta0_deriv[0])[p->first] += mderiv(0, k) * (p->second) * mnode->mo_data().n()[0];
      (n_eta0_deriv[1])[p->first] += mderiv(0, k) * (p->second) * mnode->mo_data().n()[1];
      (n_eta0_deriv[2])[p->first] += mderiv(0, k) * (p->second) * mnode->mo_data().n()[2];
    }

    for (CI p = mnode->data().get_deriv_n()[0].begin(); p != mnode->data().get_deriv_n()[0].end();
        ++p)
      (n_n_deriv[0])[p->first] += mval(k) * (p->second);
    for (CI p = mnode->data().get_deriv_n()[1].begin(); p != mnode->data().get_deriv_n()[1].end();
        ++p)
      (n_n_deriv[1])[p->first] += mval(k) * (p->second);
    for (CI p = mnode->data().get_deriv_n()[2].begin(); p != mnode->data().get_deriv_n()[2].end();
        ++p)
      (n_n_deriv[2])[p->first] += mval(k) * (p->second);
  }


  for (int j = 0; j < 3; ++j)
  {
    for (CI p = n_eta0_deriv[j].begin(); p != n_eta0_deriv[j].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second);
    for (CI p = n_n_deriv[j].begin(); p != n_n_deriv[j].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second);
  }

  return true;
}



/*----------------------------------------------------------------------*
 |  Project snode onto melement with master element normal   farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_normal_2d(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist) const
{
  if (ndim_ != 2) FOUR_C_THROW("project_s_node_by_m_normal_2d is only for 2D problems!");

  // start in the element center
  double eta[2] = {0.0, 0.0};

  // auxiliary variable for distance
  double alpha = 0.0;

  // function f (vector-valued)
  std::array<double, 3> f = {0.0, 0.0, 0.0};

  // gradient of f (df/deta[0], df/deta[1], df/dalpha)
  Core::LinAlg::Matrix<3, 3> df;

  // start iteration
  int k = 0;
  double conv = 0.0;

  for (k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************

    // build interpolation of master node coordinates for current eta
    double xm[3] = {0.0, 0.0, 0.0};
    double xs[3] = {0.0, 0.0, 0.0};
    double unormal[3] = {0.0, 0.0, 0.0};
    double normal[3] = {0.0, 0.0, 0.0};
    double normalpart[3] = {0.0, 0.0, 0.0};

    // calc xmaster
    Mortar::Utils::local_to_global<distype>(mele, eta, xm, 0);

    // calc normal part
    double length = mele.compute_unit_normal_at_xi(eta, unormal);
    for (int i = 0; i < 3; ++i) normalpart[i] = unormal[i] * alpha;

    for (int i = 0; i < 3; ++i) normal[i] = unormal[i] * length;

    // calc xslave
    for (int i = 0; i < 3; ++i) xs[i] = snode.xspatial()[i];

    // calculate F
    for (int i = 0; i < 3; ++i) f[i] = xm[i] + normalpart[i] - xs[i];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (conv <= MORTARCONVTOL) break;

    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************

    // master coordinate grad
    double meta0[3] = {0.0, 0.0, 0.0};  // x,xi_0
    double meta1[3] = {0.0, 0.0, 1.0};  // x,xi_1
    Mortar::Utils::local_to_global<distype>(mele, eta, meta0, 1);

    // normal grad
    Core::LinAlg::Matrix<1, n_> secderiv;
    std::array<double, 3> meta00 = {0.0, 0.0, 0.0};  // x , xi_0 xi_0
    std::array<double, 3> meta11 = {0.0, 0.0, 0.0};  // x , xi_1 xi_1
    std::array<double, 3> meta01 = {0.0, 0.0, 0.0};  // x , xi_0 xi_1

    Core::FE::shape_function_1d_deriv2(secderiv, eta[0], distype);

    for (int i = 0; i < n_; ++i)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[i]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int d = 0; d < 3; ++d) meta00[d] += secderiv(0, i) * mymnode->xspatial()[d];
    }

    std::array<double, 3> naux_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> naux_1 = {0.0, 0.0, 0.0};

    // normal grad xi_0
    naux_0[0] = (meta00[1] * meta1[2] - meta00[2] * meta1[1]);
    naux_0[1] = (meta00[2] * meta1[0] - meta00[0] * meta1[2]);
    naux_0[2] = (meta00[0] * meta1[1] - meta00[1] * meta1[0]);

    naux_0[0] += (meta0[1] * meta01[2] - meta0[2] * meta01[1]);
    naux_0[1] += (meta0[2] * meta01[0] - meta0[0] * meta01[2]);
    naux_0[2] += (meta0[0] * meta01[1] - meta0[1] * meta01[0]);

    // normal grad xi_1
    naux_1[0] = (meta01[1] * meta1[2] - meta01[2] * meta1[1]);
    naux_1[1] = (meta01[2] * meta1[0] - meta01[0] * meta1[2]);
    naux_1[2] = (meta01[0] * meta1[1] - meta01[1] * meta1[0]);

    naux_1[0] += (meta0[1] * meta11[2] - meta0[2] * meta11[1]);
    naux_1[1] += (meta0[2] * meta11[0] - meta0[0] * meta11[2]);
    naux_1[2] += (meta0[0] * meta11[1] - meta0[1] * meta11[0]);

    std::array<double, 3> n_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> n_1 = {0.0, 0.0, 0.0};

    double fac0 = 0.0;
    double fac1 = 0.0;

    for (int i = 0; i < 3; ++i) fac0 += naux_0[i] * normal[i];

    for (int i = 0; i < 3; ++i)
      n_0[i] = naux_0[i] / length - fac0 * normal[i] / (length * length * length);

    for (int i = 0; i < 3; ++i) fac1 += naux_1[i] * normal[i];

    for (int i = 0; i < 3; ++i)
      n_1[i] = naux_1[i] / length - fac1 * normal[i] / (length * length * length);

    // evaluate function f gradient
    for (int i = 0; i < 3; ++i)
    {
      df(i, 0) = meta0[i] + n_0[i];
      df(i, 1) = meta1[i] + n_1[i];
      df(i, 2) = unormal[i];
    }

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = Core::LinAlg::invert(df);
    if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

    // update eta and alpha
    eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
    eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
    alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];
  }  // end newton loop

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL) FOUR_C_THROW("Projector not converged!");

  // Newton iteration converged
  xi[0] = eta[0];
  mele.compute_unit_normal_at_xi(eta, normal);
  dist = alpha;

  return true;
}

/*----------------------------------------------------------------------*
 |  Project snode onto melement with master normal + Lin     farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_normal_2d_lin(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  if (ndim_ != 2) FOUR_C_THROW("project_s_node_by_m_normal_2d is only for 2D problems!");

  // start in the element center
  double eta[2] = {0.0, 0.0};

  // auxiliary variable for distance
  double alpha = 0.0;

  // function f (vector-valued)
  std::array<double, 3> f = {0.0, 0.0, 0.0};

  // gradient of f (df/deta[0], df/deta[1], df/dalpha)
  Core::LinAlg::Matrix<3, 3> df;

  // start iteration
  int k = 0;
  double conv = 0.0;

  for (k = 0; k < MORTARMAXITER; ++k)
  {
    //**********************************************
    //  F CALCULATION                             //
    //**********************************************

    // build interpolation of master node coordinates for current eta
    double xm[3] = {0.0, 0.0, 0.0};
    double xs[3] = {0.0, 0.0, 0.0};
    double unormal[3] = {0.0, 0.0, 0.0};
    double normal_k[3] = {0.0, 0.0, 0.0};
    double normalpart[3] = {0.0, 0.0, 0.0};

    // calc xmaster
    Mortar::Utils::local_to_global<distype>(mele, eta, xm, 0);

    // calc normal part
    double length = mele.compute_unit_normal_at_xi(eta, unormal);
    for (int i = 0; i < 3; ++i) normalpart[i] = unormal[i] * alpha;

    for (int i = 0; i < 3; ++i) normal_k[i] = unormal[i] * length;

    // calc xslave
    for (int i = 0; i < 3; ++i) xs[i] = snode.xspatial()[i];

    // calculate F
    for (int i = 0; i < 3; ++i) f[i] = xm[i] + normalpart[i] - xs[i];

    // check for convergence
    conv = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (conv <= MORTARCONVTOL) break;

    //**********************************************
    //   F GRADIENT CALCULATION                   //
    //**********************************************

    // master coordinate grad
    double meta0[3] = {0.0, 0.0, 0.0};  // x,xi_0
    double meta1[3] = {0.0, 0.0, 1.0};  // x,xi_1
    Mortar::Utils::local_to_global<distype>(mele, eta, meta0, 1);

    // normal grad
    Core::LinAlg::Matrix<1, n_> secderiv;
    std::array<double, 3> meta00 = {0.0, 0.0, 0.0};  // x , xi_0 xi_0
    std::array<double, 3> meta11 = {0.0, 0.0, 0.0};  // x , xi_1 xi_1
    std::array<double, 3> meta01 = {0.0, 0.0, 0.0};  // x , xi_0 xi_1

    Core::FE::shape_function_1d_deriv2(secderiv, eta[0], distype);

    for (int i = 0; i < n_; ++i)
    {
      const Node* mymnode = dynamic_cast<const Node*>(mele.nodes()[i]);
      if (!mymnode) FOUR_C_THROW("Null pointer!");
      for (int d = 0; d < 3; ++d) meta00[d] += secderiv(0, i) * mymnode->xspatial()[d];
    }

    std::array<double, 3> naux_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> naux_1 = {0.0, 0.0, 0.0};

    // normal grad xi_0
    naux_0[0] = (meta00[1] * meta1[2] - meta00[2] * meta1[1]);
    naux_0[1] = (meta00[2] * meta1[0] - meta00[0] * meta1[2]);
    naux_0[2] = (meta00[0] * meta1[1] - meta00[1] * meta1[0]);

    naux_0[0] += (meta0[1] * meta01[2] - meta0[2] * meta01[1]);
    naux_0[1] += (meta0[2] * meta01[0] - meta0[0] * meta01[2]);
    naux_0[2] += (meta0[0] * meta01[1] - meta0[1] * meta01[0]);

    // normal grad xi_1
    naux_1[0] = (meta01[1] * meta1[2] - meta01[2] * meta1[1]);
    naux_1[1] = (meta01[2] * meta1[0] - meta01[0] * meta1[2]);
    naux_1[2] = (meta01[0] * meta1[1] - meta01[1] * meta1[0]);

    naux_1[0] += (meta0[1] * meta11[2] - meta0[2] * meta11[1]);
    naux_1[1] += (meta0[2] * meta11[0] - meta0[0] * meta11[2]);
    naux_1[2] += (meta0[0] * meta11[1] - meta0[1] * meta11[0]);

    std::array<double, 3> n_0 = {0.0, 0.0, 0.0};
    std::array<double, 3> n_1 = {0.0, 0.0, 0.0};

    double fac0 = 0.0;
    double fac1 = 0.0;

    for (int i = 0; i < 3; ++i) fac0 += naux_0[i] * normal_k[i];

    for (int i = 0; i < 3; ++i)
      n_0[i] = naux_0[i] / length - fac0 * normal_k[i] / (length * length * length);

    for (int i = 0; i < 3; ++i) fac1 += naux_1[i] * normal_k[i];

    for (int i = 0; i < 3; ++i)
      n_1[i] = naux_1[i] / length - fac1 * normal_k[i] / (length * length * length);

    // evaluate function f gradient
    for (int i = 0; i < 3; ++i)
    {
      df(i, 0) = meta0[i] + n_0[i];
      df(i, 1) = meta1[i] + n_1[i];
      df(i, 2) = unormal[i];
    }

    //**********************************************
    //   solve deta = - inv(dF) * F               //
    //**********************************************
    double jacdet = Core::LinAlg::invert(df);
    if (abs(jacdet) < 1.0e-12) FOUR_C_THROW("Singular Jacobian for projection");

    // update eta and alpha
    eta[0] += -df(0, 0) * f[0] - df(0, 1) * f[1] - df(0, 2) * f[2];
    eta[1] += -df(1, 0) * f[0] - df(1, 1) * f[1] - df(1, 2) * f[2];
    alpha += -df(2, 0) * f[0] - df(2, 1) * f[1] - df(2, 2) * f[2];
  }  // end newton loop

  // Newton iteration unconverged
  if (conv > MORTARCONVTOL) FOUR_C_THROW("Projector not converged!");

  //**********************************************
  //   Get stuff                                //
  //**********************************************
  // Newton iteration converged
  xi[0] = eta[0];
  double normallength = mele.compute_unit_normal_at_xi(eta, normal);
  dist = alpha;


  //**********************************************
  //   Lin deta = - inv(dF) * Lin F             //
  //**********************************************
  // prepare linearizations
  using CI = Core::Gen::Pairedvector<int, double>::const_iterator;

  std::vector<Core::Gen::Pairedvector<int, double>> etaLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> fLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> xmLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> normalpartLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> xsLin(3, 1000);

  //--------------------------
  // master part:
  Core::LinAlg::Matrix<n_, 1> val;
  Core::FE::shape_function_1d(val, eta[0], distype);

  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const Node* mnode = dynamic_cast<const Node*>(node);

    for (int k = 0; k < 2; ++k) (xmLin[k])[mnode->dofs()[k]] += val(i);
  }

  //--------------------------
  // normal part:
  std::vector<Core::Gen::Pairedvector<int, double>> x_0Lin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> auxnormalLin(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> auxnormalunitLin(3, 1000);

  Core::LinAlg::Matrix<1, n_> deriv1;
  Core::FE::shape_function_1d_deriv1(deriv1, eta[0], distype);
  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const Node* mnode = dynamic_cast<const Node*>(node);

    for (int k = 0; k < 2; ++k) (x_0Lin[k])[mnode->dofs()[k]] += deriv1(i);
  }

  // cross product linearization
  for (CI p = x_0Lin[1].begin(); p != x_0Lin[1].end(); ++p)
    (auxnormalLin[0])[p->first] += (p->second);
  for (CI p = x_0Lin[0].begin(); p != x_0Lin[0].end(); ++p)
    (auxnormalLin[1])[p->first] -= (p->second);

  // calc normalpart without alpha
  std::array<double, 3> linnormalaux = {0.0, 0.0, 0.0};
  linnormalaux[0] = normal[0] * normallength;
  linnormalaux[1] = normal[1] * normallength;
  linnormalaux[2] = normal[2] * normallength;

  // derivative weighting matrix for current element
  Core::LinAlg::Matrix<3, 3> W;
  const double lcubeinv = 1.0 / (normallength * normallength * normallength);

  for (int j = 0; j < 3; ++j)
  {
    for (int k = 0; k < 3; ++k)
    {
      W(j, k) = -lcubeinv * linnormalaux[j] * linnormalaux[k];
      if (j == k) W(j, k) += 1 / normallength;
    }
  }

  // row loop
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = auxnormalLin[0].begin(); p != auxnormalLin[0].end(); ++p)
      (auxnormalunitLin[j])[p->first] += (p->second) * W(j, 0);

    for (CI p = auxnormalLin[1].begin(); p != auxnormalLin[1].end(); ++p)
      (auxnormalunitLin[j])[p->first] += (p->second) * W(j, 1);

    for (CI p = auxnormalLin[2].begin(); p != auxnormalLin[2].end(); ++p)
      (auxnormalunitLin[j])[p->first] += (p->second) * W(j, 2);
  }

  for (int j = 0; j < 3; ++j)
  {
    for (CI p = auxnormalunitLin[j].begin(); p != auxnormalunitLin[j].end(); ++p)
      (normalpartLin[j])[p->first] += (p->second) * alpha;
  }


  //--------------------------
  // slave part:
  for (int k = 0; k < 2; ++k) (xsLin[k])[snode.dofs()[k]] += 1.0;

  // All terms:
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = xmLin[j].begin(); p != xmLin[j].end(); ++p) (fLin[j])[p->first] += (p->second);

    for (CI p = normalpartLin[j].begin(); p != normalpartLin[j].end(); ++p)
      (fLin[j])[p->first] += (p->second);

    for (CI p = xsLin[j].begin(); p != xsLin[j].end(); ++p) (fLin[j])[p->first] -= (p->second);
  }


  for (int i = 0; i < 3; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      for (CI p = fLin[i].begin(); p != fLin[i].end(); ++p)
        (etaLin[k])[p->first] -= (p->second) * df(k, i);
    }
  }

  //**********************************************
  //   Lin N                                    //
  //**********************************************
  std::vector<Core::Gen::Pairedvector<int, double>> x_0Linnew(3, 1000);
  std::vector<Core::Gen::Pairedvector<int, double>> normaltolineLinaux(3, 1000);

  Core::LinAlg::Matrix<1, n_> deriv;
  Core::FE::shape_function_1d_deriv1(deriv, eta[0], distype);

  Core::LinAlg::Matrix<1, n_> deriv2;
  Core::FE::shape_function_1d_deriv2(deriv2, eta[0], distype);
  for (int i = 0; i < n_; ++i)
  {
    // get master node
    const Core::Nodes::Node* node = mele.nodes()[i];
    if (!node) FOUR_C_THROW("Cannot find master node");
    const Node* mnode = dynamic_cast<const Node*>(node);

    for (int k = 0; k < 2; ++k) (x_0Linnew[k])[mnode->dofs()[k]] += deriv(i);

    for (int k = 0; k < 2; ++k)
    {
      for (CI p = etaLin[0].begin(); p != etaLin[0].end(); ++p)
        (x_0Linnew[k])[p->first] += (p->second) * deriv2(i) * mnode->xspatial()[k];
    }
  }

  // cross product linearization
  for (CI p = x_0Linnew[1].begin(); p != x_0Linnew[1].end(); ++p)
    (normaltolineLinaux[0])[p->first] += (p->second);
  for (CI p = x_0Linnew[0].begin(); p != x_0Linnew[0].end(); ++p)
    (normaltolineLinaux[1])[p->first] -= (p->second);

  // normalize lin
  Core::LinAlg::Matrix<3, 3> Wfinal;
  //  const double lcubeinv = 1.0 / (normallength * normallength * normallength);

  for (int j = 0; j < 3; ++j)
  {
    for (int k = 0; k < 3; ++k)
    {
      Wfinal(j, k) = -lcubeinv * linnormalaux[j] * linnormalaux[k];
      if (j == k) Wfinal(j, k) += 1 / normallength;
    }
  }

  // row loop
  for (int j = 0; j < 3; ++j)
  {
    for (CI p = normaltolineLinaux[0].begin(); p != normaltolineLinaux[0].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second) * Wfinal(j, 0);

    for (CI p = normaltolineLinaux[1].begin(); p != normaltolineLinaux[1].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second) * Wfinal(j, 1);

    for (CI p = normaltolineLinaux[2].begin(); p != normaltolineLinaux[2].end(); ++p)
      (normaltolineLin[j])[p->first] += (p->second) * Wfinal(j, 2);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Project snode onto melement with master element normal   farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_normal(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist) const
{
  if (ndim_ == 2)
  {
    project_s_node_by_m_normal_2d(snode, mele, xi, normal, dist);
  }
  else if (ndim_ == 3)
  {
    project_s_node_by_m_normal_3d(snode, mele, xi, normal, dist);
  }
  else
  {
    FOUR_C_THROW("wrong dimension!");
  }

  return true;
}


/*----------------------------------------------------------------------*
 |  Project snode onto melement with master element normal   farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_nodal_normal_lin(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  bool success = false;

  if (ndim_ == 2)
  {
    success =
        project_s_node_by_m_nodal_normal_2d_lin(snode, mele, xi, normal, dist, normaltolineLin);
  }
  else if (ndim_ == 3)
  {
    success =
        project_s_node_by_m_nodal_normal_3d_lin(snode, mele, xi, normal, dist, normaltolineLin);
  }
  else
  {
    FOUR_C_THROW("wrong dimension!");
  }

  return success;
}

/*----------------------------------------------------------------------*
 |  Project snode onto melement with master normal           farah 01/16|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::project_s_node_by_m_normal_lin(const Mortar::Node& snode,
    const Mortar::Element& mele, double* xi, double* normal, double& dist,
    std::vector<Core::Gen::Pairedvector<int, double>>& normaltolineLin) const
{
  if (ndim_ == 2)
  {
    project_s_node_by_m_normal_2d_lin(snode, mele, xi, normal, dist, normaltolineLin);
  }
  else if (ndim_ == 3)
  {
    FOUR_C_THROW("Not yet implemented!");
  }
  else
  {
    FOUR_C_THROW("wrong dimension!");
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate F for nodal normal case (public)                 popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Mortar::ProjectorCalc<distype>::evaluate_f_nodal_normal(
    const Mortar::Node& node, const Mortar::Element& ele, const double* eta) const
{
  /* Evaluate the function F(eta) = ( Ni * xim - xs ) x ns,
   or to be more precise the third component of this vector function!

   Ni  shape functions of element
   xim coords of element nodes (master side)
   xs  coords of node to be projected (slave side)
   ns   outward normal of node to be projected (slave side)          */
  double fval = 0.0;

  // build interpolation of master node coordinates for current eta
  double nx[ndim_];
  Mortar::Utils::local_to_global<distype>(ele, eta, nx, 0);

  // subtract slave node coordinates
  for (int i = 0; i < ndim_; ++i) nx[i] -= node.xspatial()[i];

  // calculate F
  fval = nx[0] * node.mo_data().n()[1] - nx[1] * node.mo_data().n()[0];

  return fval;
}

/*----------------------------------------------------------------------*
 |  Evaluate GradF for nodal normal case (public)             popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Mortar::ProjectorCalc<distype>::evaluate_grad_f_nodal_normal(
    const Mortar::Node& node, const Mortar::Element& ele, const double* eta) const
{
  /* Evaluate the function GradF(eta)
   = Ni,eta * xim * nys - Ni,eta * yim * nxs,

   Ni,eta    shape function derivatives of element
   xim, yim  coords of element nodes (master side)
   nxs, nys   outward normal of node to be projected (slave side)   */

  double fgrad = 0.0;

  // build interpolation of master node coordinates for current eta
  // use shape function derivatives for interpolation (hence "1")
  double nxeta[ndim_];
  Mortar::Utils::local_to_global<distype>(ele, eta, nxeta, 1);

  // calculate GradF
  fgrad = nxeta[0] * node.mo_data().n()[1] - nxeta[1] * node.mo_data().n()[0];

  return fgrad;
}

/*----------------------------------------------------------------------*
 |  Evaluate F for element normal case (public)               popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Mortar::ProjectorCalc<distype>::evaluate_f_element_normal(
    const Mortar::Node& node, const Mortar::Element& ele, const double* eta) const
{
  /* Evaluate the function F(eta) = ( Ni * xis - xm ) x ( Nj * njs),
   or to be more precise the third component of this vector function!

   Ni  shape functions of element
   xis coords of element nodes (slave side)
   xm  coords of node to be projected (master side)
   nis outward normals of element nodes (slave side)                */

  double fval = 0.0;

  // collect necessary data (slave side)
  const Core::Nodes::Node* const* mynodes = ele.nodes();
  if (!mynodes) FOUR_C_THROW("evaluate_f_element_normal: Null pointer!");

  Core::LinAlg::Matrix<n_, 1> val;
  Core::LinAlg::Matrix<ndim_, n_> coord;

  // get shape function values and derivatives at gpeta
  if (distype == Core::FE::CellType::nurbs2 || distype == Core::FE::CellType::nurbs3)
  {
    Core::LinAlg::SerialDenseVector auxval(n_);
    Core::LinAlg::SerialDenseMatrix deriv(n_, 1);
    ele.evaluate_shape(eta, auxval, deriv, ele.num_node());

    for (int i = 0; i < n_; ++i) val(i) = auxval(i);
  }
  else
    Core::FE::shape_function_1d(val, eta[0], distype);

  // get interpolated normal and proj. coordinates for current eta
  double nn[ndim_];
  double nx[ndim_];
  for (int j = 0; j < ndim_; ++j)
  {
    nn[j] = 0.0;
    nx[j] = 0.0;
  }

  for (int i = 0; i < n_; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);

    for (int j = 0; j < ndim_; ++j)
    {
      nn[j] += val(i) * mymrtrnode->mo_data().n()[j];

      coord(j, i) = mymrtrnode->xspatial()[j];

      nx[j] += val(i) * coord(j, i);
    }
  }

  // subtract master node coordinates
  for (int j = 0; j < ndim_; ++j) nx[j] -= node.xspatial()[j];

  // calculate F
  fval = nx[0] * nn[1] - nx[1] * nn[0];

  return fval;
}

/*----------------------------------------------------------------------*
 |  Evaluate GradF for element normal case (public)           popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double Mortar::ProjectorCalc<distype>::evaluate_grad_f_element_normal(
    const Mortar::Node& node, const Mortar::Element& ele, const double* eta) const
{
  if (ndim_ == 3) FOUR_C_THROW("This Projector is only for 2D Problems!");

  /* Evaluate the function GradF(eta)
   = ( Ni,eta * xis ) * ( Nj * nyjs )
   + ( Ni * xis - xm ) * ( Nj,eta * nyjs )
   - ( Ni,eta * yis ) * ( Nj * nxjs )
   - ( Ni * yis - ym ) * ( Nj,eta * nxjs )

   Ni,eta      shape function derivatives of element
   xis, yis   coords of element nodes (slave side)
   xm, ym     coords of node to be projected (master side)
   nxjs, nyjs outward normals of element nodes (slave side)         */

  double fgrad = 0.0;

  // collect necessary data (slave side)
  Core::LinAlg::Matrix<n_, 1> val;
  Core::LinAlg::Matrix<ndim_ - 1, n_> deriv;
  Core::LinAlg::Matrix<ndim_, n_> coord;

  const Core::Nodes::Node* const* mynodes = ele.nodes();
  if (!mynodes) FOUR_C_THROW("evaluate_grad_f_element_normal: Null pointer!");

  // get shape function values and derivatives at gpeta
  if (distype == Core::FE::CellType::nurbs2 || distype == Core::FE::CellType::nurbs3)
  {
    Core::LinAlg::SerialDenseVector auxval(n_);
    Core::LinAlg::SerialDenseMatrix auxderiv(n_, 1);
    ele.evaluate_shape(eta, auxval, auxderiv, ele.num_node());

    for (int i = 0; i < n_; ++i)
    {
      val(i) = auxval(i);
      deriv(0, i) = auxderiv(i, 0);
    }
  }
  else
  {
    Core::FE::shape_function_1d(val, eta[0], distype);
    Core::FE::shape_function_1d_deriv1(deriv, eta[0], distype);
  }

  // get interpolated normal and proj. coordinates for current eta
  double nn[ndim_];
  double nneta[ndim_];
  double nx[ndim_];
  double nxeta[ndim_];
  for (int j = 0; j < ndim_; ++j)
  {
    nn[j] = 0.0;
    nneta[j] = 0.0;
    nx[j] = 0.0;
    nxeta[j] = 0.0;
  }

  for (int i = 0; i < n_; ++i)
  {
    const Node* mymrtrnode = dynamic_cast<const Node*>(mynodes[i]);

    for (int j = 0; j < ndim_; ++j)
    {
      nn[j] += val(i) * mymrtrnode->mo_data().n()[j];
      nneta[j] += deriv(0, i) * mymrtrnode->mo_data().n()[j];
      coord(j, i) = mymrtrnode->xspatial()[j];
      nx[j] += val(i) * coord(j, i);
      nxeta[j] += deriv(0, i) * coord(j, i);
    }
  }

  // subtract master node coordinates
  for (int j = 0; j < ndim_; ++j) nx[j] -= node.xspatial()[j];

  // calculate GradF
  fgrad = nxeta[0] * nn[1] + nx[0] * nneta[1] - nxeta[1] * nn[0] - nx[1] * nneta[0];

  return fgrad;
}

/*----------------------------------------------------------------------*
 |  Evaluate F for Gauss point case (public)                  popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
double Mortar::ProjectorCalcEleBased<distype_s, distype_m>::evaluate_f_gauss_point_2d(
    const double* gpx, const double* gpn, const Mortar::Element& ele, const double* eta) const
{
  /* Evaluate the function F(eta) = ( Ni * xim - gpx ) x gpn,
   or to be more precise the third component of this vector function!

   Ni  shape functions of element (master side)
   xim coords of element nodes (master side)
   gpx coords of GP to be projected (slave side)
   gpn outward normal of GP to be projected (slave side)          */

  double fval = 0.0;

  // build interpolation of master node coordinates for current eta
  double nx[ndim_];
  Mortar::Utils::local_to_global<distype_m>(ele, eta, nx, 0);

  // subtract GP coordinates
  nx[0] -= gpx[0];
  nx[1] -= gpx[1];

  // calculate F
  fval = nx[0] * gpn[1] - nx[1] * gpn[0];

  return fval;
}

/*----------------------------------------------------------------------*
 |  Evaluate GradF for Gauss point case (public)              popp 01/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
double Mortar::ProjectorCalcEleBased<distype_s, distype_m>::evaluate_grad_f_gauss_point_2d(
    const double* gpn, const Mortar::Element& ele, const double* eta) const
{
  /* Evaluate the function GradF(eta)
   = Ni,eta * xim * gpny - Ni,eta * yim * gpnx,

   Ni,eta     shape function derivatives of element (master side)
   xim, yim   coords of element nodes (master side)
   gpnx, gpny outward normal of GP to be projected (slave side)   */

  double fgrad = 0.0;

  // build interpolation of master node coordinates for current eta
  // use shape function derivatives for interpolation (hence "1")
  double nxeta[ndim_];
  Mortar::Utils::local_to_global<distype_m>(ele, eta, nxeta, 1);

  // calculate GradF
  fgrad = nxeta[0] * gpn[1] - nxeta[1] * gpn[0];

  return fgrad;
}

/*----------------------------------------------------------------------*
 |  Evaluate F for Gauss point case (3D)                      popp 11/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
bool Mortar::ProjectorCalcEleBased<distype_s, distype_m>::evaluate_f_gauss_point_3d(double* f,
    const double* gpx, const double* gpn, const Mortar::Element& ele, const double* eta,
    const double& alpha) const
{
  /* Evaluate the function F(eta,alpha) = Ni * xi - alpha * gpn - gpx
   which is a vector-valued function with 3 components!

   Ni      shape functions of element to project on
   xi      coords of nodes of element to project on
   gpx     coords of GP to be projected
   gpn     normal of GP along which to project                  */

  // build interpolation of ele node coordinates for current eta
  double nx[ndim_];
  Mortar::Utils::local_to_global<distype_m>(ele, eta, nx, 0);

  // evaluate function f
  for (int i = 0; i < ndim_; ++i) f[i] = nx[i] - alpha * gpn[i] - gpx[i];

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate GradF for Gauss point case (3D)                  popp 11/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype_s, Core::FE::CellType distype_m>
bool Mortar::ProjectorCalcEleBased<distype_s, distype_m>::evaluate_grad_f_gauss_point_3d(
    Core::LinAlg::Matrix<3, 3>& fgrad, const double* gpx, const double* gpn,
    const Mortar::Element& ele, const double* eta, const double& alpha) const
{
  /* Evaluate the gradient of the function F(eta,alpha) = Ni * xi -
   - alpha * gpn - gpx, which is a (3x3)-matrix!

   Ni      shape functions of element to project on
   xi      coords of nodes of element to project on
   gpx     coords of GP to be projected
   gpn     normal of GP along which to project                  */

  // build interpolation of ele node coordinates for current eta
  //  double nxeta1[3] = {0.0, 0.0, 0.0};
  //  double nxeta2[3] = {0.0, 0.0, 0.0};
  //  ele.local_to_global(eta,nxeta1,1);
  //  ele.local_to_global(eta,nxeta2,2);
  double* nxeta1 = &fgrad(0, 0);
  double* nxeta2 = &fgrad(0, 1);
  Mortar::Utils::local_to_global<distype_m>(ele, eta, nxeta1, 1);
  Mortar::Utils::local_to_global<distype_m>(ele, eta, nxeta2, 2);

  // evaluate function f gradient
  for (int i = 0; i < ndim_; ++i) fgrad(i, 2) = -gpn[i];

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate F for AuxPlane Gauss point case (3D)             popp 11/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::evaluate_f_gauss_point_auxn_3d(double* f, const double* globgp,
    const double* auxn, const Mortar::Element& ele, const double* eta, const double& alpha) const
{
  /* Evaluate the function F(eta,alpha) = Ni * xi - alpha * auxn - globgp
   which is a vector-valued function with 3 components!

   Ni      shape functions of element to project on
   xi      coords of nodes of element to project on
   globgp  coords of AuxPlaneGP to be projected
   auxn    normal of AuxPlane along which to project            */

  // build interpolation of ele node coordinates for current eta
  double nx[ndim_];
  Mortar::Utils::local_to_global<distype>(ele, eta, nx, 0);

  // evaluate function f
  for (int i = 0; i < ndim_; ++i) f[i] = nx[i] - alpha * auxn[i] - globgp[i];

  return true;
}

/*----------------------------------------------------------------------*
 |  Evaluate GradF for AuxPlane Gauss point case (3D)         popp 11/08|
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
bool Mortar::ProjectorCalc<distype>::evaluate_grad_f_gauss_point_auxn_3d(
    Core::LinAlg::Matrix<3, 3>& fgrad, const double* globgp, const double* auxn,
    const Mortar::Element& ele, const double* eta, const double& alpha) const
{
  /* Evaluate the gradient of the function F(eta,alpha) = Ni * xi -
   - alpha * auxn - globgp, which is a (3x3)-matrix!

   Ni      shape functions of element to project on
   xi      coords of nodes of element to project on
   globgp  coords of AuxPlaneGP to be projected
   auxn    normal of AuxPlane along which to project            */

  // build interpolation of ele node coordinates for current eta
  //  double nxeta1[ndim_];
  //  double nxeta2[ndim_];
  double* nxeta1 = &fgrad(0, 0);
  double* nxeta2 = &fgrad(0, 1);
  Mortar::Utils::local_to_global<distype>(ele, eta, nxeta1, 1);
  Mortar::Utils::local_to_global<distype>(ele, eta, nxeta2, 2);

  // evaluate function f gradient
  for (int i = 0; i < ndim_; ++i) fgrad(i, 2) = -auxn[i];

  return true;
}

FOUR_C_NAMESPACE_CLOSE
