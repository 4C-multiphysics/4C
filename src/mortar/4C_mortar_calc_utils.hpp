// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MORTAR_CALC_UTILS_HPP
#define FOUR_C_MORTAR_CALC_UTILS_HPP

/*----------------------------------------------------------------------*
 | Header                                                   farah 01/14 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_node.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | Utils                                                    farah 01/14 |
 *----------------------------------------------------------------------*/
namespace Mortar
{
  class Element;
  class Node;

  namespace Utils
  {
    /*----------------------------------------------------------------------*
     |  Get global coords for given local coords                 farah 01/14|
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    bool local_to_global(
        const Mortar::Element& ele, const double* xi, double* globcoord, int inttype)
    {
      // check input
      if (!xi) FOUR_C_THROW("ERROR: local_to_global called with xi=nullptr");
      if (!globcoord) FOUR_C_THROW("ERROR: local_to_global called with globcoord=nullptr");

      static constexpr int n = Core::FE::num_nodes<distype>;
      static constexpr int ndim = Core::FE::dim<distype> + 1;

      const Core::Nodes::Node* const* mynodes = ele.points();
      if (!mynodes) FOUR_C_THROW("ERROR: local_to_global: Null pointer!");

      std::fill(globcoord, globcoord + ndim, 0.0);

      //===========================================
      // Basic value
      switch (inttype)
      {
        case 0:
        {
          Core::LinAlg::Matrix<n, 1> val;
          switch (ndim)
          {
            case 2:
            {
              switch (distype)
              {
                case Core::FE::CellType::nurbs2:
                case Core::FE::CellType::nurbs3:
                {
                  Core::LinAlg::SerialDenseVector auxval(n);
                  Core::LinAlg::SerialDenseMatrix auxderiv(n, 1);
                  ele.evaluate_shape(xi, auxval, auxderiv, ele.num_node());

                  for (int i = 0; i < n; ++i) val(i) = auxval(i);

                  break;
                }
                default:
                {
                  Core::FE::shape_function_1d(val, xi[0], distype);
                  break;
                }
              }
              break;
            }
            case 3:
            {
              switch (distype)
              {
                case Core::FE::CellType::nurbs4:
                case Core::FE::CellType::nurbs8:
                case Core::FE::CellType::nurbs9:
                {
                  Core::LinAlg::SerialDenseVector auxval(n);
                  Core::LinAlg::SerialDenseMatrix auxderiv(n, 2);
                  ele.evaluate_shape(xi, auxval, auxderiv, ele.num_node());

                  for (int i = 0; i < n; ++i) val(i) = auxval(i);

                  break;
                }
                default:
                {
                  Core::FE::shape_function_2d(val, xi[0], xi[1], distype);
                  break;
                }
              }

              break;
            }
            default:
            {
              FOUR_C_THROW("Wrong Dimension");
              exit(EXIT_FAILURE);
            }
          }

          for (int i = 0; i < n; ++i)
          {
            const Node* mymrtrnode = static_cast<const Node*>(mynodes[i]);
            FOUR_C_ASSERT(mymrtrnode, "ERROR: local_to_global: Null pointer!");

            for (int j = 0; j < ndim; ++j)
            {
              // use shape function values for interpolation
              globcoord[j] += val(i) * mymrtrnode->xspatial()[j];
            }
          }
          break;
        }
        //===========================================
        // First Derivation (xi)
        case 1:
        {
          Core::LinAlg::Matrix<ndim - 1, n> deriv1;

          switch (ndim)
          {
            case 2:
            {
              switch (distype)
              {
                case Core::FE::CellType::nurbs2:
                case Core::FE::CellType::nurbs3:
                {
                  Core::LinAlg::SerialDenseVector auxval(n);
                  Core::LinAlg::SerialDenseMatrix auxderiv(n, 1);
                  ele.evaluate_shape(xi, auxval, auxderiv, ele.num_node());

                  for (int i = 0; i < n; ++i) deriv1(0, i) = auxderiv(i, 0);

                  break;
                }
                default:
                {
                  Core::FE::shape_function_1d_deriv1(deriv1, xi[0], distype);
                  break;
                }
              }

              break;
            }
            case 3:
            {
              switch (distype)
              {
                case Core::FE::CellType::nurbs4:
                case Core::FE::CellType::nurbs8:
                case Core::FE::CellType::nurbs9:
                {
                  Core::LinAlg::SerialDenseVector auxval(n);
                  Core::LinAlg::SerialDenseMatrix auxderiv(n, 2);
                  ele.evaluate_shape(xi, auxval, auxderiv, ele.num_node());

                  for (int i = 0; i < n; ++i) deriv1(0, i) = auxderiv(i, 0);

                  break;
                }
                default:
                {
                  Core::FE::shape_function_2d_deriv1(deriv1, xi[0], xi[1], distype);
                  break;
                }
              }

              break;
            }
            default:
            {
              FOUR_C_THROW("Wrong Dimension");
              break;
            }
          }
          for (int i = 0; i < n; ++i)
          {
            const Node* mymrtrnode = static_cast<const Node*>(mynodes[i]);
            FOUR_C_ASSERT(mymrtrnode, "ERROR: local_to_global: Null pointer!");

            for (int j = 0; j < ndim; ++j)
            {
              // use shape function values for interpolation
              globcoord[j] += deriv1(0, i) * mymrtrnode->xspatial()[j];
            }
          }
          break;
        }
        //===========================================
        // Second Derivation (eta)
        case 2:
        {
          Core::LinAlg::Matrix<ndim - 1, n> deriv2(Core::LinAlg::Initialization::zero);
          switch (ndim)
          {
            case 2:
            {
              switch (distype)
              {
                case Core::FE::CellType::nurbs2:
                case Core::FE::CellType::nurbs3:
                {
                  Core::LinAlg::SerialDenseVector auxval(n);
                  Core::LinAlg::SerialDenseMatrix auxderiv(n, 1);
                  ele.evaluate_shape(xi, auxval, auxderiv, ele.num_node());

                  for (int i = 0; i < n; ++i) deriv2(0, i) = auxderiv(i, 0);

                  break;
                }
                default:
                {
                  Core::FE::shape_function_1d_deriv1(deriv2, xi[0], distype);
                  break;
                }
              }

              break;
            }
            case 3:
            {
              switch (distype)
              {
                case Core::FE::CellType::nurbs4:
                case Core::FE::CellType::nurbs8:
                case Core::FE::CellType::nurbs9:
                {
                  Core::LinAlg::SerialDenseVector auxval(n);
                  Core::LinAlg::SerialDenseMatrix auxderiv(n, 2);
                  ele.evaluate_shape(xi, auxval, auxderiv, ele.num_node());

                  for (int i = 0; i < n; ++i) deriv2(1, i) = auxderiv(i, 1);

                  break;
                }
                default:
                {
                  Core::FE::shape_function_2d_deriv1(deriv2, xi[0], xi[1], distype);
                  break;
                }
              }

              break;
            }
            default:
            {
              FOUR_C_THROW("Wrong Dimension");
              exit(EXIT_FAILURE);
            }
          }

          for (int i = 0; i < n; ++i)
          {
            const Node* mymrtrnode = static_cast<const Node*>(mynodes[i]);
            FOUR_C_ASSERT(mymrtrnode, "ERROR: local_to_global: Null pointer!");

            if constexpr (ndim > 2)
            {
              for (int j = 0; j < ndim; ++j)
              {
                // use shape function values for interpolation
                globcoord[j] += deriv2(1, i) * mymrtrnode->xspatial()[j];
              }
            }
            else
              FOUR_C_THROW("Not implemented.");
          }
          break;
        }
        default:
        {
          FOUR_C_THROW("ERROR: Invalid interpolation type requested, only 0,1,2!");
          exit(EXIT_FAILURE);
        }
      }
      return true;
    };


    /*----------------------------------------------------------------------*
     |  Get global coords for given local coords (ref pos)       farah 01/14|
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    bool local_to_global(Core::Elements::Element& ele, const double* xi, double* globcoord)
    {
      // check input
      if (!xi) FOUR_C_THROW("ERROR: local_to_global called with xi=nullptr");
      if (!globcoord) FOUR_C_THROW("ERROR: local_to_global called with globcoord=nullptr");

      static constexpr int n = Core::FE::num_nodes<distype>;
      static constexpr int ndim = Core::FE::dim<distype> + 1;

      Core::Nodes::Node** mynodes = ele.nodes();
      if (!mynodes) FOUR_C_THROW("ERROR: local_to_global: Null pointer!");

      for (int i = 0; i < ndim; ++i) globcoord[i] = 0.0;

      Core::LinAlg::Matrix<ndim, n> coord;

      Core::LinAlg::Matrix<n, 1> val;
      if (ndim == 2)
        Core::FE::shape_function_1d(val, xi[0], distype);
      else if (ndim == 3)
        Core::FE::shape_function_2d(val, xi[0], xi[1], distype);
      else
        FOUR_C_THROW("Wrong Dimension");

      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < ndim; ++j)
        {
          coord(j, i) = mynodes[i]->x()[j];

          // use shape function values for interpolation
          globcoord[j] += val(i) * coord(j, i);
        }
      }


      return true;
    };
    /*---------------------------------------------------------------------------*
     |  Get global coords for given local coords (ref pos)            rauch 04/14|
     |  in current configuration.                                                |
     |                                                                           |
     |  template parameter distype is the discr. type of input element           |
     |                                                                           |
     |  ele     (in): element to consider.                                       |
     |  globdim (in): dimension of global problem                                |
     |  xi      (in): local reference coordinate within "ele".                   |
     |  edisp   (in): nodal displacements of "ele".                              |
     |                                                                           |
     |  globcoord (out): global xyz coordinate in current configuration of "xi"  |
     *---------------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    bool local_to_current_global(Core::Elements::Element& ele, const int globdim, const double* xi,
        const std::vector<double> edisp, double* globcoord)
    {
      // check input
      if (!xi) FOUR_C_THROW("ERROR: local_to_global called with xi=nullptr");
      if (!globcoord) FOUR_C_THROW("ERROR: local_to_global called with globcoord=nullptr");

      static constexpr int n = Core::FE::num_nodes<distype>;
      static constexpr int ndim = Core::FE::dim<distype>;

      if ((int)edisp.size() != n * globdim)
        FOUR_C_THROW("ERROR: vector of element displacements has wrong dimension ({} != {})",
            n * globdim, edisp.size());

      Core::Nodes::Node** mynodes = ele.nodes();
      if (!mynodes) FOUR_C_THROW("ERROR: local_to_global: Null pointer!");

      for (int i = 0; i < globdim; ++i) globcoord[i] = 0.0;

      Core::LinAlg::Matrix<ndim + 1, n> coord;

      Core::LinAlg::Matrix<n, 1> val;
      if (ndim == 1)
        Core::FE::shape_function_1d(val, xi[0], distype);
      else if (ndim == 2)
        Core::FE::shape_function_2d(val, xi[0], xi[1], distype);
      else if (ndim == 3)
        Core::FE::shape_function_3d(val, xi[0], xi[1], xi[2], distype);

      else
        FOUR_C_THROW("Wrong Dimension");

      for (int i = 0; i < n; ++i)
      {
        for (int j = 0; j < globdim; ++j)
        {
          coord(j, i) = mynodes[i]->x()[j] + edisp[i * globdim + j];

          // use shape function values for interpolation
          globcoord[j] += val(i) * coord(j, i);
        }
      }

      return true;
    };
    /*----------------------------------------------------------------------*
     |  Get local coords for given global coords (ref position)  farah 01/14|
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void global_to_local(Core::Elements::Element& ele,  // element (input)
        double* xgl,                                    // global position (input)
        double* xi)                                     // local position  (output)
    {
      static constexpr int numnod = Core::FE::num_nodes<distype>;
      static constexpr int ndim = Core::FE::dim<distype>;

      Core::LinAlg::Matrix<numnod, 1> funct(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<ndim, numnod> xref(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<ndim, ndim> xjm(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<ndim, numnod> deriv(Core::LinAlg::Initialization::zero);

      // spatial configuration of this element!
      for (int k = 0; k < numnod; ++k)
        for (int j = 0; j < ndim; ++j) xref(j, k) = ele.nodes()[k]->x()[j];

      // first estimation for parameter space coordinates
      for (int p = 0; p < ndim; ++p)
      {
        if (distype == Core::FE::CellType::quad4 or distype == Core::FE::CellType::quad8 or
            distype == Core::FE::CellType::quad9 or distype == Core::FE::CellType::hex8 or
            distype == Core::FE::CellType::hex20 or distype == Core::FE::CellType::hex27)
        {
          xi[p] = 0.0;
        }
        else if (distype == Core::FE::CellType::tri3 or distype == Core::FE::CellType::tri6 or
                 distype == Core::FE::CellType::tet4 or distype == Core::FE::CellType::tet10)
        {
          xi[p] = 1.0 / 3.0;
        }
        else if (distype == Core::FE::CellType::pyramid5)
        {
          if (p < 2)
            xi[p] = 0.0;
          else
            xi[p] = 1.0 / 3.0;
        }
        else
        {
          FOUR_C_THROW("ERROR: Element type not supported for parameter space mapping!");
        }
      }

      double rhs[ndim];
      for (int p = 0; p < ndim; ++p) rhs[p] = 0.0;

      // converged
      bool converged = false;
      int j = 0;

      //************************************************
      // loop
      while (!converged and j < 10)
      {
        // reset matriced
        xjm.clear();
        deriv.clear();

        if (ndim == 2)
        {
          Core::FE::shape_function_2d(funct, xi[0], xi[1], distype);
          Core::FE::shape_function_2d_deriv1(deriv, xi[0], xi[1], distype);
        }
        else if (ndim == 3)
        {
          Core::FE::shape_function_3d(funct, xi[0], xi[1], xi[2], distype);
          Core::FE::shape_function_3d_deriv1(deriv, xi[0], xi[1], xi[2], distype);
        }
        else
          FOUR_C_THROW("ERROR");

        for (int k = 0; k < numnod; ++k)
          for (int p = 0; p < ndim; ++p)
            for (int l = 0; l < ndim; ++l) xjm(p, l) += deriv(l, k) * xref(p, k);

        // rhs of (linearized equation)
        for (int p = 0; p < ndim; ++p) rhs[p] = 0.0;

        for (int p = 0; p < ndim; ++p) rhs[p] = -xgl[p];

        for (int k = 0; k < numnod; ++k)
          for (int p = 0; p < ndim; ++p) rhs[p] += funct(k) * xref(p, k);

        double norm = 0.0;
        for (int p = 0; p < ndim; ++p) norm += rhs[p] * rhs[p];

        if (sqrt(norm) < 1e-12)
        {
          converged = true;
        }
        //     else
        //     {
        //       std::cout << "norm= " << norm << " xi[0]=  " << xi[0]<< " xi[1]=  " << xi[1]<< "
        //       xi[2]=  " << xi[2] << std::endl;
        //     }

        // solve equation
        if (abs(xjm.determinant()) < 1e-15)
        {
          FOUR_C_THROW("*** WARNING: jacobi singular ***");
        }

        double xjm_invert = xjm.invert();
        if (abs(xjm_invert) < 1e-12) FOUR_C_THROW("ERROR: Singular Jacobian for advection map");

        double deltaxi[ndim];
        for (int p = 0; p < ndim; ++p) deltaxi[p] = 0.0;

        for (int z = 0; z < ndim; ++z)
          for (int p = 0; p < ndim; ++p) deltaxi[z] -= xjm(z, p) * rhs[p];

        // incremental update
        for (int p = 0; p < ndim; ++p) xi[p] += deltaxi[p];

        j = j + 1;
      }  // end loop
      //************************************************

      if (!converged) FOUR_C_THROW("Evaluation of element coordinates not converged!");

      return;
    };

    /*----------------------------------------------------------------------*
     |  Get local coords for given global coords (ref position)  farah 01/14|
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void global_to_local(Core::Elements::Element& ele,  // element (input)
        double* xgl,                                    // global position (input)
        double* xi,
        bool& converged)  // converged solution ?
    {
      static constexpr int numnod = Core::FE::num_nodes<distype>;
      static constexpr int ndim = Core::FE::dim<distype>;

      // converged
      converged = false;

      Core::LinAlg::Matrix<numnod, 1> funct;
      Core::LinAlg::Matrix<ndim, numnod> xref;
      Core::LinAlg::Matrix<ndim, ndim> xjm;
      Core::LinAlg::Matrix<ndim, numnod> deriv;

      // spatial configuration of this element!
      for (int k = 0; k < numnod; ++k)
        for (int j = 0; j < ndim; ++j) xref(j, k) = ele.nodes()[k]->x()[j];

      // first estimation for parameter space coordinates
      for (int p = 0; p < ndim; ++p)
      {
        if (distype == Core::FE::CellType::quad4 or distype == Core::FE::CellType::quad8 or
            distype == Core::FE::CellType::quad9 or distype == Core::FE::CellType::hex8 or
            distype == Core::FE::CellType::hex20 or distype == Core::FE::CellType::hex27)
        {
          xi[p] = 0.0;
        }
        else if (distype == Core::FE::CellType::tri3 or distype == Core::FE::CellType::tri6 or
                 distype == Core::FE::CellType::tet4 or distype == Core::FE::CellType::tet10)
        {
          xi[p] = 1.0 / 3.0;
        }
        else if (distype == Core::FE::CellType::pyramid5)
        {
          if (p < 2)
            xi[p] = 0.0;
          else
            xi[p] = 1.0 / 3.0;
        }
        else
        {
          FOUR_C_THROW("ERROR: Element type not supported for parameter space mapping!");
        }
      }
      double rhs[ndim];

      int j = 0;

      //************************************************
      // loop
      while (!converged and j < 10)
      {
        // reset matriced
        xjm.clear();
        deriv.clear();

        if (ndim == 2)
        {
          Core::FE::shape_function_2d(funct, xi[0], xi[1], distype);
          Core::FE::shape_function_2d_deriv1(deriv, xi[0], xi[1], distype);
        }
        else if (ndim == 3)
        {
          Core::FE::shape_function_3d(funct, xi[0], xi[1], xi[2], distype);
          Core::FE::shape_function_3d_deriv1(deriv, xi[0], xi[1], xi[2], distype);
        }
        else
          FOUR_C_THROW("ERROR");

        for (int k = 0; k < numnod; ++k)
          for (int p = 0; p < ndim; ++p)
            for (int l = 0; l < ndim; ++l) xjm(p, l) += deriv(l, k) * xref(p, k);

        // rhs of (linearized equation)
        for (int p = 0; p < ndim; ++p) rhs[p] = 0.0;

        for (int p = 0; p < ndim; ++p) rhs[p] = -xgl[p];

        for (int k = 0; k < numnod; ++k)
          for (int p = 0; p < ndim; ++p) rhs[p] += funct(k) * xref(p, k);

        double norm = 0.0;
        for (int p = 0; p < ndim; ++p) norm += rhs[p] * rhs[p];

        if (sqrt(norm) < 1e-12) converged = true;

        if (converged == true) break;

        // solve equation
        if (abs(xjm.determinant()) < 1e-15)
        {
          std::cout << "WARNING !!! jacobi determinant singular! In global_to_local(...)"
                    << std::endl;
          std::cout << "JAC= " << xjm.determinant() << std::endl;
          std::cout << "CONVERGED= " << converged << std::endl;
          //       FOUR_C_THROW("*** WARNING: jacobi singular ***");
          converged = false;
          break;
        }

        double xjm_invert = xjm.invert();
        if (abs(xjm_invert) < 1e-15) FOUR_C_THROW("ERROR: Singular Jacobian");

        double deltaxi[3];
        for (int p = 0; p < ndim; ++p) deltaxi[p] = 0.0;

        for (int z = 0; z < ndim; ++z)
          for (int p = 0; p < ndim; ++p) deltaxi[z] -= xjm(z, p) * rhs[p];

        // incremental update
        for (int p = 0; p < ndim; ++p) xi[p] += deltaxi[p];

        j = j + 1;
      }  // end loop
      //************************************************

      return;
    };

    /*----------------------------------------------------------------------*
     |  Get local coords for given global coords (curr position) rauch 08/14|
     *----------------------------------------------------------------------*/
    template <Core::FE::CellType distype>
    void global_to_current_local(Core::Elements::Element& ele,  // element (input)
        const double* targetdisp,
        double* xgl,  // global position (input)
        double* xi, bool& converged,
        double& residual)  // converged solution ?
    {
      static constexpr int numnod = Core::FE::num_nodes<distype>;
      static constexpr int ndim = Core::FE::dim<distype>;

      // converged
      converged = false;

      Core::LinAlg::Matrix<numnod, 1> funct;
      Core::LinAlg::Matrix<ndim, numnod> xref;
      Core::LinAlg::Matrix<ndim, ndim> xjm;
      Core::LinAlg::Matrix<ndim, numnod> deriv;

      // spatial configuration of this element!
      for (int k = 0; k < numnod; ++k)
        for (int j = 0; j < ndim; ++j)
          xref(j, k) = (ele.nodes()[k]->x()[j]) + targetdisp[k * ndim + j];

      // first estimation for parameter space coordinates
      for (int p = 0; p < 3; ++p) xi[p] = 0.0;

      double rhs[ndim];

      int j = 0;

      //************************************************
      // loop
      while (!converged and j < 50)
      {
        // reset matrices
        xjm.clear();
        deriv.clear();

        if (ndim == 2)
        {
          Core::FE::shape_function_2d(funct, xi[0], xi[1], distype);
          Core::FE::shape_function_2d_deriv1(deriv, xi[0], xi[1], distype);
        }
        else if (ndim == 3)
        {
          Core::FE::shape_function_3d(funct, xi[0], xi[1], xi[2], distype);
          Core::FE::shape_function_3d_deriv1(deriv, xi[0], xi[1], xi[2], distype);
        }
        else
          FOUR_C_THROW("ERROR");

        for (int k = 0; k < numnod; ++k)
          for (int p = 0; p < ndim; ++p)
            for (int l = 0; l < ndim; ++l) xjm(p, l) += deriv(l, k) * xref(p, k);

        for (int p = 0; p < ndim; ++p) rhs[p] = -xgl[p];

        for (int k = 0; k < numnod; ++k)
          for (int p = 0; p < ndim; ++p) rhs[p] += funct(k) * xref(p, k);

        double norm = 0.0;
        for (int p = 0; p < ndim; ++p) norm += rhs[p] * rhs[p];
        residual = sqrt(norm);
        if (residual < 1e-13) converged = true;

        if (converged == true) break;

        // solve equation
        if (abs(xjm.determinant()) < 1e-15)
        {
          std::cout << "WARNING !!! jacobi determinant singular! In GlobalToCurrentLocal(...)"
                    << std::endl;
          std::cout << "JAC= " << xjm.determinant() << std::endl;
          std::cout << "CONVERGED= " << converged << std::endl;
          //       FOUR_C_THROW("*** WARNING: jacobi singular ***");
          converged = false;
          break;
        }

        double xjm_invert = xjm.invert();
        if (abs(xjm_invert) < 1e-15) FOUR_C_THROW("ERROR: Singular Jacobian");

        double deltaxi[3];
        for (int p = 0; p < ndim; ++p) deltaxi[p] = 0.0;

        for (int z = 0; z < ndim; ++z)
          for (int p = 0; p < ndim; ++p) deltaxi[z] -= xjm(z, p) * rhs[p];

        // incremental update
        for (int p = 0; p < ndim; ++p) xi[p] += deltaxi[p];

        j = j + 1;
      }  // end loop
      //************************************************

      return;
    };

  }  // namespace Utils
}  // namespace Mortar



FOUR_C_NAMESPACE_CLOSE

#endif
