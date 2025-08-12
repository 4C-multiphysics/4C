// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_nurbs_discretization_initial_condition.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linear_solver_method_linalg.hpp"
#include "4C_utils_function.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  /*----------------------------------------------------------------------*/
  /*!
  \brief A service method allowing the application of initial conditions
         for nurbs discretisations.

  This method provides the following:

  Given an initial distribution u_0(x) of initial values (for example by a
  spatial function), we compute control point values u_cp such that they
  minimize the following least-squares problem:

                        ||                                   || 2
                    || +----+                            ||
                    ||  \                                ||
               min  ||   +    N   (x) * u     -  u   (x) ||
                    ||  /      cp       - cp     - 0     ||
               u    || +----+                            ||
               - cp ||   cp                              ||


  This is equivalent to the solution of the following linear system:


           /                         \               /                         \
          |    /                      |             |    /                      |
   +----+ |   |                       |             |   |                       |
    \     |   |                       |    dim      |   |            dim        |
     +    |   | N   (x) * N   (x) dx  | * u     =   |   | N   (x) * u   (x) dx  |
    /     |   |  cp        cp         |    cp       |   |  cp        0          |
   +----+ |   |    j         i        |      j      |   |    j                  |
     cp   |  /                        |             |  /                        |
       j   \                         /               \                         /

          |                           |             |                           |
          +---------------------------+             +---------------------------+

                 M(assmatrix)                                   rhs



  \param dis         (i) the discretisation
  \param solver      (i) a solver object for the least-squares problem
  \param startfuncno (i) the number of the startfunction defining
                         the initial field (i.e. u_0(x))
  \param initialvals (o) the initial field on output (i.e. u_cp)


  */

  void apply_nurbs_initial_condition_solve(Core::FE::Discretization& dis,
      Core::LinAlg::Solver& solver, const Core::Utils::FunctionOfSpaceTime& start_function,
      std::shared_ptr<Core::LinAlg::Vector<double>> initialvals)
  {
    // try to cast dis to a nurbs discretisation --- if possible, proceed
    // with setting initial conditions. Otherwise return.
    Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
        dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&dis);

    if (nurbsdis == nullptr)
    {
      return;
    }

    // get the knotvector from nurbs discretisation
    std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = nurbsdis->get_knot_vector();

    // get the processor ID from the communicator
    const int myrank = Core::Communication::my_mpi_rank(dis.get_comm());

    if (myrank == 0)
    {
      printf("\n");
      printf("Setting up least-squares Nurbs approximation of initial field (discretization %s)\n",
          dis.name().c_str());
    }

    // -------------------------------------------------------------------
    // get a vector layout from the discretization to construct matching
    // vectors and matrices
    //                 local <-> global dof numbering
    // -------------------------------------------------------------------
    const Core::LinAlg::Map* dofrowmap = dis.dof_row_map();

    // -------------------------------------------------------------------
    // create empty mass matrix
    // -------------------------------------------------------------------
    std::shared_ptr<Core::LinAlg::SparseMatrix> massmatrix =
        std::make_shared<Core::LinAlg::SparseMatrix>(*dofrowmap, 108, false, true);

    // -------------------------------------------------------------------
    // create empty right hand side vector
    // -------------------------------------------------------------------
    std::shared_ptr<Core::LinAlg::Vector<double>> rhs =
        Core::LinAlg::create_vector(*dofrowmap, true);

    // -------------------------------------------------------------------
    // call elements to calculate massmatrix and righthandside
    // -------------------------------------------------------------------
    {
      // call elements and assemble
      if (!nurbsdis->filled()) FOUR_C_THROW("fill_complete() was not called");
      if (!nurbsdis->have_dofs()) FOUR_C_THROW("assign_degrees_of_freedom() was not called");

      // see what we have for input
      bool assemblemat = massmatrix != nullptr;
      bool assemblevec = rhs != nullptr;

      // define element matrices and vectors
      Core::LinAlg::SerialDenseMatrix elemass;
      Core::LinAlg::SerialDenseVector elerhs;

      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;

      // loop over column elements
      const int numcolele = nurbsdis->num_my_col_elements();

      int every = numcolele / 58;
      // prevent division by zero when dividing by every later on
      if (every < 1) every = 1;

      for (int i = 0; i < numcolele; ++i)
      {
        // first 4C progress bar
        if (myrank == 0 && i % every == 0)
        {
          printf(".");
          fflush(nullptr);
        }

        Core::Elements::Element* actele = nurbsdis->l_col_element(i);

        // get element location vector, dirichlet flags and ownerships
        lm.clear();
        lmowner.clear();
        actele->location_vector(*nurbsdis, lm, lmowner, lmstride);

        // get dimension of element matrices and vectors
        // Reshape element matrices and vectors and init to zero
        const int eledim = (int)lm.size();

        if (assemblemat)
        {
          if (elemass.numRows() != eledim or elemass.numCols() != eledim)
            elemass.shape(eledim, eledim);
          else
            elemass.putScalar(0.0);
        }
        if (assemblevec)
        {
          if (elerhs.length() != eledim)
            elerhs.size(eledim);
          else
            elerhs.putScalar(0.0);
        }

        {
          int spacedim = -1;

          const Core::FE::CellType distype = actele->shape();
          switch (distype)
          {
            case Core::FE::CellType::nurbs4:
            case Core::FE::CellType::nurbs9:
            {
              spacedim = 2;
              break;
            }
            case Core::FE::CellType::nurbs8:
            case Core::FE::CellType::nurbs27:
            {
              spacedim = 3;
              break;
            }
            default:
              FOUR_C_THROW("this method is designed for usage with NurbsDiscretization only");
              break;
          }

          // set element data
          const int iel = actele->num_node();

          // dofblocks (spacedim or spacedim+1 for fluid problems)
          const int dofblock = eledim / iel;

          // get node coordinates of element
          Core::LinAlg::SerialDenseMatrix xyze(spacedim, iel);
          Core::Nodes::Node** nodes = actele->nodes();
          for (int inode = 0; inode < iel; inode++)
          {
            const auto& x = nodes[inode]->x();
            for (int dim = 0; dim < spacedim; ++dim)
            {
              xyze(dim, inode) = x[dim];
            }
          }

          // acquire weights from nodes
          Core::LinAlg::SerialDenseVector weights(iel);

          for (int inode = 0; inode < iel; ++inode)
          {
            Core::FE::Nurbs::ControlPoint* cp =
                dynamic_cast<Core::FE::Nurbs::ControlPoint*>(nodes[inode]);

            weights(inode) = cp->w();
          }

          // access elements knot span
          std::vector<Core::LinAlg::SerialDenseVector> eleknots(spacedim);

          bool zero_size = false;
          zero_size = knots->get_ele_knots(eleknots, actele->id());

          // nothing to be done for a zero sized element
          if (zero_size)
          {
            continue;
          }

          Core::LinAlg::SerialDenseVector funct(iel);
          Core::LinAlg::SerialDenseMatrix xjm(spacedim, spacedim);
          Core::LinAlg::SerialDenseMatrix deriv(spacedim, iel);
          Core::LinAlg::SerialDenseVector gp(spacedim);
          Core::LinAlg::SerialDenseVector position(
              3);  // always three-dimensional coordinates for function evaluation!
          Core::LinAlg::SerialDenseVector initialval(dofblock);

          // depending on the spatial dimension, we need a different
          // integration scheme
          switch (spacedim)
          {
            case 2:
            {
              // gaussian points
              const Core::FE::IntegrationPoints2D intpoints(Core::FE::GaussRule2D::quad_9point);

              for (int iquad = 0; iquad < intpoints.nquad; ++iquad)
              {
                // set gauss point coordinates
                for (int rr = 0; rr < spacedim; ++rr)
                {
                  gp(rr) = intpoints.qxg[iquad][rr];
                }

                Core::FE::Nurbs::nurbs_get_2d_funct_deriv(
                    funct, deriv, gp, eleknots, weights, distype);

                // get transposed Jacobian matrix and determinant
                //
                //        +-            -+ T      +-            -+
                //        | dx   dx   dx |        | dx   dy   dz |
                //        | --   --   -- |        | --   --   -- |
                //        | dr   ds   dt |        | dr   dr   dr |
                //        |              |        |              |
                //        | dy   dy   dy |        | dx   dy   dz |
                //        | --   --   -- |   =    | --   --   -- |
                //        | dr   ds   dt |        | ds   ds   ds |
                //        |              |        |              |
                //        | dz   dz   dz |        | dx   dy   dz |
                //        | --   --   -- |        | --   --   -- |
                //        | dr   ds   dt |        | dt   dt   dt |
                //        +-            -+        +-            -+
                //
                // The Jacobian is computed using the formula
                //
                //            +-----
                //   dx_j(r)   \      dN_k(r)
                //   -------  = +     ------- * (x_j)_k
                //    dr_i     /       dr_i       |
                //            +-----    |         |
                //            node k    |         |
                //                  derivative    |
                //                   of shape     |
                //                   function     |
                //                           component of
                //                          node coordinate
                //
                for (int rr = 0; rr < spacedim; ++rr)
                {
                  for (int mm = 0; mm < spacedim; ++mm)
                  {
                    xjm(rr, mm) = deriv(rr, 0) * xyze(mm, 0);
                    for (int nn = 1; nn < iel; ++nn)
                    {
                      xjm(rr, mm) += deriv(rr, nn) * xyze(mm, nn);
                    }
                  }
                }

                // The determinant is computed using Sarrus's rule
                const double det = xjm(0, 0) * xjm(1, 1) - xjm(0, 1) * xjm(1, 0);

                // get real physical coordinates of integration point
                /*
                //              +-----
                //               \
                //    pos (x) =   +      N (x) * x
                //               /        j       j
                //              +-----
                //              node j
                */
                for (int rr = 0; rr < spacedim; ++rr)
                {
                  position(rr) = funct(0) * xyze(rr, 0);
                  for (int mm = 1; mm < iel; ++mm)
                  {
                    position(rr) += funct(mm) * xyze(rr, mm);
                  }
                }
                // if spacedim < 3, ensure we define a valid z-coordinate!
                for (int rr = spacedim; rr < 3; ++rr) position(rr) = 0.0;

                for (int rr = 0; rr < dofblock; ++rr)
                {
                  // important: position has to have always three components!!
                  initialval(rr) = start_function.evaluate(position.values(), 0.0, rr);
                }


                // check for degenerated elements
                if (det < 1E-16)
                {
                  FOUR_C_THROW("GLOBAL ELEMENT NO.{}\nZERO OR NEGATIVE JACOBIAN DETERMINANT: {}",
                      actele->id(), det);
                }

                // set total integration factor
                double fac = intpoints.qwgt[iquad] * det;

                for (int vi = 0; vi < iel; ++vi)  // loop rows  (test functions)
                {
                  const int fvi = dofblock * vi;

                  for (int ui = 0; ui < iel; ++ui)  // loop columns  (test functions)
                  {
                    const int fui = dofblock * ui;

                    const double diag = fac * funct(ui) * funct(vi);

                    for (int rr = 0; rr < dofblock; ++rr)
                    {
                      elemass(fvi + rr, fui + rr) += diag;
                    }
                  }
                  for (int rr = 0; rr < dofblock; ++rr)
                  {
                    elerhs(fvi + rr) += fac * funct(vi) * initialval(rr);
                  }
                }
              }  // end gaussloop
              break;
            }
            case 3:
            {
              // gaussian points
              const Core::FE::IntegrationPoints3D intpoints(Core::FE::GaussRule3D::hex_27point);

              for (int iquad = 0; iquad < intpoints.nquad; ++iquad)
              {
                // set gauss point coordinates
                for (int rr = 0; rr < spacedim; ++rr)
                {
                  gp(rr) = intpoints.qxg[iquad][rr];
                }

                Core::FE::Nurbs::nurbs_get_3d_funct_deriv(
                    funct, deriv, gp, eleknots, weights, distype);

                // get transposed Jacobian matrix and determinant
                //
                //        +-            -+ T      +-            -+
                //        | dx   dx   dx |        | dx   dy   dz |
                //        | --   --   -- |        | --   --   -- |
                //        | dr   ds   dt |        | dr   dr   dr |
                //        |              |        |              |
                //        | dy   dy   dy |        | dx   dy   dz |
                //        | --   --   -- |   =    | --   --   -- |
                //        | dr   ds   dt |        | ds   ds   ds |
                //        |              |        |              |
                //        | dz   dz   dz |        | dx   dy   dz |
                //        | --   --   -- |        | --   --   -- |
                //        | dr   ds   dt |        | dt   dt   dt |
                //        +-            -+        +-            -+
                //
                // The Jacobian is computed using the formula
                //
                //            +-----
                //   dx_j(r)   \      dN_k(r)
                //   -------  = +     ------- * (x_j)_k
                //    dr_i     /       dr_i       |
                //            +-----    |         |
                //            node k    |         |
                //                  derivative    |
                //                   of shape     |
                //                   function     |
                //                           component of
                //                          node coordinate
                //
                for (int rr = 0; rr < spacedim; ++rr)
                {
                  for (int mm = 0; mm < spacedim; ++mm)
                  {
                    xjm(rr, mm) = deriv(rr, 0) * xyze(mm, 0);
                    for (int nn = 1; nn < iel; ++nn)
                    {
                      xjm(rr, mm) += deriv(rr, nn) * xyze(mm, nn);
                    }
                  }
                }

                // The determinant is computed using Sarrus's rule
                const double det =
                    xjm(0, 0) * xjm(1, 1) * xjm(2, 2) + xjm(2, 0) * xjm(0, 1) * xjm(1, 2) +
                    xjm(0, 2) * (xjm(1, 0) * xjm(2, 1) - xjm(2, 0) * xjm(1, 1)) -
                    xjm(2, 1) * xjm(0, 0) * xjm(1, 2) - xjm(0, 1) * xjm(1, 0) * xjm(2, 2);


                // get real physical coordinates of integration point
                /*
                //              +-----
                //               \
                //    pos (x) =   +      N (x) * x
                //               /        j       j
                //              +-----
                //              node j
                */
                for (int rr = 0; rr < spacedim; ++rr)
                {
                  position(rr) = funct(0) * xyze(rr, 0);
                  for (int mm = 1; mm < iel; ++mm)
                  {
                    position(rr) += funct(mm) * xyze(rr, mm);
                  }
                }
                // if spacedim < 3, ensure we define a valid z-coordinate!
                for (int rr = spacedim; rr < 3; ++rr) position(rr) = 0.0;

                for (int rr = 0; rr < dofblock; ++rr)
                {
                  // important: position has to have always three components!!
                  initialval(rr) = start_function.evaluate(position.values(), 0.0, rr);
                }

                // check for degenerated elements
                if (det < 0.0)
                {
                  FOUR_C_THROW(
                      "GLOBAL ELEMENT NO.{}\nNEGATIVE JACOBIAN DETERMINANT: {}", actele->id(), det);
                }

                // set total integration factor
                double fac = intpoints.qwgt[iquad] * det;

                for (int vi = 0; vi < iel; ++vi)  // loop rows  (test functions)
                {
                  const int fvi = dofblock * vi;

                  for (int ui = 0; ui < iel; ++ui)  // loop columns  (test functions)
                  {
                    const int fui = dofblock * ui;

                    const double diag = fac * funct(ui) * funct(vi);

                    for (int rr = 0; rr < dofblock; ++rr)
                    {
                      elemass(fvi + rr, fui + rr) += diag;
                    }
                  }
                  for (int rr = 0; rr < dofblock; ++rr)
                  {
                    elerhs(fvi + rr) += fac * funct(vi) * initialval(rr);
                  }
                }
              }  // end gaussloop
              break;
            }
            default:
              FOUR_C_THROW(
                  "expecting two or three-dimensional problems to set the initial conditions\n");
              break;
          }
        }

        int eid = actele->id();
        if (assemblemat) massmatrix->assemble(eid, elemass, lm, lmowner);
        if (assemblevec) Core::LinAlg::assemble(*rhs, elerhs, lm, lmowner);
      }  // for (int i=0; i<numcolele; ++i)
    }

    if (myrank == 0)
    {
      printf("\n");
      printf("\n");
      printf("Solving least-squares problem: ");
    }

    // -------------------------------------------------------------------
    // finalize the system matrix
    // -------------------------------------------------------------------
    massmatrix->complete();

    // -------------------------------------------------------------------
    // solve system
    // -------------------------------------------------------------------

    // always refactor and reset the matrix before a single new solver call

    initialvals->put_scalar(0.0);
    Core::LinAlg::SolverParams solver_params;
    solver_params.refactor = true;
    solver_params.reset = true;
    solver.solve(massmatrix, initialvals, rhs, solver_params);

    // perform resets for solver and matrix
    solver.reset();
    massmatrix->reset();

    if (myrank == 0)
    {
      printf("Initial field set.\n\n");
    }

    return;
  }
}  // namespace

/*----------------------------------------------------------------------*/
/*!
   A service method allowing the application of initial conditions
   for nurbs discretisations. recommended version with allocation
   of a separate solver!
*/
/*----------------------------------------------------------------------*/
void Core::FE::Nurbs::apply_nurbs_initial_condition(Core::FE::Discretization& dis,
    const Teuchos::ParameterList& solverparams,
    const Core::Utils::FunctionOfSpaceTime& start_function,
    std::shared_ptr<Core::LinAlg::Vector<double>> initialvals)
{
  // try to cast dis to a nurbs discretisation --- if possible, proceed
  // with setting initial conditions. Otherwise return.
  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&dis);

  if (nurbsdis == nullptr)
  {
    return;
  }

  Core::LinAlg::Solver lssolver(
      solverparams, dis.get_comm(), nullptr, Core::IO::Verbositylevel::standard);
  dis.compute_null_space_if_necessary(lssolver.params());

  apply_nurbs_initial_condition_solve(dis, lssolver, start_function, initialvals);
}

FOUR_C_NAMESPACE_CLOSE
