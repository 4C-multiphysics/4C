// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_condition.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_stru_multi_microstatic.hpp"
#include "4C_structure_aux.hpp"

FOUR_C_NAMESPACE_OPEN


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void MultiScale::MicroStatic::determine_toggle()
{
  int np = 0;  // local number of prescribed (=boundary) dofs needed for the
               // creation of vectors and matrices for homogenization
               // procedure

  std::vector<const Core::Conditions::Condition*> conds;
  discret_->get_condition("MicroBoundary", conds);
  for (auto& cond : conds)
  {
    const std::vector<int> nodeids = *cond->get_nodes();

    for (int i : nodeids)
    {
      // do only nodes in my row map
      if (!discret_->node_row_map()->my_gid(i)) continue;
      Core::Nodes::Node* actnode = discret_->g_node(i);
      if (!actnode) FOUR_C_THROW("Cannot find global node {}", i);
      std::vector<int> dofs = discret_->dof(actnode);
      const unsigned numdf = dofs.size();

      for (unsigned j = 0; j < numdf; ++j)
      {
        const int gid = dofs[j];

        const int lid = disn_->get_map().lid(gid);
        if (lid < 0) FOUR_C_THROW("Global id {} not on this proc in system vector", gid);

        if ((*dirichtoggle_).get_values()[lid] != 1.0)  // be careful not to count dofs more
                                                        // than once since nodes belong to
                                                        // several surfaces simultaneously
          ++np;

        (*dirichtoggle_).get_values()[lid] = 1.0;
      }
    }
  }

  np_ = np;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void MultiScale::MicroStatic::set_up_homogenization()
{
  int indp = 0;
  int indf = 0;

  ndof_ = discret_->dof_row_map()->num_my_elements();

  std::vector<int> pdof(np_);
  std::vector<int> fdof(ndof_ - np_);  // changed this, previously this
                                       // has been just fdof(np_),
                                       // but how should that
                                       // work for ndof_-np_>np_???

  for (int it = 0; it < ndof_; ++it)
  {
    if (dirichtoggle_->local_values_as_span()[it] == 1.0)
    {
      pdof[indp] = discret_->dof_row_map()->gid(it);
      ++indp;
    }
    else
    {
      fdof[indf] = discret_->dof_row_map()->gid(it);
      ++indf;
    }
  }

  // create map based on the determined dofs of prescribed and free nodes
  pdof_ = std::make_shared<Core::LinAlg::Map>(-1, np_, pdof.data(), 0, discret_->get_comm());
  fdof_ =
      std::make_shared<Core::LinAlg::Map>(-1, ndof_ - np_, fdof.data(), 0, discret_->get_comm());

  // create importer
  importp_ = std::make_shared<Core::LinAlg::Import>(*pdof_, (*discret_->dof_row_map()));
  importf_ = std::make_shared<Core::LinAlg::Import>(*fdof_, (*discret_->dof_row_map()));

  // create vector containing material coordinates of prescribed nodes
  Core::LinAlg::Vector<double> Xp_temp(*pdof_);

  std::vector<const Core::Conditions::Condition*> conds;
  discret_->get_condition("MicroBoundary", conds);
  for (auto& cond : conds)
  {
    const std::vector<int> nodeids = *cond->get_nodes();

    for (int i : nodeids)
    {
      // do only nodes in my row map
      if (!discret_->node_row_map()->my_gid(i)) continue;
      Core::Nodes::Node* actnode = discret_->g_node(i);
      if (!actnode) FOUR_C_THROW("Cannot find global node {}", i);

      // nodal coordinates
      const auto& x = actnode->x();

      std::vector<int> dofs = discret_->dof(actnode);

      for (int k = 0; k < 3; ++k)
      {
        const int gid = dofs[k];

        const int lid = disn_->get_map().lid(gid);
        if (lid < 0) FOUR_C_THROW("Global id {} not on this proc in system vector", gid);

        for (int l = 0; l < np_; ++l)
        {
          if (pdof[l] == gid) Xp_temp.get_values()[l] = x[k];
        }
      }
    }
  }

  material_coords_boundary_nodes_ = Core::LinAlg::create_vector(*pdof_, true);
  *material_coords_boundary_nodes_ = Xp_temp;

  // now create D and its transpose DT (following Miehe et al., 2002)
  // NOTE: d_matrix_ has the same row GIDs (0-8), but different col IDs on different procs
  // (corresponding to pdof_).
  d_matrix_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*pdof_, 9);

  for (int n = 0; n < np_ / 3; ++n)
  {
    ((*d_matrix_)(0)).get_values()[3 * n] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n];
    ((*d_matrix_)(3)).get_values()[3 * n] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 1];
    ((*d_matrix_)(6)).get_values()[3 * n] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 2];

    ((*d_matrix_)(1)).get_values()[3 * n + 1] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 1];
    ((*d_matrix_)(4)).get_values()[3 * n + 1] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 2];
    ((*d_matrix_)(7)).get_values()[3 * n + 1] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n];

    ((*d_matrix_)(2)).get_values()[3 * n + 2] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 2];
    ((*d_matrix_)(5)).get_values()[3 * n + 2] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n];
    ((*d_matrix_)(8)).get_values()[3 * n + 2] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 1];
  }

  Core::LinAlg::MultiVector<double> d_matrix_transposed(*pdof_, 9);

  for (int n = 0; n < np_ / 3; ++n)
  {
    ((d_matrix_transposed(0))).get_values()[3 * n] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n];
    ((d_matrix_transposed(1))).get_values()[3 * n + 1] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 1];
    ((d_matrix_transposed(2))).get_values()[3 * n + 2] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 2];
    ((d_matrix_transposed(3))).get_values()[3 * n] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 1];
    ((d_matrix_transposed(4))).get_values()[3 * n + 1] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 2];
    ((d_matrix_transposed(5))).get_values()[3 * n + 2] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n];
    ((d_matrix_transposed(6))).get_values()[3 * n] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 2];
    ((d_matrix_transposed(7))).get_values()[3 * n + 1] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n];
    ((d_matrix_transposed(8))).get_values()[3 * n + 2] =
        material_coords_boundary_nodes_->local_values_as_span()[3 * n + 1];
  }

  rhs_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*(discret_->dof_row_map()), 9);

  for (int i = 0; i < 9; ++i)
  {
    ((*rhs_)(i)).export_to((d_matrix_transposed(i)), *importp_, Insert);
  }
}


/*----------------------------------------------------------------------*
 |  check convergence of Newton iteration (public)              lw 12/07|
 *----------------------------------------------------------------------*/
bool MultiScale::MicroStatic::converged()
{
  // check for single norms
  bool convdis = false;
  bool convfres = false;

  // residual displacement
  switch (normtypedisi_)
  {
    case Inpar::Solid::convnorm_abs:
      convdis = normdisi_ < toldisi_;
      break;
    case Inpar::Solid::convnorm_rel:
      convdis = normdisi_ / normchardis_ < toldisi_;
      break;
    case Inpar::Solid::convnorm_mix:
      convdis = ((normdisi_ < toldisi_) or (normdisi_ / normchardis_ < toldisi_));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual displacements!");
      break;
  }

  // residual forces
  switch (normtypefres_)
  {
    case Inpar::Solid::convnorm_abs:
      convfres = normfres_ < tolfres_;
      break;
    case Inpar::Solid::convnorm_rel:
      convfres = normfres_ / normcharforce_ < tolfres_;
      break;
    case Inpar::Solid::convnorm_mix:
      convfres = ((normfres_ < tolfres_) or (normfres_ / normcharforce_ < tolfres_));
      break;
    default:
      FOUR_C_THROW("Cannot check for convergence of residual forces!");
      break;
  }

  // combine displacement-like and force-like residuals
  bool conv = false;
  if (combdisifres_ == Inpar::Solid::bop_and)
    conv = convdis and convfres;
  else if (combdisifres_ == Inpar::Solid::bop_or)
    conv = convdis or convfres;
  else
    FOUR_C_THROW("Something went terribly wrong with binary operator!");

  return conv;
}


/*----------------------------------------------------------------------*
 |  calculate reference norms for relative convergence checks   lw 12/07|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::calc_ref_norms()
{
  // The reference norms are used to scale the calculated iterative
  // displacement norm and/or the residual force norm. For this
  // purpose we only need the right order of magnitude, so we don't
  // mind evaluating the corresponding norms at possibly different
  // points within the timestep (end point, generalized midpoint).
  // In the beginning (construction of macroscale time integrator and
  // first macroscale predictor), macro displacements are generally
  // 0 leading to no load on the micro problem. Consequently, the
  // microscale reference norms are 0 in case of displacements, and
  // near 0 in case of the residual (sum of ndof numerical near zero
  // values). To enable convergence in these cases, the reference norm
  // is automatically set to 1 if the calculated values are below
  // the chosen tolerances. Simply testing against 0 only works for
  // the displacements, but not for the residual!

  normchardis_ = Solid::calculate_vector_norm(iternorm_, *dis_);
  if (normchardis_ < toldisi_) normchardis_ = 1.0;

  double fintnorm = Solid::calculate_vector_norm(iternorm_, *fintn_);
  double freactnorm = Solid::calculate_vector_norm(iternorm_, *freactn_);
  normcharforce_ = std::max(fintnorm, freactnorm);
  if (normcharforce_ < tolfres_) normcharforce_ = 1.0;
}


/*----------------------------------------------------------------------*
 |  print to screen and/or error file                           lw 12/07|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::print_newton(bool print_unconv, Teuchos::Time timer)
{
  bool relres = (normtypefres_ == Inpar::Solid::convnorm_rel);

  bool relres_reldis = ((normtypedisi_ == Inpar::Solid::convnorm_rel) && relres);

  if (relres)
  {
    normfres_ /= normcharforce_;
  }

  if (relres_reldis)
  {
    normfres_ /= normcharforce_;
    normdisi_ /= normchardis_;
  }

  if (print_unconv)
  {
    if (printscreen_ and (stepn_ % printscreen_ == 0))
    {
      if (relres)
      {
        printf("      MICROSCALE numiter %2d scaled res-norm %10.5e absolute dis-norm %20.15E\n",
            numiter_ + 1, normfres_, normdisi_);
        fflush(stdout);
      }
      else if (relres_reldis)
      {
        printf("      MICROSCALE numiter %2d scaled res-norm %10.5e scaled dis-norm %20.15E\n",
            numiter_ + 1, normfres_, normdisi_);
        fflush(stdout);
      }
      else
      {
        printf("      MICROSCALE numiter %2d absolute res-norm %10.5e absolute dis-norm %20.15E\n",
            numiter_ + 1, normfres_, normdisi_);
        fflush(stdout);
      }
    }
  }
  else
  {
    double timepernlnsolve = timer.totalElapsedTime(true);

    if (relres)
    {
      printf(
          "      MICROSCALE Newton iteration converged: numiter %d scaled res-norm %e absolute "
          "dis-norm %e time %10.5f\n\n",
          numiter_, normfres_, normdisi_, timepernlnsolve);
      fflush(stdout);
    }
    else if (relres_reldis)
    {
      printf(
          "      MICROSCALE Newton iteration converged: numiter %d scaled res-norm %e scaled "
          "dis-norm %e time %10.5f\n\n",
          numiter_, normfres_, normdisi_, timepernlnsolve);
      fflush(stdout);
    }
    else
    {
      printf(
          "      MICROSCALE Newton iteration converged: numiter %d absolute res-norm %e absolute "
          "dis-norm %e time %10.5f\n\n",
          numiter_, normfres_, normdisi_, timepernlnsolve);
      fflush(stdout);
    }
  }
}


/*----------------------------------------------------------------------*
 |  print to screen                                             lw 12/07|
 *----------------------------------------------------------------------*/
void MultiScale::MicroStatic::print_predictor()
{
  if (normtypefres_ == Inpar::Solid::convnorm_rel)
  {
    normfres_ /= normcharforce_;
    std::cout << "      MICROSCALE Predictor scaled res-norm " << normfres_ << std::endl;
  }
  else
  {
    std::cout << "      MICROSCALE Predictor absolute res-norm " << normfres_ << std::endl;
  }
  fflush(stdout);
}

FOUR_C_NAMESPACE_CLOSE
