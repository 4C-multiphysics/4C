// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_beam_to_sphere_contact_pair.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beamcontact_input.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_beaminteraction_beam_to_sphere_contact_params.hpp"
#include "4C_beaminteraction_contact_params.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN



/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::BeamToSphereContactPair()
    : beam_element_(nullptr), sphere_element_(nullptr), contactflag_(false)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::setup()
{
  check_init();

  // call setup of base class first
  BeamContactPair::setup();


  ele1pos_.clear();
  ele2pos_.clear();

  fc1_.clear();
  fc2_.clear();

  gap_ = 0.0;

  // initialize class variables for contact point coordinates
  x1_.clear();
  x2_.clear();
  normal_.clear();


  // cast first element to Beam3Base
  beam_element_ = dynamic_cast<const Discret::Elements::Beam3Base*>(element1());

  if (beam_element_ == nullptr)
    FOUR_C_THROW(
        "cast to Beam3Base failed! first element in BeamToSphereContactPair pair"
        " must be a beam element!");

  // get radius and stress-free reference length of beam element
  radius1_ = BeamInteraction::calc_ele_radius(beam_element_);
  beamele_reflength_ = beam_element_->ref_length();

  // cast second element to RigidSphere
  sphere_element_ = dynamic_cast<const Discret::Elements::Rigidsphere*>(element2());

  if (sphere_element_ == nullptr)
    FOUR_C_THROW(
        "cast to Rigidsphere failed! second element in BeamToSphereContactPair pair"
        " must be a Rigidsphere element!");

  // get radius of sphere element
  radius2_ = sphere_element_->radius();


  nodalcontactflag_.assign(2, false);

  issetup_ = true;


  // *********************** DEBUG ************************************************
  //  std::cout << "\nSuccessful Creation&Init&Setup of";
  //  print(std::cout);
  // ******************* END DEBUG ************************************************
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::pre_evaluate()
{
  // do nothing
  return;
}
/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
bool BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::evaluate(
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  unsigned int dim1 = 3 * numnodes * numnodalvalues;
  unsigned int dim2 = 3;

  // resize and initialize variables to zero
  if (forcevec1 != nullptr) forcevec1->size(dim1);
  if (forcevec2 != nullptr) forcevec2->size(dim2);

  if (stiffmat11 != nullptr) stiffmat11->shape(dim1, dim1);
  if (stiffmat12 != nullptr) stiffmat12->shape(dim1, dim2);
  if (stiffmat21 != nullptr) stiffmat21->shape(dim2, dim1);
  if (stiffmat22 != nullptr) stiffmat22->shape(dim2, dim2);

  //**********************************************************************
  // Evaluation of contact forces and stiffness
  //**********************************************************************
  // (1) Closest Point Projection
  // (2) Compute some auxiliary quantities
  //     -> normal vector, gap, shape functions, contact flag,
  //     -> linearizations of all geometric quantities
  // (3) Compute contact forces and stiffness
  //     -> stiffness terms are directly assembled to global matrix
  //     -> contact forces are only returned as global vector
  // (4) Perform some finite difference checks
  //     -> only if the flag BEAMCONTACTFDCHECKS is defined
  // set class variable for status of gapfunction

  // Note: element1 is always the beam, element2 is the sphere

  closest_point_projection();

  // vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i;
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i_xi;
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i_xixi;

  // coords and derivatives of the two contact points
  Core::LinAlg::Matrix<3, 1, TYPE> x1;    // = x1
  Core::LinAlg::Matrix<3, 1, TYPE> x2;    // = x2
  Core::LinAlg::Matrix<3, 1, TYPE> dx1;   // = x1,xi
  Core::LinAlg::Matrix<3, 1, TYPE> ddx1;  // = x1,xixi

  // initialize
  Core::LinAlg::Matrix<3, 1, TYPE> normal;
  TYPE gap = 0.0;
  TYPE norm = 0.0;

  bool validclosestpointprojection = true;

  if (std::abs(Core::FADUtils::cast_to_double(xicontact_)) <
      (1.0 + XIETATOL))  // ToDo when to reset nodalcontactflag_?
  {
    nodalcontactflag_[0] = false;
    nodalcontactflag_[1] = false;
  }
  else
  {
    contactflag_ = false;
    validclosestpointprojection = false;
  }

  if (validclosestpointprojection)
  {
    //**********************************************************************
    // (1) Compute some auxiliary quantities
    //**********************************************************************

    // call function to fill variables for shape functions and their derivatives
    get_shape_functions(N1_i, N1_i_xi, N1_i_xixi, xicontact_);

    // call function to fill variables with coords and derivs of the contact point
    compute_coords_and_derivs(x1, x2, dx1, ddx1, N1_i, N1_i_xi, N1_i_xixi);

    // call function to compute scaled normal and gap in possible contact point
    compute_normal(normal, gap, norm, x1, x2);

    // call function to evaluate contact status
    check_and_set_contact_status();

    //**********************************************************************
    // (2) Compute contact forces and stiffness
    //**********************************************************************

    // call function to evaluate contact forces
    if (forcevec1 != nullptr and forcevec2 != nullptr)
    {
      evaluate_fc_contact(*forcevec1, *forcevec2,
          params()->beam_to_sphere_contact_params()->beam_to_sphere_penalty_param(), gap, normal,
          N1_i, contactflag_);
    }

    // call function to evaluate contact stiffness
    if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
        stiffmat22 != nullptr)
    {
      evaluate_stiffc_contact(*stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22,
          params()->beam_to_sphere_contact_params()->beam_to_sphere_penalty_param(), gap, normal,
          norm, x1, x2, dx1, ddx1, N1_i, N1_i_xi, N1_i_xixi, contactflag_);
    }
  }
  else  // Fixme do this only for the nodes (physical) end points of beams in case of C1-continuous
        // beam centerline representation
  {
    if (numnodes != 2)
      FOUR_C_THROW(
          "BeamToSphereContactPair: Check for nodal contact only implemented for 2-noded beam "
          "elements!");

    //************************************************************************
    // NEW: NODAL-BEAM-TO-SPHERE CONTACT
    //************************************************************************
    // distance of beam nodes and center of sphere is computed
    // -> penalty contribution if distance is smaller than sum of radii
    // (point-to-point contact: xi=+-1, hence no contribution from linearization of xi)
    // this also has consequences for AUTOMATICDIFF: do not use xicontact_ here!
    TYPE XiContact = 0.0;
    for (unsigned int inode = 0; inode < 2; ++inode)
    {
      /* if inode=0 is active contact (gap<0), we do NOT want to run the loop for inode=1
       * because the case validclosestpointprojection=false and nodalcontactflag_[0]=true and
       * nodalcontactflag_[1]=true is highly unlikely and class variables like gap_ are overwritten
       * if we run this loop again this leads to wrong output for gap_ of active BTSPH pairs in
       * Beam3contactmanager */
      if (nodalcontactflag_[0] != true)
      {
        // Set XiContact to +-1.0
        switch (inode)
        {
          case 0:
            XiContact = -1.0;  // first node
            break;
          case 1:
            XiContact = 1.0;  // second node
            break;
          default:
            FOUR_C_THROW("This must not happen!");
            break;
        }

        // Now do exactly the same as for "normal" contact based on closest point projection
        // (except for contribution from linearization of xi)

        // TODO: make more efficient: use knowledge that contact point is a node -> matrix of shape
        // functions, etc. simplifies

        //**********************************************************************
        // (1) Compute some auxiliary quantities
        //**********************************************************************

        // call function to fill variables for shape functions and their derivatives
        get_shape_functions(N1_i, N1_i_xi, N1_i_xixi, XiContact);

        // call function to fill variables with coords and derivs of the contact point
        compute_coords_and_derivs(x1, x2, dx1, ddx1, N1_i, N1_i_xi, N1_i_xixi);

        // call function to compute scaled normal and gap in possible contact point
        compute_normal(normal, gap, norm, x1, x2);

        // evaluate nodal contact status
        if (gap_ < 0)
        {
          nodalcontactflag_[inode] = true;
        }
        else
          nodalcontactflag_[inode] = false;

        //**********************************************************************
        // (2) Compute contact forces and stiffness
        //**********************************************************************

        // call function to evaluate contact forces
        if (forcevec1 != nullptr and forcevec2 != nullptr)
        {
          evaluate_fc_contact(*forcevec1, *forcevec2,
              params()->beam_to_sphere_contact_params()->beam_to_sphere_penalty_param(), gap,
              normal, N1_i, nodalcontactflag_[inode]);
        }

        // call function to evaluate contact stiffness
        if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
            stiffmat22 != nullptr)
        {
          evaluate_stiffc_contact(*stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22,
              params()->beam_to_sphere_contact_params()->beam_to_sphere_penalty_param(), gap,
              normal, norm, x1, x2, dx1, ddx1, N1_i, N1_i_xi, N1_i_xixi, nodalcontactflag_[inode],
              false);
        }
      }
    }
  }

  return get_contact_flag();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::closest_point_projection()
{
  // local variable for beam element coordinate
  TYPE eta = 0.0;

  // vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i;
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i_xi;
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE> N1_i_xixi;

  // coords and derivatives of the two contact points
  Core::LinAlg::Matrix<3, 1, TYPE> x1;       // = x1
  Core::LinAlg::Matrix<3, 1, TYPE> x2;       // = x2
  Core::LinAlg::Matrix<3, 1, TYPE> dx1;      // = x1,xi
  Core::LinAlg::Matrix<3, 1, TYPE> ddx1;     // = x1,xixi
  Core::LinAlg::Matrix<3, 1, TYPE> delta_x;  // = x1 - x2

  // initialize function f and Jacobian df for Newton iteration
  TYPE f;
  TYPE df;

  // initial scalar residual (L2-norm of f)
  double residual = 0.0;

  //**********************************************************************
  // local Newton iteration
  //**********************************************************************
  for (int i = 0; i < BEAMCONTACTMAXITER; ++i)
  {
    // reset shape function variables to zero
    N1_i.clear();
    N1_i_xi.clear();
    N1_i_xixi.clear();

    // update shape functions and their derivatives
    get_shape_functions(N1_i, N1_i_xi, N1_i_xixi, eta);

    // update coordinates and derivatives of contact points
    compute_coords_and_derivs(x1, x2, dx1, ddx1, N1_i, N1_i_xi, N1_i_xixi);

    // compute delta_x = x1-x2
    for (unsigned int j = 0; j < 3; ++j) delta_x(j) = x1(j) - x2(j);

    // compute norm of difference vector to scale the equations
    // (this yields better conditioning)
    // Note: Even if automatic differentiation via FAD is applied, norm_delta_r has to be of type
    // double since this factor is needed for a pure scaling of the nonlinear CCP and has not to be
    // linearized!
    double norm_delta_x = Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm<3>(delta_x));

    // the closer the beams get, the smaller is norm
    // norm is not allowed to be too small, else numerical problems occur
    if (norm_delta_x < NORMTOL)
    {
      FOUR_C_THROW("Contact points x1 and x2 are identical. Choose smaller time step!");
    }

    // evaluate f at current eta
    evaluate_orthogonality_condition(f, delta_x, norm_delta_x, dx1);

    // compute the scalar residuum
    residual = abs(Core::FADUtils::cast_to_double(f));

    // check if Newton iteration has converged
    if (residual < BEAMCONTACTTOL) break;

    // evaluate Jacobian of f at current eta
    evaluate_lin_orthogonality_condition(df, delta_x, norm_delta_x, dx1, ddx1);

    // singular df
    if (abs(df) < COLINEARTOL)
    {
      FOUR_C_THROW("No solution for Closest Point Projection!");
    }
    // regular df (inversion possible)
    else
    {
      // update element coordinates of contact point
      eta += -f / df;
    }
  }
  //**********************************************************************

  // Newton iteration unconverged
  if (residual > BEAMCONTACTTOL)
  {
    eta = 1e+12;
  }

  // store final result and return
  xicontact_ = eta;

#ifdef AUTOMATICDIFF
  // Set xi1_ as (additional) primary variable for automatic differentiation
  // The dependence between the infinitesimal changes delta xi1_ and the
  // the increments of the primary displacement variables delta disp have to be given explicitly,
  // since no explicit relation between the finite quantities xi1_ and disp exists. The latter would
  // have been necessary if the full linearization had to be computed directly with Sacado!!!

  // The 3*numnodes*numnodalvalues+3 primary DoFs are the components of the nodal positions /
  // tangents. The additional degree of freedom (+1) represents the dependency on the beam parameter
  // coordinate xi, which is necessary in beam contact.
  xicontact_.diff(
      (3 * numnodes * numnodalvalues + 3 + 1) - 1, 3 * numnodes * numnodalvalues + 3 + 1);
#endif

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes,
    numnodalvalues>::evaluate_orthogonality_condition(TYPE& f,
    const Core::LinAlg::Matrix<3, 1, TYPE>& delta_x, const double norm_delta_x,
    const Core::LinAlg::Matrix<3, 1, TYPE>& dx1)
{
  // reset f
  f = 0.0;

  // evaluate f
  // see Wriggers, Computational Contact Mechanics, equation (12.5)

  for (unsigned int i = 0; i < 3; i++)
  {
    f += delta_x(i) * dx1(i) / norm_delta_x;
  }

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes,
    numnodalvalues>::evaluate_lin_orthogonality_condition(TYPE& df,
    Core::LinAlg::Matrix<3, 1, TYPE>& delta_x, const double norm_delta_x,
    const Core::LinAlg::Matrix<3, 1, TYPE>& dx1, const Core::LinAlg::Matrix<3, 1, TYPE>& ddx1)

{
  // reset df
  df = 0;

  // evaluate df
  // see Wriggers, Computational Contact Mechanics, equation (12.7)
  for (unsigned int i = 0; i < 3; i++)
  {
    df += (dx1(i) * dx1(i) + delta_x(i) * ddx1(i)) / norm_delta_x;
  }

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes,
    numnodalvalues>::check_and_set_contact_status()
{
  // check contact condition
  contactflag_ = (gap_ < 0) ? true : false;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::evaluate_fc_contact(
    Core::LinAlg::SerialDenseVector& forcevec1, Core::LinAlg::SerialDenseVector& forcevec2,
    const double& pp, const TYPE& gap, const Core::LinAlg::Matrix<3, 1, TYPE>& normal,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i, const bool contactactive)
{
  fc1_.clear();
  fc2_.clear();

  // get dimensions for vectors fc1 and fc2
  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3;

  // flag indicating assembly
  bool DoNotAssemble = false;

  //**********************************************************************
  // evaluate contact forces for active pairs
  //**********************************************************************
  if (contactactive)
  {
    //********************************************************************
    // Compute Fc1 (force acting on first element)
    //********************************************************************
    for (unsigned int i = 0; i < numnodes * numnodalvalues; ++i)
    {
      for (unsigned int j = 0; j < 3; ++j)
      {
        fc1_(3 * i + j) = (-pp * gap) * normal(j) * N1_i(i);
      }
    }

    //********************************************************************
    // Compute Fc2 (force acting on second element)
    //********************************************************************
    for (unsigned int j = 0; j < 3; ++j)
    {
      fc2_(j) = pp * gap * normal(j);
    }
  }
  //**********************************************************************
  // no forces for inactive pairs
  //**********************************************************************
  else
  {
    // set flag to avoid assembly
    DoNotAssemble = true;
  }

  //**********************************************************************
  // assemble contact forces
  //**********************************************************************
  if (!DoNotAssemble)
  {
    for (unsigned int i = 0; i < dim1; ++i) forcevec1(i) += Core::FADUtils::cast_to_double(fc1_(i));

    for (unsigned int i = 0; i < dim2; ++i) forcevec2(i) += Core::FADUtils::cast_to_double(fc2_(i));
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::evaluate_stiffc_contact(
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22,
    const double& pp, const TYPE& gap, const Core::LinAlg::Matrix<3, 1, TYPE>& normal,
    const TYPE& norm, const Core::LinAlg::Matrix<3, 1, TYPE>& x1,
    const Core::LinAlg::Matrix<3, 1, TYPE>& x2, const Core::LinAlg::Matrix<3, 1, TYPE>& dx1,
    const Core::LinAlg::Matrix<3, 1, TYPE>& ddx1,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xi,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xixi, bool activecontact,
    bool linxi)
{
  // get dimensions for vector fc1 and fc2
  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3;

  // temporary matrices for stiffness matrices
  Core::LinAlg::Matrix<dim1, dim1 + dim2, TYPE> stiffc1;
  Core::LinAlg::Matrix<dim2, dim1 + dim2, TYPE> stiffc2;

  // flag indicating assembly
  bool DoNotAssemble = false;

  // initialize stiffness to zero
  for (unsigned int i = 0; i < dim1; i++)
    for (unsigned int j = 0; j < (dim1 + dim2); j++) stiffc1(i, j) = 0.0;
  for (unsigned int i = 0; i < dim2; i++)
    for (unsigned int j = 0; j < (dim1 + dim2); j++) stiffc2(i, j) = 0.0;

  //**********************************************************************
  // evaluate contact stiffness for active pairs
  //**********************************************************************
  // initialize delta_xi here because we need it outside the if-environment for FAD Check
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPE> delta_xi;

  if (activecontact)
  {
    // auxiliary stiffmatrix for part III of linearization to avoid tensor notation
    Core::LinAlg::Matrix<dim1 + dim2, dim1 + dim2, TYPE> stiffc_III;

    // initialize storage for linearizations
    Core::LinAlg::Matrix<3, 1, TYPE> distance;
    TYPE normdist = 0.0;
    Core::LinAlg::Matrix<dim1 + dim2, 1, TYPE> delta_gap;
    Core::LinAlg::Matrix<3, dim1 + dim2, TYPE> delta_x1_minus_x2;
    Core::LinAlg::Matrix<3, dim1 + dim2, TYPE> delta_n;

    //********************************************************************
    // evaluate linearizations and distance
    //********************************************************************
    // linearization of contact point
    // (not needed in case of check for nodal points in contact)
    if (linxi)
      compute_lin_xi(delta_xi, x1, x2, dx1, ddx1, N1_i, N1_i_xi);
    else
    {
      delta_xi.clear();
    }

    // evaluation of distance
    compute_distance(distance, normdist, normal, norm);

    // linearization of gap function which is equal to delta d
    compute_lin_gap(
        delta_gap, delta_xi, x1, x2, dx1, N1_i, normdist, normal, norm, gap, delta_x1_minus_x2);

    // linearization of normal vector
    compute_lin_normal(delta_n, delta_xi, normal, norm, dx1, N1_i);

    //********************************************************************
    // evaluate contact stiffness
    // (1) stiffc1 of first element
    //********************************************************************

    //********************************************************************
    // part I
    //********************************************************************
    Core::LinAlg::Matrix<dim1, 1, TYPE> N1T_normal;
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
      {
        N1T_normal(3 * j + i) += normal(i) * N1_i(j);
      }
    }

    for (unsigned int i = 0; i < dim1; i++)
    {
      for (unsigned int j = 0; j < (dim1 + dim2); j++)
      {
        stiffc1(i, j) = -pp * N1T_normal(i) * delta_gap(j);
      }
    }

    //********************************************************************
    // part II
    //********************************************************************
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
      {
        for (unsigned int k = 0; k < dim1 + dim2; k++)
        {
          stiffc1(3 * j + i, k) -= pp * gap * N1_i(j) * delta_n(i, k);
        }
      }
    }

    //********************************************************************
    // part III
    //********************************************************************
    Core::LinAlg::Matrix<dim1, 1, TYPE> N1xiT_normal;
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
      {
        N1xiT_normal(3 * j + i) += normal(i) * N1_i_xi(j);
      }
    }

    for (unsigned int i = 0; i < dim1; i++)
    {
      for (unsigned int j = 0; j < dim1 + dim2; j++)
      {
        stiffc1(i, j) -= pp * gap * N1xiT_normal(i) * delta_xi(j);
      }
    }

    //********************************************************************
    // evaluate contact stiffness
    // (2) stiffc2 of second element
    //********************************************************************

    //********************************************************************
    // part I
    //********************************************************************

    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < dim1 + dim2; j++)
      {
        stiffc2(i, j) = pp * normal(i) * delta_gap(j);
      }
    }

    //********************************************************************
    // part II
    //********************************************************************
    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int k = 0; k < dim1 + dim2; k++)
      {
        stiffc2(i, k) -= pp * gap * delta_n(i, k);
      }
    }
    //********************************************************************
    // part III
    //********************************************************************

    // There is no third part for spheres, since no shape functions are applied!
  }
  //**********************************************************************
  // no stiffness for inactive pairs
  //**********************************************************************
  else
  {
    // set flag to avoid assembly
    DoNotAssemble = true;
  }

  //**********************************************************************
  // assemble contact stiffness
  //**********************************************************************
  if (not DoNotAssemble)
  {
#ifndef AUTOMATICDIFF
    // change sign of stiffc1 and stiffc2 due to time integration.
    // according to analytical derivation there is no minus sign, but for
    // our time integration methods the negative stiffness must be assembled.

    // Fixme check the minus sign !!!

    for (unsigned int j = 0; j < dim1; j++)
    {
      for (unsigned int i = 0; i < dim1; i++)
        stiffmat11(i, j) += -Core::FADUtils::cast_to_double(stiffc1(i, j));
      for (unsigned int i = 0; i < dim2; i++)
        stiffmat21(i, j) += -Core::FADUtils::cast_to_double(stiffc2(i, j));
    }
    for (unsigned int j = 0; j < dim2; j++)
    {
      for (unsigned int i = 0; i < dim1; i++)
        stiffmat12(i, j) += -Core::FADUtils::cast_to_double(stiffc1(i, dim1 + j));
      for (unsigned int i = 0; i < dim2; i++)
        stiffmat22(i, j) += -Core::FADUtils::cast_to_double(stiffc2(i, dim1 + j));
    }
#else
    FOUR_C_THROW(
        "check implementation of AUTOMATICDIFF for BeamToSphereContactPair before using it!");

    for (unsigned int j = 0; j < dim1 + dim2; j++)
    {
      for (unsigned int i = 0; i < dim1; i++)
        stiffc1_copy(i, j) =
            -Core::FADUtils::cast_to_double(fc1_(i).dx(j) + fc1_(i).dx(dim1 + dim2) * delta_xi(j));
      for (unsigned int i = 0; i < dim2; i++)
        stiffc2_copy(i, j) =
            -Core::FADUtils::cast_to_double(fc2_(i).dx(j) + fc2_(i).dx(dim1 + dim2) * delta_xi(j));
    }

#ifdef FADCHECKS
    FOUR_C_THROW("check implementation of FADCHECKS for BeamToSphereContactPair before using it!");

    std::cout << "BTSPH Contact Pair: " << Element1()->Id() << " / " << Element2()->Id()
              << std::endl;

    std::cout << "stiffc1: " << std::endl;
    for (unsigned int i = 0; i < dim1; i++)
    {
      for (unsigned int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << std::setw(14) << -stiffc1(i, j).val();
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "stiffc1_FAD: " << std::endl;
    for (unsigned int i = 0; i < dim1; i++)
    {
      for (unsigned int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << std::setw(14) << stiffc1_copy(i, j);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "stiffc2: " << std::endl;
    for (unsigned int i = 0; i < dim2; i++)
    {
      for (unsigned int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << std::setw(14) << -stiffc2(i, j).val();
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "stiffc2_FAD: " << std::endl;
    for (unsigned int i = 0; i < dim2; i++)
    {
      for (unsigned int j = 0; j < dim1 + dim2; j++)
      {
        std::cout << std::setw(14) << stiffc2_copy(i, j);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

#endif

#endif  // AUTOMATICDIFF
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_normal(
    Core::LinAlg::Matrix<3, 1, TYPE>& normal, TYPE& gap, TYPE& norm,
    const Core::LinAlg::Matrix<3, 1, TYPE>& x1, const Core::LinAlg::Matrix<3, 1, TYPE>& x2)
{
  // compute non-unit normal
  for (unsigned int i = 0; i < 3; i++) normal(i) = x1(i) - x2(i);

  // compute length of normal
  norm = Core::FADUtils::vector_norm<3>(normal);
  if (norm < NORMTOL) FOUR_C_THROW("ERROR: Normal of length zero! --> change time step!");

  // compute unit normal and store it in class variable
  for (unsigned int i = 0; i < 3; i++)
  {
    normal(i) /= norm;
    normal_(i) = normal(i);
  }

  // evaluate scalar gap function
  compute_gap(gap, norm);

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_gap(
    TYPE& gap, const TYPE& norm)
{
  // compute gap to be returned
  gap = norm - radius1_ - radius2_;
  gap_ = gap;

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_coords_and_derivs(
    Core::LinAlg::Matrix<3, 1, TYPE>& x1, Core::LinAlg::Matrix<3, 1, TYPE>& x2,
    Core::LinAlg::Matrix<3, 1, TYPE>& dx1, Core::LinAlg::Matrix<3, 1, TYPE>& ddx1,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xi,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xixi)
{
  // reset input variables
  x1.clear();
  x2.clear();
  dx1.clear();
  ddx1.clear();

#ifdef AUTOMATICDIFF
  // The 3*numnodes*numnodalvalues+3 primary DoFs are the components of the nodal positions /
  // tangents. The additional degree of freedom (+1) represents the dependency on the parameter
  // coordinates xi, which is necessary in beam contact.
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; i++)
    ele1pos_(i).diff(i, 3 * numnodes * numnodalvalues + 3 + 1);

  for (unsigned int i = 0; i < 3; i++)
    ele2pos_(i).diff(3 * numnodes * numnodalvalues + i, 3 * numnodes * numnodalvalues + 3 + 1);
#endif  // AUTOMATICDIFF

  // compute output variables
  for (unsigned int i = 0; i < 3; i++) x2(i) = ele2pos_(i);

  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
    {
      x1(i) += N1_i(j) * ele1pos_(3 * j + i);         // x1 = N1 * x~1
      dx1(i) += N1_i_xi(j) * ele1pos_(3 * j + i);     // dx1 = N1,xi * x~1
      ddx1(i) += N1_i_xixi(j) * ele1pos_(3 * j + i);  // ddx1 = N1,xixi * x~1
    }
  }

  // store coordinates of contact point into class variables
  for (unsigned int i = 0; i < 3; i++)
  {
    x1_(i) = x1(i);
    x2_(i) = x2(i);
  }

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::get_shape_functions(
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xi,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xixi, const TYPE& eta)
{
  // get both discretization types
  const Core::FE::CellType distype1 = beam_element()->shape();

  if (numnodalvalues == 1)
  {
    // get values and derivatives of shape functions
    Core::FE::shape_function_1d(N1_i, eta, distype1);
    Core::FE::shape_function_1d_deriv1(N1_i_xi, eta, distype1);
    Core::FE::shape_function_1d_deriv2(N1_i_xixi, eta, distype1);
  }
  else if (numnodalvalues == 2)
  {
    /* TODO hard set distype to line2 in case of numnodalvalues_=2 because
     *  only 3rd order Hermite interpolation is used (always 2 nodes) */
    const Core::FE::CellType distype1herm = Core::FE::CellType::line2;

    // get values and derivatives of shape functions
    Core::FE::shape_function_hermite_1d(N1_i, eta, beamele_reflength_, distype1herm);
    Core::FE::shape_function_hermite_1d_deriv1(N1_i_xi, eta, beamele_reflength_, distype1herm);
    Core::FE::shape_function_hermite_1d_deriv2(N1_i_xixi, eta, beamele_reflength_, distype1herm);
  }
  else
    FOUR_C_THROW(
        "Only beam elements with one (nodal positions) or two (nodal positions + nodal tangents) "
        "values are valid!");

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_lin_xi(
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3, 1, TYPE>& delta_xi,
    const Core::LinAlg::Matrix<3, 1, TYPE>& x1, const Core::LinAlg::Matrix<3, 1, TYPE>& x2,
    const Core::LinAlg::Matrix<3, 1, TYPE>& dx1, const Core::LinAlg::Matrix<3, 1, TYPE>& ddx1,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i_xi)
{
  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3;

  // matrices to compute Lin_Xi (in case of beam-sphere contact: scalar and vector)
  TYPE L = 0.0;
  Core::LinAlg::Matrix<1, dim1 + dim2, TYPE> B;

  Core::LinAlg::Matrix<3, 1, TYPE> delta_x;

  // compute L
  for (unsigned int i = 0; i < 3; i++)
  {
    delta_x(i) = x1(i) - x2(i);
    L += dx1(i) * dx1(i) + delta_x(i) * ddx1(i);
  }

  if (L == 0) FOUR_C_THROW("ERROR: L = 0");


  for (unsigned int i = 0; i < 3; ++i)
  {
    for (unsigned int j = 0; j < numnodes * numnodalvalues; ++j)
    {
      B(3 * j + i) += -delta_x(i) * N1_i_xi(j) - dx1(i) * N1_i(j);
    }
  }

  for (unsigned int i = 0; i < dim2; ++i) B(dim1 + i) += dx1(i);

  // finally the linearizations / directional derivatives
  for (unsigned int i = 0; i < dim1 + dim2; i++)
  {
    delta_xi(i) = B(i) / L;
  }

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_distance(
    Core::LinAlg::Matrix<3, 1, TYPE>& distance, TYPE& normdist,
    const Core::LinAlg::Matrix<3, 1, TYPE>& normal, const TYPE& norm)
{
  // compute distance vector
  for (unsigned int i = 0; i < 3; i++) distance(i) = normal(i) * norm;

  // compute scalar distance
  normdist = norm;

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_lin_gap(
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3, 1, TYPE>& delta_gap,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3, 1, TYPE>& delta_xi,
    const Core::LinAlg::Matrix<3, 1, TYPE>& x1, const Core::LinAlg::Matrix<3, 1, TYPE>& x2,
    const Core::LinAlg::Matrix<3, 1, TYPE>& dx1,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i, const TYPE& normdist,
    const Core::LinAlg::Matrix<3, 1, TYPE>& normal, const TYPE& norm, const TYPE& gap,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3, TYPE>& delta_x1_minus_x2)
{
  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3;


  // delta g := delta_r/||delta_r||*auxiliary_matri1 delta d, with auxiliary_matri1 =
  // (r1_xi*delta_xi-r2_xi*delta_eta + (N1, -N2))

  Core::LinAlg::Matrix<3, dim1 + dim2, TYPE> auxiliary_matrix1(Core::LinAlg::Initialization::zero);

  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < dim1 + dim2; j++)
    {
      auxiliary_matrix1(i, j) += dx1(i) * delta_xi(j);
    }
  }

  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
    {
      auxiliary_matrix1(i, 3 * j + i) += N1_i(j);
    }
  }

  for (unsigned int i = 0; i < 3; i++)
  {
    auxiliary_matrix1(i, i + dim1) += -1.0;
  }

  // compute linearization of gap
  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < dim1 + dim2; j++)
    {
      delta_gap(j) += (x1(i) - x2(i)) * auxiliary_matrix1(i, j) / normdist;
    }
  }

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::compute_lin_normal(
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3, TYPE>& delta_normal,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3, 1, TYPE>& delta_xi,
    const Core::LinAlg::Matrix<3, 1, TYPE>& normal, const TYPE& norm_delta_x,
    const Core::LinAlg::Matrix<3, 1, TYPE>& x1_xi,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPE>& N1_i)
{
  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3;

  // delta n := auxiliary_matri2*auxiliary_matrix1* delta d, with auxiliary_matri2 =
  // (I-nxn)/||r1-r2|| and auxiliary_matri1 = (r1_xi*delta_xi-r2_xi*delta_eta + (N1, -N2))

  Core::LinAlg::Matrix<3, dim1 + dim2, TYPE> auxiliary_matrix1(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, TYPE> auxiliary_matrix2(Core::LinAlg::Initialization::zero);

  // compute auxiliary_matrix1
  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < dim1 + dim2; j++)
    {
      auxiliary_matrix1(i, j) += x1_xi(i) * delta_xi(j);
    }
  }

  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
    {
      auxiliary_matrix1(i, 3 * j + i) += N1_i(j);
    }
  }

  for (unsigned int i = 0; i < 3; i++)
  {
    auxiliary_matrix1(i, dim1 + i) += -1.0;
  }

  // compute auxiliary_matrix2
  for (unsigned int i = 0; i < 3; i++)
  {
    auxiliary_matrix2(i, i) += 1.0 / norm_delta_x;
    for (unsigned int j = 0; j < 3; j++)
    {
      auxiliary_matrix2(i, j) += -normal(i) * normal(j) / norm_delta_x;
    }
  }

  // compute linearization of normal vector
  for (unsigned int i = 0; i < 3; i++)
    for (unsigned int j = 0; j < 3; j++)
      for (unsigned int k = 0; k < dim1 + dim2; k++)
        delta_normal(i, k) += auxiliary_matrix2(i, j) * auxiliary_matrix1(j, k);

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::reset_state(
    const std::vector<double>& centerline_dofvec_ele1,
    const std::vector<double>& centerline_dofvec_ele2)
{
  if (centerline_dofvec_ele1.size() != 3 * numnodes * numnodalvalues)
    FOUR_C_THROW("size mismatch! expected {} values for centerline_dofvec_ele1, but got {}",
        3 * numnodes * numnodalvalues, centerline_dofvec_ele1.size());

  if (centerline_dofvec_ele2.size() != 3)
    FOUR_C_THROW("size mismatch! expected {} values for centerline_dofvec_ele2, but got {}", 3,
        centerline_dofvec_ele1.size());

  for (unsigned int i = 0; i < 3 * numnodalvalues * numnodalvalues; ++i)
    ele1pos_(i) = centerline_dofvec_ele1[i];

  for (unsigned int i = 0; i < 3; ++i) ele2pos_(i) = centerline_dofvec_ele2[i];
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes, numnodalvalues>::print(
    std::ostream& out) const
{
  // ToDo add further information here
  out << "\nInstance of BeamToSphereContactPair: element GIDs " << element1()->id()
      << " (beam) and " << element2()->id() << " (sphere)";
  out << "\ncontactflag_: " << contactflag_ << "\tnodalcontactflag: " << nodalcontactflag_[0]
      << nodalcontactflag_[1];
  out << "\ngap_: " << gap_;
  out << "\nxicontact_: " << xicontact_;
  out << "\nnormal_: " << normal_;
  out << "\nx1_: " << x1_;
  out << "\nx2_: " << x2_;
  out << "\nele1pos_: " << ele1pos_;
  out << "\nele2pos_: " << ele2pos_;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSphereContactPair<numnodes,
    numnodalvalues>::print_summary_one_line_per_active_segment_pair(std::ostream& out) const
{
  check_init_setup();

  if (get_contact_flag())
  {
    out << "    " << std::setw(5) << std::left << element1()->id() << "(" << std::setw(3)
        << std::right << -1 << "/" << std::setw(3) << -1 << ")"
        << " " << std::setw(5) << std::left << element2()->id() << "(" << std::setw(3) << std::right
        << -1 << "/" << std::setw(3) << -1 << ")"
        << "  BTSPH ";

    out << std::setw(9) << std::left << std::setprecision(2) << xicontact_ << std::setw(9)
        << std::left << std::setprecision(2) << -1 << std::setw(9) << std::left
        << std::setprecision(3) << -1 << std::setw(12) << std::left << std::scientific << gap_
        << std::setw(12) << std::left << std::scientific
        << Core::FADUtils::cast_to_double<TYPE, 3, 1>(fc2_).norm2() << std::setprecision(6)
        << std::resetiosflags(std::ios::scientific) << std::right;

    out << "\n";
  }
}

// explicit template instantiations
template class BeamInteraction::BeamToSphereContactPair<2, 1>;
template class BeamInteraction::BeamToSphereContactPair<3, 1>;
template class BeamInteraction::BeamToSphereContactPair<4, 1>;
template class BeamInteraction::BeamToSphereContactPair<5, 1>;
template class BeamInteraction::BeamToSphereContactPair<2, 2>;

FOUR_C_NAMESPACE_CLOSE
