// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beamcontact_beam3tosolidcontact.hpp"

#include "4C_beam3_euler_bernoulli.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_tangentsmoothing.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_comm_exporter.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beamcontact.hpp"
#include "4C_inpar_contact.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | Constructor (public)                                     meier 01/14 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::Beam3tosolidcontact(
    const Core::FE::Discretization& pdiscret, const Core::FE::Discretization& cdiscret,
    const std::map<int, int>& dofoffsetmap, Core::Elements::Element* element1,
    Core::Elements::Element* element2, Teuchos::ParameterList beamcontactparams)
    : pdiscret_(pdiscret),
      cdiscret_(cdiscret),
      dofoffsetmap_(dofoffsetmap),
      element1_(element1),
      element2_(element2),
      sgn_(1.0),
      firstcall_(true),
      gap_(0.0),
      gap_original_(0.0),
      contactflag_(false),
      elementscolinear_(false),
      elementscrossing_(false),
      shiftnodalvalues_(false),
      xi1_(0.0),
      xi2_(0.0),
      gmsh_debug_points_()
{
  for (int i = 0; i < 3 * numnodes * numnodalvalues; i++) ele1pos_(i) = 0.0;

  for (int i = 0; i < 3 * numnodessol; i++) ele2pos_(i) = 0.0;

  normalsets_.clear();
  normalsets_.resize(0);

  normalsets_old_.clear();
  normalsets_old_.resize(0);

  return;
}
/*----------------------------------------------------------------------*
 | End: constructor                                                     |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | copy-constructor (public)                                            |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::Beam3tosolidcontact(
    const Beam3tosolidcontact& old)
    : pdiscret_(old.pdiscret_), cdiscret_(old.cdiscret_), dofoffsetmap_(old.dofoffsetmap_)
{
  FOUR_C_THROW("ERROR: Copy constructor incomplete");
  return;
}
/*----------------------------------------------------------------------*
 | End: copy-constructor                                                |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Evaluate the element (public)                                        |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
bool CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::evaluate(
    Core::LinAlg::SparseMatrix& stiffmatrix, Core::LinAlg::Vector<double>& fint, const double& pp)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

#if defined(GMSHDEBUG) || defined(SAVEFORCE) || defined(GMSHFORCE)
  // Clear vector for Gmsh debug
  gmshDebugPoints_.clear();
  gmshDebugPoints_.resize(0);
#endif


  // Find contact interval borders
  // -----------------------------------------------------------------

  // Vector for contact interval border parameter sets (xi1, xi2, eta and index of fixed paramater)
  std::vector<std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>> parsets;

  // Find contact interval borders
  get_contact_interval_borders(parsets);

  // -----------------------------------------------------------------
  // End: Find contact interval borders


  // Compute contact forces and stiffness for all contact intervals
  // -----------------------------------------------------------------

  // Temporary vectors for contact forces
  Core::LinAlg::Matrix<dim1, 1, TYPEBTS> fc1(true);
  Core::LinAlg::Matrix<dim2, 1, TYPEBTS> fc2(true);

  // Temporary matrices for contact stiffness
  Core::LinAlg::Matrix<dim1, dim1 + dim2, TYPEBTS> stiffc1(true);
  Core::LinAlg::Matrix<dim2, dim1 + dim2, TYPEBTS> stiffc2(true);

  // Temporary matrices for contact stiffness calculated via FAD
  // TODO: Declare stiffc1_FAD and stiffc2_FAD only if needed (#ifdef AUTOMATICDIFFBTS)
  Core::LinAlg::Matrix<dim1, dim1 + dim2, TYPEBTS> stiffc1_FAD(true);
  Core::LinAlg::Matrix<dim2, dim1 + dim2, TYPEBTS> stiffc2_FAD(true);

  // Set total gap for beam to solid contact pair to zero
  // gap_ is used for calculating the contact flag of the beam to solid contact contact pair
  gap_ = 0.0;

  // Clear vector containing pairs of surface unit normal vector nD and beam parameter eta of
  // current time step
  normalsets_.clear();
  normalsets_.resize(0);

  // Initialize flag indicating if the beam to solid contact pair has contact
  // For active contact assemble contact forces and stiffness
  bool doAssemble = false;

  // Loop over all contact intervals, calculate and sum up contact forces (fc1, fc2) and stiffness
  // (stiffc1, stiffc2)
  for (int i = 0; i < (int)parsets.size() - 1; i += 2)
  {
    // Initialize flag indicating if the current contact interval has contact
    bool doAssembleContactInterval = false;

    // Two parameter sets indicate the beginning (a) and the end (b) of the contact interval
    // TODO: Use stiffc1_FAD and stiffc2_FAD only if needed (#ifdef AUTOMATICDIFFBTS)
    evaluate_contact_interval(pp, parsets[i], parsets[i + 1], fc1, fc2, stiffc1, stiffc2,
        doAssembleContactInterval, stiffc1_FAD, stiffc2_FAD);

    // If at least one contact interval has contact, set the doAssemble flag true
    if (!doAssemble && doAssembleContactInterval)
    {
      doAssemble = true;
    }
  }

  // -----------------------------------------------------------------
  // End: Compute contact forces and stiffness for all contact intervals


  // Assemble contact forces and contact stiffness
  // -----------------------------------------------------------------

  // Evaluate contact status for beam to solid contact pair
  contactflag_ = (gap_ < 0.0);

#ifdef FDCHECKSTIFFNESS
  // Calculate contact stiffness with finite difference
  if (doAssemble)
  {
    fd_check_stiffness(pp, fc1, fc2, stiffc1, stiffc2);
  }
#endif

#ifdef FADCHECKSTIFFNESS
  // Compare calculated contact stiffness with FAD
  if (doAssemble)
  {
    fad_check_stiffness(stiffc1, stiffc2, stiffc1_FAD, stiffc2_FAD);
  }
#endif

  // Finally assemble fc1, fc2 and stiffc1, stiffc2 for beam to solid contact pair
  if (doAssemble)
  {
    assemble_fc_and_stiffc_contact(fc1, fc2, &fint, stiffc1, stiffc2, stiffmatrix);
  }

  // -----------------------------------------------------------------
  // end: Assemble contact forces and contact stiffness

  return true;
}
/*----------------------------------------------------------------------*
 | End: Evaluate the element                                            |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Evaluate contact interval                                            |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::evaluate_contact_interval(
    const double& pp, const std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>& parset_a,
    const std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>& parset_b,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS>& fc1,
    Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS>& fc2,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
    Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
        stiffc2,
    bool& doAssembleContactInterval,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1_FAD,
    Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
        stiffc2_FAD)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // NOTE: Linearization lin() = (),d * lin_d with directional derivative (),d

  // Beam and surface element parameters
  TYPEBTS xi1 = 0.0;
  TYPEBTS xi2 = 0.0;
  TYPEBTS eta = 0.0;

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS> N1(true);         // = N1
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS> N1_eta(true);     // = N1,eta
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS> N1_etaeta(true);  // = N1,etaeta

  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2(true);         // = N2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi1(true);     // = N2,xi1
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi2(true);     // = N2,xi2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi1xi1(true);  // = N2,xi1xi1
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi2xi2(true);  // = N2,xi2xi2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi1xi2(true);  // = N2,xi1xi2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi2xi1(true);  // = N2,xi2xi1

  // Coords and derivatives of beam and surface element
  Core::LinAlg::Matrix<3, 1, TYPEBTS> r1(true);         // = r1
  Core::LinAlg::Matrix<3, 1, TYPEBTS> r1_eta(true);     // = r1,eta
  Core::LinAlg::Matrix<3, 1, TYPEBTS> r1_etaeta(true);  // = r1,etaeta

  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2(true);         // = x2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi1(true);     // = x2,xi1
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi2(true);     // = x2,xi2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi1xi1(true);  // = x2,xi1xi1
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi2xi2(true);  // = x2,xi2xi2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi1xi2(true);  // = x2,xi1xi2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi2xi1(true);  // = x2,xi2xi1

  // Distance vector its norm and distance unit vector
  Core::LinAlg::Matrix<3, 1, TYPEBTS> rD(true);  // = r1 - x2
  TYPEBTS norm_rD = 0.0;                         // = ||rD||
  Core::LinAlg::Matrix<3, 1, TYPEBTS> nD(true);  // = rD / norm_rD

  // Surface tangent cross product, its norm and unit surface normal vector
  Core::LinAlg::Matrix<3, 1, TYPEBTS> a2(true);  // = x2_xi1 x x2_xi2
  TYPEBTS norm_a2 = 0.0;                         // = ||a||
  Core::LinAlg::Matrix<3, 1, TYPEBTS> n2(true);  // = a / norm_a

  // Sign variable and gap function
  double sgn = 1.0;   // = sign(nD * n2)
  TYPEBTS gap = 0.0;  // = n2 * rD - radius1


  // Calculate radius of beam element via moment of inertia (only valid for beams with circular
  // cross section)
  const Discret::Elements::Beam3Base* beamele1 =
      static_cast<const Discret::Elements::Beam3Base*>(element1_);

  if (beamele1 == nullptr) FOUR_C_THROW("cast to beam base failed!");

  const double radius1 =
      MANIPULATERADIUS * beamele1->get_circular_cross_section_radius_for_interactions();

  // Get contact interval borders eta_a and eta_b
  TYPEBTS eta_a = parset_a.first(2);
  TYPEBTS eta_b = parset_b.first(2);


  // Calculate linearization of contact interval borders
  // TODO: Calculate only if the current contact interval has contact
  // -----------------------------------------------------------------

  // Initialize directional derivatives of contact interval borders
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> eta_a_d(true);
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> eta_b_d(true);

  // Compute directional derivatives eta_a_d (i = 0) and eta_b_d (i = 1)
  // TODO: May this can be done in a more beautiful way
  for (int i = 0; i < 2; i++)
  {
    Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> eta_d(true);
    std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int> parset;
    switch (i)
    {
      case 0:
        parset = parset_a;
        break;
      case 1:
        parset = parset_b;
        break;
    }

    // Get parameters from contact interval border parameter set
    TYPEBTS xi1 = parset.first(0);
    TYPEBTS xi2 = parset.first(1);
    TYPEBTS eta = parset.first(2);
    const int fixed_par = parset.second;

    // Initialize directional derivatives of surface parameters xi1 and xi2. They are used
    // temporary, because we are only interested in the directional derivative of the beam parameter
    // eta
    Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi1_d(true);
    Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi2_d(true);

#ifdef FADCHECKLINCONTACTINTERVALBORDER
    // Set known parameters xi1, xi2, eta and element positions as primary variables
    // for checking the linearization of the contact interval borders with FAD
    BeamContact::SetFADParCoordDofs<numnodessol, numnodes, numnodalvalues>(xi1, xi2, eta);
    BeamContact::SetFADDispDofs<numnodessol, numnodes, numnodalvalues>(ele1pos_, ele2pos_, 3);
#endif

    // Update shape functions and their derivatives for beam and surface element
    get_beam_shape_functions(N1, N1_eta, N1_etaeta, eta);
    get_surf_shape_functions(
        N2, N2_xi1, N2_xi2, N2_xi1xi1, N2_xi2xi2, N2_xi1xi2, N2_xi2xi1, xi1, xi2);

    // Update coordinates and derivatives for beam and surface element
    compute_beam_coords_and_derivs(r1, r1_eta, r1_etaeta, N1, N1_eta, N1_etaeta);
    compute_surf_coords_and_derivs(x2, x2_xi1, x2_xi2, x2_xi1xi1, x2_xi2xi2, x2_xi1xi2, x2_xi2xi1,
        N2, N2_xi1, N2_xi2, N2_xi1xi1, N2_xi2xi2, N2_xi1xi2, N2_xi2xi1);

    // Compute unit distance vector nD
    compute_distance_normal(r1, x2, rD, norm_rD, nD);

    // Add nD and eta to normalsets_ of the current time step
    std::pair<TYPEBTS, Core::LinAlg::Matrix<3, 1, TYPEBTS>> normalset;
    normalset.first = eta;
    normalset.second = nD;
    normalsets_.push_back(normalset);

#if defined(GMSHDEBUG) || defined(GMSHFORCE) || defined(SAVEFORCE)
    // Compute some variables at contact interval border
    compute_surface_normal(x2_xi1, x2_xi2, a2, norm_a2, n2);
    gap = Core::FADUtils::ScalarProduct(n2, rD) - radius1;
    double fp = 0.0;
    if (gap < 0)
    {
      TYPEBTS dfp = 0.0;
      evaluate_penalty_force_law(pp, gap, fp, dfp);
    }

    // Store variables at contact interval border in gmshDebugPoints_
    gmshDebugPoint gmshDebugPoint;
    gmshDebugPoint.r1 = r1;
    gmshDebugPoint.x2 = x2;
    gmshDebugPoint.n2 = n2;
    gmshDebugPoint.gap = gap;
    gmshDebugPoint.fp = fp;
    gmshDebugPoint.type =
        -1 + 2 * i;  // -1: Left contact interval border, +1: Right contact interval border
    gmshDebugPoints_.push_back(gmshDebugPoint);
#endif

    // If eta is fixed then its linearization is zero, otherwise compute directional derivative
    if (fixed_par == 2)
    {
      // Set directional derivative of contact interval border to zero
      eta_d.clear();
    }
    else
    {
      xi1_d.clear();
      xi2_d.clear();
      eta_d.clear();

      // Compute directional derivatives of not fixed parameters at contact interval border,
      // only the directional derivative eta_d is needed
      compute_lin_parameter(fixed_par, xi1_d, xi2_d, eta_d, rD, r1_eta, x2_xi1, x2_xi2, x2_xi1xi1,
          x2_xi2xi2, x2_xi1xi2, x2_xi2xi1, N1, N2, N2_xi1, N2_xi2);

#ifdef FADCHECKLINCONTACTINTERVALBORDER
      // Print some information
      std::cout << std::endl
                << "FAD-Check: Linearization of contact interval borders at "
                << "xi1: " << xi1.val() << ", xi2: " << xi2.val() << ", eta: " << eta.val()
                << " with fixed_par: " << fixed_par;
      switch (i)
      {
        case 0:
          std::cout << " (eta_a_d): " << std::endl;
          break;
        case 1:
          std::cout << " (eta_b_d): " << std::endl;
          break;
      }

      // Directional derivatives of eta and temporary derivatives of xi1 and xi2
      Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi_1d_FAD(true);
      Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi_2d_FAD(true);
      Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> eta_d_FAD(true);

      // Compute linearization of contact interval borders eta_a (if i == 0) or eta_b (if i == 1)
      // with FAD
      fad_check_lin_parameter(
          fixed_par, rD, x2_xi1, x2_xi2, xi_1d_FAD, xi_2d_FAD, eta_d_FAD, xi1_d, xi2_d, eta_d);
#endif
    }

    switch (i)
    {
      case 0:
        eta_a_d = eta_d;
        break;
      case 1:
        eta_b_d = eta_d;
        break;
    }
  }
  // -----------------------------------------------------------------
  // End: Calculate linearization of contact interval borders


  // Evaluate contact force and stiffness at each Gauss point
  // -----------------------------------------------------------------

  // Initialize directional derivative of eta at Gauss point
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> eta_d(true);

  // Get Gauss points and weights in interval [-1, 1]
  Core::FE::IntegrationPoints1D gaussPoints = Core::FE::IntegrationPoints1D(GAUSSRULE);

  // Loop over all Gauss points in contact interval [eta_a, eta_b]
  for (int i_gp = 0; i_gp < gaussPoints.nquad; i_gp++)
  {
    // Initialize flag indicating if Gauss point has contact
    bool contactflag = false;

    // Initialize flag indicating if projection at Gauss point is allowed
    bool proj_allowed = false;

    // Set surface parameters and gap to zero
    xi1 = 0.0;
    xi2 = 0.0;
    gap = 0.0;

    // Get integration point and weight
    const double x_gp = gaussPoints.qxg[i_gp][0];
    const double w_gp = gaussPoints.qwgt[i_gp];

    // Calculate beam parameter eta at Gauss point using a linear transformation
    eta = 0.5 * (eta_b - eta_a) * x_gp + 0.5 * (eta_a + eta_b);

    // Calculate directional derivative of eta at Gauss point
    eta_d.clear();
    for (int i = 0; i < dim1 + dim2; i++)
    {
      eta_d(i) = 0.5 * (1 - x_gp) * eta_a_d(i) + 0.5 * (1 + x_gp) * eta_b_d(i);
    }

    // Projection of Gauss point on surface with fixed eta to get xi1 and xi2
    projection(2, xi1, xi2, eta, proj_allowed);

    // Check if the projected point found for this contact pair is really on the
    // considered surface element, i.e. xi1 and xi2 in [-1, 1]
    if (fabs(xi1) > 1.0 + XIETATOL || fabs(xi2) > 1.0 + XIETATOL)
    {
      FOUR_C_THROW("Projection on surface is not on the considered surface element.");

      // std::cout << "Projection of Gauss point " << i_gp + 1 << " of " << gaussPoints.nquad
      //     << " (at eta: " << eta << ") on surface is not on the considered surface element." <<
      //     std::endl;
      // std::cout << "xi1: " << xi1 << std::endl;
      // std::cout << "xi2: " << xi2 << std::endl;
      // continue;
    }

#ifdef FADCHECKSTIFFNESS
    // Set known parameters xi1, xi2, contact interval borders eta_a, eta_b and element
    // positions as primary variables for checking the contact stiffness with FAD
    BeamContact::SetFADParCoordDofs<numnodessol, numnodes, numnodalvalues>(xi1, xi2, eta_a, eta_b);
    BeamContact::SetFADDispDofs<numnodessol, numnodes, numnodalvalues>(ele1pos_, ele2pos_, 4);

    // Calculate eta again depending on eta_a and eta_b, because at this point all FAD paramater
    // coordinates are known
    eta = 0.5 * (eta_b - eta_a) * x_gp + 0.5 * (eta_a + eta_b);
#endif

#ifdef FADCHECKLINGAUSSPOINT
    // Set known parameters xi1, xi2, eta and element positions as primary variables for
    // checking the linearization of surface parameters xi1 and xi2, gap and surface unit normal
    // vector n2
    BeamContact::SetFADParCoordDofs<numnodessol, numnodes, numnodalvalues>(xi1, xi2, eta);
    BeamContact::SetFADDispDofs<numnodessol, numnodes, numnodalvalues>(ele1pos_, ele2pos_, 3);
#endif

    // Update shape functions and their derivatives for beam and surface element
    get_beam_shape_functions(N1, N1_eta, N1_etaeta, eta);
    get_surf_shape_functions(
        N2, N2_xi1, N2_xi2, N2_xi1xi1, N2_xi2xi2, N2_xi1xi2, N2_xi2xi1, xi1, xi2);

    // Update coordinates and derivatives for beam and surface element
    compute_beam_coords_and_derivs(r1, r1_eta, r1_etaeta, N1, N1_eta, N1_etaeta);
    compute_surf_coords_and_derivs(x2, x2_xi1, x2_xi2, x2_xi1xi1, x2_xi2xi2, x2_xi1xi2, x2_xi2xi1,
        N2, N2_xi1, N2_xi2, N2_xi1xi1, N2_xi2xi2, N2_xi1xi2, N2_xi2xi1);

    // Compute distance vector rD, its norm norm_rD and unit distance vector nD
    compute_distance_normal(r1, x2, rD, norm_rD, nD);

    // Compute tangent cross product a2, its norm norm_a2 and surface unit normal vector n2
    compute_surface_normal(x2_xi1, x2_xi2, a2, norm_a2, n2);

    // Compute sign of projection of distance vector rD on surface unit normal vector n2
    sgn = Core::FADUtils::cast_to_double(
        Core::FADUtils::signum(Core::FADUtils::scalar_product(nD, n2)));

    // Compute gap at Gauss point
    gap = Core::FADUtils::scalar_product(n2, rD) - radius1;  // = sgn * norm_rD - radius1;

    // If the beam element center line lies under the surface element and there is no change of sgn
    // from the last to the current time step, the contact is not active (here called projection not
    // allowed) and contactflag is false (initialization). Otherwise check the contact status at
    // this Gauss point.
    if (proj_allowed)
    {
      check_contact_status(pp, gap, contactflag);
    }

#if defined(GMSHDEBUG) || defined(GMSHFORCE) || defined(SAVEFORCE)
    if (sgn > 0 || contactflag)
    {
      // Insert variables at current Gauss point in gmshDebugPoints_ before variables of contact
      // interval end
      double fp = 0.0;
      if (gap < 0)
      {
        TYPEBTS dfp = 0.0;
        evaluate_penalty_force_law(pp, gap, fp, dfp);
      }
      gmshDebugPoint gmshDebugPoint;
      gmshDebugPoint.r1 = r1;
      gmshDebugPoint.x2 = x2;
      gmshDebugPoint.n2 = n2;
      gmshDebugPoint.gap = gap;
      gmshDebugPoint.fp = fp;
      gmshDebugPoint.type = 0;  // 0: Gauss point
      gmshDebugPoints_.insert(gmshDebugPoints_.end() - 1, gmshDebugPoint);
    }
#endif

    // Evaluate and sum up contact forces and stiffness only if the current Gauss point has contact
    if (contactflag)
    {
      // Sum up all negative gaps for getting a total gap for the current beam to solid contact pair
      gap_ += gap;

      // Get Jacobi factor of beam element
      const double jacobi = (static_cast<Discret::Elements::Beam3eb*>(element1_))->jacobi();

      // Evaluate penalty force law with penalty parameter pp and gap to get force fp and derivative
      // dfp = dfp / dgap
      TYPEBTS fp = 0.0;
      TYPEBTS dfp = 0.0;
      evaluate_penalty_force_law(pp, gap, fp, dfp);

      // Evaluate contact forces at current Gauss point
      evaluate_fc_contact(fp, fc1, fc2, eta_a, eta_b, w_gp, sgn, nD, n2, N1, N2, jacobi);

      // Evaluate contact stiffness at current Gauss point
      evaluate_stiffc_contact(fp, dfp, gap, stiffc1, stiffc2, sgn, eta_a, eta_b, eta_d, eta_a_d,
          eta_b_d, w_gp, rD, norm_rD, nD, a2, norm_a2, n2, r1_eta, x2_xi1, x2_xi2, x2_xi1xi1,
          x2_xi2xi2, x2_xi1xi2, x2_xi2xi1, N1, N1_eta, N2, N2_xi1, N2_xi2, jacobi, stiffc1_FAD,
          stiffc2_FAD);

      // Set the doAssembleContactInterval flag true, if at least one Gauss point
      // inside the current contact interval has active contact
      if (!doAssembleContactInterval)
      {
        doAssembleContactInterval = true;
      }
    }

    // Save current parameter eta and unit distance vector nD in normalsets_ of the current time
    // step by inserting this pair before the last normalset in normalsets_ (contact interval end)
    std::pair<TYPEBTS, Core::LinAlg::Matrix<3, 1, TYPEBTS>> normalset;
    normalset.first = eta;
    normalset.second = nD;
    normalsets_.insert(normalsets_.end() - 1, normalset);
  }
  // -----------------------------------------------------------------
  // End: Evaluate contact force and stiffness at each Gauss point

  return;
}
/*----------------------------------------------------------------------*
 | End: Evaluate contact interval                                       |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Evaluate penalty force law                                           |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::evaluate_penalty_force_law(const double& pp, const TYPEBTS& gap, TYPEBTS& fp,
    TYPEBTS& dfp)
{
#ifdef LINPENALTY
  // Linear penalty force law
  fp = -pp * gap;
  dfp = -pp;
#endif

#ifdef QUADPENALTY
  // Quadratic penalty force law
  fp = pp * gap * gap;
  dfp = 2 * pp * gap;
#endif

#ifdef ARBITPENALTY
  const double g0 = G0;

  // Quadratic penalty force law
  if (ARBITPENALTY == 1)  // Cubic regularization for positive gaps
  {
    double k = 2.0;
    double f0 = pp * g0 / k;
    double factor_a = -pp / (g0 * g0) + 2 * f0 / (g0 * g0 * g0);
    double factor_b = 2 * pp / g0 - 3 * f0 / (g0 * g0);
    double factor_c = -pp;
    double factor_d = f0;
    if (gap > 0.0)
    {
      // std::cout << "Regularized Penalty!" << std::endl;
      fp = factor_a * gap * gap * gap + factor_b * gap * gap + factor_c * gap + factor_d;
      dfp = 3 * factor_a * gap * gap + 2 * factor_b * gap + factor_c;
    }
    else
    {
      // std::cout << "Linear Penalty!" << std::endl;
      fp = f0 - pp * gap;
      dfp = -pp;
    }
  }
  else if (ARBITPENALTY == 2)  // Quadratic regularization for negative gaps
  {
    if (gap > -g0)
    {
      // std::cout << "Regularized Penalty!" << std::endl;
      fp = pp / (2.0 * g0) * gap * gap;
      dfp = pp / g0 * gap;
    }
    else
    {
      // std::cout << "Linear Penalty!" << std::endl;
      fp = -pp * (gap + g0 / 2.0);
      dfp = -pp;
    }
  }
  else if (ARBITPENALTY == 3)  // Quadratic regularization for positiv gaps
  {
    double f0 = g0 * pp / 2.0;
    double factor_a = pp / g0 - f0 / (g0 * g0);
    double factor_b = -pp;
    double factor_c = f0;

    if (gap > 0)
    {
      // std::cout << "Regularized Penalty!" << std::endl;
      fp = factor_a * gap * gap + factor_b * gap + factor_c;
      dfp = 2 * factor_a * gap + factor_b;
    }
    else
    {
      // std::cout << "Linear Penalty!" << std::endl;
      fp = f0 - pp * gap;
      dfp = -pp;
    }
  }
  else if (ARBITPENALTY == 4)  // Double quadratic regularization for positiv gaps
  {
    double f0 = 0.25;
    double g1 = 1.8 * f0 / pp;
    double c_tilde = f0;
    double b_tilde = -pp;
    double a_bar = (2 * f0 - pp * g1) / (2 * g0 * (g0 - g1));
    double b_bar = -2 * g0 * a_bar;
    double c_bar = -g0 * g0 * a_bar - g0 * b_bar;
    double a_tilde = (2 * g1 * a_bar + b_bar - b_tilde) / (2 * g1);

    if (gap > g1)
    {
      // std::cout << "Regularized Penalty: g1 < gap < g0!" << std::endl;
      fp = a_bar * gap * gap + b_bar * gap + c_bar;
      dfp = 2 * a_bar * gap + b_bar;
    }
    else if (gap > 0)
    {
      // std::cout << "Regularized Penalty: 0 < gap < g1!" << std::endl;
      fp = a_tilde * gap * gap + b_tilde * gap + c_tilde;
      dfp = 2 * a_tilde * gap + b_tilde;
    }
    else
    {
      // std::cout << "Linear Penalty!" << std::endl;
      fp = f0 - pp * gap;
      dfp = -pp;
    }
  }
  else if (ARBITPENALTY == 5)  // Exponential regularization for positiv gaps
  {
    double f0 = 0.25;

    if (gap > 0)
    {
      // std::cout << "Regularized Penalty: 0 < gap < g1!" << std::endl;
      fp = f0 * exp(-pp * gap / f0);
      dfp = -pp * exp(-pp * gap / f0);
    }
    else
    {
      // std::cout << "Linear Penalty!" << std::endl;
      fp = f0 - pp * gap;
      dfp = -pp;
    }
  }
  else
    FOUR_C_THROW("Only the values 1,2,3,4 are possible for ARBITPENALTY!");
#endif
}
/*----------------------------------------------------------------------*
 | End: Evaluate penalty force law                                      |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Compute contact forces                                               |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::evaluate_fc_contact(
    const TYPEBTS& fp, Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS>& fc1,
    Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS>& fc2, const TYPEBTS& eta_a,
    const TYPEBTS& eta_b, const double& w_gp, const double& sgn,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2, const double& jacobi)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // NOTE: Macauley-bracket <-gap> = -gap if gap < 0


  // Evaluate contact forces acting on beam element
  // -----------------------------------------------------------------

  // Compute fc1 at Gauss point for beam element
  for (int i = 0; i < dim1; i++)
    for (int j = 0; j < 3; j++)
      fc1(i) += jacobi * w_gp * fp * N1(j, i) * n2(j) * 0.5 * (eta_b - eta_a);

  // -----------------------------------------------------------------
  // end: Evaluate contact forces acting on beam element


  // Evaluate contact forces acting on solid element
  // -----------------------------------------------------------------

  // Compute fc2 at Gauss point for surface element
  for (int i = 0; i < dim2; i++)
    for (int j = 0; j < 3; j++)
      fc2(i) += -jacobi * w_gp * fp * N2(j, i) * n2(j) * 0.5 * (eta_b - eta_a);

  // -----------------------------------------------------------------
  // end: Evaluate contact forces acting on solid element

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute contact forces                                          |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Evaluate contact stiffness                                           |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::evaluate_stiffc_contact(
    const TYPEBTS& fp, const TYPEBTS& dfp, const TYPEBTS& gap,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
    Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
        stiffc2,
    const double& sgn, const TYPEBTS& eta_a, const TYPEBTS& eta_b,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_a_d,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_b_d,
    const double& w_gp, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD, const TYPEBTS& norm_rD,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& a2,
    const TYPEBTS& norm_a2, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_eta,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi1,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1_eta,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi2, const double& jacobi,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1_FAD,
    Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
        stiffc2_FAD)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // NOTE: Linearization lin() = (),d * lin_d with directional derivative (),d

  // Initialize storage for directional derivatives
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi1_d(true);
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi2_d(true);
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> gap_d(true);
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> rD_d(true);
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> nD_d(true);
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> n2_d(true);

  // Compute directional derivative of parameters xi1 and xi2 with given eta_d (par_fixed = 2)
  compute_lin_parameter(2, xi1_d, xi2_d, eta_d, rD, r1_eta, x2_xi1, x2_xi2, x2_xi1xi1, x2_xi2xi2,
      x2_xi1xi2, x2_xi2xi1, N1, N2, N2_xi1, N2_xi2);

  // Compute directional derivatives of gap and distance vector rD
  compute_lin_gap(
      gap_d, xi1_d, xi2_d, eta_d, sgn, rD, nD, n2, norm_rD, r1_eta, x2_xi1, x2_xi2, N1, N2, rD_d);

  // Compute directional derivative of normal vector n2
  compute_lin_normal(nD_d, nD, norm_rD, n2_d, n2, norm_a2, rD_d, xi1_d, xi2_d, x2_xi1, x2_xi2,
      x2_xi1xi1, x2_xi2xi2, x2_xi1xi2, x2_xi2xi1, N2_xi1, N2_xi2);


#ifdef FADCHECKLINGAUSSPOINT
  // Print information for linearization of surface parameters at Gauss point
  std::cout << std::endl
            << "FAD-Check: Linearization of surface parameters at Gauss point" << std::endl;

  // Initialize directional derivatives of surface parameters xi1 and xi2 for FAD
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi_1d_FAD(true);
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> xi_2d_FAD(true);

  // At this point the directional derivative of eta is already known and is used for calculating
  // xi1_d and xi2_d
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> eta_d_FAD = eta_d;

  // Compute directional derivatives of surface parameters xi1 and xi2 for fixed eta (par_fixed = 2)
  // with FAD
  fad_check_lin_parameter(
      2, rD, x2_xi1, x2_xi2, xi_1d_FAD, xi_2d_FAD, eta_d_FAD, xi1_d, xi2_d, eta_d);

  // Print information for linearization of gap and distance vector
  std::cout << std::endl << "FAD-Check: Linearization of gap and distance vector" << std::endl;

  // Initialize directional derivative of gap and distance vector rD for FAD
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> gap_d_FAD(true);
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> rD_d_FAD(true);

  // Compute directional derivative of gap and distance vector rD with FAD
  fad_check_lin_gap_and_distance_vector(
      gap, rD, xi1_d, xi2_d, eta_d, gap_d_FAD, rD_d_FAD, gap_d, rD_d);

  // Print information for linearization of normal vector
  std::cout << std::endl << "FAD-Check: Linearization of normal vector" << std::endl;

  // Initialize directional derivatives of unit distance vector and normal vector n2 for FAD
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> nD_d_FAD(true);
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> n_2d_FAD(true);

  // Compute directional derivatives of unit distance vector and normal vector n2 with FAD
  fad_check_lin_normal(nD, n2, xi1_d, xi2_d, eta_d, nD_d_FAD, n_2d_FAD, nD_d, n2_d);
#endif


  // Initialize flag indicating if the complete stiffness is used
  bool completestiff = true;
#ifdef GAP0
  // In order to accelerate convergence, we only apply the basic stiffness part in case of very
  // large negative gaps NOTE: GAP0 > 0
  if (gap < -GAP0)
  {
    completestiff = false;
    std::cout << "gap: " << gap << " < -GAP0: " << -GAP0 << " => completestiff: " << completestiff
              << std::endl;
  }
#endif


  // NOTE: Macaulay bracket <-gap> = -gap and Heaviside function H(-gap) = 1 if gap < 0,
  // lin_<-gap> = -H(-gap) * lin_gap = -lin_gap <=> <-gap>_d = -gap_d


  // Evaluate contact stiffness of beam element
  // -----------------------------------------------------------------

  // Part I: gap_d
  Core::LinAlg::Matrix<dim1, 1, TYPEBTS> N1T_n2(true);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1; j++) N1T_n2(j) += N1(i, j) * n2(i);

  for (int i = 0; i < dim1; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      stiffc1(i, j) += jacobi * w_gp * dfp * N1T_n2(i) * gap_d(j) * 0.5 * (eta_b - eta_a);

  if (completestiff)
  {
    // Part II: n2_d
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < dim1; j++)
        for (int k = 0; k < dim1 + dim2; k++)
          stiffc1(j, k) += jacobi * w_gp * fp * N1(i, j) * n2_d(i, k) * 0.5 * (eta_b - eta_a);

    // Part III: N1_d
    Core::LinAlg::Matrix<dim1, 1, TYPEBTS> N1_etaT_n2(true);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < dim1; j++) N1_etaT_n2(j) += N1_eta(i, j) * n2(i);
    for (int i = 0; i < dim1; i++)
      for (int j = 0; j < dim1 + dim2; j++)
        stiffc1(i, j) += jacobi * w_gp * fp * N1_etaT_n2(i) * eta_d(j) * 0.5 * (eta_b - eta_a);

    // Part IV: eta_a_d and eta_b_d (Jacobi factor contact interval)
    for (int i = 0; i < dim1; i++)
      for (int j = 0; j < dim1 + dim2; j++)
        stiffc1(i, j) += jacobi * w_gp * fp * N1T_n2(i) * 0.5 * (eta_b_d(j) - eta_a_d(j));
  }

  // -----------------------------------------------------------------
  // end: Evaluate contact stiffness of beam element


  // Evaluate contact stiffness of solid element
  // -----------------------------------------------------------------

  // Part I: gap_d
  Core::LinAlg::Matrix<dim2, 1, TYPEBTS> N2T_n2(true);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim2; j++) N2T_n2(j) += N2(i, j) * n2(i);

  for (int i = 0; i < dim2; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      stiffc2(i, j) += -jacobi * w_gp * dfp * N2T_n2(i) * gap_d(j) * 0.5 * (eta_b - eta_a);

  if (completestiff)
  {
    // Part II: n2_d
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < dim2; j++)
        for (int k = 0; k < dim1 + dim2; k++)
          stiffc2(j, k) += -jacobi * w_gp * fp * N2(i, j) * n2_d(i, k) * 0.5 * (eta_b - eta_a);

    // Part III: N2_d
    Core::LinAlg::Matrix<dim2, 1, TYPEBTS> N2_xi1T_n2(true);
    Core::LinAlg::Matrix<dim2, 1, TYPEBTS> N2_xi2T_n2(true);
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < dim2; j++)
      {
        N2_xi1T_n2(j) += N2_xi1(i, j) * n2(i);
        N2_xi2T_n2(j) += N2_xi2(i, j) * n2(i);
      }
    }
    for (int i = 0; i < dim2; i++)
      for (int j = 0; j < dim1 + dim2; j++)
        stiffc2(i, j) += -jacobi * w_gp * fp *
                         (N2_xi1T_n2(i) * xi1_d(j) + N2_xi2T_n2(i) * xi2_d(j)) * 0.5 *
                         (eta_b - eta_a);

    // Part IV: eta_a_d and eta_b_d (Jacobi factor contact interval)
    for (int i = 0; i < dim2; i++)
      for (int j = 0; j < dim1 + dim2; j++)
        stiffc2(i, j) += -jacobi * w_gp * fp * N2T_n2(i) * 0.5 * (eta_b_d(j) - eta_a_d(j));
  }

  // -----------------------------------------------------------------
  // end: Evaluate contact stiffness of solid element


#ifdef FADCHECKSTIFFNESS

  Core::LinAlg::Matrix<dim1, 1, TYPEBTS> fc1_FAD(true);
  Core::LinAlg::Matrix<dim2, 1, TYPEBTS> fc2_FAD(true);

  // Evaluate contact forces at current Gauss point (fc1_FAD, fc2_FAD)
  evaluate_fc_contact(fp, fc1_FAD, fc2_FAD, eta_a, eta_b, w_gp, sgn, nD, n2, N1, N2, jacobi);

  // Calculate contact stiffness (stiffc1_FAD, stiffc2_FAD) with FAD
  for (int j = 0; j < dim1 + dim2; j++)
  {
    // Beam element
    for (int i = 0; i < dim1; i++)
    {
      stiffc1_FAD(i, j) += fc1_FAD(i).dx(j) + fc1_FAD(i).dx(dim1 + dim2) * xi1_d(j) +
                           fc1_FAD(i).dx(dim1 + dim2 + 1) * xi2_d(j) +
                           fc1_FAD(i).dx(dim1 + dim2 + 2) * eta_a_d(j) +
                           fc1_FAD(i).dx(dim1 + dim2 + 3) * eta_b_d(j);
    }

    // Surface element
    for (int i = 0; i < dim2; i++)
    {
      stiffc2_FAD(i, j) += fc2_FAD(i).dx(j) + fc2_FAD(i).dx(dim1 + dim2) * xi1_d(j) +
                           fc2_FAD(i).dx(dim1 + dim2 + 1) * xi2_d(j) +
                           fc2_FAD(i).dx(dim1 + dim2 + 2) * eta_a_d(j) +
                           fc2_FAD(i).dx(dim1 + dim2 + 3) * eta_b_d(j);
    }
  }
#endif

  return;
}
/*----------------------------------------------------------------------*
 | End: Evaluate contact stiffness
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Compute directional derivatives of element parameters                |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::compute_lin_parameter(
    const int& fixed_par,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_eta,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi1,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi2)
{
  // Compute linearization of two parameters with one known parameter linearization
  //
  // NOTE: Linearization lin() = (),d * lin_d with directional derivative (),d
  //
  // Solve the following system of equations:
  //                                        _       _
  //  _                               _    | lin_xi1 |    _                _     _      _
  // | Lpar(1,1)  Lpar(1,2)  Lpar(1,3) |   |         |   | B'(1,1)  B'(1,2) |   | lin_d1 |
  // |                                 | * | lin_xi2 | = |                  | * |        |
  // |_Lpar(2,1)  Lpar(2,2)  Lpar(2,3)_|   |         |   |_B'(2,1)  B'(2,2)_|   |_lin_d2_|
  //                                       |_lin_eta_|
  //
  // This is only possible if one parameter linerization is known and that's the case here.
  //
  // Assemble needed entries of Lpar in L:
  //  _              _     _        _     _                _     _      _     _      _
  // | L(1,1)  L(1,2) |   | lin_par1 |   | B'(1,1)  B'(1,2) |   | lin_d1 |   | L(1,3) |
  // |                | * |          | = |                  | * |        | - |        | * lin_par3
  // |_L(2,1)  L(2,2)_|   |_lin_par2_|   |_B'(2,1)  B'(2,2)_|   |_lin_d2_|   |_L(2,3)_|
  //
  // Solution by inverting matrix L:
  //
  // [lin_par1; lin_par2] = L^-1 * (B' * [lin_d1; lin_d2] - [L(1,3); L(2,3)] * lin_par3) = L^-1 * B
  //
  // with B = B' * [lin_d1; lin_d2] - [L(1,3); L(2,3)] * lin_par3
  //
  // [par1_d; par2_d] = D = L^-1 * B

  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;


  // Initialize vector with parameters xi1, xi2, eta
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> par_d[3];
  par_d[0] = xi1_d;
  par_d[1] = xi2_d;
  par_d[2] = eta_d;

  // Get indices of not fixed parameters
  int par_i[2];
  int j = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != fixed_par)
    {
      par_i[j] = i;
      j++;
    }
  }

  // Clear not fixed directional derivatives
  for (int i = 0; i < 2; i++)
  {
    par_d[par_i[i]].clear();
  }

  // Initialize matrices
  Core::LinAlg::Matrix<2, 2, TYPEBTS> L(true);
  Core::LinAlg::Matrix<2, 3, TYPEBTS> Lpar(true);
  Core::LinAlg::Matrix<2, 2, TYPEBTS> L_inv(true);
  Core::LinAlg::Matrix<2, dim1 + dim2, TYPEBTS> B(true);
  Core::LinAlg::Matrix<2, dim1 + dim2, TYPEBTS> D(true);

  // Compute Lpar for all parametes
  Lpar(0, 0) = -Core::FADUtils::scalar_product(x2_xi1, x2_xi1) +
               Core::FADUtils::scalar_product(rD, x2_xi1xi1);
  Lpar(1, 0) = -Core::FADUtils::scalar_product(x2_xi1, x2_xi2) +
               Core::FADUtils::scalar_product(rD, x2_xi2xi1);
  Lpar(0, 1) = -Core::FADUtils::scalar_product(x2_xi2, x2_xi1) +
               Core::FADUtils::scalar_product(rD, x2_xi1xi2);
  Lpar(1, 1) = -Core::FADUtils::scalar_product(x2_xi2, x2_xi2) +
               Core::FADUtils::scalar_product(rD, x2_xi2xi2);
  Lpar(0, 2) = Core::FADUtils::scalar_product(r1_eta, x2_xi1);
  Lpar(1, 2) = Core::FADUtils::scalar_product(r1_eta, x2_xi2);

  // Assemble needed entries of Lpar in L
  L(0, 0) = Lpar(0, par_i[0]);
  L(1, 0) = Lpar(1, par_i[0]);
  L(0, 1) = Lpar(0, par_i[1]);
  L(1, 1) = Lpar(1, par_i[1]);

  // invert L by hand
  TYPEBTS det_L = L(0, 0) * L(1, 1) - L(0, 1) * L(1, 0);
  if (Core::FADUtils::cast_to_double(Core::FADUtils::norm(det_L)) < DETERMINANTTOL)
  {
    FOUR_C_THROW("ERROR: determinant of L = 0");
  }
  L_inv(0, 0) = L(1, 1) / det_L;
  L_inv(0, 1) = -L(0, 1) / det_L;
  L_inv(1, 0) = -L(1, 0) / det_L;
  L_inv(1, 1) = L(0, 0) / det_L;

  // Compute B
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1; j++)
    {
      B(0, j) += -N1(i, j) * x2_xi1(i);
      B(1, j) += -N1(i, j) * x2_xi2(i);
    }
  }
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim2; j++)
    {
      B(0, j + dim1) += x2_xi1(i) * N2(i, j) - rD(i) * N2_xi1(i, j);
      B(1, j + dim1) += x2_xi2(i) * N2(i, j) - rD(i) * N2_xi2(i, j);
    }
  }

  // Evaluation at Gauss point: In this case the directional derivative of the fixed beam parameter
  // eta is already known and is now used to calculate the directional derivatives of the both
  // surface parameters xi1 and xi2 NOTE: This is not implemented for fixed xi1 or xi2 yet, because
  // for computing the linearization of the contact interval borders lin_xi1 or lin_xi2 is zero
  // (projection of surface edges)
  if (fixed_par == 2)  // eta fixed
  {
    for (int i = 0; i < dim1 + dim2; i++)
    {
      B(0, i) += -Lpar(0, 2) * eta_d(i);
      B(1, i) += -Lpar(1, 2) * eta_d(i);
    }
  }

  // Compute D = L^-1 * B
  D.multiply(L_inv, B);

  // Finally get directional derivatives
  for (int i = 0; i < dim1 + dim2; i++)
  {
    par_d[par_i[0]](i) = D(0, i);
    par_d[par_i[1]](i) = D(1, i);
  }

  xi1_d = par_d[0];
  xi2_d = par_d[1];
  eta_d = par_d[2];

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute directional derivatives of element parameters           |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Compute directional derivative of gap                                |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::compute_lin_gap(
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& gap_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
    const double sgn, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
    const TYPEBTS& norm_rD, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1_eta,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& rD_d)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Compute gap_d and rD_d
  //
  // NOTE: Linearization lin() = (),d * lin_d with directional derivative (),d
  //
  // NOTE: lin_<-gap> = -H(-gap) * lin_gap with the Heaviside function H
  //
  // lin_gap = gap_d * lin_d = n2 * rD_d * lin_d
  //
  // with rD_d = r1_eta * eta_d - x2_xi1 * xi1_d - x2_xi2 * xi2_d + [N1, -N2] and gap_d = n2 * rD_d

  // First compute directional derivative of distance vector rD
  rD_d.clear();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      rD_d(i, j) += r1_eta(i) * eta_d(j) - x2_xi1(i) * xi1_d(j) - x2_xi2(i) * xi2_d(j);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1; j++) rD_d(i, j) += N1(i, j);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim2; j++) rD_d(i, j + dim1) += -N2(i, j);


  // TODO: Old/alternative linearization
  // Finally compute directional derivative of gap
  // gap_d.clear();
  // for (int i = 0; i < 3; i++)
  //  for (int j = 0; j < dim1+dim2; j++)
  //    gap_d(j) += sgn * rD(i) * rD_d(i,j) / norm_rD;
  // std::cout << "gap_d (old linearization): " << gap_d << std::endl;


  // Finally compute directional derivative of gap (new linearization)
  // TODO: Use directly gap_d = n2 * r1_eta * eta_d + n2 * [N1, -N2] instead of
  // gap_d = n2 * rD_d using rD_d, because n2 * x2_xi1 * xi1_d = n2 * x2_xi2 * xi2_d = 0
  gap_d.clear();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1 + dim2; j++) gap_d(j) += n2(i) * rD_d(i, j);
  // std::cout << "gap_d (new linearization): " << gap_d << std::endl;

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute directional derivative of gap                           |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Compute directional derivative of surface unit normal vector         |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::compute_lin_normal(
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& nD_d,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD, const TYPEBTS& norm_rD,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& n2_d,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2, const TYPEBTS& norm_a2,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& rD_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1xi2,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2xi1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi1,
    const Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N2_xi2)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Compute nD_d
  //
  // NOTE: Linearization lin() = (),d * lin_d with directional derivative (),d
  //
  // lin_nD = nD_d * lin_d = auxiliary_matrix * a2_d * lin_d
  //
  // with auxiliary_matrix = (I - n2 x n2) / norm_a2

  // Initialize auxiliary_matrix
  Core::LinAlg::Matrix<3, 3, TYPEBTS> auxiliary_matrix(true);

  // TODO: Old/alternative linearization
  // Compute auxiliary_matrix
  for (int i = 0; i < 3; i++)
  {
    auxiliary_matrix(i, i) += 1.0 / norm_rD;
    for (int j = 0; j < 3; j++) auxiliary_matrix(i, j) += -nD(i) * nD(j) / norm_rD;
  }
  // Finally compute derivatives of unit distance vector
  nD_d.clear();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < dim1 + dim2; k++) nD_d(i, k) += auxiliary_matrix(i, j) * rD_d(j, k);
  // std::cout << "nD_d (old linearization): " << nD_d << std::endl;


  // Skew-symmetric matrix to replace cross product with matrix multiplication
  // TODO: Use a function for calculating cross products
  Core::LinAlg::Matrix<3, 3, TYPEBTS> x2_xi1_tilde(true);
  x2_xi1_tilde(2, 1) = x2_xi1(0);
  x2_xi1_tilde(2, 0) = -x2_xi1(1);
  x2_xi1_tilde(1, 0) = x2_xi1(2);
  x2_xi1_tilde(1, 2) = -x2_xi1_tilde(2, 1);
  x2_xi1_tilde(0, 2) = -x2_xi1_tilde(2, 0);
  x2_xi1_tilde(0, 1) = -x2_xi1_tilde(1, 0);
  Core::LinAlg::Matrix<3, 3, TYPEBTS> x2_xi2_tilde(true);
  x2_xi2_tilde(2, 1) = x2_xi2(0);
  x2_xi2_tilde(2, 0) = -x2_xi2(1);
  x2_xi2_tilde(1, 0) = x2_xi2(2);
  x2_xi2_tilde(1, 2) = -x2_xi2_tilde(2, 1);
  x2_xi2_tilde(0, 2) = -x2_xi2_tilde(2, 0);
  x2_xi2_tilde(0, 1) = -x2_xi2_tilde(1, 0);

  // First compute compute directional derivative of tangent cross product a2
  Core::LinAlg::Matrix<3, dim1 + dim2, TYPEBTS> a2_d(true);
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        a2_d(i, j) +=
            (x2_xi1_tilde(i, k) * x2_xi2xi1(k) - x2_xi2_tilde(i, k) * x2_xi1xi1(k)) * xi1_d(j) +
            (x2_xi1_tilde(i, k) * x2_xi2xi2(k) - x2_xi2_tilde(i, k) * x2_xi1xi2(k)) * xi2_d(j);
      }
    }
  }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim2; j++)
      for (int k = 0; k < 3; k++)
        a2_d(i, j + dim1) += x2_xi1_tilde(i, k) * N2_xi2(k, j) - x2_xi2_tilde(i, k) * N2_xi1(k, j);

  // Compute auxiliary_matrix
  auxiliary_matrix.clear();
  for (int i = 0; i < 3; i++)
  {
    auxiliary_matrix(i, i) += 1.0 / norm_a2;
    for (int j = 0; j < 3; j++) auxiliary_matrix(i, j) += -n2(i) * n2(j) / norm_a2;
  }

  // Finally compute directional derivative of surface unit normal vector n2 (new linearization)
  n2_d.clear();
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < dim1 + dim2; k++) n2_d(i, k) += auxiliary_matrix(i, j) * a2_d(j, k);
  // std::cout << "n2_d (new linearization): " << n2_d << std::endl;

  return;
}
/*----------------------------------------------------------------------*
 | end: Compute directional derivative of surface unit normal vector    |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Assemble contact force and stiffness                                 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::
    assemble_fc_and_stiffc_contact(
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS> fc1,
        const Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS> fc2,
        Core::LinAlg::Vector<double>* fint,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
            3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>
            stiffc1,
        const Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
            TYPEBTS>
            stiffc2,
        Core::LinAlg::SparseMatrix& stiffmatrix)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Node ids of elements
  const int* node_ids1 = element1_->node_ids();
  const int* node_ids2 = element2_->node_ids();

  // Temporary vectors for contact forces, DOF-GIDs and owning procs
  Core::LinAlg::SerialDenseVector fcontact1(dim1);
  Core::LinAlg::SerialDenseVector fcontact2(dim2);
  std::vector<int> lm1(dim1);
  std::vector<int> lm2(dim2);
  std::vector<int> lmowner1(dim1);
  std::vector<int> lmowner2(dim2);

  // Temporary matrices for stiffness and vectors for DOF-GIDs and owning procs
  Core::LinAlg::SerialDenseMatrix stiffcontact1(dim1, dim1 + dim2);
  Core::LinAlg::SerialDenseMatrix stiffcontact2(dim2, dim1 + dim2);
  std::vector<int> lmrow1(dim1);
  std::vector<int> lmrow2(dim2);
  std::vector<int> lmrowowner1(dim1);
  std::vector<int> lmrowowner2(dim2);
  std::vector<int> lmcol1(dim1 + dim2);
  std::vector<int> lmcol2(dim1 + dim2);


  // Assemble contact forces acting on beam and solid
  // -----------------------------------------------------------------

  // Prepare assembly

  // Fill lm1 and lmowner1
  for (int i = 0; i < numnodes; ++i)
  {
    // Get node pointer and dof ids
    Core::Nodes::Node* node = contact_discret().g_node(node_ids1[i]);
    std::vector<int> NodeDofGIDs = get_global_dofs(node);

    for (int j = 0; j < 3 * numnodalvalues; ++j)
    {
      lm1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      lmowner1[3 * numnodalvalues * i + j] = node->owner();
    }
  }

  // Fill lm2 and lmowner2
  for (int i = 0; i < numnodessol; ++i)
  {
    // Get node pointer and dof ids
    Core::Nodes::Node* node = contact_discret().g_node(node_ids2[i]);
    std::vector<int> NodeDofGIDs = get_global_dofs(node);

    for (int j = 0; j < 3; ++j)
    {
      lm2[3 * i + j] = NodeDofGIDs[j];
      lmowner2[3 * i + j] = node->owner();
    }
  }

  // Assemble fc1 and fc2 into global contact force vector
  for (int i = 0; i < dim1; i++) fcontact1[i] = Core::FADUtils::cast_to_double(fc1(i));
  for (int i = 0; i < dim2; i++) fcontact2[i] = Core::FADUtils::cast_to_double(fc2(i));

  Core::LinAlg::assemble(*fint, fcontact1, lm1, lmowner1);
  Core::LinAlg::assemble(*fint, fcontact2, lm2, lmowner2);

  // -----------------------------------------------------------------
  // End: Assemble contact forces acting on beam and solid


  // Assemble contact stiffness for beam and solid
  // -----------------------------------------------------------------

  // Prepare assembly

  // Fill lmrow1 and lmrowowner1
  lmrow1 = lm1;
  lmrowowner1 = lmowner1;

  // Fill lmrow2 and lmrowowner2
  lmrow2 = lm2;
  lmrowowner2 = lmowner2;

  // Fill lmcol1 and lmcol2
  for (int i = 0; i < numnodes; ++i)
  {
    // Get pointer and node ids
    Core::Nodes::Node* node = contact_discret().g_node(node_ids1[i]);
    std::vector<int> NodeDofGIDs = get_global_dofs(node);

    for (int j = 0; j < 3 * numnodalvalues; ++j)
    {
      lmcol1[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
      lmcol2[3 * numnodalvalues * i + j] = NodeDofGIDs[j];
    }
  }
  for (int i = 0; i < numnodessol; ++i)
  {
    // Get pointer and node ids
    Core::Nodes::Node* node = contact_discret().g_node(node_ids2[i]);
    std::vector<int> NodeDofGIDs = get_global_dofs(node);

    for (int j = 0; j < 3; ++j)
    {
      lmcol1[dim1 + 3 * i + j] = NodeDofGIDs[j];
      lmcol2[dim1 + 3 * i + j] = NodeDofGIDs[j];
    }
  }

  // Now finally assemble stiffc1 and stiffc2
  for (int j = 0; j < dim1 + dim2; j++)
  {
    // Change sign of stiffc1 and stiffc2 due to time integration. According to analytical
    // derivation there is no minus sign, but for our time integration methods the negative
    // stiffness must be assembled.
    for (int i = 0; i < dim1; i++)
      stiffcontact1(i, j) = -Core::FADUtils::cast_to_double(stiffc1(i, j));
    for (int i = 0; i < dim2; i++)
      stiffcontact2(i, j) = -Core::FADUtils::cast_to_double(stiffc2(i, j));
  }

  stiffmatrix.assemble(stiffcontact1, lmrow1, lmrowowner1, lmcol1);
  stiffmatrix.assemble(stiffcontact2, lmrow2, lmrowowner2, lmcol2);

  // ----------------------------------------------------------------
  // End: Assemble contact stiffness for beam and solid

  // Print some debug information
  const bool output = false;
  if (output)
  {
    std::cout << "element1_->Id(): " << element1_->id() << std::endl;

    std::cout << "lmrow1: " << std::endl;
    for (int i = 0; i < dim1; i++) std::cout << lmrow1[i] << std::endl;

    std::cout << "lmcol1: " << std::endl;
    for (int i = 0; i < dim1 + dim2; i++) std::cout << lmcol1[i] << std::endl;

    std::cout << "element2_->Id(): " << element2_->id() << std::endl;

    std::cout << "lmrow2: " << std::endl;
    for (int i = 0; i < dim2; i++) std::cout << lmrow2[i] << std::endl;

    std::cout << "lmcol2: " << std::endl;
    for (int i = 0; i < dim1 + dim2; i++) std::cout << lmcol2[i] << std::endl;
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Assemble contact force and stiffness                            |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Get contact interval borders                                         |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::
    get_contact_interval_borders(
        std::vector<std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>>& parsets)
{
  // Enable or disable console output
  const bool output = false;
  if (output)
  {
    std::cout << "get_contact_interval_borders output:" << std::endl;
  }

  // Initialize limit for parameter values (interval [-limit, limit])
  const double limit = 1.0 + XIETATOL;

  // Clear and resize current vector for parameter sets
  parsets.clear();
  parsets.resize(0);

  // Temporary vector containg parsets and flags for allowed or not allowed projections
  // TODO: May this can be done in a more beautiful way without using the temporary vector
  // parsetstmp
  std::vector<std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, Core::LinAlg::Matrix<2, 1, int>>>
      parsetstmp;
  std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, Core::LinAlg::Matrix<2, 1, int>> parsettmp;

  // Initialize flags indicating if the projection of beam start and end point are necessary
  bool projbeamstart = true;
  bool projbeamend = true;


  // Find projection of surface edges xi1 = +-1 and xi2 = +-1
  // Only implemented for quad surface elements such as quad4, quad8
  // TODO: Implementation for tri surface elements
  // -----------------------------------------------------------------

  if (output)
  {
    std::cout << "Projection surface edges:" << std::endl;
  }

  // Fixed parameter xi1 or xi2 (i = 0: xi1 fixed, i = 1: xi2 fixed)
  for (int i = 0; i < 2; i++)
  {
    // Value for xi1 or xi2 (j = 0: xi = -1, j = 1: xi = +1)
    for (int j = 0; j < 2; j++)
    {
      // Set start point for Newton iteration
      TYPEBTS xi1 = 0.0;
      TYPEBTS xi2 = 0.0;
      TYPEBTS eta = 0.0;

      // Set value for fixed parameter xi1 or xi2
      switch (i)
      {
        case 0:
          xi1 = -1.0 + 2.0 * j;
          break;
        case 1:
          xi2 = -1.0 + 2.0 * j;
          break;
      }

      // Projection of surface edge
      bool proj_allowed = true;
      projection(i, xi1, xi2, eta, proj_allowed);

      if (output)
      {
        std::cout << "  xi1: " << xi1 << ", xi2: " << xi2 << ", eta: " << eta
                  << ", fixed par: " << i << ", proj_allowed: " << proj_allowed << std::endl;
      }

      // Check if the parameter eta found within this projection is already a contact interval
      // border, e.g. if the projected beam element centerline crosses an surface edge point
      bool alreadyfound = false;
      for (int k = 0; k < (int)parsetstmp.size(); k++)
      {
        if (fabs(parsetstmp[k].first(2) - eta) < XIETATOL)
        {
          alreadyfound = true;
          break;
        }
      }

      if (!alreadyfound && fabs(xi1) < limit && fabs(xi2) < limit && fabs(eta) < limit)
      {
        // Insert contact interval border parameter set
        parsettmp.first(0) = xi1;
        parsettmp.first(1) = xi2;
        parsettmp.first(2) = eta;
        parsettmp.second(0) = i;
        parsettmp.second(1) = proj_allowed;
        parsetstmp.push_back(parsettmp);

        // Check if the projection of the beam start and end point is needed
        if (fabs(eta + 1.0) < XIETATOL)
        {
          // Start point projection not needed
          projbeamstart = false;
        }
        if (fabs(eta - 1.0) < XIETATOL)
        {
          // End point projection not needed
          projbeamend = false;
        }
      }
    }
  }

  // Sort vector with parameter sets from smallest to biggest eta
  std::sort(parsetstmp.begin(), parsetstmp.end(), compare_parsets);

  // -----------------------------------------------------------------
  // End: Find projection of surface edges xi1 = +-1 and xi2 = +-1


  // Find projection of beam start and end point eta = +-1
  // -----------------------------------------------------------------

  if (output)
  {
    std::cout << "Projection of beam start and end point:" << std::endl;
  }

  // Check if the projected start point of beam (eta = -1) lies inside the surface
  if (projbeamstart)
  {
    // Set start point for newton iteration
    TYPEBTS xi1 = 0.0;
    TYPEBTS xi2 = 0.0;

    // Set beam parameter
    TYPEBTS eta = -1.0;

    // Projection of beam start point
    bool proj_allowed = true;
    projection(2, xi1, xi2, eta, proj_allowed);

    if (fabs(xi1) < limit && fabs(xi2) < limit)
    {
      // Insert contact interval border parameter set at first position
      parsettmp.first(0) = xi1;
      parsettmp.first(1) = xi2;
      parsettmp.first(2) = eta;
      parsettmp.second(0) = 2;
      parsettmp.second(1) = proj_allowed;
      parsetstmp.insert(parsetstmp.begin(), parsettmp);
      if (output)
      {
        std::cout << "  xi1: " << xi1 << ", xi2: " << xi2 << ", eta: " << eta
                  << ", fixed par: " << 2 << ", proj_allowed: " << proj_allowed
                  << "(beam start point)" << std::endl;
      }
    }
  }

  // Check if projected end point of beam (eta = 1) lies inside the surface
  if (projbeamend)
  {
    // Set start point for newton iteration
    TYPEBTS xi1 = 0.0;
    TYPEBTS xi2 = 0.0;

    // Set beam parameter
    TYPEBTS eta = 1.0;

    // Projection of beam end point
    bool proj_allowed = true;
    projection(2, xi1, xi2, eta, proj_allowed);

    if (fabs(xi1) < limit && fabs(xi2) < limit)
    {
      // Insert contact interval border parameter set at last position
      parsettmp.first(0) = xi1;
      parsettmp.first(1) = xi2;
      parsettmp.first(2) = eta;
      parsettmp.second(0) = 2;
      parsettmp.second(1) = proj_allowed;
      parsetstmp.push_back(parsettmp);
      if (output)
      {
        std::cout << "  xi1: " << xi1 << ", xi2: " << xi2 << ", eta: " << eta
                  << ", fixed par: " << 2 << ", proj_allowed: " << proj_allowed
                  << "(beam end point)" << std::endl;
      }
    }
  }

  // -----------------------------------------------------------------
  // End: Find projection of beam start and end point eta = +-1


  // Sort contact interval borders to identify contact intervals
  // -----------------------------------------------------------------

  // Print sorted vector with all parameter sets
  if (output)
  {
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Number of parameter sets found for this beam to solid contact pair: "
              << parsetstmp.size() << std::endl;
    if (parsetstmp.size() > 0)
    {
      for (int i = 0; i < (int)parsetstmp.size(); i++)
      {
        std::cout << "  xi1: " << parsetstmp[i].first(0) << ", xi2: " << parsetstmp[i].first(1)
                  << ", eta: " << parsetstmp[i].first(2)
                  << ", fixed par: " << parsetstmp[i].second(0)
                  << ", proj_allowed: " << parsetstmp[i].second(1) << std::endl;
      }
    }
  }

  // Check if all found contact intervals are closed
  if (parsetstmp.size() % 2 != 0)
  {
    // TODO: Point contact
    if (parsetstmp.size() == 1)
    {
      parsetstmp.clear();
      parsetstmp.resize(0);
    }
    else
    {
      // std::cout << "Contact interval not closed" << std::endl;
      FOUR_C_THROW("Contact interval not closed.");
    }
  }

  // Insert parsetstmp in parsets and sort out contact interval border parameter sets with not
  // allowed projection
  std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int> parset;
  for (int i = 0; i < (int)parsetstmp.size() - 1; i += 2)
  {
    // Check if the contact interval is allowed
    if (parsetstmp[i].second(1) && parsetstmp[i + 1].second(1))
    {
      for (int j = 0; j < 2; j++)
      {
        parset.first = parsetstmp[i + j].first;
        parset.second = parsetstmp[i + j].second(0);
        parsets.push_back(parset);
      }
    }
  }

  // Print final sorted vector with parameter sets
  if (output)
  {
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Final number of parameter sets allowed for this beam to solid contact pair: "
              << parsets.size() << std::endl;
    if (parsets.size() > 0)
    {
      for (int i = 0; i < (int)parsets.size(); i++)
      {
        std::cout << "  xi1: " << parsets[i].first(0) << ", xi2: " << parsets[i].first(1)
                  << ", eta: " << parsets[i].first(2) << ", fixed par: " << parsets[i].second
                  << std::endl;
      }
    }
    std::cout << std::endl;
  }

  // -----------------------------------------------------------------
  // End: Sort contact interval borders to identify contact intervals

  return;
}
/*----------------------------------------------------------------------*
 | End: Get contact interval borders                                    |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Projection                                                           |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::projection(
    const int& fixed_par, TYPEBTS& xi1, TYPEBTS& xi2, TYPEBTS& eta, bool& proj_allowed)
{
  proj_allowed = true;
  bool parallel = false;

  // Initialize limit for parameter values (interval [-limit, limit])
  const double limit = 1.0 + XIETATOL;

  const bool output = false;
  if (output)
  {
    std::cout << "Projection output:" << std::endl;
    std::cout << "Start parameters xi1: " << xi1 << ", xi2: " << xi2 << ", eta: " << eta
              << " and fixed_par: " << fixed_par << std::endl;
  }

  // Vectors for shape functions and their derivatives
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS> N1(true);         // = N1
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS> N1_eta(true);     // = N1,eta
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS> N1_etaeta(true);  // = N1,etaeta

  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2(true);         // = N2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi1(true);     // = N2,xi1
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi2(true);     // = N2,xi2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi1xi1(true);  // = N2,xi1xi1
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi2xi2(true);  // = N2,xi2xi2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi1xi2(true);  // = N2,xi1xi2
  Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS> N2_xi2xi1(true);  // = N2,xi2xi1

  // Coords and derivatives of beam and surface element
  Core::LinAlg::Matrix<3, 1, TYPEBTS> r1(true);         // = r1
  Core::LinAlg::Matrix<3, 1, TYPEBTS> r1_eta(true);     // = r1,eta
  Core::LinAlg::Matrix<3, 1, TYPEBTS> r1_etaeta(true);  // = r1,etaeta

  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2(true);         // = x2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi1(true);     // = x2,xi1
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi2(true);     // = x2,xi2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi1xi1(true);  // = x2,xi1xi1
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi2xi2(true);  // = x2,xi2xi2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi1xi2(true);  // = x2,xi1xi2
  Core::LinAlg::Matrix<3, 1, TYPEBTS> x2_xi2xi1(true);  // = x2,xi2xi1

  // Distance vector, its norm and unit distance vector
  Core::LinAlg::Matrix<3, 1, TYPEBTS> rD(true);  // = r1 - x2
  TYPEBTS norm_rD = 0.0;                         // = ||rD||
  Core::LinAlg::Matrix<3, 1, TYPEBTS> nD(true);  // = rD / norm_rD

  // Surface tangent cross product, its norm and unit surface normal vector
  Core::LinAlg::Matrix<3, 1, TYPEBTS> a2(true);  // = x2_xi1 x x2_xi2
  TYPEBTS norm_a2 = 0.0;                         // = ||a||
  Core::LinAlg::Matrix<3, 1, TYPEBTS> n2(true);  // = a / norm_a


  // Initialize function f and Jacobian J for Newton iteration
  Core::LinAlg::Matrix<2, 1, TYPEBTS> f(true);
  Core::LinAlg::Matrix<2, 3, TYPEBTS> Jpar(true);
  Core::LinAlg::Matrix<2, 2, TYPEBTS> J(true);
  Core::LinAlg::Matrix<2, 2, TYPEBTS> Jinv(true);

  // Initialize vector with parameters xi1, xi2, eta
  std::array<TYPEBTS, 3> par = {xi1, xi2, eta};

  // Get indices of not fixed parameters
  int par_i[2];
  int j = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != fixed_par)
    {
      par_i[j] = i;
      j++;
    }
  }

#ifdef FADCHECKLINORTHOGONALITYCONDITION
  // Print information about current parameters
  std::cout << std::endl
            << "FAD-Check: Linearization of orthogonality condition with start parameters "
               "xi1: "
            << xi1.val() << ", xi2: " << xi2.val() << ", eta: " << eta.val()
            << " and fixed_par: " << fixed_par << std::endl;
#endif

  // Initial scalar residual (L2-norm of f)
  TYPEBTS residual = 0.0;


  // Local netwon iteration
  // -----------------------------------------------------------------

  int iter;
  for (iter = 0; iter < BEAMCONTACTMAXITER; iter++)
  {
#ifdef FADCHECKLINORTHOGONALITYCONDITION
    // Set known parameters xi1, xi2, eta and element positions as primary variables for checking
    // the linearization of the orthogonality conditions (Jacobi-matrix J) borders with FAD
    BeamContact::SetFADParCoordDofs<numnodessol, numnodes, numnodalvalues>(xi1, xi2, eta);
    BeamContact::SetFADDispDofs<numnodessol, numnodes, numnodalvalues>(ele1pos_, ele2pos_, 3);
#endif

    // Update shape functions and their derivatives for beam and surface element
    get_beam_shape_functions(N1, N1_eta, N1_etaeta, eta);
    get_surf_shape_functions(
        N2, N2_xi1, N2_xi2, N2_xi1xi1, N2_xi2xi2, N2_xi1xi2, N2_xi2xi1, xi1, xi2);

    // Update coordinates and derivatives for beam and surface element
    compute_beam_coords_and_derivs(r1, r1_eta, r1_etaeta, N1, N1_eta, N1_etaeta);
    compute_surf_coords_and_derivs(x2, x2_xi1, x2_xi2, x2_xi1xi1, x2_xi2xi2, x2_xi1xi2, x2_xi2xi1,
        N2, N2_xi1, N2_xi2, N2_xi1xi1, N2_xi2xi2, N2_xi1xi2, N2_xi2xi1);

    // Compute distance vector rD = r1 - x2
    for (int i = 0; i < 3; i++)
    {
      rD(i) = r1(i) - x2(i);
    }

    // Compute norm of difference vector
    norm_rD = Core::FADUtils::vector_norm<3>(rD);

    // If automatic differentiation via FAD is applied, norm_rD has to be of type double since
    // this factor is needed for a pure scaling of the orthogonality conditions and has not to be
    // linearized
    double norm_rD_scale = Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm<3>(rD));

    // The closer the beam and surface element get, the smaller is norm_rD, but norm_rD is not
    // allowed to be too small, else numerical problems occur. Since in this case |eta| > 1,
    // |xi1| > 1 or |xi2| > 1 they will be sorted out later anyways.
    if (norm_rD_scale < NORMTOL)
    {
      break;
    }

    // Evaluate f at current xi1, xi2, eta
    f.clear();
    for (int i = 0; i < 3; i++)
    {
      f(0) += rD(i) * x2_xi1(i) / norm_rD_scale;
      f(1) += rD(i) * x2_xi2(i) / norm_rD_scale;
    }

    // Compute scalar residuum
    residual = Core::FADUtils::vector_norm<2>(f);

    // Reset matrices
    J.clear();
    Jpar.clear();
    Jinv.clear();

    // Evaluate derivatives of f at current xi1, xi2, eta
    for (int i = 0; i < 3; i++)
    {
      if (fixed_par != 0)  // xi1 is not fixed
      {
        // xi1 derivate of f
        Jpar(0, 0) += (-x2_xi1(i) * x2_xi1(i) + rD(i) * x2_xi1xi1(i)) / norm_rD_scale;
        Jpar(1, 0) += (-x2_xi1(i) * x2_xi2(i) + rD(i) * x2_xi2xi1(i)) / norm_rD_scale;
      }
      if (fixed_par != 1)  // xi2 is not fixed
      {
        // xi2 derivate of f
        Jpar(0, 1) += (-x2_xi2(i) * x2_xi1(i) + rD(i) * x2_xi1xi2(i)) / norm_rD_scale;
        Jpar(1, 1) += (-x2_xi2(i) * x2_xi2(i) + rD(i) * x2_xi2xi2(i)) / norm_rD_scale;
      }
      if (fixed_par != 2)  // eta is not fixed
      {
        // eta derivate of f
        Jpar(0, 2) += r1_eta(i) * x2_xi1(i) / norm_rD_scale;
        Jpar(1, 2) += r1_eta(i) * x2_xi2(i) / norm_rD_scale;
      }
    }

    // Assemble needed derivatives of f in J
    J(0, 0) = Jpar(0, par_i[0]);
    J(1, 0) = Jpar(1, par_i[0]);
    J(0, 1) = Jpar(0, par_i[1]);
    J(1, 1) = Jpar(1, par_i[1]);

#ifdef FADCHECKLINORTHOGONALITYCONDITION
    // Print information about current iteration
    std::cout << "Local Newton iteration " << iter + 1 << ":" << std::endl;

    // Check linearization of orthogonality conditions with FAD
    Core::LinAlg::Matrix<2, 2, TYPEBTS> J_FAD(true);
    fad_check_lin_orthogonality_condition(fixed_par, rD, x2_xi1, x2_xi2, J_FAD, J);
#endif

    // Inverting 2x2-matrix J by hard coded formula, so that it is possible
    // to handle colinear vectors, because they lead to det(J) = 0
    TYPEBTS det_J = J(0, 0) * J(1, 1) - J(1, 0) * J(0, 1);

    // If det_J = 0 we assume, that the beam centerline and the surface edge are parallel.
    // These projection is not needed due the fact that the contact interval can also be
    // identified by two contact interval borders found with the GetContactLines method
    parallel = Core::FADUtils::cast_to_double(Core::FADUtils::norm(det_J)) < COLINEARTOL;

    // Check if the local Newton iteration has converged
    // If the start point fulfills the orthogonalty conditions (residual < BEAMCONTACTTOL), we also
    // check if the beam centerline and the surface edge are parallel. This is done by calculating
    // det_J before checking if the local Newton iteration has converged by fulfilling the condition
    // residual < BEAMCONTACTTOL
    if (Core::FADUtils::cast_to_double(residual) < BEAMCONTACTTOL && !parallel)
    {
      if (output)
      {
        std::cout << "Local Newton iteration converged after " << iter << " iterations"
                  << std::endl;
        std::cout << "Found point at xi1: " << xi1 << ", xi2: " << xi2 << ", eta: " << eta
                  << " with residual: " << residual << std::endl;
      }
      // Local Newton iteration converged
      break;
    }
    else if (output && iter > 0)
    {
      std::cout << "New point at xi1: " << xi1 << ", xi2: " << xi2 << ", eta: " << eta
                << " with residual: " << residual << std::endl;
    }

    // Singular J
    if (parallel)
    {
      // Sort out
      if (output)
      {
        std::cout << "elementscolinear: det_J = "
                  << Core::FADUtils::cast_to_double(Core::FADUtils::norm(det_J)) << std::endl;
      }
      break;
    }
    // Regular J (inversion possible)
    else
    {
      // Do not sort out

      // invert J
      Jinv(0, 0) = J(1, 1) / det_J;
      Jinv(0, 1) = -J(0, 1) / det_J;
      Jinv(1, 0) = -J(1, 0) / det_J;
      Jinv(1, 1) = J(0, 0) / det_J;
    }

    par[par_i[0]] += -Jinv(0, 0) * f(0) - Jinv(0, 1) * f(1);
    par[par_i[1]] += -Jinv(1, 0) * f(0) - Jinv(1, 1) * f(1);

    xi1 = par[0];
    xi2 = par[1];
    eta = par[2];
  }
  // -----------------------------------------------------------------
  // End: Local Newton iteration


  // Local Newton iteration unconverged after BEAMCONTACTMAXITER
  if (residual > BEAMCONTACTTOL || parallel)
  {
    par[par_i[0]] = 1e+12;
    par[par_i[1]] = 1e+12;

    xi1 = par[0];
    xi2 = par[1];
    eta = par[2];

    if (output)
      std::cout << "Local Newton iteration unconverged (!) after " << iter + 1 << " iterations"
                << std::endl;
  }
  else if (fabs(xi1) < limit && fabs(xi2) < limit && fabs(eta) < limit)
  {
    Core::LinAlg::Matrix<3, 1, TYPEBTS> n2(true);
    double sgn = 1.0;

    // Compute distance vector rD, its norm norm_rD and unit distance vector nD
    compute_distance_normal(r1, x2, rD, norm_rD, nD);

    // Compute surface tangent cross product a2, its norm_a2 and surface unit normal vector n2
    compute_surface_normal(x2_xi1, x2_xi2, a2, norm_a2, n2);

    // Compute sign of projection of distance vector rD on surface unit normal vector n2
    sgn = Core::FADUtils::cast_to_double(
        Core::FADUtils::signum(Core::FADUtils::scalar_product(nD, n2)));

    if (sgn < 0)
    {
      // If there is no information about the unit distance vector nD of the last time step (e.g. in
      // the first call after the contact pair element has been created), there is no change of the
      // dircetion of nD
      bool normaldir_changed = false;

      // Otherwise check if the direction of nD has changed as follows
      if (normalsets_old_.size() > 0)
      {
        // Find the nearest eta from the last time step to the current eta
        int nearest_index = 0;
        if (normalsets_old_.size() > 1)
        {
          int lower_index = std::lower_bound(normalsets_old_.begin(), normalsets_old_.end(), eta,
                                compare_normalsets) -
                            normalsets_old_.begin();
          lower_index = std::max(1, std::min(lower_index, (int)normalsets_old_.size() - 1));
          if (fabs(normalsets_old_[lower_index].first - eta) <
              fabs(normalsets_old_[lower_index - 1].first - eta))
          {
            nearest_index = lower_index;
          }
          else
          {
            nearest_index = lower_index - 1;
          }
        }

        // Check if the direction of nD has changed
        if (Core::FADUtils::scalar_product(nD, normalsets_old_[nearest_index].second) < 0)
        {
          normaldir_changed = true;
        }
      }

      // If there is no a change of the direction of the normal vector, the projection is not
      // allowed and the found parameters are not treated in the later caluclation
      if (!normaldir_changed)
      {
        proj_allowed = false;
      }
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Projection                                                      |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Evaluate beam shape functions and derivatives                        |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::get_beam_shape_functions(
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_eta,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_etaeta, const TYPEBTS& eta)
{
  // Clear shape functions and derivatives
  N.clear();
  N_eta.clear();
  N_etaeta.clear();

  // Get discretization type
  const Core::FE::CellType distype = element1_->shape();

  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS> N_i(true);
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS> N_i_eta(true);
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS> N_i_etaeta(true);

  if (numnodalvalues == 1)
  {
    // Get values and derivatives of shape functions
    Core::FE::shape_function_1d(N_i, eta, distype);
    Core::FE::shape_function_1d_deriv1(N_i_eta, eta, distype);
    Core::FE::shape_function_1d_deriv2(N_i_etaeta, eta, distype);
  }
  else if (numnodalvalues == 2)
  {
    if (element1_->element_type() != Discret::Elements::Beam3ebType::instance())
      FOUR_C_THROW("Only elements of type Beam3eb are valid for the case numnodalvalues=2!");

    double length = 2 * (static_cast<Discret::Elements::Beam3eb*>(element1_))->jacobi();

    // Get values and derivatives of shape functions
    Core::FE::shape_function_hermite_1d(N_i, eta, length, distype);
    Core::FE::shape_function_hermite_1d_deriv1(N_i_eta, eta, length, distype);
    Core::FE::shape_function_hermite_1d_deriv2(N_i_etaeta, eta, length, distype);
  }
  else
    FOUR_C_THROW(
        "Only beam elements with one (nodal positions) or two (nodal positions + nodal tangents) "
        "values are valid!");

  // Assemble the individual shape functions in matrices, such that: r = N * d, r_eta = N_eta * d,
  // r_etaeta = N_etaeta * d
  assemble_beam_shapefunctions(N_i, N_i_eta, N_i_etaeta, N, N_eta, N_etaeta);

  return;
}
/*----------------------------------------------------------------------*
 | End: Evaluate beam shape functions and derivatives                   |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Evaluate solid surface shape functions and derivatives               |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::get_surf_shape_functions(
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi1,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi2,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi2,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi1, const TYPEBTS& xi1,
    const TYPEBTS& xi2)
{
  // Clear shape functions and derivatives
  N.clear();
  N_xi1.clear();
  N_xi2.clear();
  N_xi1xi1.clear();
  N_xi2xi2.clear();
  N_xi1xi2.clear();
  N_xi2xi1.clear();

  Core::LinAlg::Matrix<1, numnodessol, TYPEBTS> N_i(true);
  Core::LinAlg::Matrix<2, numnodessol, TYPEBTS> N_i_xi(true);
  Core::LinAlg::Matrix<3, numnodessol, TYPEBTS> N_i_xixi(true);

  Core::FE::shape_function_2d(N_i, xi1, xi2, element2_->shape());
  Core::FE::shape_function_2d_deriv1(N_i_xi, xi1, xi2, element2_->shape());
  Core::FE::shape_function_2d_deriv2(N_i_xixi, xi1, xi2, element2_->shape());

  // Assemble the individual shape functions in matrices, such that: r = N * d, r_xi = N_xi * d,
  // r_xi1xi1 = N_xi1xi1 * d, ...
  assemble_surf_shapefunctions(
      N_i, N_i_xi, N_i_xixi, N, N_xi1, N_xi2, N_xi1xi1, N_xi2xi2, N_xi1xi2, N_xi2xi1);

  return;
}
/*----------------------------------------------------------------------*
 | End: Evaluate solid surface shape functions and derivatives          |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Assemble beam shape functions                                        |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::assemble_beam_shapefunctions(const Core::LinAlg::Matrix<1,
                                                      numnodes * numnodalvalues, TYPEBTS>& N_i,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS>& N_i_eta,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, TYPEBTS>& N_i_etaeta,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_eta,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_etaeta)
{
  // assembly_N is just an array to help assemble the matrices of the shape functions
  // it determines, which shape function is used in which column of N
  std::array<std::array<unsigned int, 3 * numnodes * numnodalvalues>, 3> assembly_N{};

  // Set number of shape functions for each 3x3 block:
  // e.g. second order Reissner beam (numnodes = 3, numnodalvalues = 1)
  // int assembly_N[3][9] = {{1,0,0,2,0,0,3,0,0},
  //                         {0,1,0,0,2,0,0,3,0},
  //                         {0,0,1,0,0,2,0,0,3}};
  //
  // e.g. Kirchhoff beam (numnodes = 2, numnodalvalues = 2)
  // int assembly_N[3][12] = {{1,0,0,2,0,0,3,0,0,4,0,0},
  //                          {0,1,0,0,2,0,0,3,0,0,4,0},
  //                          {0,0,1,0,0,2,0,0,3,0,0,4}};

  for (int i = 0; i < numnodes * numnodalvalues; i++)
  {
    assembly_N[0][3 * i] = i + 1;
    assembly_N[1][3 * i + 1] = i + 1;
    assembly_N[2][3 * i + 2] = i + 1;
  }

  // Assemble the matrices of the shape functions
  for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (assembly_N[j][i] == 0)
      {
        N(j, i) = 0;
        N_eta(j, i) = 0;
        N_etaeta(j, i) = 0;
      }
      else
      {
        int k = assembly_N[j][i] - 1;
        N(j, i) = N_i(k);
        N_eta(j, i) = N_i_eta(k);
        N_etaeta(j, i) = N_i_etaeta(k);
      }
    }
  }
  return;
}
/*----------------------------------------------------------------------*
 | End: Assemble beam shape functions                                   |
 *----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*
 | Assemble solid surface shape functions                               |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::
    assemble_surf_shapefunctions(const Core::LinAlg::Matrix<1, numnodessol, TYPEBTS>& N_i,
        const Core::LinAlg::Matrix<2, numnodessol, TYPEBTS>& N_i_xi,
        const Core::LinAlg::Matrix<3, numnodessol, TYPEBTS>& N_i_xixi,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi1,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi2,
        Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi1)
{
  // assembly_N is just an array to help assemble the matrices of the shape functions
  // it determines, which shape function is used in which column of N
  std::array<std::array<int, 3 * numnodessol>, 3> assembly_N{};

  // Set number of shape functions for each 3x3 block:
  // e.g. quad4 surface element (numnodesol = 4)
  // int assembly_N[3][12] = {{1,0,0,2,0,0,3,0,0,4,0,0},
  //                          {0,1,0,0,2,0,0,3,0,0,4,0},
  //                          {0,0,1,0,0,2,0,0,3,0,0,4}};

  for (int i = 0; i < numnodessol; i++)
  {
    assembly_N[0][3 * i] = i + 1;
    assembly_N[1][3 * i + 1] = i + 1;
    assembly_N[2][3 * i + 2] = i + 1;
  }

  // Assemble the matrices of the shape functions
  for (int i = 0; i < 3 * numnodessol; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (assembly_N[j][i] == 0)
      {
        N(j, i) = 0;
        N_xi1(j, i) = 0;
        N_xi2(j, i) = 0;
        N_xi1xi1(j, i) = 0;
        N_xi2xi2(j, i) = 0;
        N_xi1xi2(j, i) = 0;
        N_xi2xi1(j, i) = 0;
      }
      else
      {
        int k = assembly_N[j][i] - 1;
        N(j, i) = N_i(k);
        N_xi1(j, i) = N_i_xi(0, k);
        N_xi2(j, i) = N_i_xi(1, k);
        N_xi1xi1(j, i) = N_i_xixi(0, k);
        N_xi2xi2(j, i) = N_i_xixi(1, k);
        N_xi1xi2(j, i) = N_i_xixi(2, k);
        N_xi2xi1(j, i) = N_i_xixi(2, k);  // = N_xi1xi2
      }
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Assemble solid surface shape functions                          |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Compute beam point coordinates and their derivatives                 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::compute_beam_coords_and_derivs(Core::LinAlg::Matrix<3, 1, TYPEBTS>& r,
    Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_eta, Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_etaeta,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_eta,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, TYPEBTS>& N_etaeta)
{
  r.clear();
  r_eta.clear();
  r_etaeta.clear();

  // Compute output variable
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3 * numnodes * numnodalvalues; j++)
    {
      r(i) += N(i, j) * ele1pos_(j);
      r_eta(i) += N_eta(i, j) * ele1pos_(j);
      r_etaeta(i) += N_etaeta(i, j) * ele1pos_(j);
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute beam contact point coordinates and their derivative     |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Compute solid surface point coordinates and their derivatives        |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::compute_surf_coords_and_derivs(Core::LinAlg::Matrix<3, 1, TYPEBTS>& r,
    Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi1, Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi2,
    Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi1xi1, Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi2xi2,
    Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi1xi2, Core::LinAlg::Matrix<3, 1, TYPEBTS>& r_xi2xi1,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi1,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi2,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi1xi2,
    Core::LinAlg::Matrix<3, 3 * numnodessol, TYPEBTS>& N_xi2xi1)
{
  r.clear();
  r_xi1.clear();
  r_xi2.clear();
  r_xi1xi1.clear();
  r_xi2xi2.clear();
  r_xi1xi2.clear();
  r_xi2xi1.clear();

  // Compute output variable
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3 * numnodessol; j++)
    {
      r(i) += N(i, j) * ele2pos_(j);
      r_xi1(i) += N_xi1(i, j) * ele2pos_(j);
      r_xi2(i) += N_xi2(i, j) * ele2pos_(j);
      r_xi1xi1(i) += N_xi1xi1(i, j) * ele2pos_(j);
      r_xi2xi2(i) += N_xi2xi2(i, j) * ele2pos_(j);
      r_xi1xi2(i) += N_xi1xi2(i, j) * ele2pos_(j);
      r_xi2xi1(i) += N_xi2xi1(i, j) * ele2pos_(j);
    }
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute solid surface point coordinates and their derivatives   |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Compute distance vector                                              |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::compute_distance_normal(
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& r1, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2,
    Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD, TYPEBTS& norm_rD,
    Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD)
{
  // Reset variables
  rD.clear();
  norm_rD = 0.0;
  nD.clear();

  // Compute distance vector rD = r1 - x2 (not normalized)
  for (int i = 0; i < 3; i++)
  {
    rD(i) = r1(i) - x2(i);
  }

  // Compute norm of distance vector
  norm_rD = Core::FADUtils::vector_norm<3>(rD);

  if (Core::FADUtils::cast_to_double(norm_rD) < NORMTOL)
  {
    FOUR_C_THROW("ERROR: Distance vector of length zero! --> Change time step!");
  }

  // Compute unit distance vector
  for (int i = 0; i < 3; i++)
  {
    nD(i) = rD(i) / norm_rD;
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute distance vector                                         |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Compute surface normal vector                                        |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::compute_surface_normal(
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2, Core::LinAlg::Matrix<3, 1, TYPEBTS>& a2,
    TYPEBTS& norm_a2, Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2)
{
  // Reset variables
  n2.clear();
  norm_a2 = 0;
  a2.clear();

  // Compute surface normal vector (not normalized)
  a2(0) = x2_xi1(1) * x2_xi2(2) - x2_xi1(2) * x2_xi2(1);
  a2(1) = x2_xi1(2) * x2_xi2(0) - x2_xi1(0) * x2_xi2(2);
  a2(2) = x2_xi1(0) * x2_xi2(1) - x2_xi1(1) * x2_xi2(0);

  // Compute norm of surface normal vector
  norm_a2 = Core::FADUtils::vector_norm<3>(a2);

  if (Core::FADUtils::cast_to_double(norm_a2) < NORMTOL)
  {
    FOUR_C_THROW("ERROR: Surface normal vector of length zero! --> Change time step!");
  }

  // Compute unit surface normal vector
  for (int i = 0; i < 3; i++)
  {
    n2(i) = a2(i) / norm_a2;
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: Compute surface normal vector                                   |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Check if contact is active or inactive                                |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::check_contact_status(
    const double& pp, const TYPEBTS& gap, bool& contactflag)
{
#ifdef LINPENALTY
  // Linear penalty force law
  if (gap < 0)
    contactflag = true;
  else
    contactflag = false;
#endif

#ifdef QUADPENALTY
  // Linear penalty force law
  if (gap < 0)
    contactflag = true;
  else
    contactflag = false;
#endif

#ifdef ARBITPENALTY
  const double g0 = G0;

  // Linear penalty force law
  if (ARBITPENALTY == 1 or ARBITPENALTY == 3 or ARBITPENALTY == 4 or ARBITPENALTY == 5)
  {
    if (gap < g0)
      contactflag = true;
    else
      contactflag = false;
  }
  else if (ARBITPENALTY == 2)
  {
    if (gap < 0)
      contactflag = true;
    else
      contactflag = false;
  }
#endif

  return;
}
/*----------------------------------------------------------------------*
 | End: Check if contact is active or inactive                           |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Get global dofs of a node                                meier 02/14 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
std::vector<int>
CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::get_global_dofs(
    const Core::Nodes::Node* node)
{
  // Get dofs in beam contact discretization
  const std::vector<int> cdofs = contact_discret().dof(node);

  // Get dofs in problem discretization via offset
  std::vector<int> pdofs((int)(cdofs.size()));
  for (int k = 0; k < (int)(cdofs.size()); ++k)
  {
    pdofs[k] = (dofoffsetmap_.find(cdofs[k]))->second;
  }

  return pdofs;
}
/*----------------------------------------------------------------------*
 | End: Get global dofs of a node                                       |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Change the sign of the normal vector                   meier 02/2014 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::invert_normal()
{
  //  for (int i=0; i<3;i++)
  //    normal_(i) = -normal_(i);
}
/*----------------------------------------------------------------------*
 | End: Change the sign of the old normal vector                        |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Update all class variables at the end of time step                   |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::update_class_variables_step()
{
  // Update history variables
  const bool output = false;
  if (output)
  {
    std::cout << "update_class_variables_step:" << std::endl;
    for (int i = 0; i < (int)normalsets_.size(); i++)
    {
      std::cout << "normalset " << i << ":" << std::endl;
      std::cout << "  eta: " << normalsets_[i].first << std::endl;
      std::cout << "  nD: " << normalsets_[i].second(0) << ", " << normalsets_[i].second(1) << ", "
                << normalsets_[i].second(2) << std::endl;
    }
  }
  normalsets_old_ = normalsets_;
}
/*----------------------------------------------------------------------*
 | End: Update all class variables at the end of time step              |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Shift current normal vector to old normal vector       meier 02/2014 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::shift_normal()
{
  //  for (int j=0;j<3;j++)
  //    normal_old_(j) = normal_(j);
}
/*----------------------------------------------------------------------*
 | End: Shift current normal vector to old normal vector                |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Check if there is a difference of old and new gap      meier 02/2014 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
bool CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::get_new_gap_status()
{
  //  TYPEBTS gap_diff = gap_-gap_original_;
  //
  //  if (Core::FADUtils::cast_to_double(Core::FADUtils::Norm(gap_diff)) < GAPTOL)
  //    return false;
  //  else
  return true;
}
/*----------------------------------------------------------------------*
 | End: Check if there is a difference of old and new gap               |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Update nodal coordinates (public)                        meier 02/14 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::update_ele_pos(
    Core::LinAlg::SerialDenseMatrix& newele1pos, Core::LinAlg::SerialDenseMatrix& newele2pos)
{
  // Beam element positions
  for (int i = 0; i < 3 * numnodalvalues; i++)
    for (int j = 0; j < numnodes; j++) ele1pos_(3 * numnodalvalues * j + i) = newele1pos(i, j);

  // Solid element positions
  for (int i = 0; i < 3; i++)              // Loop over nodal dofs
    for (int j = 0; j < numnodessol; j++)  // Loop over nodes
      ele2pos_(3 * j + i) = newele2pos(i, j);

  return;
}
/*----------------------------------------------------------------------*
 | End: Update nodal coordinates (public)                               |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Update nodal tangents for tangent smoothing (public)     meier 02/14 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::
    update_ele_smooth_tangents(std::map<int, Core::LinAlg::Matrix<3, 1>>& currentpositions)
{
  //  //Tangent smoothing is only possible for Reissner beam elements --> FOUR_C_THROW() otherwise
  //  if (numnodalvalues>1)
  //    FOUR_C_THROW("Tangent smoothing only possible for Reissner beam elements
  //    (numnodalvalues=1)!!!");
  //
  //  Core::LinAlg::Matrix<3*numnodes,1> elepos_aux(true);
  //  //Tangent smoothing only possible with data type double (not with Sacado FAD)
  //  for (int i=0;i<3*numnodes;i++)
  //    elepos_aux(i)=Core::FADUtils::cast_to_double(ele1pos_(i));
  //
  //  nodaltangentssmooth1_=CONTACT::Beam3TangentSmoothing::CalculateNodalTangents<numnodes>(currentpositions,elepos_aux
  //  ,element1_,neighbors1_);
  //
  //  elepos_aux.clear();
  //  //Tangent smoothing only possible with data type double (not with Sacado FAD)
  //  for (int i=0;i<3*numnodes;i++)
  //    elepos_aux(i)=Core::FADUtils::cast_to_double(ele2pos_(i));
  //
  //  nodaltangentssmooth2_=CONTACT::Beam3TangentSmoothing::CalculateNodalTangents<numnodes>(currentpositions,elepos_aux
  //  ,element2_,neighbors2_);
}
/*----------------------------------------------------------------------*
 | End: Update nodal coordinates (public)                               |
 *----------------------------------------------------------------------*/


/*----------------------------------------------------------------------*
 | Shift Nodal positions (public)                           meier 02/14 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::shift_nodal_positions()
{
  //  //Reissner beams
  //  if (numnodalvalues == 1)
  //  {
  //    for (int i=0; i<numnodes; i++)
  //    {
  //      for (int j=0;j<3;j++)
  //      {
  //        ele1pos_(3*i + j) = ele1pos_(3*i + j) + SHIFTVALUE * normal_old_(j);
  //      }
  //    }
  //  }
  //  //Kirchhoff beams
  //  else if (numnodalvalues == 2)
  //  {
  //    if (numnodes == 2)
  //    {
  //      for (int j=0;j<3;j++)
  //      {
  //        ele1pos_(j) = ele1pos_(j) + SHIFTVALUE * normal_old_(j);
  //        ele1pos_(6+j) = ele1pos_(6+j) + SHIFTVALUE * normal_old_(j);
  //      }
  //    }
  //    else
  //    {
  //      FOUR_C_THROW("Only numnodes = 2 possible for Kirchhoff beams!!!");
  //    }
  //  }
  //  else
  //  {
  //    FOUR_C_THROW("The parameter numnodalvalues can only have the values 1 or 2!!!");
  //  }
  return;
}
/*----------------------------------------------------------------------*
 | End: Shift Nodal positions (public)                                  |
 *----------------------------------------------------------------------*/


std::shared_ptr<CONTACT::Beam3tosolidcontactinterface> CONTACT::Beam3tosolidcontactinterface::impl(
    const int numnodessol, const int numnodes, const int numnodalvalues,
    const Core::FE::Discretization& pdiscret, const Core::FE::Discretization& cdiscret,
    const std::map<int, int>& dofoffsetmap, Core::Elements::Element* element1,
    Core::Elements::Element* element2, Teuchos::ParameterList beamcontactparams)
{
  if (numnodalvalues != 1 and numnodalvalues != 2)
    FOUR_C_THROW("Only the values 1 and 2 are valid for numnodalvalues!");

  if (numnodalvalues != 2 and numnodes != 2)
    FOUR_C_THROW(
        "Only the values numnodes=2 is possible for Kirchhoff beams, i.e. if numnodalvalues=2!");

  if (numnodes != 2 and numnodes != 3 and numnodes != 4 and numnodes != 5)
    FOUR_C_THROW("Only the values 2, 3, 4 and 5 are valid for numnodes!");

  if (numnodessol != 3 and numnodessol != 6 and numnodessol != 4 and numnodessol != 8 and
      numnodessol != 9)
    FOUR_C_THROW("Only the values 3, 4, 6, 8 and 9 are valid for numnodessol!");


  switch (numnodessol)
  {
      //    case 3:
      //    {
      //      switch (numnodalvalues)
      //      {
      //        case 1:
      //        {
      //          switch (numnodes)
      //          {
      //            case 2:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<3,2,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 3:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<3,3,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 4:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<3,4,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 5:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<3,5,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //          }
      //          break;
      //        }
      //        case 2:
      //        {
      //          return Teuchos::rcp (new
      //          CONTACT::Beam3tosolidcontact<3,2,2>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //        }
      //      }
      //      break;
      //    }
    case 4:
    {
      switch (numnodalvalues)
      {
          //        case 1:
          //        {
          //          switch (numnodes)
          //          {
          //            case 2:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<4,2,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //            case 3:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<4,3,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //            case 4:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<4,4,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //            case 5:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<4,5,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //          }
          //          break;
          //        }
        case 2:
        {
          return std::make_shared<CONTACT::Beam3tosolidcontact<4, 2, 2>>(
              pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams);
        }
      }
      break;
    }
      //    case 6:
      //    {
      //      switch (numnodalvalues)
      //      {
      //        case 1:
      //        {
      //          switch (numnodes)
      //          {
      //            case 2:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<6,2,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 3:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<6,3,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 4:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<6,4,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 5:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<6,5,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //          }
      //          break;
      //        }
      //        case 2:
      //        {
      //          return Teuchos::rcp (new
      //          CONTACT::Beam3tosolidcontact<6,2,2>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //        }
      //      }
      //      break;
      //    }
    case 8:
    {
      switch (numnodalvalues)
      {
          //        case 1:
          //        {
          //          switch (numnodes)
          //          {
          //            case 2:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<8,2,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //            case 3:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<8,3,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //            case 4:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<8,4,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //            case 5:
          //            {
          //              return Teuchos::rcp (new
          //              CONTACT::Beam3tosolidcontact<8,5,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
          //            }
          //          }
          //          break;
          //        }
        case 2:
        {
          return std::make_shared<CONTACT::Beam3tosolidcontact<8, 2, 2>>(
              pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams);
        }
      }
      break;
    }
      //    case 9:
      //    {
      //      switch (numnodalvalues)
      //      {
      //        case 1:
      //        {
      //          switch (numnodes)
      //          {
      //            case 2:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<9,2,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 3:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<9,3,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 4:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<9,4,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //            case 5:
      //            {
      //              return Teuchos::rcp (new
      //              CONTACT::Beam3tosolidcontact<9,5,1>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //            }
      //          }
      //          break;
      //        }
      //        case 2:
      //        {
      //          return Teuchos::rcp (new
      //          CONTACT::Beam3tosolidcontact<9,2,2>(pdiscret,cdiscret,dofoffsetmap,element1,element2,beamcontactparams));
      //        }
      //      }
      //      break;
      //    }
  }

  return nullptr;
}

#if defined(FADCHECKLINCONTACTINTERVALBORDER) || defined(FADCHECKLINGAUSSPOINT)
/*----------------------------------------------------------------------*
 | FAD-Check for linearization of element parameters                    |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::fad_check_lin_parameter(
    const int& fixed_par, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi_1d_FAD,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi_2d_FAD,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d_FAD,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Initialize array with directional derivatives of parameters xi1, xi2, eta for FAD
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> par_d_FAD[3];
  par_d_FAD[0] = xi_1d_FAD;
  par_d_FAD[1] = xi_2d_FAD;
  par_d_FAD[2] = eta_d_FAD;

  // Get indices of not fixed parameters
  int par_i[2];
  int j = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != fixed_par)
    {
      par_i[j] = i;
      j++;
    }
  }

  // Clear not fixed directional derivatives
  for (int i = 0; i < 2; i++)
  {
    par_d_FAD[par_i[i]].clear();
  }

  // Compute norm of distance vector rD to scale the equations (this yields better conditioning)
  // NOTE: Even if automatic differentiation via FAD is applied, norm_rD has to be of type double
  // since this factor is needed for a pure scaling of the nonlinear orthogonality conditions and
  // has not to be linearized!
  double norm_rD_scale = Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm<3>(rD));

  // Evaluate f of orthogonality conditions
  Core::LinAlg::Matrix<2, 1, TYPEBTS> f(true);
  for (int i = 0; i < 3; i++)
  {
    f(0) += rD(i) * x2_xi1(i) / norm_rD_scale;
    f(1) += rD(i) * x2_xi2(i) / norm_rD_scale;
  }

  // Initialize matrices of system of equations showed in method compute_lin_parameter
  Core::LinAlg::Matrix<2, 2, TYPEBTS> L(true);
  Core::LinAlg::Matrix<2, 2, TYPEBTS> L_inv(true);
  Core::LinAlg::Matrix<2, dim1 + dim2, TYPEBTS> B(true);
  Core::LinAlg::Matrix<2, dim1 + dim2, TYPEBTS> D(true);

  // Compute L elementwise for the not fixed parameters
  L(0, 0) = f(0).dx(dim1 + dim2 + par_i[0]);
  L(0, 1) = f(0).dx(dim1 + dim2 + par_i[1]);
  L(1, 0) = f(1).dx(dim1 + dim2 + par_i[0]);
  L(1, 1) = f(1).dx(dim1 + dim2 + par_i[1]);

  // invert L by hand
  TYPEBTS det_L = L(0, 0) * L(1, 1) - L(0, 1) * L(1, 0);
  if (Core::FADUtils::cast_to_double(Core::FADUtils::Norm(det_L)) < DETERMINANTTOL)
    FOUR_C_THROW("ERROR: Determinant of L = 0");
  L_inv(0, 0) = L(1, 1) / det_L;
  L_inv(0, 1) = -L(0, 1) / det_L;
  L_inv(1, 0) = -L(1, 0) / det_L;
  L_inv(1, 1) = L(0, 0) / det_L;

  // Compute B
  for (int j = 0; j < dim1 + dim2; j++)
  {
    B(0, j) = -f(0).dx(j);
    B(1, j) = -f(1).dx(j);
  }
  if (fixed_par == 2)  // eta fixed (eta_d_FAD is known)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      B(0, j) += -f(0).dx(dim1 + dim2 + 2) * eta_d_FAD(j);
      B(1, j) += -f(1).dx(dim1 + dim2 + 2) * eta_d_FAD(j);
    }
  }

  // Compute D = L^-1 * B
  D.multiply(L_inv, B);

  // Finally compute directional derivatives of not fixed parameters
  for (int i = 0; i < dim1 + dim2; i++)
  {
    par_d_FAD[par_i[0]](i) = D(0, i);
    par_d_FAD[par_i[1]](i) = D(1, i);
  }

  // Store directional derivatives parameters in original variables
  xi_1d_FAD = par_d_FAD[0];
  xi_2d_FAD = par_d_FAD[1];
  eta_d_FAD = par_d_FAD[2];

  // Store analytical calculated directional derivatives of parameters xi1, xi2, eta in an array for
  // comparison
  Core::LinAlg::Matrix<dim1 + dim2, 1, TYPEBTS> par_d[3];
  par_d[0] = xi1_d;
  par_d[1] = xi2_d;
  par_d[2] = eta_d;

  // Compare linearizations
  for (int i = 0; i < 2; i++)
  {
    // Print name of parameter
    switch (par_i[i])
    {
      case 0:
        std::cout << "xi1_d: " << std::endl;
        break;
      case 1:
        std::cout << "xi2_d: " << std::endl;
        break;
      case 2:
        std::cout << "eta_d: " << std::endl;
        break;
    }

    // Print directional derivatives
    std::cout << "Lin: " << std::endl;
    for (int j = 0; j < dim1 + dim2; j++) std::cout << par_d[par_i[i]](j).val() << "  ";
    std::cout << std::endl;

    // Print directional derivatives calculated via FAD
    std::cout << "FAD: " << std::endl;
    for (int j = 0; j < dim1 + dim2; j++)
    {
      std::cout << par_d_FAD[par_i[i]](j).val() << "  ";
    }
    std::cout << std::endl;

    // Calculate maximal difference between par_d and par_d_FAD
    const double tolerance = 1.0e-12;
    double diff = 0.0;
    for (int j = 0; j < dim1 + dim2; j++)
    {
      diff = std::max(diff, fabs(par_d[par_i[i]](j).val() - par_d_FAD[par_i[i]](j).val()));
    }
    std::cout << "Maximal difference: " << diff;
    if (diff > tolerance)
    {
      std::cout << " > " << tolerance << " (tolerance)";
    }
    std::cout << std::endl;
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: FAD-Check for linearization of element parameters               |
 *----------------------------------------------------------------------*/
#endif

#ifdef FADCHECKLINGAUSSPOINT
/*----------------------------------------------------------------------*
 | FAD-Check for linearization of gap and distance vector               |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::fad_check_lin_gap_and_distance_vector(const TYPEBTS& gap,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& gap_d_FAD,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& rD_d_FAD,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& gap_d,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& rD_d)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Compute directional derivative of gap with FAD
  for (int i = 0; i < dim1 + dim2; i++)
  {
    gap_d_FAD(i) = gap.dx(i) + gap.dx(dim1 + dim2) * xi1_d(i) + gap.dx(dim1 + dim2 + 1) * xi2_d(i) +
                   gap.dx(dim1 + dim2 + 2) * eta_d(i);
  }

  // Compute directional derivative of distance vector rD with FAD
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      rD_d_FAD(i, j) = rD(i).dx(j) + rD(i).dx(dim1 + dim2) * xi1_d(j) +
                       rD(i).dx(dim1 + dim2 + 1) * xi2_d(j) + rD(i).dx(dim1 + dim2 + 2) * eta_d(j);
    }
  }

  // Print name of directional derivative of gap
  std::cout << "gap_d: " << std::endl;

  // Print directional derivative
  std::cout << "Lin: " << std::endl;
  for (int i = 0; i < dim1 + dim2; i++) std::cout << gap_d(i).val() << "  ";
  std::cout << std::endl;
  // Print directional derivative calculated via FAD
  std::cout << "FAD: " << std::endl;
  for (int i = 0; i < dim1 + dim2; i++) std::cout << gap_d_FAD(i).val() << "  ";
  std::cout << std::endl;

  const double tolerance = 1.0e-12;

  // Calculate maximal difference between gap_d and gap_d_FAD
  double diff = 0.0;
  for (int i = 0; i < dim1 + dim2; i++)
    diff = std::max(diff, fabs(gap_d(i).val() - gap_d_FAD(i).val()));
  std::cout << "Maximal difference: " << diff;
  if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
  std::cout << std::endl;


  // Print name of directional derivative of distance vector rD
  std::cout << "rD_d: " << std::endl;

  // Print directional derivative
  std::cout << "Lin: " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << rD_d(i, j).val() << "  ";
    std::cout << std::endl;
  }
  // Print directional derivative calculated via FAD
  std::cout << "FAD: " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << rD_d_FAD(i, j).val() << "  ";
    std::cout << std::endl;
  }

  // Calculate maximal difference between rD_d and rD_d_FAD
  diff = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      diff = std::max(diff, fabs(rD_d(i, j).val() - rD_d_FAD(i, j).val()));
  std::cout << "Maximal difference: " << diff;
  if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
  std::cout << std::endl;
}
/*----------------------------------------------------------------------*
 | End: FAD-Check for linearization of gap and distance vector          |
 *----------------------------------------------------------------------*/
#endif

#ifdef FADCHECKLINGAUSSPOINT
/*----------------------------------------------------------------------*
 | FAD-Check for linearization of normal vector                         |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::fad_check_lin_normal(
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& nD, const Core::LinAlg::Matrix<3, 1, TYPEBTS>& n2,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi1_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& xi2_d,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues + 3 * numnodessol, 1, TYPEBTS>& eta_d,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& nD_d_FAD,
    Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& n_2d_FAD,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& nD_d,
    const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& n2_d)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Calculate directional derivative of unit distance vector nD with FAD
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      nD_d_FAD(i, j) = nD(i).dx(j) + nD(i).dx(dim1 + dim2) * xi1_d(j) +
                       nD(i).dx(dim1 + dim2 + 1) * xi2_d(j) + nD(i).dx(dim1 + dim2 + 2) * eta_d(j);
    }
  }

  // Calculate directional derivative of unit surface normal vector n2 with FAD
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++)
    {
      n_2d_FAD(i, j) = n2(i).dx(j) + n2(i).dx(dim1 + dim2) * xi1_d(j) +
                       n2(i).dx(dim1 + dim2 + 1) * xi2_d(j) + n2(i).dx(dim1 + dim2 + 2) * eta_d(j);
    }
  }

  // Print name of directional derivative of unit distance vector nD
  std::cout << "nD_d: " << std::endl;

  // Print directional derivative
  std::cout << "Lin: " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << nD_d(i, j).val() << "  ";
    std::cout << std::endl;
  }
  // Print directional derivative calculated via FAD
  std::cout << "FAD: " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << nD_d_FAD(i, j).val() << "  ";
    std::cout << std::endl;
  }

  const double tolerance = 1.0e-12;

  // Calculate maximal difference between nD_d and nD_d_FAD
  double diff = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      diff = std::max(diff, fabs(nD_d(i, j).val() - nD_d_FAD(i, j).val()));
  std::cout << "Maximal difference: " << diff;
  if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
  std::cout << std::endl;


  // Print name of directional derivative of unit surface normal vector n2
  std::cout << "n2_d: " << std::endl;

  // Print directional derivative
  std::cout << "Lin: " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << n2_d(i, j).val() << "  ";
    std::cout << std::endl;
  }
  // Print directional derivative calculated via FAD
  std::cout << "FAD: " << std::endl;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << n_2d_FAD(i, j).val() << "  ";
    std::cout << std::endl;
  }

  // Calculate maximal difference between n2_d and n_2d_FAD
  diff = 0.0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      diff = std::max(diff, fabs(n2_d(i, j).val() - n_2d_FAD(i, j).val()));
  std::cout << "Maximal difference: " << diff;
  if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
  std::cout << std::endl;
}
/*----------------------------------------------------------------------*
 | End: FAD-Check for linearization of normal vector                    |
 *----------------------------------------------------------------------*/
#endif

#ifdef FADCHECKLINORTHOGONALITYCONDITION
/*----------------------------------------------------------------------*
 | FAD-Check for linearizations of orthogonality conditions             |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes,
    numnodalvalues>::fad_check_lin_orthogonality_condition(const int& fixed_par,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& rD,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi1,
    const Core::LinAlg::Matrix<3, 1, TYPEBTS>& x2_xi2, Core::LinAlg::Matrix<2, 2, TYPEBTS>& J_FAD,
    const Core::LinAlg::Matrix<2, 2, TYPEBTS>& J)
{
  // Get indices of not fixed parameters
  int par_i[2];
  int j = 0;
  for (int i = 0; i < 3; i++)
  {
    if (i != fixed_par)
    {
      par_i[j] = i;
      j++;
    }
  }

  // Compute norm of distance vector rD to scale the equations (this yields better conditioning)
  // NOTE: Even if automatic differentiation via FAD is applied, norm_rD has to be of type double
  // since this factor is needed for a pure scaling of the nonlinear orthogonality conditions and
  // has not to be linearized!
  double norm_rD_scale = Core::FADUtils::cast_to_double(Core::FADUtils::vector_norm<3>(rD));

  // Evaluate f of orthogonality conditions
  Core::LinAlg::Matrix<2, 1, TYPEBTS> f(true);
  for (int i = 0; i < 3; i++)
  {
    f(0) += rD(i) * x2_xi1(i) / norm_rD_scale;
    f(1) += rD(i) * x2_xi2(i) / norm_rD_scale;
  }

  // Calculate linearization of orthogonality conditions (Jacobi-matrix J)
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      J_FAD(i, j) = f(i).dx(3 * numnodes * numnodalvalues + 3 * numnodessol + par_i[j]);

  // Print linearization of f
  std::cout << "Lin: " << std::endl;
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) std::cout << J(i, j).val() << "  ";
    std::cout << std::endl;
  }

  // Print linearization of f calculated via FAD
  std::cout << "FAD: " << std::endl;
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 2; j++) std::cout << J_FAD(i, j).val() << "  ";
    std::cout << std::endl;
  }

  return;
}
/*----------------------------------------------------------------------*
 | End: FAD-Check for linearizations of orthogonality conditions        |
 *----------------------------------------------------------------------*/
#endif

#ifdef FDCHECKSTIFFNESS
/*----------------------------------------------------------------------*
 | Differentation with finite difference for contact stiffness          |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::fd_check_stiffness(
    const double& pp, const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPEBTS>& fc1,
    const Core::LinAlg::Matrix<3 * numnodessol, 1, TYPEBTS>& fc2,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
    Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>&
        stiffc2)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Options for finite difference check
  const double h = 1.0e-9;
  const bool use_stiffc_FD = false;
  const bool output = true;

#ifdef GMSHDEBUG
  // Store class variables before adding h
  std::vector<gmshDebugPoint> gmshDebugPoints = gmshDebugPoints_;
  std::vector<std::pair<TYPEBTS, Core::LinAlg::Matrix<3, 1, TYPEBTS>>> normalsets = normalsets_;
#endif

  // Intialize matrices for contact stiffness
  Core::LinAlg::Matrix<dim1, dim1 + dim2, TYPEBTS> stiffc1_FD(true);
  Core::LinAlg::Matrix<dim2, dim1 + dim2, TYPEBTS> stiffc2_FD(true);

  // Loop over all coloums, size of displacement vector d = [d1, d2]^T
  for (int col = 0; col < dim1 + dim2; col++)
  {
    // Initialize temporary varaibles
    TYPEBTS ele1pos_col = 0.0;
    TYPEBTS ele2pos_col = 0.0;

    // Store current element position before adding h and then add h to d(col)
    if (col < dim1)
    {
      // Beam element positions
      ele1pos_col = ele1pos_(col);
      ele1pos_(col) += h;
    }
    else
    {
      // Solid surface element positions
      ele2pos_col = ele2pos_(col - dim1);
      ele2pos_(col - dim1) += h;
    }


    // Find contact interval borders
    // -----------------------------------------------------------------

    // Vector for contact interval border parameter sets (xi1, xi2, eta and index of fixed
    // paramater)
    std::vector<std::pair<Core::LinAlg::Matrix<3, 1, TYPEBTS>, int>> parsets;

    // Find contact interval borders
    get_contact_interval_borders(parsets);

    // -----------------------------------------------------------------
    // End: Find contact interval borders


    // Compute contact forces and stiffness for all contact intervals
    // -----------------------------------------------------------------

    // Temporary vectors for contact forces at position with added h
    Core::LinAlg::Matrix<dim1, 1, TYPEBTS> fc1_h(true);
    Core::LinAlg::Matrix<dim2, 1, TYPEBTS> fc2_h(true);

    // Temporary matrices for contact stiffness at position with added h
    Core::LinAlg::Matrix<dim1, dim1 + dim2, TYPEBTS> stiffc1_h(true);
    Core::LinAlg::Matrix<dim2, dim1 + dim2, TYPEBTS> stiffc2_h(true);

    // Temporary matrices for contact stiffness calculated via FAD
    // TODO: Declare stiffc1_FAD and stiffc2_FAD only if needed (#ifdef AUTOMATICDIFFBTS)
    Core::LinAlg::Matrix<dim1, dim1 + dim2, TYPEBTS> stiffc1_FAD(true);
    Core::LinAlg::Matrix<dim2, dim1 + dim2, TYPEBTS> stiffc2_FAD(true);

#ifdef GMSHDEBUG
    // Clear class variables in each loop, because the information at positions with added h is not
    // needed
    gmshDebugPoints_.clear();
    gmshDebugPoints_.resize(0);
    normalsets_.clear();
    normalsets_.resize(0);
#endif

    // Loop over all contact intervals, calculate and sum up contact forces (fc1, fc2) and stiffness
    // (stiffc1, stiffc2)
    for (int i = 0; i < (int)parsets.size() - 1; i += 2)
    {
      bool doAssembleContactInterval = false;

      // Two parameter sets indicate the beginning (a) and the end (b) of the contact interval
      // TODO: Use stiffc1_FAD and stiffc2_FAD only if needed (#ifdef AUTOMATICDIFFBTS)
      evaluate_contact_interval(pp, parsets[i], parsets[i + 1], fc1_h, fc2_h, stiffc1_h, stiffc2_h,
          doAssembleContactInterval, stiffc1_FAD, stiffc2_FAD);
    }

    // -----------------------------------------------------------------
    // End: Compute contact forces and stiffness for all contact intervals


    // Restore element positions after the contact forces and stiffness at the position with added h
    // are calculated
    if (col < dim1)
      ele1pos_(col) = ele1pos_col;
    else
      ele2pos_(col - dim1) = ele2pos_col;

    // Compute contact stiffness with finite difference
    for (int i = 0; i < dim1; i++)
    {
      // Beam element
      stiffc1_FD(i, col) = (fc1_h(i) - fc1(i)) / h;
    }
    for (int i = 0; i < dim2; i++)
    {
      // Solid element
      stiffc2_FD(i, col) = (fc2_h(i) - fc2(i)) / h;
    }
  }

  if (output)
  {
    // Print contact force fc1, stiffc1 and stiffc1_FD calculated via finite difference
    std::cout << "fc1: " << fc1 << std::endl;
    std::cout << "stiffc1: " << stiffc1 << std::endl;
    std::cout << "stiffc1_FD (h: " << h << "): " << stiffc1_FD << std::endl;

    // Set tolerance for comparison
    const double tolerance = 1.0e-12;

    // Calculate maximal difference between stiffc1 and stiffc1_FD
    double diff = 0.0;
    for (int i = 0; i < dim1; i++)
      for (int j = 0; j < dim1 + dim2; j++)
        diff = std::max(diff, fabs(stiffc1(i, j) - stiffc1_FD(i, j)));
    std::cout << "Maximal difference: " << diff;
    if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
    std::cout << std::endl;


    // Print contact force fc2, stiffc2 and stiffc2_FD calculated via finite difference
    std::cout << "fc2: " << fc2 << std::endl;
    std::cout << "stiffc2: " << stiffc2 << std::endl;
    std::cout << "stiffc2_FD (h: " << h << "): " << stiffc2_FD << std::endl;

    // Calculate maximal difference between stiffc2 and stiffc2_FD
    diff = 0.0;
    for (int i = 0; i < dim2; i++)
      for (int j = 0; j < dim1 + dim2; j++)
        diff = std::max(diff, fabs(stiffc2(i, j) - stiffc2_FD(i, j)));
    std::cout << "Maximal difference: " << diff;
    if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
    std::cout << std::endl;
  }

  // Assemble stiffc1_FD and stiffc2_FD instead of stiffc1 and stiffc2
  if (use_stiffc_FD)
  {
    stiffc1 = stiffc1_FD;
    stiffc2 = stiffc2_FD;
  }

#ifdef GMSHDEBUG
  // Restore class variables at position without added h
  gmshDebugPoints_ = gmshDebugPoints;
  normalsets_ = normalsets;
#endif

  return;
}
/*----------------------------------------------------------------------*
 | End: Differentation with finite difference for contact stiffness     |
 *----------------------------------------------------------------------*/
#endif

#ifdef FADCHECKSTIFFNESS
/*----------------------------------------------------------------------*
 | Check difference between stiffness and FAD stiffness                 |
 *----------------------------------------------------------------------*/
template <const int numnodessol, const int numnodes, const int numnodalvalues>
void CONTACT::Beam3tosolidcontact<numnodessol, numnodes, numnodalvalues>::fad_check_stiffness(
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1,
    const Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
        TYPEBTS>& stiffc2,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues,
        3 * numnodes * numnodalvalues + 3 * numnodessol, TYPEBTS>& stiffc1_FAD,
    const Core::LinAlg::Matrix<3 * numnodessol, 3 * numnodes * numnodalvalues + 3 * numnodessol,
        TYPEBTS>& stiffc2_FAD)
{
  const int dim1 = 3 * numnodes * numnodalvalues;
  const int dim2 = 3 * numnodessol;

  // Print information
  std::cout << std::endl << "FAD-Check: Stiffmatrix:" << std::endl;

  // Print stiffc1 and stiffc1_FAD calculated via FAD
  std::cout << "stiffc1:" << std::endl;

  // Print stiffmatrix
  std::cout << "Lin:" << std::endl;
  for (int i = 0; i < dim1; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << stiffc1(i, j).val() << "  ";
    std::cout << std::endl;
  }

  // Print stiffmatrix calculated via FAD
  std::cout << "FAD:" << std::endl;
  for (int i = 0; i < dim1; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << stiffc1_FAD(i, j).val() << "  ";
    std::cout << std::endl;
  }

  const double tolerance = 1.0e-12;

  // Calculate maximal difference between stiffc1 and stiffc1_FAD
  double diff = 0.0;
  for (int i = 0; i < dim1; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      diff = std::max(diff, fabs(stiffc1(i, j).val() - stiffc1_FAD(i, j).val()));
  std::cout << "Maximal difference: " << diff;
  if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
  std::cout << std::endl;


  // Print stiffc2 and stiffc2_FAD calculated via FAD
  std::cout << "stiffc2:" << std::endl;

  // Print stiffmatrix
  std::cout << "Lin:" << std::endl;
  for (int i = 0; i < dim2; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << stiffc2(i, j).val() << "  ";
    std::cout << std::endl;
  }

  // Print stiffmatrix calculated via FAD
  std::cout << "FAD:" << std::endl;
  for (int i = 0; i < dim2; i++)
  {
    for (int j = 0; j < dim1 + dim2; j++) std::cout << stiffc2_FAD(i, j).val() << "  ";
    std::cout << std::endl;
  }

  // Calculate maximal difference between stiffc2 and stiffc2_FAD
  diff = 0.0;
  for (int i = 0; i < dim2; i++)
    for (int j = 0; j < dim1 + dim2; j++)
      diff = std::max(diff, fabs(stiffc2(i, j).val() - stiffc2_FAD(i, j).val()));
  std::cout << "Maximal difference: " << diff;
  if (diff > tolerance) std::cout << " > " << tolerance << " (tolerance)";
  std::cout << std::endl;
}
/*----------------------------------------------------------------------*
 | End: Check difference between stiffness and FAD stiffness            |
 *----------------------------------------------------------------------*/
#endif

// Possible template cases: this is necessary for the compiler
// template class CONTACT::Beam3tosolidcontact<3,2,1>;
// template class CONTACT::Beam3tosolidcontact<3,3,1>;
// template class CONTACT::Beam3tosolidcontact<3,4,1>;
// template class CONTACT::Beam3tosolidcontact<3,5,1>;
// template class CONTACT::Beam3tosolidcontact<3,2,2>;

// template class CONTACT::Beam3tosolidcontact<4,2,1>;
// template class CONTACT::Beam3tosolidcontact<4,3,1>;
// template class CONTACT::Beam3tosolidcontact<4,4,1>;
// template class CONTACT::Beam3tosolidcontact<4,5,1>;
template class CONTACT::Beam3tosolidcontact<4, 2,
    2>;  // Hermite beam element, quad4 surface element

// template class CONTACT::Beam3tosolidcontact<6,2,1>;
// template class CONTACT::Beam3tosolidcontact<6,3,1>;
// template class CONTACT::Beam3tosolidcontact<6,4,1>;
// template class CONTACT::Beam3tosolidcontact<6,5,1>;
// template class CONTACT::Beam3tosolidcontact<6,2,2>;

// template class CONTACT::Beam3tosolidcontact<8,2,1>;
// template class CONTACT::Beam3tosolidcontact<8,3,1>;
// template class CONTACT::Beam3tosolidcontact<8,4,1>;
// template class CONTACT::Beam3tosolidcontact<8,5,1>;
template class CONTACT::Beam3tosolidcontact<8, 2,
    2>;  // Hermite beam element, quad8 surface element

// template class CONTACT::Beam3tosolidcontact<9,2,1>;
// template class CONTACT::Beam3tosolidcontact<9,3,1>;
// template class CONTACT::Beam3tosolidcontact<9,4,1>;
// template class CONTACT::Beam3tosolidcontact<9,5,1>;
// template class CONTACT::Beam3tosolidcontact<9,2,2>;

FOUR_C_NAMESPACE_CLOSE
