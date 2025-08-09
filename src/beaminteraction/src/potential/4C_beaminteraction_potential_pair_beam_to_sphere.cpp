// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_pair_beam_to_sphere.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_utils.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Teuchos_TimeMonitor.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::BeamToSpherePotentialPair()
    : BeamPotentialPair(),
      beam_element_(nullptr),
      sphere_element_(nullptr),
      time_(0.0),
      k_(0.0),
      m_(0.0),
      beamele_reflength_(0.0),
      radius1_(0.0),
      radius2_(0.0),
      interaction_potential_(0.0)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::setup()
{
  check_init();

  // call setup of base class first
  BeamPotentialPair::setup();


  ele1pos_.clear();
  ele2pos_.clear();

  fpot1_.clear();
  fpot2_.clear();
  stiffpot1_.clear();
  stiffpot2_.clear();


  // cast first element to Beam3Base
  beam_element_ = dynamic_cast<const Discret::Elements::Beam3Base*>(element1());

  if (beam_element_ == nullptr)
    FOUR_C_THROW(
        "cast to Beam3Base failed! first element in BeamToSpherePotentialPair pair"
        " must be a beam element!");

  // get radius and stress-free reference length of beam element
  radius1_ = BeamInteraction::calc_ele_radius(beam_element_);
  beamele_reflength_ = beam_element_->ref_length();

  // cast second element to RigidSphere
  sphere_element_ = dynamic_cast<const Discret::Elements::Rigidsphere*>(element2());

  if (sphere_element_ == nullptr)
    FOUR_C_THROW(
        "cast to Rigidsphere failed! second element in BeamToSpherePotentialPair pair"
        " must be a Rigidsphere element!");

  // get radius of sphere element
  radius2_ = sphere_element_->radius();

  // initialize charge conditions applied to beam and sphere elements
  chargeconds_.resize(2);

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
bool BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::evaluate(
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22,
    const std::vector<Core::Conditions::Condition*> chargeconds, const double k, const double m)
{
  // nothing to do in case of k==0.0
  if (k == 0.0) return false;

  // reset fpot and stiffpot class variables
  fpot1_.clear();
  fpot2_.clear();
  stiffpot1_.clear();
  stiffpot2_.clear();

  unsigned int dim1 = 3 * numnodes * numnodalvalues;
  unsigned int dim2 = 3;

  // set class variables
  if (chargeconds.size() == 2)
  {
    if (chargeconds[0]->type() == Core::Conditions::BeamPotential_LineChargeDensity)
      chargeconds_[0] = chargeconds[0];
    else
      FOUR_C_THROW("Provided condition is not of correct type BeamPotential_LineChargeDensity!");

    if (chargeconds[1]->type() == Core::Conditions::RigidspherePotential_PointCharge)
      chargeconds_[1] = chargeconds[1];
    else
      FOUR_C_THROW("Provided condition is not of correct type RigidspherePotential_PointCharge!");
  }
  else
    FOUR_C_THROW(
        "Expected TWO charge conditions for a (beam,rigidsphere) potential-based interaction "
        "pair!");

  k_ = k;
  m_ = m;

  // prepare FAD
#ifdef AUTOMATICDIFF
  // The 2*3*numnodes*numnodalvalues primary DoFs are the components of the nodal positions /
  // tangents.
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; i++)
    ele1pos_(i).diff(i, 3 * numnodes * numnodalvalues + 3);

  for (unsigned int i = 0; i < 3; i++)
    ele2pos_(i).diff(3 * numnodes * numnodalvalues + i, 3 * numnodes * numnodalvalues + 3);
#endif


  // compute the values for element residual vectors ('force') and linearizations ('stiff')
  // Todo allow for independent choice of strategy for beam-to-sphere potentials
  switch (params()->strategy)
  {
    case BeamInteraction::Potential::Strategy::double_length_specific_large_separations:
    {
      evaluate_fpotand_stiffpot_large_sep_approx();
      break;
    }

    default:
      FOUR_C_THROW("Invalid strategy to evaluate beam-to-sphere interaction potential!");
  }

  // resize variables and fill with pre-computed values
  if (forcevec1 != nullptr)
  {
    forcevec1->size(dim1);
    for (unsigned int i = 0; i < dim1; ++i)
      (*forcevec1)(i) = Core::FADUtils::cast_to_double(fpot1_(i));
  }
  if (forcevec2 != nullptr)
  {
    forcevec2->size(dim2);
    for (unsigned int i = 0; i < dim2; ++i)
      (*forcevec2)(i) = Core::FADUtils::cast_to_double(fpot2_(i));
  }

  if (stiffmat11 != nullptr)
  {
    stiffmat11->shape(dim1, dim1);
    for (unsigned int irow = 0; irow < dim1; ++irow)
      for (unsigned int icol = 0; icol < dim1; ++icol)
        (*stiffmat11)(irow, icol) = Core::FADUtils::cast_to_double(stiffpot1_(irow, icol));
  }
  if (stiffmat12 != nullptr)
  {
    stiffmat12->shape(dim1, dim2);
    for (unsigned int irow = 0; irow < dim1; ++irow)
      for (unsigned int icol = 0; icol < dim2; ++icol)
        (*stiffmat12)(irow, icol) = Core::FADUtils::cast_to_double(stiffpot1_(irow, dim1 + icol));
  }
  if (stiffmat21 != nullptr)
  {
    stiffmat21->shape(dim2, dim1);
    for (unsigned int irow = 0; irow < dim2; ++irow)
      for (unsigned int icol = 0; icol < dim1; ++icol)
        (*stiffmat21)(irow, icol) = Core::FADUtils::cast_to_double(stiffpot2_(irow, icol));
  }
  if (stiffmat22 != nullptr)
  {
    stiffmat22->shape(dim2, dim2);
    for (unsigned int irow = 0; irow < dim2; ++irow)
      for (unsigned int icol = 0; icol < dim2; ++icol)
        (*stiffmat22)(irow, icol) = Core::FADUtils::cast_to_double(stiffpot2_(irow, dim1 + icol));
  }

  return (true);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes,
    numnodalvalues>::evaluate_fpotand_stiffpot_large_sep_approx()
{
  // get cutoff radius
  const std::optional<double> cutoff_radius = params()->cutoff_radius;

  // Set gauss integration rule
  Core::FE::GaussRule1D gaussrule = get_gauss_rule();

  // Get gauss points (gp) for integration
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);
  // number of gps
  const int numgp = gausspoints.nquad;

  // vectors for shape functions and their derivatives
  // Attention: these are individual shape function values, NOT shape function matrices
  // values at all gauss points are stored in advance
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues>> N1_i(numgp);     // = N1_i
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues>> N1_i_xi(numgp);  // = N1_i,xi

  // coords and derivatives of the two gauss points
  Core::LinAlg::Matrix<3, 1, TYPE> r1(Core::LinAlg::Initialization::zero);    // = r1
  Core::LinAlg::Matrix<3, 1, TYPE> r2(Core::LinAlg::Initialization::zero);    // = r2
  Core::LinAlg::Matrix<3, 1, TYPE> dist(Core::LinAlg::Initialization::zero);  // = r1-r2
  TYPE norm_dist = 0.0;                                                       // = |r1-r2|

  // Evaluate shape functions at gauss points and store values
  get_shape_functions(N1_i, N1_i_xi, gausspoints);

  // evaluate charge density from DLINE charge condition specified in input file
  double q1 = chargeconds_[0]->parameters().get<double>("VAL");

  // read charge of rigid sphere; note: this is NOT a charge density but the total charge of the
  // sphere!!!
  double q2 = chargeconds_[1]->parameters().get<double>("VAL");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  auto function_number = chargeconds_[0]->parameters().get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    q1 *= Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
              .evaluate(time_);

  function_number = chargeconds_[1]->parameters().get<std::optional<int>>("FUNCT");

  if (function_number.has_value() and function_number.value() > 0)
    q2 *= Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
              .evaluate(time_);


  // auxiliary variable
  Core::LinAlg::Matrix<3, 1, TYPE> fpot_tmp(Core::LinAlg::Initialization::zero);

  // determine prefactor of the integral (depends on whether surface or volume potential is applied)
  double prefactor = k_ * m_;

  switch (params()->type)  // Todo do we need a own Beam-to-sphere potential type here?
  {
    case BeamInteraction::Potential::Type::surface:
      prefactor *= 2 * radius1_ * M_PI;
      break;
    case BeamInteraction::Potential::Type::volume:
      prefactor *= std::pow(radius1_, 2) * M_PI;
      break;
    default:
      FOUR_C_THROW("No valid TYPE specified. Choose either `surface` `volume` in input file!");
  }

  // get sphere midpoint position
  for (int i = 0; i < 3; ++i) r2(i) = ele2pos_(i);

  // reset interaction potential of this pair
  interaction_potential_ = 0.0;

  // loop over gauss points on ele1
  for (int gp1 = 0; gp1 < numgp; ++gp1)
  {
    compute_coords(r1, N1_i[gp1], ele1pos_);

    dist = Core::FADUtils::diff_vector(r1, r2);

    norm_dist = Core::FADUtils::vector_norm<3>(dist);

    // check cutoff criterion: if specified, contributions are neglected at larger separation
    if (cutoff_radius.has_value() and
        Core::FADUtils::cast_to_double(norm_dist) > cutoff_radius.value())
      continue;

    // auxiliary variables to store pre-calculated common terms
    TYPE norm_dist_exp1 = 0.0;
    if (norm_dist != 0.0)
    {
      norm_dist_exp1 = std::pow(norm_dist, -m_ - 2);
    }
    else
    {
      FOUR_C_THROW(
          "\n|r1-r2|=0 ! Interacting points are identical! Potential law not defined in this case!"
          " Think about shifting nodes in unconverged state?!");
    }

    double q1q2_JacFac_GaussWeights =
        q1 * q2 * beam_element()->get_jacobi_fac_at_xi(gausspoints.qxg[gp1][0]) *
        gausspoints.qwgt[gp1];

    // compute fpot_tmp here, same for both element forces
    for (unsigned int i = 0; i < 3; ++i)
      fpot_tmp(i) = q1q2_JacFac_GaussWeights * norm_dist_exp1 * dist(i);

    //********************************************************************
    // calculate fpot1: force on element 1
    //********************************************************************
    // sum up the contributions of all nodes (in all dimensions)
    for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
    {
      // loop over dimensions
      for (unsigned int j = 0; j < 3; ++j)
      {
        fpot1_(3 * i + j) -= N1_i[gp1](i) * fpot_tmp(j);
      }
    }

    //********************************************************************
    // calculate fpot2: force on element 2
    //********************************************************************
    // loop over dimensions
    for (unsigned int j = 0; j < 3; ++j)
    {
      fpot2_(j) += fpot_tmp(j);
    }


    //********************************************************************
    // calculate stiffpot1
    //********************************************************************
    // auxiliary variables (same for both elements)
    TYPE norm_dist_exp2 = (m_ + 2) * std::pow(norm_dist, -m_ - 4);

    Core::LinAlg::Matrix<3, 3, TYPE> dist_dist_T(Core::LinAlg::Initialization::zero);

    for (unsigned int i = 0; i < 3; ++i)
    {
      for (unsigned int j = 0; j <= i; ++j)
      {
        dist_dist_T(i, j) = dist(i) * dist(j);
        if (i != j) dist_dist_T(j, i) = dist_dist_T(i, j);
      }
    }

    for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
    {
      // d (Res_1) / d (d_1)
      for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
      {
        for (unsigned int idim = 0; idim < 3; ++idim)
        {
          stiffpot1_(3 * i + idim, 3 * j + idim) -=
              norm_dist_exp1 * N1_i[gp1](i) * N1_i[gp1](j) * q1q2_JacFac_GaussWeights;

          for (unsigned int jdim = 0; jdim < 3; ++jdim)
          {
            stiffpot1_(3 * i + idim, 3 * j + jdim) += norm_dist_exp2 * N1_i[gp1](i) *
                                                      dist_dist_T(idim, jdim) * N1_i[gp1](j) *
                                                      q1q2_JacFac_GaussWeights;
          }
        }
      }


      // d (Res_1) / d (d_2)
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffpot1_(3 * i + idim, 3 * (numnodes * numnodalvalues) + idim) +=
            norm_dist_exp1 * N1_i[gp1](i) * q1q2_JacFac_GaussWeights;

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffpot1_(3 * i + idim, 3 * (numnodes * numnodalvalues) + jdim) -=
              norm_dist_exp2 * N1_i[gp1](i) * dist_dist_T(idim, jdim) * q1q2_JacFac_GaussWeights;
        }
      }
    }

    //********************************************************************
    // calculate stiffpot2
    //********************************************************************
    // d (Res_2) / d (d_1)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffpot2_(idim, 3 * j + idim) += norm_dist_exp1 * N1_i[gp1](j) * q1q2_JacFac_GaussWeights;

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffpot2_(idim, 3 * j + jdim) -=
              norm_dist_exp2 * dist_dist_T(idim, jdim) * N1_i[gp1](j) * q1q2_JacFac_GaussWeights;
        }
      }
    }

    // d (Res_2) / d (d_2)
    for (unsigned int idim = 0; idim < 3; ++idim)
    {
      stiffpot2_(idim, 3 * (numnodes * numnodalvalues) + idim) -=
          norm_dist_exp1 * q1q2_JacFac_GaussWeights;

      for (unsigned int jdim = 0; jdim < 3; ++jdim)
      {
        stiffpot2_(idim, 3 * (numnodes * numnodalvalues) + jdim) +=
            norm_dist_exp2 * dist_dist_T(idim, jdim) * q1q2_JacFac_GaussWeights;
      }
    }

    // store for energy output
    interaction_potential_ += prefactor / m_ * q1q2_JacFac_GaussWeights *
                              std::pow(Core::FADUtils::cast_to_double(norm_dist), -m_);
  }


  // apply constant prefactor
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i) fpot1_(i) *= prefactor;
  for (unsigned int i = 0; i < 3; ++i) fpot2_(i) *= prefactor;

  stiffpot1_.scale(prefactor);
  stiffpot2_.scale(prefactor);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::print(
    std::ostream& out) const
{
  check_init_setup();

  out << "\nInstance of BeamToSpherePotentialPair (EleGIDs " << element1()->id() << " & "
      << element2()->id() << "):";
  out << "\nbeamele dofvec: " << ele1pos_;
  out << "\nspherele dofvec: " << ele2pos_;

  out << "\n";
  // Todo add more relevant information here
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes,
    numnodalvalues>::print_summary_one_line_per_active_segment_pair(std::ostream& out) const
{
  check_init_setup();

  // Todo difficulty here is that the same element pair is evaluated more than once
  //      to be more precise, once for every common potlaw;
  //      contribution of previous evaluations is overwritten if multiple potlaws are applied
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::get_shape_functions(
    std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues>>& N1_i,
    std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues>>& N1_i_xi,
    Core::FE::IntegrationPoints1D& gausspoints)
{
  // get discretization type
  const Core::FE::CellType distype1 = element1()->shape();

  if (numnodalvalues == 1)
  {
    for (int gp = 0; gp < gausspoints.nquad; ++gp)
    {
      // get values and derivatives of shape functions
      Core::FE::shape_function_1d(N1_i[gp], gausspoints.qxg[gp][0], distype1);
      Core::FE::shape_function_1d_deriv1(N1_i_xi[gp], gausspoints.qxg[gp][0], distype1);
    }
  }
  else if (numnodalvalues == 2)
  {
    /* TODO hard set distype to line2 in case of numnodalvalues_=2 because
     *  only 3rd order Hermite interpolation is used (always 2 nodes) */
    const Core::FE::CellType distype1herm = Core::FE::CellType::line2;

    for (int gp = 0; gp < gausspoints.nquad; ++gp)
    {
      // get values and derivatives of shape functions
      Core::FE::shape_function_hermite_1d(
          N1_i[gp], gausspoints.qxg[gp][0], beamele_reflength_, distype1herm);
      Core::FE::shape_function_hermite_1d_deriv1(
          N1_i_xi[gp], gausspoints.qxg[gp][0], beamele_reflength_, distype1herm);
    }
  }
  else
    FOUR_C_THROW(
        "Only beam elements with one (nodal positions) or two (nodal positions + nodal tangents)"
        " values are valid!");

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::compute_coords(
    Core::LinAlg::Matrix<3, 1, TYPE>& r,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues>& N_i,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, TYPE> elepos)
{
  r.clear();

  // compute output variable
  for (unsigned int i = 0; i < 3; i++)
  {
    for (unsigned int j = 0; j < numnodes * numnodalvalues; j++)
    {
      r(i) += N_i(j) * elepos(3 * j + i);
    }
  }

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BeamInteraction::BeamToSpherePotentialPair<numnodes, numnodalvalues>::reset_state(double time,
    const std::vector<double>& centerline_dofvec_ele1,
    const std::vector<double>& centerline_dofvec_ele2)
{
  time_ = time;

  if (centerline_dofvec_ele1.size() != 3 * numnodes * numnodalvalues)
    FOUR_C_THROW("size mismatch! expected {} values for centerline_dofvec_ele1, but got {}",
        3 * numnodes * numnodalvalues, centerline_dofvec_ele1.size());

  if (centerline_dofvec_ele2.size() != 3)
    FOUR_C_THROW("size mismatch! expected {} values for centerline_dofvec_ele2, but got {}", 3,
        centerline_dofvec_ele1.size());


  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele1pos_(i) = centerline_dofvec_ele1[i];

  for (unsigned int i = 0; i < 3; ++i) ele2pos_(i) = centerline_dofvec_ele2[i];
}


// Possible template cases: this is necessary for the compiler
template class BeamInteraction::BeamToSpherePotentialPair<2, 1>;
template class BeamInteraction::BeamToSpherePotentialPair<3, 1>;
template class BeamInteraction::BeamToSpherePotentialPair<4, 1>;
template class BeamInteraction::BeamToSpherePotentialPair<5, 1>;
template class BeamInteraction::BeamToSpherePotentialPair<2, 2>;

FOUR_C_NAMESPACE_CLOSE
