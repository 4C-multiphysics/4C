// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_potential_pair_beam_to_beam.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_beaminteraction_potential_geometry_utils.hpp"
#include "4C_beaminteraction_potential_input.hpp"
#include "4C_fem_general_largerotations.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_fad.hpp"
#include "4C_utils_function_of_time.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::BeamToBeamPotentialPair()
    : BeamPotentialPair(),
      beam_element1_(nullptr),
      beam_element2_(nullptr),
      time_(0.0),
      k_(0.0),
      m_(0.0),
      ele1length_(0.0),
      ele2length_(0.0),
      radius1_(0.0),
      radius2_(0.0),
      interaction_potential_(0.0)
{
  // empty constructor
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::setup()
{
  check_init();

  // call setup of base class first
  BeamPotentialPair::setup();


  ele1pos_.clear();
  ele2pos_.clear();

  /* take care of how to assign the role of target and source (only applicable to
   * "SingleLengthSpecific" approach). Immediately before this setup(), the element with smaller GID
   * has been assigned as element1_, i.e., source. */
  if (params()->choice_source_target ==
      BeamInteraction::Potential::SourceTargetChoice::higher_eleGID_is_source)
  {
    // interchange order, i.e., role of elements
    Core::Elements::Element const* tmp_ele_ptr = element1();
    set_element1(element2());
    set_element2(tmp_ele_ptr);
  }

  // get initial length of beam elements
  beam_element1_ = dynamic_cast<const Discret::Elements::Beam3Base*>(element1());
  ele1length_ = beam_element1()->ref_length();
  beam_element2_ = dynamic_cast<const Discret::Elements::Beam3Base*>(element2());
  ele2length_ = beam_element2()->ref_length();

  radius1_ = beam_element1()->get_circular_cross_section_radius_for_interactions();
  radius2_ = beam_element2()->get_circular_cross_section_radius_for_interactions();

  if (element1()->element_type() != element2()->element_type())
    FOUR_C_THROW(
        "The class BeamToBeamPotentialPair currently only supports element "
        "pairs of the same beam element type!");

  // initialize line charge conditions applied to element1 and element2
  linechargeconds_.resize(2);

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::evaluate(
    Core::LinAlg::SerialDenseVector* forcevec1, Core::LinAlg::SerialDenseVector* forcevec2,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22,
    const std::vector<const Core::Conditions::Condition*> linechargeconds, const double k,
    const double m)
{
  // no need to evaluate this pair in case of separation by far larger than cutoff or prefactor zero
  if ((params()->cutoff_radius.has_value() and
          are_elements_much_more_separated_than_cutoff_distance()) or
      k == 0.0)
    return false;


  // set class variables
  if (linechargeconds.size() == 2)
  {
    for (unsigned int i = 0; i < 2; ++i)
    {
      if (linechargeconds[i]->type() == Core::Conditions::BeamPotential_LineChargeDensity)
        linechargeconds_[i] = linechargeconds[i];
      else
        FOUR_C_THROW(
            "Provided line charge condition is not of correct type"
            "BeamPotential_LineChargeDensity!");
    }
  }
  else
    FOUR_C_THROW("Expected TWO dline charge conditions!");

  k_ = k;
  m_ = m;



  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3 * numnodes * numnodalvalues;

  Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot2(
      Core::LinAlg::Initialization::zero);


  if (stiffmat11 != nullptr) stiffmat11->shape(dim1, dim1);
  if (stiffmat12 != nullptr) stiffmat12->shape(dim1, dim2);
  if (stiffmat21 != nullptr) stiffmat21->shape(dim2, dim1);
  if (stiffmat22 != nullptr) stiffmat22->shape(dim2, dim2);

  if (params()->strategy !=
          BeamInteraction::Potential::Strategy::single_length_specific_small_separations_simple &&
      params()->two_half_pass)
    FOUR_C_THROW(
        "The two-halfway approach is only available for the "
        "single_length_specific_small_separations_simple strategy!");

  // compute the values for element residual vectors ('force') and linearizations ('stiff')
  switch (params()->strategy)
  {
    case BeamInteraction::Potential::Strategy::double_length_specific_large_separations:
    {
      evaluate_fpotand_stiffpot_double_length_specific_large_sep_approx(
          force_pot1, force_pot2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);
      break;
    }

    case BeamInteraction::Potential::Strategy::double_length_specific_small_separations:
    {
      evaluate_fpotand_stiffpot_double_length_specific_small_sep_approx(
          force_pot1, force_pot2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);
      break;
    }

    case BeamInteraction::Potential::Strategy::single_length_specific_small_separations:
    case BeamInteraction::Potential::Strategy::single_length_specific_small_separations_simple:
    {
      evaluate_fpotand_stiffpot_single_length_specific_small_sep_approx(
          force_pot1, force_pot2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);


      if (params()->two_half_pass)
      {
        exchange_source_target_allocation_for_two_half_pass();

        // evaluate switched pair contribution a second time for two-half-pass approach
        // interaction potential is averaged
        // Note: the residual vectors and stiffness matrices need to be switched to correctly
        // account for the changed target/source role
        evaluate_fpotand_stiffpot_single_length_specific_small_sep_approx(
            force_pot2, force_pot1, stiffmat22, stiffmat21, stiffmat12, stiffmat11);

        // revert to original target/source allocation (necessary for next evaluation within same
        // time step)
        exchange_source_target_allocation_for_two_half_pass();
      }
      break;
    }

    default:
      FOUR_C_THROW("Invalid strategy to evaluate beam interaction potential!");
  }


  // resize variables and fill with pre-computed values
  if (forcevec1 != nullptr)
  {
    forcevec1->size(dim1);
    for (unsigned int i = 0; i < dim1; ++i)
      (*forcevec1)(i) = Core::FADUtils::cast_to_double(force_pot1(i));
  }
  if (forcevec2 != nullptr)
  {
    forcevec2->size(dim2);
    for (unsigned int i = 0; i < dim2; ++i)
      (*forcevec2)(i) = Core::FADUtils::cast_to_double(force_pot2(i));
  }

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    evaluate_fpotand_stiffpot_double_length_specific_large_sep_approx(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot2,
        Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
        Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // prepare differentiation via FAD if desired
  set_automatic_differentiation_variables_if_required(ele1pos_, ele2pos_);

  // get cutoff radius
  const std::optional<double> cutoff_radius = params()->cutoff_radius;

  // number of integration segments per element
  const unsigned int n_integration_segments = params()->n_integration_segments;

  // Set Gauss integration rule applied in each integration segment
  Core::FE::GaussRule1D gaussrule = get_gauss_rule();

  // Get Gauss points (gp) for integration
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);
  // number of Gauss points per integration segment and in total per element
  int numgp_persegment = gausspoints.nquad;
  int numgp_perelement = n_integration_segments * numgp_persegment;

  // vectors for shape functions
  // Attention: these are individual shape function values, NOT shape function matrices
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double>> N1_i(numgp_persegment);
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double>> N2_i(numgp_persegment);

  // Evaluate shape functions at gauss points and store values
  // Todo think about pre-computing and storing values for inner Gauss point loops here

  // coords of the two gauss points
  Core::LinAlg::Matrix<3, 1, T> r1(Core::LinAlg::Initialization::zero);    // = r1
  Core::LinAlg::Matrix<3, 1, T> r2(Core::LinAlg::Initialization::zero);    // = r2
  Core::LinAlg::Matrix<3, 1, T> dist(Core::LinAlg::Initialization::zero);  // = r1-r2
  T norm_dist = 0.0;                                                       // = |r1-r2|

  // evaluate charge densities from DLINE charge condition specified in input file
  double q1 = linechargeconds_[0]->parameters().template get<double>("VAL");
  double q2 = linechargeconds_[1]->parameters().template get<double>("VAL");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  auto function_number =
      linechargeconds_[0]->parameters().template get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    q1 *= Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
              .evaluate(time_);

  function_number = linechargeconds_[1]->parameters().template get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    q2 *= Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
              .evaluate(time_);


  // auxiliary variable
  Core::LinAlg::Matrix<3, 1, T> fpot_tmp(Core::LinAlg::Initialization::zero);

  // determine prefactor of the integral (depends on whether surface or volume potential is applied)
  double prefactor = k_ * m_;

  switch (params()->type)
  {
    case BeamInteraction::Potential::Type::surface:
      prefactor *= 4 * radius1_ * radius2_ * std::numbers::pi * std::numbers::pi;
      break;
    case BeamInteraction::Potential::Type::volume:
      prefactor *= radius1_ * radius1_ * radius2_ * radius2_ * std::numbers::pi * std::numbers::pi;
      break;
    default:
      FOUR_C_THROW("No valid TYPE specified. Choose either `surface` or `volume` in input file!");
  }

  // prepare data storage for visualization
  centerline_coords_gp_1_.resize(numgp_perelement);
  centerline_coords_gp_2_.resize(numgp_perelement);
  forces_pot_gp_1_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));
  forces_pot_gp_2_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));
  // note: no moments appear in this approach, vectors are initialized to zero
  moments_pot_gp_1_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));
  moments_pot_gp_2_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));

  for (unsigned int isegment1 = 0; isegment1 < n_integration_segments; ++isegment1)
  {
    // compute element parameter coordinate for lower and upper limit of current integration segment
    double integration_segment1_lower_limit = -1.0 + isegment1 * 2.0 / n_integration_segments;
    double integration_segment1_upper_limit = -1.0 + (isegment1 + 1) * 2.0 / n_integration_segments;

    double jacobifactor_segment1 =
        0.5 * (integration_segment1_upper_limit - integration_segment1_lower_limit);

    Discret::Utils::Beam::evaluate_shape_functions_all_gps<numnodes, numnodalvalues>(gausspoints,
        N1_i, beam_element1()->shape(), ele1length_, integration_segment1_lower_limit,
        integration_segment1_upper_limit);

    for (unsigned int isegment2 = 0; isegment2 < n_integration_segments; ++isegment2)
    {
      // compute element parameter coordinate for lower and upper limit of current integration
      // segment
      double integration_segment2_lower_limit = -1.0 + isegment2 * 2.0 / n_integration_segments;
      double integration_segment2_upper_limit =
          -1.0 + (isegment2 + 1) * 2.0 / n_integration_segments;

      double jacobifactor_segment2 =
          0.5 * (integration_segment2_upper_limit - integration_segment2_lower_limit);

      Discret::Utils::Beam::evaluate_shape_functions_all_gps<numnodes, numnodalvalues>(gausspoints,
          N2_i, beam_element2()->shape(), ele2length_, integration_segment2_lower_limit,
          integration_segment2_upper_limit);


      // loop over Gauss points in current segment on ele1
      for (int igp1 = 0; igp1 < numgp_persegment; ++igp1)
      {
        int igp1_total = isegment1 * numgp_persegment + igp1;

        // Get location of GP in element parameter space xi \in [-1;1]
        const double xi_GP1_tilde = gausspoints.qxg[igp1][0];

        /* do a mapping into integration segment, i.e. coordinate transformation
         * note: this has no effect if integration interval is [-1;1] */
        const double xi_GP1 = 0.5 * ((1.0 - xi_GP1_tilde) * integration_segment1_lower_limit +
                                        (1.0 + xi_GP1_tilde) * integration_segment1_upper_limit);

        compute_centerline_position(r1, N1_i[igp1], ele1pos_);

        // store for visualization
        centerline_coords_gp_1_[igp1_total] = Core::FADUtils::cast_to_double<T, 3, 1>(r1);

        double jacobifac1 = beam_element1()->get_jacobi_fac_at_xi(xi_GP1);

        // loop over Gauss points in current segment on ele2
        for (int igp2 = 0; igp2 < numgp_persegment; ++igp2)
        {
          int igp2_total = isegment2 * numgp_persegment + igp2;

          // Get location of GP in element parameter space xi \in [-1;1]
          const double xi_GP2_tilde = gausspoints.qxg[igp2][0];

          /* do a mapping into integration segment, i.e. coordinate transformation
           * note: this has no effect if integration interval is [-1;1] */
          const double xi_GP2 = 0.5 * ((1.0 - xi_GP2_tilde) * integration_segment2_lower_limit +
                                          (1.0 + xi_GP2_tilde) * integration_segment2_upper_limit);

          // compute coord vector
          compute_centerline_position(r2, N2_i[igp2], ele2pos_);

          // store for visualization
          centerline_coords_gp_2_[igp2_total] = Core::FADUtils::cast_to_double<T, 3, 1>(r2);

          double jacobifac2 = beam_element2()->get_jacobi_fac_at_xi(xi_GP2);

          dist = Core::FADUtils::diff_vector(r1, r2);

          norm_dist = Core::FADUtils::vector_norm(dist);

          // check cutoff criterion: if specified, contributions are neglected at larger separation
          if (cutoff_radius.has_value() and
              Core::FADUtils::cast_to_double(norm_dist) > cutoff_radius.value())
            continue;

          // auxiliary variables to store pre-calculated common terms
          T norm_dist_exp1 = 0.0;
          if (norm_dist != 0.0)
          {
            norm_dist_exp1 = std::pow(norm_dist, -m_ - 2);
          }
          else
          {
            FOUR_C_THROW(
                "\n|r1-r2|=0 ! Interacting points are identical! Potential law not defined in this"
                " case! Think about shifting nodes in unconverged state?!");
          }

          double q1q2_JacFac_GaussWeights = q1 * q2 * jacobifac1 * jacobifactor_segment1 *
                                            jacobifac2 * jacobifactor_segment2 *
                                            gausspoints.qwgt[igp1] * gausspoints.qwgt[igp2];

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
              force_pot1(3 * i + j) -= N1_i[igp1](i) * fpot_tmp(j);
            }
          }

          //********************************************************************
          // calculate fpot2: force on element 2
          //********************************************************************
          // sum up the contributions of all nodes (in all dimensions)
          for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
          {
            // loop over dimensions
            for (unsigned int j = 0; j < 3; ++j)
            {
              force_pot2(3 * i + j) += N2_i[igp2](i) * fpot_tmp(j);
            }
          }

          // evaluate analytic contributions to linearization
          if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
              stiffmat22 != nullptr)
          {
            evaluate_stiffpot_analytic_contributions_large_sep_approx(dist, norm_dist,
                norm_dist_exp1, q1q2_JacFac_GaussWeights, N1_i[igp1], N2_i[igp2], *stiffmat11,
                *stiffmat12, *stiffmat21, *stiffmat22);
          }

          // store for visualization
          forces_pot_gp_1_[igp1_total].update(
              1.0 * prefactor * q1 * q2 * Core::FADUtils::cast_to_double(norm_dist_exp1) *
                  jacobifac2 * jacobifactor_segment2 * gausspoints.qwgt[igp2],
              Core::FADUtils::cast_to_double<T, 3, 1>(dist), 1.0);
          forces_pot_gp_2_[igp2_total].update(
              -1.0 * prefactor * q1 * q2 * Core::FADUtils::cast_to_double(norm_dist_exp1) *
                  jacobifac1 * jacobifactor_segment1 * gausspoints.qwgt[igp1],
              Core::FADUtils::cast_to_double<T, 3, 1>(dist), 1.0);

          // store for energy output
          interaction_potential_ += prefactor / m_ * q1q2_JacFac_GaussWeights *
                                    std::pow(Core::FADUtils::cast_to_double(norm_dist), -m_);

        }  // end gauss quadrature loop (element 2)
      }  // end gauss quadrature loop (element 1)

    }  // end: loop over integration segments of element 2
  }  // end: loop over integration segments of element 1

  // apply constant prefactor
  force_pot1.scale(prefactor);
  force_pot2.scale(prefactor);

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    scale_stiffpot_analytic_contributions_if_required(
        prefactor, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);

    calc_stiffmat_automatic_differentiation_if_required(
        force_pot1, force_pot2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::evaluate_stiffpot_analytic_contributions_large_sep_approx(Core::LinAlg::Matrix<3, 1,
                                                                      double> const& dist,
    double const& norm_dist, double const& norm_dist_exp1, double q1q2_JacFac_GaussWeights,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N1_i_GP1,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N2_i_GP2,
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22) const
{
  //********************************************************************
  // calculate stiffpot1
  //********************************************************************
  // auxiliary variables (same for both elements)
  double norm_dist_exp2 = (m_ + 2) * std::pow(norm_dist, -m_ - 4);

  Core::LinAlg::Matrix<3, 3, double> dist_dist_T(Core::LinAlg::Initialization::zero);

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
        stiffmat11(3 * i + idim, 3 * j + idim) -=
            norm_dist_exp1 * N1_i_GP1(i) * N1_i_GP1(j) * q1q2_JacFac_GaussWeights;

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat11(3 * i + idim, 3 * j + jdim) += norm_dist_exp2 * N1_i_GP1(i) *
                                                    dist_dist_T(idim, jdim) * N1_i_GP1(j) *
                                                    q1q2_JacFac_GaussWeights;
        }
      }
    }

    // d (Res_1) / d (d_2)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffmat12(3 * i + idim, 3 * j + idim) +=
            norm_dist_exp1 * N1_i_GP1(i) * N2_i_GP2(j) * q1q2_JacFac_GaussWeights;

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat12(3 * i + idim, 3 * j + jdim) -= norm_dist_exp2 * N1_i_GP1(i) *
                                                    dist_dist_T(idim, jdim) * N2_i_GP2(j) *
                                                    q1q2_JacFac_GaussWeights;
        }
      }
    }
  }

  //********************************************************************
  // calculate stiffpot2
  //********************************************************************
  for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
  {
    // d (Res_2) / d (d_1)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffmat21(3 * i + idim, 3 * j + idim) +=
            norm_dist_exp1 * N2_i_GP2(i) * N1_i_GP1(j) * q1q2_JacFac_GaussWeights;

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat21(3 * i + idim, 3 * j + jdim) -= norm_dist_exp2 * N2_i_GP2(i) *
                                                    dist_dist_T(idim, jdim) * N1_i_GP1(j) *
                                                    q1q2_JacFac_GaussWeights;
        }
      }
    }

    // d (Res_2) / d (d_2)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffmat22(3 * i + idim, 3 * j + idim) -=
            norm_dist_exp1 * N2_i_GP2(i) * N2_i_GP2(j) * q1q2_JacFac_GaussWeights;

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat22(3 * i + idim, 3 * j + jdim) += norm_dist_exp2 * N2_i_GP2(i) *
                                                    dist_dist_T(idim, jdim) * N2_i_GP2(j) *
                                                    q1q2_JacFac_GaussWeights;
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    evaluate_fpotand_stiffpot_double_length_specific_small_sep_approx(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot2,
        Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
        Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // safety check
  if (m_ < 3.5)
    FOUR_C_THROW(
        "This strategy to evaluate the interaction potential is not applicable for exponents "
        "of the point potential law smaller than 3.5!");

  // prepare differentiation via FAD if desired
  set_automatic_differentiation_variables_if_required(ele1pos_, ele2pos_);

  // get cutoff radius
  const std::optional<double> cutoff_radius = params()->cutoff_radius;

  // get regularization type and separation
  const BeamInteraction::Potential::RegularizationType regularization_type =
      params()->regularization.type;

  const double regularization_separation = params()->regularization.separation;

  // number of integration segments per element
  const unsigned int n_integration_segments = params()->n_integration_segments;

  // Set Gauss integration rule applied in each integration segment
  Core::FE::GaussRule1D gaussrule = get_gauss_rule();

  // Get Gauss points (gp) for integration
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);
  // number of Gauss points per integration segment and in total per element
  int numgp_persegment = gausspoints.nquad;
  int numgp_perelement = n_integration_segments * numgp_persegment;

  // vectors for shape function values
  // Attention: these are individual shape function values, NOT shape function matrices
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double>> N1_i(numgp_persegment);
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double>> N2_i(numgp_persegment);

  // Evaluate shape functions at gauss points and store values
  // Todo think about pre-computing and storing values for inner Gauss point loops here

  // coords of the two gauss points
  Core::LinAlg::Matrix<3, 1, T> r1(Core::LinAlg::Initialization::zero);    // = r1
  Core::LinAlg::Matrix<3, 1, T> r2(Core::LinAlg::Initialization::zero);    // = r2
  Core::LinAlg::Matrix<3, 1, T> dist(Core::LinAlg::Initialization::zero);  // = r1-r2
  T norm_dist = 0.0;                                                       // = |r1-r2|
  T gap = 0.0;                                                             // = |r1-r2|-R1-R2
  T gap_regularized = 0.0;  // modified gap if a regularization of the force law is applied

  // evaluate charge/particle densities from DLINE charge condition specified in input file
  double q1 = linechargeconds_[0]->parameters().template get<double>("VAL");
  double q2 = linechargeconds_[1]->parameters().template get<double>("VAL");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  auto function_number =
      linechargeconds_[0]->parameters().template get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    q1 *= Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
              .evaluate(time_);

  function_number = linechargeconds_[1]->parameters().template get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    q2 *= Global::Problem::instance()
              ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
              .evaluate(time_);


  // Evaluation of the Gamma-Function term:
  // gamma(nue-3.5)*gamma(0.5*(nue-1))/gamma(nue-2)/gamma(0.5*nue-1)
  double C = 0.0;

  // safety check via split of exponent m in integer and fractional part
  double integerpart;
  if (std::modf(m_, &integerpart) != 0.0)
    FOUR_C_THROW("You specified a non-integer exponent of the point potential law!");

  switch ((int)m_)
  {
    case 4:
      C = 1.570796326794897;
      break;
    case 5:
      C = 0.5;
      break;
    case 6:
      C = 0.294524311274043;
      break;
    case 7:
      C = 0.208333333333333;
      break;
    case 8:
      C = 0.161067982727992;
      break;
    case 9:
      C = 0.13125;
      break;
    case 10:
      C = 0.110734238125495;
      break;
    case 11:
      C = 0.095758928571429;
      break;
    case 12:
      C = 0.084348345447154;
      break;
    default:
      FOUR_C_THROW("Gamma-Function values not known for this exponent m of potential law");
      break;
  }



  // prepare data storage for visualization
  centerline_coords_gp_1_.resize(numgp_perelement);
  centerline_coords_gp_2_.resize(numgp_perelement);
  forces_pot_gp_1_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));
  forces_pot_gp_2_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));
  // note: no moments appear in this approach, vectors are initialized to zero
  moments_pot_gp_1_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));
  moments_pot_gp_2_.resize(
      numgp_perelement, Core::LinAlg::Matrix<3, 1, double>(Core::LinAlg::Initialization::zero));

  // auxiliary variables
  Core::LinAlg::Matrix<3, 1, T> fpot_tmp(Core::LinAlg::Initialization::zero);

  double prefactor = k_ * 2 * std::numbers::pi * (m_ - 3.5) / (m_ - 2) / (m_ - 2) *
                     std::sqrt(2 * radius1_ * radius2_ / (radius1_ + radius2_)) * C;

  for (unsigned int isegment1 = 0; isegment1 < n_integration_segments; ++isegment1)
  {
    // compute element parameter coordinate for lower and upper limit of current integration segment
    double integration_segment1_lower_limit = -1.0 + isegment1 * 2.0 / n_integration_segments;
    double integration_segment1_upper_limit = -1.0 + (isegment1 + 1) * 2.0 / n_integration_segments;

    double jacobifactor_segment1 =
        0.5 * (integration_segment1_upper_limit - integration_segment1_lower_limit);

    Discret::Utils::Beam::evaluate_shape_functions_all_gps<numnodes, numnodalvalues>(gausspoints,
        N1_i, beam_element1()->shape(), ele1length_, integration_segment1_lower_limit,
        integration_segment1_upper_limit);

    for (unsigned int isegment2 = 0; isegment2 < n_integration_segments; ++isegment2)
    {
      // compute element parameter coordinate for lower and upper limit of current integration
      // segment
      double integration_segment2_lower_limit = -1.0 + isegment2 * 2.0 / n_integration_segments;
      double integration_segment2_upper_limit =
          -1.0 + (isegment2 + 1) * 2.0 / n_integration_segments;

      double jacobifactor_segment2 =
          0.5 * (integration_segment2_upper_limit - integration_segment2_lower_limit);

      Discret::Utils::Beam::evaluate_shape_functions_all_gps<numnodes, numnodalvalues>(gausspoints,
          N2_i, beam_element2()->shape(), ele2length_, integration_segment2_lower_limit,
          integration_segment2_upper_limit);

      // loop over gauss points of current segment on element 1
      for (int igp1 = 0; igp1 < numgp_persegment; ++igp1)
      {
        int igp1_total = isegment1 * numgp_persegment + igp1;

        // Get location of GP in element parameter space xi \in [-1;1]
        const double xi_GP1_tilde = gausspoints.qxg[igp1][0];

        /* do a mapping into integration segment, i.e. coordinate transformation
         * note: this has no effect if integration interval is [-1;1] */
        const double xi_GP1 = 0.5 * ((1.0 - xi_GP1_tilde) * integration_segment1_lower_limit +
                                        (1.0 + xi_GP1_tilde) * integration_segment1_upper_limit);

        // compute coord vector
        compute_centerline_position(r1, N1_i[igp1], ele1pos_);

        // store for visualization
        centerline_coords_gp_1_[igp1_total] = Core::FADUtils::cast_to_double<T, 3, 1>(r1);

        double jacobifac1 = beam_element1()->get_jacobi_fac_at_xi(xi_GP1);

        // loop over gauss points of current segment on element 2
        for (int igp2 = 0; igp2 < numgp_persegment; ++igp2)
        {
          int igp2_total = isegment2 * numgp_persegment + igp2;

          // Get location of GP in element parameter space xi \in [-1;1]
          const double xi_GP2_tilde = gausspoints.qxg[igp2][0];

          /* do a mapping into integration segment, i.e. coordinate transformation
           * note: this has no effect if integration interval is [-1;1] */
          const double xi_GP2 = 0.5 * ((1.0 - xi_GP2_tilde) * integration_segment2_lower_limit +
                                          (1.0 + xi_GP2_tilde) * integration_segment2_upper_limit);

          // compute coord vector
          compute_centerline_position(r2, N2_i[igp2], ele2pos_);

          // store for visualization
          centerline_coords_gp_2_[igp2_total] = Core::FADUtils::cast_to_double<T, 3, 1>(r2);

          double jacobifac2 = beam_element2()->get_jacobi_fac_at_xi(xi_GP2);

          dist = Core::FADUtils::diff_vector(r1, r2);

          norm_dist = Core::FADUtils::vector_norm(dist);

          if (norm_dist == 0.0)
          {
            this->print(std::cout);
            std::cout << "\nGP pair: igp1_total=" << igp1_total << " & igp2_total=" << igp2_total
                      << ": |r1-r2|=" << norm_dist;

            FOUR_C_THROW("centerline separation |r1-r2|=0!");
          }

          // check cutoff criterion: if specified, contributions are neglected at larger separation
          if (cutoff_radius.has_value() and
              Core::FADUtils::cast_to_double(norm_dist) > cutoff_radius.value())
            continue;

          gap = norm_dist - radius1_ - radius2_;



          if (regularization_type == BeamInteraction::Potential::RegularizationType::none and
              gap <= 0.0)
          {
            this->print(std::cout);
            std::cout << "\nGP pair: igp1_total=" << igp1_total << " & igp2_total=" << igp2_total
                      << ": gap=" << gap;

            FOUR_C_THROW(
                "gap<=0! Force law resulting from specified interaction potential law is "
                "not defined for zero/negative gaps! Use/implement a regularization!");
          }

          gap_regularized = gap;

          if ((regularization_type == BeamInteraction::Potential::RegularizationType::constant or
                  regularization_type == BeamInteraction::Potential::RegularizationType::linear) and
              gap < regularization_separation)
          {
            gap_regularized = regularization_separation;
          }

          if (gap_regularized <= 0)
            FOUR_C_THROW(
                "regularized gap <= 0! Fatal error since force law is not defined for "
                "zero/negative gaps! Use positive regularization separation!");


          // auxiliary variables to store pre-calculated common terms
          // Todo: a more intuitive, reasonable auxiliary quantity would be the scalar force
          //    which is equivalent to gap_exp1 apart from the constant factor "prefactor"
          T gap_exp1 = std::pow(gap_regularized, -m_ + 2.5);

          double q1q2_JacFac_GaussWeights = q1 * q2 * jacobifac1 * jacobifactor_segment1 *
                                            jacobifac2 * jacobifactor_segment2 *
                                            gausspoints.qwgt[igp1] * gausspoints.qwgt[igp2];

          // store for energy output
          interaction_potential_ +=
              prefactor / (m_ - 3.5) * q1q2_JacFac_GaussWeights *
              std::pow(Core::FADUtils::cast_to_double(gap_regularized), -m_ + 3.5);

          if ((regularization_type == BeamInteraction::Potential::RegularizationType::constant or
                  regularization_type == BeamInteraction::Potential::RegularizationType::linear) and
              gap < regularization_separation)
          {
            // potential law is linear in the regime of constant extrapolation of force law
            // and quadratic in case of linear extrapolation
            // add the linear contribution from this part of the force law
            interaction_potential_ +=
                prefactor * q1q2_JacFac_GaussWeights * Core::FADUtils::cast_to_double(gap_exp1) *
                (regularization_separation - Core::FADUtils::cast_to_double(gap));
          }


          if (regularization_type == BeamInteraction::Potential::RegularizationType::linear and
              gap < regularization_separation)
          {
            // Todo: a more intuitive, reasonable auxiliary quantity would be the derivative of the
            //    scalar force which is equivalent to gap_exp2 apart from the constant factors
            //    "prefactor" and -(m_ - 2.5) ?!
            T gap_exp2 = std::pow(gap_regularized, -m_ + 1.5);

            gap_exp1 += (m_ - 2.5) * gap_exp2 * (regularization_separation - gap);

            // add the quadratic contribution from this part of the force law
            interaction_potential_ +=
                prefactor * q1q2_JacFac_GaussWeights * 0.5 * (m_ - 2.5) *
                Core::FADUtils::cast_to_double(gap_exp2) *
                (regularization_separation - Core::FADUtils::cast_to_double(gap)) *
                (regularization_separation - Core::FADUtils::cast_to_double(gap));
          }

          // auxiliary term, same for both element forces
          for (unsigned int i = 0; i < 3; i++)
            fpot_tmp(i) = q1q2_JacFac_GaussWeights * dist(i) / norm_dist * gap_exp1;


          //********************************************************************
          // calculate fpot1: force on element 1
          //********************************************************************
          // sum up the contributions of all nodes (in all dimensions)
          for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
          {
            // loop over dimensions
            for (unsigned int j = 0; j < 3; ++j)
            {
              force_pot1(3 * i + j) -= N1_i[igp1](i) * fpot_tmp(j);
            }
          }

          //********************************************************************
          // calculate fpot2: force on element 2
          //********************************************************************
          // sum up the contributions of all nodes (in all dimensions)
          for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
          {
            // loop over dimensions
            for (unsigned int j = 0; j < 3; ++j)
            {
              force_pot2(3 * i + j) += N2_i[igp2](i) * fpot_tmp(j);
            }
          }

          // evaluate analytic contributions to linearization
          if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
              stiffmat22 != nullptr)
          {
            evaluate_stiffpot_analytic_contributions_double_length_specific_small_sep_approx(dist,
                norm_dist, gap, gap_regularized, gap_exp1, q1q2_JacFac_GaussWeights, N1_i[igp1],
                N2_i[igp2], *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
          }

          // store for visualization
          forces_pot_gp_1_[igp1_total].update(
              1.0 * prefactor * q1 * q2 * Core::FADUtils::cast_to_double(gap_exp1) /
                  Core::FADUtils::cast_to_double(norm_dist) * jacobifac2 * jacobifactor_segment2 *
                  gausspoints.qwgt[igp2],
              Core::FADUtils::cast_to_double<T, 3, 1>(dist), 1.0);
          forces_pot_gp_2_[igp2_total].update(
              -1.0 * prefactor * q1 * q2 * Core::FADUtils::cast_to_double(gap_exp1) /
                  Core::FADUtils::cast_to_double(norm_dist) * jacobifac1 * jacobifactor_segment1 *
                  gausspoints.qwgt[igp1],
              Core::FADUtils::cast_to_double<T, 3, 1>(dist), 1.0);

        }  // end: loop over gauss points of element 2
      }  // end: loop over gauss points of element 1

    }  // end: loop over integration segments of element 2
  }  // end: loop over integration segments of element 1

  // apply constant prefactor
  force_pot1.scale(prefactor);
  force_pot2.scale(prefactor);

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    scale_stiffpot_analytic_contributions_if_required(
        prefactor, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);

    calc_stiffmat_automatic_differentiation_if_required(
        force_pot1, force_pot2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    evaluate_stiffpot_analytic_contributions_double_length_specific_small_sep_approx(
        Core::LinAlg::Matrix<3, 1, double> const& dist, double const& norm_dist, double const& gap,
        double const& gap_regularized, double const& gap_exp1, double q1q2_JacFac_GaussWeights,
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N1_i_GP1,
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N2_i_GP2,
        Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
        Core::LinAlg::SerialDenseMatrix& stiffmat21,
        Core::LinAlg::SerialDenseMatrix& stiffmat22) const
{
  //********************************************************************
  // calculate stiffpot1
  //********************************************************************
  // auxiliary variables (same for both elements)
  double gap_exp2 = std::pow(gap_regularized, -m_ + 1.5);

  if (params()->regularization.type == BeamInteraction::Potential::RegularizationType::constant and
      gap < params()->regularization.separation)
  {
    /* in case of constant extrapolation of force law, the derivative of the force is zero
     * and this contribution to the stiffness matrix vanishes */
    gap_exp2 = 0.0;
  }

  double aux_fac1 = gap_exp1 / norm_dist * q1q2_JacFac_GaussWeights;
  double aux_fac2 = (gap_exp1 / norm_dist / norm_dist / norm_dist +
                        (m_ - 2.5) * gap_exp2 / norm_dist / norm_dist) *
                    q1q2_JacFac_GaussWeights;

  Core::LinAlg::Matrix<3, 3, double> dist_dist_T(Core::LinAlg::Initialization::zero);

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
        stiffmat11(3 * i + idim, 3 * j + idim) -= aux_fac1 * N1_i_GP1(i) * N1_i_GP1(j);

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat11(3 * i + idim, 3 * j + jdim) +=
              aux_fac2 * N1_i_GP1(i) * dist_dist_T(idim, jdim) * N1_i_GP1(j);
        }
      }
    }

    // d (Res_1) / d (d_2)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffmat12(3 * i + idim, 3 * j + idim) += aux_fac1 * N1_i_GP1(i) * N2_i_GP2(j);

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat12(3 * i + idim, 3 * j + jdim) -=
              aux_fac2 * N1_i_GP1(i) * dist_dist_T(idim, jdim) * N2_i_GP2(j);
        }
      }
    }
  }

  //********************************************************************
  // calculate stiffpot2
  //********************************************************************
  for (unsigned int i = 0; i < (numnodes * numnodalvalues); ++i)
  {
    // d (Res_2) / d (d_1)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffmat21(3 * i + idim, 3 * j + idim) += aux_fac1 * N2_i_GP2(i) * N1_i_GP1(j);

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat21(3 * i + idim, 3 * j + jdim) -=
              aux_fac2 * N2_i_GP2(i) * dist_dist_T(idim, jdim) * N1_i_GP1(j);
        }
      }
    }

    // d (Res_2) / d (d_2)
    for (unsigned int j = 0; j < (numnodes * numnodalvalues); ++j)
    {
      for (unsigned int idim = 0; idim < 3; ++idim)
      {
        stiffmat22(3 * i + idim, 3 * j + idim) -= aux_fac1 * N2_i_GP2(i) * N2_i_GP2(j);

        for (unsigned int jdim = 0; jdim < 3; ++jdim)
        {
          stiffmat22(3 * i + idim, 3 * j + jdim) +=
              aux_fac2 * N2_i_GP2(i) * dist_dist_T(idim, jdim) * N2_i_GP2(j);
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    evaluate_fpotand_stiffpot_single_length_specific_small_sep_approx(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot2,
        Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
        Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // safety checks
  if (m_ < 6.0)
    FOUR_C_THROW(
        "Invalid exponent m={}. The strategy 'SingleLengthSpecific_SmallSepApprox' to evaluate the "
        "interaction potential is only applicable for exponents m>=6 of the point potential law, "
        "e.g. van der Waals (m=6) or the repulsive part of Lennard-Jones (m=12)!",
        m_);

  if (not params()->automatic_differentiation and
      params()->strategy ==
          BeamInteraction::Potential::Strategy::single_length_specific_small_separations)
  {
    FOUR_C_THROW(
        "The strategy 'SingleLengthSpecific_SmallSepApprox' to evaluate the interaction "
        "potential requires automatic differentiation via FAD!");
  }

  if (params()->strategy ==
          BeamInteraction::Potential::Strategy::single_length_specific_small_separations &&
      params()->potential_reduction_length.has_value())
  {
    FOUR_C_THROW(
        "The potential reduction strategy is currently not implemented for the beam interaction "
        "strategy 'SingleLengthSpecific_SmallSepApprox'!");
  }

  if (radius1_ != radius2_)
    FOUR_C_THROW(
        "The strategy 'SingleLengthSpecific_SmallSepApprox' to evaluate the interaction "
        "potential requires the beam radii to be identical!");

  // get cutoff radius
  const std::optional<double> cutoff_radius = params()->cutoff_radius;

  // get potential reduction length
  const std::optional<double> potential_reduction_length = params()->potential_reduction_length;

  // get length from current target beam element to beam edge
  double length_prior_left = 0.0;
  double length_prior_right = 0.0;

  if (potential_reduction_length.has_value())
  {
    std::tie(length_prior_left, length_prior_right) =
        params()->ele_gid_prior_length_map.at(element2()->id());

    if ((length_prior_left >= 0.0) && (length_prior_right >= 0.0) &&
        ((ele2length_ - 2 * potential_reduction_length.value() + length_prior_left +
             length_prior_right) < 0.0))
    {
      FOUR_C_THROW(
          "ERROR: Target beam endpoint reduction strategy would interfere on current target beam "
          "element! Potential reduction would overlap due to too small prior element lengths on "
          "both sides");
    }
  }

  /* parameter coordinate of the closest point on the target beam,
   * determined via point-to-curve projection */
  T xi_target = 0.0;

  // prepare differentiation via FAD if desired
  /* xi_target is used as additional, auxiliary primary Dof
   * since there is no closed-form expression for how xi_target depends on the 'real' primary Dofs.
   * It is determined iteratively via point-to-curve projection */
  set_automatic_differentiation_variables_if_required(ele1pos_, ele2pos_, xi_target);


  // number of integration segments per element
  const unsigned int n_integration_segments = params()->n_integration_segments;

  // Set Gauss integration rule applied in each integration segment
  Core::FE::GaussRule1D gaussrule = get_gauss_rule();

  // Get Gauss points (gp) for integration
  Core::FE::IntegrationPoints1D gausspoints(gaussrule);

  // number of Gauss points per integration segment and in total
  int numgp_persegment = gausspoints.nquad;
  int numgp_total = n_integration_segments * numgp_persegment;

  double xi_GP_tilde = 0.0;
  double xi_GP = 0.0;

  double rho1rho2_JacFac_GaussWeight = 0.0;

  const unsigned int num_initial_values = 9;
  const std::array<double, num_initial_values> xi_target_initial_guess_values = {
      0.0, -1.0, 1.0, -0.5, 0.5, -0.75, -0.25, 0.25, 0.75};

  unsigned int iter_projection = 0;

  // vectors for shape functions and their derivatives
  // Attention: these are individual shape function values, NOT shape function matrices
  // values at all Gauss points are stored in advance (more efficient espec. for many segments)
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double>> N_i_source(
      numgp_persegment);  // = N1_i
  std::vector<Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double>> N_i_xi_source(
      numgp_persegment);  // = N1_i,xi

  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> N_i_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> N_i_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> N_i_xixi_target(
      Core::LinAlg::Initialization::zero);

  // assembled shape function matrices: Todo maybe avoid these matrices
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, double> N_source(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, T> N_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, T> N_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, T> N_xixi_target(
      Core::LinAlg::Initialization::zero);


  // coords and derivatives of the source Gauss point and projected point on target element
  Core::LinAlg::Matrix<3, 1, T> r_source(
      Core::LinAlg::Initialization::zero);  // centerline position vector on source
  Core::LinAlg::Matrix<3, 1, T> r_xi_source(
      Core::LinAlg::Initialization::zero);  // centerline tangent vector on source
  Core::LinAlg::Matrix<3, 1, T> t_source(
      Core::LinAlg::Initialization::zero);  // unit centerline tangent vector on source
  Core::LinAlg::Matrix<3, 1, T> r_target(
      Core::LinAlg::Initialization::zero);  // centerline position vector on target
  Core::LinAlg::Matrix<3, 1, T> r_xi_target(
      Core::LinAlg::Initialization::zero);  // centerline tangent vector on target
  Core::LinAlg::Matrix<3, 1, T> r_xixi_target(
      Core::LinAlg::Initialization::zero);  // 2nd deriv of target curve
  Core::LinAlg::Matrix<3, 1, T> t_target(
      Core::LinAlg::Initialization::zero);  // unit centerline tangent vector on target

  T norm_r_xi_source = 0.0;
  T norm_r_xi_target = 0.0;

  Core::LinAlg::Matrix<3, 1, T> dist_ul(Core::LinAlg::Initialization::zero);  // = r_source-r_target
  T norm_dist_ul = 0.0;

  T alpha = 0.0;      // mutual angle of tangent vectors
  T cos_alpha = 0.0;  // cosine of mutual angle of tangent vectors

  T interaction_potential_GP = 0.0;

  // components from variation of parameter coordinate on target beam
  Core::LinAlg::Matrix<1, 3, T> xi_target_partial_r_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 3, T> xi_target_partial_r_target(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 3, T> xi_target_partial_r_xi_target(Core::LinAlg::Initialization::zero);


  // linearization of parameter coordinate on target resulting from point-to-curve projection
  Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, T> lin_xi_source_targetDofs(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, T> lin_xi_target_targetDofs(
      Core::LinAlg::Initialization::zero);

  /* contribution of one Gauss point (required for automatic differentiation with contributions
   * from xi_target) */
  Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot_source_GP(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot_target_GP(
      Core::LinAlg::Initialization::zero);

  // evaluate charge/particle densities from DLINE charge condition specified in input file
  double rho1 = linechargeconds_[0]->parameters().template get<double>("VAL");
  double rho2 = linechargeconds_[1]->parameters().template get<double>("VAL");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  auto function_number =
      linechargeconds_[0]->parameters().template get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    rho1 *= Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
                .evaluate(time_);

  function_number = linechargeconds_[1]->parameters().template get<std::optional<int>>("FUNCT");

  if (function_number.has_value() && function_number.value() > 0)
    rho2 *= Global::Problem::instance()
                ->function_by_id<Core::Utils::FunctionOfTime>(function_number.value())
                .evaluate(time_);


  // constant prefactor of the disk-cylinder interaction potential
  double prefactor = k_ * std::numbers::pi * std::numbers::pi;

  switch ((int)m_)
  {
    case 6:
      prefactor /= 3.0;
      break;
    case 12:
      prefactor *= 286.0 / 15.0;
      break;
    default:
      FOUR_C_THROW(
          "Please implement the prefactor of the analytical disk-cylinder potential law for "
          "exponent m={}. So far, only exponent 6 and 12 is supported.",
          m_);
      break;
  }

  // store for visualization
  // Note: the visualization for the two half way approach does not need special treatment
  // since the interaction potential is halved and the resulting GPs are not coincident, i.e. two
  // distributed forces/moments are written per beam pair. This could be improved in the future by
  // merging the two resulting force/moment contributions per beam pair (this necessitates a new
  // algorithm for the visulaziation which would no longer be based on GPs)
  double prefactor_visualization_data = -1.0 * prefactor * rho1 * rho2;

  // prepare data storage for visualization
  centerline_coords_gp_1_.reserve(numgp_total);
  centerline_coords_gp_2_.reserve(numgp_total);
  forces_pot_gp_1_.reserve(numgp_total);
  forces_pot_gp_2_.reserve(numgp_total);
  moments_pot_gp_1_.reserve(numgp_total);
  moments_pot_gp_2_.reserve(numgp_total);

  for (unsigned int isegment = 0; isegment < n_integration_segments; ++isegment)
  {
    // compute element parameter coordinate for lower and upper limit of current integration segment
    double integration_segment_lower_limit =
        -1.0 + (double)isegment * 2.0 / (double)n_integration_segments;
    double integration_segment_upper_limit =
        -1.0 + (double)(isegment + 1) * 2.0 / (double)n_integration_segments;

    double jacobifactor_segment =
        0.5 * (integration_segment_upper_limit - integration_segment_lower_limit);

    // Evaluate shape functions at Gauss points of source element and store values
    Discret::Utils::Beam::evaluate_shape_functions_and_derivs_all_gps<numnodes, numnodalvalues>(
        gausspoints, N_i_source, N_i_xi_source, beam_element1()->shape(), ele1length_,
        integration_segment_lower_limit, integration_segment_upper_limit);

    // loop over gauss points of element 1
    // so far, element 1 is always treated as the source element!
    for (int igp = 0; igp < numgp_persegment; ++igp)
    {
      // add zero-initialized visualization data for this Gauss point
      // coordinates will be updated as they are computed, forces/moments at the end if contributing
      centerline_coords_gp_1_.emplace_back(Core::LinAlg::Initialization::zero);
      centerline_coords_gp_2_.emplace_back(Core::LinAlg::Initialization::zero);
      forces_pot_gp_1_.emplace_back(Core::LinAlg::Initialization::zero);
      forces_pot_gp_2_.emplace_back(Core::LinAlg::Initialization::zero);
      moments_pot_gp_1_.emplace_back(Core::LinAlg::Initialization::zero);
      moments_pot_gp_2_.emplace_back(Core::LinAlg::Initialization::zero);

      // Get location of GP in element parameter space xi \in [-1;1]
      xi_GP_tilde = gausspoints.qxg[igp][0];

      /* do a mapping into integration segment, i.e. coordinate transformation
       * note: this has no effect if integration interval is [-1;1] */
      xi_GP = 0.5 * ((1.0 - xi_GP_tilde) * integration_segment_lower_limit +
                        (1.0 + xi_GP_tilde) * integration_segment_upper_limit);


      // compute coord vector and tangent vector on source side
      compute_centerline_position(r_source, N_i_source[igp], ele1pos_);
      compute_centerline_tangent(r_xi_source, N_i_xi_source[igp], ele1pos_);

      norm_r_xi_source = Core::FADUtils::vector_norm(r_xi_source);

      t_source.update(1.0 / norm_r_xi_source, r_xi_source);

      // store for visualization
      centerline_coords_gp_1_.back() = Core::FADUtils::cast_to_double<T, 3, 1>(r_source);

      rho1rho2_JacFac_GaussWeight = rho1 * rho2 * jacobifactor_segment *
                                    beam_element1()->get_jacobi_fac_at_xi(xi_GP) *
                                    gausspoints.qwgt[igp];

      /* point-to-curve projection, i.e. 'unilateral' closest-point projection
       * to determine point on target beam (i.e. parameter coordinate xi_target) */
      iter_projection = 0;

      while (
          iter_projection < num_initial_values and
          (not BeamInteraction::Potential::Geo::point_to_curve_projection<numnodes, numnodalvalues,
              T>(r_source, xi_target, xi_target_initial_guess_values[iter_projection], ele2pos_,
              element2()->shape(), ele2length_)))
      {
        iter_projection++;
      }

      if (iter_projection == num_initial_values)
      {
        std::cout << "\nWARNING: Point-to-Curve Projection ultimately failed at "
                     "xi_source="
                  << xi_GP << " of ele pair " << element1()->id() << " & " << element2()->id()
                  << "\nFallback strategy: Assume invalid projection and skip this GP..."
                  << std::endl;
        continue;
      }

      // Todo: specify tolerance value in a more central place
      if (Core::FADUtils::norm(xi_target) > 1.0 + 1.0e-10)
      {
        continue;
      }
      // ensure no end node is utilized to prevent a possible doubled potential contribution
      else if (Core::FADUtils::norm(xi_target) >= 1.0 - 1.0e-10)
      {
        FOUR_C_THROW(
            "Point-to-curve projection yields xi_target= {}. This is a critical case "
            "since it is very close to the element boundary!",
            Core::FADUtils::cast_to_double(xi_target));
      }

      Discret::Utils::Beam::evaluate_shape_functions_and_derivs_and2nd_derivs_at_xi<numnodes,
          numnodalvalues>(xi_target, N_i_target, N_i_xi_target, N_i_xixi_target,
          beam_element2()->shape(), ele2length_);

      // compute coord vector and tangent vector on target side
      compute_centerline_position(r_target, N_i_target, ele2pos_);
      compute_centerline_tangent(r_xi_target, N_i_xi_target, ele2pos_);
      compute_centerline_tangent(r_xixi_target, N_i_xixi_target, ele2pos_);


      norm_r_xi_target = Core::FADUtils::vector_norm(r_xi_target);

      t_target.update(1.0 / norm_r_xi_target, r_xi_target);

      // store for visualization
      centerline_coords_gp_2_.back() = Core::FADUtils::cast_to_double<T, 3, 1>(r_target);

      // distance vector between unilateral closest points
      dist_ul.update(1.0, r_source, -1.0, r_target);

      norm_dist_ul = Core::FADUtils::vector_norm(dist_ul);

      if (Core::FADUtils::cast_to_double(norm_dist_ul) == 0.0)
      {
        this->print(std::cout);
        FOUR_C_THROW("centerline separation |r1-r2|=0!");
      }

      // check cutoff criterion: if specified, contributions are neglected at larger separation
      if (cutoff_radius.has_value() and
          Core::FADUtils::cast_to_double(norm_dist_ul) > cutoff_radius.value())
      {
        continue;
      }

      // mutual angle of tangent vectors at unilateral closest points
      BeamInteraction::Potential::Geo::calc_enclosed_angle(
          alpha, cos_alpha, r_xi_source, r_xi_target);

      if (alpha < 0.0 or alpha > std::numbers::pi / 2)
        FOUR_C_THROW("alpha={}, should be in [0,pi/2]", Core::FADUtils::cast_to_double(alpha));

      // Todo: maybe avoid this assembly of shape fcns into matrices
      // Fixme: at least, do this only in case of FAD-based linearization
      Discret::Utils::Beam::assemble_shape_functions<numnodes, numnodalvalues>(
          N_i_source[igp], N_source);

      Discret::Utils::Beam::assemble_shape_functions_and_derivs_and2nd_derivs<numnodes,
          numnodalvalues>(
          N_i_target, N_i_xi_target, N_i_xixi_target, N_target, N_xi_target, N_xixi_target);

      BeamInteraction::Potential::Geo::
          calc_linearization_point_to_curve_projection_parameter_coord_target<numnodes,
              numnodalvalues>(lin_xi_source_targetDofs, lin_xi_target_targetDofs, dist_ul,
              r_xi_target, r_xixi_target, N_source, N_target, N_xi_target);


      BeamInteraction::Potential::Geo::
          calc_point_to_curve_projection_parameter_coord_target_partial_derivs(
              xi_target_partial_r_source, xi_target_partial_r_target, xi_target_partial_r_xi_target,
              dist_ul, r_xi_target, r_xixi_target);


      // evaluate all quantities which depend on the applied disk-cylinder potential law

      // 'full' disk-cylinder interaction potential
      if (params()->strategy ==
          BeamInteraction::Potential::Strategy::single_length_specific_small_separations)
      {
        if (not evaluate_full_disk_cylinder_potential(interaction_potential_GP, force_pot_source_GP,
                force_pot_target_GP, r_source, r_xi_source, t_source, r_target, r_xi_target,
                r_xixi_target, t_target, alpha, cos_alpha, dist_ul, xi_target_partial_r_source,
                xi_target_partial_r_target, xi_target_partial_r_xi_target,
                prefactor_visualization_data, forces_pot_gp_1_.back(), forces_pot_gp_2_.back(),
                moments_pot_gp_1_.back(), moments_pot_gp_2_.back(), rho1rho2_JacFac_GaussWeight,
                N_i_source[igp], N_i_xi_source[igp], N_i_target, N_i_xi_target))
          continue;
      }
      // reduced, simpler variant of the disk-cylinder interaction potential
      else if (params()->strategy == BeamInteraction::Potential::Strategy::
                                         single_length_specific_small_separations_simple)
      {
        if (not evaluate_simple_disk_cylinder_potential(dist_ul, norm_dist_ul, alpha, cos_alpha,
                r_source, r_xi_source, norm_r_xi_source, t_source, r_target, r_xi_target,
                norm_r_xi_target, r_xixi_target, t_target, xi_target, xi_target_partial_r_source,
                xi_target_partial_r_target, xi_target_partial_r_xi_target,
                potential_reduction_length, length_prior_right, length_prior_left,
                interaction_potential_GP, N_i_source[igp], N_i_xi_source[igp], N_i_target,
                N_i_xi_target, N_i_xixi_target, force_pot_source_GP, force_pot_target_GP,
                prefactor_visualization_data, rho1rho2_JacFac_GaussWeight, forces_pot_gp_1_.back(),
                forces_pot_gp_2_.back(), moments_pot_gp_1_.back(), moments_pot_gp_2_.back(),
                stiffmat11, stiffmat12, stiffmat21, stiffmat22))
          continue;
      }

      // apply constant prefactor
      force_pot_source_GP.scale(prefactor);
      force_pot_target_GP.scale(prefactor);

      // sum contributions from all Gauss points
      force_pot1.update(1.0, force_pot_source_GP, 1.0);
      force_pot2.update(1.0, force_pot_target_GP, 1.0);

      if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
          stiffmat22 != nullptr)
      {
        add_stiffmat_contributions_xi_target_automatic_differentiation_if_required(
            force_pot_source_GP, force_pot_target_GP, lin_xi_source_targetDofs,
            lin_xi_target_targetDofs, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
      }

      // do this scaling down here, because the value without the prefactors is meant to be used
      // in the force calculation above!
      interaction_potential_GP *= prefactor;

      // store for energy output
      interaction_potential_ += Core::FADUtils::cast_to_double(interaction_potential_GP);

    }  // end loop over Gauss points per segment
  }  // end loop over integration segments

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    scale_stiffpot_analytic_contributions_if_required(
        prefactor, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);


    calc_stiffmat_automatic_differentiation_if_required(
        force_pot1, force_pot2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    evaluate_stiffpot_analytic_contributions_single_length_specific_small_sep_approx_simple(
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_source,
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_source,
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_target,
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_target,
        Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xixi_target,
        double const& xi_target, Core::LinAlg::Matrix<3, 1, double> const& r_xi_source,
        Core::LinAlg::Matrix<3, 1, double> const& r_xi_target,
        Core::LinAlg::Matrix<3, 1, double> const& r_xixi_target, double const& norm_dist_ul,
        Core::LinAlg::Matrix<3, 1, double> const& normal_ul, double const& pot_red_fac,
        double const& pot_red_fac_deriv_xi_target, double const& pot_red_fac_2ndderiv_xi_target,
        double const& pot_ia, double const& pot_ia_deriv_gap_ul,
        double const& pot_ia_deriv_cos_alpha, double const& pot_ia_2ndderiv_gap_ul,
        double const& pot_ia_deriv_gap_ul_deriv_cos_alpha, double const& pot_ia_2ndderiv_cos_alpha,
        Core::LinAlg::Matrix<3, 1, double> const& gap_ul_deriv_r_source,
        Core::LinAlg::Matrix<3, 1, double> const& gap_ul_deriv_r_target,
        Core::LinAlg::Matrix<3, 1, double> const& cos_alpha_deriv_r_source,
        Core::LinAlg::Matrix<3, 1, double> const& cos_alpha_deriv_r_target,
        Core::LinAlg::Matrix<3, 1, double> const& cos_alpha_deriv_r_xi_source,
        Core::LinAlg::Matrix<3, 1, double> const& cos_alpha_deriv_r_xi_target,
        Core::LinAlg::Matrix<1, 3, double> const& xi_target_partial_r_source,
        Core::LinAlg::Matrix<1, 3, double> const& xi_target_partial_r_target,
        Core::LinAlg::Matrix<1, 3, double> const& xi_target_partial_r_xi_target,
        Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
        Core::LinAlg::SerialDenseMatrix& stiffmat21,
        Core::LinAlg::SerialDenseMatrix& stiffmat22) const
{
  const unsigned int num_spatial_dimensions = 3;
  const unsigned int num_dofs_per_spatial_dimension = numnodes * numnodalvalues;


  // auxiliary and intermediate quantities required for second derivatives of the unilateral gap
  Core::LinAlg::Matrix<3, 3, double> unit_matrix(Core::LinAlg::Initialization::zero);
  for (unsigned int i = 0; i < 3; ++i) unit_matrix(i, i) = 1.0;

  Core::LinAlg::Matrix<3, 3, double> dist_ul_deriv_r_source(unit_matrix);
  Core::LinAlg::Matrix<3, 3, double> dist_ul_deriv_r_target(unit_matrix);
  dist_ul_deriv_r_target.scale(-1.0);
  Core::LinAlg::Matrix<3, 3, double> dist_ul_deriv_r_xi_target(Core::LinAlg::Initialization::zero);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      dist_ul_deriv_r_source(irow, icol) -= r_xi_target(irow) * xi_target_partial_r_source(icol);

      dist_ul_deriv_r_target(irow, icol) -= r_xi_target(irow) * xi_target_partial_r_target(icol);

      dist_ul_deriv_r_xi_target(irow, icol) -=
          r_xi_target(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  Core::LinAlg::Matrix<3, 3, double> normal_ul_deriv_dist_ul(Core::LinAlg::Initialization::zero);

  for (unsigned int i = 0; i < 3; ++i)
  {
    normal_ul_deriv_dist_ul(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j)
      normal_ul_deriv_dist_ul(i, j) -= normal_ul(i) * normal_ul(j);
  }
  normal_ul_deriv_dist_ul.scale(1.0 / norm_dist_ul);


  // second derivatives of the unilateral gap
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_source_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_source_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_source_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_target_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_target_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_target_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);


  gap_ul_deriv_r_source_deriv_r_source.multiply(
      1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_source);
  gap_ul_deriv_r_source_deriv_r_target.multiply(
      1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_target);
  gap_ul_deriv_r_source_deriv_r_xi_target.multiply(
      1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_xi_target);

  gap_ul_deriv_r_target_deriv_r_source.multiply(
      -1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_source);
  gap_ul_deriv_r_target_deriv_r_target.multiply(
      -1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_target);
  gap_ul_deriv_r_target_deriv_r_xi_target.multiply(
      -1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_xi_target);

  // add contributions from linearization of (variation of r_target) according to the chain
  // rule
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_xi_target_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_xi_target_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> gap_ul_deriv_r_xi_target_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      gap_ul_deriv_r_xi_target_deriv_r_source(irow, icol) -=
          normal_ul(irow) * xi_target_partial_r_source(icol);

      gap_ul_deriv_r_xi_target_deriv_r_target(irow, icol) -=
          normal_ul(irow) * xi_target_partial_r_target(icol);

      gap_ul_deriv_r_xi_target_deriv_r_xi_target(irow, icol) -=
          normal_ul(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  // second derivatives of cos(alpha)
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_source_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_source_deriv_r_xi_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_source_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_source_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_source_deriv_r_xixi_target(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_source_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_source_deriv_r_xi_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_source_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_source_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_target_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_target_deriv_r_xi_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_target_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_target_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_target_deriv_r_xixi_target(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_target_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_target_deriv_r_xi_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_target_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_target_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xi_target_deriv_r_xixi_target(
      Core::LinAlg::Initialization::zero);


  // auxiliary and intermediate quantities required for second derivatives of cos(alpha)

  double norm_r_xi_source_inverse = 1.0 / Core::FADUtils::vector_norm(r_xi_source);
  double norm_r_xi_target_inverse = 1.0 / Core::FADUtils::vector_norm(r_xi_target);

  Core::LinAlg::Matrix<3, 1, double> t_source(Core::LinAlg::Initialization::zero);
  t_source.update(norm_r_xi_source_inverse, r_xi_source);
  Core::LinAlg::Matrix<3, 1, double> t_target(Core::LinAlg::Initialization::zero);
  t_target.update(norm_r_xi_target_inverse, r_xi_target);

  double t_source_dot_t_target = t_source.dot(t_target);
  double signum_tangentsscalarproduct = Core::FADUtils::signum(t_source_dot_t_target);

  Core::LinAlg::Matrix<3, 3, double> t_source_tensorproduct_t_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> t_source_tensorproduct_t_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> t_target_tensorproduct_t_target(
      Core::LinAlg::Initialization::zero);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      t_source_tensorproduct_t_source(i, j) = t_source(i) * t_source(j);
      t_source_tensorproduct_t_target(i, j) = t_source(i) * t_target(j);
      t_target_tensorproduct_t_target(i, j) = t_target(i) * t_target(j);
    }

  Core::LinAlg::Matrix<3, 3, double> t_source_partial_r_xi_source(
      Core::LinAlg::Initialization::zero);
  t_source_partial_r_xi_source.update(norm_r_xi_source_inverse, unit_matrix,
      -1.0 * norm_r_xi_source_inverse, t_source_tensorproduct_t_source);

  Core::LinAlg::Matrix<3, 3, double> t_target_partial_r_xi_target(
      Core::LinAlg::Initialization::zero);
  t_target_partial_r_xi_target.update(norm_r_xi_target_inverse, unit_matrix,
      -1.0 * norm_r_xi_target_inverse, t_target_tensorproduct_t_target);


  Core::LinAlg::Matrix<3, 1, double> t_source_partial_r_xi_source_mult_t_target(
      Core::LinAlg::Initialization::zero);
  t_source_partial_r_xi_source_mult_t_target.multiply(t_source_partial_r_xi_source, t_target);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      cos_alpha_deriv_r_xi_source_deriv_r_xi_source(i, j) -=
          norm_r_xi_source_inverse * t_source_partial_r_xi_source_mult_t_target(i) * t_source(j);

  cos_alpha_deriv_r_xi_source_deriv_r_xi_source.update(
      -1.0 * norm_r_xi_source_inverse * t_source_dot_t_target, t_source_partial_r_xi_source, 1.0);

  Core::LinAlg::Matrix<3, 3, double> tmp_mat(Core::LinAlg::Initialization::zero);
  tmp_mat.multiply(t_source_tensorproduct_t_target, t_source_partial_r_xi_source);
  cos_alpha_deriv_r_xi_source_deriv_r_xi_source.update(
      -1.0 * norm_r_xi_source_inverse, tmp_mat, 1.0);


  cos_alpha_deriv_r_xi_source_deriv_r_xi_target.update(
      norm_r_xi_source_inverse, t_target_partial_r_xi_target, 1.0);

  tmp_mat.multiply(t_source_tensorproduct_t_source, t_target_partial_r_xi_target);
  cos_alpha_deriv_r_xi_source_deriv_r_xi_target.update(
      -1.0 * norm_r_xi_source_inverse, tmp_mat, 1.0);



  Core::LinAlg::Matrix<3, 1, double> t_target_partial_r_xi_target_mult_t_source(
      Core::LinAlg::Initialization::zero);
  t_target_partial_r_xi_target_mult_t_source.multiply(t_target_partial_r_xi_target, t_source);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      cos_alpha_deriv_r_xi_target_deriv_r_xi_target(i, j) -=
          norm_r_xi_target_inverse * t_target_partial_r_xi_target_mult_t_source(i) * t_target(j);

  cos_alpha_deriv_r_xi_target_deriv_r_xi_target.update(
      -1.0 * norm_r_xi_target_inverse * t_source_dot_t_target, t_target_partial_r_xi_target, 1.0);

  tmp_mat.clear();
  tmp_mat.multiply_tn(t_source_tensorproduct_t_target, t_target_partial_r_xi_target);
  cos_alpha_deriv_r_xi_target_deriv_r_xi_target.update(
      -1.0 * norm_r_xi_target_inverse, tmp_mat, 1.0);


  cos_alpha_deriv_r_xi_target_deriv_r_xi_source.update(
      norm_r_xi_target_inverse, t_source_partial_r_xi_source, 1.0);

  tmp_mat.multiply(t_target_tensorproduct_t_target, t_source_partial_r_xi_source);
  cos_alpha_deriv_r_xi_target_deriv_r_xi_source.update(
      -1.0 * norm_r_xi_target_inverse, tmp_mat, 1.0);


  // add contributions from variation of target parameter coordinate xi_target
  // to [.]_deriv_r_xi_target_[.] expressions (according to chain rule)
  Core::LinAlg::Matrix<1, 3, double> tmp_vec;
  tmp_vec.multiply_tn(r_xixi_target, cos_alpha_deriv_r_xi_target_deriv_r_xi_source);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_source_deriv_r_xi_source(irow, icol) +=
          xi_target_partial_r_source(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_target_deriv_r_xi_source(irow, icol) +=
          xi_target_partial_r_target(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_xi_source(irow, icol) +=
          xi_target_partial_r_xi_target(irow) * tmp_vec(icol);
    }
  }

  tmp_vec.multiply_tn(r_xixi_target, cos_alpha_deriv_r_xi_target_deriv_r_xi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_source_deriv_r_xi_target(irow, icol) +=
          xi_target_partial_r_source(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_target_deriv_r_xi_target(irow, icol) +=
          xi_target_partial_r_target(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_xi_target(irow, icol) +=
          xi_target_partial_r_xi_target(irow) * tmp_vec(icol);
    }
  }


  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_source_deriv_r_xixi_target(irow, icol) +=
          xi_target_partial_r_source(irow) * t_target_partial_r_xi_target_mult_t_source(icol);

      cos_alpha_deriv_r_target_deriv_r_xixi_target(irow, icol) +=
          xi_target_partial_r_target(irow) * t_target_partial_r_xi_target_mult_t_source(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_xixi_target(irow, icol) +=
          xi_target_partial_r_xi_target(irow) * t_target_partial_r_xi_target_mult_t_source(icol);
    }
  }


  // add contributions from linearization of target parameter coordinate xi_target
  // to [.]_deriv_r_xi_target expressions (according to chain rule)
  Core::LinAlg::Matrix<3, 1, double> tmp_vec2;
  tmp_vec2.multiply(cos_alpha_deriv_r_source_deriv_r_xi_target, r_xixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_source_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_source_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_source_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  tmp_vec2.multiply(cos_alpha_deriv_r_xi_source_deriv_r_xi_target, r_xixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xi_source_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_xi_source_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_xi_source_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  tmp_vec2.multiply(cos_alpha_deriv_r_target_deriv_r_xi_target, r_xixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_target_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_target_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_target_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  tmp_vec2.multiply(cos_alpha_deriv_r_xi_target_deriv_r_xi_target, r_xixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xi_target_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  // add contributions from linearization of target parameter coordinate xi_target
  // also to [.]_deriv_r_xixi_target expressions (according to chain rule)
  Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> N_i_xixixi_target(
      Core::LinAlg::Initialization::zero);

  Discret::Utils::Beam::evaluate_shape_function3rd_derivs_at_xi<numnodes, numnodalvalues>(
      xi_target, N_i_xixixi_target, beam_element2()->shape(), ele2length_);

  Core::LinAlg::Matrix<3, 1, double> r_xixixi_target(Core::LinAlg::Initialization::zero);

  Discret::Utils::Beam::calc_interpolation<numnodes, numnodalvalues, 3>(
      Core::FADUtils::cast_to_double(ele2pos_), N_i_xixixi_target, r_xixixi_target);

  tmp_vec2.multiply(cos_alpha_deriv_r_source_deriv_r_xixi_target, r_xixixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_source_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_source_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_source_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }

  tmp_vec2.multiply(cos_alpha_deriv_r_target_deriv_r_xixi_target, r_xixixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_target_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_target_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_target_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }

  tmp_vec2.multiply(cos_alpha_deriv_r_xi_target_deriv_r_xixi_target, r_xixixi_target);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xi_target_deriv_r_source(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_xi_target_deriv_r_xi_target(irow, icol) +=
          tmp_vec2(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  // add contributions from linearization of (variation of r_xi_target) according to the chain
  // rule
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xixi_target_deriv_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xixi_target_deriv_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> cos_alpha_deriv_r_xixi_target_deriv_r_xi_target(
      Core::LinAlg::Initialization::zero);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xixi_target_deriv_r_source(irow, icol) +=
          t_target_partial_r_xi_target_mult_t_source(irow) * xi_target_partial_r_source(icol);

      cos_alpha_deriv_r_xixi_target_deriv_r_target(irow, icol) +=
          t_target_partial_r_xi_target_mult_t_source(irow) * xi_target_partial_r_target(icol);

      cos_alpha_deriv_r_xixi_target_deriv_r_xi_target(irow, icol) +=
          t_target_partial_r_xi_target_mult_t_source(irow) * xi_target_partial_r_xi_target(icol);
    }
  }


  // contributions from linarization of the (variation of target parameter coordinate xi_target)
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_source_partial_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_source_partial_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_source_partial_r_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_source_partial_r_xixi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_target_partial_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_target_partial_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_target_partial_r_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_target_partial_r_xixi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xi_target_partial_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xi_target_partial_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xi_target_partial_r_xi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xi_target_partial_r_xixi_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xixi_target_partial_r_source(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xixi_target_partial_r_target(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, double> xi_target_partial_r_xixi_target_partial_r_xi_target(
      Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 1, double> dist_ul(Core::LinAlg::Initialization::zero);
  dist_ul.update(norm_dist_ul, normal_ul);

  BeamInteraction::Potential::Geo::
      calc_point_to_curve_projection_parameter_coord_target_partial2nd_derivs(
          xi_target_partial_r_source_partial_r_source, xi_target_partial_r_source_partial_r_target,
          xi_target_partial_r_source_partial_r_xi_target,
          xi_target_partial_r_source_partial_r_xixi_target,
          xi_target_partial_r_target_partial_r_source, xi_target_partial_r_target_partial_r_target,
          xi_target_partial_r_target_partial_r_xi_target,
          xi_target_partial_r_target_partial_r_xixi_target,
          xi_target_partial_r_xi_target_partial_r_source,
          xi_target_partial_r_xi_target_partial_r_target,
          xi_target_partial_r_xi_target_partial_r_xi_target,
          xi_target_partial_r_xi_target_partial_r_xixi_target,
          xi_target_partial_r_xixi_target_partial_r_source,
          xi_target_partial_r_xixi_target_partial_r_target,
          xi_target_partial_r_xixi_target_partial_r_xi_target, xi_target_partial_r_source,
          xi_target_partial_r_target, xi_target_partial_r_xi_target, dist_ul_deriv_r_source,
          dist_ul_deriv_r_target, dist_ul_deriv_r_xi_target, dist_ul, r_xi_target, r_xixi_target,
          r_xixixi_target);

  double r_xixi_target_dot_v2 = r_xixi_target.dot(t_target_partial_r_xi_target_mult_t_source);

  cos_alpha_deriv_r_source_deriv_r_source.update(
      r_xixi_target_dot_v2, xi_target_partial_r_source_partial_r_source, 1.0);

  cos_alpha_deriv_r_source_deriv_r_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_source_partial_r_target, 1.0);

  cos_alpha_deriv_r_source_deriv_r_xi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_source_partial_r_xi_target, 1.0);

  cos_alpha_deriv_r_source_deriv_r_xixi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_source_partial_r_xixi_target, 1.0);


  cos_alpha_deriv_r_target_deriv_r_source.update(
      r_xixi_target_dot_v2, xi_target_partial_r_target_partial_r_source, 1.0);

  cos_alpha_deriv_r_target_deriv_r_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_target_partial_r_target, 1.0);

  cos_alpha_deriv_r_target_deriv_r_xi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_target_partial_r_xi_target, 1.0);

  cos_alpha_deriv_r_target_deriv_r_xixi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_target_partial_r_xixi_target, 1.0);


  cos_alpha_deriv_r_xi_target_deriv_r_source.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xi_target_partial_r_source, 1.0);

  cos_alpha_deriv_r_xi_target_deriv_r_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xi_target_partial_r_target, 1.0);

  cos_alpha_deriv_r_xi_target_deriv_r_xi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xi_target_partial_r_xi_target, 1.0);

  cos_alpha_deriv_r_xi_target_deriv_r_xixi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xi_target_partial_r_xixi_target, 1.0);


  cos_alpha_deriv_r_xixi_target_deriv_r_source.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xixi_target_partial_r_source, 1.0);

  cos_alpha_deriv_r_xixi_target_deriv_r_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xixi_target_partial_r_target, 1.0);

  cos_alpha_deriv_r_xixi_target_deriv_r_xi_target.update(
      r_xixi_target_dot_v2, xi_target_partial_r_xixi_target_partial_r_xi_target, 1.0);


  cos_alpha_deriv_r_source_deriv_r_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_source_deriv_r_xi_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_source_deriv_r_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_source_deriv_r_xi_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_source_deriv_r_xixi_target.scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_xi_source_deriv_r_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_source_deriv_r_xi_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_source_deriv_r_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_source_deriv_r_xi_target.scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_target_deriv_r_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_target_deriv_r_xi_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_target_deriv_r_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_target_deriv_r_xi_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_target_deriv_r_xixi_target.scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_xi_target_deriv_r_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_target_deriv_r_xi_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_target_deriv_r_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_target_deriv_r_xi_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_target_deriv_r_xixi_target.scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_xixi_target_deriv_r_source.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xixi_target_deriv_r_target.scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xixi_target_deriv_r_xi_target.scale(signum_tangentsscalarproduct);

  // Distributed endpoint moment compensation strategy
  double endpoint_moment_compensation = 1.0;
  if (params()->potential_reduction_endpoint_moment_compensation)
  {
    // if endpoint moment compensation is enabled, the distributed end point moment arising from the
    // variation of the closest point projection is compensated. Here we need to remove certain
    // contributions from the stiffness matrix that correspond to the distributed end point moment,
    // which is equivalent to setting the endpoint_moment_compensation factor to zero.
    endpoint_moment_compensation = 0.0;
  }


  // assemble all pre-computed terms into the stiffness matrices
  for (unsigned int irowdofperdim = 0; irowdofperdim < num_dofs_per_spatial_dimension;
      ++irowdofperdim)
  {
    for (unsigned int icolumndofperdim = 0; icolumndofperdim < num_dofs_per_spatial_dimension;
        ++icolumndofperdim)
    {
      for (unsigned int irowdim = 0; irowdim < num_spatial_dimensions; ++irowdim)
      {
        for (unsigned int icolumndim = 0; icolumndim < num_spatial_dimensions; ++icolumndim)
        {
          // variation of xi_target * linearization of (pot_red_fac_deriv_xi_target * pot_ia)
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_source(irowdofperdim) * xi_target_partial_r_source(irowdim) *
              (pot_red_fac_2ndderiv_xi_target * xi_target_partial_r_source(icolumndim) *
                      N_i_source(icolumndofperdim) * pot_ia +
                  pot_red_fac_deriv_xi_target *
                      (pot_ia_deriv_gap_ul * gap_ul_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_xi_source(icolumndim) *
                              N_i_xi_source(icolumndofperdim)));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_source(irowdofperdim) * xi_target_partial_r_source(irowdim) *
              (pot_red_fac_2ndderiv_xi_target *
                      (xi_target_partial_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)) *
                      pot_ia +
                  pot_red_fac_deriv_xi_target *
                      (pot_ia_deriv_gap_ul * gap_ul_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_target(irowdofperdim) * xi_target_partial_r_target(irowdim) +
                  N_i_xi_target(irowdofperdim) * endpoint_moment_compensation *
                      xi_target_partial_r_xi_target(irowdim)) *
              (pot_red_fac_2ndderiv_xi_target * xi_target_partial_r_source(icolumndim) *
                      N_i_source(icolumndofperdim) * pot_ia +
                  pot_red_fac_deriv_xi_target *
                      (pot_ia_deriv_gap_ul * gap_ul_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_xi_source(icolumndim) *
                              N_i_xi_source(icolumndofperdim)));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_target(irowdofperdim) * xi_target_partial_r_target(irowdim) +
                  N_i_xi_target(irowdofperdim) * endpoint_moment_compensation *
                      xi_target_partial_r_xi_target(irowdim)) *
              (pot_red_fac_2ndderiv_xi_target *
                      (xi_target_partial_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)) *
                      pot_ia +
                  pot_red_fac_deriv_xi_target *
                      (pot_ia_deriv_gap_ul * gap_ul_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)));

          // linearization of the (variation of xi_target) * (pot_red_fac_deriv_xi_target * pot_ia)
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac_deriv_xi_target * pot_ia * N_i_source(irowdofperdim) *
              xi_target_partial_r_source_partial_r_source(irowdim, icolumndim) *
              N_i_source(icolumndofperdim);

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac_deriv_xi_target * pot_ia * N_i_source(irowdofperdim) *
              (xi_target_partial_r_source_partial_r_target(irowdim, icolumndim) *
                      N_i_target(icolumndofperdim) +
                  xi_target_partial_r_source_partial_r_xi_target(irowdim, icolumndim) *
                      N_i_xi_target(icolumndofperdim) +
                  xi_target_partial_r_source_partial_r_xixi_target(irowdim, icolumndim) *
                      N_i_xixi_target(icolumndofperdim));

          // The direct variation of r_xi_target is always active; the remaining terms are scaled
          // by the distributed endpoint moment compensation in the potential reduction strategy
          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac_deriv_xi_target * pot_ia *
              (N_i_target(irowdofperdim) *
                      xi_target_partial_r_target_partial_r_source(irowdim, icolumndim) *
                      N_i_source(icolumndofperdim) +
                  N_i_xi_target(irowdofperdim) *
                      (endpoint_moment_compensation *
                              xi_target_partial_r_xi_target_partial_r_source(irowdim, icolumndim) +
                          (1.0 - endpoint_moment_compensation) *
                              xi_target_partial_r_target(irowdim) *
                              xi_target_partial_r_source(icolumndim)) *
                      N_i_source(icolumndofperdim) +
                  N_i_xixi_target(irowdofperdim) * endpoint_moment_compensation *
                      (xi_target_partial_r_xixi_target_partial_r_source(irowdim, icolumndim) *
                          N_i_source(icolumndofperdim)));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac_deriv_xi_target * pot_ia *
              (N_i_target(irowdofperdim) *
                      (xi_target_partial_r_target_partial_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          xi_target_partial_r_target_partial_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim) +
                          xi_target_partial_r_target_partial_r_xixi_target(irowdim, icolumndim) *
                              N_i_xixi_target(icolumndofperdim)) +
                  N_i_xi_target(irowdofperdim) *
                      ((endpoint_moment_compensation *
                               xi_target_partial_r_xi_target_partial_r_target(irowdim, icolumndim) +
                           (1.0 - endpoint_moment_compensation) *
                               xi_target_partial_r_target(irowdim) *
                               xi_target_partial_r_target(icolumndim)) *
                              N_i_target(icolumndofperdim) +
                          (endpoint_moment_compensation *
                                  xi_target_partial_r_xi_target_partial_r_xi_target(
                                      irowdim, icolumndim) +
                              (1.0 - endpoint_moment_compensation) *
                                  xi_target_partial_r_target(irowdim) *
                                  xi_target_partial_r_xi_target(icolumndim)) *
                              N_i_xi_target(icolumndofperdim) +
                          endpoint_moment_compensation *
                              xi_target_partial_r_xi_target_partial_r_xixi_target(
                                  irowdim, icolumndim) *
                              N_i_xixi_target(icolumndofperdim)) +
                  N_i_xixi_target(irowdofperdim) * endpoint_moment_compensation *
                      (xi_target_partial_r_xixi_target_partial_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xixi_target_partial_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim)));

          // variation of gap_ul * linearization of (pot_red_fac * pot_ia_deriv_gap_ul)
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_source(irowdofperdim) * gap_ul_deriv_r_source(irowdim) *
              (pot_red_fac_deriv_xi_target * xi_target_partial_r_source(icolumndim) *
                      N_i_source(icolumndofperdim) * pot_ia_deriv_gap_ul +
                  pot_red_fac *
                      (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_source(icolumndim) * N_i_source(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_xi_source(icolumndim) *
                              N_i_xi_source(icolumndofperdim)));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_source(irowdofperdim) * gap_ul_deriv_r_source(irowdim) *
              (pot_red_fac_deriv_xi_target *
                      (xi_target_partial_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)) *
                      pot_ia_deriv_gap_ul +
                  pot_red_fac *
                      (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_target(irowdofperdim) * gap_ul_deriv_r_target(irowdim) *
              (pot_red_fac_deriv_xi_target * xi_target_partial_r_source(icolumndim) *
                      N_i_source(icolumndofperdim) * pot_ia_deriv_gap_ul +
                  pot_red_fac *
                      (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_source(icolumndim) * N_i_source(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_xi_source(icolumndim) *
                              N_i_xi_source(icolumndofperdim)));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_target(irowdofperdim) * gap_ul_deriv_r_target(irowdim) *
              (pot_red_fac_deriv_xi_target *
                      (xi_target_partial_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)) *
                      pot_ia_deriv_gap_ul +
                  pot_red_fac *
                      (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha *
                              cos_alpha_deriv_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)));

          // linearization of the (variation of gap_ul) * (pot_red_fac * pot_ia_deriv_gap_ul)
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_gap_ul * N_i_source(irowdofperdim) *
              gap_ul_deriv_r_source_deriv_r_source(irowdim, icolumndim) *
              N_i_source(icolumndofperdim);

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_gap_ul * N_i_source(irowdofperdim) *
              (gap_ul_deriv_r_source_deriv_r_target(irowdim, icolumndim) *
                      N_i_target(icolumndofperdim) +
                  gap_ul_deriv_r_source_deriv_r_xi_target(irowdim, icolumndim) *
                      N_i_xi_target(icolumndofperdim));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_gap_ul *
              (N_i_target(irowdofperdim) *
                      gap_ul_deriv_r_target_deriv_r_source(irowdim, icolumndim) +
                  N_i_xi_target(irowdofperdim) *
                      gap_ul_deriv_r_xi_target_deriv_r_source(irowdim, icolumndim)) *
              N_i_source(icolumndofperdim);

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_gap_ul *
              (N_i_target(irowdofperdim) *
                      (gap_ul_deriv_r_target_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          gap_ul_deriv_r_target_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim)) +
                  N_i_xi_target(irowdofperdim) *
                      (gap_ul_deriv_r_xi_target_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          gap_ul_deriv_r_xi_target_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim)));


          // variation of cos_alpha * linearization of (pot_red_fac * pot_ia_deriv_cos_alpha)
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_source(irowdofperdim) * cos_alpha_deriv_r_source(irowdim) +
                  N_i_xi_source(irowdofperdim) * cos_alpha_deriv_r_xi_source(irowdim)) *
              (pot_red_fac_deriv_xi_target * xi_target_partial_r_source(icolumndim) *
                      N_i_source(icolumndofperdim) * pot_ia_deriv_cos_alpha +
                  pot_red_fac *
                      (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_source(icolumndim) *
                              N_i_xi_source(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim)));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_source(irowdofperdim) * cos_alpha_deriv_r_source(irowdim) +
                  N_i_xi_source(irowdofperdim) * cos_alpha_deriv_r_xi_source(irowdim)) *
              (pot_red_fac_deriv_xi_target *
                      (xi_target_partial_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)) *
                      pot_ia_deriv_cos_alpha +
                  pot_red_fac *
                      (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim)));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_target(irowdofperdim) * cos_alpha_deriv_r_target(irowdim) +
                  N_i_xi_target(irowdofperdim) * cos_alpha_deriv_r_xi_target(irowdim)) *
              (pot_red_fac_deriv_xi_target * xi_target_partial_r_source(icolumndim) *
                      N_i_source(icolumndofperdim) * pot_ia_deriv_cos_alpha +
                  pot_red_fac *
                      (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim) +
                          pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_source(icolumndim) *
                              N_i_xi_source(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_source(icolumndim) *
                              N_i_source(icolumndofperdim)));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_target(irowdofperdim) * cos_alpha_deriv_r_target(irowdim) +
                  N_i_xi_target(irowdofperdim) * cos_alpha_deriv_r_xi_target(irowdim)) *
              (pot_red_fac_deriv_xi_target *
                      (xi_target_partial_r_target(icolumndim) * N_i_target(icolumndofperdim) +
                          xi_target_partial_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim)) *
                      pot_ia_deriv_cos_alpha +
                  pot_red_fac *
                      (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim) +
                          pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_target(icolumndim) *
                              N_i_xi_target(icolumndofperdim) +
                          pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_target(icolumndim) *
                              N_i_target(icolumndofperdim)));


          // linearization of the (variation of cos_alpha) * (pot_red_fac * pot_ia_deriv_cos_alpha)
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_cos_alpha *
              (N_i_source(irowdofperdim) *
                      (cos_alpha_deriv_r_source_deriv_r_source(irowdim, icolumndim) *
                              N_i_source(icolumndofperdim) +
                          cos_alpha_deriv_r_source_deriv_r_xi_source(irowdim, icolumndim) *
                              N_i_xi_source(icolumndofperdim)) +
                  N_i_xi_source(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_source_deriv_r_source(irowdim, icolumndim) *
                              N_i_source(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_source_deriv_r_xi_source(irowdim, icolumndim) *
                              N_i_xi_source(icolumndofperdim)));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_cos_alpha *
              (N_i_source(irowdofperdim) *
                      (cos_alpha_deriv_r_source_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          cos_alpha_deriv_r_source_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim) +
                          cos_alpha_deriv_r_source_deriv_r_xixi_target(irowdim, icolumndim) *
                              N_i_xixi_target(icolumndofperdim)) +
                  N_i_xi_source(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_source_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_source_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim)));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_cos_alpha *
              (N_i_target(irowdofperdim) *
                      (cos_alpha_deriv_r_target_deriv_r_source(irowdim, icolumndim) *
                              N_i_source(icolumndofperdim) +
                          cos_alpha_deriv_r_target_deriv_r_xi_source(irowdim, icolumndim) *
                              N_i_xi_source(icolumndofperdim)) +
                  N_i_xi_target(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_target_deriv_r_source(irowdim, icolumndim) *
                              N_i_source(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_target_deriv_r_xi_source(irowdim, icolumndim) *
                              N_i_xi_source(icolumndofperdim)) +
                  N_i_xixi_target(irowdofperdim) *
                      (cos_alpha_deriv_r_xixi_target_deriv_r_source(irowdim, icolumndim) *
                          N_i_source(icolumndofperdim)));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_red_fac * pot_ia_deriv_cos_alpha *
              (N_i_target(irowdofperdim) *
                      (cos_alpha_deriv_r_target_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          cos_alpha_deriv_r_target_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim) +
                          cos_alpha_deriv_r_target_deriv_r_xixi_target(irowdim, icolumndim) *
                              N_i_xixi_target(icolumndofperdim)) +
                  N_i_xi_target(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_target_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_target_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_target_deriv_r_xixi_target(irowdim, icolumndim) *
                              N_i_xixi_target(icolumndofperdim)) +
                  N_i_xixi_target(irowdofperdim) *
                      (cos_alpha_deriv_r_xixi_target_deriv_r_target(irowdim, icolumndim) *
                              N_i_target(icolumndofperdim) +
                          cos_alpha_deriv_r_xixi_target_deriv_r_xi_target(irowdim, icolumndim) *
                              N_i_xi_target(icolumndofperdim)));
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::evaluate_full_disk_cylinder_potential(T& interaction_potential_GP,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot_source_GP,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot_target_GP,
    Core::LinAlg::Matrix<3, 1, T> const& r_source, Core::LinAlg::Matrix<3, 1, T> const& r_xi_source,
    Core::LinAlg::Matrix<3, 1, T> const& t1_source, Core::LinAlg::Matrix<3, 1, T> const& r_target,
    Core::LinAlg::Matrix<3, 1, T> const& r_xi_target,
    Core::LinAlg::Matrix<3, 1, T> const& r_xixi_target,
    Core::LinAlg::Matrix<3, 1, T> const& t1_target, T alpha, T cos_alpha,
    Core::LinAlg::Matrix<3, 1, T> const& dist_ul,
    Core::LinAlg::Matrix<1, 3, T> const& xi_target_partial_r_source,
    Core::LinAlg::Matrix<1, 3, T> const& xi_target_partial_r_target,
    Core::LinAlg::Matrix<1, 3, T> const& xi_target_partial_r_xi_target,
    double prefactor_visualization_data,
    Core::LinAlg::Matrix<3, 1, double>& vtk_force_pot_source_GP,
    Core::LinAlg::Matrix<3, 1, double>& vtk_force_pot_target_GP,
    Core::LinAlg::Matrix<3, 1, double>& vtk_moment_pot_source_GP,
    Core::LinAlg::Matrix<3, 1, double>& vtk_moment_pot_target_GP,
    double rho1rho2_JacFac_GaussWeight,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_source,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_source,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> const& N_i_target,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> const& N_i_xi_target)
{
  // get regularization type and separation
  const BeamInteraction::Potential::RegularizationType regularization_type =
      params()->regularization.type;

  const double regularization_separation = params()->regularization.separation;


  T sin_alpha = 0.0;   // sine of mutual angle of tangent vectors
  T sin_2alpha = 0.0;  // sine of 2*mutual angle of tangent vectors
  Core::LinAlg::Matrix<3, 1, T> normal_bl(
      Core::LinAlg::Initialization::zero);  // normal vector at bilateral closest point
  T norm_normal_bl_tilde = 0.0;             // norm of vector defining bilateral normal vector
  T gap_bl = 0.0;                           // gap of bilateral closest point
  T x = 0.0;  // distance between Gauss point and bilateral closest point on source
  Core::LinAlg::Matrix<3, 1, T> aux_plane_normal(
      Core::LinAlg::Initialization::zero);  // normal vector of auxiliary plane n*

  T beta = 0.0;       // auxiliary quantity
  T beta_exp2 = 0.0;  // beta^2
  T beta_exp3 = 0.0;  // beta^3
  T beta_exp4 = 0.0;  // beta^3
  T a = 0.0;          // auxiliary quantity
  T Delta = 0.0;      // auxiliary quantity
  T Delta_regularized = 0.0;

  T beta_partial_x = 0.0;
  T beta_partial_gap_bl = 0.0;
  T beta_partial_cos_alpha = 0.0;

  T a_partial_beta = 0.0;
  T a_partial_x = 0.0;
  T a_partial_gap_bl = 0.0;
  T a_partial_cos_alpha = 0.0;

  T Delta_partial_beta = 0.0;
  T Delta_partial_x = 0.0;
  T Delta_partial_gap_bl = 0.0;
  T Delta_partial_cos_alpha = 0.0;

  T pot_ia_partial_beta = 0.0;
  T pot_ia_partial_a = 0.0;
  T pot_ia_partial_Delta = 0.0;
  T pot_ia_partial_Delta_atregsep = 0.0;
  T pot_ia_partial_2ndderiv_Delta = 0.0;

  T pot_ia_partial_Delta_partial_beta = 0.0;
  T pot_ia_partial_2ndderiv_Delta_partial_beta = 0.0;

  T pot_ia_partial_Delta_partial_a = 0.0;
  T pot_ia_partial_2ndderiv_Delta_partial_a = 0.0;

  T pot_ia_partial_Delta_partial_gap_bl = 0.0;
  T pot_ia_partial_2ndderiv_Delta_partial_gap_bl = 0.0;

  T pot_ia_partial_x = 0.0;
  T pot_ia_partial_cos_alpha = 0.0;
  T pot_ia_partial_gap_bl = 0.0;


  // components from variation of bilateral gap
  Core::LinAlg::Matrix<3, 1, T> gap_bl_partial_r_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> gap_bl_partial_r_target(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> gap_bl_partial_r_xi_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> gap_bl_partial_r_xi_target(Core::LinAlg::Initialization::zero);
  T gap_bl_partial_xi_target = 0.0;

  // components from variation of cosine of enclosed angle
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_partial_r_xi_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_partial_r_xi_target(Core::LinAlg::Initialization::zero);
  T cos_alpha_partial_xi_target = 0.0;

  // components from variation of distance from bilateral closest point on source
  Core::LinAlg::Matrix<3, 1, T> x_partial_r_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> x_partial_r_target(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> x_partial_r_xi_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> x_partial_r_xi_target(Core::LinAlg::Initialization::zero);

  Core::LinAlg::Matrix<3, 1, T> x_partial_aux_plane_normal(Core::LinAlg::Initialization::zero);
  T x_partial_xi_target = 0.0;


  // auxiliary variables
  Core::LinAlg::Matrix<3, 1, T> fpot_tmp(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 3, T> v_mat_tmp(Core::LinAlg::Initialization::zero);

  sin_alpha = std::sin(alpha);
  sin_2alpha = std::sin(2 * alpha);

  const double BEAMSCOLINEARANGLETHRESHOLD = 5.0 / 180.0 * std::numbers::pi;  // 5 works best so far

  if (Core::FADUtils::norm(alpha) < BEAMSCOLINEARANGLETHRESHOLD)
  {
    // there is no unique bilateral closest point pair in case of alpha=0
    // hence, we can use the current (unilateral) closest point pair
    normal_bl = dist_ul;
    norm_normal_bl_tilde = Core::FADUtils::vector_norm(normal_bl);
    normal_bl.scale(1.0 / norm_normal_bl_tilde);

    aux_plane_normal.clear();
    x = 0.0;
  }
  else
  {
    // normal vector at bilateral closest point Fixme
    normal_bl.cross_product(r_xi_source, r_xi_target);
    norm_normal_bl_tilde = Core::FADUtils::vector_norm(normal_bl);
    normal_bl.scale(1.0 / norm_normal_bl_tilde);

    // distance between Gauss point and bilateral closest point on source
    aux_plane_normal.update(r_xi_target.dot(r_xi_target), r_xi_source,
        -1.0 * r_xi_target.dot(r_xi_source), r_xi_target);

    x = Core::FADUtils::vector_norm(r_xi_source) *
        (r_target.dot(aux_plane_normal) - r_source.dot(aux_plane_normal)) /
        r_xi_source.dot(aux_plane_normal);
  }

  // gap of bilateral closest point (also valid for special case alpha=0)
  gap_bl = Core::FADUtils::norm(dist_ul.dot(normal_bl)) - radius1_ - radius2_;

  const double MAXNEGATIVEBILATERALGAP = -0.9 * radius2_;

  if (Core::FADUtils::norm(alpha) >= BEAMSCOLINEARANGLETHRESHOLD and
      gap_bl < MAXNEGATIVEBILATERALGAP)
  {
    if (Core::FADUtils::norm(x) < 20 * radius2_)
    {
      FOUR_C_THROW(
          "Ignoring this GP with negative gap_bl in the non-parallel case "
          "violates the assumption that this GP is far from the bilateral CP");
    }
    else
    {
      return false;
    }
  }

  if (std::norm(Core::FADUtils::cast_to_double(gap_bl) + radius2_) < 1e-14)
    FOUR_C_THROW(
        "bilateral gap={} is close to negative radius and thus the interaction potential is "
        "close to singular! Fatal error. Take care of this case!",
        Core::FADUtils::cast_to_double(gap_bl));

  beta = Core::FADUtils::sqrt<T>(
      (gap_bl + radius2_) * (gap_bl + radius2_) + x * x * sin_alpha * sin_alpha);

  if (beta < 1e-14)
    FOUR_C_THROW("beta={} is negative or very close to zero! Fatal error. Take care of this case!",
        Core::FADUtils::cast_to_double(beta));

  beta_exp2 = beta * beta;
  beta_exp3 = beta_exp2 * beta;
  beta_exp4 = beta_exp2 * beta_exp2;

  a = 0.5 / beta *
      ((gap_bl + radius2_) / radius2_ + cos_alpha * cos_alpha -
          x * x * sin_2alpha * sin_2alpha / (4.0 * beta_exp2));

  Delta = 4 * a * (beta - radius2_) - x * x * sin_2alpha * sin_2alpha / (4 * beta_exp2);

  if (regularization_type == BeamInteraction::Potential::RegularizationType::none and Delta < 1e-14)
  {
    this->print(std::cout);

    std::cout << "\ngap_bl: " << Core::FADUtils::cast_to_double(gap_bl);
    std::cout << "\nalpha: " << Core::FADUtils::cast_to_double(alpha * 180 / std::numbers::pi)
              << "degrees";
    std::cout << "\nx: " << Core::FADUtils::cast_to_double(x) << std::endl;

    std::cout << "\nbeta: " << Core::FADUtils::cast_to_double(beta);
    std::cout << "\na: " << Core::FADUtils::cast_to_double(a);

    FOUR_C_THROW("Delta={} is negative or very close to zero! Use a regularization to handle this!",
        Core::FADUtils::cast_to_double(Delta));
  }

  Delta_regularized = Delta;

  if (regularization_type == BeamInteraction::Potential::RegularizationType::linear and
      Delta < regularization_separation)
  {
    Delta_regularized = regularization_separation;
  }

  if (Delta_regularized <= 0)
    FOUR_C_THROW(
        "regularized Delta <= 0! Fatal error since force law is not defined for "
        "zero/negative Delta! Use positive regularization separation!");


  // interaction potential
  interaction_potential_GP =
      beta / (gap_bl + radius2_) * std::pow(a, m_ - 5) * std::pow(Delta_regularized, -m_ + 4.5);


  // partial derivatives of beta
  beta_partial_x = x * sin_alpha * sin_alpha / beta;
  beta_partial_gap_bl = (gap_bl + radius2_) / beta;
  beta_partial_cos_alpha = -x * x * cos_alpha / beta;

  // partial derivatives of a
  a_partial_beta = -a / beta + x * x * sin_2alpha * sin_2alpha / (4 * beta_exp4);

  a_partial_x = a_partial_beta * beta_partial_x - x * sin_2alpha * sin_2alpha / (4 * beta_exp3);

  a_partial_gap_bl = a_partial_beta * beta_partial_gap_bl + 0.5 / (radius2_ * beta);

  a_partial_cos_alpha = a_partial_beta * beta_partial_cos_alpha + cos_alpha / beta -
                        x * x * cos_alpha * (1 - 2 * cos_alpha * cos_alpha) / beta_exp3;

  // partial derivatives of Delta
  Delta_partial_beta =
      2.0 * (((-(gap_bl + radius2_) / radius2_ - cos_alpha * cos_alpha) / beta_exp2 +
                 3.0 * x * x * sin_2alpha * sin_2alpha / 4.0 / beta_exp4) *
                    (beta - radius2_) +
                (gap_bl + radius2_) / beta / radius2_ + cos_alpha * cos_alpha / beta);

  Delta_partial_x = Delta_partial_beta * beta_partial_x -
                    x * sin_2alpha * sin_2alpha / beta_exp3 * (1.5 * beta - radius2_);

  Delta_partial_gap_bl =
      Delta_partial_beta * beta_partial_gap_bl + 2 / beta / radius2_ * (beta - radius2_);

  Delta_partial_cos_alpha =
      Delta_partial_beta * beta_partial_cos_alpha +
      4.0 *
          (cos_alpha / beta -
              x * x * (cos_alpha - 2.0 * cos_alpha * cos_alpha * cos_alpha) / beta_exp3) *
          (beta - radius2_) -
      2.0 * x * x * (cos_alpha - 2.0 * cos_alpha * cos_alpha * cos_alpha) / beta_exp2;

  // partial derivatives of single length specific interaction potential

  pot_ia_partial_beta = interaction_potential_GP / beta;

  pot_ia_partial_a = (m_ - 5.0) * interaction_potential_GP / a;

  pot_ia_partial_Delta = (-m_ + 4.5) * interaction_potential_GP / Delta_regularized;


  if (regularization_type == BeamInteraction::Potential::RegularizationType::linear and
      Delta < regularization_separation)
  {
    // partial derivative w.r.t. Delta

    // store this as an intermediate result that is needed below
    pot_ia_partial_Delta_atregsep = pot_ia_partial_Delta;

    pot_ia_partial_2ndderiv_Delta = (-m_ + 3.5) / Delta_regularized * pot_ia_partial_Delta_atregsep;

    pot_ia_partial_Delta += pot_ia_partial_2ndderiv_Delta * (Delta - regularization_separation);

    // partial derivative w.r.t. beta
    pot_ia_partial_Delta_partial_beta = pot_ia_partial_Delta_atregsep / beta;
    pot_ia_partial_2ndderiv_Delta_partial_beta = pot_ia_partial_2ndderiv_Delta / beta;

    pot_ia_partial_beta += pot_ia_partial_Delta_partial_beta * (Delta - regularization_separation) +
                           0.5 * pot_ia_partial_2ndderiv_Delta_partial_beta *
                               (Delta - regularization_separation) *
                               (Delta - regularization_separation);

    // partial derivative w.r.t. a
    pot_ia_partial_Delta_partial_a = (m_ - 5.0) * pot_ia_partial_Delta_atregsep / a;
    pot_ia_partial_2ndderiv_Delta_partial_a = (m_ - 5.0) * pot_ia_partial_2ndderiv_Delta / a;

    pot_ia_partial_a += pot_ia_partial_Delta_partial_a * (Delta - regularization_separation) +
                        0.5 * pot_ia_partial_2ndderiv_Delta_partial_a *
                            (Delta - regularization_separation) *
                            (Delta - regularization_separation);
  }

  pot_ia_partial_x = pot_ia_partial_beta * beta_partial_x + pot_ia_partial_a * a_partial_x +
                     pot_ia_partial_Delta * Delta_partial_x;

  pot_ia_partial_gap_bl =
      pot_ia_partial_beta * beta_partial_gap_bl + pot_ia_partial_a * a_partial_gap_bl +
      pot_ia_partial_Delta * Delta_partial_gap_bl - interaction_potential_GP / (gap_bl + radius2_);

  pot_ia_partial_cos_alpha = pot_ia_partial_beta * beta_partial_cos_alpha +
                             pot_ia_partial_a * a_partial_cos_alpha +
                             pot_ia_partial_Delta * Delta_partial_cos_alpha;


  if (regularization_type == BeamInteraction::Potential::RegularizationType::linear and
      Delta < regularization_separation)
  {
    // partial derivative w.r.t. bilateral gap
    pot_ia_partial_Delta_partial_gap_bl =
        (-1.0) * pot_ia_partial_Delta_atregsep / (gap_bl + radius2_);
    pot_ia_partial_2ndderiv_Delta_partial_gap_bl =
        (-1.0) * pot_ia_partial_2ndderiv_Delta / (gap_bl + radius2_);

    pot_ia_partial_gap_bl +=
        pot_ia_partial_Delta_partial_gap_bl * (Delta - regularization_separation) +
        0.5 * pot_ia_partial_2ndderiv_Delta_partial_gap_bl * (Delta - regularization_separation) *
            (Delta - regularization_separation);
  }



  /* now that we don't need the interaction potential at Delta=regularization_separation as
   * an intermediate result anymore, we can
   * add the additional (linear and quadratic) contributions in case of active
   * regularization also to the interaction potential */
  if (regularization_type == BeamInteraction::Potential::RegularizationType::linear and
      Delta < regularization_separation)
  {
    interaction_potential_GP +=
        pot_ia_partial_Delta_atregsep * (Delta - regularization_separation) +
        0.5 * pot_ia_partial_2ndderiv_Delta * (Delta - regularization_separation) *
            (Delta - regularization_separation);
  }


  // components from variation of bilateral gap
  T signum_dist_bl_tilde = Core::FADUtils::signum(dist_ul.dot(normal_bl));

  gap_bl_partial_r_source.update(signum_dist_bl_tilde, normal_bl);
  gap_bl_partial_r_target.update(-1.0 * signum_dist_bl_tilde, normal_bl);

  // Todo: check and remove: the following term should vanish since r_xi_target \perp normal_bl
  gap_bl_partial_xi_target = -1.0 * signum_dist_bl_tilde * r_xi_target.dot(normal_bl);

  v_mat_tmp.clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= normal_bl(i) * normal_bl(j);
  }

  Core::LinAlg::Matrix<3, 1, T> vec_tmp(Core::LinAlg::Initialization::zero);
  vec_tmp.multiply(v_mat_tmp, dist_ul);


  if (alpha < BEAMSCOLINEARANGLETHRESHOLD)
  {
    // Todo: check and remove: signum_dist_bl_tilde should always be +1.0 in this case!

    gap_bl_partial_r_xi_source.clear();

    gap_bl_partial_r_source.update(signum_dist_bl_tilde / norm_normal_bl_tilde, vec_tmp, 1.0);

    gap_bl_partial_r_xi_target.clear();

    gap_bl_partial_r_target.update(
        -1.0 * signum_dist_bl_tilde / norm_normal_bl_tilde, vec_tmp, 1.0);

    // Todo: check and remove: the following term should vanish
    gap_bl_partial_xi_target -=
        signum_dist_bl_tilde / norm_normal_bl_tilde * r_xi_target.dot(vec_tmp);
  }
  else
  {
    gap_bl_partial_r_xi_source.update(signum_dist_bl_tilde / norm_normal_bl_tilde,
        Core::FADUtils::vector_product(r_xi_target, vec_tmp));

    gap_bl_partial_r_xi_target.update(-1.0 * signum_dist_bl_tilde / norm_normal_bl_tilde,
        Core::FADUtils::vector_product(r_xi_source, vec_tmp));

    gap_bl_partial_xi_target += r_xixi_target.dot(gap_bl_partial_r_xi_target);
  }


  // components from variation of cosine of enclosed angle
  T signum_tangentsscalarproduct = Core::FADUtils::signum(r_xi_source.dot(r_xi_target));

  v_mat_tmp.clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t1_source(i) * t1_source(j);
  }

  cos_alpha_partial_r_xi_source.multiply(
      signum_tangentsscalarproduct / Core::FADUtils::vector_norm(r_xi_source), v_mat_tmp,
      t1_target);

  v_mat_tmp.clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t1_target(i) * t1_target(j);
  }

  cos_alpha_partial_r_xi_target.multiply(
      signum_tangentsscalarproduct / Core::FADUtils::vector_norm(r_xi_target), v_mat_tmp,
      t1_source);

  cos_alpha_partial_xi_target = r_xixi_target.dot(cos_alpha_partial_r_xi_target);


  // components from variation of distance from bilateral closest point on source
  if (Core::FADUtils::cast_to_double(alpha) < BEAMSCOLINEARANGLETHRESHOLD)
  {
    /* set the following quantities to zero since they are not required in this case
     * because pot_ia_partial_x is zero anyway */
    x_partial_r_source.clear();
    x_partial_r_target.clear();
    x_partial_r_xi_source.clear();
    x_partial_r_xi_target.clear();
    x_partial_xi_target = 0.0;
  }
  else
  {
    x_partial_r_source.update(-1.0 / t1_source.dot(aux_plane_normal), aux_plane_normal);
    x_partial_r_target.update(1.0 / t1_source.dot(aux_plane_normal), aux_plane_normal);

    v_mat_tmp.clear();
    for (unsigned int i = 0; i < 3; ++i)
    {
      v_mat_tmp(i, i) += 1.0;
      for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t1_source(i) * t1_source(j);
    }

    x_partial_r_xi_source.multiply(1.0 / Core::FADUtils::vector_norm(r_xi_source) *
                                       dist_ul.dot(aux_plane_normal) /
                                       std::pow(t1_source.dot(aux_plane_normal), 2),
        v_mat_tmp, aux_plane_normal);

    x_partial_aux_plane_normal.update(-1.0 / t1_source.dot(aux_plane_normal), dist_ul,
        dist_ul.dot(aux_plane_normal) / std::pow(t1_source.dot(aux_plane_normal), 2), t1_source);

    v_mat_tmp.clear();
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) += r_xi_target(i) * r_xi_target(j);

    x_partial_r_xi_source.update(r_xi_target.dot(r_xi_target), x_partial_aux_plane_normal, 1.0);

    x_partial_r_xi_source.multiply(-1.0, v_mat_tmp, x_partial_aux_plane_normal, 1.0);


    x_partial_r_xi_target.update(-1.0 * r_xi_target.dot(r_xi_source), x_partial_aux_plane_normal);

    v_mat_tmp.clear();
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        v_mat_tmp(i, j) += 2.0 * r_xi_target(i) * r_xi_source(j);
        v_mat_tmp(i, j) -= 1.0 * r_xi_source(i) * r_xi_target(j);
      }

    x_partial_r_xi_target.multiply(1.0, v_mat_tmp, x_partial_aux_plane_normal, 1.0);

    x_partial_xi_target = r_xixi_target.dot(x_partial_r_xi_target);
  }


  /* store for visualization
   * note: contributions to the force vector can either be visualized as distributed forces (in
   * combination with the variation of the centerline) or distributed moments (in combination with
   * the variation of the centerline derivative with respect to the parameter coordinate xi).
   */

  /* note on distributed force visualization
   * contrary to the distributed moment visualization, the distributed force does not need any form
   * of transformation, since the contribution to the force vector in combination with the variation
   * of the centerline is already a distributed force
   */

  // distributed force on source
  vtk_force_pot_source_GP.update(
      prefactor_visualization_data * Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl),
      Core::FADUtils::cast_to_double<T, 3, 1>(gap_bl_partial_r_source), 1.0);

  vtk_force_pot_source_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl) *
                                       Core::FADUtils::cast_to_double(gap_bl_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_source), 1.0);

  vtk_force_pot_source_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_ia_partial_cos_alpha) *
                                       Core::FADUtils::cast_to_double(cos_alpha_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_source), 1.0);

  vtk_force_pot_source_GP.update(
      prefactor_visualization_data * Core::FADUtils::cast_to_double(pot_ia_partial_x),
      Core::FADUtils::cast_to_double<T, 3, 1>(x_partial_r_source), 1.0);

  vtk_force_pot_source_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_ia_partial_x) *
                                       Core::FADUtils::cast_to_double(x_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_source), 1.0);

  // distributed force on target
  vtk_force_pot_target_GP.update(
      prefactor_visualization_data * Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl),
      Core::FADUtils::cast_to_double<T, 3, 1>(gap_bl_partial_r_target), 1.0);

  vtk_force_pot_target_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl) *
                                       Core::FADUtils::cast_to_double(gap_bl_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_target), 1.0);

  vtk_force_pot_target_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_ia_partial_cos_alpha) *
                                       Core::FADUtils::cast_to_double(cos_alpha_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_target), 1.0);

  vtk_force_pot_target_GP.update(
      prefactor_visualization_data * Core::FADUtils::cast_to_double(pot_ia_partial_x),
      Core::FADUtils::cast_to_double<T, 3, 1>(x_partial_r_target), 1.0);

  vtk_force_pot_target_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_ia_partial_x) *
                                       Core::FADUtils::cast_to_double(x_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_target), 1.0);

  /* note on distributed moment visualization
   * Contrary to the distributed force visualization, the distributed moment can not be directly
   * visualized from the force vector because this only represents a pseudo-moment. This
   * pseudo-moment can be transformed into a distributed moment as follows:
   *
   * \f[
   *    \delta W = \delta \boldsymbol{r}_2^{\shortmid T} \cdot \boldsymbol{m}_{\text{pseudo}}
   *             = \delta \boldsymbol{\theta}_\perp^T \cdot \boldsymbol{m}
   * \f]
   * where \f$\boldsymbol{m}_{\text{pseudo}}\f$ is the contribution from the force vector and
   * \f$\boldsymbol{m}\f$ is the distributed moment for visualization.
   *
   * Note: Here we only consider bending moments which are perpendicular to the tangent vector
   *       r_xi. That's why we only consider the transversal part of the rotation vector
   *       theta_perp which is perpendicular to the tangent vector r_xi.
   *
   * The variation of the rotation vector \f$\delta \boldsymbol{\theta}_\perp\f$ can be found,
   * for example, in "Meier, C.: Geometrically exact finite element formulations for slender
   * beams and their contact interaction, Technical University of Munich, Germany, 2016",
   * formula (2.26).
   *
   * Rearranging the equation above results in the actual distributed moment we want to visualize:
   * \f[
   *    \boldsymbol{m} = \boldsymbol{r}_2^{\shortmid T} \cross \boldsymbol{m}_{\text{pseudo}}
   * \f]
   * or in direct notation => m = r_xi x m_pseudo)
   */

  Core::LinAlg::Matrix<3, 1, double> m_pseudo(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> m(Core::LinAlg::Initialization::zero);

  // distributed moment on source
  m_pseudo.update(Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl),
      Core::FADUtils::cast_to_double<T, 3, 1>(gap_bl_partial_r_xi_source));

  m_pseudo.update(Core::FADUtils::cast_to_double(pot_ia_partial_cos_alpha),
      Core::FADUtils::cast_to_double<T, 3, 1>(cos_alpha_partial_r_xi_source), 1.0);

  m_pseudo.update(Core::FADUtils::cast_to_double(pot_ia_partial_x),
      Core::FADUtils::cast_to_double<T, 3, 1>(x_partial_r_xi_source), 1.0);

  m.cross_product(Core::FADUtils::cast_to_double<T, 3, 1>(r_xi_source), m_pseudo);

  vtk_moment_pot_source_GP.update(prefactor_visualization_data, m, 1.0);

  // distributed moment on target
  m_pseudo.update(Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl),
      Core::FADUtils::cast_to_double<T, 3, 1>(gap_bl_partial_r_xi_target));

  m_pseudo.update_t(Core::FADUtils::cast_to_double(pot_ia_partial_gap_bl) *
                        Core::FADUtils::cast_to_double(gap_bl_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_xi_target), 1.0);

  m_pseudo.update(Core::FADUtils::cast_to_double(pot_ia_partial_cos_alpha),
      Core::FADUtils::cast_to_double<T, 3, 1>(cos_alpha_partial_r_xi_target), 1.0);

  m_pseudo.update_t(Core::FADUtils::cast_to_double(pot_ia_partial_cos_alpha) *
                        Core::FADUtils::cast_to_double(cos_alpha_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_xi_target), 1.0);

  m_pseudo.update(Core::FADUtils::cast_to_double(pot_ia_partial_x),
      Core::FADUtils::cast_to_double<T, 3, 1>(x_partial_r_xi_target), 1.0);

  m_pseudo.update_t(Core::FADUtils::cast_to_double(pot_ia_partial_x) *
                        Core::FADUtils::cast_to_double(x_partial_xi_target),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_xi_target), 1.0);

  m.cross_product(Core::FADUtils::cast_to_double<T, 3, 1>(r_xi_target), m_pseudo);

  vtk_moment_pot_target_GP.update(prefactor_visualization_data, m, 1.0);

  // integration factor
  interaction_potential_GP *= rho1rho2_JacFac_GaussWeight;
  pot_ia_partial_gap_bl *= rho1rho2_JacFac_GaussWeight;
  pot_ia_partial_cos_alpha *= rho1rho2_JacFac_GaussWeight;
  pot_ia_partial_x *= rho1rho2_JacFac_GaussWeight;

  //********************************************************************
  // calculate fpot1: force/residual vector on source element
  //********************************************************************
  force_pot_source_GP.clear();
  // sum up the contributions of all nodes (in all dimensions)
  for (unsigned int idofperdim = 0; idofperdim < (numnodes * numnodalvalues); ++idofperdim)
  {
    // loop over dimensions
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
    {
      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_source(idofperdim) * pot_ia_partial_gap_bl *
          (gap_bl_partial_r_source(jdim) +
              gap_bl_partial_xi_target * xi_target_partial_r_source(jdim));

      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_xi_source(idofperdim) * pot_ia_partial_gap_bl * gap_bl_partial_r_xi_source(jdim);


      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_source(idofperdim) * pot_ia_partial_cos_alpha * cos_alpha_partial_xi_target *
          xi_target_partial_r_source(jdim);

      force_pot_source_GP(3 * idofperdim + jdim) += N_i_xi_source(idofperdim) *
                                                    pot_ia_partial_cos_alpha *
                                                    cos_alpha_partial_r_xi_source(jdim);


      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_source(idofperdim) * pot_ia_partial_x *
          (x_partial_r_source(jdim) + x_partial_xi_target * xi_target_partial_r_source(jdim));

      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_xi_source(idofperdim) * pot_ia_partial_x * x_partial_r_xi_source(jdim);
    }
  }

  //********************************************************************
  // calculate fpot2: force/residual vector on target element
  //********************************************************************
  force_pot_target_GP.clear();
  // sum up the contributions of all nodes (in all dimensions)
  for (unsigned int idofperdim = 0; idofperdim < (numnodes * numnodalvalues); ++idofperdim)
  {
    // loop over dimensions
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
    {
      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_target(idofperdim) * pot_ia_partial_gap_bl *
          (gap_bl_partial_r_target(jdim) +
              gap_bl_partial_xi_target * xi_target_partial_r_target(jdim));

      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_xi_target(idofperdim) * pot_ia_partial_gap_bl *
          (gap_bl_partial_r_xi_target(jdim) +
              gap_bl_partial_xi_target * xi_target_partial_r_xi_target(jdim));


      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_target(idofperdim) * pot_ia_partial_cos_alpha * cos_alpha_partial_xi_target *
          xi_target_partial_r_target(jdim);

      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_xi_target(idofperdim) * pot_ia_partial_cos_alpha *
          (cos_alpha_partial_r_xi_target(jdim) +
              cos_alpha_partial_xi_target * xi_target_partial_r_xi_target(jdim));


      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_target(idofperdim) * pot_ia_partial_x *
          (x_partial_r_target(jdim) + x_partial_xi_target * xi_target_partial_r_target(jdim));

      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_xi_target(idofperdim) * pot_ia_partial_x *
          (x_partial_r_xi_target(jdim) + x_partial_xi_target * xi_target_partial_r_xi_target(jdim));
    }
  }

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::evaluate_simple_disk_cylinder_potential(Core::LinAlg::Matrix<3, 1, T> const& dist_ul,
    T norm_dist_ul, T alpha, T cos_alpha, Core::LinAlg::Matrix<3, 1, T> const& r_source,
    Core::LinAlg::Matrix<3, 1, T> const& r_xi_source, T norm_r_xi_source,
    Core::LinAlg::Matrix<3, 1, T> const& t_source, Core::LinAlg::Matrix<3, 1, T> const& r_target,
    Core::LinAlg::Matrix<3, 1, T> const& r_xi_target, T norm_r_xi_target,
    Core::LinAlg::Matrix<3, 1, T> const& r_xixi_target,
    Core::LinAlg::Matrix<3, 1, T> const& t_target, T xi_target,
    Core::LinAlg::Matrix<1, 3, T> const& xi_target_partial_r_source,
    Core::LinAlg::Matrix<1, 3, T> const& xi_target_partial_r_target,
    Core::LinAlg::Matrix<1, 3, T> const& xi_target_partial_r_xi_target,
    std::optional<double> potential_reduction_length, double length_prior_right,
    double length_prior_left, T& interaction_potential_GP,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_source,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_source,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> const& N_i_target,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> const& N_i_xi_target,
    Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T> const& N_i_xixi_target,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot_source_GP,
    Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot_target_GP,
    double prefactor_visualization_data, double rho1rho2_JacFac_GaussWeight,
    Core::LinAlg::Matrix<3, 1, double>& vtk_force_pot_source_GP,
    Core::LinAlg::Matrix<3, 1, double>& vtk_force_pot_target_GP,
    Core::LinAlg::Matrix<3, 1, double>& vtk_moment_pot_source_GP,
    Core::LinAlg::Matrix<3, 1, double>& vtk_moment_pot_target_GP,
    Core::LinAlg::SerialDenseMatrix* stiffmat11, Core::LinAlg::SerialDenseMatrix* stiffmat12,
    Core::LinAlg::SerialDenseMatrix* stiffmat21, Core::LinAlg::SerialDenseMatrix* stiffmat22)
{
  // get regularization type and separation
  const BeamInteraction::Potential::RegularizationType regularization_type =
      params()->regularization.type;

  const double regularization_separation = params()->regularization.separation;

  T signum_tangentsscalarproduct = 0.0;

  // unilateral normal vector
  Core::LinAlg::Matrix<3, 1, T> normal_ul(Core::LinAlg::Initialization::zero);

  // gap of unilateral closest points
  T gap_ul = 0.0;
  // regularized gap of unilateral closest points
  T gap_ul_regularized = 0.0;

  // auxiliary quantity
  T a = 0.0;
  T a_deriv_cos_alpha = 0.0;

  T potential_reduction_factor_GP = 1.0;

  // derivatives of the interaction potential w.r.t. gap_ul and cos(alpha)
  T pot_ia_deriv_gap_ul = 0.0;
  T pot_ia_deriv_cos_alpha = 0.0;

  T pot_ia_2ndderiv_gap_ul = 0.0;
  T pot_ia_2ndderiv_cos_alpha = 0.0;
  T pot_ia_deriv_gap_ul_deriv_cos_alpha = 0.0;

  // third derivatives
  T pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep = 0.0;
  T pot_ia_deriv_gap_ul_2ndderiv_cos_alpha_at_regsep = 0.0;

  // fourth derivative
  T pot_ia_2ndderiv_gap_ul_2ndderiv_cos_alpha_at_regsep = 0.0;

  // derivatives of the potential reduction factor w.r.t. xi_target
  T pot_red_fac_deriv_l_edge = 0.0;
  T pot_red_fac_2ndderiv_l_edge = 0.0;
  T l_edge_deriv_xi_target = 0.0;
  T pot_red_fac_deriv_xi_target = 0.0;
  T pot_red_fac_2ndderiv_xi_target = 0.0;

  // components from variation of unilateral gap
  Core::LinAlg::Matrix<3, 1, T> gap_ul_deriv_r_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> gap_ul_deriv_r_target(Core::LinAlg::Initialization::zero);

  // components from variation of cosine of enclosed angle
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_partial_r_xi_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_partial_r_xi_target(Core::LinAlg::Initialization::zero);
  T cos_alpha_partial_xi_target = 0.0;

  Core::LinAlg::Matrix<3, 1, T> cos_alpha_deriv_r_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_deriv_r_target(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_deriv_r_xi_source(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, T> cos_alpha_deriv_r_xi_target(Core::LinAlg::Initialization::zero);

  // gap of unilateral closest points
  gap_ul = norm_dist_ul - radius1_ - radius2_;

  // regularized gap of unilateral closest points
  gap_ul_regularized = gap_ul;


  if (regularization_type != BeamInteraction::Potential::RegularizationType::none and
      gap_ul < regularization_separation)
  {
    gap_ul_regularized = regularization_separation;
  }
  else if (regularization_type == BeamInteraction::Potential::RegularizationType::none and
           gap_ul < 1e-14)
  {
    this->print(std::cout);

    std::cout << "\ngap_ul: " << Core::FADUtils::cast_to_double(gap_ul);
    std::cout << "\nalpha: " << Core::FADUtils::cast_to_double(alpha * 180 / std::numbers::pi)
              << "degrees";

    FOUR_C_THROW(
        "gap_ul={} is negative or very close to zero! Fatal error. Use regularization to"
        " handle this!",
        Core::FADUtils::cast_to_double(gap_ul));
  }


  // unilateral normal vector
  normal_ul.update(1.0 / norm_dist_ul, dist_ul);


  // auxiliary quantity
  a = 0.5 / radius1_ + 0.5 * cos_alpha * cos_alpha / radius2_;

  // safety check
  if (Core::FADUtils::cast_to_double(a) <= 0.0)
  {
    this->print(std::cout);
    FOUR_C_THROW("auxiliary quantity a<=0!");
  }

  // interaction potential
  interaction_potential_GP = std::pow(4 * gap_ul_regularized, -m_ + 4.5) / Core::FADUtils::sqrt(a);

  // for two-half-pass approach the interaction potential is halved
  if (params()->two_half_pass) interaction_potential_GP *= 0.5;


  // compute derivatives of the interaction potential w.r.t. gap_ul and cos(alpha)
  pot_ia_deriv_gap_ul = (-m_ + 4.5) / gap_ul_regularized * interaction_potential_GP;

  a_deriv_cos_alpha = cos_alpha / radius2_;
  pot_ia_deriv_cos_alpha = -0.5 / a * interaction_potential_GP * a_deriv_cos_alpha;


  // Todo: strictly speaking, the following second derivatives of the interaction potential
  // are only required in case of active regularization OR for analytic linearization

  pot_ia_deriv_gap_ul_deriv_cos_alpha = (-m_ + 4.5) / gap_ul_regularized * pot_ia_deriv_cos_alpha;

  pot_ia_2ndderiv_gap_ul = (-m_ + 3.5) / gap_ul_regularized * pot_ia_deriv_gap_ul;

  pot_ia_2ndderiv_cos_alpha =
      -0.5 / (a * radius2_) * interaction_potential_GP +
      0.75 * a_deriv_cos_alpha * a_deriv_cos_alpha / (a * a) * interaction_potential_GP;

  // third derivatives
  pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep = (-m_ + 4.5) * (-m_ + 3.5) /
                                                    (gap_ul_regularized * gap_ul_regularized) *
                                                    pot_ia_deriv_cos_alpha;

  pot_ia_deriv_gap_ul_2ndderiv_cos_alpha_at_regsep =
      (-m_ + 4.5) / gap_ul_regularized * pot_ia_2ndderiv_cos_alpha;

  // fourth derivative
  pot_ia_2ndderiv_gap_ul_2ndderiv_cos_alpha_at_regsep = (-m_ + 4.5) * (-m_ + 3.5) /
                                                        (gap_ul_regularized * gap_ul_regularized) *
                                                        pot_ia_2ndderiv_cos_alpha;

  /* now that we don't need the interaction potential at gap_ul=regularization_separation as
   * an intermediate result anymore, we can add the additional (linear and quadratic)
   * contributions in case of active regularization to the interaction potential */
  if (regularization_type != BeamInteraction::Potential::RegularizationType::none and
      gap_ul < regularization_separation)
  {
    interaction_potential_GP += pot_ia_deriv_gap_ul * (gap_ul - regularization_separation);

    pot_ia_deriv_cos_alpha +=
        pot_ia_deriv_gap_ul_deriv_cos_alpha * (gap_ul - regularization_separation);


    if (regularization_type == BeamInteraction::Potential::RegularizationType::linear)
    {
      interaction_potential_GP += 0.5 * pot_ia_2ndderiv_gap_ul *
                                  (gap_ul - regularization_separation) *
                                  (gap_ul - regularization_separation);

      pot_ia_deriv_gap_ul += pot_ia_2ndderiv_gap_ul * (gap_ul - regularization_separation);

      pot_ia_deriv_cos_alpha += 0.5 * pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep *
                                (gap_ul - regularization_separation) *
                                (gap_ul - regularization_separation);
    }
  }

  /* adapt also the second derivatives (required for analytic linearization) in
   * case of active regularization*/
  if (regularization_type != BeamInteraction::Potential::RegularizationType::none and
      gap_ul < regularization_separation)
  {
    if (regularization_type == BeamInteraction::Potential::RegularizationType::constant)
      pot_ia_2ndderiv_gap_ul = 0.0;

    pot_ia_2ndderiv_cos_alpha +=
        pot_ia_deriv_gap_ul_2ndderiv_cos_alpha_at_regsep * (gap_ul - regularization_separation);

    if (regularization_type == BeamInteraction::Potential::RegularizationType::linear)
    {
      pot_ia_deriv_gap_ul_deriv_cos_alpha +=
          pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep * (gap_ul - regularization_separation);

      pot_ia_2ndderiv_cos_alpha += 0.5 * pot_ia_2ndderiv_gap_ul_2ndderiv_cos_alpha_at_regsep *
                                   (gap_ul - regularization_separation) *
                                   (gap_ul - regularization_separation);
    }
  }

  // determine potential reduction factor for target beam endpoint strategy
  if (potential_reduction_length.has_value())
  {
    T left_length_to_edge = length_prior_left + ele2length_ * 0.5 * (1 + xi_target);
    T right_length_to_edge = length_prior_right + ele2length_ * 0.5 * (1 - xi_target);
    T length_to_edge = -1.0;
    bool right_node = false;

    // determine potential reduction factor (target beam can only consist of one element!)
    if (left_length_to_edge < potential_reduction_length.value())
    {
      length_to_edge = left_length_to_edge;
    }
    else if (right_length_to_edge < potential_reduction_length.value())
    {
      length_to_edge = right_length_to_edge;
      right_node = true;
    }

    if (length_to_edge != -1.0)
    {
      if (params()->potential_reduction_function ==
          BeamInteraction::Potential::ReductionFunction::cosine)
      {
        potential_reduction_factor_GP = 0.5 - 0.5 * std::cos(std::numbers::pi * length_to_edge /
                                                             potential_reduction_length.value());
        pot_red_fac_deriv_l_edge =
            0.5 * std::numbers::pi / potential_reduction_length.value() *
            std::sin(std::numbers::pi * length_to_edge / potential_reduction_length.value());
        pot_red_fac_2ndderiv_l_edge =
            0.5 * std::numbers::pi * std::numbers::pi /
            (potential_reduction_length.value() * potential_reduction_length.value()) *
            std::cos(std::numbers::pi * length_to_edge / potential_reduction_length.value());
      }
      else if (params()->potential_reduction_function ==
               BeamInteraction::Potential::ReductionFunction::polynomial_5)
      {
        Core::FE::Polynomial polynomial =
            Core::FE::Polynomial({0, 0, 0, 10 / std::pow(potential_reduction_length.value(), 3),
                -15 / std::pow(potential_reduction_length.value(), 4),
                6 / std::pow(potential_reduction_length.value(), 5)});

        potential_reduction_factor_GP =
            polynomial.evaluate(Core::FADUtils::cast_to_double(length_to_edge));

        pot_red_fac_deriv_l_edge =
            polynomial.evaluate_derivative(Core::FADUtils::cast_to_double(length_to_edge), 1);

        pot_red_fac_2ndderiv_l_edge =
            polynomial.evaluate_derivative(Core::FADUtils::cast_to_double(length_to_edge), 2);
      }
      else if (params()->potential_reduction_function ==
               BeamInteraction::Potential::ReductionFunction::polynomial_7)
      {
        Core::FE::Polynomial polynomial =
            Core::FE::Polynomial({0, 0, 0, 0, 35 / std::pow(potential_reduction_length.value(), 4),
                -84 / std::pow(potential_reduction_length.value(), 5),
                70 / std::pow(potential_reduction_length.value(), 6),
                -20 / std::pow(potential_reduction_length.value(), 7)});

        potential_reduction_factor_GP =
            polynomial.evaluate(Core::FADUtils::cast_to_double(length_to_edge));
        pot_red_fac_deriv_l_edge =
            polynomial.evaluate_derivative(Core::FADUtils::cast_to_double(length_to_edge), 1);

        pot_red_fac_2ndderiv_l_edge =
            polynomial.evaluate_derivative(Core::FADUtils::cast_to_double(length_to_edge), 2);
      }
    }

    l_edge_deriv_xi_target = 0.5 * ele2length_;

    if (right_node) l_edge_deriv_xi_target *= -1.0;

    pot_red_fac_deriv_xi_target = pot_red_fac_deriv_l_edge * l_edge_deriv_xi_target;
    pot_red_fac_2ndderiv_xi_target =
        pot_red_fac_2ndderiv_l_edge * l_edge_deriv_xi_target * l_edge_deriv_xi_target;
  }

  // compute components from variation of cosine of enclosed angle
  signum_tangentsscalarproduct = Core::FADUtils::signum(r_xi_source.dot(r_xi_target));
  // auxiliary variables
  Core::LinAlg::Matrix<3, 3, T> v_mat_tmp(Core::LinAlg::Initialization::zero);

  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t_source(i) * t_source(j);
  }

  cos_alpha_partial_r_xi_source.multiply(
      signum_tangentsscalarproduct / Core::FADUtils::vector_norm(r_xi_source), v_mat_tmp, t_target);

  v_mat_tmp.clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t_target(i) * t_target(j);
  }

  cos_alpha_partial_r_xi_target.multiply(
      signum_tangentsscalarproduct / Core::FADUtils::vector_norm(r_xi_target), v_mat_tmp, t_source);

  cos_alpha_partial_xi_target = r_xixi_target.dot(cos_alpha_partial_r_xi_target);


  cos_alpha_deriv_r_source.update_t(cos_alpha_partial_xi_target, xi_target_partial_r_source);
  cos_alpha_deriv_r_target.update_t(cos_alpha_partial_xi_target, xi_target_partial_r_target);

  cos_alpha_deriv_r_xi_source.update(cos_alpha_partial_r_xi_source);
  cos_alpha_deriv_r_xi_target.update(1.0, cos_alpha_partial_r_xi_target);
  cos_alpha_deriv_r_xi_target.update_t(
      cos_alpha_partial_xi_target, xi_target_partial_r_xi_target, 1.0);


  // compute components from variation of the unilateral gap
  gap_ul_deriv_r_source.update(normal_ul);
  gap_ul_deriv_r_target.update(-1.0, normal_ul);


  /* store for visualization
   * note: contributions to the force vector can either be visualized as distributed forces (in
   * combination with the variation of the centerline) or distributed moments (in combination with
   * the variation of the centerline derivative with respect to the parameter coordinate xi).
   */

  /* note on distributed force visualization
   * contrary to the distributed moment visualization, the distributed force does not need any form
   * of transformation, since the contribution to the force vector in combination with the variation
   * of the centerline is already a distributed force
   */

  // distributed force on source
  vtk_force_pot_source_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_red_fac_deriv_xi_target) *
                                       Core::FADUtils::cast_to_double(interaction_potential_GP),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_source), 1.0);

  vtk_force_pot_source_GP.update(prefactor_visualization_data *
                                     Core::FADUtils::cast_to_double(potential_reduction_factor_GP) *
                                     Core::FADUtils::cast_to_double(pot_ia_deriv_gap_ul),
      Core::FADUtils::cast_to_double<T, 3, 1>(gap_ul_deriv_r_source), 1.0);

  vtk_force_pot_source_GP.update(prefactor_visualization_data *
                                     Core::FADUtils::cast_to_double(potential_reduction_factor_GP) *
                                     Core::FADUtils::cast_to_double(pot_ia_deriv_cos_alpha),
      Core::FADUtils::cast_to_double<T, 3, 1>(cos_alpha_deriv_r_source), 1.0);

  // distributed force on target
  vtk_force_pot_target_GP.update_t(prefactor_visualization_data *
                                       Core::FADUtils::cast_to_double(pot_red_fac_deriv_xi_target) *
                                       Core::FADUtils::cast_to_double(interaction_potential_GP),
      Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_target), 1.0);

  vtk_force_pot_target_GP.update(prefactor_visualization_data *
                                     Core::FADUtils::cast_to_double(potential_reduction_factor_GP) *
                                     Core::FADUtils::cast_to_double(pot_ia_deriv_gap_ul),
      Core::FADUtils::cast_to_double<T, 3, 1>(gap_ul_deriv_r_target), 1.0);

  vtk_force_pot_target_GP.update(prefactor_visualization_data *
                                     Core::FADUtils::cast_to_double(potential_reduction_factor_GP) *
                                     Core::FADUtils::cast_to_double(pot_ia_deriv_cos_alpha),
      Core::FADUtils::cast_to_double<T, 3, 1>(cos_alpha_deriv_r_target), 1.0);

  /* note on distributed moment visualization
   * Contrary to the distributed force visualization, the distributed moment can not be directly
   * visualized from the force vector because this only represents a pseudo-moment. This
   * pseudo-moment can be transformed into a distributed moment as follows:
   *
   * \f[
   *    \delta W = \delta \boldsymbol{r}_2^{\shortmid T} \cdot \boldsymbol{m}_{\text{pseudo}}
   *             = \delta \boldsymbol{\theta}_\perp^T \cdot \boldsymbol{m}
   * \f]
   * where \f$\boldsymbol{m}_{\text{pseudo}}\f$ is the contribution from the force vector and
   * \f$\boldsymbol{m}\f$ is the distributed moment for visualization.
   *
   * Note: Here we only consider bending moments which are perpendicular to the tangent vector
   *       r_xi. That's why we only consider the transversal part of the rotation vector
   *       theta_perp which is perpendicular to the tangent vector r_xi.
   *
   * The variation of the rotation vector \f$\delta \boldsymbol{\theta}_\perp\f$ can be found,
   * for example, in "Meier, C.: Geometrically exact finite element formulations for slender
   * beams and their contact interaction, Technical University of Munich, Germany, 2016",
   * formula (2.26).
   *
   * Rearranging the equation above results in the actual distributed moment we want to visualize:
   * \f[
   *    \boldsymbol{m} = \boldsymbol{r}_2^{\shortmid T} \cross \boldsymbol{m}_{\text{pseudo}}
   * \f]
   * or in direct notation => m = r_xi x m_pseudo)
   */

  Core::LinAlg::Matrix<3, 1, double> m_pseudo(Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<3, 1, double> m(Core::LinAlg::Initialization::zero);

  // distributed moment on source
  m_pseudo.update(Core::FADUtils::cast_to_double(potential_reduction_factor_GP) *
                      Core::FADUtils::cast_to_double(pot_ia_deriv_cos_alpha),
      Core::FADUtils::cast_to_double<T, 3, 1>(cos_alpha_deriv_r_xi_source));

  m.cross_product(Core::FADUtils::cast_to_double<T, 3, 1>(r_xi_source), m_pseudo);

  vtk_moment_pot_source_GP.update(prefactor_visualization_data, m, 1.0);


  // distributed moment on target
  // if parameter is true, we do not evaluate this contribution
  if (!params()->potential_reduction_endpoint_moment_compensation)
  {
    m_pseudo.update_t(Core::FADUtils::cast_to_double(pot_red_fac_deriv_xi_target) *
                          Core::FADUtils::cast_to_double(interaction_potential_GP),
        Core::FADUtils::cast_to_double<T, 1, 3>(xi_target_partial_r_xi_target));
  }

  m_pseudo.update(Core::FADUtils::cast_to_double(potential_reduction_factor_GP) *
                      Core::FADUtils::cast_to_double(pot_ia_deriv_cos_alpha),
      Core::FADUtils::cast_to_double<T, 3, 1>(cos_alpha_deriv_r_xi_target), 1.0);

  m.cross_product(Core::FADUtils::cast_to_double<T, 3, 1>(r_xi_target), m_pseudo);

  vtk_moment_pot_target_GP.update(prefactor_visualization_data, m, 1.0);


  // now apply scalar integration factor
  // (after computing and storing distributed force/moment for vtk output)
  interaction_potential_GP *= rho1rho2_JacFac_GaussWeight;

  pot_ia_deriv_gap_ul *= rho1rho2_JacFac_GaussWeight;
  pot_ia_deriv_cos_alpha *= rho1rho2_JacFac_GaussWeight;

  pot_ia_2ndderiv_gap_ul *= rho1rho2_JacFac_GaussWeight;
  pot_ia_deriv_gap_ul_deriv_cos_alpha *= rho1rho2_JacFac_GaussWeight;
  pot_ia_2ndderiv_cos_alpha *= rho1rho2_JacFac_GaussWeight;


  //********************************************************************
  // calculate force/residual vector of source and target element
  //********************************************************************
  force_pot_source_GP.clear();
  force_pot_target_GP.clear();

  // sum up the contributions of all nodes (in all dimensions)
  for (unsigned int idofperdim = 0; idofperdim < (numnodes * numnodalvalues); ++idofperdim)
  {
    // loop over dimensions
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
    {
      // source beam => axial pull-off force
      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_source(idofperdim) * pot_red_fac_deriv_xi_target * interaction_potential_GP *
          xi_target_partial_r_source(jdim);

      // source beam => radial (attractive / repulsive) force
      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_source(idofperdim) * potential_reduction_factor_GP * pot_ia_deriv_gap_ul *
          gap_ul_deriv_r_source(jdim);

      // source beam => force arising from not parallel beams
      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_source(idofperdim) * potential_reduction_factor_GP * pot_ia_deriv_cos_alpha *
          cos_alpha_deriv_r_source(jdim);

      // source beam => moment arising from not parallel beams
      force_pot_source_GP(3 * idofperdim + jdim) +=
          N_i_xi_source(idofperdim) * potential_reduction_factor_GP * pot_ia_deriv_cos_alpha *
          cos_alpha_deriv_r_xi_source(jdim);

      // target beam => axial pull-off force
      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_target(idofperdim) * pot_red_fac_deriv_xi_target * interaction_potential_GP *
          xi_target_partial_r_target(jdim);

      // target beam => end point moment
      // if compensation is on, we do not evaluate this contribution
      if (!params()->potential_reduction_endpoint_moment_compensation)
      {
        force_pot_target_GP(3 * idofperdim + jdim) +=
            N_i_xi_target(idofperdim) * pot_red_fac_deriv_xi_target * interaction_potential_GP *
            xi_target_partial_r_xi_target(jdim);
      }

      // target beam => radial (attractive / repulsive) force
      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_target(idofperdim) * potential_reduction_factor_GP * pot_ia_deriv_gap_ul *
          gap_ul_deriv_r_target(jdim);

      // target beam => force arising from not parallel beams
      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_target(idofperdim) * potential_reduction_factor_GP * pot_ia_deriv_cos_alpha *
          cos_alpha_deriv_r_target(jdim);

      // target beam => moment due to not parallel beams
      force_pot_target_GP(3 * idofperdim + jdim) +=
          N_i_xi_target(idofperdim) * potential_reduction_factor_GP * pot_ia_deriv_cos_alpha *
          cos_alpha_deriv_r_xi_target(jdim);
    }
  }

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    // evaluate contributions to linearization based on analytical expression
    evaluate_stiffpot_analytic_contributions_single_length_specific_small_sep_approx_simple(
        N_i_source, N_i_xi_source, N_i_target, N_i_xi_target, N_i_xixi_target, xi_target,
        r_xi_source, r_xi_target, r_xixi_target, norm_dist_ul, normal_ul,
        potential_reduction_factor_GP, pot_red_fac_deriv_xi_target, pot_red_fac_2ndderiv_xi_target,
        interaction_potential_GP, pot_ia_deriv_gap_ul, pot_ia_deriv_cos_alpha,
        pot_ia_2ndderiv_gap_ul, pot_ia_deriv_gap_ul_deriv_cos_alpha, pot_ia_2ndderiv_cos_alpha,
        gap_ul_deriv_r_source, gap_ul_deriv_r_target, cos_alpha_deriv_r_source,
        cos_alpha_deriv_r_target, cos_alpha_deriv_r_xi_source, cos_alpha_deriv_r_xi_target,
        xi_target_partial_r_source, xi_target_partial_r_target, xi_target_partial_r_xi_target,
        *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }

  // add potential reduction factor to interaction potential for correct energy output
  interaction_potential_GP *= potential_reduction_factor_GP;

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::scale_stiffpot_analytic_contributions_if_required(double const& scalefactor,
    Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
    Core::LinAlg::SerialDenseMatrix& stiffmat21, Core::LinAlg::SerialDenseMatrix& stiffmat22) const
{
  stiffmat11.scale(scalefactor);
  stiffmat12.scale(scalefactor);
  stiffmat21.scale(scalefactor);
  stiffmat22.scale(scalefactor);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    calc_stiffmat_automatic_differentiation_if_required(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot2,
        Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
        Core::LinAlg::SerialDenseMatrix& stiffmat21,
        Core::LinAlg::SerialDenseMatrix& stiffmat22) const
{
  for (unsigned int idof = 0; idof < 3 * numnodes * numnodalvalues; ++idof)
  {
    for (unsigned int jdof = 0; jdof < 3 * numnodes * numnodalvalues; ++jdof)
    {
      // d (Res_1) / d (d_1)
      stiffmat11(idof, jdof) += force_pot1(idof).dx(jdof);

      // d (Res_1) / d (d_2)
      stiffmat12(idof, jdof) += force_pot1(idof).dx(3 * numnodes * numnodalvalues + jdof);

      // d (Res_2) / d (d_1)
      stiffmat21(idof, jdof) += force_pot2(idof).dx(jdof);

      // d (Res_2) / d (d_2)
      stiffmat22(idof, jdof) += force_pot2(idof).dx(3 * numnodes * numnodalvalues + jdof);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    add_stiffmat_contributions_xi_target_automatic_differentiation_if_required(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot2,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_source_targetDofs,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_target_targetDofs,
        Core::LinAlg::SerialDenseMatrix& stiffmat11, Core::LinAlg::SerialDenseMatrix& stiffmat12,
        Core::LinAlg::SerialDenseMatrix& stiffmat21,
        Core::LinAlg::SerialDenseMatrix& stiffmat22) const
{
  const unsigned int dim = 3 * numnodes * numnodalvalues;

  for (unsigned int idof = 0; idof < dim; ++idof)
  {
    for (unsigned int jdof = 0; jdof < dim; ++jdof)
    {
      // d (Res_1) / d (d_1)
      stiffmat11(idof, jdof) += force_pot1(idof).dx(2 * dim) *
                                Core::FADUtils::cast_to_double(lin_xi_source_targetDofs(jdof));

      // d (Res_1) / d (d_2)
      stiffmat12(idof, jdof) += force_pot1(idof).dx(2 * dim) *
                                Core::FADUtils::cast_to_double(lin_xi_target_targetDofs(jdof));

      // d (Res_2) / d (d_1)
      stiffmat21(idof, jdof) += force_pot2(idof).dx(2 * dim) *
                                Core::FADUtils::cast_to_double(lin_xi_source_targetDofs(jdof));

      // d (Res_2) / d (d_2)
      stiffmat22(idof, jdof) += force_pot2(idof).dx(2 * dim) *
                                Core::FADUtils::cast_to_double(lin_xi_target_targetDofs(jdof));
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    calc_fpot_gausspoint_automatic_differentiation_if_required(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, double>& force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, double>& force_pot2,
        Sacado::Fad::DFad<double> const& interaction_potential,
        Sacado::Fad::DFad<double> const& potential_reduction_factor,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_source_targetDofs,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_target_targetDofs) const
{
  const unsigned int dim = 3 * numnodalvalues * numnodes;

  for (unsigned int idof = 0; idof < dim; ++idof)
  {
    force_pot1(idof) = (potential_reduction_factor * interaction_potential).dx(idof) +
                       (potential_reduction_factor * interaction_potential).dx(2 * dim) *
                           Core::FADUtils::cast_to_double(lin_xi_source_targetDofs(idof));

    force_pot2(idof) = (potential_reduction_factor * interaction_potential).dx(dim + idof) +
                       (potential_reduction_factor * interaction_potential).dx(2 * dim) *
                           Core::FADUtils::cast_to_double(lin_xi_target_targetDofs(idof));
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    calc_fpot_gausspoint_automatic_differentiation_if_required(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            force_pot1,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            force_pot2,
        Sacado::Fad::DFad<double> const& interaction_potential,
        Sacado::Fad::DFad<double> const& potential_reduction_factor,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_source_targetDofs,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_target_targetDofs) const
{
  const unsigned int dim = 3 * numnodalvalues * numnodes;

  for (unsigned int idof = 0; idof < dim; ++idof)
  {
    force_pot1(idof) = (potential_reduction_factor * interaction_potential).dx(idof) +
                       (potential_reduction_factor * interaction_potential).dx(2 * dim) *
                           lin_xi_source_targetDofs(idof);

    force_pot2(idof) = (potential_reduction_factor * interaction_potential).dx(dim + idof) +
                       (potential_reduction_factor * interaction_potential).dx(2 * dim) *
                           lin_xi_target_targetDofs(idof);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::print(
    std::ostream& out) const
{
  check_init_setup();

  out << "\nInstance of BeamToBeamPotentialPair (EleGIDs " << element1()->id() << " & "
      << element2()->id() << "):";
  out << "\nele1 dofvec: "
      << Core::FADUtils::cast_to_double<T, 3 * numnodes * numnodalvalues, 1>(ele1pos_);
  out << "\nele2 dofvec: "
      << Core::FADUtils::cast_to_double<T, 3 * numnodes * numnodalvalues, 1>(ele2pos_);

  out << "\n";
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
template <typename T2>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::compute_centerline_position(Core::LinAlg::Matrix<3, 1, T>& r,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T2>& N_i,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T> eledofvec) const
{
  Discret::Utils::Beam::calc_interpolation<numnodes, numnodalvalues, 3, T, T2>(eledofvec, N_i, r);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
template <typename T2>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::compute_centerline_tangent(Core::LinAlg::Matrix<3, 1, T>& r_xi,
    const Core::LinAlg::Matrix<1, numnodes * numnodalvalues, T2>& N_i_xi,
    const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T> eledofvec) const
{
  Discret::Utils::Beam::calc_interpolation<numnodes, numnodalvalues, 3, T, T2>(
      eledofvec, N_i_xi, r_xi);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::reset_state(double time,
    const std::vector<double>& centerline_dofvec_ele1,
    const std::vector<double>& centerline_dofvec_ele2)
{
  time_ = time;

  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
  {
    ele1pos_(i) = centerline_dofvec_ele1[i];
    ele2pos_(i) = centerline_dofvec_ele2[i];
  }

  // reset interaction potential as well as interaction forces and moments of this pair
  interaction_potential_ = 0.0;

  // clear visualization data vectors (not just their elements) to avoid accumulation across passes
  centerline_coords_gp_1_.clear();
  centerline_coords_gp_2_.clear();
  forces_pot_gp_1_.clear();
  forces_pot_gp_2_.clear();
  moments_pot_gp_1_.clear();
  moments_pot_gp_2_.clear();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    set_automatic_differentiation_variables_if_required(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele1centerlinedofvec,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele2centerlinedofvec)
{
  // The 2*3*numnodes*numnodalvalues primary DoFs consist of all nodal positions and tangents
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele1centerlinedofvec(i).diff(i, 2 * 3 * numnodes * numnodalvalues);

  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele2centerlinedofvec(i).diff(
        3 * numnodes * numnodalvalues + i, 2 * 3 * numnodes * numnodalvalues);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    set_automatic_differentiation_variables_if_required(
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele1centerlinedofvec,
        Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele2centerlinedofvec,
        Sacado::Fad::DFad<double>& xi_target)
{
  // The 2*3*numnodes*numnodalvalues primary DoFs consist of all nodal positions and tangents
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele1centerlinedofvec(i).diff(i, 2 * 3 * numnodes * numnodalvalues + 1);

  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele2centerlinedofvec(i).diff(
        3 * numnodes * numnodalvalues + i, 2 * 3 * numnodes * numnodalvalues + 1);

  // Additionally, we set the parameter coordinate on the target side xi_target as primary Dof
  xi_target.diff(2 * 3 * numnodes * numnodalvalues, 2 * 3 * numnodes * numnodalvalues + 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::are_elements_much_more_separated_than_cutoff_distance()
{
  const unsigned int num_spatial_dim = 3;

  // multiple of the cutoff radius to be considered as "far enough" apart
  const double safety_factor = 2.0;

  // extract the current position of both elements' boundary nodes
  Core::LinAlg::Matrix<num_spatial_dim, 1> position_element1node1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<num_spatial_dim, 1> position_element1node2(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<num_spatial_dim, 1> position_element2node1(
      Core::LinAlg::Initialization::zero);
  Core::LinAlg::Matrix<num_spatial_dim, 1> position_element2node2(
      Core::LinAlg::Initialization::zero);

  for (unsigned int i_dim = 0; i_dim < num_spatial_dim; ++i_dim)
  {
    position_element1node1(i_dim) = Core::FADUtils::cast_to_double(ele1pos_(0 + i_dim));
    position_element1node2(i_dim) =
        Core::FADUtils::cast_to_double(ele1pos_(num_spatial_dim * 1 * numnodalvalues + i_dim));

    position_element2node1(i_dim) = Core::FADUtils::cast_to_double(ele2pos_(0 + i_dim));
    position_element2node2(i_dim) =
        Core::FADUtils::cast_to_double(ele2pos_(num_spatial_dim * 1 * numnodalvalues + i_dim));
  }


  // the following rough estimate of the elements' separation is based on spherical bounding boxes
  Core::LinAlg::Matrix<num_spatial_dim, 1> spherical_boxes_midpoint_distance(
      Core::LinAlg::Initialization::zero);

  spherical_boxes_midpoint_distance.update(
      0.5, position_element1node1, 0.5, position_element1node2);

  spherical_boxes_midpoint_distance.update(
      -0.5, position_element2node1, -0.5, position_element2node2, 1.0);


  Core::LinAlg::Matrix<num_spatial_dim, 1> element1_boundary_nodes_distance(
      Core::LinAlg::Initialization::zero);
  element1_boundary_nodes_distance.update(
      1.0, position_element1node1, -1.0, position_element1node2);
  double element1_spherical_box_radius = 0.5 * element1_boundary_nodes_distance.norm2();

  Core::LinAlg::Matrix<num_spatial_dim, 1> element2_boundary_nodes_distance(
      Core::LinAlg::Initialization::zero);
  element2_boundary_nodes_distance.update(
      1.0, position_element2node1, -1.0, position_element2node2);
  double element2_spherical_box_radius = 0.5 * element2_boundary_nodes_distance.norm2();

  double estimated_minimal_centerline_separation = spherical_boxes_midpoint_distance.norm2() -
                                                   element1_spherical_box_radius -
                                                   element2_spherical_box_radius;


  return (
      estimated_minimal_centerline_separation > safety_factor * params()->cutoff_radius.value());
}

template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BeamInteraction::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::exchange_source_target_allocation_for_two_half_pass()
{
  // interchange elements (also swap base class pointers!)
  Core::Elements::Element const* tmp_ele = element1();
  set_element1(element2());
  set_element2(tmp_ele);

  beam_element1_ = dynamic_cast<const Discret::Elements::Beam3Base*>(element1());
  beam_element2_ = dynamic_cast<const Discret::Elements::Beam3Base*>(element2());

  // swap geometric data
  std::swap(ele1length_, ele2length_);
  std::swap(radius1_, radius2_);
  std::swap(ele1pos_, ele2pos_);

  // swap line charge conditions
  std::swap(linechargeconds_[0], linechargeconds_[1]);

  // swap visualization output storage
  std::swap(centerline_coords_gp_1_, centerline_coords_gp_2_);
  std::swap(forces_pot_gp_1_, forces_pot_gp_2_);
  std::swap(moments_pot_gp_1_, moments_pot_gp_2_);
}

// explicit template instantiations
template class BeamInteraction::BeamToBeamPotentialPair<2, 1, double>;
template class BeamInteraction::BeamToBeamPotentialPair<2, 1, Sacado::Fad::DFad<double>>;
template class BeamInteraction::BeamToBeamPotentialPair<3, 1, double>;
template class BeamInteraction::BeamToBeamPotentialPair<3, 1, Sacado::Fad::DFad<double>>;
template class BeamInteraction::BeamToBeamPotentialPair<4, 1, double>;
template class BeamInteraction::BeamToBeamPotentialPair<4, 1, Sacado::Fad::DFad<double>>;
template class BeamInteraction::BeamToBeamPotentialPair<5, 1, double>;
template class BeamInteraction::BeamToBeamPotentialPair<5, 1, Sacado::Fad::DFad<double>>;
template class BeamInteraction::BeamToBeamPotentialPair<2, 2, double>;
template class BeamInteraction::BeamToBeamPotentialPair<2, 2, Sacado::Fad::DFad<double>>;

FOUR_C_NAMESPACE_CLOSE
