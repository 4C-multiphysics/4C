/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief One beam-to-beam potential-based interacting pair (two beam elements)

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/

#include "4C_beaminteraction_beam_to_beam_potential_pair.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_spatial_discretization_utils.hpp"
#include "4C_beaminteraction_geometry_utils.hpp"
#include "4C_beaminteraction_potential_params.hpp"
#include "4C_discretization_fem_general_largerotations.hpp"
#include "4C_discretization_fem_general_utils_integration.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_beampotential.hpp"
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
BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::BeamToBeamPotentialPair()
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
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::Setup()
{
  CheckInit();

  // call setup of base class first
  BeamPotentialPair::Setup();


  ele1pos_.Clear();
  ele2pos_.Clear();

  /* take care of how to assign the role of master and slave (only applicable to
   * "SingleLengthSpecific" approach). Immediately before this Setup(), the element with smaller GID
   * has been assigned as element1_, i.e., slave. */
  if (Params()->ChoiceMasterSlave() ==
      INPAR::BEAMPOTENTIAL::MasterSlaveChoice::higher_eleGID_is_slave)
  {
    // interchange order, i.e., role of elements
    DRT::Element const* tmp_ele_ptr = Element1();
    SetElement1(Element2());
    SetElement2(tmp_ele_ptr);
  }

  // get initial length of beam elements
  beam_element1_ = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(Element1());
  ele1length_ = BeamElement1()->RefLength();
  beam_element2_ = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(Element2());
  ele2length_ = BeamElement2()->RefLength();

  radius1_ = BeamElement1()->GetCircularCrossSectionRadiusForInteractions();
  radius2_ = BeamElement2()->GetCircularCrossSectionRadiusForInteractions();

  if (Element1()->ElementType() != Element2()->ElementType())
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
bool BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::Evaluate(
    CORE::LINALG::SerialDenseVector* forcevec1, CORE::LINALG::SerialDenseVector* forcevec2,
    CORE::LINALG::SerialDenseMatrix* stiffmat11, CORE::LINALG::SerialDenseMatrix* stiffmat12,
    CORE::LINALG::SerialDenseMatrix* stiffmat21, CORE::LINALG::SerialDenseMatrix* stiffmat22,
    const std::vector<DRT::Condition*> linechargeconds, const double k, const double m)
{
  // no need to evaluate this pair in case of separation by far larger than cutoff or prefactor zero
  if ((Params()->CutoffRadius() != -1.0 and AreElementsMuchMoreSeparatedThanCutoffDistance()) or
      k == 0.0)
    return false;


  // set class variables
  if (linechargeconds.size() == 2)
  {
    for (unsigned int i = 0; i < 2; ++i)
    {
      if (linechargeconds[i]->Type() == DRT::Condition::BeamPotential_LineChargeDensity)
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

  CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot1(true);
  CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot2(true);


  if (stiffmat11 != nullptr) stiffmat11->shape(dim1, dim1);
  if (stiffmat12 != nullptr) stiffmat12->shape(dim1, dim2);
  if (stiffmat21 != nullptr) stiffmat21->shape(dim2, dim1);
  if (stiffmat22 != nullptr) stiffmat22->shape(dim2, dim2);


  // compute the values for element residual vectors ('force') and linearizations ('stiff')
  switch (Params()->Strategy())
  {
    case INPAR::BEAMPOTENTIAL::strategy_doublelengthspec_largesepapprox:
    {
      EvaluateFpotandStiffpot_LargeSepApprox(
          force_pot1, force_pot2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);
      break;
    }

    case INPAR::BEAMPOTENTIAL::strategy_doublelengthspec_smallsepapprox:
    {
      EvaluateFpotandStiffpot_DoubleLengthSpecific_SmallSepApprox(
          force_pot1, force_pot2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);
      break;
    }

    case INPAR::BEAMPOTENTIAL::strategy_singlelengthspec_smallsepapprox:
    case INPAR::BEAMPOTENTIAL::strategy_singlelengthspec_smallsepapprox_simple:
    {
      EvaluateFpotandStiffpot_SingleLengthSpecific_SmallSepApprox(
          force_pot1, force_pot2, stiffmat11, stiffmat12, stiffmat21, stiffmat22);
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
      (*forcevec1)(i) = CORE::FADUTILS::CastToDouble(force_pot1(i));
  }
  if (forcevec2 != nullptr)
  {
    forcevec2->size(dim2);
    for (unsigned int i = 0; i < dim2; ++i)
      (*forcevec2)(i) = CORE::FADUTILS::CastToDouble(force_pot2(i));
  }

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::EvaluateFpotandStiffpot_LargeSepApprox(CORE::LINALG::Matrix<3 * numnodes * numnodalvalues,
                                                   1, T>& force_pot1,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot2,
    CORE::LINALG::SerialDenseMatrix* stiffmat11, CORE::LINALG::SerialDenseMatrix* stiffmat12,
    CORE::LINALG::SerialDenseMatrix* stiffmat21, CORE::LINALG::SerialDenseMatrix* stiffmat22)
{
  // prepare differentiation via FAD if desired
  SetAutomaticDifferentiationVariablesIfRequired(ele1pos_, ele2pos_);

  // get cutoff radius
  const double cutoff_radius = Params()->CutoffRadius();

  // number of integration segments per element
  const unsigned int num_integration_segments = Params()->NumberIntegrationSegments();

  // Set Gauss integration rule applied in each integration segment
  CORE::FE::GaussRule1D gaussrule = GetGaussRule();

  // Get Gauss points (gp) for integration
  CORE::FE::IntegrationPoints1D gausspoints(gaussrule);
  // number of Gauss points per integration segment and in total per element
  int numgp_persegment = gausspoints.nquad;
  int numgp_perelement = num_integration_segments * numgp_persegment;

  // vectors for shape functions
  // Attention: these are individual shape function values, NOT shape function matrices
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>> N1_i(numgp_persegment);
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>> N2_i(numgp_persegment);

  // Evaluate shape functions at gauss points and store values
  // Todo think about pre-computing and storing values for inner Gauss point loops here

  // coords of the two gauss points
  CORE::LINALG::Matrix<3, 1, T> r1(true);    // = r1
  CORE::LINALG::Matrix<3, 1, T> r2(true);    // = r2
  CORE::LINALG::Matrix<3, 1, T> dist(true);  // = r1-r2
  T norm_dist = 0.0;                         // = |r1-r2|

  // evaluate charge densities from DLINE charge condition specified in input file
  double q1 = linechargeconds_[0]->Get<double>("val");
  double q2 = linechargeconds_[1]->Get<double>("val");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  int function_number = linechargeconds_[0]->Get<int>("funct");

  if (function_number != -1)
    q1 *= GLOBAL::Problem::Instance()
              ->FunctionById<CORE::UTILS::FunctionOfTime>(function_number - 1)
              .Evaluate(time_);

  function_number = linechargeconds_[1]->Get<int>("funct");

  if (function_number != -1)
    q2 *= GLOBAL::Problem::Instance()
              ->FunctionById<CORE::UTILS::FunctionOfTime>(function_number - 1)
              .Evaluate(time_);


  // auxiliary variable
  CORE::LINALG::Matrix<3, 1, T> fpot_tmp(true);

  // determine prefactor of the integral (depends on whether surface or volume potential is applied)
  double prefactor = k_ * m_;

  switch (Params()->PotentialType())
  {
    case INPAR::BEAMPOTENTIAL::beampot_surf:
      prefactor *= 4 * radius1_ * radius2_ * M_PI * M_PI;
      break;
    case INPAR::BEAMPOTENTIAL::beampot_vol:
      prefactor *= radius1_ * radius1_ * radius2_ * radius2_ * M_PI * M_PI;
      break;
    default:
      FOUR_C_THROW(
          "No valid BEAMPOTENTIAL_TYPE specified. Choose either Surface or Volume in input file!");
  }

  // prepare data storage for visualization
  centerline_coords_gp_1_.resize(numgp_perelement);
  centerline_coords_gp_2_.resize(numgp_perelement);
  forces_pot_gp_1_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));
  forces_pot_gp_2_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));
  moments_pot_gp_1_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));
  moments_pot_gp_2_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));

  for (unsigned int isegment1 = 0; isegment1 < num_integration_segments; ++isegment1)
  {
    // compute element parameter coordinate for lower and upper limit of current integration segment
    double integration_segment1_lower_limit = -1.0 + isegment1 * 2.0 / num_integration_segments;
    double integration_segment1_upper_limit =
        -1.0 + (isegment1 + 1) * 2.0 / num_integration_segments;

    double jacobifactor_segment1 =
        0.5 * (integration_segment1_upper_limit - integration_segment1_lower_limit);

    DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<numnodes, numnodalvalues>(gausspoints, N1_i,
        BeamElement1()->Shape(), ele1length_, integration_segment1_lower_limit,
        integration_segment1_upper_limit);

    for (unsigned int isegment2 = 0; isegment2 < num_integration_segments; ++isegment2)
    {
      // compute element parameter coordinate for lower and upper limit of current integration
      // segment
      double integration_segment2_lower_limit = -1.0 + isegment2 * 2.0 / num_integration_segments;
      double integration_segment2_upper_limit =
          -1.0 + (isegment2 + 1) * 2.0 / num_integration_segments;

      double jacobifactor_segment2 =
          0.5 * (integration_segment2_upper_limit - integration_segment2_lower_limit);

      DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<numnodes, numnodalvalues>(gausspoints, N2_i,
          BeamElement2()->Shape(), ele2length_, integration_segment2_lower_limit,
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

        ComputeCenterlinePosition(r1, N1_i[igp1], ele1pos_);

        // store for visualization
        centerline_coords_gp_1_[igp1_total] = CORE::FADUTILS::CastToDouble<T, 3, 1>(r1);

        double jacobifac1 = BeamElement1()->GetJacobiFacAtXi(xi_GP1);

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
          ComputeCenterlinePosition(r2, N2_i[igp2], ele2pos_);

          // store for visualization
          centerline_coords_gp_2_[igp2_total] = CORE::FADUTILS::CastToDouble<T, 3, 1>(r2);

          double jacobifac2 = BeamElement2()->GetJacobiFacAtXi(xi_GP2);

          dist = CORE::FADUTILS::DiffVector(r1, r2);

          norm_dist = CORE::FADUTILS::VectorNorm(dist);

          // check cutoff criterion: if specified, contributions are neglected at larger separation
          if (cutoff_radius != -1.0 and CORE::FADUTILS::CastToDouble(norm_dist) > cutoff_radius)
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
            EvaluateStiffpotAnalyticContributions_LargeSepApprox(dist, norm_dist, norm_dist_exp1,
                q1q2_JacFac_GaussWeights, N1_i[igp1], N2_i[igp2], *stiffmat11, *stiffmat12,
                *stiffmat21, *stiffmat22);
          }

          // store for visualization
          forces_pot_gp_1_[igp1_total].Update(
              1.0 * prefactor * q1 * q2 * CORE::FADUTILS::CastToDouble(norm_dist_exp1) *
                  jacobifac2 * jacobifactor_segment2 * gausspoints.qwgt[igp2],
              CORE::FADUTILS::CastToDouble<T, 3, 1>(dist), 1.0);
          forces_pot_gp_2_[igp2_total].Update(
              -1.0 * prefactor * q1 * q2 * CORE::FADUTILS::CastToDouble(norm_dist_exp1) *
                  jacobifac1 * jacobifactor_segment1 * gausspoints.qwgt[igp1],
              CORE::FADUTILS::CastToDouble<T, 3, 1>(dist), 1.0);

          // store for energy output
          interaction_potential_ += prefactor / m_ * q1q2_JacFac_GaussWeights *
                                    std::pow(CORE::FADUTILS::CastToDouble(norm_dist), -m_);

        }  // end gauss quadrature loop (element 2)
      }    // end gauss quadrature loop (element 1)

    }  // end: loop over integration segments of element 2
  }    // end: loop over integration segments of element 1

  // apply constant prefactor
  force_pot1.Scale(prefactor);
  force_pot2.Scale(prefactor);

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    ScaleStiffpotAnalyticContributionsIfRequired(
        prefactor, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);

    CalcStiffmatAutomaticDifferentiationIfRequired(
        force_pot1, force_pot2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::EvaluateStiffpotAnalyticContributions_LargeSepApprox(CORE::LINALG::Matrix<3, 1,
                                                                 double> const& dist,
    double const& norm_dist, double const& norm_dist_exp1, double q1q2_JacFac_GaussWeights,
    CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N1_i_GP1,
    CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N2_i_GP2,
    CORE::LINALG::SerialDenseMatrix& stiffmat11, CORE::LINALG::SerialDenseMatrix& stiffmat12,
    CORE::LINALG::SerialDenseMatrix& stiffmat21, CORE::LINALG::SerialDenseMatrix& stiffmat22) const
{
  //********************************************************************
  // calculate stiffpot1
  //********************************************************************
  // auxiliary variables (same for both elements)
  double norm_dist_exp2 = (m_ + 2) * std::pow(norm_dist, -m_ - 4);

  CORE::LINALG::Matrix<3, 3, double> dist_dist_T(true);

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
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    EvaluateFpotandStiffpot_DoubleLengthSpecific_SmallSepApprox(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot1,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot2,
        CORE::LINALG::SerialDenseMatrix* stiffmat11, CORE::LINALG::SerialDenseMatrix* stiffmat12,
        CORE::LINALG::SerialDenseMatrix* stiffmat21, CORE::LINALG::SerialDenseMatrix* stiffmat22)
{
  // safety check
  if (m_ < 3.5)
    FOUR_C_THROW(
        "This strategy to evaluate the interaction potential is not applicable for exponents "
        "of the point potential law smaller than 3.5!");

  // prepare differentiation via FAD if desired
  SetAutomaticDifferentiationVariablesIfRequired(ele1pos_, ele2pos_);

  // get cutoff radius
  const double cutoff_radius = Params()->CutoffRadius();

  // get regularization type and separation
  const INPAR::BEAMPOTENTIAL::BeamPotentialRegularizationType regularization_type =
      Params()->RegularizationType();

  const double regularization_separation = Params()->RegularizationSeparation();

  // number of integration segments per element
  const unsigned int num_integration_segments = Params()->NumberIntegrationSegments();

  // Set Gauss integration rule applied in each integration segment
  CORE::FE::GaussRule1D gaussrule = GetGaussRule();

  // Get Gauss points (gp) for integration
  CORE::FE::IntegrationPoints1D gausspoints(gaussrule);
  // number of Gauss points per integration segment and in total per element
  int numgp_persegment = gausspoints.nquad;
  int numgp_perelement = num_integration_segments * numgp_persegment;

  // vectors for shape function values
  // Attention: these are individual shape function values, NOT shape function matrices
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>> N1_i(numgp_persegment);
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>> N2_i(numgp_persegment);

  // Evaluate shape functions at gauss points and store values
  // Todo think about pre-computing and storing values for inner Gauss point loops here

  // coords of the two gauss points
  CORE::LINALG::Matrix<3, 1, T> r1(true);    // = r1
  CORE::LINALG::Matrix<3, 1, T> r2(true);    // = r2
  CORE::LINALG::Matrix<3, 1, T> dist(true);  // = r1-r2
  T norm_dist = 0.0;                         // = |r1-r2|
  T gap = 0.0;                               // = |r1-r2|-R1-R2
  T gap_regularized = 0.0;  // modified gap if a regularization of the force law is applied

  // evaluate charge/particle densities from DLINE charge condition specified in input file
  double q1 = linechargeconds_[0]->Get<double>("val");
  double q2 = linechargeconds_[1]->Get<double>("val");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  int function_number = linechargeconds_[0]->Get<int>("funct");

  if (function_number != -1)
    q1 *= GLOBAL::Problem::Instance()
              ->FunctionById<CORE::UTILS::FunctionOfTime>(function_number - 1)
              .Evaluate(time_);

  function_number = linechargeconds_[1]->Get<int>("funct");

  if (function_number != -1)
    q2 *= GLOBAL::Problem::Instance()
              ->FunctionById<CORE::UTILS::FunctionOfTime>(function_number - 1)
              .Evaluate(time_);


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
  forces_pot_gp_1_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));
  forces_pot_gp_2_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));
  moments_pot_gp_1_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));
  moments_pot_gp_2_.resize(numgp_perelement, CORE::LINALG::Matrix<3, 1, double>(true));

  // auxiliary variables
  CORE::LINALG::Matrix<3, 1, T> fpot_tmp(true);

  double prefactor = k_ * 2 * M_PI * (m_ - 3.5) / (m_ - 2) / (m_ - 2) *
                     std::sqrt(2 * radius1_ * radius2_ / (radius1_ + radius2_)) * C;

  for (unsigned int isegment1 = 0; isegment1 < num_integration_segments; ++isegment1)
  {
    // compute element parameter coordinate for lower and upper limit of current integration segment
    double integration_segment1_lower_limit = -1.0 + isegment1 * 2.0 / num_integration_segments;
    double integration_segment1_upper_limit =
        -1.0 + (isegment1 + 1) * 2.0 / num_integration_segments;

    double jacobifactor_segment1 =
        0.5 * (integration_segment1_upper_limit - integration_segment1_lower_limit);

    DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<numnodes, numnodalvalues>(gausspoints, N1_i,
        BeamElement1()->Shape(), ele1length_, integration_segment1_lower_limit,
        integration_segment1_upper_limit);

    for (unsigned int isegment2 = 0; isegment2 < num_integration_segments; ++isegment2)
    {
      // compute element parameter coordinate for lower and upper limit of current integration
      // segment
      double integration_segment2_lower_limit = -1.0 + isegment2 * 2.0 / num_integration_segments;
      double integration_segment2_upper_limit =
          -1.0 + (isegment2 + 1) * 2.0 / num_integration_segments;

      double jacobifactor_segment2 =
          0.5 * (integration_segment2_upper_limit - integration_segment2_lower_limit);

      DRT::UTILS::BEAM::EvaluateShapeFunctionsAllGPs<numnodes, numnodalvalues>(gausspoints, N2_i,
          BeamElement2()->Shape(), ele2length_, integration_segment2_lower_limit,
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
        ComputeCenterlinePosition(r1, N1_i[igp1], ele1pos_);

        // store for visualization
        centerline_coords_gp_1_[igp1_total] = CORE::FADUTILS::CastToDouble<T, 3, 1>(r1);

        double jacobifac1 = BeamElement1()->GetJacobiFacAtXi(xi_GP1);

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
          ComputeCenterlinePosition(r2, N2_i[igp2], ele2pos_);

          // store for visualization
          centerline_coords_gp_2_[igp2_total] = CORE::FADUTILS::CastToDouble<T, 3, 1>(r2);

          double jacobifac2 = BeamElement2()->GetJacobiFacAtXi(xi_GP2);

          dist = CORE::FADUTILS::DiffVector(r1, r2);

          norm_dist = CORE::FADUTILS::VectorNorm(dist);

          if (norm_dist == 0.0)
          {
            this->Print(std::cout);
            std::cout << "\nGP pair: igp1_total=" << igp1_total << " & igp2_total=" << igp2_total
                      << ": |r1-r2|=" << norm_dist;

            FOUR_C_THROW("centerline separation |r1-r2|=0! Fatal error.");
          }

          // check cutoff criterion: if specified, contributions are neglected at larger separation
          if (cutoff_radius != -1.0 and CORE::FADUTILS::CastToDouble(norm_dist) > cutoff_radius)
            continue;

          gap = norm_dist - radius1_ - radius2_;



          if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_none and gap <= 0.0)
          {
            this->Print(std::cout);
            std::cout << "\nGP pair: igp1_total=" << igp1_total << " & igp2_total=" << igp2_total
                      << ": gap=" << gap;

            FOUR_C_THROW(
                "gap<=0! Force law resulting from specified interaction potential law is "
                "not defined for zero/negative gaps! Use/implement a regularization!");
          }

          gap_regularized = gap;

          if ((regularization_type == INPAR::BEAMPOTENTIAL::regularization_constant or
                  regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear) and
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
              std::pow(CORE::FADUTILS::CastToDouble(gap_regularized), -m_ + 3.5);

          if ((regularization_type == INPAR::BEAMPOTENTIAL::regularization_constant or
                  regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear) and
              gap < regularization_separation)
          {
            // potential law is linear in the regime of constant extrapolation of force law
            // and quadratic in case of linear extrapolation
            // add the linear contribution from this part of the force law
            interaction_potential_ +=
                prefactor * q1q2_JacFac_GaussWeights * CORE::FADUTILS::CastToDouble(gap_exp1) *
                (regularization_separation - CORE::FADUTILS::CastToDouble(gap));
          }


          if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear and
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
                CORE::FADUTILS::CastToDouble(gap_exp2) *
                (regularization_separation - CORE::FADUTILS::CastToDouble(gap)) *
                (regularization_separation - CORE::FADUTILS::CastToDouble(gap));
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
            EvaluateStiffpotAnalyticContributions_DoubleLengthSpecific_SmallSepApprox(dist,
                norm_dist, gap, gap_regularized, gap_exp1, q1q2_JacFac_GaussWeights, N1_i[igp1],
                N2_i[igp2], *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
          }

          // store for visualization
          forces_pot_gp_1_[igp1_total].Update(
              1.0 * prefactor * q1 * q2 * CORE::FADUTILS::CastToDouble(gap_exp1) /
                  CORE::FADUTILS::CastToDouble(norm_dist) * jacobifac2 * jacobifactor_segment2 *
                  gausspoints.qwgt[igp2],
              CORE::FADUTILS::CastToDouble<T, 3, 1>(dist), 1.0);
          forces_pot_gp_2_[igp2_total].Update(
              -1.0 * prefactor * q1 * q2 * CORE::FADUTILS::CastToDouble(gap_exp1) /
                  CORE::FADUTILS::CastToDouble(norm_dist) * jacobifac1 * jacobifactor_segment1 *
                  gausspoints.qwgt[igp1],
              CORE::FADUTILS::CastToDouble<T, 3, 1>(dist), 1.0);

        }  // end: loop over gauss points of element 2
      }    // end: loop over gauss points of element 1

    }  // end: loop over integration segments of element 2
  }    // end: loop over integration segments of element 1

  // apply constant prefactor
  force_pot1.Scale(prefactor);
  force_pot2.Scale(prefactor);

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    ScaleStiffpotAnalyticContributionsIfRequired(
        prefactor, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);

    CalcStiffmatAutomaticDifferentiationIfRequired(
        force_pot1, force_pot2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    EvaluateStiffpotAnalyticContributions_DoubleLengthSpecific_SmallSepApprox(
        CORE::LINALG::Matrix<3, 1, double> const& dist, double const& norm_dist, double const& gap,
        double const& gap_regularized, double const& gap_exp1, double q1q2_JacFac_GaussWeights,
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N1_i_GP1,
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N2_i_GP2,
        CORE::LINALG::SerialDenseMatrix& stiffmat11, CORE::LINALG::SerialDenseMatrix& stiffmat12,
        CORE::LINALG::SerialDenseMatrix& stiffmat21,
        CORE::LINALG::SerialDenseMatrix& stiffmat22) const
{
  //********************************************************************
  // calculate stiffpot1
  //********************************************************************
  // auxiliary variables (same for both elements)
  double gap_exp2 = std::pow(gap_regularized, -m_ + 1.5);

  if (Params()->RegularizationType() == INPAR::BEAMPOTENTIAL::regularization_constant and
      gap < Params()->RegularizationSeparation())
  {
    /* in case of constant extrapolation of force law, the derivative of the force is zero
     * and this contribution to the stiffness matrix vanishes */
    gap_exp2 = 0.0;
  }

  double aux_fac1 = gap_exp1 / norm_dist * q1q2_JacFac_GaussWeights;
  double aux_fac2 = (gap_exp1 / norm_dist / norm_dist / norm_dist +
                        (m_ - 2.5) * gap_exp2 / norm_dist / norm_dist) *
                    q1q2_JacFac_GaussWeights;

  CORE::LINALG::Matrix<3, 3, double> dist_dist_T(true);

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
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    EvaluateFpotandStiffpot_SingleLengthSpecific_SmallSepApprox(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot1,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot2,
        CORE::LINALG::SerialDenseMatrix* stiffmat11, CORE::LINALG::SerialDenseMatrix* stiffmat12,
        CORE::LINALG::SerialDenseMatrix* stiffmat21, CORE::LINALG::SerialDenseMatrix* stiffmat22)
{
  // safety checks
  if (m_ < 6.0)
    FOUR_C_THROW(
        "Invalid exponent m=%f. The strategy 'SingleLengthSpecific_SmallSepApprox' to evaluate the "
        "interaction potential is only applicable for exponents m>=6 of the point potential law, "
        "e.g. van der Waals (m=6) or the repulsive part of Lennard-Jones (m=12)!",
        m_);

  if (not Params()->UseFAD() and
      Params()->Strategy() == INPAR::BEAMPOTENTIAL::strategy_singlelengthspec_smallsepapprox)
  {
    FOUR_C_THROW(
        "The strategy 'SingleLengthSpecific_SmallSepApprox' to evaluate the interaction "
        "potential requires automatic differentiation via FAD!");
  }

  if (radius1_ != radius2_)
    FOUR_C_THROW(
        "The strategy 'SingleLengthSpecific_SmallSepApprox' to evaluate the interaction "
        "potential requires the beam radii to be identical!");


  // get cutoff radius
  const double cutoff_radius = Params()->CutoffRadius();

  // get regularization type and separation
  const INPAR::BEAMPOTENTIAL::BeamPotentialRegularizationType regularization_type =
      Params()->RegularizationType();

  const double regularization_separation = Params()->RegularizationSeparation();

  /* parameter coordinate of the closest point on the master beam,
   * determined via point-to-curve projection */
  T xi_master = 0.0;

  // prepare differentiation via FAD if desired
  /* xi_master is used as additional, auxiliary primary Dof
   * since there is no closed-form expression for how xi_master depends on the 'real' primary Dofs.
   * It is determined iteratively via point-to-curve projection */
  SetAutomaticDifferentiationVariablesIfRequired(ele1pos_, ele2pos_, xi_master);


  // number of integration segments per element
  const unsigned int num_integration_segments = Params()->NumberIntegrationSegments();

  // Set Gauss integration rule applied in each integration segment
  CORE::FE::GaussRule1D gaussrule = GetGaussRule();

  // Get Gauss points (gp) for integration
  CORE::FE::IntegrationPoints1D gausspoints(gaussrule);

  // number of Gauss points per integration segment and in total
  int numgp_persegment = gausspoints.nquad;
  int numgp_total = num_integration_segments * numgp_persegment;

  int igp_total = 0;
  double xi_GP_tilde = 0.0;
  double xi_GP = 0.0;

  double rho1rho2_JacFac_GaussWeight = 0.0;

  const unsigned int num_initial_values = 9;
  const std::array<double, num_initial_values> xi_master_initial_guess_values = {
      0.0, -1.0, 1.0, -0.5, 0.5, -0.75, -0.25, 0.25, 0.75};

  unsigned int iter_projection = 0;

  // vectors for shape functions and their derivatives
  // Attention: these are individual shape function values, NOT shape function matrices
  // values at all Gauss points are stored in advance (more efficient espec. for many segments)
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>> N_i_slave(
      numgp_persegment);  // = N1_i
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>> N_i_xi_slave(
      numgp_persegment);  // = N1_i,xi

  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> N_i_master(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> N_i_xi_master(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> N_i_xixi_master(true);

  // assembled shape function matrices: Todo maybe avoid these matrices
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, double> N_slave(true);

  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, T> N_master(true);
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, T> N_xi_master(true);
  CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, T> N_xixi_master(true);


  // coords and derivatives of the slave Gauss point and projected point on master element
  CORE::LINALG::Matrix<3, 1, T> r_slave(true);        // centerline position vector on slave
  CORE::LINALG::Matrix<3, 1, T> r_xi_slave(true);     // centerline tangent vector on slave
  CORE::LINALG::Matrix<3, 1, T> t_slave(true);        // unit centerline tangent vector on slave
  CORE::LINALG::Matrix<3, 1, T> r_master(true);       // centerline position vector on master
  CORE::LINALG::Matrix<3, 1, T> r_xi_master(true);    // centerline tangent vector on master
  CORE::LINALG::Matrix<3, 1, T> r_xixi_master(true);  // 2nd deriv of master curve
  CORE::LINALG::Matrix<3, 1, T> t_master(true);       // unit centerline tangent vector on master

  T norm_r_xi_slave = 0.0;
  T norm_r_xi_master = 0.0;

  T signum_tangentsscalarproduct = 0.0;

  CORE::LINALG::Matrix<3, 1, T> dist_ul(true);  // = r_slave-r_master
  T norm_dist_ul = 0.0;

  // unilateral normal vector
  CORE::LINALG::Matrix<3, 1, T> normal_ul(true);

  // gap of unilateral closest points
  T gap_ul = 0.0;
  // regularized gap of unilateral closest points
  T gap_ul_regularized = 0.0;

  T alpha = 0.0;      // mutual angle of tangent vectors
  T cos_alpha = 0.0;  // cosine of mutual angle of tangent vectors

  // auxiliary quantity
  T a = 0.0;

  T interaction_potential_GP = 0.0;


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


  // components from variation of unilateral gap
  CORE::LINALG::Matrix<3, 1, T> gap_ul_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 1, T> gap_ul_deriv_r_master(true);

  // components from variation of cosine of enclosed angle
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_partial_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_partial_r_xi_master(true);
  T cos_alpha_partial_xi_master = 0.0;

  CORE::LINALG::Matrix<3, 1, T> cos_alpha_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_deriv_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_deriv_r_xi_master(true);

  T a_deriv_cos_alpha = 0.0;

  // components from variation of parameter coordinate on master beam
  CORE::LINALG::Matrix<1, 3, T> xi_master_partial_r_slave(true);
  CORE::LINALG::Matrix<1, 3, T> xi_master_partial_r_master(true);
  CORE::LINALG::Matrix<1, 3, T> xi_master_partial_r_xi_master(true);


  // linearization of parameter coordinate on master resulting from point-to-curve projection
  CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, T> lin_xi_master_slaveDofs(true);
  CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, T> lin_xi_master_masterDofs(true);

  /* contribution of one Gauss point (required for automatic differentiation with contributions
   * from xi_master) */
  CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot_slave_GP(true);
  CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T> force_pot_master_GP(true);

  // Todo remove the following two variables, which are required for verification via FAD
  CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, double> force_pot_slave_GP_calc_via_FAD(
      true);
  CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, double> force_pot_master_GP_calc_via_FAD(
      true);

  // evaluate charge/particle densities from DLINE charge condition specified in input file
  double rho1 = linechargeconds_[0]->Get<double>("val");
  double rho2 = linechargeconds_[1]->Get<double>("val");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  int function_number = linechargeconds_[0]->Get<int>("funct");

  if (function_number != -1)
    rho1 *= GLOBAL::Problem::Instance()
                ->FunctionById<CORE::UTILS::FunctionOfTime>(function_number - 1)
                .Evaluate(time_);

  function_number = linechargeconds_[1]->Get<int>("funct");

  if (function_number != -1)
    rho2 *= GLOBAL::Problem::Instance()
                ->FunctionById<CORE::UTILS::FunctionOfTime>(function_number - 1)
                .Evaluate(time_);


  // constant prefactor of the disk-cylinder interaction potential
  double prefactor = k_ * M_PI * M_PI;

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
          "exponent m=%f. So far, only exponent 6 and 12 is supported.",
          m_);
      break;
  }

  // store for visualization
  double prefactor_visualization_data = -1.0 * prefactor * rho1 * rho2;

  //************************** DEBUG ******************************************
  //  this->Print(std::cout);
  //  std::cout << "\n\n\nStart evaluation via Gauss integration..." << std::endl;
  //  std::cout << "\nnumgp_total=" << numgp_total;
  //*********************** END DEBUG *****************************************


  // prepare data storage for visualization
  centerline_coords_gp_1_.resize(numgp_total);
  centerline_coords_gp_2_.resize(numgp_total);
  forces_pot_gp_1_.resize(numgp_total, CORE::LINALG::Matrix<3, 1, double>(true));
  forces_pot_gp_2_.resize(numgp_total, CORE::LINALG::Matrix<3, 1, double>(true));
  moments_pot_gp_1_.resize(numgp_total, CORE::LINALG::Matrix<3, 1, double>(true));
  moments_pot_gp_2_.resize(numgp_total, CORE::LINALG::Matrix<3, 1, double>(true));


  CORE::LINALG::Matrix<3, 1, double> moment_pot_tmp(true);
  CORE::LINALG::Matrix<3, 3, double> spin_pseudo_moment_tmp(true);


  for (unsigned int isegment = 0; isegment < num_integration_segments; ++isegment)
  {
    // compute element parameter coordinate for lower and upper limit of current integration segment
    double integration_segment_lower_limit =
        -1.0 + (double)isegment * 2.0 / (double)num_integration_segments;
    double integration_segment_upper_limit =
        -1.0 + (double)(isegment + 1) * 2.0 / (double)num_integration_segments;

    double jacobifactor_segment =
        0.5 * (integration_segment_upper_limit - integration_segment_lower_limit);

    // Evaluate shape functions at Gauss points of slave element and store values
    DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAllGPs<numnodes, numnodalvalues>(gausspoints,
        N_i_slave, N_i_xi_slave, BeamElement1()->Shape(), ele1length_,
        integration_segment_lower_limit, integration_segment_upper_limit);

    // loop over gauss points of element 1
    // so far, element 1 is always treated as the slave element!
    for (int igp = 0; igp < numgp_persegment; ++igp)
    {
      igp_total = isegment * numgp_persegment + igp;

      //************************** DEBUG ******************************************
      //      std::cout << "\nEvaluate igp_total=" << igp_total;
      //*********************** END DEBUG *****************************************


      // Get location of GP in element parameter space xi \in [-1;1]
      xi_GP_tilde = gausspoints.qxg[igp][0];

      /* do a mapping into integration segment, i.e. coordinate transformation
       * note: this has no effect if integration interval is [-1;1] */
      xi_GP = 0.5 * ((1.0 - xi_GP_tilde) * integration_segment_lower_limit +
                        (1.0 + xi_GP_tilde) * integration_segment_upper_limit);


      // compute coord vector and tangent vector on slave side
      ComputeCenterlinePosition(r_slave, N_i_slave[igp], ele1pos_);
      ComputeCenterlineTangent(r_xi_slave, N_i_xi_slave[igp], ele1pos_);

      norm_r_xi_slave = CORE::FADUTILS::VectorNorm(r_xi_slave);

      t_slave.Update(1.0 / norm_r_xi_slave, r_xi_slave);

      // store for visualization
      centerline_coords_gp_1_[igp_total] = CORE::FADUTILS::CastToDouble<T, 3, 1>(r_slave);

      rho1rho2_JacFac_GaussWeight = rho1 * rho2 * jacobifactor_segment *
                                    BeamElement1()->GetJacobiFacAtXi(xi_GP) * gausspoints.qwgt[igp];

      //************************** DEBUG ******************************************
      //      std::cout << "\n\nGP " << igp_total << ":";
      //      std::cout << "\nr_slave: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(r_slave);
      //      std::cout << "\nr_xi_slave: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(r_xi_slave);
      //      std::cout << "\n|r_xi_slave|: "
      //                << CORE::FADUTILS::VectorNorm<3>(CORE::FADUTILS::CastToDouble<T, 3,
      //                1>(r_xi_slave));
      //      std::cout << "\ng1_slave: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(g1_slave);
      //      std::cout << "\n|g1_slave|: "
      //                << CORE::FADUTILS::VectorNorm<3>(CORE::FADUTILS::CastToDouble<T, 3,
      //                1>(g1_slave));
      //*********************** END DEBUG *****************************************

      /* point-to-curve projection, i.e. 'unilateral' closest-point projection
       * to determine point on master beam (i.e. parameter coordinate xi_master) */
      iter_projection = 0;

      while (iter_projection < num_initial_values and
             (not BEAMINTERACTION::GEO::PointToCurveProjection<numnodes, numnodalvalues, T>(r_slave,
                 xi_master, xi_master_initial_guess_values[iter_projection], ele2pos_,
                 Element2()->Shape(), ele2length_)))
      {
        iter_projection++;
      }

      if (iter_projection == num_initial_values)
      {
        std::cout << "\nWARNING: Point-to-Curve Projection ultimately failed at "
                     "xi_slave="
                  << xi_GP << " of ele pair " << Element1()->Id() << " & " << Element2()->Id()
                  << "\nFallback strategy: Assume invalid projection and skip this GP..."
                  << std::endl;
        continue;
      }

      //************************** DEBUG ******************************************
      //    if ( i!=0 )
      //    {
      //      std::cout << "\n\nINFO: Point-to-Curve Projection succeeded with initial guess "
      //          << xi_master_initial_guess_values[i] << ": xi_master="
      //          << CORE::FADUTILS::CastToDouble(xi_master) << std::endl;
      //    }
      //*********************** END DEBUG *****************************************

      //************************** DEBUG ******************************************
      //    std::cout << "\nxi_master: " << CORE::FADUTILS::CastToDouble( xi_master );
      //*********************** END DEBUG *****************************************

      // Todo: specify tolerance value in a more central place
      if (CORE::FADUTILS::Norm(xi_master) > 1.0 + 1.0e-10)
      {
        //************************** DEBUG ******************************************
        //      std::cout << "\nxi_master not in valid range ... proceed to next GP\n";
        //*********************** END DEBUG *****************************************
        continue;
      }
      else if (CORE::FADUTILS::Norm(xi_master) >= 1.0 - 1.0e-10)
      {
        FOUR_C_THROW(
            "Point-to-curve projection yields xi_master= %f. This is a critical case "
            "since it is very close to the element boundary!",
            CORE::FADUTILS::CastToDouble(xi_master));
      }

      DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAnd2ndDerivsAtXi<numnodes, numnodalvalues>(
          xi_master, N_i_master, N_i_xi_master, N_i_xixi_master, BeamElement2()->Shape(),
          ele2length_);

      // compute coord vector and tangent vector on master side
      ComputeCenterlinePosition(r_master, N_i_master, ele2pos_);
      ComputeCenterlineTangent(r_xi_master, N_i_xi_master, ele2pos_);
      ComputeCenterlineTangent(r_xixi_master, N_i_xixi_master, ele2pos_);


      norm_r_xi_master = CORE::FADUTILS::VectorNorm(r_xi_master);

      t_master.Update(1.0 / norm_r_xi_master, r_xi_master);


      //************************** DEBUG ******************************************
      //      std::cout << "\nr_master: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(r_master);
      //      std::cout << "\nr_xi_master: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(r_xi_master);
      //      std::cout << "\n|r_xi_master|: "
      //                << CORE::FADUTILS::VectorNorm<3>(CORE::FADUTILS::CastToDouble<T, 3,
      //                1>(r_xi_master));
      //      std::cout << "\ng1_master: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(g1_master);
      //      std::cout << "\n|g1_master|: "
      //                << CORE::FADUTILS::VectorNorm<3>(CORE::FADUTILS::CastToDouble<T, 3,
      //                1>(g1_master));
      //*********************** END DEBUG *****************************************

      // store for visualization
      centerline_coords_gp_2_[igp_total] = CORE::FADUTILS::CastToDouble<T, 3, 1>(r_master);

      // distance vector between unilateral closest points
      dist_ul.Update(1.0, r_slave, -1.0, r_master);

      norm_dist_ul = CORE::FADUTILS::VectorNorm(dist_ul);

      if (CORE::FADUTILS::CastToDouble(norm_dist_ul) == 0.0)
      {
        this->Print(std::cout);
        FOUR_C_THROW("centerline separation |r1-r2|=0! Fatal error.");
      }

      //************************** DEBUG ******************************************
      //      std::cout << "\ndist_ul: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(dist_ul);
      //*********************** END DEBUG *****************************************

      // check cutoff criterion: if specified, contributions are neglected at larger separation
      if (cutoff_radius != -1.0 and CORE::FADUTILS::CastToDouble(norm_dist_ul) > cutoff_radius)
      {
        //************************** DEBUG ******************************************
        // std::cout << "\nINFO: Ignored GP (ele GIDs " << Element1()->Id() << "&" <<
        // Element2()->Id()
        //          << ": iGP " << igp_total
        //          << ") with |dist_ul|=" <<
        //          CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(dist_ul))
        //          << " > cutoff=" << cutoff_radius << std::endl;
        //*********************** END DEBUG *****************************************
        continue;
      }

      // mutual angle of tangent vectors at unilateral closest points
      BEAMINTERACTION::GEO::CalcEnclosedAngle(alpha, cos_alpha, r_xi_slave, r_xi_master);


      //************************** DEBUG ******************************************
      //    std::cout << "\nalpha: " << CORE::FADUTILS::CastToDouble(alpha * 180 / M_PI) <<
      //    "degrees"; std::cout << "\ncos(alpha): " << CORE::FADUTILS::CastToDouble( cos_alpha );
      //    std::cout << "\nsin(alpha): " << CORE::FADUTILS::CastToDouble( sin_alpha );
      //*********************** END DEBUG *****************************************


      //************************** DEBUG ******************************************
      if (alpha < 0.0 or alpha > M_PI_2)
        FOUR_C_THROW("alpha=%f, should be in [0,pi/2]", CORE::FADUTILS::CastToDouble(alpha));
      //*********************** END DEBUG *****************************************



      // Todo: maybe avoid this assembly of shape fcns into matrices
      // Fixme: at least, do this only in case of FAD-based linearization
      DRT::UTILS::BEAM::AssembleShapeFunctions<numnodes, numnodalvalues>(N_i_slave[igp], N_slave);

      DRT::UTILS::BEAM::AssembleShapeFunctionsAndDerivsAnd2ndDerivs<numnodes, numnodalvalues>(
          N_i_master, N_i_xi_master, N_i_xixi_master, N_master, N_xi_master, N_xixi_master);

      BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<numnodes,
          numnodalvalues>(lin_xi_master_slaveDofs, lin_xi_master_masterDofs, dist_ul, r_xi_master,
          r_xixi_master, N_slave, N_master, N_xixi_master);


      BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartialDerivs(
          xi_master_partial_r_slave, xi_master_partial_r_master, xi_master_partial_r_xi_master,
          dist_ul, r_xi_master, r_xixi_master);


      // evaluate all quantities which depend on the the applied disk-cylinder potential law

      // 'full' disk-cylinder interaction potential
      if (Params()->Strategy() == INPAR::BEAMPOTENTIAL::strategy_singlelengthspec_smallsepapprox)
      {
        if (not EvaluateFullDiskCylinderPotential(interaction_potential_GP, force_pot_slave_GP,
                force_pot_master_GP, r_slave, r_xi_slave, t_slave, r_master, r_xi_master,
                r_xixi_master, t_master, alpha, cos_alpha, dist_ul, xi_master_partial_r_slave,
                xi_master_partial_r_master, xi_master_partial_r_xi_master,
                prefactor_visualization_data, forces_pot_gp_1_[igp_total],
                forces_pot_gp_2_[igp_total], moments_pot_gp_1_[igp_total],
                moments_pot_gp_2_[igp_total], rho1rho2_JacFac_GaussWeight, N_i_slave[igp],
                N_i_xi_slave[igp], N_i_master, N_i_xi_master))
          continue;
      }
      // reduced, simpler variant of the disk-cylinder interaction potential
      else if (Params()->Strategy() ==
               INPAR::BEAMPOTENTIAL::strategy_singlelengthspec_smallsepapprox_simple)
      {
        // gap of unilateral closest points
        gap_ul = norm_dist_ul - radius1_ - radius2_;

        // regularized gap of unilateral closest points
        gap_ul_regularized = gap_ul;


        if (regularization_type != INPAR::BEAMPOTENTIAL::regularization_none and
            gap_ul < regularization_separation)
        {
          gap_ul_regularized = regularization_separation;

          //************************** DEBUG ******************************************
          //          std::cout << "\ngap_ul: " << CORE::FADUTILS::CastToDouble(gap_ul) << ":
          //          regularization active!"; std::cout << "\ngap_ul_regularized: " <<
          //          CORE::FADUTILS::CastToDouble(gap_ul_regularized);
          // this->Print(std::cout);
          // std::cout << "\nigp_total: " << igp_total;
          //*********************** END DEBUG *****************************************
        }
        else if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_none and
                 gap_ul < 1e-14)
        {
          this->Print(std::cout);

          std::cout << "\ngap_ul: " << CORE::FADUTILS::CastToDouble(gap_ul);
          std::cout << "\nalpha: " << CORE::FADUTILS::CastToDouble(alpha * 180 / M_PI) << "degrees";

          FOUR_C_THROW(
              "gap_ul=%f is negative or very close to zero! Fatal error. Use regularization to"
              " handle this!",
              CORE::FADUTILS::CastToDouble(gap_ul));
        }


        // unilateral normal vector
        normal_ul.Update(1.0 / norm_dist_ul, dist_ul);


        // auxiliary quantity
        a = 0.5 / radius1_ + 0.5 * cos_alpha * cos_alpha / radius2_;

        // safety check
        if (CORE::FADUTILS::CastToDouble(a) <= 0.0)
        {
          this->Print(std::cout);
          FOUR_C_THROW("auxiliary quantity a<=0! Fatal error.");
        }

        // interaction potential
        interaction_potential_GP =
            std::pow(4 * gap_ul_regularized, -m_ + 4.5) / CORE::FADUTILS::sqrt(a);


        // compute derivatives of the interaction potential w.r.t. gap_ul and cos(alpha)
        pot_ia_deriv_gap_ul = (-m_ + 4.5) / gap_ul_regularized * interaction_potential_GP;

        a_deriv_cos_alpha = cos_alpha / radius2_;
        pot_ia_deriv_cos_alpha = -0.5 / a * interaction_potential_GP * a_deriv_cos_alpha;


        // Todo: strictly speaking, the following second derivatives of the interaction potential
        // are only required in case of active regularization OR for analytic linearization

        pot_ia_deriv_gap_ul_deriv_cos_alpha =
            (-m_ + 4.5) / gap_ul_regularized * pot_ia_deriv_cos_alpha;

        pot_ia_2ndderiv_gap_ul = (-m_ + 3.5) / gap_ul_regularized * pot_ia_deriv_gap_ul;

        pot_ia_2ndderiv_cos_alpha =
            -0.5 / (a * radius2_) * interaction_potential_GP +
            0.75 * a_deriv_cos_alpha * a_deriv_cos_alpha / (a * a) * interaction_potential_GP;

        // third derivatives
        pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep =
            (-m_ + 4.5) * (-m_ + 3.5) / (gap_ul_regularized * gap_ul_regularized) *
            pot_ia_deriv_cos_alpha;

        pot_ia_deriv_gap_ul_2ndderiv_cos_alpha_at_regsep =
            (-m_ + 4.5) / gap_ul_regularized * pot_ia_2ndderiv_cos_alpha;

        // fourth derivative
        pot_ia_2ndderiv_gap_ul_2ndderiv_cos_alpha_at_regsep =
            (-m_ + 4.5) * (-m_ + 3.5) / (gap_ul_regularized * gap_ul_regularized) *
            pot_ia_2ndderiv_cos_alpha;

        /* now that we don't need the interaction potential at gap_ul=regularization_separation as
         * an intermediate result anymore, we can add the additional (linear and quadratic)
         * contributions in case of active regularization to the interaction potential */
        if (regularization_type != INPAR::BEAMPOTENTIAL::regularization_none and
            gap_ul < regularization_separation)
        {
          //************************** DEBUG ******************************************
          //          std::cout << "\ninteraction_potential_GP_atregsep: "
          //                    << CORE::FADUTILS::CastToDouble(interaction_potential_GP);
          //          std::cout << "\npot_ia_deriv_gap_ul_atregsep: "
          //                    << CORE::FADUTILS::CastToDouble(pot_ia_deriv_gap_ul);
          //          std::cout << "\npot_ia_deriv_cos_alpha_atregsep: "
          //                    << CORE::FADUTILS::CastToDouble(pot_ia_deriv_cos_alpha);
          // this->Print(std::cout);
          // std::cout << "\nigp_total: " << igp_total;
          //*********************** END DEBUG *****************************************

          interaction_potential_GP += pot_ia_deriv_gap_ul * (gap_ul - regularization_separation);

          pot_ia_deriv_cos_alpha +=
              pot_ia_deriv_gap_ul_deriv_cos_alpha * (gap_ul - regularization_separation);


          if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear)
          {
            interaction_potential_GP += 0.5 * pot_ia_2ndderiv_gap_ul *
                                        (gap_ul - regularization_separation) *
                                        (gap_ul - regularization_separation);

            pot_ia_deriv_gap_ul += pot_ia_2ndderiv_gap_ul * (gap_ul - regularization_separation);

            pot_ia_deriv_cos_alpha += 0.5 * pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep *
                                      (gap_ul - regularization_separation) *
                                      (gap_ul - regularization_separation);
          }

          //************************** DEBUG ******************************************
          //          std::cout << "\ninteraction_potential_GP_regularized: "
          //                    << CORE::FADUTILS::CastToDouble(interaction_potential_GP);
          //          std::cout << "\npot_ia_deriv_gap_ul_regularized: "
          //                    << CORE::FADUTILS::CastToDouble(pot_ia_deriv_gap_ul);
          //          std::cout << "\npot_ia_deriv_cos_alpha_regularized: "
          //                    << CORE::FADUTILS::CastToDouble(pot_ia_deriv_cos_alpha);
          // this->Print(std::cout);
          // std::cout << "\nigp_total: " << igp_total;
          //*********************** END DEBUG *****************************************
        }

        /* adapt also the second derivatives (required for analytic linearization) in
         * case of active regularization*/
        if (regularization_type != INPAR::BEAMPOTENTIAL::regularization_none and
            gap_ul < regularization_separation)
        {
          if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_constant)
            pot_ia_2ndderiv_gap_ul = 0.0;

          pot_ia_2ndderiv_cos_alpha += pot_ia_deriv_gap_ul_2ndderiv_cos_alpha_at_regsep *
                                       (gap_ul - regularization_separation);

          if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear)
          {
            pot_ia_deriv_gap_ul_deriv_cos_alpha += pot_ia_2ndderiv_gap_ul_deriv_cos_alpha_atregsep *
                                                   (gap_ul - regularization_separation);

            pot_ia_2ndderiv_cos_alpha += 0.5 * pot_ia_2ndderiv_gap_ul_2ndderiv_cos_alpha_at_regsep *
                                         (gap_ul - regularization_separation) *
                                         (gap_ul - regularization_separation);
          }
        }


        // compute components from variation of cosine of enclosed angle
        signum_tangentsscalarproduct = CORE::FADUTILS::Signum(r_xi_slave.Dot(r_xi_master));
        // auxiliary variables
        CORE::LINALG::Matrix<3, 3, T> v_mat_tmp(true);

        for (unsigned int i = 0; i < 3; ++i)
        {
          v_mat_tmp(i, i) += 1.0;
          for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t_slave(i) * t_slave(j);
        }

        cos_alpha_partial_r_xi_slave.Multiply(
            signum_tangentsscalarproduct / CORE::FADUTILS::VectorNorm(r_xi_slave), v_mat_tmp,
            t_master);

        v_mat_tmp.Clear();
        for (unsigned int i = 0; i < 3; ++i)
        {
          v_mat_tmp(i, i) += 1.0;
          for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t_master(i) * t_master(j);
        }

        cos_alpha_partial_r_xi_master.Multiply(
            signum_tangentsscalarproduct / CORE::FADUTILS::VectorNorm(r_xi_master), v_mat_tmp,
            t_slave);

        cos_alpha_partial_xi_master = r_xixi_master.Dot(cos_alpha_partial_r_xi_master);


        cos_alpha_deriv_r_slave.UpdateT(cos_alpha_partial_xi_master, xi_master_partial_r_slave);
        cos_alpha_deriv_r_master.UpdateT(cos_alpha_partial_xi_master, xi_master_partial_r_master);

        cos_alpha_deriv_r_xi_slave.Update(cos_alpha_partial_r_xi_slave);
        cos_alpha_deriv_r_xi_master.Update(1.0, cos_alpha_partial_r_xi_master);
        cos_alpha_deriv_r_xi_master.UpdateT(
            cos_alpha_partial_xi_master, xi_master_partial_r_xi_master, 1.0);


        // compute components from variation of the unilateral gap
        gap_ul_deriv_r_slave.Update(normal_ul);
        gap_ul_deriv_r_master.Update(-1.0, normal_ul);



        // store for visualization
        forces_pot_gp_1_[igp_total].Update(
            prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_deriv_gap_ul),
            CORE::FADUTILS::CastToDouble<T, 3, 1>(gap_ul_deriv_r_slave), 1.0);

        forces_pot_gp_1_[igp_total].Update(
            prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_deriv_cos_alpha),
            CORE::FADUTILS::CastToDouble<T, 3, 1>(cos_alpha_deriv_r_slave), 1.0);



        moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_deriv_cos_alpha),
            CORE::FADUTILS::CastToDouble<T, 3, 1>(cos_alpha_deriv_r_xi_slave));

        /* note: relation between variation of tangent vector r_xi and variation of (transversal
         * part of) rotation vector theta_perp describing cross-section orientation can be used to
         *       identify (distributed) moments as follows: m_pot = 1/|r_xi| * ( m_pot_pseudo x g1 )
         */
        CORE::LARGEROTATIONS::computespin(spin_pseudo_moment_tmp, moment_pot_tmp);

        moment_pot_tmp.Multiply(1.0 / CORE::FADUTILS::CastToDouble(norm_r_xi_slave),
            spin_pseudo_moment_tmp, CORE::FADUTILS::CastToDouble<T, 3, 1>(t_slave));

        moments_pot_gp_1_[igp_total].Update(prefactor_visualization_data, moment_pot_tmp, 1.0);


        forces_pot_gp_2_[igp_total].Update(
            prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_deriv_gap_ul),
            CORE::FADUTILS::CastToDouble<T, 3, 1>(gap_ul_deriv_r_master), 1.0);

        forces_pot_gp_2_[igp_total].Update(
            prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_deriv_cos_alpha),
            CORE::FADUTILS::CastToDouble<T, 3, 1>(cos_alpha_deriv_r_master), 1.0);


        moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_deriv_cos_alpha),
            CORE::FADUTILS::CastToDouble<T, 3, 1>(cos_alpha_deriv_r_xi_master));

        CORE::LARGEROTATIONS::computespin(spin_pseudo_moment_tmp, moment_pot_tmp);

        moment_pot_tmp.Multiply(1.0 / CORE::FADUTILS::CastToDouble(norm_r_xi_master),
            spin_pseudo_moment_tmp, CORE::FADUTILS::CastToDouble<T, 3, 1>(t_master));

        moments_pot_gp_2_[igp_total].Update(prefactor_visualization_data, moment_pot_tmp, 1.0);



        // now apply scalar integration factor
        // (after computing and storing distributed force/moment for vtk output)
        pot_ia_deriv_gap_ul *= rho1rho2_JacFac_GaussWeight;
        pot_ia_deriv_cos_alpha *= rho1rho2_JacFac_GaussWeight;

        pot_ia_2ndderiv_gap_ul *= rho1rho2_JacFac_GaussWeight;
        pot_ia_deriv_gap_ul_deriv_cos_alpha *= rho1rho2_JacFac_GaussWeight;
        pot_ia_2ndderiv_cos_alpha *= rho1rho2_JacFac_GaussWeight;


        //********************************************************************
        // calculate force/residual vector of slave and master element
        //********************************************************************
        force_pot_slave_GP.Clear();
        force_pot_master_GP.Clear();

        // sum up the contributions of all nodes (in all dimensions)
        for (unsigned int idofperdim = 0; idofperdim < (numnodes * numnodalvalues); ++idofperdim)
        {
          // loop over dimensions
          for (unsigned int jdim = 0; jdim < 3; ++jdim)
          {
            force_pot_slave_GP(3 * idofperdim + jdim) +=
                N_i_slave[igp](idofperdim) * pot_ia_deriv_gap_ul * gap_ul_deriv_r_slave(jdim);

            force_pot_slave_GP(3 * idofperdim + jdim) +=
                N_i_slave[igp](idofperdim) * pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_slave(jdim);

            force_pot_slave_GP(3 * idofperdim + jdim) += N_i_xi_slave[igp](idofperdim) *
                                                         pot_ia_deriv_cos_alpha *
                                                         cos_alpha_deriv_r_xi_slave(jdim);


            force_pot_master_GP(3 * idofperdim + jdim) +=
                N_i_master(idofperdim) * pot_ia_deriv_gap_ul * gap_ul_deriv_r_master(jdim);

            force_pot_master_GP(3 * idofperdim + jdim) +=
                N_i_master(idofperdim) * pot_ia_deriv_cos_alpha * cos_alpha_deriv_r_master(jdim);

            force_pot_master_GP(3 * idofperdim + jdim) += N_i_xi_master(idofperdim) *
                                                          pot_ia_deriv_cos_alpha *
                                                          cos_alpha_deriv_r_xi_master(jdim);
          }
        }


        if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
            stiffmat22 != nullptr)
        {
          // evaluate contributions to linearization based on analytical expression
          EvaluateStiffpotAnalyticContributions_SingleLengthSpecific_SmallSepApprox_Simple(
              N_i_slave[igp], N_i_xi_slave[igp], N_i_master, N_i_xi_master, N_i_xixi_master,
              xi_master, r_xi_slave, r_xi_master, r_xixi_master, norm_dist_ul, normal_ul,
              pot_ia_deriv_gap_ul, pot_ia_deriv_cos_alpha, pot_ia_2ndderiv_gap_ul,
              pot_ia_deriv_gap_ul_deriv_cos_alpha, pot_ia_2ndderiv_cos_alpha, gap_ul_deriv_r_slave,
              gap_ul_deriv_r_master, cos_alpha_deriv_r_slave, cos_alpha_deriv_r_master,
              cos_alpha_deriv_r_xi_slave, cos_alpha_deriv_r_xi_master, xi_master_partial_r_slave,
              xi_master_partial_r_master, xi_master_partial_r_xi_master, *stiffmat11, *stiffmat12,
              *stiffmat21, *stiffmat22);
        }
      }

      // apply constant prefactor
      force_pot_slave_GP.Scale(prefactor);
      force_pot_master_GP.Scale(prefactor);

      // sum contributions from all Gauss points
      force_pot1.Update(1.0, force_pot_slave_GP, 1.0);
      force_pot2.Update(1.0, force_pot_master_GP, 1.0);

      if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
          stiffmat22 != nullptr)
      {
        AddStiffmatContributionsXiMasterAutomaticDifferentiationIfRequired(force_pot_slave_GP,
            force_pot_master_GP, lin_xi_master_slaveDofs, lin_xi_master_masterDofs, *stiffmat11,
            *stiffmat12, *stiffmat21, *stiffmat22);
      }

      // do this scaling down here, because the value without the prefactors is meant to be used
      // in the force calculation above!
      interaction_potential_GP *= rho1rho2_JacFac_GaussWeight * prefactor;

      // store for energy output
      interaction_potential_ += CORE::FADUTILS::CastToDouble(interaction_potential_GP);

      //************************** DEBUG ******************************************
      //      std::cout << "\ninteraction_potential_GP: "
      //                << CORE::FADUTILS::CastToDouble(interaction_potential_GP);
      //*********************** END DEBUG *****************************************

    }  // end loop over Gauss points per segment
  }    // end loop over integration segments

  if (stiffmat11 != nullptr and stiffmat12 != nullptr and stiffmat21 != nullptr and
      stiffmat22 != nullptr)
  {
    ScaleStiffpotAnalyticContributionsIfRequired(
        prefactor, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);


    CalcStiffmatAutomaticDifferentiationIfRequired(
        force_pot1, force_pot2, *stiffmat11, *stiffmat12, *stiffmat21, *stiffmat22);
  }
}


/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    EvaluateStiffpotAnalyticContributions_SingleLengthSpecific_SmallSepApprox_Simple(
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_slave,
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_slave,
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_master,
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_master,
        CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xixi_master,
        double const& xi_master, CORE::LINALG::Matrix<3, 1, double> const& r_xi_slave,
        CORE::LINALG::Matrix<3, 1, double> const& r_xi_master,
        CORE::LINALG::Matrix<3, 1, double> const& r_xixi_master, double const& norm_dist_ul,
        CORE::LINALG::Matrix<3, 1, double> const& normal_ul, double const& pot_ia_deriv_gap_ul,
        double const& pot_ia_deriv_cos_alpha, double const& pot_ia_2ndderiv_gap_ul,
        double const& pot_ia_deriv_gap_ul_deriv_cos_alpha, double const& pot_ia_2ndderiv_cos_alpha,
        CORE::LINALG::Matrix<3, 1, double> const& gap_ul_deriv_r_slave,
        CORE::LINALG::Matrix<3, 1, double> const& gap_ul_deriv_r_master,
        CORE::LINALG::Matrix<3, 1, double> const& cos_alpha_deriv_r_slave,
        CORE::LINALG::Matrix<3, 1, double> const& cos_alpha_deriv_r_master,
        CORE::LINALG::Matrix<3, 1, double> const& cos_alpha_deriv_r_xi_slave,
        CORE::LINALG::Matrix<3, 1, double> const& cos_alpha_deriv_r_xi_master,
        CORE::LINALG::Matrix<1, 3, double> const& xi_master_partial_r_slave,
        CORE::LINALG::Matrix<1, 3, double> const& xi_master_partial_r_master,
        CORE::LINALG::Matrix<1, 3, double> const& xi_master_partial_r_xi_master,
        CORE::LINALG::SerialDenseMatrix& stiffmat11, CORE::LINALG::SerialDenseMatrix& stiffmat12,
        CORE::LINALG::SerialDenseMatrix& stiffmat21,
        CORE::LINALG::SerialDenseMatrix& stiffmat22) const
{
  const unsigned int num_spatial_dimensions = 3;
  const unsigned int num_dofs_per_spatial_dimension = numnodes * numnodalvalues;


  // auxiliary and intermediate quantities required for second derivatives of the unilateral gap
  CORE::LINALG::Matrix<3, 3, double> unit_matrix(true);
  for (unsigned int i = 0; i < 3; ++i) unit_matrix(i, i) = 1.0;

  CORE::LINALG::Matrix<3, 3, double> dist_ul_deriv_r_slave(unit_matrix);
  CORE::LINALG::Matrix<3, 3, double> dist_ul_deriv_r_master(unit_matrix);
  dist_ul_deriv_r_master.Scale(-1.0);
  CORE::LINALG::Matrix<3, 3, double> dist_ul_deriv_r_xi_master(true);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      dist_ul_deriv_r_slave(irow, icol) -= r_xi_master(irow) * xi_master_partial_r_slave(icol);

      dist_ul_deriv_r_master(irow, icol) -= r_xi_master(irow) * xi_master_partial_r_master(icol);

      dist_ul_deriv_r_xi_master(irow, icol) -=
          r_xi_master(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  CORE::LINALG::Matrix<3, 3, double> normal_ul_deriv_dist_ul(true);

  for (unsigned int i = 0; i < 3; ++i)
  {
    normal_ul_deriv_dist_ul(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j)
      normal_ul_deriv_dist_ul(i, j) -= normal_ul(i) * normal_ul(j);
  }
  normal_ul_deriv_dist_ul.Scale(1.0 / norm_dist_ul);


  // second derivatives of the unilateral gap
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_slave_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_slave_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_slave_deriv_r_xi_master(true);

  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_master_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_master_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_master_deriv_r_xi_master(true);


  gap_ul_deriv_r_slave_deriv_r_slave.Multiply(1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_slave);
  gap_ul_deriv_r_slave_deriv_r_master.Multiply(
      1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_master);
  gap_ul_deriv_r_slave_deriv_r_xi_master.Multiply(
      1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_xi_master);

  gap_ul_deriv_r_master_deriv_r_slave.Multiply(
      -1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_slave);
  gap_ul_deriv_r_master_deriv_r_master.Multiply(
      -1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_master);
  gap_ul_deriv_r_master_deriv_r_xi_master.Multiply(
      -1.0, normal_ul_deriv_dist_ul, dist_ul_deriv_r_xi_master);

  // add contributions from linearization of (variation of r_master) according to the chain
  // rule
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_xi_master_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_xi_master_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> gap_ul_deriv_r_xi_master_deriv_r_xi_master(true);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      gap_ul_deriv_r_xi_master_deriv_r_slave(irow, icol) -=
          normal_ul(irow) * xi_master_partial_r_slave(icol);

      gap_ul_deriv_r_xi_master_deriv_r_master(irow, icol) -=
          normal_ul(irow) * xi_master_partial_r_master(icol);

      gap_ul_deriv_r_xi_master_deriv_r_xi_master(irow, icol) -=
          normal_ul(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  // second derivatives of cos(alpha)
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_slave_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_slave_deriv_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_slave_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_slave_deriv_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_slave_deriv_r_xixi_master(true);

  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_slave_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_slave_deriv_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_slave_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_slave_deriv_r_xi_master(true);

  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_master_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_master_deriv_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_master_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_master_deriv_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_master_deriv_r_xixi_master(true);

  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_master_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_master_deriv_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_master_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_master_deriv_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xi_master_deriv_r_xixi_master(true);


  // auxiliary and intermediate quantities required for second derivatives of cos(alpha)

  double norm_r_xi_slave_inverse = 1.0 / CORE::FADUTILS::VectorNorm(r_xi_slave);
  double norm_r_xi_master_inverse = 1.0 / CORE::FADUTILS::VectorNorm(r_xi_master);

  CORE::LINALG::Matrix<3, 1, double> t_slave(true);
  t_slave.Update(norm_r_xi_slave_inverse, r_xi_slave);
  CORE::LINALG::Matrix<3, 1, double> t_master(true);
  t_master.Update(norm_r_xi_master_inverse, r_xi_master);

  double t_slave_dot_t_master = t_slave.Dot(t_master);
  double signum_tangentsscalarproduct = CORE::FADUTILS::Signum(t_slave_dot_t_master);

  CORE::LINALG::Matrix<3, 3, double> t_slave_tensorproduct_t_slave(true);
  CORE::LINALG::Matrix<3, 3, double> t_slave_tensorproduct_t_master(true);
  CORE::LINALG::Matrix<3, 3, double> t_master_tensorproduct_t_master(true);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
    {
      t_slave_tensorproduct_t_slave(i, j) = t_slave(i) * t_slave(j);
      t_slave_tensorproduct_t_master(i, j) = t_slave(i) * t_master(j);
      t_master_tensorproduct_t_master(i, j) = t_master(i) * t_master(j);
    }

  CORE::LINALG::Matrix<3, 3, double> t_slave_partial_r_xi_slave(true);
  t_slave_partial_r_xi_slave.Update(norm_r_xi_slave_inverse, unit_matrix,
      -1.0 * norm_r_xi_slave_inverse, t_slave_tensorproduct_t_slave);

  CORE::LINALG::Matrix<3, 3, double> t_master_partial_r_xi_master(true);
  t_master_partial_r_xi_master.Update(norm_r_xi_master_inverse, unit_matrix,
      -1.0 * norm_r_xi_master_inverse, t_master_tensorproduct_t_master);


  CORE::LINALG::Matrix<3, 1, double> t_slave_partial_r_xi_slave_mult_t_master(true);
  t_slave_partial_r_xi_slave_mult_t_master.Multiply(t_slave_partial_r_xi_slave, t_master);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      cos_alpha_deriv_r_xi_slave_deriv_r_xi_slave(i, j) -=
          norm_r_xi_slave_inverse * t_slave_partial_r_xi_slave_mult_t_master(i) * t_slave(j);

  cos_alpha_deriv_r_xi_slave_deriv_r_xi_slave.Update(
      -1.0 * norm_r_xi_slave_inverse * t_slave_dot_t_master, t_slave_partial_r_xi_slave, 1.0);

  CORE::LINALG::Matrix<3, 3, double> tmp_mat(true);
  tmp_mat.Multiply(t_slave_tensorproduct_t_master, t_slave_partial_r_xi_slave);
  cos_alpha_deriv_r_xi_slave_deriv_r_xi_slave.Update(-1.0 * norm_r_xi_slave_inverse, tmp_mat, 1.0);


  cos_alpha_deriv_r_xi_slave_deriv_r_xi_master.Update(
      norm_r_xi_slave_inverse, t_master_partial_r_xi_master, 1.0);

  tmp_mat.Multiply(t_slave_tensorproduct_t_slave, t_master_partial_r_xi_master);
  cos_alpha_deriv_r_xi_slave_deriv_r_xi_master.Update(-1.0 * norm_r_xi_slave_inverse, tmp_mat, 1.0);



  CORE::LINALG::Matrix<3, 1, double> t_master_partial_r_xi_master_mult_t_slave(true);
  t_master_partial_r_xi_master_mult_t_slave.Multiply(t_master_partial_r_xi_master, t_slave);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      cos_alpha_deriv_r_xi_master_deriv_r_xi_master(i, j) -=
          norm_r_xi_master_inverse * t_master_partial_r_xi_master_mult_t_slave(i) * t_master(j);

  cos_alpha_deriv_r_xi_master_deriv_r_xi_master.Update(
      -1.0 * norm_r_xi_master_inverse * t_slave_dot_t_master, t_master_partial_r_xi_master, 1.0);

  tmp_mat.Clear();
  tmp_mat.MultiplyTN(t_slave_tensorproduct_t_master, t_master_partial_r_xi_master);
  cos_alpha_deriv_r_xi_master_deriv_r_xi_master.Update(
      -1.0 * norm_r_xi_master_inverse, tmp_mat, 1.0);


  cos_alpha_deriv_r_xi_master_deriv_r_xi_slave.Update(
      norm_r_xi_master_inverse, t_slave_partial_r_xi_slave, 1.0);

  tmp_mat.Multiply(t_master_tensorproduct_t_master, t_slave_partial_r_xi_slave);
  cos_alpha_deriv_r_xi_master_deriv_r_xi_slave.Update(
      -1.0 * norm_r_xi_master_inverse, tmp_mat, 1.0);


  // add contributions from variation of master parameter coordinate xi_master
  // to [.]_deriv_r_xi_master_[.] expressions (according to chain rule)
  CORE::LINALG::Matrix<1, 3, double> tmp_vec;
  tmp_vec.MultiplyTN(r_xixi_master, cos_alpha_deriv_r_xi_master_deriv_r_xi_slave);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_slave_deriv_r_xi_slave(irow, icol) +=
          xi_master_partial_r_slave(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_master_deriv_r_xi_slave(irow, icol) +=
          xi_master_partial_r_master(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_xi_slave(irow, icol) +=
          xi_master_partial_r_xi_master(irow) * tmp_vec(icol);
    }
  }

  tmp_vec.MultiplyTN(r_xixi_master, cos_alpha_deriv_r_xi_master_deriv_r_xi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_slave_deriv_r_xi_master(irow, icol) +=
          xi_master_partial_r_slave(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_master_deriv_r_xi_master(irow, icol) +=
          xi_master_partial_r_master(irow) * tmp_vec(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_xi_master(irow, icol) +=
          xi_master_partial_r_xi_master(irow) * tmp_vec(icol);
    }
  }


  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_slave_deriv_r_xixi_master(irow, icol) +=
          xi_master_partial_r_slave(irow) * t_master_partial_r_xi_master_mult_t_slave(icol);

      cos_alpha_deriv_r_master_deriv_r_xixi_master(irow, icol) +=
          xi_master_partial_r_master(irow) * t_master_partial_r_xi_master_mult_t_slave(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_xixi_master(irow, icol) +=
          xi_master_partial_r_xi_master(irow) * t_master_partial_r_xi_master_mult_t_slave(icol);
    }
  }


  // add contributions from linearization of master parameter coordinate xi_master
  // to [.]_deriv_r_xi_master expressions (according to chain rule)
  CORE::LINALG::Matrix<3, 1, double> tmp_vec2;
  tmp_vec2.Multiply(cos_alpha_deriv_r_slave_deriv_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_slave_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_slave_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_slave_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  tmp_vec2.Multiply(cos_alpha_deriv_r_xi_slave_deriv_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xi_slave_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_xi_slave_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_xi_slave_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  tmp_vec2.Multiply(cos_alpha_deriv_r_master_deriv_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_master_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_master_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_master_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  tmp_vec2.Multiply(cos_alpha_deriv_r_xi_master_deriv_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xi_master_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  // add contributions from linearization of master parameter coordinate xi_master
  // also to [.]_deriv_r_xixi_master expressions (according to chain rule)
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> N_i_xixixi_master(true);

  DRT::UTILS::BEAM::EvaluateShapeFunction3rdDerivsAtXi<numnodes, numnodalvalues>(
      xi_master, N_i_xixixi_master, BeamElement2()->Shape(), ele2length_);

  CORE::LINALG::Matrix<3, 1, double> r_xixixi_master(true);

  DRT::UTILS::BEAM::CalcInterpolation<numnodes, numnodalvalues, 3>(
      CORE::FADUTILS::CastToDouble(ele2pos_), N_i_xixixi_master, r_xixixi_master);

  tmp_vec2.Multiply(cos_alpha_deriv_r_slave_deriv_r_xixi_master, r_xixixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_slave_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_slave_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_slave_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }

  tmp_vec2.Multiply(cos_alpha_deriv_r_master_deriv_r_xixi_master, r_xixixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_master_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_master_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_master_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }

  tmp_vec2.Multiply(cos_alpha_deriv_r_xi_master_deriv_r_xixi_master, r_xixixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xi_master_deriv_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_xi_master_deriv_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  // add contributions from linearization of (variation of r_xi_master) according to the chain
  // rule
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xixi_master_deriv_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xixi_master_deriv_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> cos_alpha_deriv_r_xixi_master_deriv_r_xi_master(true);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      cos_alpha_deriv_r_xixi_master_deriv_r_slave(irow, icol) +=
          t_master_partial_r_xi_master_mult_t_slave(irow) * xi_master_partial_r_slave(icol);

      cos_alpha_deriv_r_xixi_master_deriv_r_master(irow, icol) +=
          t_master_partial_r_xi_master_mult_t_slave(irow) * xi_master_partial_r_master(icol);

      cos_alpha_deriv_r_xixi_master_deriv_r_xi_master(irow, icol) +=
          t_master_partial_r_xi_master_mult_t_slave(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  // contributions from linarization of the (variation of master parameter coordinate xi_master)
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_slave_partial_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_slave_partial_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_slave_partial_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_slave_partial_r_xixi_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_master_partial_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_master_partial_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_master_partial_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_master_partial_r_xixi_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xi_master_partial_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xi_master_partial_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xi_master_partial_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xi_master_partial_r_xixi_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xixi_master_partial_r_slave(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xixi_master_partial_r_master(true);
  CORE::LINALG::Matrix<3, 3, double> xi_master_partial_r_xixi_master_partial_r_xi_master(true);

  CORE::LINALG::Matrix<3, 1, double> dist_ul(true);
  dist_ul.Update(norm_dist_ul, normal_ul);

  BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartial2ndDerivs(
      xi_master_partial_r_slave_partial_r_slave, xi_master_partial_r_slave_partial_r_master,
      xi_master_partial_r_slave_partial_r_xi_master,
      xi_master_partial_r_slave_partial_r_xixi_master, xi_master_partial_r_master_partial_r_slave,
      xi_master_partial_r_master_partial_r_master, xi_master_partial_r_master_partial_r_xi_master,
      xi_master_partial_r_master_partial_r_xixi_master,
      xi_master_partial_r_xi_master_partial_r_slave, xi_master_partial_r_xi_master_partial_r_master,
      xi_master_partial_r_xi_master_partial_r_xi_master,
      xi_master_partial_r_xi_master_partial_r_xixi_master,
      xi_master_partial_r_xixi_master_partial_r_slave,
      xi_master_partial_r_xixi_master_partial_r_master,
      xi_master_partial_r_xixi_master_partial_r_xi_master, xi_master_partial_r_slave,
      xi_master_partial_r_master, xi_master_partial_r_xi_master, dist_ul_deriv_r_slave,
      dist_ul_deriv_r_master, dist_ul_deriv_r_xi_master, dist_ul, r_xi_master, r_xixi_master,
      r_xixixi_master);

  double r_xixi_master_dot_v2 = r_xixi_master.Dot(t_master_partial_r_xi_master_mult_t_slave);

  cos_alpha_deriv_r_slave_deriv_r_slave.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_slave_partial_r_slave, 1.0);

  cos_alpha_deriv_r_slave_deriv_r_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_slave_partial_r_master, 1.0);

  cos_alpha_deriv_r_slave_deriv_r_xi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_slave_partial_r_xi_master, 1.0);

  cos_alpha_deriv_r_slave_deriv_r_xixi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_slave_partial_r_xixi_master, 1.0);


  cos_alpha_deriv_r_master_deriv_r_slave.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_master_partial_r_slave, 1.0);

  cos_alpha_deriv_r_master_deriv_r_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_master_partial_r_master, 1.0);

  cos_alpha_deriv_r_master_deriv_r_xi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_master_partial_r_xi_master, 1.0);

  cos_alpha_deriv_r_master_deriv_r_xixi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_master_partial_r_xixi_master, 1.0);


  cos_alpha_deriv_r_xi_master_deriv_r_slave.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xi_master_partial_r_slave, 1.0);

  cos_alpha_deriv_r_xi_master_deriv_r_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xi_master_partial_r_master, 1.0);

  cos_alpha_deriv_r_xi_master_deriv_r_xi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xi_master_partial_r_xi_master, 1.0);

  cos_alpha_deriv_r_xi_master_deriv_r_xixi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xi_master_partial_r_xixi_master, 1.0);


  cos_alpha_deriv_r_xixi_master_deriv_r_slave.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xixi_master_partial_r_slave, 1.0);

  cos_alpha_deriv_r_xixi_master_deriv_r_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xixi_master_partial_r_master, 1.0);

  cos_alpha_deriv_r_xixi_master_deriv_r_xi_master.Update(
      r_xixi_master_dot_v2, xi_master_partial_r_xixi_master_partial_r_xi_master, 1.0);


  cos_alpha_deriv_r_slave_deriv_r_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_slave_deriv_r_xi_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_slave_deriv_r_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_slave_deriv_r_xi_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_slave_deriv_r_xixi_master.Scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_xi_slave_deriv_r_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_slave_deriv_r_xi_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_slave_deriv_r_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_slave_deriv_r_xi_master.Scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_master_deriv_r_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_master_deriv_r_xi_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_master_deriv_r_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_master_deriv_r_xi_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_master_deriv_r_xixi_master.Scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_xi_master_deriv_r_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_master_deriv_r_xi_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_master_deriv_r_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_master_deriv_r_xi_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xi_master_deriv_r_xixi_master.Scale(signum_tangentsscalarproduct);

  cos_alpha_deriv_r_xixi_master_deriv_r_slave.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xixi_master_deriv_r_master.Scale(signum_tangentsscalarproduct);
  cos_alpha_deriv_r_xixi_master_deriv_r_xi_master.Scale(signum_tangentsscalarproduct);


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
          // variation of gap_ul * linearization of pot_ia_deriv_gap_ul
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_slave(irowdofperdim) * gap_ul_deriv_r_slave(irowdim) *
              (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_xi_slave(icolumndim) *
                      N_i_xi_slave(icolumndofperdim));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_slave(irowdofperdim) * gap_ul_deriv_r_slave(irowdim) *
              (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_xi_master(icolumndim) *
                      N_i_xi_master(icolumndofperdim));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_master(irowdofperdim) * gap_ul_deriv_r_master(irowdim) *
              (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_xi_slave(icolumndim) *
                      N_i_xi_slave(icolumndofperdim));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              N_i_master(irowdofperdim) * gap_ul_deriv_r_master(irowdim) *
              (pot_ia_2ndderiv_gap_ul * gap_ul_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * cos_alpha_deriv_r_xi_master(icolumndim) *
                      N_i_xi_master(icolumndofperdim));


          // linearization of the (variation of gap_ul) * pot_ia_deriv_gap_ul
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_gap_ul * N_i_slave(irowdofperdim) *
              gap_ul_deriv_r_slave_deriv_r_slave(irowdim, icolumndim) * N_i_slave(icolumndofperdim);

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_gap_ul * N_i_slave(irowdofperdim) *
              (gap_ul_deriv_r_slave_deriv_r_master(irowdim, icolumndim) *
                      N_i_master(icolumndofperdim) +
                  gap_ul_deriv_r_slave_deriv_r_xi_master(irowdim, icolumndim) *
                      N_i_xi_master(icolumndofperdim));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_gap_ul *
              (N_i_master(irowdofperdim) *
                      gap_ul_deriv_r_master_deriv_r_slave(irowdim, icolumndim) +
                  N_i_xi_master(irowdofperdim) *
                      gap_ul_deriv_r_xi_master_deriv_r_slave(irowdim, icolumndim)) *
              N_i_slave(icolumndofperdim);

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_gap_ul *
              (N_i_master(irowdofperdim) *
                      (gap_ul_deriv_r_master_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          gap_ul_deriv_r_master_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim)) +
                  N_i_xi_master(irowdofperdim) *
                      (gap_ul_deriv_r_xi_master_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          gap_ul_deriv_r_xi_master_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim)));


          // variation of cos_alpha * linearization of pot_ia_deriv_cos_alpha
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_slave(irowdofperdim) * cos_alpha_deriv_r_slave(irowdim) +
                  N_i_xi_slave(irowdofperdim) * cos_alpha_deriv_r_xi_slave(irowdim)) *
              (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim) +
                  pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_slave(icolumndim) *
                      N_i_xi_slave(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_slave(irowdofperdim) * cos_alpha_deriv_r_slave(irowdim) +
                  N_i_xi_slave(irowdofperdim) * cos_alpha_deriv_r_xi_slave(irowdim)) *
              (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim) +
                  pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_master(icolumndim) *
                      N_i_xi_master(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_master(irowdofperdim) * cos_alpha_deriv_r_master(irowdim) +
                  N_i_xi_master(irowdofperdim) * cos_alpha_deriv_r_xi_master(irowdim)) *
              (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim) +
                  pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_slave(icolumndim) *
                      N_i_xi_slave(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_slave(icolumndim) *
                      N_i_slave(icolumndofperdim));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              (N_i_master(irowdofperdim) * cos_alpha_deriv_r_master(irowdim) +
                  N_i_xi_master(irowdofperdim) * cos_alpha_deriv_r_xi_master(irowdim)) *
              (pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim) +
                  pot_ia_2ndderiv_cos_alpha * cos_alpha_deriv_r_xi_master(icolumndim) *
                      N_i_xi_master(icolumndofperdim) +
                  pot_ia_deriv_gap_ul_deriv_cos_alpha * gap_ul_deriv_r_master(icolumndim) *
                      N_i_master(icolumndofperdim));


          // linearization of the (variation of cos_alpha) * pot_ia_deriv_cos_alpha
          stiffmat11(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_cos_alpha *
              (N_i_slave(irowdofperdim) *
                      (cos_alpha_deriv_r_slave_deriv_r_slave(irowdim, icolumndim) *
                              N_i_slave(icolumndofperdim) +
                          cos_alpha_deriv_r_slave_deriv_r_xi_slave(irowdim, icolumndim) *
                              N_i_xi_slave(icolumndofperdim)) +
                  N_i_xi_slave(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_slave_deriv_r_slave(irowdim, icolumndim) *
                              N_i_slave(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_slave_deriv_r_xi_slave(irowdim, icolumndim) *
                              N_i_xi_slave(icolumndofperdim)));

          stiffmat12(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_cos_alpha *
              (N_i_slave(irowdofperdim) *
                      (cos_alpha_deriv_r_slave_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          cos_alpha_deriv_r_slave_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim) +
                          cos_alpha_deriv_r_slave_deriv_r_xixi_master(irowdim, icolumndim) *
                              N_i_xixi_master(icolumndofperdim)) +
                  N_i_xi_slave(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_slave_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_slave_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim)));

          stiffmat21(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_cos_alpha *
              (N_i_master(irowdofperdim) *
                      (cos_alpha_deriv_r_master_deriv_r_slave(irowdim, icolumndim) *
                              N_i_slave(icolumndofperdim) +
                          cos_alpha_deriv_r_master_deriv_r_xi_slave(irowdim, icolumndim) *
                              N_i_xi_slave(icolumndofperdim)) +
                  N_i_xi_master(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_master_deriv_r_slave(irowdim, icolumndim) *
                              N_i_slave(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_master_deriv_r_xi_slave(irowdim, icolumndim) *
                              N_i_xi_slave(icolumndofperdim)) +
                  N_i_xixi_master(irowdofperdim) *
                      (cos_alpha_deriv_r_xixi_master_deriv_r_slave(irowdim, icolumndim) *
                          N_i_slave(icolumndofperdim)));

          stiffmat22(3 * irowdofperdim + irowdim, 3 * icolumndofperdim + icolumndim) +=
              pot_ia_deriv_cos_alpha *
              (N_i_master(irowdofperdim) *
                      (cos_alpha_deriv_r_master_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          cos_alpha_deriv_r_master_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim) +
                          cos_alpha_deriv_r_master_deriv_r_xixi_master(irowdim, icolumndim) *
                              N_i_xixi_master(icolumndofperdim)) +
                  N_i_xi_master(irowdofperdim) *
                      (cos_alpha_deriv_r_xi_master_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_master_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim) +
                          cos_alpha_deriv_r_xi_master_deriv_r_xixi_master(irowdim, icolumndim) *
                              N_i_xixi_master(icolumndofperdim)) +
                  N_i_xixi_master(irowdofperdim) *
                      (cos_alpha_deriv_r_xixi_master_deriv_r_master(irowdim, icolumndim) *
                              N_i_master(icolumndofperdim) +
                          cos_alpha_deriv_r_xixi_master_deriv_r_xi_master(irowdim, icolumndim) *
                              N_i_xi_master(icolumndofperdim)));
        }
      }
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::EvaluateFullDiskCylinderPotential(T& interaction_potential_GP,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot_slave_GP,
    CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& force_pot_master_GP,
    CORE::LINALG::Matrix<3, 1, T> const& r_slave, CORE::LINALG::Matrix<3, 1, T> const& r_xi_slave,
    CORE::LINALG::Matrix<3, 1, T> const& t1_slave, CORE::LINALG::Matrix<3, 1, T> const& r_master,
    CORE::LINALG::Matrix<3, 1, T> const& r_xi_master,
    CORE::LINALG::Matrix<3, 1, T> const& r_xixi_master,
    CORE::LINALG::Matrix<3, 1, T> const& t1_master, T alpha, T cos_alpha,
    CORE::LINALG::Matrix<3, 1, T> const& dist_ul,
    CORE::LINALG::Matrix<1, 3, T> const& xi_master_partial_r_slave,
    CORE::LINALG::Matrix<1, 3, T> const& xi_master_partial_r_master,
    CORE::LINALG::Matrix<1, 3, T> const& xi_master_partial_r_xi_master,
    double prefactor_visualization_data, CORE::LINALG::Matrix<3, 1, double>& vtk_force_pot_slave_GP,
    CORE::LINALG::Matrix<3, 1, double>& vtk_force_pot_master_GP,
    CORE::LINALG::Matrix<3, 1, double>& vtk_moment_pot_slave_GP,
    CORE::LINALG::Matrix<3, 1, double>& vtk_moment_pot_master_GP,
    double rho1rho2_JacFac_GaussWeight,
    CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_slave,
    CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double> const& N_i_xi_slave,
    CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> const& N_i_master,
    CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> const& N_i_xi_master)
{
  // get regularization type and separation
  const INPAR::BEAMPOTENTIAL::BeamPotentialRegularizationType regularization_type =
      Params()->RegularizationType();

  const double regularization_separation = Params()->RegularizationSeparation();


  T sin_alpha = 0.0;                              // sine of mutual angle of tangent vectors
  T sin_2alpha = 0.0;                             // sine of 2*mutual angle of tangent vectors
  CORE::LINALG::Matrix<3, 1, T> normal_bl(true);  // normal vector at bilateral closest point
  T norm_normal_bl_tilde = 0.0;                   // norm of vector defining bilateral normal vector
  T gap_bl = 0.0;                                 // gap of bilateral closest point
  T x = 0.0;  // distance between Gauss point and bilateral closest point on slave
  CORE::LINALG::Matrix<3, 1, T> aux_plane_normal(true);  // normal vector of auxiliary plane n*

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
  CORE::LINALG::Matrix<3, 1, T> gap_bl_partial_r_slave(true);
  CORE::LINALG::Matrix<3, 1, T> gap_bl_partial_r_master(true);
  CORE::LINALG::Matrix<3, 1, T> gap_bl_partial_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 1, T> gap_bl_partial_r_xi_master(true);
  T gap_bl_partial_xi_master = 0.0;

  // components from variation of cosine of enclosed angle
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_partial_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 1, T> cos_alpha_partial_r_xi_master(true);
  T cos_alpha_partial_xi_master = 0.0;

  // components from variation of distance from bilateral closest point on slave
  CORE::LINALG::Matrix<3, 1, T> x_partial_r_slave(true);
  CORE::LINALG::Matrix<3, 1, T> x_partial_r_master(true);
  CORE::LINALG::Matrix<3, 1, T> x_partial_r_xi_slave(true);
  CORE::LINALG::Matrix<3, 1, T> x_partial_r_xi_master(true);

  CORE::LINALG::Matrix<3, 1, T> x_partial_aux_plane_normal(true);
  T x_partial_xi_master = 0.0;


  // auxiliary variables
  CORE::LINALG::Matrix<3, 1, T> fpot_tmp(true);
  CORE::LINALG::Matrix<3, 3, T> v_mat_tmp(true);

  sin_alpha = std::sin(alpha);
  sin_2alpha = std::sin(2 * alpha);

  const double BEAMSCOLINEARANGLETHRESHOLD = 5.0 / 180.0 * M_PI;  // 5 works best so far

  if (CORE::FADUTILS::Norm(alpha) < BEAMSCOLINEARANGLETHRESHOLD)
  {
    //************************** DEBUG ******************************************
    //      std::cout << "\n\nINFO: Enclosed angle is close to zero: alpha="
    //          << CORE::FADUTILS::CastToDouble(alpha)*180/M_PI << "degrees\n"
    //          << std::endl;
    //*********************** END DEBUG *****************************************

    // there is no unique bilateral closest point pair in case of alpha=0
    // hence, we can use the current (unilateral) closest point pair
    normal_bl = dist_ul;
    norm_normal_bl_tilde = CORE::FADUTILS::VectorNorm(normal_bl);
    normal_bl.Scale(1.0 / norm_normal_bl_tilde);

    aux_plane_normal.Clear();
    x = 0.0;
  }
  else
  {
    // normal vector at bilateral closest point Fixme
    normal_bl.CrossProduct(r_xi_slave, r_xi_master);
    norm_normal_bl_tilde = CORE::FADUTILS::VectorNorm(normal_bl);
    normal_bl.Scale(1.0 / norm_normal_bl_tilde);

    // distance between Gauss point and bilateral closest point on slave
    aux_plane_normal.Update(
        r_xi_master.Dot(r_xi_master), r_xi_slave, -1.0 * r_xi_master.Dot(r_xi_slave), r_xi_master);

    x = CORE::FADUTILS::VectorNorm(r_xi_slave) *
        (r_master.Dot(aux_plane_normal) - r_slave.Dot(aux_plane_normal)) /
        r_xi_slave.Dot(aux_plane_normal);
  }

  // gap of bilateral closest point (also valid for special case alpha=0)
  gap_bl = CORE::FADUTILS::Norm(dist_ul.Dot(normal_bl)) - radius1_ - radius2_;

  const double MAXNEGATIVEBILATERALGAP = -0.9 * radius2_;

  if (CORE::FADUTILS::Norm(alpha) >= BEAMSCOLINEARANGLETHRESHOLD and
      gap_bl < MAXNEGATIVEBILATERALGAP)
  {
    //************************** DEBUG **********************************************
    // std::cout << "\nINFO: Ignored GP (ele GIDs " << Element1()->Id() << "&" <<
    // Element2()->Id()
    //          << ": iGP " << igp_total
    //          << ") with alpha=" << CORE::FADUTILS::CastToDouble(alpha) * 180 / M_PI
    //          << "degrees >= " << BEAMSCOLINEARANGLETHRESHOLD * 180 / M_PI
    //          << "degrees and gap_bl/R=" << CORE::FADUTILS::CastToDouble(gap_bl) / radius2_ << " <
    //          "
    //          << MAXNEGATIVEBILATERALGAP / radius2_
    //          << " and x/R=" << CORE::FADUTILS::CastToDouble(x) / radius2_ << std::endl;
    //*********************** END DEBUG *********************************************

    if (CORE::FADUTILS::Norm(x) < 20 * radius2_)
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

  if (std::norm(CORE::FADUTILS::CastToDouble(gap_bl) + radius2_) < 1e-14)
    FOUR_C_THROW(
        "bilateral gap=%f is close to negative radius and thus the interaction potential is "
        "close to singular! Fatal error. Take care of this case!",
        CORE::FADUTILS::CastToDouble(gap_bl));

  //************************** DEBUG ******************************************
  //      std::cout << "\nnormal_bl: " << CORE::FADUTILS::CastToDouble<T, 3, 1>(normal_bl);
  //      std::cout << "\ngap_bl: " << CORE::FADUTILS::CastToDouble(gap_bl);
  //      std::cout << "\naux_plane_normal: " << CORE::FADUTILS::CastToDouble<T, 3,
  //      1>(aux_plane_normal); std::cout << "\nx: " << CORE::FADUTILS::CastToDouble(x) <<
  //      std::endl;
  //*********************** END DEBUG *****************************************

  beta = CORE::FADUTILS::sqrt<T>(
      (gap_bl + radius2_) * (gap_bl + radius2_) + x * x * sin_alpha * sin_alpha);

  if (beta < 1e-14)
    FOUR_C_THROW("beta=%f is negative or very close to zero! Fatal error. Take care of this case!",
        CORE::FADUTILS::CastToDouble(beta));

  beta_exp2 = beta * beta;
  beta_exp3 = beta_exp2 * beta;
  beta_exp4 = beta_exp2 * beta_exp2;

  a = 0.5 / beta *
      ((gap_bl + radius2_) / radius2_ + cos_alpha * cos_alpha -
          x * x * sin_2alpha * sin_2alpha / (4.0 * beta_exp2));

  Delta = 4 * a * (beta - radius2_) - x * x * sin_2alpha * sin_2alpha / (4 * beta_exp2);


  //      if (gap_bl < 0.0)
  //      {
  //      ************************** DEBUG **********************************************
  //      std::cout << "\nINFO: GP with negative gap_bl (ele GIDs " << Element1()->Id() << "&"
  //                << Element2()->Id() << ": iGP " << igp_total
  //                << ") with alpha=" << CORE::FADUTILS::CastToDouble(alpha) * 180 / M_PI
  //                << "degrees and gap_bl/R=" << CORE::FADUTILS::CastToDouble(gap_bl) / radius2_
  //                << " and x/R=" << CORE::FADUTILS::CastToDouble(x) / radius2_;
  //
  //      std::cout << "\n|dist_ul|: " <<
  //      CORE::FADUTILS::CastToDouble(CORE::FADUTILS::Norm(dist_ul)); std::cout << "\nbeta: " <<
  //      CORE::FADUTILS::CastToDouble(beta); std::cout << "\na: " <<
  //      CORE::FADUTILS::CastToDouble(a); std::cout << "\nDelta: " <<
  //      CORE::FADUTILS::CastToDouble(Delta);
  //      *********************** END DEBUG *********************************************
  //      }

  //************************** DEBUG ******************************************
  //    std::cout << "\nbeta: " << CORE::FADUTILS::CastToDouble(beta);
  //    std::cout << "\nbeta^2: " << CORE::FADUTILS::CastToDouble( beta*beta );
  //    std::cout << "\nbeta^3: " << CORE::FADUTILS::CastToDouble( beta*beta*beta );
  //    std::cout << "\nDelta: " << CORE::FADUTILS::CastToDouble(Delta);
  //*********************** END DEBUG *****************************************


  if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_none and Delta < 1e-14)
  {
    this->Print(std::cout);

    std::cout << "\ngap_bl: " << CORE::FADUTILS::CastToDouble(gap_bl);
    std::cout << "\nalpha: " << CORE::FADUTILS::CastToDouble(alpha * 180 / M_PI) << "degrees";
    std::cout << "\nx: " << CORE::FADUTILS::CastToDouble(x) << std::endl;

    std::cout << "\nbeta: " << CORE::FADUTILS::CastToDouble(beta);
    std::cout << "\na: " << CORE::FADUTILS::CastToDouble(a);

    FOUR_C_THROW("Delta=%f is negative or very close to zero! Use a regularization to handle this!",
        CORE::FADUTILS::CastToDouble(Delta));
  }

  Delta_regularized = Delta;

  if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear and
      Delta < regularization_separation)
  {
    Delta_regularized = regularization_separation;

    //************************** DEBUG ******************************************
    //    std::cout << "\nDelta: " << CORE::FADUTILS::CastToDouble(Delta) << ": regularization
    //    active!"; std::cout << "\nDelta_regularized: " <<
    //    CORE::FADUTILS::CastToDouble(Delta_regularized);
    //
    // this->Print(std::cout);
    // std::cout << "\nigp_total: " << igp_total;
    //
    // std::cout << "\ngap_bl: " << CORE::FADUTILS::CastToDouble(gap_bl);
    // std::cout << "\nalpha: " << CORE::FADUTILS::CastToDouble(alpha * 180 / M_PI) <<  "degrees";
    // std::cout << "\nx: " << CORE::FADUTILS::CastToDouble(x) << std::endl;
    //
    // std::cout << "\nbeta: " << CORE::FADUTILS::CastToDouble(beta);
    // std::cout << "\na: " << CORE::FADUTILS::CastToDouble(a);
    //*********************** END DEBUG *****************************************
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


  //************************** DEBUG ******************************************
  //    std::cout << "\nDelta_partial_beta: " << CORE::FADUTILS::CastToDouble( Delta_partial_beta );
  //    std::cout << "\nbeta_partial_cos_alpha: " << CORE::FADUTILS::CastToDouble(
  //    beta_partial_cos_alpha ); std::cout << "\nDelta_partial_cos_alpha: " <<
  //    CORE::FADUTILS::CastToDouble( Delta_partial_cos_alpha );
  //*********************** END DEBUG *****************************************


  // partial derivatives of single length specific interaction potential

  pot_ia_partial_beta = interaction_potential_GP / beta;

  pot_ia_partial_a = (m_ - 5.0) * interaction_potential_GP / a;

  pot_ia_partial_Delta = (-m_ + 4.5) * interaction_potential_GP / Delta_regularized;


  if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear and
      Delta < regularization_separation)
  {
    // partial derivative w.r.t. Delta

    // store this as an intermediate result that is needed below
    pot_ia_partial_Delta_atregsep = pot_ia_partial_Delta;

    //************************** DEBUG ******************************************
    //        std::cout << "\npot_ia_partial_Delta_atregsep: "
    //                  << CORE::FADUTILS::CastToDouble(pot_ia_partial_Delta_atregsep);
    //*********************** END DEBUG *****************************************

    pot_ia_partial_2ndderiv_Delta = (-m_ + 3.5) / Delta_regularized * pot_ia_partial_Delta_atregsep;

    //************************** DEBUG ******************************************
    //        std::cout << "\npot_ia_partial_2ndderiv_Delta_atregsep: "
    //                  << CORE::FADUTILS::CastToDouble(pot_ia_partial_2ndderiv_Delta);
    //*********************** END DEBUG *****************************************

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


    //************************** DEBUG ******************************************
    //        std::cout << ", pot_ia_partial_Delta: " <<
    //        CORE::FADUTILS::CastToDouble(pot_ia_partial_Delta);
    //*********************** END DEBUG *****************************************
  }

  pot_ia_partial_x = pot_ia_partial_beta * beta_partial_x + pot_ia_partial_a * a_partial_x +
                     pot_ia_partial_Delta * Delta_partial_x;

  pot_ia_partial_gap_bl =
      pot_ia_partial_beta * beta_partial_gap_bl + pot_ia_partial_a * a_partial_gap_bl +
      pot_ia_partial_Delta * Delta_partial_gap_bl - interaction_potential_GP / (gap_bl + radius2_);

  pot_ia_partial_cos_alpha = pot_ia_partial_beta * beta_partial_cos_alpha +
                             pot_ia_partial_a * a_partial_cos_alpha +
                             pot_ia_partial_Delta * Delta_partial_cos_alpha;


  if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear and
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
  if (regularization_type == INPAR::BEAMPOTENTIAL::regularization_linear and
      Delta < regularization_separation)
  {
    interaction_potential_GP +=
        pot_ia_partial_Delta_atregsep * (Delta - regularization_separation) +
        0.5 * pot_ia_partial_2ndderiv_Delta * (Delta - regularization_separation) *
            (Delta - regularization_separation);
  }


  // components from variation of bilateral gap
  T signum_dist_bl_tilde = CORE::FADUTILS::Signum(dist_ul.Dot(normal_bl));

  gap_bl_partial_r_slave.Update(signum_dist_bl_tilde, normal_bl);
  gap_bl_partial_r_master.Update(-1.0 * signum_dist_bl_tilde, normal_bl);

  // Todo: check and remove: the following term should vanish since r_xi_master \perp normal_bl
  gap_bl_partial_xi_master = -1.0 * signum_dist_bl_tilde * r_xi_master.Dot(normal_bl);

  v_mat_tmp.Clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= normal_bl(i) * normal_bl(j);
  }

  CORE::LINALG::Matrix<3, 1, T> vec_tmp(true);
  vec_tmp.Multiply(v_mat_tmp, dist_ul);


  if (alpha < BEAMSCOLINEARANGLETHRESHOLD)
  {
    // Todo: check and remove: signum_dist_bl_tilde should always be +1.0 in this case!

    gap_bl_partial_r_xi_slave.Clear();

    gap_bl_partial_r_slave.Update(signum_dist_bl_tilde / norm_normal_bl_tilde, vec_tmp, 1.0);

    gap_bl_partial_r_xi_master.Clear();

    gap_bl_partial_r_master.Update(
        -1.0 * signum_dist_bl_tilde / norm_normal_bl_tilde, vec_tmp, 1.0);

    // Todo: check and remove: the following term should vanish
    gap_bl_partial_xi_master -=
        signum_dist_bl_tilde / norm_normal_bl_tilde * r_xi_master.Dot(vec_tmp);
  }
  else
  {
    gap_bl_partial_r_xi_slave.Update(signum_dist_bl_tilde / norm_normal_bl_tilde,
        CORE::FADUTILS::VectorProduct(r_xi_master, vec_tmp));

    gap_bl_partial_r_xi_master.Update(-1.0 * signum_dist_bl_tilde / norm_normal_bl_tilde,
        CORE::FADUTILS::VectorProduct(r_xi_slave, vec_tmp));

    gap_bl_partial_xi_master += r_xixi_master.Dot(gap_bl_partial_r_xi_master);
  }


  // components from variation of cosine of enclosed angle
  T signum_tangentsscalarproduct = CORE::FADUTILS::Signum(r_xi_slave.Dot(r_xi_master));

  v_mat_tmp.Clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t1_slave(i) * t1_slave(j);
  }

  cos_alpha_partial_r_xi_slave.Multiply(
      signum_tangentsscalarproduct / CORE::FADUTILS::VectorNorm(r_xi_slave), v_mat_tmp, t1_master);

  v_mat_tmp.Clear();
  for (unsigned int i = 0; i < 3; ++i)
  {
    v_mat_tmp(i, i) += 1.0;
    for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t1_master(i) * t1_master(j);
  }

  cos_alpha_partial_r_xi_master.Multiply(
      signum_tangentsscalarproduct / CORE::FADUTILS::VectorNorm(r_xi_master), v_mat_tmp, t1_slave);

  cos_alpha_partial_xi_master = r_xixi_master.Dot(cos_alpha_partial_r_xi_master);


  // components from variation of distance from bilateral closest point on slave
  if (CORE::FADUTILS::CastToDouble(alpha) < BEAMSCOLINEARANGLETHRESHOLD)
  {
    /* set the following quantities to zero since they are not required in this case
     * because pot_ia_partial_x is zero anyway */
    x_partial_r_slave.Clear();
    x_partial_r_master.Clear();
    x_partial_r_xi_slave.Clear();
    x_partial_r_xi_master.Clear();
    x_partial_xi_master = 0.0;
  }
  else
  {
    x_partial_r_slave.Update(-1.0 / t1_slave.Dot(aux_plane_normal), aux_plane_normal);
    x_partial_r_master.Update(1.0 / t1_slave.Dot(aux_plane_normal), aux_plane_normal);

    v_mat_tmp.Clear();
    for (unsigned int i = 0; i < 3; ++i)
    {
      v_mat_tmp(i, i) += 1.0;
      for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) -= t1_slave(i) * t1_slave(j);
    }

    x_partial_r_xi_slave.Multiply(1.0 / CORE::FADUTILS::VectorNorm(r_xi_slave) *
                                      dist_ul.Dot(aux_plane_normal) /
                                      std::pow(t1_slave.Dot(aux_plane_normal), 2),
        v_mat_tmp, aux_plane_normal);

    x_partial_aux_plane_normal.Update(-1.0 / t1_slave.Dot(aux_plane_normal), dist_ul,
        dist_ul.Dot(aux_plane_normal) / std::pow(t1_slave.Dot(aux_plane_normal), 2), t1_slave);

    v_mat_tmp.Clear();
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j) v_mat_tmp(i, j) += r_xi_master(i) * r_xi_master(j);

    x_partial_r_xi_slave.Update(r_xi_master.Dot(r_xi_master), x_partial_aux_plane_normal, 1.0);

    x_partial_r_xi_slave.Multiply(-1.0, v_mat_tmp, x_partial_aux_plane_normal, 1.0);


    x_partial_r_xi_master.Update(-1.0 * r_xi_master.Dot(r_xi_slave), x_partial_aux_plane_normal);

    v_mat_tmp.Clear();
    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
      {
        v_mat_tmp(i, j) += 2.0 * r_xi_master(i) * r_xi_slave(j);
        v_mat_tmp(i, j) -= 1.0 * r_xi_slave(i) * r_xi_master(j);
      }

    x_partial_r_xi_master.Multiply(1.0, v_mat_tmp, x_partial_aux_plane_normal, 1.0);

    x_partial_xi_master = r_xixi_master.Dot(x_partial_r_xi_master);
  }


  // store for vtk visualization
  vtk_force_pot_slave_GP.Update(
      prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(gap_bl_partial_r_slave), 1.0);

  vtk_force_pot_slave_GP.UpdateT(
      prefactor_visualization_data *
          CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl * gap_bl_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_slave), 1.0);

  vtk_force_pot_slave_GP.UpdateT(
      prefactor_visualization_data *
          CORE::FADUTILS::CastToDouble(pot_ia_partial_cos_alpha * cos_alpha_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_slave), 1.0);

  vtk_force_pot_slave_GP.Update(
      prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_partial_x),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(x_partial_r_slave), 1.0);

  vtk_force_pot_slave_GP.UpdateT(
      prefactor_visualization_data *
          CORE::FADUTILS::CastToDouble(pot_ia_partial_x * x_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_slave), 1.0);

  CORE::LINALG::Matrix<3, 1, double> moment_pot_tmp(true);

  moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(gap_bl_partial_r_xi_slave));

  moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_partial_cos_alpha),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(cos_alpha_partial_r_xi_slave), 1.0);

  moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_partial_x),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(x_partial_r_xi_slave), 1.0);

  /* note: relation between variation of tangent vector r_xi and variation of (transversal part
   *       of) rotation vector theta_perp describing cross-section orientation can be used to
   *       identify (distributed) moments as follows: m_pot = 1/|r_xi| * ( m_pot_pseudo x g1 )
   */
  CORE::LINALG::Matrix<3, 3, double> spin_pseudo_moment_tmp(true);

  CORE::LARGEROTATIONS::computespin(spin_pseudo_moment_tmp, moment_pot_tmp);

  T norm_r_xi_slave = CORE::FADUTILS::VectorNorm(r_xi_slave);

  moment_pot_tmp.Multiply(1.0 / CORE::FADUTILS::CastToDouble(norm_r_xi_slave),
      spin_pseudo_moment_tmp, CORE::FADUTILS::CastToDouble<T, 3, 1>(t1_slave));

  vtk_moment_pot_slave_GP.Update(prefactor_visualization_data, moment_pot_tmp, 1.0);


  vtk_force_pot_master_GP.Update(
      prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(gap_bl_partial_r_master), 1.0);

  vtk_force_pot_master_GP.UpdateT(
      prefactor_visualization_data *
          CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl * gap_bl_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_master), 1.0);

  vtk_force_pot_master_GP.UpdateT(
      prefactor_visualization_data *
          CORE::FADUTILS::CastToDouble(pot_ia_partial_cos_alpha * cos_alpha_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_master), 1.0);

  vtk_force_pot_master_GP.Update(
      prefactor_visualization_data * CORE::FADUTILS::CastToDouble(pot_ia_partial_x),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(x_partial_r_master), 1.0);

  vtk_force_pot_master_GP.UpdateT(
      prefactor_visualization_data *
          CORE::FADUTILS::CastToDouble(pot_ia_partial_x * x_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_master), 1.0);


  moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(gap_bl_partial_r_xi_master));

  moment_pot_tmp.UpdateT(
      CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl * gap_bl_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_xi_master), 1.0);

  moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_partial_cos_alpha),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(cos_alpha_partial_r_xi_master), 1.0);

  moment_pot_tmp.UpdateT(
      CORE::FADUTILS::CastToDouble(pot_ia_partial_cos_alpha * cos_alpha_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_xi_master), 1.0);

  moment_pot_tmp.Update(CORE::FADUTILS::CastToDouble(pot_ia_partial_x),
      CORE::FADUTILS::CastToDouble<T, 3, 1>(x_partial_r_xi_master), 1.0);

  moment_pot_tmp.UpdateT(CORE::FADUTILS::CastToDouble(pot_ia_partial_x * x_partial_xi_master),
      CORE::FADUTILS::CastToDouble<T, 1, 3>(xi_master_partial_r_xi_master), 1.0);

  CORE::LARGEROTATIONS::computespin(spin_pseudo_moment_tmp, moment_pot_tmp);

  T norm_r_xi_master = CORE::FADUTILS::VectorNorm(r_xi_master);

  moment_pot_tmp.Multiply(1.0 / CORE::FADUTILS::CastToDouble(norm_r_xi_master),
      spin_pseudo_moment_tmp, CORE::FADUTILS::CastToDouble<T, 3, 1>(t1_master));

  vtk_moment_pot_master_GP.Update(prefactor_visualization_data, moment_pot_tmp, 1.0);


  // integration factor
  pot_ia_partial_gap_bl *= rho1rho2_JacFac_GaussWeight;
  pot_ia_partial_cos_alpha *= rho1rho2_JacFac_GaussWeight;
  pot_ia_partial_x *= rho1rho2_JacFac_GaussWeight;

  //************************** DEBUG ******************************************
  //      std::cout << "\nprefactor: " << CORE::FADUTILS::CastToDouble(prefactor);
  //      std::cout << "\nrho1rho2_JacFac_GaussWeight: "
  //                << CORE::FADUTILS::CastToDouble(rho1rho2_JacFac_GaussWeight);
  //
  //      std::cout << "\npot_ia_partial_x: " << CORE::FADUTILS::CastToDouble(pot_ia_partial_x);
  //      std::cout << "\npot_ia_partial_gap_bl: " <<
  //      CORE::FADUTILS::CastToDouble(pot_ia_partial_gap_bl); std::cout <<
  //      "\npot_ia_partial_cos_alpha: "
  //                << CORE::FADUTILS::CastToDouble(pot_ia_partial_cos_alpha);
  //*********************** END DEBUG *****************************************


  //********************************************************************
  // calculate fpot1: force/residual vector on slave element
  //********************************************************************
  force_pot_slave_GP.Clear();
  // sum up the contributions of all nodes (in all dimensions)
  for (unsigned int idofperdim = 0; idofperdim < (numnodes * numnodalvalues); ++idofperdim)
  {
    // loop over dimensions
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
    {
      force_pot_slave_GP(3 * idofperdim + jdim) +=
          N_i_slave(idofperdim) * pot_ia_partial_gap_bl *
          (gap_bl_partial_r_slave(jdim) +
              gap_bl_partial_xi_master * xi_master_partial_r_slave(jdim));

      force_pot_slave_GP(3 * idofperdim + jdim) +=
          N_i_xi_slave(idofperdim) * pot_ia_partial_gap_bl * gap_bl_partial_r_xi_slave(jdim);


      force_pot_slave_GP(3 * idofperdim + jdim) +=
          N_i_slave(idofperdim) * pot_ia_partial_cos_alpha * cos_alpha_partial_xi_master *
          xi_master_partial_r_slave(jdim);

      force_pot_slave_GP(3 * idofperdim + jdim) +=
          N_i_xi_slave(idofperdim) * pot_ia_partial_cos_alpha * cos_alpha_partial_r_xi_slave(jdim);


      force_pot_slave_GP(3 * idofperdim + jdim) +=
          N_i_slave(idofperdim) * pot_ia_partial_x *
          (x_partial_r_slave(jdim) + x_partial_xi_master * xi_master_partial_r_slave(jdim));

      force_pot_slave_GP(3 * idofperdim + jdim) +=
          N_i_xi_slave(idofperdim) * pot_ia_partial_x * x_partial_r_xi_slave(jdim);
    }
  }

  //********************************************************************
  // calculate fpot2: force/residual vector on master element
  //********************************************************************
  force_pot_master_GP.Clear();
  // sum up the contributions of all nodes (in all dimensions)
  for (unsigned int idofperdim = 0; idofperdim < (numnodes * numnodalvalues); ++idofperdim)
  {
    // loop over dimensions
    for (unsigned int jdim = 0; jdim < 3; ++jdim)
    {
      force_pot_master_GP(3 * idofperdim + jdim) +=
          N_i_master(idofperdim) * pot_ia_partial_gap_bl *
          (gap_bl_partial_r_master(jdim) +
              gap_bl_partial_xi_master * xi_master_partial_r_master(jdim));

      force_pot_master_GP(3 * idofperdim + jdim) +=
          N_i_xi_master(idofperdim) * pot_ia_partial_gap_bl *
          (gap_bl_partial_r_xi_master(jdim) +
              gap_bl_partial_xi_master * xi_master_partial_r_xi_master(jdim));


      force_pot_master_GP(3 * idofperdim + jdim) +=
          N_i_master(idofperdim) * pot_ia_partial_cos_alpha * cos_alpha_partial_xi_master *
          xi_master_partial_r_master(jdim);

      force_pot_master_GP(3 * idofperdim + jdim) +=
          N_i_xi_master(idofperdim) * pot_ia_partial_cos_alpha *
          (cos_alpha_partial_r_xi_master(jdim) +
              cos_alpha_partial_xi_master * xi_master_partial_r_xi_master(jdim));


      force_pot_master_GP(3 * idofperdim + jdim) +=
          N_i_master(idofperdim) * pot_ia_partial_x *
          (x_partial_r_master(jdim) + x_partial_xi_master * xi_master_partial_r_master(jdim));

      force_pot_master_GP(3 * idofperdim + jdim) +=
          N_i_xi_master(idofperdim) * pot_ia_partial_x *
          (x_partial_r_xi_master(jdim) + x_partial_xi_master * xi_master_partial_r_xi_master(jdim));
    }
  }

  return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::ScaleStiffpotAnalyticContributionsIfRequired(double const& scalefactor,
    CORE::LINALG::SerialDenseMatrix& stiffmat11, CORE::LINALG::SerialDenseMatrix& stiffmat12,
    CORE::LINALG::SerialDenseMatrix& stiffmat21, CORE::LINALG::SerialDenseMatrix& stiffmat22) const
{
  stiffmat11.scale(scalefactor);
  stiffmat12.scale(scalefactor);
  stiffmat21.scale(scalefactor);
  stiffmat22.scale(scalefactor);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    CalcStiffmatAutomaticDifferentiationIfRequired(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot1,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot2,
        CORE::LINALG::SerialDenseMatrix& stiffmat11, CORE::LINALG::SerialDenseMatrix& stiffmat12,
        CORE::LINALG::SerialDenseMatrix& stiffmat21,
        CORE::LINALG::SerialDenseMatrix& stiffmat22) const
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
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    AddStiffmatContributionsXiMasterAutomaticDifferentiationIfRequired(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot1,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>> const&
            force_pot2,
        CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_master_slaveDofs,
        CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_master_masterDofs,
        CORE::LINALG::SerialDenseMatrix& stiffmat11, CORE::LINALG::SerialDenseMatrix& stiffmat12,
        CORE::LINALG::SerialDenseMatrix& stiffmat21,
        CORE::LINALG::SerialDenseMatrix& stiffmat22) const
{
  const unsigned int dim = 3 * numnodes * numnodalvalues;

  for (unsigned int idof = 0; idof < dim; ++idof)
  {
    for (unsigned int jdof = 0; jdof < dim; ++jdof)
    {
      // d (Res_1) / d (d_1)
      stiffmat11(idof, jdof) += force_pot1(idof).dx(2 * dim) *
                                CORE::FADUTILS::CastToDouble(lin_xi_master_slaveDofs(jdof));

      // d (Res_1) / d (d_2)
      stiffmat12(idof, jdof) += force_pot1(idof).dx(2 * dim) *
                                CORE::FADUTILS::CastToDouble(lin_xi_master_masterDofs(jdof));

      // d (Res_2) / d (d_1)
      stiffmat21(idof, jdof) += force_pot2(idof).dx(2 * dim) *
                                CORE::FADUTILS::CastToDouble(lin_xi_master_slaveDofs(jdof));

      // d (Res_2) / d (d_2)
      stiffmat22(idof, jdof) += force_pot2(idof).dx(2 * dim) *
                                CORE::FADUTILS::CastToDouble(lin_xi_master_masterDofs(jdof));
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    CalcFpotGausspointAutomaticDifferentiationIfRequired(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, double>& force_pot1,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, double>& force_pot2,
        Sacado::Fad::DFad<double> const& interaction_potential,
        CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_master_slaveDofs,
        CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_master_masterDofs) const
{
  const unsigned int dim = 3 * numnodalvalues * numnodes;

  for (unsigned int idof = 0; idof < dim; ++idof)
  {
    force_pot1(idof) = interaction_potential.dx(idof) +
                       interaction_potential.dx(2 * dim) *
                           CORE::FADUTILS::CastToDouble(lin_xi_master_slaveDofs(idof));

    force_pot2(idof) = interaction_potential.dx(dim + idof) +
                       interaction_potential.dx(2 * dim) *
                           CORE::FADUTILS::CastToDouble(lin_xi_master_masterDofs(idof));
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    CalcFpotGausspointAutomaticDifferentiationIfRequired(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            force_pot1,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            force_pot2,
        Sacado::Fad::DFad<double> const& interaction_potential,
        CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_master_slaveDofs,
        CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, Sacado::Fad::DFad<double>> const&
            lin_xi_master_masterDofs) const
{
  const unsigned int dim = 3 * numnodalvalues * numnodes;

  for (unsigned int idof = 0; idof < dim; ++idof)
  {
    force_pot1(idof) = interaction_potential.dx(idof) +
                       interaction_potential.dx(2 * dim) * lin_xi_master_slaveDofs(idof);

    force_pot2(idof) = interaction_potential.dx(dim + idof) +
                       interaction_potential.dx(2 * dim) * lin_xi_master_masterDofs(idof);
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::Print(
    std::ostream& out) const
{
  CheckInitSetup();

  out << "\nInstance of BeamToBeamPotentialPair (EleGIDs " << Element1()->Id() << " & "
      << Element2()->Id() << "):";
  out << "\nele1 dofvec: "
      << CORE::FADUTILS::CastToDouble<T, 3 * numnodes * numnodalvalues, 1>(ele1pos_);
  out << "\nele2 dofvec: "
      << CORE::FADUTILS::CastToDouble<T, 3 * numnodes * numnodalvalues, 1>(ele2pos_);

  out << "\n";
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::PrintSummaryOneLinePerActiveSegmentPair(std::ostream& out) const
{
  CheckInitSetup();

  // Todo difficulty here is that the same element pair is evaluated more than once
  //      to be more precise, once for every common potlaw;
  //      contribution of previous evaluations is overwritten if multiple potlaws are applied
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::GetShapeFunctions(
    std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>>& N1_i,
    std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>>& N2_i,
    std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>>& N1_i_xi,
    std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues, double>>& N2_i_xi,
    CORE::FE::IntegrationPoints1D& gausspoints) const
{
  DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAllGPs<numnodes, numnodalvalues>(
      gausspoints, N1_i, N1_i_xi, BeamElement1()->Shape(), ele1length_);

  DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAllGPs<numnodes, numnodalvalues>(
      gausspoints, N2_i, N2_i_xi, BeamElement2()->Shape(), ele2length_);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/

template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
template <typename T2>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::ComputeCenterlinePosition(CORE::LINALG::Matrix<3, 1, T>& r,
    const CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T2>& N_i,
    const CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T> eledofvec) const
{
  DRT::UTILS::BEAM::CalcInterpolation<numnodes, numnodalvalues, 3, T, T2>(eledofvec, N_i, r);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
template <typename T2>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::ComputeCenterlineTangent(CORE::LINALG::Matrix<3, 1, T>& r_xi,
    const CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T2>& N_i_xi,
    const CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T> eledofvec) const
{
  DRT::UTILS::BEAM::CalcInterpolation<numnodes, numnodalvalues, 3, T, T2>(eledofvec, N_i_xi, r_xi);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::ResetState(double time,
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

  for (auto& forcevec : forces_pot_gp_1_) forcevec.Clear();
  for (auto& forcevec : forces_pot_gp_2_) forcevec.Clear();
  for (auto& momentvec : moments_pot_gp_1_) momentvec.Clear();
  for (auto& momentvec : moments_pot_gp_2_) momentvec.Clear();
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    SetAutomaticDifferentiationVariablesIfRequired(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele1centerlinedofvec,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
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
void BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues, T>::
    SetAutomaticDifferentiationVariablesIfRequired(
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele1centerlinedofvec,
        CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, Sacado::Fad::DFad<double>>&
            ele2centerlinedofvec,
        Sacado::Fad::DFad<double>& xi_master)
{
  // The 2*3*numnodes*numnodalvalues primary DoFs consist of all nodal positions and tangents
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele1centerlinedofvec(i).diff(i, 2 * 3 * numnodes * numnodalvalues + 1);

  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele2centerlinedofvec(i).diff(
        3 * numnodes * numnodalvalues + i, 2 * 3 * numnodes * numnodalvalues + 1);

  // Additionally, we set the parameter coordinate on the master side xi_master as primary Dof
  xi_master.diff(2 * 3 * numnodes * numnodalvalues, 2 * 3 * numnodes * numnodalvalues + 1);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BEAMINTERACTION::BeamToBeamPotentialPair<numnodes, numnodalvalues,
    T>::AreElementsMuchMoreSeparatedThanCutoffDistance()
{
  const unsigned int num_spatial_dim = 3;

  // multiple of the cutoff radius to be considered as "far enough" apart
  const double safety_factor = 2.0;

  // extract the current position of both elements' boundary nodes
  CORE::LINALG::Matrix<num_spatial_dim, 1> position_element1node1(true);
  CORE::LINALG::Matrix<num_spatial_dim, 1> position_element1node2(true);
  CORE::LINALG::Matrix<num_spatial_dim, 1> position_element2node1(true);
  CORE::LINALG::Matrix<num_spatial_dim, 1> position_element2node2(true);

  for (unsigned int i_dim = 0; i_dim < num_spatial_dim; ++i_dim)
  {
    position_element1node1(i_dim) = CORE::FADUTILS::CastToDouble(ele1pos_(0 + i_dim));
    position_element1node2(i_dim) =
        CORE::FADUTILS::CastToDouble(ele1pos_(num_spatial_dim * 1 * numnodalvalues + i_dim));

    position_element2node1(i_dim) = CORE::FADUTILS::CastToDouble(ele2pos_(0 + i_dim));
    position_element2node2(i_dim) =
        CORE::FADUTILS::CastToDouble(ele2pos_(num_spatial_dim * 1 * numnodalvalues + i_dim));
  }


  // the following rough estimate of the elements' separation is based on spherical bounding boxes
  CORE::LINALG::Matrix<num_spatial_dim, 1> spherical_boxes_midpoint_distance(true);

  spherical_boxes_midpoint_distance.Update(
      0.5, position_element1node1, 0.5, position_element1node2);

  spherical_boxes_midpoint_distance.Update(
      -0.5, position_element2node1, -0.5, position_element2node2, 1.0);


  CORE::LINALG::Matrix<num_spatial_dim, 1> element1_boundary_nodes_distance(true);
  element1_boundary_nodes_distance.Update(
      1.0, position_element1node1, -1.0, position_element1node2);
  double element1_spherical_box_radius = 0.5 * element1_boundary_nodes_distance.Norm2();

  CORE::LINALG::Matrix<num_spatial_dim, 1> element2_boundary_nodes_distance(true);
  element2_boundary_nodes_distance.Update(
      1.0, position_element2node1, -1.0, position_element2node2);
  double element2_spherical_box_radius = 0.5 * element2_boundary_nodes_distance.Norm2();

  double estimated_minimal_centerline_separation = spherical_boxes_midpoint_distance.Norm2() -
                                                   element1_spherical_box_radius -
                                                   element2_spherical_box_radius;


  if (estimated_minimal_centerline_separation > safety_factor * Params()->CutoffRadius())
    return true;
  else
    return false;
}

// explicit template instantiations
template class BEAMINTERACTION::BeamToBeamPotentialPair<2, 1, double>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<2, 1, Sacado::Fad::DFad<double>>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<3, 1, double>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<3, 1, Sacado::Fad::DFad<double>>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<4, 1, double>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<4, 1, Sacado::Fad::DFad<double>>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<5, 1, double>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<5, 1, Sacado::Fad::DFad<double>>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<2, 2, double>;
template class BEAMINTERACTION::BeamToBeamPotentialPair<2, 2, Sacado::Fad::DFad<double>>;

FOUR_C_NAMESPACE_CLOSE
