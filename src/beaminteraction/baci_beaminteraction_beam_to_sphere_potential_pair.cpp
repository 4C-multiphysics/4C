/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief One beam-to-sphere potential-based interacting pair

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/

#include "baci_beaminteraction_beam_to_sphere_potential_pair.H"

#include "baci_beam3_base.H"
#include "baci_beaminteraction_beam_to_beam_contact_defines.H"
#include "baci_beaminteraction_beam_to_beam_contact_utils.H"
#include "baci_beaminteraction_potential_params.H"
#include "baci_discretization_fem_general_utils_fem_shapefunctions.H"
#include "baci_inpar_beampotential.H"
#include "baci_inpar_contact.H"
#include "baci_lib_discret.H"
#include "baci_lib_exporter.H"
#include "baci_lib_function_of_time.H"
#include "baci_lib_globalproblem.H"
#include "baci_linalg_serialdensematrix.H"
#include "baci_linalg_serialdensevector.H"
#include "baci_linalg_utils_sparse_algebra_math.H"
#include "baci_rigidsphere.H"
#include "baci_utils_exceptions.H"
#include "baci_utils_fad.H"

#include <Teuchos_TimeMonitor.hpp>

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::BeamToSpherePotentialPair()
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
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::Setup()
{
  CheckInit();

  // call setup of base class first
  BeamPotentialPair::Setup();


  ele1pos_.Clear();
  ele2pos_.Clear();

  fpot1_.Clear();
  fpot2_.Clear();
  stiffpot1_.Clear();
  stiffpot2_.Clear();


  // cast first element to Beam3Base
  beam_element_ = dynamic_cast<const DRT::ELEMENTS::Beam3Base*>(Element1());

  if (beam_element_ == nullptr)
    dserror(
        "cast to Beam3Base failed! first element in BeamToSpherePotentialPair pair"
        " must be a beam element!");

  // get radius and stress-free reference length of beam element
  radius1_ = BEAMINTERACTION::CalcEleRadius(beam_element_);
  beamele_reflength_ = beam_element_->RefLength();

  // cast second element to RigidSphere
  sphere_element_ = dynamic_cast<const DRT::ELEMENTS::Rigidsphere*>(Element2());

  if (sphere_element_ == nullptr)
    dserror(
        "cast to Rigidsphere failed! second element in BeamToSpherePotentialPair pair"
        " must be a Rigidsphere element!");

  // get radius of sphere element
  radius2_ = sphere_element_->Radius();

  // initialize charge conditions applied to beam and sphere elements
  chargeconds_.resize(2);

  issetup_ = true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
bool BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::Evaluate(
    CORE::LINALG::SerialDenseVector* forcevec1, CORE::LINALG::SerialDenseVector* forcevec2,
    CORE::LINALG::SerialDenseMatrix* stiffmat11, CORE::LINALG::SerialDenseMatrix* stiffmat12,
    CORE::LINALG::SerialDenseMatrix* stiffmat21, CORE::LINALG::SerialDenseMatrix* stiffmat22,
    const std::vector<DRT::Condition*> chargeconds, const double k, const double m)
{
  // nothing to do in case of k==0.0
  if (k == 0.0) return false;

  // reset fpot and stiffpot class variables
  fpot1_.Clear();
  fpot2_.Clear();
  stiffpot1_.Clear();
  stiffpot2_.Clear();

  unsigned int dim1 = 3 * numnodes * numnodalvalues;
  unsigned int dim2 = 3;

  // set class variables
  if (chargeconds.size() == 2)
  {
    if (chargeconds[0]->Type() == DRT::Condition::BeamPotential_LineChargeDensity)
      chargeconds_[0] = chargeconds[0];
    else
      dserror("Provided condition is not of correct type BeamPotential_LineChargeDensity!");

    if (chargeconds[1]->Type() == DRT::Condition::RigidspherePotential_PointCharge)
      chargeconds_[1] = chargeconds[1];
    else
      dserror("Provided condition is not of correct type RigidspherePotential_PointCharge!");
  }
  else
    dserror(
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
  switch (Params()->Strategy())
  {
    case INPAR::BEAMPOTENTIAL::strategy_doublelengthspec_largesepapprox:
    {
      EvaluateFpotandStiffpot_LargeSepApprox();
      break;
    }

    default:
      dserror("Invalid strategy to evaluate beam-to-sphere interaction potential!");
  }

  // resize variables and fill with pre-computed values
  if (forcevec1 != nullptr)
  {
    forcevec1->size(dim1);
    for (unsigned int i = 0; i < dim1; ++i)
      (*forcevec1)(i) = CORE::FADUTILS::CastToDouble(fpot1_(i));
  }
  if (forcevec2 != nullptr)
  {
    forcevec2->size(dim2);
    for (unsigned int i = 0; i < dim2; ++i)
      (*forcevec2)(i) = CORE::FADUTILS::CastToDouble(fpot2_(i));
  }

  if (stiffmat11 != nullptr)
  {
    stiffmat11->shape(dim1, dim1);
    for (unsigned int irow = 0; irow < dim1; ++irow)
      for (unsigned int icol = 0; icol < dim1; ++icol)
        (*stiffmat11)(irow, icol) = CORE::FADUTILS::CastToDouble(stiffpot1_(irow, icol));
  }
  if (stiffmat12 != nullptr)
  {
    stiffmat12->shape(dim1, dim2);
    for (unsigned int irow = 0; irow < dim1; ++irow)
      for (unsigned int icol = 0; icol < dim2; ++icol)
        (*stiffmat12)(irow, icol) = CORE::FADUTILS::CastToDouble(stiffpot1_(irow, dim1 + icol));
  }
  if (stiffmat21 != nullptr)
  {
    stiffmat21->shape(dim2, dim1);
    for (unsigned int irow = 0; irow < dim2; ++irow)
      for (unsigned int icol = 0; icol < dim1; ++icol)
        (*stiffmat21)(irow, icol) = CORE::FADUTILS::CastToDouble(stiffpot2_(irow, icol));
  }
  if (stiffmat22 != nullptr)
  {
    stiffmat22->shape(dim2, dim2);
    for (unsigned int irow = 0; irow < dim2; ++irow)
      for (unsigned int icol = 0; icol < dim2; ++icol)
        (*stiffmat22)(irow, icol) = CORE::FADUTILS::CastToDouble(stiffpot2_(irow, dim1 + icol));
  }

  return (true);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes,
    numnodalvalues>::EvaluateFpotandStiffpot_LargeSepApprox()
{
  // get cutoff radius
  const double cutoff_radius = Params()->CutoffRadius();

  // Set gauss integration rule
  CORE::DRT::UTILS::GaussRule1D gaussrule = GetGaussRule();

  // Get gauss points (gp) for integration
  CORE::DRT::UTILS::IntegrationPoints1D gausspoints(gaussrule);
  // number of gps
  const int numgp = gausspoints.nquad;

  // vectors for shape functions and their derivatives
  // Attention: these are individual shape function values, NOT shape function matrices
  // values at all gauss points are stored in advance
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues>> N1_i(numgp);     // = N1_i
  std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues>> N1_i_xi(numgp);  // = N1_i,xi

  // coords and derivatives of the two gauss points
  CORE::LINALG::Matrix<3, 1, TYPE> r1(true);    // = r1
  CORE::LINALG::Matrix<3, 1, TYPE> r2(true);    // = r2
  CORE::LINALG::Matrix<3, 1, TYPE> dist(true);  // = r1-r2
  TYPE norm_dist = 0.0;                         // = |r1-r2|

  // Evaluate shape functions at gauss points and store values
  GetShapeFunctions(N1_i, N1_i_xi, gausspoints);

  // evaluate charge density from DLINE charge condition specified in input file
  double q1 = chargeconds_[0]->GetDouble("val");

  // read charge of rigid sphere; note: this is NOT a charge density but the total charge of the
  // sphere!!!
  double q2 = chargeconds_[1]->GetDouble("val");

  // evaluate function in time if specified in line charge conditions
  // TODO allow for functions in space, i.e. varying charge along beam centerline
  int function_number = chargeconds_[0]->GetInt("funct");

  if (function_number != -1)
    q1 *= DRT::Problem::Instance()
              ->FunctionById<DRT::UTILS::FunctionOfTime>(function_number - 1)
              .Evaluate(time_);

  function_number = chargeconds_[1]->GetInt("funct");

  if (function_number != -1)
    q2 *= DRT::Problem::Instance()
              ->FunctionById<DRT::UTILS::FunctionOfTime>(function_number - 1)
              .Evaluate(time_);


  // auxiliary variable
  CORE::LINALG::Matrix<3, 1, TYPE> fpot_tmp(true);

  // determine prefactor of the integral (depends on whether surface or volume potential is applied)
  double prefactor = k_ * m_;

  switch (Params()->PotentialType())  // Todo do we need a own Beam-to-sphere potential type here?
  {
    case INPAR::BEAMPOTENTIAL::beampot_surf:
      prefactor *= 2 * radius1_ * M_PI;
      break;
    case INPAR::BEAMPOTENTIAL::beampot_vol:
      prefactor *= std::pow(radius1_, 2) * M_PI;
      break;
    default:
      dserror(
          "No valid BEAMPOTENTIAL_TYPE specified. Choose either Surface or Volume in input file!");
  }

  // get sphere midpoint position
  for (int i = 0; i < 3; ++i) r2(i) = ele2pos_(i);

  // reset interaction potential of this pair
  interaction_potential_ = 0.0;

  // loop over gauss points on ele1
  for (int gp1 = 0; gp1 < numgp; ++gp1)
  {
    ComputeCoords(r1, N1_i[gp1], ele1pos_);

    dist = CORE::FADUTILS::DiffVector(r1, r2);

    norm_dist = CORE::FADUTILS::VectorNorm<3>(dist);

    // check cutoff criterion: if specified, contributions are neglected at larger separation
    if (cutoff_radius != -1.0 and CORE::FADUTILS::CastToDouble(norm_dist) > cutoff_radius) continue;


    // temporary hacks for cell-ecm interaction
    //    // get radius of rigid sphere element
    //    double radius = 4.0;//sphere_element_->Radius();
    //    double deltaradius = radius/10.0;
    //    if(
    //        norm_dist < (radius-2.0*deltaradius)
    //        or
    //        norm_dist > (radius +2.0* deltaradius)
    //        )
    //    {
    //      fpot1_.PutScalar(0.0);
    //      fpot2_.PutScalar(0.0);
    //      stiffpot1_.PutScalar(0.0);
    //      stiffpot2_.PutScalar(0.0);;
    //      return;
    //    }
    //    else if (
    //        norm_dist > ( radius - 2.0 * deltaradius)
    //        and
    //        norm_dist < ( radius + 2.0 * deltaradius)
    //        )
    //    if( norm_dist < sphere_element_->Radius() )
    //    {
    //      dist.Scale( sphere_element_->Radius() / norm_dist );
    //      norm_dist = CORE::FADUTILS::VectorNorm<3>(dist);
    //    }
    //
    //    if(norm_dist > 0.5)
    //      dist.Scale(10.0/norm_dist);
    //    norm_dist = CORE::FADUTILS::VectorNorm<3>(dist);

    // auxiliary variables to store pre-calculated common terms
    TYPE norm_dist_exp1 = 0.0;
    if (norm_dist != 0.0)
    {
      norm_dist_exp1 = std::pow(norm_dist, -m_ - 2);
    }
    else
    {
      dserror(
          "\n|r1-r2|=0 ! Interacting points are identical! Potential law not defined in this case!"
          " Think about shifting nodes in unconverged state?!");
    }

    double q1q2_JacFac_GaussWeights =
        q1 * q2 * BeamElement()->GetJacobiFacAtXi(gausspoints.qxg[gp1][0]) * gausspoints.qwgt[gp1];

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

    CORE::LINALG::Matrix<3, 3, TYPE> dist_dist_T(true);

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
                              std::pow(CORE::FADUTILS::CastToDouble(norm_dist), -m_);
  }


  // apply constant prefactor
  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i) fpot1_(i) *= prefactor;
  for (unsigned int i = 0; i < 3; ++i) fpot2_(i) *= prefactor;

  stiffpot1_.Scale(prefactor);
  stiffpot2_.Scale(prefactor);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::Print(
    std::ostream& out) const
{
  CheckInitSetup();

  out << "\nInstance of BeamToSpherePotentialPair (EleGIDs " << Element1()->Id() << " & "
      << Element2()->Id() << "):";
  out << "\nbeamele dofvec: " << ele1pos_;
  out << "\nspherele dofvec: " << ele2pos_;

  out << "\n";
  // Todo add more relevant information here
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes,
    numnodalvalues>::PrintSummaryOneLinePerActiveSegmentPair(std::ostream& out) const
{
  CheckInitSetup();

  // Todo difficulty here is that the same element pair is evaluated more than once
  //      to be more precise, once for every common potlaw;
  //      contribution of previous evaluations is overwritten if multiple potlaws are applied
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::GetShapeFunctions(
    std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues>>& N1_i,
    std::vector<CORE::LINALG::Matrix<1, numnodes * numnodalvalues>>& N1_i_xi,
    CORE::DRT::UTILS::IntegrationPoints1D& gausspoints)
{
  // get discretization type
  const DRT::Element::DiscretizationType distype1 = Element1()->Shape();

  if (numnodalvalues == 1)
  {
    for (int gp = 0; gp < gausspoints.nquad; ++gp)
    {
      // get values and derivatives of shape functions
      CORE::DRT::UTILS::shape_function_1D(N1_i[gp], gausspoints.qxg[gp][0], distype1);
      CORE::DRT::UTILS::shape_function_1D_deriv1(N1_i_xi[gp], gausspoints.qxg[gp][0], distype1);
    }
  }
  else if (numnodalvalues == 2)
  {
    /* TODO hard set distype to line2 in case of numnodalvalues_=2 because
     *  only 3rd order Hermite interpolation is used (always 2 nodes) */
    const DRT::Element::DiscretizationType distype1herm = DRT::Element::DiscretizationType::line2;

    for (int gp = 0; gp < gausspoints.nquad; ++gp)
    {
      // get values and derivatives of shape functions
      CORE::DRT::UTILS::shape_function_hermite_1D(
          N1_i[gp], gausspoints.qxg[gp][0], beamele_reflength_, distype1herm);
      CORE::DRT::UTILS::shape_function_hermite_1D_deriv1(
          N1_i_xi[gp], gausspoints.qxg[gp][0], beamele_reflength_, distype1herm);
    }
  }
  else
    dserror(
        "Only beam elements with one (nodal positions) or two (nodal positions + nodal tangents)"
        " values are valid!");

  return;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues>
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::ComputeCoords(
    CORE::LINALG::Matrix<3, 1, TYPE>& r,
    const CORE::LINALG::Matrix<1, numnodes * numnodalvalues>& N_i,
    const CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, TYPE> elepos)
{
  r.Clear();

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
void BEAMINTERACTION::BeamToSpherePotentialPair<numnodes, numnodalvalues>::ResetState(double time,
    const std::vector<double>& centerline_dofvec_ele1,
    const std::vector<double>& centerline_dofvec_ele2)
{
  time_ = time;

  if (centerline_dofvec_ele1.size() != 3 * numnodes * numnodalvalues)
    dserror("size mismatch! expected %d values for centerline_dofvec_ele1, but got %d",
        3 * numnodes * numnodalvalues, centerline_dofvec_ele1.size());

  if (centerline_dofvec_ele2.size() != 3)
    dserror("size mismatch! expected %d values for centerline_dofvec_ele2, but got %d", 3,
        centerline_dofvec_ele1.size());


  for (unsigned int i = 0; i < 3 * numnodes * numnodalvalues; ++i)
    ele1pos_(i) = centerline_dofvec_ele1[i];

  for (unsigned int i = 0; i < 3; ++i) ele2pos_(i) = centerline_dofvec_ele2[i];
}


// Possible template cases: this is necessary for the compiler
template class BEAMINTERACTION::BeamToSpherePotentialPair<2, 1>;
template class BEAMINTERACTION::BeamToSpherePotentialPair<3, 1>;
template class BEAMINTERACTION::BeamToSpherePotentialPair<4, 1>;
template class BEAMINTERACTION::BeamToSpherePotentialPair<5, 1>;
template class BEAMINTERACTION::BeamToSpherePotentialPair<2, 2>;
