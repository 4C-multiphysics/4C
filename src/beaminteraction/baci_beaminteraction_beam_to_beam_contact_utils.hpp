/*----------------------------------------------------------------------------*/
/*! \file

\brief A set of utility functions for beam contact

\level 2

*/
/*----------------------------------------------------------------------------*/
#ifndef FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_CONTACT_UTILS_HPP
#define FOUR_C_BEAMINTERACTION_BEAM_TO_BEAM_CONTACT_UTILS_HPP

#include "baci_config.hpp"

#include "baci_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "baci_discretization_fem_general_largerotations.hpp"
#include "baci_lib_discret.hpp"
#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_linalg_serialdensematrix.hpp"
#include "baci_linalg_serialdensevector.hpp"

#include <Epetra_Vector.h>
#include <Sacado.hpp>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN

typedef Sacado::Fad::DFad<double> FAD;

namespace BEAMINTERACTION
{
  /*!
  \brief Calculate angle encompassed by two lines: returns an angle \in [0;pi/2]
  */
  double CalcAngle(CORE::LINALG::Matrix<3, 1, double> a, CORE::LINALG::Matrix<3, 1, double> b);

  /*!
  \brief Get closest distance between the endpoints of two lines
  */
  template <typename type>
  type GetClosestEndpointDist(CORE::LINALG::Matrix<3, 1, type> r1_a,
      CORE::LINALG::Matrix<3, 1, type> r1_b, CORE::LINALG::Matrix<3, 1, type> r2_a,
      CORE::LINALG::Matrix<3, 1, type> r2_b);

  /*!
  \brief Set primary displacement DoFs for automatic differentiation with Sacado
  */
  template <int numnodes, int numnodalvalues>
  void SetFADDispDofs(CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, FAD>& ele1pos_,
      CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, FAD>& ele2pos_)
  {
    // The 2*3*numnodes*numnodalvalues primary DoFs are the components of the nodal positions /
    // tangents. The two (+2) additional degrees of freedom represent the dependency on the
    // parameter coordinates xi and eta, which is necessary in beam contact.
    for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
      ele1pos_(i).diff(i, 2 * 3 * numnodes * numnodalvalues + 2);

    for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
      ele2pos_(i).diff(3 * numnodes * numnodalvalues + i, 2 * 3 * numnodes * numnodalvalues + 2);

    return;
  }

  /*!
\brief BTS-Contact: Set primary displacement DoFs for automatic differentiation with Sacado
*/
  template <int numnodessol, int numnodes, int numnodalvalues>
  void SetFADDispDofs(CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, FAD>& ele1pos,
      CORE::LINALG::Matrix<3 * numnodessol, 1, FAD>& ele2pos, const int& numvar)
  {
    for (int i = 0; i < 3 * numnodes * numnodalvalues; i++)
      ele1pos(i).diff(i, 3 * numnodes * numnodalvalues + 3 * numnodessol + numvar);

    for (int i = 0; i < 3 * numnodessol; i++)
      ele2pos(i).diff(3 * numnodes * numnodalvalues + i,
          3 * numnodes * numnodalvalues + 3 * numnodessol + numvar);

    return;
  }

  /*!
  \brief Set primary parameter coordinate DoFs for automatic differentiation with Sacado
  */
  template <int numnodes, int numnodalvalues>
  void SetFADParCoordDofs(FAD& xi, FAD& eta)
  {
    // The 2*3*numnodes*numnodalvalues primary DoFs are the components of the nodal positions /
    // tangents. The two (+2) additional degrees of freedom represent the dependency on the
    // parameter coordinates xi and eta, which is necessary in beam contact.
    xi.diff((2 * 3 * numnodes * numnodalvalues + 1) - 1, 2 * 3 * numnodes * numnodalvalues + 2);
    eta.diff((2 * 3 * numnodes * numnodalvalues + 2) - 1, 2 * 3 * numnodes * numnodalvalues + 2);

    return;
  }

  /*!
  \brief BTS-Contact: Set primary parameter coordinate DoFs for automatic differentiation with
  Sacado
  */
  template <int numnodessol, int numnodes, int numnodalvalues>
  void SetFADParCoordDofs(FAD& xi1, FAD& xi2, FAD& eta)
  {
    xi1.diff(3 * numnodes * numnodalvalues + 3 * numnodessol,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 3);
    xi2.diff(3 * numnodes * numnodalvalues + 3 * numnodessol + 1,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 3);
    eta.diff(3 * numnodes * numnodalvalues + 3 * numnodessol + 2,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 3);

    return;
  }

  /*!
  \brief BTS-Contact: Set primary parameter coordinate DoFs for automatic differentiation with
  Sacado
  */
  template <int numnodessol, int numnodes, int numnodalvalues>
  void SetFADParCoordDofs(FAD& xi1, FAD& xi2, FAD& eta_a, FAD& eta_b)
  {
    xi1.diff(3 * numnodes * numnodalvalues + 3 * numnodessol,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 4);
    xi2.diff(3 * numnodes * numnodalvalues + 3 * numnodessol + 1,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 4);
    eta_a.diff(3 * numnodes * numnodalvalues + 3 * numnodessol + 2,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 4);
    eta_b.diff(3 * numnodes * numnodalvalues + 3 * numnodessol + 3,
        3 * numnodes * numnodalvalues + 3 * numnodessol + 4);

    return;
  }

  /*!
  \brief Check, if current node is a solid contact element
  */
  bool SolidContactElement(const DRT::Element& element);

  /*
  \brief Check, if two elements share a node -> neighbor elements
  */
  bool ElementsShareNode(const DRT::Element& element1, const DRT::Element& element2);

  /*
  \brief Calculate beam radius
  */
  double CalcEleRadius(const DRT::Element* ele);

  /*
  \brief Intersect two parallel cylinders
  */
  bool IntersectParallelCylinders(CORE::LINALG::Matrix<3, 1, double>& r1_a,
      CORE::LINALG::Matrix<3, 1, double>& r1_b, CORE::LINALG::Matrix<3, 1, double>& r2_a,
      CORE::LINALG::Matrix<3, 1, double>& r2_b, double& distancelimit);

  /*
  \brief Intersect two non-parallel, arbitrary oriented cylinders
  */
  bool IntersectArbitraryCylinders(CORE::LINALG::Matrix<3, 1, double>& r1_a,
      CORE::LINALG::Matrix<3, 1, double>& r1_b, CORE::LINALG::Matrix<3, 1, double>& r2_a,
      CORE::LINALG::Matrix<3, 1, double>& r2_b, double& distancelimit,
      std::pair<double, double>&
          closestpoints,  // The closest point are only set, if we have detected an intersection at
                          // a valid closest point with eta1_seg, eta2_seg \in [-1.0;1.0]
      bool& etaset);      // bool to check, if the closest point coordinates have been set or not

  /*
  \brief Calculate closest distance of a point and a line
  */
  double CalcPointLineDist(CORE::LINALG::Matrix<3, 1, double>& rline_a,
      CORE::LINALG::Matrix<3, 1, double>& rline_b, CORE::LINALG::Matrix<3, 1, double>& rp,
      double& eta);

  /*
  \brief Determine inpute parameter representing the additive searchbox increment
  */
  double DetermineSearchboxInc(Teuchos::ParameterList& beamcontactparams);

  /*
  \brief Check if a given double lies within a prescribed interval (enlarged by the tolerance
  XIETATOL)
  */
  inline bool WithinInterval(double& testpoint, double& leftbound, double& rightbound)
  {
    // The tolerance XIETATOL makes the test more conservative, i.e. the testpoint
    // is assumed to be within the interval even if it actually is slightly outside
    if (testpoint > leftbound - XIETATOL and testpoint < rightbound + XIETATOL)
      return true;
    else
      return false;
  }

  /*
  \brief Get interval-id out of numberofintervals intervals, in which the given point lies
  */
  inline int GetIntervalId(double& point, int numberofintervals, bool leftbound)
  {
    int interval_id = 0;
    double unrounded_id = 0.0;

    // With the following formula we would get the exact interval Ids when inserting the coordinate
    // of the left bound of the interval. By inserting any double value lying within the interval,
    // we get as result a double value that is larger than the sought-after interval ID but smaller
    // than the next highter ID, i.e. we have to round down the solution.
    unrounded_id = (point + 1.0) / 2.0 * numberofintervals;
    interval_id = floor(unrounded_id);

    // Size Check: If the size of the interval becomes smaller than RELSEGMENTTOL times the interval
    // length we simply shift the point to the next higher/lower interval
    double segmenttol = RELSEGMENTTOL * numberofintervals / 2.0;

    // The necessary shifting procedure depends on the fact, whether we are searching for the left
    // bound
    if (leftbound)
    {
      if (fabs(1 + interval_id - unrounded_id) < segmenttol)
        FOUR_C_THROW("Such small segmented integration intervals are not possible so far!");
    }
    // or for the right bound of the integration interval
    else
    {
      if (fabs(interval_id - unrounded_id) < segmenttol)
        FOUR_C_THROW("Such small segmented integration intervals are not possible so far!");
    }

    if (interval_id < 0) FOUR_C_THROW("Interval-ID can't be negative!");

    return interval_id;
  }

  /*
  \brief Get segment-id out of numberofsegments segmenets, in which the given point lies
  */
  inline int GetSegmentId(double& point, int numberofsegments)
  {
    int segment_id = 0;
    double unrounded_id = 0.0;

    // The method is similar to GetIntervalId() above. However, since it is purely used as output
    // quantity and it does does not influence algorithmic quantities such as the integration
    // interval length, we don't need a Size Check such as the method GetIntervalId() above. By
    // inserting any double value lying within the segment, we get as result a double value that is
    // larger than the sought-after segment ID but smaller than the next highter ID, i.e. we have to
    // round down the solution.
    unrounded_id = (point + 1.0) / 2.0 * numberofsegments;
    segment_id = floor(unrounded_id);

    if (segment_id < 0) FOUR_C_THROW("Segment-ID can't be negative!");

    return segment_id;
  }

}  // namespace BEAMINTERACTION

FOUR_C_NAMESPACE_CLOSE

#endif
