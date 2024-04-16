/*----------------------------------------------------------------------*/
/*! \file

\brief main file containing routines for calculation of artery element

\level 3


*----------------------------------------------------------------------*/

#ifndef FOUR_C_ART_NET_ARTERY_ELE_CALC_HPP
#define FOUR_C_ART_NET_ARTERY_ELE_CALC_HPP

#include "baci_config.hpp"

#include "baci_art_net_artery.hpp"
#include "baci_art_net_artery_ele_action.hpp"
#include "baci_art_net_artery_ele_interface.hpp"
#include "baci_discretization_fem_general_utils_local_connectivity_matrices.hpp"
#include "baci_linalg_fixedsizematrix.hpp"

BACI_NAMESPACE_OPEN


namespace DRT
{
  namespace ELEMENTS
  {
    /// Internal artery implementation
    /*!
      This internal class keeps all the working arrays needed to
      calculate the artery element. Additionally, the method Sysmat()
      provides a clean and fast element implementation.

      <h3>Purpose</h3>

      The idea is to separate the element maintenance (class Artery) from the
      mathematical contents (this class). There are different
      implementations of the artery element, this is just one such
      implementation.

      The artery element will allocate exactly one object of this class for all
      artery elements with the same number of nodes in the mesh. This
      allows us to use exactly matching working arrays (and keep them
      around.)

      The code is meant to be as clean as possible. This is the only way
      to keep it fast. The number of working arrays has to be reduced to
      a minimum so that the element fits into the cache. (There might be
      room for improvements.)

      <h3>Usability</h3>

      The calculations are done by the Evaluate ... () methods which are
      inherited from the ArteryEleInterface.

      \author kremheller
      \date 03/18
    */

    template <CORE::FE::CellType distype>
    class ArteryEleCalc : public ArteryEleInterface
    {
     protected:
      /// (private) protected constructor, since we are a Singleton.
      /// this constructor is called from a derived class
      /// -> therefore, it has to be protected instead of private
      ArteryEleCalc(const int numdofpernode, const std::string& disname);

     public:
      //! number of nodes
      static constexpr int iel_ = CORE::FE::num_nodes<distype>;

      /*!
       * \brief  calculate element length
       * \param  ele[in] element whose length should be calculated
       * \return         the length of the element
       *
       * \note   only checked for line2 elements
       */
      virtual double CalculateEleLength(Artery* ele);


     protected:
      //! array for shape functions
      CORE::LINALG::Matrix<iel_, 1> funct_;
      //! array for shape function derivatives w.r.t s
      CORE::LINALG::Matrix<1, iel_> deriv_;
      //! transposed array for shape function derivatives w.r.t s
      CORE::LINALG::Matrix<iel_, 1> tderiv_;
      //! transposed jacobian "dx/ds"
      CORE::LINALG::Matrix<1, 1> xjm_;
      //! inverse of transposed jacobian "ds/dx"
      CORE::LINALG::Matrix<1, 1> xji_;
      //! global derivatives of shape functions w.r.t s
      CORE::LINALG::Matrix<2, iel_> derxy_;
    };

  }  // namespace ELEMENTS

}  // namespace DRT



BACI_NAMESPACE_CLOSE

#endif
