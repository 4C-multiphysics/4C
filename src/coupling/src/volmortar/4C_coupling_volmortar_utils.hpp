/*----------------------------------------------------------------------*/
/*! \file

 \brief utils functions related to volmortar

\level 2

 *----------------------------------------------------------------------*/


#ifndef FOUR_C_COUPLING_VOLMORTAR_UTILS_HPP
#define FOUR_C_COUPLING_VOLMORTAR_UTILS_HPP

#include "4C_config.hpp"

#include <Teuchos_RCPDecl.hpp>

#include <vector>

FOUR_C_NAMESPACE_OPEN

/*---------------------------------------------------------------------*
 | forward declarations                                    vuong 09/14 |
 *---------------------------------------------------------------------*/
namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Coupling::VolMortar
{
  class VolMortarCoupl;

  namespace UTILS
  {
    /// Helper class for assigning materials for volumetric coupling of non conforming meshes
    /*!
     When coupling two overlapping discretizations, most often one discretization needs access
     to the corresponding element/material on the other side. For conforming meshes this is straight
     forward as there is one unique element on the other side and therefore one unique material,
     which can be accessed. However, for non conforming meshes there are potentially several
     elements overlapping. Therefore, some rule for assigning materials is needed. This class is
     meant to do that. It gets the element to which it shall assign a material and a vector of IDs
     of the overlapping elements of the other discretization.

     The default strategy will just assign the material of the first element in the vector to the
     other element. This is fine for constant material properties, for instance. If there is furhter
     work to be done it is meant to derive from this class (see e.g. TSIMaterialStrategy).

     \author vuong 10/14
     */

    class DefaultMaterialStrategy
    {
     public:
      //! standard constructor
      DefaultMaterialStrategy(){};

      //! virtual destructor
      virtual ~DefaultMaterialStrategy() = default;

      //! assign material of discretization B
      virtual void assign_material2_to1(const Coupling::VolMortar::VolMortarCoupl* volmortar,
          Core::Elements::Element* ele1, const std::vector<int>& ids_2,
          Teuchos::RCP<Core::FE::Discretization> dis1, Teuchos::RCP<Core::FE::Discretization> dis2);

      //! assign material of discretization B
      virtual void assign_material1_to2(const Coupling::VolMortar::VolMortarCoupl* volmortar,
          Core::Elements::Element* ele2, const std::vector<int>& ids_1,
          Teuchos::RCP<Core::FE::Discretization> dis1, Teuchos::RCP<Core::FE::Discretization> dis2);
    };
  }  // namespace UTILS
}  // namespace Coupling::VolMortar


FOUR_C_NAMESPACE_CLOSE

#endif
