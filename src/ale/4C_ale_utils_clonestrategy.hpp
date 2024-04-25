/*----------------------------------------------------------------------------*/
/*! \file

\brief Strategy to clone ALE discretization form other discretization

\level 1

*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
#ifndef FOUR_C_ALE_UTILS_CLONESTRATEGY_HPP
#define FOUR_C_ALE_UTILS_CLONESTRATEGY_HPP

/*----------------------------------------------------------------------------*/
/*header inclusions */
#include "4C_config.hpp"

#include <Teuchos_RCP.hpp>

#include <map>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/* forward declarations */
namespace DRT
{
  class Element;
}

/*----------------------------------------------------------------------------*/
/* definition of classes */
namespace ALE
{
  namespace UTILS
  {
    /*!
    \brief Implementation of special clone strategy for automatic generation
           of ale from a given fluid discretization

    */
    class AleCloneStrategy
    {
     public:
      /// constructor
      explicit AleCloneStrategy() {}
      /// destructor
      virtual ~AleCloneStrategy() = default;

     protected:
      /// determine element type string and whether element is copied or not
      bool DetermineEleType(DRT::Element* actele,  ///< current element
          const bool ismyele,                      ///< true if element belongs to my proc
          std::vector<std::string>& eletype        ///< element type
      );

      /*! \brief Set element-specific data (material etc.)
       *
       *  We need to set material and possibly other things to complete element
       *  setup. This is again really ugly as we have to extract the actual
       *  element type in order to access the material property.
       */
      void SetElementData(
          Teuchos::RCP<DRT::Element> newele,  ///< newly created element where data has to be set
          DRT::Element* oldele,               ///< existing element, that has been cloned
          const int matid,                    ///< ID of material law
          const bool nurbsdis                 ///< Is this a Nurbs-based discretization?
      );

      /// returns conditions names to be copied (source and target name)
      std::map<std::string, std::string> ConditionsToCopy() const;

      /// check for correct material
      void CheckMaterialType(const int matid);

     private:
    };  // class AleCloneStrategy
  }     // namespace UTILS
}  // namespace ALE

FOUR_C_NAMESPACE_CLOSE

#endif
