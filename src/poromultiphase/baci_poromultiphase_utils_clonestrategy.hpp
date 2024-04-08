/*----------------------------------------------------------------------*/
/*! \file
 \brief utils methods for cloning the porofluid discretization


   \level 3

 *----------------------------------------------------------------------*/

#ifndef FOUR_C_POROMULTIPHASE_UTILS_CLONESTRATEGY_HPP
#define FOUR_C_POROMULTIPHASE_UTILS_CLONESTRATEGY_HPP

#include "baci_config.hpp"

#include <Teuchos_RCP.hpp>

#include <vector>

BACI_NAMESPACE_OPEN

namespace DRT
{
  class Element;
}

namespace POROMULTIPHASE
{
  namespace UTILS
  {
    /*!
    \brief implementation of special clone strategy for automatic generation
           of scatra from a given fluid discretization

     */
    class PoroFluidMultiPhaseCloneStrategy
    {
     public:
      /// constructor
      explicit PoroFluidMultiPhaseCloneStrategy() {}
      /// destructor
      virtual ~PoroFluidMultiPhaseCloneStrategy() = default;
      /// returns conditions names to be copied (source and target name)
      virtual std::map<std::string, std::string> ConditionsToCopy() const;

     protected:
      /// determine element type std::string and whether element is copied or not
      virtual bool DetermineEleType(
          DRT::Element* actele, const bool ismyele, std::vector<std::string>& eletype);

      /// set element-specific data (material etc.)
      void SetElementData(Teuchos::RCP<DRT::Element> newele, DRT::Element* oldele, const int matid,
          const bool isnurbs);

      /// check for correct material
      void CheckMaterialType(const int matid);

     private:
    };  // class PoroFluidMultiPhaseCloneStrategy

  }  // namespace UTILS
}  // namespace POROMULTIPHASE

BACI_NAMESPACE_CLOSE

#endif  // POROMULTIPHASE_UTILS_CLONESTRATEGY_H
