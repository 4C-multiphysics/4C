/*----------------------------------------------------------------------------*/
/*! \file

\brief interface class for beam contact

\level 2

*/
/*----------------------------------------------------------------------------*/

#include "4C_beamcontact_beam3contactinterface.hpp"

#include "4C_beamcontact_beam3contact.hpp"
#include "4C_beamcontact_beam3contactnew.hpp"
#include "4C_beaminteraction_beam_to_beam_contact_defines.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_inpar_beamcontact.hpp"

FOUR_C_NAMESPACE_OPEN

Teuchos::RCP<CONTACT::Beam3contactinterface> CONTACT::Beam3contactinterface::impl(
    const int numnodes, const int numnodalvalues, const Core::FE::Discretization& pdiscret,
    const Core::FE::Discretization& cdiscret, const std::map<int, int>& dofoffsetmap,
    Core::Elements::Element* element1, Core::Elements::Element* element2,
    Teuchos::ParameterList& beamcontactparams)
{
  // Decide, if beam contact with subsegment creation (beam3contact) or pure element based beam
  // contact (beam3contactnew) should be applied
  const bool beamssegcon = beamcontactparams.get<bool>("BEAMS_SEGCON");

  // note: numnodes is to be interpreted as number of nodes used for centerline interpolation.

  if (!beamssegcon)
  {
    switch (numnodalvalues)
    {
      case 1:
      {
        switch (numnodes)
        {
          case 2:
          {
            return Teuchos::rcp(new CONTACT::Beam3contactnew<2, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          case 3:
          {
            return Teuchos::rcp(new CONTACT::Beam3contactnew<3, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          case 4:
          {
            return Teuchos::rcp(new CONTACT::Beam3contactnew<4, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          case 5:
          {
            return Teuchos::rcp(new CONTACT::Beam3contactnew<5, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          default:
            FOUR_C_THROW(
                "No valid template parameter for the number of nodes (numnodes = 2,3,4,5 for "
                "Reissner beams) available!");
            break;
        }
        break;
      }
      case 2:
      {
        switch (numnodes)
        {
          case 2:
          {
            return Teuchos::rcp(new CONTACT::Beam3contactnew<2, 2>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          default:
            FOUR_C_THROW(
                "No valid template parameter combination for the number of nodes and number of "
                "types of nodal DoFs"
                "(only numnodes = 2 in combination with numnodalvalues=2 possible so far, i.e. 3rd "
                "order Hermite interpolation)!");
            break;
        }
        break;
      }
      default:
        FOUR_C_THROW(
            "No valid template parameter for the number of types of nodal DoFs used for centerline "
            "interpolation!\n"
            "(numnodalvalues = 1, i.e. positions              for Lagrange interpolation,\n"
            " numnodalvalues = 2, i.e. positions AND tangents for Hermite interpolation)");
        break;
    }
  }
  else
  {
    switch (numnodalvalues)
    {
      case 1:
      {
        switch (numnodes)
        {
          case 2:
          {
            return Teuchos::rcp(new CONTACT::Beam3contact<2, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          case 3:
          {
            return Teuchos::rcp(new CONTACT::Beam3contact<3, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          case 4:
          {
            return Teuchos::rcp(new CONTACT::Beam3contact<4, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          case 5:
          {
            return Teuchos::rcp(new CONTACT::Beam3contact<5, 1>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          default:
            FOUR_C_THROW(
                "No valid template parameter for the number of nodes (numnodes = 2,3,4,5 for "
                "Reissner beams) available!");
            break;
        }
        break;
      }
      case 2:
      {
        switch (numnodes)
        {
          case 2:
          {
            return Teuchos::rcp(new CONTACT::Beam3contact<2, 2>(
                pdiscret, cdiscret, dofoffsetmap, element1, element2, beamcontactparams));
          }
          default:
            FOUR_C_THROW(
                "No valid template parameter combination for the number of nodes and number of "
                "types of nodal DoFs"
                "(only numnodes = 2 in combination with numnodalvalues=2 possible so far, i.e. 3rd "
                "order Hermite interpolation)!");
            break;
        }
        break;
      }
      default:
        FOUR_C_THROW(
            "No valid template parameter for the number of types of nodal DoFs used for centerline "
            "interpolation!\n"
            "(numnodalvalues = 1, i.e. positions              for Lagrange interpolation,\n"
            " numnodalvalues = 2, i.e. positions AND tangents for Hermite interpolation)");
        break;
    }
  }
  return Teuchos::null;
}

FOUR_C_NAMESPACE_CLOSE
