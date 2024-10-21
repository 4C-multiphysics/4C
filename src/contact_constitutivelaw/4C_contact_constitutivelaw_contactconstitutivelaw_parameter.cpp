#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"

#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


CONTACT::CONSTITUTIVELAW::Parameter::Parameter(
    const Teuchos::RCP<const CONTACT::CONSTITUTIVELAW::Container>
        coconstlawdata  ///< read and validate contactconstitutivelaw data (of 'slow' access)
    )
    : offset_(coconstlawdata->get<double>("Offset")){};
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
CONTACT::CONSTITUTIVELAW::Container::Container(
    const int id, const Inpar::CONTACT::ConstitutiveLawType type, const std::string name)
    : Core::IO::InputParameterContainer(), id_(id), type_(type), name_(name), params_(Teuchos::null)
{
}
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void CONTACT::CONSTITUTIVELAW::Container::print(std::ostream& os) const
{
  os << "ContactConstitutiveLaw " << id() << " " << name() << " :: ";

  Core::IO::InputParameterContainer::print(os);
}

FOUR_C_NAMESPACE_CLOSE
