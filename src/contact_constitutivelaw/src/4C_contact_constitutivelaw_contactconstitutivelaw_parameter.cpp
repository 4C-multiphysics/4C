// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_contact_constitutivelaw_contactconstitutivelaw_parameter.hpp"

FOUR_C_NAMESPACE_OPEN


CONTACT::CONSTITUTIVELAW::Parameter::Parameter(const Core::IO::InputParameterContainer&
        coconstlawdata  ///< read and validate contactconstitutivelaw data (of 'slow' access)
    )
    : offset_(coconstlawdata.get<double>("Offset"))
{
}

FOUR_C_NAMESPACE_CLOSE
