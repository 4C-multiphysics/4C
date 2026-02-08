// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_CONTACTCONSTITUTIVELAW_PARAMETER_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_CONTACTCONSTITUTIVELAW_PARAMETER_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT::CONSTITUTIVELAW
{
  class ConstitutiveLaw;
}  // namespace CONTACT::CONSTITUTIVELAW

namespace CONTACT::CONSTITUTIVELAW
{

  /// Type of contact constitutive law
  enum class ConstitutiveLawType
  {
    colaw_none,              ///< Undefined
    colaw_brokenrational,    ///< Brokenrational constitutive law
    colaw_linear,            ///< Linear constitutive law
    colaw_cubic,             ///< Cubic constitutive law
    colaw_power,             ///< Power law as constitutive law
    colaw_mirco,             ///< MIRCO constitutive law
    colaw_python_surrogate,  ///< Python-based surrogate contact law
  };

  /**
   * \brief Base object to hold 'quick' access contact constitutive law parameters
   */
  class Parameter
  {
   public:
    Parameter() = delete;

    /**
     * Construct the parameter object from the raw input parameters.
     */
    Parameter(const Core::IO::InputParameterContainer& coconstlawdata);

    /// destructor
    virtual ~Parameter() = default;

    // Access offset of the function
    double get_offset() const { return offset_; }

    /**
     * \brief Offset from the edge (gap==0) from where the constitutive law will be used
     *
     * When regarding different smoothness patches, the maximum peaks of the patches are in
     * general not aligned. To model this phenomenon, an offset is introduced into the
     * constitutive laws
     */
    const double offset_;
  };  // class Parameter
}  // namespace CONTACT::CONSTITUTIVELAW

FOUR_C_NAMESPACE_CLOSE

#endif
