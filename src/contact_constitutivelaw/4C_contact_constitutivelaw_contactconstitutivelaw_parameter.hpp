// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_CONTACTCONSTITUTIVELAW_PARAMETER_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_CONTACTCONSTITUTIVELAW_PARAMETER_HPP


/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/* forward declarations */

namespace CONTACT::CONSTITUTIVELAW
{
  class ConstitutiveLaw;
  class Container;
}  // namespace CONTACT::CONSTITUTIVELAW



namespace Inpar::CONTACT
{
  /// Type of contact constitutive law
  enum class ConstitutiveLawType
  {
    colaw_none,            ///< undefined
    colaw_brokenrational,  ///< brokenrational constitutive law
    colaw_linear,          ///< linear constitutive law
    colaw_cubic,           ///< cubic constitutive law
    colaw_power,           ///< simple power law as constitutive law
    colaw_mirco            ///< mirco constitutive law
  };
}  // namespace Inpar::CONTACT


/*----------------------------------------------------------------------*/
/* declarations */


namespace CONTACT::CONSTITUTIVELAW
{
  /**
   * \brief Base object to hold 'quick' access contact constitutive law parameters
   */
  class Parameter
  {
   public:
    Parameter() = delete;
    /** construct the contact constitutive law object given the parameters
     * params[in] contactconstitutivelawdata "dumb" container containing the
     * contactconstitutivelaw data from the input file
     */
    Parameter(const std::shared_ptr<const CONTACT::CONSTITUTIVELAW::Container> coconstlawdata);

    /// destructor
    virtual ~Parameter() = default;

    /// create CoConstLaw instance of matching type with my parameters
    virtual std::shared_ptr<CONTACT::CONSTITUTIVELAW::ConstitutiveLaw>
    create_constitutive_law() = 0;

    // Access offset of the function
    double get_offset() { return offset_; }

    /**
     * \brief Offset from the edge (gap==0) from where the constitutive law will be used
     *
     * When regarding different smoothness patches, the maximum peaks of the patches are in
     * general not aligned. To model this phenomenon, an offset is introduced into the
     * constitutive laws
     */
    const double offset_;
  };  // class Parameter

  /**
   * \brief Container to pass Contact Constitutive Law parameters around
   */

  class Container : public Core::IO::InputParameterContainer
  {
   public:
    /// @name life span
    //@{

    /// standard constructor
    Container(const int id,                              ///< unique contact constitutivelaw ID
        const Inpar::CONTACT::ConstitutiveLawType type,  ///< type of contact constitutivelaw
        const std::string name                           ///< name of contact constitutivelaw
    );


    //@}

    /// @name Query methods
    //@{

    /// Return material id
    [[nodiscard]] inline virtual int id() const
    {
      return id_;
    }  // todo does not override anything.. is it supposed to be this way?

    /// Return material name
    [[nodiscard]] inline virtual std::string name() const
    {
      return name_;
    }  // todo does not override anything.. is it supposed to be this way?

    /// Print this ConstitutiveLaw
    void print(std::ostream& os) const override;

    /// Return type of constitutivelaw
    [[nodiscard]] inline virtual Inpar::CONTACT::ConstitutiveLawType type() const { return type_; }

    /**
     * \brief Return quickly accessible material parameter data
     *
     * These quick access parameters are stored in separate member #params_;
     * whereas the originally read ones are stored in Core::IO::InputParameterContainer base
     */
    [[nodiscard]] inline CONTACT::CONSTITUTIVELAW::Parameter* parameter() const
    {
      return params_.get();
    }

    //@}

   protected:
    /// Unique ID of this ConstitutiveLaw, no second ConstitutiveLaw of same ID may exist
    int id_{};

    /// Type of this condition
    Inpar::CONTACT::ConstitutiveLawType type_{};

    /// Name
    std::string name_{};

    /// Unwrapped constitutivelaw data for 'quick' access
    std::shared_ptr<CONTACT::CONSTITUTIVELAW::Parameter> params_{};
  };
}  // namespace CONTACT::CONSTITUTIVELAW

FOUR_C_NAMESPACE_CLOSE

#endif
