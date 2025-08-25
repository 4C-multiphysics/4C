// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONSTRAINT_LAGPENCONSTRAINT_NOXINTERFACE_HPP
#define FOUR_C_CONSTRAINT_LAGPENCONSTRAINT_NOXINTERFACE_HPP


#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_constraint_interface_preconditioner.hpp"
#include "4C_solver_nonlin_nox_constraint_interface_required.hpp"
#include "4C_structure_new_timint_basedataglobalstate.hpp"

FOUR_C_NAMESPACE_OPEN


namespace LAGPENCONSTRAINT
{
  class NoxInterface : public NOX::Nln::CONSTRAINT::Interface::Required
  {
   public:
    /// constructor
    NoxInterface();

    /// initialize important member variables
    void init(const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr);

    /** \brief Setup important new member variables
     *
     *  Supposed to be overloaded by derived classes. */
    virtual void setup();

    /// @name Supported basic interface functions
    /// @{
    //! Returns the constraint right-hand-side norms [derived]
    double get_constraint_rhs_norms(const Core::LinAlg::Vector<double>& F,
        NOX::Nln::StatusTest::QuantityType chQ, ::NOX::Abstract::Vector::NormType type,
        bool isScaled) const override;

    /// Returns the root mean square (abbr.: RMS) of the Lagrange multiplier updates [derived]
    double get_lagrange_multiplier_update_rms(const Core::LinAlg::Vector<double>& xNew,
        const Core::LinAlg::Vector<double>& xOld, double aTol, double rTol,
        NOX::Nln::StatusTest::QuantityType checkQuantity,
        bool disable_implicit_weighting) const override;

    /// Returns the increment norm of the largange multiplier DoFs
    double get_lagrange_multiplier_update_norms(const Core::LinAlg::Vector<double>& xNew,
        const Core::LinAlg::Vector<double>& xOld, NOX::Nln::StatusTest::QuantityType checkQuantity,
        ::NOX::Abstract::Vector::NormType type, bool isScaled) const override;

    /// Returns the previous solution norm of the largange multiplier DoFs
    double get_previous_lagrange_multiplier_norms(const Core::LinAlg::Vector<double>& xOld,
        NOX::Nln::StatusTest::QuantityType checkQuantity, ::NOX::Abstract::Vector::NormType type,
        bool isScaled) const override;
    /// @}

   protected:
    /// get the init indicator state
    inline const bool& is_init() const { return isinit_; };

    /// get the setup indicator state
    inline const bool& is_setup() const { return issetup_; };

    /// Check if init() has been called
    inline void check_init() const
    {
      if (not is_init()) FOUR_C_THROW("Call init() first!");
    };

    /// Check if init() and setup() have been called, yet.
    inline void check_init_setup() const
    {
      if (not is_init() or not is_setup()) FOUR_C_THROW("Call init() and setup() first!");
    };


   protected:
    /// flag indicating if init() has been called
    bool isinit_;

    /// flag indicating if setup() has been called
    bool issetup_;

   private:
    //! global state data container
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;
  };


  class NoxInterfacePrec : public NOX::Nln::CONSTRAINT::Interface::Preconditioner
  {
   public:
    /// constructor
    NoxInterfacePrec();

    /// initialize important member variables
    void init(const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr);

    /** \brief Setup important new member variables
     *
     *  Supposed to be overloaded by derived classes. */
    virtual void setup();


    bool is_saddle_point_system() const override;

    bool is_condensed_system() const override;

    void fill_maps_for_preconditioner(
        std::vector<Teuchos::RCP<Core::LinAlg::Map>>& maps) const override;
    /// @}

   protected:
    /// get the init indicator state
    inline const bool& is_init() const { return isinit_; };

    /// get the setup indicator state
    inline const bool& is_setup() const { return issetup_; };

    /// Check if init() has been called
    inline void check_init() const
    {
      if (not is_init()) FOUR_C_THROW("Call init() first!");
    };

    /// Check if init() and setup() have been called, yet.
    inline void check_init_setup() const
    {
      if (not is_init() or not is_setup()) FOUR_C_THROW("Call init() and setup() first!");
    };

   protected:
    /// flag indicating if init() has been called
    bool isinit_;

    /// flag indicating if setup() has been called
    bool issetup_;

   private:
    //! global state data container
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;
  };

}  // namespace LAGPENCONSTRAINT


FOUR_C_NAMESPACE_CLOSE

#endif
