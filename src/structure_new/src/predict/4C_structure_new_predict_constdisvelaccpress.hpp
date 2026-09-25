// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_PREDICT_CONSTDISVELACCPRESS_HPP
#define FOUR_C_STRUCTURE_NEW_PREDICT_CONSTDISVELACCPRESS_HPP

#include "4C_config.hpp"

#include "4C_structure_new_predict_generic.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Solid
{
  namespace Predict
  {
    class ConstDisVelAccPress : public Generic
    {
     public:
      //! constructor
      ConstDisVelAccPress();

      //! setup class specific stuff
      void setup() override;

      //! do the class specific predictor step
      void compute(::NOX::Abstract::Group& grp) override;

      void reset_state() override
      {
        Generic::reset_state();
        if (tangdis_ptr_) tangdis_ptr_->reset_state();
      }

     private:
      std::shared_ptr<Solid::Predict::Generic> tangdis_ptr_;
    };  // class ConstDisVelAccPress
  }  // namespace Predict
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
