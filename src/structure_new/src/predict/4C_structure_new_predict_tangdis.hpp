// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_PREDICT_TANGDIS_HPP
#define FOUR_C_STRUCTURE_NEW_PREDICT_TANGDIS_HPP

#include "4C_config.hpp"

#include "4C_solver_nonlin_nox_abstract_prepostoperator.hpp"
#include "4C_structure_new_predict_generic.hpp"

#include <Teuchos_RCPDecl.hpp>

// forward declaration

FOUR_C_NAMESPACE_OPEN

namespace NOX
{
  namespace Nln
  {
    class Group;
  }  // namespace Nln
}  // namespace NOX
namespace Solid
{
  namespace Predict
  {
    class TangDis : public Generic
    {
     public:
      TangDis();
      //! setup class specific stuff
      void setup() override;

      //! do the class specific predictor step
      void compute(::NOX::Abstract::Group& grp) override;

      void reset_state() override;

      //! return the dbc increment
      const Core::LinAlg::Vector<double>& get_dbc_incr() const;

      //! return the switch for the pre/post operator
      const bool& is_apply_linear_reaction_forces() const;

      //! derived
      bool pre_apply_force_external(Core::LinAlg::Vector<double>& fextnp) const override;

     private:
      std::shared_ptr<Core::LinAlg::Vector<double>> dbc_incr_ptr_;

      bool apply_linear_reaction_forces_;
    };  // class TangDis
  }  // namespace Predict
}  // namespace Solid

namespace NOX
{
  namespace Nln
  {
    namespace GROUP
    {
      namespace PrePostOp
      {
        /*! \brief Tangential Displacement helper class
         *
         *  This class is an implementation of the NOX::Nln::Abstract::PrePostOperator
         *  and is used to modify the computeF() routines of the given NOX::Nln::Group
         *  (see Solid::Predict::TangDis). It's called by the wrapper class
         *  NOX::Nln::GROUP::PrePostOperator.
         *
         *  */
        class TangDis : public NOX::Nln::Abstract::PrePostOperator
        {
         public:
          //! constructor
          TangDis(const Teuchos::RCP<const Solid::Predict::TangDis>& tang_predict_ptr);

          //! add the linear reaction forces
          void run_post_compute_f(
              Core::LinAlg::Vector<double>& F, const NOX::Nln::Group& grp) override;

         private:
          //! pointer to the tangdis object (read-only)
          Teuchos::RCP<const Solid::Predict::TangDis> tang_predict_ptr_;
        };  // class TangDis
      }  // namespace PrePostOp
    }  // namespace GROUP
  }  // namespace Nln
}  // namespace NOX

FOUR_C_NAMESPACE_CLOSE

#endif
