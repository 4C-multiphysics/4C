// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_STRUCTURE_NEW_PREDICT_GENERIC_HPP
#define FOUR_C_STRUCTURE_NEW_PREDICT_GENERIC_HPP

#include "4C_config.hpp"

#include "4C_structure_new_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>

// forward declaration ...

namespace NOX
{
  namespace Abstract
  {
    class Group;
  }  // namespace Abstract
}  // namespace NOX

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  template <typename T>
  class Vector;
}
namespace Solid
{
  class Dbc;
  namespace IMPLICIT
  {
    class Generic;
  }  // namespace IMPLICIT
  namespace TimeInt
  {
    class BaseDataGlobalState;
    class BaseDataIO;
  }  // namespace TimeInt
  namespace Predict
  {
    class Generic
    {
     public:
      //! constructor
      Generic();

      //! destructor
      virtual ~Generic() = default;

      //! initialize the base class variables
      virtual void init(const Solid::PredictorType& type,
          const std::shared_ptr<Solid::IMPLICIT::Generic>& implint_ptr,
          const std::shared_ptr<Solid::Dbc>& dbc_ptr,
          const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& gstate_ptr,
          const std::shared_ptr<Solid::TimeInt::BaseDataIO>& iodata_ptr,
          const std::shared_ptr<Teuchos::ParameterList>& noxparams_ptr);

      //! setup of the specific predictor
      virtual void setup() = 0;

      //! Get the predictor type enum
      const Solid::PredictorType& get_type() const { return type_; };

      //! returns the name of the used predictor
      virtual std::string name() const;

      //! Preprocess the predictor step
      virtual void pre_predict(::NOX::Abstract::Group& grp);

      //! Pre-/Postprocess the specific predictor step
      void predict(::NOX::Abstract::Group& grp);

      //! Reset the predictor state. Derived classes can override this function to also reset their
      //! specific state.
      virtual void reset_state();

      //! Calculate the specific predictor step
      virtual void compute(::NOX::Abstract::Group& grp) = 0;

      //! Postprocess the predictor step
      virtual void post_predict(::NOX::Abstract::Group& grp);

      //! return a constant reference to the global state object (read only)
      const Solid::TimeInt::BaseDataGlobalState& global_state() const;

      //! print the result of the predictor step
      void print() const;

      //! Run before the external force are computed and assembled
      virtual bool pre_apply_force_external(Core::LinAlg::Vector<double>& fextnp) const;

     protected:
      //! returns init state
      const bool& is_init() const { return isinit_; };

      //! returns setup state
      const bool& is_setup() const { return issetup_; };

      void check_init() const;

      void check_init_setup() const;

      std::shared_ptr<Solid::IMPLICIT::Generic>& impl_int_ptr();
      Solid::IMPLICIT::Generic& impl_int();

      std::shared_ptr<Solid::Dbc>& dbc_ptr();
      Solid::Dbc& dbc();

      std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr();
      Solid::TimeInt::BaseDataGlobalState& global_state();

      std::shared_ptr<Solid::TimeInt::BaseDataIO>& io_data_ptr();
      Solid::TimeInt::BaseDataIO& io_data();

      std::shared_ptr<Teuchos::ParameterList>& nox_params_ptr();
      Teuchos::ParameterList& nox_params();

     protected:
      //! indicates if the init() function has been called
      bool isinit_;

      //! indicates if the setup() function has been called
      bool issetup_;

     private:
      //! predictor type
      Solid::PredictorType type_;

      //! pointer to the implicit integrator
      std::shared_ptr<Solid::IMPLICIT::Generic> implint_ptr_;

      //! pointer to the dirichlet boundary condition object
      std::shared_ptr<Solid::Dbc> dbc_ptr_;

      //! global state pointer
      std::shared_ptr<Solid::TimeInt::BaseDataGlobalState> gstate_ptr_;

      //! input/output data pointer
      std::shared_ptr<Solid::TimeInt::BaseDataIO> iodata_ptr_;

      std::shared_ptr<Teuchos::ParameterList> noxparams_ptr_;
    };  // class  Generic
  }  // namespace Predict
}  // namespace Solid

FOUR_C_NAMESPACE_CLOSE

#endif
