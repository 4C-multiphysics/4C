// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_input_field.hpp"

#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

Core::IO::InputFieldReference Core::IO::InputFieldRegistry::register_field_reference(
    const std::string& ref_name)
{
  // Access the field data, with the side-effect of creating it if it does not exist.
  fields[ref_name];
  return InputFieldReference{
      .ref_name = ref_name,
      .registry = this,
  };
}


void Core::IO::InputFieldRegistry::attach_input_field(
    InputFieldReference ref, InitFunction init, RedistributeFunction redistribute, void* field_ptr)
{
  FOUR_C_ASSERT(ref.registry == this,
      "Internal error: InputFieldReference does not refer to this InputFieldRegistry.");

  FOUR_C_ASSERT(fields.contains(ref.ref_name),
      "Internal error: Input field '{}' is not registered.", ref.ref_name);

  SetupFunctions& field_ref_data = fields[ref.ref_name];
  field_ref_data.init_functions[field_ptr] = std::move(init);
  field_ref_data.redistribute_functions[field_ptr] = std::move(redistribute);
}


void Core::IO::InputFieldRegistry::detach_input_field(InputFieldReference ref, void* field_ptr)
{
  FOUR_C_ASSERT(ref.registry == this,
      "Internal error: InputFieldReference does not refer to this InputFieldRegistry.");

  FOUR_C_ASSERT(fields.contains(ref.ref_name),
      "Internal error: Input field '{}' is not registered.", ref.ref_name);

  auto& init_functions = fields[ref.ref_name].init_functions;
  FOUR_C_ASSERT(init_functions.contains(field_ptr),
      "Input field '{}' does not have an init function for the given field pointer.", ref.ref_name);
  init_functions.erase(field_ptr);

  auto& redistribute_functions = fields[ref.ref_name].redistribute_functions;
  FOUR_C_ASSERT(redistribute_functions.contains(field_ptr),
      "Input field '{}' does not have a redistribute function for the given field pointer.",
      ref.ref_name);
  redistribute_functions.erase(field_ptr);
}


Core::IO::InputFieldRegistry& Core::IO::global_input_field_registry()
{
  static auto singleton_owner =
      Core::Utils::make_singleton_owner([]() { return std::make_unique<InputFieldRegistry>(); });
  return *singleton_owner.instance(Utils::SingletonAction::create);
}

FOUR_C_NAMESPACE_CLOSE