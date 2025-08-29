// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_singleton_owner.hpp"

#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

void Core::Utils::SingletonOwnerRegistry::finalize()
{
  for (const auto& [_, deleter] : instance().deleters_)
  {
    deleter();
  }
}

void Core::Utils::SingletonOwnerRegistry::register_deleter(
    void* owner, std::function<void()> deleter)
{
  instance().deleters_.emplace(owner, std::move(deleter));
}

void Core::Utils::SingletonOwnerRegistry::unregister(void* owner)
{
  instance().deleters_.erase(owner);
}

void Core::Utils::SingletonOwnerRegistry::initialize() { instance(); }

Core::Utils::SingletonOwnerRegistry& Core::Utils::SingletonOwnerRegistry::instance()
{
  static SingletonOwnerRegistry instance;
  return instance;
}

FOUR_C_NAMESPACE_CLOSE