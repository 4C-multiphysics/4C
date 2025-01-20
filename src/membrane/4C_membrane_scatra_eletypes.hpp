// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MEMBRANE_SCATRA_ELETYPES_HPP
#define FOUR_C_MEMBRANE_SCATRA_ELETYPES_HPP

#include "4C_config.hpp"

#include "4C_membrane_eletypes.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    /*----------------------------------------------------------------------*
     |  TRI 3 Element                                                       |
     *----------------------------------------------------------------------*/
    class MembraneScatraTri3Type : public MembraneTri3Type
    {
     public:
      std::string name() const override { return "MembraneScatra_tri3Type"; }

      static MembraneScatraTri3Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static MembraneScatraTri3Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  TRI 6 Element                                                       |
     *----------------------------------------------------------------------*/
    class MembraneScatraTri6Type : public MembraneTri6Type
    {
     public:
      std::string name() const override { return "MembraneScatra_tri6Type"; }

      static MembraneScatraTri6Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static MembraneScatraTri6Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  QUAD 4 Element                                                      |
     *----------------------------------------------------------------------*/
    class MembraneScatraQuad4Type : public MembraneQuad4Type
    {
     public:
      std::string name() const override { return "MembraneScatra_quad4Type"; }

      static MembraneScatraQuad4Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static MembraneScatraQuad4Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  QUAD 9 Element                                                      |
     *----------------------------------------------------------------------*/
    class MembraneScatraQuad9Type : public MembraneQuad9Type
    {
     public:
      std::string name() const override { return "MembraneScatra_quad9Type"; }

      static MembraneScatraQuad9Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static MembraneScatraQuad9Type instance_;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
