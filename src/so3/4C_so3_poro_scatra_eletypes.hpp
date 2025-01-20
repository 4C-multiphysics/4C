// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_PORO_SCATRA_ELETYPES_HPP
#define FOUR_C_SO3_PORO_SCATRA_ELETYPES_HPP

#include "4C_config.hpp"

#include "4C_so3_poro_eletypes.hpp"

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
     |  HEX 8 Element                                         schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class SoHex8PoroScatraType : public SoHex8PoroType
    {
     public:
      std::string name() const override { return "So_hex8PoroScatraType"; }

      static SoHex8PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoHex8PoroScatraType instance_;

      std::string get_element_type_string() const { return "SOLIDH8POROSCATRA"; }
    };

    /*----------------------------------------------------------------------*
     |  TET 4 Element                                         schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class SoTet4PoroScatraType : public SoTet4PoroType
    {
     public:
      std::string name() const override { return "So_tet4PoroScatraType"; }

      static SoTet4PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoTet4PoroScatraType instance_;

      std::string get_element_type_string() const { return "SOLIDT4POROSCATRA"; }
    };


    /*----------------------------------------------------------------------*
     |  HEX 27 Element                                        schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class SoHex27PoroScatraType : public SoHex27PoroType
    {
     public:
      std::string name() const override { return "So_hex27PoroScatraType"; }

      static SoHex27PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoHex27PoroScatraType instance_;

      std::string get_element_type_string() const { return "SOLIDH27POROSCATRA"; }
    };

    /*----------------------------------------------------------------------*
     |  TET 10 Element                                        schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class SoTet10PoroScatraType : public SoTet10PoroType
    {
     public:
      std::string name() const override { return "So_tet10PoroScatraType"; }

      static SoTet10PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoTet10PoroScatraType instance_;

      std::string get_element_type_string() const { return "SOLIDT10POROSCATRA"; }
    };

    /*----------------------------------------------------------------------*
     |  NURBS 27 Element                                      schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class SoNurbs27PoroScatraType : public SoNurbs27PoroType
    {
     public:
      std::string name() const override { return "So_nurbs27PoroScatraType"; }

      static SoNurbs27PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoNurbs27PoroScatraType instance_;

      std::string get_element_type_string() const { return "SONURBS27POROSCATRA"; }
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
