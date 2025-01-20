// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_PORO_ELETYPES_HPP
#define FOUR_C_SO3_PORO_ELETYPES_HPP

#include "4C_config.hpp"

#include "4C_so3_hex27.hpp"
#include "4C_so3_hex8.hpp"
#include "4C_so3_nurbs27.hpp"
#include "4C_so3_tet10.hpp"
#include "4C_so3_tet4.hpp"

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
     |  HEX 8 Element                                       |
     *----------------------------------------------------------------------*/
    class SoHex8PoroType : public SoHex8Type
    {
     public:
      std::string name() const override { return "So_hex8PoroType"; }

      static SoHex8PoroType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoHex8PoroType instance_;

      std::string get_element_type_string() const { return "SOLIDH8PORO"; }
    };

    /*----------------------------------------------------------------------*
     |  TET 4 Element                                       |
     *----------------------------------------------------------------------*/
    class SoTet4PoroType : public SoTet4Type
    {
     public:
      std::string name() const override { return "So_tet4PoroType"; }

      static SoTet4PoroType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoTet4PoroType instance_;

      std::string get_element_type_string() const { return "SOLIDT4PORO"; }
    };


    /*----------------------------------------------------------------------*
     |  HEX 27 Element                                       |
     *----------------------------------------------------------------------*/
    class SoHex27PoroType : public SoHex27Type
    {
     public:
      std::string name() const override { return "So_hex27PoroType"; }

      static SoHex27PoroType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoHex27PoroType instance_;

      std::string get_element_type_string() const { return "SOLIDH27PORO"; }
    };

    /*----------------------------------------------------------------------*
     |  TET 10 Element                                       |
     *----------------------------------------------------------------------*/
    class SoTet10PoroType : public SoTet10Type
    {
     public:
      std::string name() const override { return "So_tet10PoroType"; }

      static SoTet10PoroType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoTet10PoroType instance_;

      std::string get_element_type_string() const { return "SOLIDT10PORO"; }
    };

    /*----------------------------------------------------------------------*
     |  NURBS 27 Element                                       |
     *----------------------------------------------------------------------*/
    class SoNurbs27PoroType : public Nurbs::SoNurbs27Type
    {
     public:
      std::string name() const override { return "So_nurbs27PoroType"; }

      static SoNurbs27PoroType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoNurbs27PoroType instance_;

      std::string get_element_type_string() const { return "SONURBS27PORO"; }
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
