// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_PORO_SCATRA_ELETYPES_HPP
#define FOUR_C_W1_PORO_SCATRA_ELETYPES_HPP

#include "4C_config.hpp"

#include "4C_w1_poro.hpp"

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
     |  QUAD 4 Element                                        schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class WallQuad4PoroScatraType : public Discret::Elements::WallQuad4PoroType
    {
     public:
      std::string name() const override { return "WallQuad4PoroScatraType"; }

      static WallQuad4PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallQuad4PoroScatraType instance_;
    };

    /*----------------------------------------------------------------------*
     |  QUAD 9 Element                                        schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class WallQuad9PoroScatraType : public Discret::Elements::WallQuad9PoroType
    {
     public:
      std::string name() const override { return "WallQuad9PoroScatraType"; }

      static WallQuad9PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallQuad9PoroScatraType instance_;
    };

    /*----------------------------------------------------------------------*
     |  NURBS 4 Element                                       schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class WallNurbs4PoroScatraType : public Discret::Elements::WallNurbs4PoroType
    {
     public:
      std::string name() const override { return "WallNurbs4PoroScatraType"; }

      static WallNurbs4PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallNurbs4PoroScatraType instance_;
    };

    /*----------------------------------------------------------------------*
     |  NURBS 9 Element                                       schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class WallNurbs9PoroScatraType : public Discret::Elements::WallNurbs9PoroType
    {
     public:
      std::string name() const override { return "WallNurbs9PoroScatraType"; }

      static WallNurbs9PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallNurbs9PoroScatraType instance_;
    };

    /*----------------------------------------------------------------------*
     |  TRI 3 Element                                         schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class WallTri3PoroScatraType : public Discret::Elements::WallTri3PoroType
    {
     public:
      std::string name() const override { return "WallTri3PoroScatraType"; }

      static WallTri3PoroScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallTri3PoroScatraType instance_;
    };

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
