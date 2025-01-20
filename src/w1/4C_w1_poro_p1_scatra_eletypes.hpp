// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_PORO_P1_SCATRA_ELETYPES_HPP
#define FOUR_C_W1_PORO_P1_SCATRA_ELETYPES_HPP

#include "4C_config.hpp"

#include "4C_w1_poro_p1_eletypes.hpp"

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
    class WallQuad4PoroP1ScatraType : public Discret::Elements::WallQuad4PoroP1Type
    {
     public:
      std::string name() const override { return "WallQuad4PoroP1ScatraType"; }

      static WallQuad4PoroP1ScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallQuad4PoroP1ScatraType instance_;
    };

    /*----------------------------------------------------------------------*
     |  QUAD 9 Element                                        schmidt 09/17 |
     *----------------------------------------------------------------------*/
    class WallQuad9PoroP1ScatraType : public Discret::Elements::WallQuad9PoroP1Type
    {
     public:
      std::string name() const override { return "WallQuad9PoroP1ScatraType"; }

      static WallQuad9PoroP1ScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallQuad9PoroP1ScatraType instance_;
    };

    /*----------------------------------------------------------------------*
     |  TRI 3 Element                                         schmidt 09/17 ||
     *----------------------------------------------------------------------*/
    class WallTri3PoroP1ScatraType : public Discret::Elements::WallTri3PoroP1Type
    {
     public:
      std::string name() const override { return "WallTri3PoroP1ScatraType"; }

      static WallTri3PoroP1ScatraType& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static WallTri3PoroP1ScatraType instance_;
    };

  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
