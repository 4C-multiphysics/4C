// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_PORO_P1_ELETYPES_HPP
#define FOUR_C_W1_PORO_P1_ELETYPES_HPP

#include "4C_config.hpp"

#include "4C_w1_poro_eletypes.hpp"

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
     |  QUAD 4 Element                                                      |
     *----------------------------------------------------------------------*/
    class WallQuad4PoroP1Type : public Discret::Elements::WallQuad4PoroType
    {
     public:
      std::string name() const override { return "WallQuad4PoroP1Type"; }

      static WallQuad4PoroP1Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static WallQuad4PoroP1Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  QUAD 9 Element                                                      |
     *----------------------------------------------------------------------*/
    class WallQuad9PoroP1Type : public Discret::Elements::WallQuad9PoroType
    {
     public:
      std::string name() const override { return "WallQuad9PoroP1Type"; }

      static WallQuad9PoroP1Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static WallQuad9PoroP1Type instance_;
    };

    /*----------------------------------------------------------------------*
     |  TRI 3 Element                                                       |
     *----------------------------------------------------------------------*/
    class WallTri3PoroP1Type : public Discret::Elements::WallTri3PoroType
    {
     public:
      std::string name() const override { return "WallTri3PoroP1Type"; }

      static WallTri3PoroP1Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      int initialize(Core::FE::Discretization& dis) override;

      void setup_element_definition(
          std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
          override;

     private:
      static WallTri3PoroP1Type instance_;
    };

  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
