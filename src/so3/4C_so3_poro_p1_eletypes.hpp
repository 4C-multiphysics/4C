// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_SO3_PORO_P1_ELETYPES_HPP
#define FOUR_C_SO3_PORO_P1_ELETYPES_HPP

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
     |  HEX 8 Element                                                       |
     *----------------------------------------------------------------------*/
    class SoHex8PoroP1Type : public SoHex8PoroType
    {
     public:
      std::string name() const override { return "So_hex8PoroP1Type"; }

      static SoHex8PoroP1Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoHex8PoroP1Type instance_;

      std::string get_element_type_string() const { return "SOLIDH8POROP1"; }
    };

    /*----------------------------------------------------------------------*
     |  TET 4 Element                                                       |
     *----------------------------------------------------------------------*/
    class SoTet4PoroP1Type : public SoTet4PoroType
    {
     public:
      std::string name() const override { return "So_tet4PoroP1Type"; }

      static SoTet4PoroP1Type& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
          const std::string eledistype, const int id, const int owner) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      int initialize(Core::FE::Discretization& dis) override;

      void nodal_block_information(
          Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

      Core::LinAlg::SerialDenseMatrix compute_null_space(
          Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;

     private:
      static SoTet4PoroP1Type instance_;

      std::string get_element_type_string() const { return "SOLIDT4POROP1"; }
    };

  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
