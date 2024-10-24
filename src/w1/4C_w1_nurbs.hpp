// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_W1_NURBS_HPP
#define FOUR_C_W1_NURBS_HPP

#include "4C_config.hpp"

#include "4C_w1.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    namespace Nurbs
    {
      class Wall1NurbsType : public Core::Elements::ElementType
      {
       public:
        std::string name() const override { return "Wall1NurbsType"; }

        static Wall1NurbsType& instance();

        Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

        Teuchos::RCP<Core::Elements::Element> create(const std::string eletype,
            const std::string eledistype, const int id, const int owner) override;

        Teuchos::RCP<Core::Elements::Element> create(const int id, const int owner) override;

        void setup_element_definition(
            std::map<std::string, std::map<std::string, Input::LineDefinition>>& definitions)
            override;

        void nodal_block_information(
            Core::Elements::Element* dwele, int& numdf, int& dimns, int& nv, int& np) override;

        Core::LinAlg::SerialDenseMatrix compute_null_space(
            Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

       private:
        static Wall1NurbsType instance_;
      };


      class Wall1Nurbs : public Discret::Elements::Wall1
      {
       public:
        /*!
        \brief Standard Constructor

        \param id    : A unique global id
        \param owner : proc id that will own this element
        */
        Wall1Nurbs(int id, int owner);

        /*!
        \brief Copy Constructor

        Makes a deep copy of a Element

        */
        Wall1Nurbs(const Wall1Nurbs& old);


        /*!
        \brief Deep copy this instance of Wall1 and return pointer to it
        */

        Core::Elements::Element* clone() const override;


        /*!
        \brief Return unique ParObject id

        every class implementing ParObject needs a unique id defined at the
        top of this file.

        \return my parobject id
        */
        int unique_par_object_id() const override
        {
          return Wall1NurbsType::instance().unique_par_object_id();
        }


        /// Print this element
        void print(std::ostream& os) const override;

        Core::Elements::ElementType& element_type() const override
        {
          return Wall1NurbsType::instance();
        }

        /*!
        \brief Get shape type of element

        \return nurbs4 or nurbs9

        */
        Core::FE::CellType shape() const override;


        /*!
        \brief Return number of lines of this element.
        */
        int num_line() const override
        {
          if (num_node() == 9 || num_node() == 4)
          {
            return 4;
          }
          else
          {
            FOUR_C_THROW("Could not determine number of lines");
            return -1;
          }
        }


        /*!
        \brief Get vector of Teuchos::RCPs to the lines of this element
        */
        std::vector<Teuchos::RCP<Core::Elements::Element>> lines() override;


        /*!
        \brief Get vector of Teuchos::RCPs to the surfaces of this element
        */
        std::vector<Teuchos::RCP<Core::Elements::Element>> surfaces() override;


       private:
      };

    }  // namespace Nurbs
  }    // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
