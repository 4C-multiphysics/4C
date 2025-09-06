// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_ALE_ALE3_NURBS_HPP
#define FOUR_C_ALE_ALE3_NURBS_HPP

/*----------------------------------------------------------------------------*/
/* header inclusions */
#include "4C_config.hpp"

#include "4C_ale_ale3.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    namespace Nurbs
    {
      class Ale3NurbsType : public Core::Elements::ElementType
      {
       public:
        std::string name() const override { return "Ale3_NurbsType"; }

        static Ale3NurbsType& instance();

        Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

        std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
            const std::string eledistype, const int id, const int owner) override;

        std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

        void nodal_block_information(
            Core::Elements::Element* dwele, int& numdf, int& dimns) override;

        Core::LinAlg::SerialDenseMatrix compute_null_space(
            Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override;

        void setup_element_definition(
            std::map<std::string, std::map<Core::FE::CellType, Core::IO::InputSpec>>& definitions)
            override
        {
          // do nothing. Definition inserted by normal wall element.
        }

       private:
        static Ale3NurbsType instance_;
      };

      /*----------------------------------------------------------------------------*/
      /*----------------------------------------------------------------------------*/
      class Ale3Nurbs : public Discret::Elements::Ale3
      {
       public:
        /*!
        \brief Standard Constructor

        \param id    : A unique global id
        \param owner : proc id that will own this element
        */
        Ale3Nurbs(int id, int owner);

        /*!
        \brief Copy Constructor

        Makes a deep copy of a Element

        */
        Ale3Nurbs(const Ale3Nurbs& old);



        /*!
        \brief Return unique ParObject id

        every class implementing ParObject needs a unique id defined at the
        top of this file.

        \return my parobject id
        */
        int unique_par_object_id() const override
        {
          return Ale3NurbsType::instance().unique_par_object_id();
        }

        virtual Core::Elements::ElementType& element_type() { return Ale3NurbsType::instance(); }

        /// Print this element
        void print(std::ostream& os) const override;


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
          return (0);
          /*
          if (num_node()==27 || num_node()==8)
          {
            return 12;
          }
          else {
            FOUR_C_THROW("Could not determine number of lines");
            return -1;
          }
          */
        }


        /*!
        \brief Return number of surfaces of this element
        */
        int num_surface() const override
        {
          if (num_node() == 27 || num_node() == 8)
          {
            return 6;
          }
          else
          {
            FOUR_C_THROW("Could not determine number of surfaces");
            return -1;
          }
        }

       private:
      };

    }  // namespace Nurbs
  }  // namespace Elements
}  // namespace Discret


FOUR_C_NAMESPACE_CLOSE

#endif
