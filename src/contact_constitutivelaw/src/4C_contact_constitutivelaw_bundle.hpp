// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_CONTACT_CONSTITUTIVELAW_BUNDLE_HPP
#define FOUR_C_CONTACT_CONSTITUTIVELAW_BUNDLE_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"

FOUR_C_NAMESPACE_OPEN

namespace CONTACT
{
  namespace CONSTITUTIVELAW
  {
    class Parameter;
    /*----------------------------------------------------------------------*/
    /**
     * \brief This bundle is used to hold all contact constitutive laws from the input file
     *
     * Basically it is a map, mapping IDs to contact constitutive laws which is wrapped to make some
     * sanity checks
     */
    class Bundle
    {
     public:
      /// construct
      Bundle();

      /** \brief insert new container holding contact constitutive law parameter and ID
       * \param[in] id ID od the contact constitutive law in the input file
       * \law[in] container holding the law parameter read from the input file
       */
      void insert(int id, Core::IO::InputParameterContainer container);

      /** \brief check if a contact constitutive law exists for provided ID
       *
       *\param[in] id ID of the contact constitutive law in the input file
       * \return Upon failure -1 is returned, otherwise >=0
       */
      int find(const int id) const;

      /// make quick access parameters
      void make_parameters();

      /// return number of defined materials
      int num() const;

      /** return contact constitutive law by ID
       *
       * \param[in] id ID of the contact constitutive law given in the input file
       */
      Core::IO::InputParameterContainer& by_id(const int id);

      /// return problem index to read from
      int get_read_from_problem() const { return readfromproblem_; }

     private:
      /// the map linking contact constitutive law IDs to input constitutive laws
      std::map<int, Core::IO::InputParameterContainer> map_;

      /// the index of problem instance of which contact constitutive law read-in shall be performed
      int readfromproblem_;
    };

  }  // namespace CONSTITUTIVELAW

}  // namespace CONTACT


FOUR_C_NAMESPACE_CLOSE

#endif
