// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PAR_BUNDLE_HPP
#define FOUR_C_MAT_PAR_BUNDLE_HPP

/*----------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_material_parameter_base.hpp"
#include "4C_utils_lazy_ptr.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// bundle holds all read-in materials of a #Global::Problem
    ///
    /// <h4>About</h4>
    /// The bundle provides an interface between unique material IDs and
    /// associated material parameters/data. The material ID is set via
    /// input file. It has to be unique, larger than zero or equal zero.
    /// Material ID and data are hold in #matmap_.
    ///
    /// <h4>Special issues for multi-problem-instance applications</h4>
    /// We have for each Global::Problem instance an individual material bundle.
    /// However, this fact is not transparent at the read time of the elements
    /// which are only aware of their material ID. The variable #materialreadfromproblem_
    /// of the material bundle of the 0th Global::Problem instance make it possible to switch
    /// among different Global::Problem. (The variable #materialreadfromproblem_ is redundant
    /// in the material bundles of all non-0th Global::Problem instances.)
    ///
    /// \author bborn
    /// \date 02/09
    class Bundle
    {
     public:
      /**
       * Insert new pair of material ID and the input data. The input data is set up for lazy
       * construction the first time the material is accessed.
       */
      void insert(int matid, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter> mat);

      /**
       * Check whether material parameters exist for provided @p id.
       *
       * @note This call does not check whether material parameters are already constructed. It only
       * checks whether the material ID is known.
       */
      [[nodiscard]] bool id_exists(int id) const;

      /// provide access to material map (a li'l dirty)
      [[nodiscard]] const std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>>& map()
          const
      {
        return matmap_;
      }

      /// return number of defined materials
      int num() const { return matmap_.size(); }

      /// return material parameters
      Core::Mat::PAR::Parameter* parameter_by_id(
          const int num  ///< request is made for this material ID
      ) const;

      /// return (first) ID by material type
      ///
      /// \return The ID of searched for material type.
      ///         If the search is unsuccessful -1 is returned
      int first_id_by_type(const Core::Materials::MaterialType type) const;

      /// return problem index to read from
      int get_read_from_problem() const { return materialreadfromproblem_; }

      /// set problem index to read from
      void set_read_from_problem(const int p  ///< index of Global::Problem instance to read for
      )
      {
        materialreadfromproblem_ = p;
      }

      /// reset problem index to read from, i.e. to index 0
      void reset_read_from_problem() { materialreadfromproblem_ = 0; }

     private:
      /// The map linking material IDs to input parameters. The data is stored as a lazy pointer to
      /// allow for lazy construction of material parameters in arbitrary order.
      std::map<int, Core::Utils::LazyPtr<Core::Mat::PAR::Parameter>> matmap_;

      /// the index of problem instance of which material read-in shall be performed
      int materialreadfromproblem_{};
    };

  }  // namespace PAR

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
