// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_COUPLING_ADAPTER_BASE_HPP
#define FOUR_C_COUPLING_ADAPTER_BASE_HPP

/*----------------------------------------------------------------------------*/
/* headers */
#include "4C_config.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_vector.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*/
/* definition of classes */
namespace Coupling::Adapter
{
  /*! \class CouplingBase
   *  \brief Abstract interface to coupling managers
   *
   *  This class is an interface to the coupling adapters.
   */
  class CouplingBase
  {
   public:
    /// empty constructor
    CouplingBase() {};

    /// virtual destructor
    virtual ~CouplingBase() = default;

    /// @name Conversion between target and source
    //!@{

    /// transfer a dof vector from target to source
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> target_to_source(
        const Core::LinAlg::Vector<double>& mv  ///< target vector (to be transferred)
    ) const = 0;

    /// transfer a dof vector from source to target
    virtual std::shared_ptr<Core::LinAlg::Vector<double>> source_to_target(
        const Core::LinAlg::Vector<double>& sv  ///< source vector (to be transferred)
    ) const = 0;

    /// transfer a dof vector from target to source
    virtual std::shared_ptr<Core::LinAlg::MultiVector<double>> target_to_source(
        const Core::LinAlg::MultiVector<double>& mv  ///< target vector (to be transferred)
    ) const = 0;

    /// transfer a dof vector from source to target
    virtual std::shared_ptr<Core::LinAlg::MultiVector<double>> source_to_target(
        const Core::LinAlg::MultiVector<double>& sv  ///< source vector (to be transferred)
    ) const = 0;

    /// transfer a dof vector from target to source
    virtual void target_to_source(
        const Core::LinAlg::MultiVector<double>& mv,  ///< target vector (to be transferred)
        Core::LinAlg::MultiVector<double>& sv         ///< source vector (containing result)
    ) const = 0;

    /// transfer a dof vector from source to target
    virtual void source_to_target(
        const Core::LinAlg::MultiVector<double>& sv,  ///< source vector (to be transferred)
        Core::LinAlg::MultiVector<double>& mv         ///< target vector (containing result)
    ) const = 0;

    //!@}

    //! @name Coupled maps
    //!@{

    /// the interface dof map of the target side
    virtual std::shared_ptr<const Core::LinAlg::Map> target_dof_map() const = 0;

    /// the interface dof map of the source side
    virtual std::shared_ptr<const Core::LinAlg::Map> source_dof_map() const = 0;

    //!@}
  };
}  // namespace Coupling::Adapter

FOUR_C_NAMESPACE_CLOSE

#endif
