// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_EXTENSION_BASE_HPP
#define FOUR_C_MAT_ANISOTROPY_EXTENSION_BASE_HPP

#include "4C_config.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication
namespace Mat
{
  // forward declaration
  class Anisotropy;

  class BaseAnisotropyExtension
  {
    // Anisotropy calls the private notification methods on_global_element_data_initialized
    // and on_global_gp_data_initialized.
    friend class Anisotropy;

   public:
    /// If element fibers are used, they are stored at the beginning of the list
    static constexpr int GPDEFAULT = 0;

    virtual ~BaseAnisotropyExtension() = default;

    ///@name Packing and Unpacking
    /// @{

    /*!
     * \brief Pack all data for parallel distribution and restart
     *
     * \param data
     */
    virtual void pack_anisotropy(Core::Communication::PackBuffer& data) const = 0;

    /*!
     * \brief Unpack all data from parallel distribution or restart
     */
    virtual void unpack_anisotropy(Core::Communication::UnpackBuffer& buffer) = 0;
    /// @}

    /*!
     * \brief This method will be called by Mat::Anisotropy if element and Gauss point fibers are
     * available
     */
    virtual void on_global_data_initialized(Anisotropy& anisotropy) = 0;

   private:
    /*!
     * \brief This method will be called by Mat::Anisotropy to notify that element information is
     * available.
     */
    virtual void on_global_element_data_initialized(Anisotropy& anisotropy) = 0;


    /*!
     * \brief This method will be called by Mat::Anisotropy to notify that Gauss point information
     * is available.
     */
    virtual void on_global_gp_data_initialized(Anisotropy& anisotropy) = 0;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif