// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_EXTENSION_CYLINDER_COSY_HPP
#define FOUR_C_MAT_ANISOTROPY_EXTENSION_CYLINDER_COSY_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_cylinder_coordinate_system_manager.hpp"
#include "4C_mat_anisotropy_extension_base.hpp"

#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication
namespace Mat
{
  // Forward declaration
  class CoordinateSystemProvider;
  class CylinderCoordinateSystemProvider;

  /*!
   * @brief definition, which kind of fibers should be used (Element fibers or nodal (aka Gauss
   * point) fibers)
   */
  enum class CosyLocation
  {
    /// Undefined fiber location
    None,
    /// Cosy is constant per element
    ElementCosy,
    /// Cosy is defined on the GP
    GPCosy
  };

  class CylinderCoordinateSystemAnisotropyExtension : public Mat::BaseAnisotropyExtension
  {
   public:
    CylinderCoordinateSystemAnisotropyExtension();
    ///@name Packing and Unpacking
    /// @{

    /*!
     * \brief Pack all data for parallel distribution and restart
     *
     * \param data
     */
    void pack_anisotropy(Core::Communication::PackBuffer& data) const override;

    /*!
     * \brief Unpack all data from parallel distribution or restart
     */
    void unpack_anisotropy(Core::Communication::UnpackBuffer& buffer) override;
    /// @}

    /*!
     * \brief This method will be called by Mat::Anisotropy if element and Gauss point fibers are
     * available
     */
    void on_global_data_initialized(Anisotropy& anisotropy) override;

    /*!
     * \brief Returns the cylinder coordinate system for a specific Gausspoint.
     *
     *
     * \note If the coordinate system is only given on the element, then this is returned.
     *
     * \param gp Gauss point
     * \return const CylinderCoordinateSystemProvider& Reference to the cylinder coordinate system
     * provider
     */
    [[nodiscard]] const CylinderCoordinateSystemProvider& get_cylinder_coordinate_system(
        int gp) const;

    [[nodiscard]] std::shared_ptr<Mat::CoordinateSystemProvider> get_coordinate_system_provider(
        int gp) const;

   private:
    /*!
     * \brief This method will be called by Mat::Anisotropy to notify that element information is
     * available.
     */
    void on_global_element_data_initialized(Anisotropy& anisotropy) override;


    /*!
     * \brief This method will be called by Mat::Anisotropy to notify that Gauss point information
     * is available.
     */
    void on_global_gp_data_initialized(Anisotropy& anisotropy) override;

    /// flag where the coordinate system is located
    CosyLocation cosy_location_;

    /// element-level cylinder coordinate system manager supplied by the Anisotropy class during
    /// setup
    std::optional<Mat::CylinderCoordinateSystemManager> element_cosy_;

    /// Gauss-point-level cylinder coordinate system managers supplied by the Anisotropy class
    /// during setup
    std::vector<Mat::CylinderCoordinateSystemManager> gp_cosy_;
  };
}  // namespace Mat
FOUR_C_NAMESPACE_CLOSE

#endif
