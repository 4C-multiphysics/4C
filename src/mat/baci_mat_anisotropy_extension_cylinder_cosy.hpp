/*----------------------------------------------------------------------*/
/*! \file

\brief Declaration of a cylinder coordinate system anisotropy extension to be used by anisotropic
materials with @MAT::Anisotropy

\level 3


*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MAT_ANISOTROPY_EXTENSION_CYLINDER_COSY_HPP
#define FOUR_C_MAT_ANISOTROPY_EXTENSION_CYLINDER_COSY_HPP

#include "baci_config.hpp"

#include "baci_mat_anisotropy_extension_base.hpp"

#include <Teuchos_RCPDecl.hpp>

BACI_NAMESPACE_OPEN

// forward declarations
namespace CORE::COMM
{
  class PackBuffer;
}
namespace MAT
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

  class CylinderCoordinateSystemAnisotropyExtension : public MAT::BaseAnisotropyExtension
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
    void PackAnisotropy(CORE::COMM::PackBuffer& data) const override;

    /*!
     * \brief Unpack all data from parallel distribution or restart
     *
     * \param data whole data array
     * \param position position of the current reader
     */
    void UnpackAnisotropy(
        const std::vector<char>& data, std::vector<char>::size_type& position) override;
    /// @}

    /*!
     * \brief This method will be called by MAT::Anisotropy if element and Gauss point fibers are
     * available
     */
    void OnGlobalDataInitialized() override;

    /*!
     * \brief Retrns the cylinder coordinate system for a specific Gausspoint.
     *
     *
     * \note If the coordinate system is only given on the element, then this is returned.
     *
     * \param gp Gauss point
     * \return const CylinderCoordinateSystemProvider& Reference to the cylinder coordinate system
     * provider
     */
    const CylinderCoordinateSystemProvider& GetCylinderCoordinateSystem(int gp) const;

    Teuchos::RCP<MAT::CoordinateSystemProvider> GetCoordinateSystemProvider(int gp) const;

   private:
    /*!
     * \brief This method will be called by MAT::Anisotropy to notify that element information is
     * available.
     */
    void OnGlobalElementDataInitialized() override;


    /*!
     * \brief This method will be called by MAT::Anisotropy to notify that Gauss point information
     * is available.
     */
    void OnGlobalGPDataInitialized() override;

    /// flag where the coordinate system is located
    CosyLocation cosyLocation_;
  };
}  // namespace MAT
BACI_NAMESPACE_CLOSE

#endif