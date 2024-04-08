/*----------------------------------------------------------------------*/
/*! \file

\brief Declaration of the input functionality of anisotropy and coordinate systems

\level 3


*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MAT_ANISOTROPY_HPP
#define FOUR_C_MAT_ANISOTROPY_HPP

#include "baci_config.hpp"

#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_mat_anisotropy_cylinder_coordinate_system_manager.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

#include <optional>

BACI_NAMESPACE_OPEN

// forward declarations
namespace CORE::COMM
{
  class PackBuffer;
}

namespace INPUT
{
  class LineDefinition;
}

namespace MAT
{
  // forward declarations
  namespace ELASTIC
  {
    class StructuralTensorStrategyBase;
  }  // namespace ELASTIC
  class BaseAnisotropyExtension;

  /*!
   * Class that handles the initialization of the anisotropic parts of materials.
   *
   * \note: This is an optional class. There are anisotropic materials that do not use this
   * interface.
   */
  class Anisotropy
  {
   public:
    //! Numerical tolerance used to check for unit vectors
    static constexpr float TOLERANCE = 1e-9;

    /*!
     * Constructor of the anisotropy class
     */
    Anisotropy();

    /// Destructor
    virtual ~Anisotropy() = default;

    ///@name Packing and Unpacking
    //@{

    /*!
     * Pack all data for parallel distribution
     *
     * @param data (in/out) : data object
     */
    void PackAnisotropy(CORE::COMM::PackBuffer& data) const;

    /*!
     * Unpack all data from another processor
     *
     * @param data (in) : data object
     * @param position (in/out) : current position in the data
     */
    void UnpackAnisotropy(const std::vector<char>& data, std::vector<char>::size_type& position);

    /*!
     * This method should be called as soon as the number of Gauss points is known.
     * @param numgp Number of Gauss points
     */
    void SetNumberOfGaussPoints(int numgp);

    /*!
     * \brief Returns the number of Gauss points
     *
     * \return int Number of Gauss integration points
     */
    int GetNumberOfGaussPoints() const;

    /*!
     * \brief Returns the number of element fibers
     *
     * \return int Number of fibers per element
     */
    int GetNumberOfElementFibers() const;

    /*!
     * \brief Returns the number of Gauss point fibers
     *
     * \return int number of fibers per Gauss point
     */
    int GetNumberOfGPFibers() const;

    /*!
     * \brief Flag whether the element has given a cylinder coordinate system
     *
     * \return true
     * \return false
     */
    bool HasElementCylinderCoordinateSystem() const;

    /*!
     * \brief Flag whether cylinder coordinate system are defined in the Gauss point
     *
     * \return true
     * \return false
     */
    bool HasGPCylinderCoordinateSystem() const;

    /*!
     * \brief Returns the cylinder coordinate system that are defined on the element
     *
     * \return const CylinderCoordinateSystemManager&
     */
    const CylinderCoordinateSystemManager& GetElementCylinderCoordinateSystem() const;

    /*!
     * \brief Returns the cylinder coordinate system that belongs to the Gauss point
     *
     * \param gp (in) : Gauss point
     * \return const CylinderCoordinateSystemManager&
     */
    const CylinderCoordinateSystemManager& GetGPCylinderCoordinateSystem(int gp) const;

    /*!
     * \brief Set vector of element fibers
     *
     * \param fibers Vector of element fibers
     */
    void SetElementFibers(const std::vector<CORE::LINALG::Matrix<3, 1>>& fibers);

    /*!
     * \brief Set Gauss point fibers. First index represents the Gauss point, second index the
     * fiber id
     *
     * @param fibers Vector of Gauss point fibers
     */
    void SetGaussPointFibers(const std::vector<std::vector<CORE::LINALG::Matrix<3, 1>>>& fibers);

    /*!
     * Reads the line definition of an element to get the fibers defined on the element
     *
     * @param linedef (in) : Input line of the corresponding element
     */
    void ReadAnisotropyFromElement(INPUT::LineDefinition* linedef);

    /*!
     * This method extracts the Gauss-point fibers written by the elements into the ParameterList
     * and stores them internally. This method should be called during the PostSetup. This method
     * will only check for Gauss-point fibers if the initialization mode is
     * #INIT_MODE_NODAL_FIBERS.
     *
     * @param params Container that hold the Gauss-point fibers.
     */
    void ReadAnisotropyFromParameterList(const Teuchos::ParameterList& params);

    /*!
     * \brief A notifier method that calls all extensions that element fibers are initialized
     */
    void OnElementFibersInitialized();


    /*!
     * \brief A notifier method that calls all extensions that Gauss point fibers are initialized
     */
    void OnGPFibersInitialized();

    /// @name Getter methods for the fibers
    ///@{

    /*!
     * \brief Returns the i-th element fiber
     *
     * \param i zerobased Id of the fiber
     * \return const CORE::LINALG::Matrix<3, 1>& Reference to the fiber vector
     */
    const CORE::LINALG::Matrix<3, 1>& GetElementFiber(unsigned int i) const;

    /*!
     * \brief Returns an vector of all element fibers
     *
     * \return const std::vector<CORE::LINALG::Matrix<3, 1>>&
     */
    const std::vector<CORE::LINALG::Matrix<3, 1>>& GetElementFibers() const;

    /*!
     * \brief Returns the a vector of all Gauss point fibers. The first index is the GP, the second
     * index is the zero pased fiber id
     *
     * \return const std::vector<std::vector<CORE::LINALG::Matrix<3, 1>>>&
     */
    const std::vector<std::vector<CORE::LINALG::Matrix<3, 1>>>& GetGPFibers() const;

    /*!
     * \brief Returns the i-th Gauss-point fiber at the Gauss point
     *
     * \param gp Gauss point
     * \param i zerobased fiber id
     * \return const CORE::LINALG::Matrix<3, 1>&
     */
    const CORE::LINALG::Matrix<3, 1>& GetGPFiber(unsigned int gp, unsigned int i) const;
    ///@}

    /*!
     * \brief Register an external fiber extension. To be used by a material implementation.
     *
     * An anisotropy extension handles all needed structural tensors and fiber vector calculation.
     *
     * \param extension
     */
    void RegisterAnisotropyExtension(BaseAnisotropyExtension& extension);

   private:
    void InsertFibers(std::vector<CORE::LINALG::Matrix<3, 1>> fiber);
    /// Number of Gauss points
    unsigned numgp_ = 0;

    /// Flag whether element fibers were set
    bool element_fibers_initialized_;

    /// Flag whether GP fibers are initialized
    bool gp_fibers_initialized_;

    /**
     * Fibers of the element. The first index is for the Gauss points, the second index is for the
     * fiber id
     */
    std::vector<CORE::LINALG::Matrix<3, 1>> elementFibers_;

    /**
     * Fibers of the element. The first index is for the Gauss points, the second index is for the
     * fiber id
     */
    std::vector<std::vector<CORE::LINALG::Matrix<3, 1>>> gpFibers_;

    /// Cylinder coordinate system manager on element level
    std::optional<CylinderCoordinateSystemManager> elementCylinderCoordinateSystemManager_;

    /// Cylinder coordinate system manager on gp level
    std::vector<CylinderCoordinateSystemManager> gpCylinderCoordinateSystemManagers_;

    /// Anisotropy to extend the functionality
    std::vector<Teuchos::RCP<BaseAnisotropyExtension>> extensions_;
  };
}  // namespace MAT

BACI_NAMESPACE_CLOSE

#endif  // MAT_ANISOTROPY_H
