// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_HPP
#define FOUR_C_MAT_ANISOTROPY_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_anisotropy_cylinder_coordinate_system_manager.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <optional>

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
}

namespace Mat
{
  // forward declarations
  namespace Elastic
  {
    class StructuralTensorStrategyBase;
  }  // namespace Elastic
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
    void pack_anisotropy(Core::Communication::PackBuffer& data) const;

    /*!
     * Unpack all data from another processor
     */
    void unpack_anisotropy(Core::Communication::UnpackBuffer& buffer);

    /*!
     * This method should be called as soon as the number of Gauss points is known.
     * @param numgp Number of Gauss points
     */
    void set_number_of_gauss_points(int numgp);

    /*!
     * \brief Returns the number of Gauss points
     *
     * \return int Number of Gauss integration points
     */
    int get_number_of_gauss_points() const;

    /*!
     * \brief Returns the number of element fibers
     *
     * \return int Number of fibers per element
     */
    int get_number_of_element_fibers() const;

    /*!
     * \brief Returns the number of Gauss point fibers
     *
     * \return int number of fibers per Gauss point
     */
    int get_number_of_gauss_point_fibers() const;

    /*!
     * \brief Flag whether the element has given a cylinder coordinate system
     *
     * \return true
     * \return false
     */
    bool has_element_cylinder_coordinate_system() const;

    /*!
     * \brief Flag whether cylinder coordinate system are defined in the Gauss point
     *
     * \return true
     * \return false
     */
    bool has_gp_cylinder_coordinate_system() const;

    /*!
     * \brief Returns the cylinder coordinate system that are defined on the element
     *
     * \return const CylinderCoordinateSystemManager&
     */
    const CylinderCoordinateSystemManager& get_element_cylinder_coordinate_system() const;

    /*!
     * \brief Returns the cylinder coordinate system that belongs to the Gauss point
     *
     * \param gp (in) : Gauss point
     * \return const CylinderCoordinateSystemManager&
     */
    const CylinderCoordinateSystemManager& get_gp_cylinder_coordinate_system(int gp) const;

    /*!
     * \brief Set vector of element fibers
     *
     * \param fibers Vector of element fibers
     */
    void set_element_fibers(const std::vector<Core::LinAlg::Matrix<3, 1>>& fibers);

    /*!
     * \brief Set Gauss point fibers. First index represents the Gauss point, second index the
     * fiber id
     *
     * @param fibers Vector of Gauss point fibers
     */
    void set_gauss_point_fibers(const std::vector<std::vector<Core::LinAlg::Matrix<3, 1>>>& fibers);

    /*!
     * Reads the input parameter container of an element to get the coordinate system defined on the
     * element
     *
     * @param container (in) : a container of data to create the corresponding element
     */
    void read_anisotropy_from_element(const Core::IO::InputParameterContainer& container);

    /*!
     * This method extracts the Gauss-point fibers written by the elements into the ParameterList
     * and stores them internally. This method should be called during the post_setup. This method
     * will only check for Gauss-point fibers if the initialization mode is
     * #INIT_MODE_NODAL_FIBERS.
     *
     * @param params Container that hold the Gauss-point fibers.
     */
    void read_anisotropy_from_parameter_list(const Teuchos::ParameterList& params);

    /*!
     * \brief A notifier method that calls all extensions that element fibers are initialized
     */
    void on_element_fibers_initialized();


    /*!
     * \brief A notifier method that calls all extensions that Gauss point fibers are initialized
     */
    void on_gp_fibers_initialized();

    /// @name Getter methods for the fibers
    ///@{

    /*!
     * \brief Returns the i-th element fiber
     *
     * \param i zerobased Id of the fiber
     * \return const Core::LinAlg::Matrix<3, 1>& Reference to the fiber vector
     */
    const Core::LinAlg::Matrix<3, 1>& get_element_fiber(unsigned int i) const;

    /*!
     * \brief Returns an vector of all element fibers
     *
     * \return const std::vector<Core::LinAlg::Matrix<3, 1>>&
     */
    const std::vector<Core::LinAlg::Matrix<3, 1>>& get_element_fibers() const;

    /*!
     * \brief Returns the a vector of all Gauss point fibers. The first index is the GP, the second
     * index is the zero-based fiber id
     *
     * \return const std::vector<std::vector<Core::LinAlg::Matrix<3, 1>>>&
     */
    const std::vector<std::vector<Core::LinAlg::Matrix<3, 1>>>& get_gauss_point_fibers() const;

    /*!
     * \brief Returns the i-th Gauss-point fiber at the Gauss point
     *
     * \param gp Gauss point
     * \param i zerobased fiber id
     * \return const Core::LinAlg::Matrix<3, 1>&
     */
    const Core::LinAlg::Matrix<3, 1>& get_gauss_point_fiber(unsigned int gp, unsigned int i) const;
    ///@}

    /*!
     * \brief Register an external fiber extension. To be used by a material implementation.
     *
     * An anisotropy extension handles all needed structural tensors and fiber vector calculation.
     *
     * \param extension
     */
    void register_anisotropy_extension(BaseAnisotropyExtension& extension);

   private:
    void insert_fibers(std::vector<Core::LinAlg::Matrix<3, 1>> fiber);
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
    std::vector<Core::LinAlg::Matrix<3, 1>> element_fibers_;

    /**
     * Fibers of the element. The first index is for the Gauss points, the second index is for the
     * fiber id
     */
    std::vector<std::vector<Core::LinAlg::Matrix<3, 1>>> gp_fibers_;

    /// Cylinder coordinate system manager on element level
    std::optional<CylinderCoordinateSystemManager> element_cylinder_coordinate_system_manager_;

    /// Cylinder coordinate system manager on gp level
    std::vector<CylinderCoordinateSystemManager> gp_cylinder_coordinate_system_managers_;

    /// Anisotropy to extend the functionality
    std::vector<std::shared_ptr<BaseAnisotropyExtension>> extensions_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
