// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_CYLINDER_COORDINATE_SYSTEM_MANAGER_HPP
#define FOUR_C_MAT_ANISOTROPY_CYLINDER_COORDINATE_SYSTEM_MANAGER_HPP

#include "4C_config.hpp"

#include "4C_io_input_parameter_container.hpp"
#include "4C_mat_anisotropy_cylinder_coordinate_system_provider.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::Communication
{
  class PackBuffer;
  class UnpackBuffer;
}  // namespace Core::Communication

namespace Mat
{
  /*!
   * \brief A manager that handles reading of a cylinder coordinate manager, distribution to
   * multiple processors and getter/setter methods
   */
  class CylinderCoordinateSystemManager : public CylinderCoordinateSystemProvider
  {
   public:
    /*!
     * \brief Constructor of the cylinder coordinate system manager
     */
    explicit CylinderCoordinateSystemManager();

    ///@name Packing and Unpacking
    ///@{
    /*!
     * Pack all data for parallel distribution
     *
     * @param data (in/out) : data object
     */
    void pack(Core::Communication::PackBuffer& data) const;

    /*!
     * Unpack all data from another processor
     */
    void unpack(Core::Communication::UnpackBuffer& buffer);

    ///@}


    /*!
     * Reads the input parameter container of an element to get the coordinate system defined on the
     * element
     *
     * @param container (in) : a container of data to create the corresponding element
     */
    void read_from_element_line_definition(const Core::IO::InputParameterContainer& container);

    /*!
     * \brief Flag
     *
     * \return true
     * \return false
     */
    bool is_defined() const { return is_defined_; }

    const Core::LinAlg::Matrix<3, 1>& get_rad() const override
    {
      if (!is_defined_)
      {
        FOUR_C_THROW("The coordinate system is not yet defined.");
      }
      return radial_;
    };

    const Core::LinAlg::Matrix<3, 1>& get_axi() const override
    {
      if (!is_defined_)
      {
        FOUR_C_THROW("The coordinate system is not yet defined.");
      }
      return axial_;
    }

    const Core::LinAlg::Matrix<3, 1>& get_cir() const override
    {
      if (!is_defined_)
      {
        FOUR_C_THROW("The coordinate system is not yet defined.");
      }
      return circumferential_;
    };

    /*!
     * \brief Evaluation
     *
     * \param cosy
     */
    void evaluate_local_coordinate_system(Core::LinAlg::Matrix<3, 3>& cosy) const;

   private:
    /// Flag whether coordinate system is already set
    bool is_defined_ = false;

    /*!
     * \brief Unit vector in radial direction
     */
    Core::LinAlg::Matrix<3, 1> radial_;

    /*!
     * \brief unit vector in axial direction
     */
    Core::LinAlg::Matrix<3, 1> axial_;


    /*!
     * \brief unit vector in circumferential direction
     */
    Core::LinAlg::Matrix<3, 1> circumferential_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
