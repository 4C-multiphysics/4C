// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_EXTENSION_DEFAULT_HPP
#define FOUR_C_MAT_ANISOTROPY_EXTENSION_DEFAULT_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /*!
   * \brief Default anisotropy extension
   *
   * This is an anisotropy extension that contains the functionality of most of the anisotropic
   * materials. Fibers are either initialized using the FIBERX notation, via a cylinder coordinate
   * system or in x direction.
   *
   * \tparam numfib Number of fibers
   */
  template <unsigned int numfib>
  class DefaultAnisotropyExtension : public FiberAnisotropyExtension<numfib>
  {
   public:
    //! @name Initialization modes of the fibers
    /// @{
    //! Fibers defined in material on element basis
    static constexpr int INIT_MODE_ELEMENT_EXTERNAL = 0;
    //! Fibers defined in input file on element basis
    static constexpr int INIT_MODE_ELEMENT_FIBERS = 1;
    //! Fibers defined in material on Gauss point basis
    static constexpr int INIT_MODE_NODAL_EXTERNAL = 4;
    //! Fibers defined in input file on Gauss point basis
    static constexpr int INIT_MODE_NODAL_FIBERS = 3;
    /// @}

    /*!
     * \brief Constructor of the default anisotropy extension
     *
     * \param init_mode initialization mode. Use one of the following:
     * DefaultAnisotropyExtension::INIT_MODE_ELEMENT_EXTERNAL,
     * DefaultAnisotropyExtension::INIT_MODE_ELEMENT_FIBERS,
     * DefaultAnisotropyExtension::INIT_MODE_NODAL_EXTERNAL,
     * DefaultAnisotropyExtension::INIT_MODE_NODAL_FIBERS
     *
     * \param gamma angle
     *
     * \param adapt_angle flag whether the angle is subject to growth and remodeling
     *
     * \param stucturalTensorStrategy Reference to the structural tensor strategy
     * \param fiber_ids List of ids of the fibers
     */
    DefaultAnisotropyExtension(int init_mode, double gamma, bool adapt_angle,
        const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& stucturalTensorStrategy,
        std::array<int, numfib> fiber_ids);

    ///@name Packing and Unpacking
    /// @{
    void pack_anisotropy(Core::Communication::PackBuffer& data) const override;

    void unpack_anisotropy(Core::Communication::UnpackBuffer& buffer) override;
    /// @}

    /*!
     * \brief Initializes the Element fibers
     *
     * \return true if the material is parametrized so that element fibers should be used
     * \return false otherwise
     */
    bool do_element_fiber_initialization(Anisotropy& anisotropy) override;

    /*!
     * \brief Initializes Gauss point fibers
     *
     * \return true if the materials is parametrized so that Gauss point fibers should be used
     * \return false otherwise
     */
    bool do_gp_fiber_initialization(Anisotropy& anisotropy) override;

    /*!
     * \brief Set Fiber vectors by a new angle gamma in the current configuration
     *
     * \note this method is here for backwards compatibility
     *
     * \param newgamma New angle
     * \param locsys local coordinate system
     * \param defgrd deformation gradient
     */
    void set_fiber_vecs(double newgamma, const Core::LinAlg::Tensor<double, 3, 3>& locsys,
        const Core::LinAlg::Tensor<double, 3, 3>& defgrd);

    /*!
     * \brief Set the new element fibers directly
     *
     * \param fibervec unit vector pointing in fiber direction
     */
    void set_fiber_vecs(const Core::LinAlg::Tensor<double, 3>& fibervec);

    /*!
     * \brief Status of fiber initialization
     *
     * \return true In case the fibers are initialized
     * \return false In case the fibers are not yet initialized
     */
    bool fibers_initialized() const { return initialized_; }

   protected:
    void on_fibers_initialized() override
    {
      FiberAnisotropyExtension<numfib>::on_fibers_initialized();
      initialized_ = true;
    }

    /*!
     * \brief The single fiber is aligned in z-direction
     */
    virtual void do_external_fiber_initialization();

   private:
    /// Initialization mode
    const int init_mode_;

    /// angle of the fiber
    const double gamma_;

    /// flag whether the angle should be adapted with growth and remodeling (for comoatibility
    /// reasons)
    const bool adapt_angle_;

    /// Ids of the fiber to use
    const std::array<int, numfib> fiber_ids_;

    /// Flag that shows the initialization state of the fibers
    bool initialized_ = false;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
