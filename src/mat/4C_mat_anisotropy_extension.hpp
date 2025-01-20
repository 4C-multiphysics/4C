// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ANISOTROPY_EXTENSION_HPP
#define FOUR_C_MAT_ANISOTROPY_EXTENSION_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_extension_base.hpp"
#include "4C_mat_anisotropy_fiber_provider.hpp"
#include "4C_utils_exceptions.hpp"

#include <cstdint>
#include <functional>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  /*!
   * @brief definition, which kind of fibers should be used (Element fibers or nodal (aka Gauss
   * point) fibers)
   */
  enum class FiberLocation
  {
    /// Undefined fiber location
    None,
    /// Fibers are constant per element
    ElementFibers,
    /// Fibers are defined on the GP
    GPFibers
  };

  /*!
   * \brief A fiber extension to be used with Mat::Anisotropy
   *
   * The anisotropy extension is the actual provider of the fibers for the material law.
   * Mat::Anisotropy initializes the anisotropy information (fibers and coordinate systems) and
   * notifies the Mat::FiberAnisotropyExtension to build their needed fibers and structural tensors.
   * Every anisotropic material should have an Mat::FiberAnisotropyExtension and should register it
   * to the Anisotropy framework.
   *
   * After the setup, the fibers and structural tensors (or any other needed quantity) is computed
   * and stored. Anisotropy extension can handle element and Gauss-point fibers
   *
   * \tparam numfib Number of fibers
   */
  template <unsigned int numfib>
  class FiberAnisotropyExtension : public BaseAnisotropyExtension, public FiberProvider
  {
    // Anisotropy is a friend to create back reference
    friend class Anisotropy;

   public:
    //! @name Tensors needed for the evaluation
    /// @{
    /// The material evaluation needs the fiber vectors
    static constexpr std::uint_fast8_t FIBER_VECTORS{1U << 0U};
    /// The material evaluation needs the structural tensors
    static constexpr std::uint_fast8_t STRUCTURAL_TENSOR{1U << 1U};
    /// The material evaluation needs the structural tensors in stress like voigt notation
    static constexpr std::uint_fast8_t STRUCTURAL_TENSOR_STRESS{1U << 2U};
    /// @}

    /*!
     * \brief Create an anisotropy extension with a structural tensor strategy
     *
     * \param stucturalTensorStrategy
     */
    explicit FiberAnisotropyExtension(
        const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& stucturalTensorStrategy);

    /*!
     * \brief Create an anisotropy extension without a structural tensor strategy. Computing
     * structural tensors may not work.
     */
    FiberAnisotropyExtension();

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
     *
     * \param data whole data array
     * \param position position of the current reader
     */
    void unpack_anisotropy(Core::Communication::UnpackBuffer& buffer) override;
    /// @}

    /// @name Getter methods for the fibers
    //@{
    /**
     * \brief Returns the i-th fiber vector at the Integration point
     *
     * \note Use gp=#GPDEFAULT if element fibers are used
     *
     * @param gp (in) : Id of the integration point (use #GPDEFAULT for Element fibers)
     * @param i (in) : Id of the fiber
     * @return Reference to the vector of the fiber
     */
    const Core::LinAlg::Matrix<3, 1>& get_fiber(int gp, int i) const override;

    /**
     * \brief Returns the i-th structural tensor at the Integration point in stress-like Voigt
     * notation
     *
     * \note Use gp=#GPDEFAULT if element fibers are used
     *
     * @param gp (in) : Id of the integration point (use #GPDEFAULT for Element fibers)
     * @param i (in) : Id of the fiber
     * @return Martix of the structural tensor in stress-like Voigt notation
     */
    const Core::LinAlg::Matrix<6, 1>& get_structural_tensor_stress(int gp, int i) const override;

    /**
     * \brief Returns the i-th structural tensor at the Integration point in tensor notation
     *
     * \note Use gp=#GPDEFAULT if element fibers are used
     *
     * @param gp (in) : Id of the integration point (use #GPDEFAULT for Element fibers)
     * @param i (in) : Id of the fiber
     * @return Reference to Matrix of the structural tensor in tensor notation
     */
    const Core::LinAlg::Matrix<3, 3>& get_structural_tensor(int gp, int i) const override;
    //@}

    /*!
     * \brief Needed structural tensors should be registered here before setup so that only needed
     * structural tensors are computed
     *
     * Use FiberAnisotropyExtension::FIBER_VECTORS, FiberAnisotropyExtension::STRUCTURAL_TENSOR or
     * FiberAnisotropyExtension::STRUCTURAL_TENSOR_STRESS
     *
     * They can be combined with binary or, hence
     * FiberAnisotropyExtension::FIBER_VECTORS|FiberAnisotropyExtension::STRUCTURAL_TENSOR
     *
     * \param tensor_flags Flags of the needed structural tensors
     */
    void register_needed_tensors(std::uint_fast8_t tensor_flags) { tensor_flags_ |= tensor_flags; }

    /*!
     * \brief Returns the Id of the fiber to be returned at the Gauss point. If Element fibers are
     * used, this returns FiberAnisotropyExtension::GPDEFAULT. In case of Gauss point fibers, it
     * returns the Gauss point
     *
     * \param gp Current Gauss point
     *
     * \return int
     */
    int get_virtual_gauss_point(int gp) const;

    /*!
     * \brief Returns the number of fibers per element. If Element fiber are used, it returns 1. In
     * case of Gauss point fibers, it is the number of Gauss points
     *
     * \return int
     */
    int get_fibers_per_element() const;

   protected:
    /*!
     * \brief Set all fibers at the Gauss point gp
     *
     * \param gp Gauss point. Use FiberAnisotropyExtension::GPDEFAULT in case of element fibers
     * \param fibers Vector of all fibers
     */
    void set_fibers(int gp, const std::array<Core::LinAlg::Matrix<3, 1>, numfib>& fibers);

    /*!
     * \brief Set all fibers of the element
     *
     * \param fibers The first index are the Gauss points, the second index the fibers. In case of
     * element fiebers, the first vector should only contain one element.
     */
    void set_fibers(const std::vector<std::array<Core::LinAlg::Matrix<3, 1>, numfib>>& fibers);

    /*!
     * \brief Method that compute all structural tensors. Should be executed after a change of the
     * fibers.
     */
    void compute_needed_structural_tensors();

    /*!
     * \brief Method that initializes element fibers.
     *
     * This method should only return true if element fibers are used.
     *
     * \return true if the fibers are initialized
     * \return false if the fibers are not initialized
     */
    virtual bool do_element_fiber_initialization() { return false; }

    /*!
     * \brief Method that initialized Gauss point fibers.
     *
     * This method should only return true if Gauss point fibers are used.
     *
     * \return true if the fibers are initialized
     * \return false if the fibers are not initialized
     */
    virtual bool do_gp_fiber_initialization() { return false; }

    /*!
     * \brief Method that will be called of the fibers are initialized.
     */
    virtual void on_fibers_initialized()
    {
      // do nothing in the default case
    }

    /*!
     * \brief Set the flag where fibers lie (Gausspoint or Element fibers)
     *
     * \param location
     */
    void set_fiber_location(FiberLocation location);

    /*!
     * \brief Returns the location where the fibers are stored (elemet/node)
     *
     * \return FiberLocation
     */
    virtual FiberLocation get_fiber_location() const { return fiber_location_; }

    /*!
     * \brief This method will be called by Mat::Anisotropy if element and Gauss point fibers are
     * available
     */
    void on_global_data_initialized() override {}

   private:
    /*!
     * \brief This method will be called by Mat::Anisotropy to notify that element information is
     * available.
     */
    void on_global_element_data_initialized() override
    {
      const bool initialized = do_element_fiber_initialization();
      if (initialized) on_fibers_initialized();
    }

    /*!
     * \brief This method will be called by Mat::Anisotropy to notify that Gauss point information
     * is available.
     */
    void on_global_gp_data_initialized() override
    {
      const bool initialized = do_gp_fiber_initialization();
      if (initialized) on_fibers_initialized();
    }
    /// \}

    /*!
     * \brief Method that computes structural tensors from all given fibers
     */
    void compute_structural_tensors();

    /*!
     * \brief Method that computes all structural tensors in stress like Voigt notation from all
     * given fibers.
     */
    void compute_structural_tensors_stress();

    /// Indication of the fiber location
    FiberLocation fiber_location_ = FiberLocation::None;

    /// Tensors needed for the evaluation
    std::uint_fast8_t tensor_flags_{};

    /**
     * Fibers of the element. The first index is for the Gauss points, the second index is for
     * the fiber id
     */
    std::vector<std::array<Core::LinAlg::Matrix<3, 1>, numfib>> fibers_;

    /**
     * Structural tensors of the fibers in stress like Voigt notation. The ordering is the same as
     * in #fibers_
     */
    std::vector<std::array<Core::LinAlg::Matrix<6, 1>, numfib>> fiber_structural_tensors_stress_;

    /**
     * Structural tensors of the fibers. The ordering is the same as in #fibers_
     */
    std::vector<std::array<Core::LinAlg::Matrix<3, 3>, numfib>> fiber_structural_tensors_;

    /// Structural tensor strategy
    const std::shared_ptr<Elastic::StructuralTensorStrategyBase> structural_tensor_strategy_ =
        nullptr;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
