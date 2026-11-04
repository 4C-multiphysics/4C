// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_fluidporo.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat::FLUIDPORO
{
  /*! @brief Compute structure tensor from a given direction vector
   *
   * Templated for the number of spatial dimensions, compute the structure tensor corresponding to a
   * direction vector taking the dyadic product of the vector with itself.
   *
   * @param [in] direction_vector     Non-zero direction vector (not necessarily unit-length)
   *                                  (@p dim dimensional vector)
   * @param [out] structure_tensor    Normalized structure tensor of the direction vector
   *                                  (@p dim x @p dim dyadic)
   */
  template <unsigned int dim>
  void create_structure_tensor_from_vector(
      const std::vector<double>& direction_vector, Core::LinAlg::Matrix<dim, dim>& structure_tensor)
  {
    structure_tensor.clear();

    // Factor to normalize the structure tensor
    const double square_length = std::inner_product(
        direction_vector.begin(), direction_vector.end(), direction_vector.begin(), 0.0);

    if (square_length > 1e-8)
    {
      for (unsigned int i = 0; i < dim; ++i)
      {
        for (unsigned int j = 0; j < dim; ++j)
        {
          structure_tensor(i, j) = direction_vector[i] * direction_vector[j];
        }
      }

      structure_tensor.scale(1. / square_length);
    }
    else
      FOUR_C_THROW("Check for a zero direction vector");
  }

  /*!
   * @brief  Base class for the anisotropy strategy in the FluidPoro material.
   *
   * For materials with anisotropic permeability properties, the reaction tensor will represent
   * the type of anisotropy and therefore will be calculated in different ways. This base class
   * defines the methods that are affected by the anisotropy. These methods are responsible for
   * computations regarding the reaction tensor. Some methods are already implemented by the
   * base class. They can be overridden depending on the needs of the specific type of
   * anisotropy.
   */
  class PoroAnisotropyStrategyBase
  {
   public:
    /*!
     * @brief Simple constructor
     *
     * @param [in] params Material parameters
     */
    PoroAnisotropyStrategyBase(const Mat::PAR::FluidPoro* params) : params_(params) {};

    //! Virtual default destructor
    virtual ~PoroAnisotropyStrategyBase() = default;

    //! Compute the reaction coefficient
    virtual double compute_reaction_coeff() const = 0;

    /*!
     * @brief Compute the material reaction tensor for the fluid - 2D
     *
     * This is the base class method. Is is overridden by derived classes to compute the reaction
     * tensor depending on the respective anisotropy strategy. The reaction tensor is the inverse
     * of the permeability tensor multiplied by the dynamic viscosity of the fluid.
     *
     * @param [out] reaction_tensor    The 2D reaction tensor computed according to the
     *                                    anisotropy strategy
     * @param [in] J  determinant of the deformation gradient in case the reaction tensor is
     *                dependent on it
     * @param [in] porosity   Porosity value in case the reaction tensor is dependent on it
     * @param [in] anisotropic_permeability_directions    Principal directions of anisotropy
     *                                                    required to construct the reaction
     *                                                    tensor
     * @param [in] anisotropic_permeability_coeffs  Scaling coefficients for the permeability in
     *                                              anisotropy directions
     */
    virtual void compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor,
        const double& J, const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        const std::vector<double>& anisotropic_permeability_coeffs) const = 0;

    /*!
     * @brief Compute the material reaction tensor for the fluid - 3D
     *
     * This is the base class method. Is is overridden by derived classes to compute the reaction
     * tensor depending on the respective anisotropy strategy. The reaction tensor is the inverse
     * of the permeability tensor multiplied by the dynamic viscosity of the fluid.
     *
     * @param [out] reaction_tensor   The 3D reaction tensor computed according to the
     *                                anisotropy strategy (3x3 dyadic)
     * @param [in] J  determinant of the deformation gradient
     * @param [in] porosity   Porosity value
     * @param [in] anisotropic_permeability_directions    Principal directions of anisotropy
     *                                                    required to construct the reaction
     *                                                    tensor
     * @param [in] anisotropic_permeability_coeffs  Scaling coefficients for the permeability in
     *                                              anisotropy directions
     */
    virtual void compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor,
        const double& J, const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        const std::vector<double>& anisotropic_permeability_coeffs) const = 0;

    /*! @brief Compute the derivatives of the material reaction tensor - 2D
     *
     * In case the material reaction tensor depends on the determinant of the deformation gradient
     * and/or the porosity (e.g. for the Kozeny-Carman equation), this method and its overriding
     * methods compute the derivative tensors.
     *
     * @param [out] linreac_dphi  Derivative of the material reaction tensor w.r.t. the porosity
     * (2x2 dyadic)
     * @param [out] linreac_dJ    Derivative of the material reaction tensor w.r.t. the
     *                            determinant of the deformation gradient (2x2 dyadic)
     * @param [in] J  determinant of the deformation gradient
     * @param [in] porosity   Porosity value
     */
    virtual void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<2, 2>& linreac_dphi,
        Core::LinAlg::Matrix<2, 2>& linreac_dJ, const double& J, const double& porosity) const
    {
      // Derivative is zero, clearing the tensors is enough
      // This is the standard case, so it is implemented in the base class
      linreac_dphi.clear();
      linreac_dJ.clear();
    };

    /*! @brief Compute the derivatives of the material reaction tensor - 3D
     *
     * In case the material reaction tensor depends on the determinant of the deformation gradient
     * and/or the porosity (e.g. for the Kozeny-Carman equation), this method and its overriding
     * methods compute the derivative tensors.
     *
     * @param [out] linreac_dphi  Derivative of the material reaction tensor w.r.t. the porosity
     *                            (3x3 dyadic)
     * @param [out] linreac_dJ    Derivative of the material reaction tensor w.r.t. the
     *                            determinant of the deformation gradient (3x3 dyadic)
     * @param [in] J  determinant of the deformation gradient
     * @param [in] porosity   Porosity value
     */
    virtual void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<3, 3>& linreac_dphi,
        Core::LinAlg::Matrix<3, 3>& linreac_dJ, const double& J, const double& porosity) const
    {
      // Derivative is zero, clearing the tensors is enough
      // This is the standard case, so it is implemented in the base class
      linreac_dphi.clear();
      linreac_dJ.clear();
    };

   protected:
    //! Material parameters
    const Mat::PAR::FluidPoro* params_;
  };

  /*!
   * @brief Reaction tensor (anisotropy) strategy for isotropy.
   *
   * In the case of isotropy, both 2D and 3D simulations are allowed. Furthermore, the isotropic
   * case implements two different strategies for the permeability-porosity dependency which are
   *  -# Constant material permeability, and
   *  -# Kozeny-Carman-Equation.
   */
  class PoroIsotropyStrategy : public PoroAnisotropyStrategyBase
  {
   public:
    //! Simple constructor
    PoroIsotropyStrategy(const Mat::PAR::FluidPoro* params) : PoroAnisotropyStrategyBase(params) {};

    //! compute isotropy reaction coefficient
    double compute_reaction_coeff() const override
    {
      const double viscosity = params_->viscosity_;
      const double permeability = params_->permeability_;

      if (viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      if (permeability <= 0.0) FOUR_C_THROW("zero or negative permeability");

      // trace of the reaction tensor divided by the dimension
      const double reacoeff = viscosity / permeability;

      return reacoeff;
    };

    //! compute isotropic reaction tensor - 2D
    void compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor, const double& J,
        const double& porosity,
        [[maybe_unused]] const std::vector<std::vector<double>>&
            anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      compute_reaction_tensor_for_isotropy<2>(reaction_tensor, J, porosity,
          anisotropic_permeability_directions, anisotropic_permeability_coeffs);
    };

    //! compute isotropic reaction tensor - 3D
    void compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor, const double& J,
        const double& porosity,
        [[maybe_unused]] const std::vector<std::vector<double>>&
            anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      compute_reaction_tensor_for_isotropy<3>(reaction_tensor, J, porosity,
          anisotropic_permeability_directions, anisotropic_permeability_coeffs);
    };

    /*!
     * @brief Compute reaction tensor for isotropy
     *
     * Templated for the number of spatial dimensions, compute the specific reaction tensor in the
     * case of isotropic permeability.
     *
     * @tparam dim  Number of spatial dimensions
     * @param [out] reaction_tensor Reaction tensor (@tp dim x @p dim dyadic)
     * @param [in] J  determinant of the deformation gradient
     * @param [in] porosity   Porosity value
     * @param [in] anisotropic_permeability_directions  Unused - Principal directions of
     *                                                  anisotropy required to construct the
     *                                                  reaction tensor
     * @param [in] anisotropic_permeability_coeffs  Unused - Scaling coefficients for the
     *                                              permeability in anisotropy directions
     */
    template <unsigned int dim>
    void compute_reaction_tensor_for_isotropy(Core::LinAlg::Matrix<dim, dim>& reaction_tensor,
        const double& J, const double& porosity,
        [[maybe_unused]] const std::vector<std::vector<double>>&
            anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const
    {
      double reacoeff = compute_reaction_coeff();
      const double permeability_correction_factor = params_->permeability_correction_factor_;
      const auto permeability_function = params_->permeability_func_;

      reaction_tensor.clear();

      if (permeability_function == Mat::PAR::kozeny_carman)
      {
        reacoeff *= (1 - porosity * porosity * J * J) /
                    (porosity * porosity * porosity * J * J * J) * permeability_correction_factor;
      }

      for (unsigned int i = 0; i < dim; i++) reaction_tensor(i, i) = reacoeff;
    };

    //! compute linearization of isotropic reaction tensor - 2D
    void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<2, 2>& linreac_dphi,
        Core::LinAlg::Matrix<2, 2>& linreac_dJ, const double& J,
        const double& porosity) const override
    {
      compute_lin_mat_reaction_tensor_for_isotropy<2>(linreac_dphi, linreac_dJ, J, porosity);
    };

    //! compute linearization of isotropic reaction tensor - 3D
    void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<3, 3>& linreac_dphi,
        Core::LinAlg::Matrix<3, 3>& linreac_dJ, const double& J,
        const double& porosity) const override
    {
      compute_lin_mat_reaction_tensor_for_isotropy<3>(linreac_dphi, linreac_dJ, J, porosity);
    };

    /*!
     * @brief Compute linearization reaction tensor for isotropy
     *
     * Templated for the number of spatial dimensions, compute the specific linearizations of the
     * reaction tensor w.r.t. the porosity and J in the case of isotropic permeability. A
     * distinction between constant permeability and the Kozeny-Carman permeability function is
     * made, which results in different derivatives.
     *
     * @tparam dim  Number of spatial dimensions
     * @param [out] linreac_dphi    Derivative of the material reaction tensor w.r.t. the porosity
     *                              (@p dim x @p dim dyadic)
     * @param [out] linreac_dJ  Derivative of the material reaction tensor w.r.t. the
     *                          determinant of the deformation gradient (@p dim x @p dim dyadic)
     * @param [in] J    determinant of the deformation gradient
     * @param [in] porosity Porosity value
     */
    template <unsigned int dim>
    void compute_lin_mat_reaction_tensor_for_isotropy(Core::LinAlg::Matrix<dim, dim>& linreac_dphi,
        Core::LinAlg::Matrix<dim, dim>& linreac_dJ, const double& J, const double& porosity) const
    {
      const double reacoeff = compute_reaction_coeff();
      const double permeability_correction_factor = params_->permeability_correction_factor_;
      const auto permeability_function = params_->permeability_func_;

      linreac_dphi.clear();
      linreac_dJ.clear();

      if (permeability_function == Mat::PAR::constant)
      {
        // Permeability is not a function of porosity or J
        return;
      }
      else if (permeability_function == Mat::PAR::kozeny_carman)
      {
        // d(isotropic_mat_reactiontensor)/d(phi) = reacoeff * [(J * phi)^2 - 3] / ( J^3 * phi^4 )
        // d(isotropic_mat_reactiontensor)/d(J) = reacoeff * [(J * phi)^2 - 3] / ( J^4 * phi^3 )

        const double linreac_tmp = reacoeff * ((J * J * porosity * porosity) - 3.0) /
                                   (J * J * J * porosity * porosity * porosity) *
                                   permeability_correction_factor;
        linreac_dphi(0, 0) = linreac_tmp / porosity;
        linreac_dJ(0, 0) = linreac_tmp / J;

        for (unsigned int i = 1; i < dim; i++)
        {
          linreac_dphi(i, i) = linreac_dphi(0, 0);
          linreac_dJ(i, i) = linreac_dJ(0, 0);
        }
      }
    };
  };

  /*!
   * @brief Reaction tensor (anisotropy) strategy for constant material transverse isotropy.
   *
   * In the case of transverse isotropy, both 2D and 3D simulations can be done. For now,
   * only constant material permeabilities are allowed, i.e. there is no dependency between the
   * permeability and porosity in the material reaction tensor.
   */
  class PoroConstantMaterialTransverseIsotropyStrategy : public PoroAnisotropyStrategyBase
  {
   public:
    PoroConstantMaterialTransverseIsotropyStrategy(const Mat::PAR::FluidPoro* params)
        : PoroAnisotropyStrategyBase(params) {};

    //! compute reaction coefficient for constant material transverse isotropy
    double compute_reaction_coeff() const override
    {
      const double viscosity = params_->viscosity_;
      const double permeability = params_->permeability_;
      const double axial_permeability = params_->axial_permeability_;

      if (viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      if (permeability <= 0.0) FOUR_C_THROW("zero or negative permeability");
      if (axial_permeability <= 0.0) FOUR_C_THROW("zero or negative axial permeability");

      // trace of the 3D reaction tensor divided by 3 (even if the problem is in 2D)
      const double reacoeff = viscosity / 3. * (1. / axial_permeability + 2. / permeability);

      return reacoeff;
    };

    //! compute reaction tensor for constant material transverse isotropy - 2D
    void compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      compute_reaction_tensor_for_constant_material_transverse_isotropy<2>(reaction_tensor, J,
          porosity, anisotropic_permeability_directions, anisotropic_permeability_coeffs);
    };

    //! compute reaction tensor for constant material transverse isotropy - 3D
    void compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      compute_reaction_tensor_for_constant_material_transverse_isotropy<3>(reaction_tensor, J,
          porosity, anisotropic_permeability_directions, anisotropic_permeability_coeffs);
    };

    /*!
     * @brief Compute reaction tensor for constant material transverse isotropy
     *
     * Templated for the number of spatial dimensions, compute the specific reaction tensor in the
     * case of constant material transversely isotropic permeability.
     *
     * @tparam dim  Number of spatial dimensions
     * @param [out] reaction_tensor   Reaction tensor (@p dim x @p dim dyadic)
     * @param [in] J  determinant of the deformation gradient
     * @param [in] porosity   Porosity value
     * @param [in] anisotropic_permeability_directions    Principal directions of anisotropy
     *                                                    required to construct the reaction
     *                                                    tensor
     * @param [in] anisotropic_permeability_coeffs  Unused - Scaling coefficients for the
     *                                              permeability in anisotropy directions
     */
    template <unsigned int dim>
    void compute_reaction_tensor_for_constant_material_transverse_isotropy(
        Core::LinAlg::Matrix<dim, dim>& reaction_tensor, const double& J, const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const
    {
      const double dynamic_viscosity = params_->viscosity_;
      const double permeability = params_->permeability_;
      const double axial_permeability = params_->axial_permeability_;

      if (dynamic_viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      if (permeability <= 0.0) FOUR_C_THROW("zero or negative permeability");
      if (axial_permeability <= 0.0) FOUR_C_THROW("zero or negative axial permeability");
      if (anisotropic_permeability_directions.empty())
        FOUR_C_THROW("orthotropy directions not specified");

      reaction_tensor.clear();

      Core::LinAlg::Matrix<dim, dim> permeability_tensor(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<dim, dim> structure_tensor(Core::LinAlg::Initialization::zero);
      create_structure_tensor_from_vector<dim>(
          anisotropic_permeability_directions[0], structure_tensor);

      // Transverse component of the transversely isotropic permeability tensor
      for (unsigned int i = 0; i < dim; ++i) permeability_tensor(i, i) += permeability;

      // Axial component of the transversely isotropic permeability tensor
      permeability_tensor.update(axial_permeability - permeability, structure_tensor, 1.0);

      reaction_tensor.invert(permeability_tensor);
      reaction_tensor.scale(dynamic_viscosity);
    }
  };

  /*!
   * @brief Reaction tensor (anisotropy) strategy for constant material orthotropy.
   *
   * Orthotropy can only be used in 3D simulations. For now, only constant material permeabilities
   * are allowed, i.e. there is no dependency between the permeability and porosity in the
   * material reaction tensor.
   */
  class PoroConstantMaterialOrthotropyStrategy : public PoroAnisotropyStrategyBase
  {
   public:
    //! Simple constructor
    PoroConstantMaterialOrthotropyStrategy(const Mat::PAR::FluidPoro* params)
        : PoroAnisotropyStrategyBase(params) {};

    //! compute reaction coefficient for constant material orthotropy
    double compute_reaction_coeff() const override
    {
      const double viscosity = params_->viscosity_;
      const std::vector<double> orthotropic_permeabilities = params_->orthotropic_permeabilities_;

      if (viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      for (int dim = 0; dim < 3; ++dim)
        if (orthotropic_permeabilities[dim] <= 0.0)
          FOUR_C_THROW("zero or negative permeability in direction {}", dim + 1);

      // trace of the reaction tensor divided by 3
      const double reacoeff =
          viscosity / 3. *
          (1. / orthotropic_permeabilities[0] + 1. / orthotropic_permeabilities[1] +
              1. / orthotropic_permeabilities[2]);

      return reacoeff;
    };

    //! constant material orthotropy in 2D is not allowed
    void compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      FOUR_C_THROW("Use transverse isotropy in 2D!");
    };

    //! compute reaction tensor for constant material orthotropy - 3D
    void compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        [[maybe_unused]] const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      const double dynamic_viscosity = params_->viscosity_;
      const std::vector<double> orthotropic_permeabilities = params_->orthotropic_permeabilities_;

      if (dynamic_viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      for (int dim = 0; dim < 3; ++dim)
        if (orthotropic_permeabilities[dim] <= 0.0)
          FOUR_C_THROW("zero or negative permeability in direction {}", dim + 1);
      if (anisotropic_permeability_directions.empty())
        FOUR_C_THROW("orthotropy directions not specified");

      reaction_tensor.clear();

      Core::LinAlg::Matrix<3, 3> permeability_tensor(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<3, 3> structure_tensor(Core::LinAlg::Initialization::zero);

      for (int dim = 0; dim < 3; ++dim)
      {
        create_structure_tensor_from_vector<3>(
            anisotropic_permeability_directions[dim], structure_tensor);
        permeability_tensor.update(orthotropic_permeabilities[dim], structure_tensor, 1.0);
      }

      reaction_tensor.invert(permeability_tensor);
      reaction_tensor.scale(dynamic_viscosity);
    };

    //! constant material orthotropy in 2D is not allowed
    void compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<2, 2>& linreac_dphi,
        Core::LinAlg::Matrix<2, 2>& linreac_dJ, const double& J,
        const double& porosity) const override
    {
      // constant material orthotropy in 2D is not allowed
      FOUR_C_THROW("Use transverse isotropy in 2D!");
    };
  };

  /*!
   * @brief Reaction tensor (anisotropy) strategy for nodal material orthotropy.
   *
   * Orthotropy can only be used in 3D simulations. For now, only constant material permeabilities
   * are allowed, i.e. there is no dependency between the permeability and porosity in the
   * material reaction tensor.
   *
   * This orthotropy strategy does not obtain the different permeability values from the material
   * properties. Instead, it works with a basis permeability value from the material properties and
   * scales it in different directions with coefficients that are provided by the evaluation
   * methods. Obtaining the scaling coefficients from the evaluation methods allows to have varying
   * permeabilities inside the elements, where each element has prescribed nodal values for the
   * coefficients.
   */
  class PoroConstantMaterialNodalOrthotropyStrategy : public PoroAnisotropyStrategyBase
  {
   public:
    //! Simple constructor
    PoroConstantMaterialNodalOrthotropyStrategy(const Mat::PAR::FluidPoro* params)
        : PoroAnisotropyStrategyBase(params) {};

    //! compute reaction coefficient for constant material nodal orthotropy
    double compute_reaction_coeff() const override
    {
      const double viscosity = params_->viscosity_;
      const double permeability = params_->permeability_;

      if (viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      if (permeability <= 0.0) FOUR_C_THROW("zero or negative permeability");

      // reaction coefficient computed with the base permeability as a good approximation
      const double reacoeff = viscosity / permeability;

      return reacoeff;
    };

    //!  compute reaction tensor for constant material nodal orthotropy - 2D
    void compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      compute_reaction_tensor_for_constant_material_nodal_orthotropy<2>(reaction_tensor, J,
          porosity, anisotropic_permeability_directions, anisotropic_permeability_coeffs);
    };

    //! compute reaction tensor for constant material nodal orthotropy - 3D
    void compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor, const double& J,
        const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        const std::vector<double>& anisotropic_permeability_coeffs) const override
    {
      compute_reaction_tensor_for_constant_material_nodal_orthotropy<3>(reaction_tensor, J,
          porosity, anisotropic_permeability_directions, anisotropic_permeability_coeffs);
    };

    /*!
     * @brief Compute reaction tensor for constant material nodal orthotropy
     *
     * Templated for the number of spatial dimensions, compute the specific reaction tensor in the
     * case of constant material nodal orthotropic permeability.
     *
     * @tparam dim  Number of spatial dimensions
     * @param [out] reaction_tensor Reaction tensor (@p dim x @p dim dyadic)
     * @param [in] J  determinant of the deformation gradient
     * @param [in] porosity   Porosity value
     * @param [in] anisotropic_permeability_directions    Principal directions of anisotropy
     *                                                    required to construct the reaction
     *                                                    tensor
     * @param [in] anisotropic_permeability_coeffs  Scaling coefficients for the permeability in
     *                                              anisotropy directions
     */
    template <unsigned int dim>
    void compute_reaction_tensor_for_constant_material_nodal_orthotropy(
        Core::LinAlg::Matrix<dim, dim>& reaction_tensor, const double& J, const double& porosity,
        const std::vector<std::vector<double>>& anisotropic_permeability_directions,
        const std::vector<double>& anisotropic_permeability_coeffs) const
    {
      const double dynamic_viscosity = params_->viscosity_;
      const double permeability = params_->permeability_;

      if (dynamic_viscosity <= 0.0) FOUR_C_THROW("zero or negative viscosity");
      if (permeability <= 0.0) FOUR_C_THROW("zero or negative permeability");
      if (anisotropic_permeability_directions.empty())
        FOUR_C_THROW("orthotropy directions not specified");
      if (anisotropic_permeability_coeffs.empty())
        FOUR_C_THROW("orthotropy coefficients not specified");

      reaction_tensor.clear();

      Core::LinAlg::Matrix<dim, dim> permeability_tensor(Core::LinAlg::Initialization::zero);
      Core::LinAlg::Matrix<dim, dim> structure_tensor(Core::LinAlg::Initialization::zero);

      for (unsigned int i = 0; i < dim; ++i)
      {
        create_structure_tensor_from_vector<dim>(
            anisotropic_permeability_directions[i], structure_tensor);
        permeability_tensor.update(
            permeability * anisotropic_permeability_coeffs[i], structure_tensor, 1.0);
      }

      reaction_tensor.invert(permeability_tensor);
      reaction_tensor.scale(dynamic_viscosity);
    }
  };

  std::shared_ptr<Mat::FLUIDPORO::PoroAnisotropyStrategyBase> create_anisotropy_strategy(
      const Mat::PAR::FluidPoro* params)
  {
    switch (params->permeability_func_)
    {
      case Mat::PAR::constant:
      case Mat::PAR::kozeny_carman:
        return std::make_shared<Mat::FLUIDPORO::PoroIsotropyStrategy>(params);
      case Mat::PAR::const_material_transverse:
        return std::make_shared<Mat::FLUIDPORO::PoroConstantMaterialTransverseIsotropyStrategy>(
            params);
      case Mat::PAR::const_material_orthotropic:
        return std::make_shared<Mat::FLUIDPORO::PoroConstantMaterialOrthotropyStrategy>(params);
      case Mat::PAR::const_material_nodal_orthotropic:
        return std::make_shared<Mat::FLUIDPORO::PoroConstantMaterialNodalOrthotropyStrategy>(
            params);
      default:
        return nullptr;
    }
  }

}  // namespace Mat::FLUIDPORO

Mat::PAR::FluidPoro::FluidPoro(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      viscosity_(matdata.parameters.get<double>("DYNVISCOSITY")),
      density_(matdata.parameters.get<double>("DENSITY")),
      permeability_(matdata.parameters.get<double>("PERMEABILITY")),
      axial_permeability_(matdata.parameters.get<double>("AXIALPERMEABILITY")),
      type_(undefined),
      varying_permeability_(false),
      permeability_func_(Mat::PAR::pf_undefined),
      permeability_correction_factor_(1.0),
      initial_porosity_(1.0)
{
  const auto& typestring = matdata.parameters.get<std::string>("TYPE");

  if (typestring == "Darcy")
    type_ = darcy;
  else if (typestring == "Darcy-Brinkman")
    type_ = darcy_brinkman;

  const auto& pfuncstring = matdata.parameters.get<std::string>("PERMEABILITYFUNCTION");

  if (pfuncstring == "Const")
    permeability_func_ = Mat::PAR::constant;
  else if (pfuncstring == "Kozeny_Carman")
    permeability_func_ = Mat::PAR::kozeny_carman;
  else if (pfuncstring == "Const_Material_Transverse")
    permeability_func_ = Mat::PAR::const_material_transverse;
  else if (pfuncstring == "Const_Material_Orthotropy")
    permeability_func_ = Mat::PAR::const_material_orthotropic;
  else if (pfuncstring == "Const_Material_Nodal_Orthotropy")
    permeability_func_ = Mat::PAR::const_material_nodal_orthotropic;
  else
    FOUR_C_THROW("Unknown permeability function: {}", pfuncstring);

  orthotropic_permeabilities_.resize(3, 0.0);
  if (permeability_func_ == Mat::PAR::const_material_orthotropic)
  {
    for (int dim = 0; dim < 3; ++dim)
      orthotropic_permeabilities_[dim] =
          matdata.parameters.get<double>("ORTHOPERMEABILITY" + std::to_string(dim + 1));
  }
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::FluidPoro::create_material()
{
  return std::make_shared<Mat::FluidPoro>(this);
}

void Mat::PAR::FluidPoro::set_initial_porosity(double initial_porosity)
{
  initial_porosity_ = initial_porosity;

  if (permeability_func_ == Mat::PAR::kozeny_carman)
  {
    // c = (phi0^3 / (1 - phi0^2))
    permeability_correction_factor_ = initial_porosity_ * initial_porosity_ * initial_porosity_ /
                                      (1 - initial_porosity_ * initial_porosity_);
  }
  else
  {
    permeability_correction_factor_ = 1.0;
  }
}

Mat::FluidPoroType Mat::FluidPoroType::instance_;

Core::Communication::ParObject* Mat::FluidPoroType::create(
    Core::Communication::UnpackBuffer& buffer)
{
  auto* fluid_poro = new Mat::FluidPoro();
  fluid_poro->unpack(buffer);
  return fluid_poro;
}

Mat::FluidPoro::FluidPoro() : params_(nullptr), anisotropy_strategy_(nullptr) {}

Mat::FluidPoro::FluidPoro(Mat::PAR::FluidPoro* params) : params_(params)
{
  anisotropy_strategy_ = Mat::FLUIDPORO::create_anisotropy_strategy(params);
}

void Mat::FluidPoro::pack(Core::Communication::PackBuffer& data) const
{
  // pack type of this instance of ParObject
  int type = unique_par_object_id();
  add_to_pack(data, type);

  // matid
  int matid = -1;
  if (params_ != nullptr) matid = params_->id();  // in case we are in post-process mode
  add_to_pack(data, matid);
}

void Mat::FluidPoro::unpack(Core::Communication::UnpackBuffer& buffer)
{
  Core::Communication::extract_and_assert_id(buffer, unique_par_object_id());

  // matid
  int matid;
  extract_from_pack(buffer, matid);
  params_ = nullptr;
  if (Global::Problem::instance()->materials() != nullptr)
  {
    if (Global::Problem::instance()->materials()->num() != 0)
    {
      const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
      Core::Mat::PAR::Parameter* mat =
          Global::Problem::instance(probinst)->materials()->parameter_by_id(matid);
      if (mat->type() == material_type())
        params_ = static_cast<Mat::PAR::FluidPoro*>(mat);
      else
        FOUR_C_THROW("Type of parameter material {} does not fit to calling type {}", mat->type(),
            material_type());
    }
  }

  // Only execute if not in post-process mode
  if (params_ != nullptr)
    anisotropy_strategy_ = Mat::FLUIDPORO::create_anisotropy_strategy(params_);
}

double Mat::FluidPoro::compute_reaction_coeff() const
{
  return anisotropy_strategy_->compute_reaction_coeff();
}

void Mat::FluidPoro::compute_reaction_tensor(Core::LinAlg::Matrix<2, 2>& reaction_tensor,
    const double& J, const double& porosity,
    const std::vector<std::vector<double>>& anisotropic_permeability_directions,
    const std::vector<double>& anisotropic_permeability_coeffs) const
{
  anisotropy_strategy_->compute_reaction_tensor(reaction_tensor, J, porosity,
      anisotropic_permeability_directions, anisotropic_permeability_coeffs);
}

void Mat::FluidPoro::compute_reaction_tensor(Core::LinAlg::Matrix<3, 3>& reaction_tensor,
    const double& J, const double& porosity,
    const std::vector<std::vector<double>>& anisotropic_permeability_directions,
    const std::vector<double>& anisotropic_permeability_coeffs) const
{
  anisotropy_strategy_->compute_reaction_tensor(reaction_tensor, J, porosity,
      anisotropic_permeability_directions, anisotropic_permeability_coeffs);
}

void Mat::FluidPoro::compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<2, 2>& linreac_dphi,
    Core::LinAlg::Matrix<2, 2>& linreac_dJ, const double& J, const double& porosity) const
{
  anisotropy_strategy_->compute_lin_mat_reaction_tensor(linreac_dphi, linreac_dJ, J, porosity);
}

void Mat::FluidPoro::compute_lin_mat_reaction_tensor(Core::LinAlg::Matrix<3, 3>& linreac_dphi,
    Core::LinAlg::Matrix<3, 3>& linreac_dJ, const double& J, const double& porosity) const
{
  anisotropy_strategy_->compute_lin_mat_reaction_tensor(linreac_dphi, linreac_dJ, J, porosity);
}

double Mat::FluidPoro::effective_viscosity() const
{
  // set zero viscosity and only modify it for Darcy-Stokes problems
  if (type() == PAR::darcy)
    return 0.0;
  else if (type() == PAR::darcy_brinkman)
    return viscosity();
  else
    FOUR_C_THROW("Unknown flow type for porous flow");
}

FOUR_C_NAMESPACE_CLOSE
