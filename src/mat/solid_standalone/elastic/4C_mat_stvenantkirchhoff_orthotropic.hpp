// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_STVENANTKIRCHHOFF_ORTHOTROPIC_HPP
#define FOUR_C_MAT_STVENANTKIRCHHOFF_ORTHOTROPIC_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_input_field.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /// material parameters for orthotropic St. Venant--Kirchhoff
    class StVenantKirchhoffOrthotropic : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      StVenantKirchhoffOrthotropic(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Young's moduli E1, E2, E3 along principal directions
      const Core::IO::InputField<std::array<double, 3>> youngs_;
      /// Shear moduli G12, G23, G13
      const Core::IO::InputField<std::array<double, 3>> shear_;
      /// Possion's ratios nu12, nu23, nu13
      const Core::IO::InputField<std::array<double, 3>> poissonratio_;
      /// Mass density
      const double density_;

      //@}

      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class StVenantKirchhoffOrthotropic
  }  // namespace PAR

  class StVenantKirchhoffOrthotropicType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "StVenantKirchhoffOrthotropicType"; }

    static StVenantKirchhoffOrthotropicType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static StVenantKirchhoffOrthotropicType instance_;
  };

  /*
   * Wrapper for St.-Venant-Kirchhoff material with an orthotropic material tensor.
   * The behavior of the material is governed by 9 independent parameters, 3 Young's moduli,
   * 3 shear moduli and 3 Poisson ratios.
   */
  class StVenantKirchhoffOrthotropic : public So3Material
  {
   public:
    /// construct empty material object
    StVenantKirchhoffOrthotropic();

    /// construct the material object given material parameters
    explicit StVenantKirchhoffOrthotropic(Mat::PAR::StVenantKirchhoffOrthotropic* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return StVenantKirchhoffOrthotropicType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! @name Access methods

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_orthostvenant;
    }

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::linear && kinem != Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<StVenantKirchhoffOrthotropic>(*this);
    }

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //@}

    //! @name Evaluation methods

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const EvaluationContext<3>& context, int gp, int eleGID) const override;
    //@}

    static constexpr Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_stress(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat)
    {
      return Core::LinAlg::ddot(cmat, glstrain);
    }


    /**
     * @brief Build the stiffness tensor used for stress linearization of an orthotropic
     * St.-Venant-Kirchhoff material.
     *
     * This function constructs the 4th-order elastic stiffness tensor \f$\mathbb{C}\f$  from
     * orthotropic engineering constants: Young's moduli \f$E_1,E_2,E_3\f$, shear moduli
     * \f$G_{12},G_{23},G_{13}\f$, and Poisson's ratios
     * \f$\nu_{12},\nu_{23},\nu_{13}\f$.
     *
     * The tensor is populated via a 6x6 stress-like Voigt view:
     * \f[
     *   \mathbf{C}_\text{Voigt} =
     *   \begin{bmatrix}
     *     C_{11} & C_{12} & C_{13} & 0      & 0      & 0 \\
     *     C_{21} & C_{22} & C_{23} & 0      & 0      & 0 \\
     *     C_{31} & C_{32} & C_{33} & 0      & 0      & 0 \\
     *     0      & 0      & 0      & G_{23} & 0      & 0 \\
     *     0      & 0      & 0      & 0      & G_{13} & 0 \\
     *     0      & 0      & 0      & 0      & 0      & G_{12}
     *   \end{bmatrix}.
     * \f]
     *
     * The reciprocal Poisson ratios are computed using orthotropic reciprocity:
     * \f[
     *   \nu_{21}=\nu_{12}\frac{E_2}{E_1},\quad
     *   \nu_{32}=\nu_{23}\frac{E_3}{E_2},\quad
     *   \nu_{31}=\nu_{13}\frac{E_3}{E_1}.
     * \f]
     *
     * A common denominator is used for the normal components:
     * \f[
     *   \text{factor} =
     *   1 - \nu_{12}\nu_{21} - \nu_{23}\nu_{32} - \nu_{31}\nu_{13} - 2\nu_{12}\nu_{23}\nu_{31}.
     * \f]
     *
     * @param[in] E  Young's moduli \f$(E_1,E_2,E_3)\f$ in material principal directions.
     * @param[in] G  Shear moduli \f$(G_{12},G_{23},G_{13})\f$ (ordering as used by the
     * implementation).
     * @param[in] nu Poisson ratios \f$(\nu_{12},\nu_{23},\nu_{13})\f$ (ordering as used by the
     * implementation).
     *
     * @return The orthotropic stiffness tensor \f$\mathbb{C}\f$ as a symmetric 4th-order tensor.
     */
    static Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> evaluate_stress_linearization(
        const std::array<double, 3>& E, const std::array<double, 3>& G,
        const std::array<double, 3>& nu)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat{};
      Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

      const double E1 = E[0];
      const double E2 = E[1];
      const double E3 = E[2];

      const double G12 = G[0];
      const double G23 = G[1];
      const double G13 = G[2];

      const double nu12 = nu[0];
      const double nu23 = nu[1];
      const double nu13 = nu[2];

      const double nu21 = nu12 * (E2 / E1);
      const double nu32 = nu23 * (E3 / E2);
      const double nu31 = nu13 * (E3 / E1);

      const double inverse_factor =
          1. / (1 - nu12 * nu21 - nu23 * nu32 - nu31 * nu13 - 2 * nu12 * nu23 * nu31);

      // write non-zero components
      cmat_view(0, 0) = E1 * (1 - nu23 * nu32) * inverse_factor;
      cmat_view(0, 1) = E1 * (nu21 + nu31 * nu23) * inverse_factor;
      cmat_view(0, 2) = E1 * (nu31 + nu21 * nu32) * inverse_factor;
      cmat_view(1, 0) = E2 * (nu12 + nu13 * nu32) * inverse_factor;
      cmat_view(1, 1) = E2 * (1 - nu31 * nu13) * inverse_factor;
      cmat_view(1, 2) = E2 * (nu32 + nu12 * nu31) * inverse_factor;
      cmat_view(2, 0) = E3 * (nu13 + nu12 * nu23) * inverse_factor;
      cmat_view(2, 1) = E3 * (nu23 + nu21 * nu13) * inverse_factor;
      cmat_view(2, 2) = E3 * (1 - nu12 * nu21) * inverse_factor;
      // ~~~
      cmat_view(3, 3) = G23;
      cmat_view(4, 4) = G13;
      cmat_view(5, 5) = G12;

      return cmat;
    }


   private:
    /// my material parameters
    Mat::PAR::StVenantKirchhoffOrthotropic* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
