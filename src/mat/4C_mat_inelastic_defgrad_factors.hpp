// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_HPP
#define FOUR_C_MAT_INELASTIC_DEFGRAD_FACTORS_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_comm_pack_helpers.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_densematrix_funct.hpp"
#include "4C_linalg_utils_tensor_interpolation.hpp"
#include "4C_mat_elast_couptransverselyisotropic.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_multiplicative_split_defgrad_elasthyper.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <boost/graph/visitors.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Discret::Utils
{
  class FunctionOfTime;
}

namespace Mat
{
  namespace PAR
  {
    enum class InelasticSource;

    /*----------------------------------------------------------------------*/
    /*! \class InelasticDeformationDirection
     *
     * Calculates and holds growth direction in matrix format for anisotropic growth
     */
    class InelasticDeformationDirection
    {
     public:
      /*!
       * @brief standard constructor
       * @param[in] growthdirection  direction of anisotropic growth
       */
      explicit InelasticDeformationDirection(std::vector<double> growthdirection);

      /// reference to matrix that determines growth direction
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& growth_dir_mat() const
      {
        return growth_dir_mat_;
      }

     private:
      /// matrix that determines growth direction
      Core::LinAlg::SymmetricTensor<double, 3, 3> growth_dir_mat_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradNoGrowth
     *
     * This is a parameter class that is only needed to implement the pure virtual method
     * 'create_material()'.
     */
    class InelasticDefgradNoGrowth : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradNoGrowth(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }
    };


    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradScalar
     *
     * This is a parameter class holding parameters for evaluation of inelastic deformation (incl.
     * linearization) induced by a scalar.
     */
    class InelasticDefgradScalar : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradScalar(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }

      /// scalar that causes growth
      int scalar1() const { return scalar1_; }

      //! concentration, at which no growth occurs
      double scalar1_ref_conc() const { return scalar1_ref_conc_; }

     private:
      /// scalar that causes growth
      const int scalar1_;

      //! concentration, at which no growth occurs
      const double scalar1_ref_conc_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradTimeFunct
     *
     * This is a parameter class holding parameters for evaluation of inelastic deformation induced
     * by a given time-dependent function.
     */
    class InelasticDefgradTimeFunct : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradTimeFunct(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; }

      /// function number that sets determinant of inelastic def. grad.
      int funct_num() const { return funct_num_; }

     private:
      /// function number that sets determinant of inelastic def. grad.
      const int funct_num_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradLinScalar
     *
     * This is a specialized parameter class that holds the growth factor for linear growth
     */
    class InelasticDefgradLinScalar : public InelasticDefgradScalar
    {
     public:
      /// standard constructor
      explicit InelasticDefgradLinScalar(const Core::Mat::PAR::Parameter::Data& matdata);

      //! molar factor that causes growth
      double scalar1_molar_growth_fac() { return scalar1_molar_growth_fac_; }

     private:
      //! molar factor that causes growth
      const double scalar1_molar_growth_fac_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradLinScalarAniso
     *
     * This is a specialized parameter class that can return the anisotropic growth direction
     * represented as a growth matrix
     */
    class InelasticDefgradLinScalarAniso : public InelasticDefgradLinScalar
    {
     public:
      /// standard constructor
      explicit InelasticDefgradLinScalarAniso(const Core::Mat::PAR::Parameter::Data& matdata);

      /// reference to matrix that determines growth direction
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& growth_dir_mat()
      {
        return growth_dir_->growth_dir_mat();
      }

     private:
      /// calculation of direction of inelastic deformation
      std::shared_ptr<const InelasticDeformationDirection> growth_dir_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradIntercalFrac
     *
     * This parameter class provides all electrochemical quantities that are needed to calculate the
     * intercalation fraction from a given species concentration.
     */
    class InelasticDefgradIntercalFrac : public InelasticDefgradScalar
    {
     public:
      explicit InelasticDefgradIntercalFrac(const Core::Mat::PAR::Parameter::Data& matdata);

      /// saturation concentration of material
      double cmax() const { return c_max_; }
      /// intercalation fraction at saturation concentration of material
      double chimax() const { return chi_max_; }

     private:
      /// saturation concentration of material
      double c_max_;
      /// intercalation fraction at saturation concentration of material
      double chi_max_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradPolyIntercalFrac
     *
     * This parameter class provides the value of the polynomial that models the growth evaluated in
     * the reference configuration.
     */
    class InelasticDefgradPolyIntercalFrac : public InelasticDefgradIntercalFrac
    {
     public:
      explicit InelasticDefgradPolyIntercalFrac(const Core::Mat::PAR::Parameter::Data& matdata);

      /// return value of polynomial at reference intercalation fraction
      double get_polynom_reference_value() const { return polynom_reference_value_; }

      // set value of polynomial at reference intercalation fraction
      void set_polynom_reference_value(double polynomReferenceValue)
      {
        polynom_reference_value_ = polynomReferenceValue;
      }

      //! polynomial coefficients that describe the growth law
      std::vector<double> poly_coeffs() const { return poly_coeffs_; }

      //! upper bound of polynomial
      double x_max() const { return x_max_; }

      //! lower bound of polynomial
      double x_min() const { return x_min_; }

     private:
      const std::vector<double> poly_coeffs_;

      /// value of polynomial at reference intercalation fraction
      double polynom_reference_value_;

      //! upper bound of polynomial
      const double x_max_;

      //! lower bound of polynomial
      const double x_min_;
    };

    /*----------------------------------------------------------------------
     *----------------------------------------------------------------------*/
    /*! \class InelasticDefgradPolyIntercalFracAniso
     *
     * This is a specialized parameter class that can return the anisotropic growth direction
     * represented as a growth matrix
     */
    class InelasticDefgradPolyIntercalFracAniso : public InelasticDefgradPolyIntercalFrac
    {
     public:
      /// standard constructor
      explicit InelasticDefgradPolyIntercalFracAniso(
          const Core::Mat::PAR::Parameter::Data& matdata);

      /// return reference to matrix that determines growth direction
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& growth_dir_mat() const
      {
        return growth_dir_->growth_dir_mat();
      };

     private:
      /// pointer to object, that calculates and holds direction of inelastic deformation
      std::shared_ptr<InelasticDeformationDirection> growth_dir_;
    };

    /*----------------------------------------------------------------------
    ----------------------------------------------------------------------*/
    /*! \class InelasticDefgradLinTempIso

    Parameter class of InelasticDefgradLinTempIso.
    */
    class InelasticDefgradLinTempIso : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradLinTempIso(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      /// return temperature related growth factor
      double get_temp_growth_fac() const { return temp_growth_fac_; };

      /// return value of temperature that causes no growth
      double ref_temp() const { return ref_temp_; };

     private:
      /// value of temperature that causes no growth
      const double ref_temp_;

      /// growth factor
      const double temp_growth_fac_;
    };

    /*----------------------------------------------------------------------
    ---------------------------------------------------------------------*/
    /*! \class InelasticDefgradTransvIsotropElastViscoplast
     * Parameter class of InelasticDefgradTransvIsotropElastViscoplast.
     */
    class InelasticDefgradTransvIsotropElastViscoplast : public Core::Mat::PAR::Parameter
    {
     public:
      explicit InelasticDefgradTransvIsotropElastViscoplast(
          const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

      //! get ID of the viscoplasticity law
      [[nodiscard]] int viscoplastic_law_id() const { return viscoplastic_law_id_; };
      //! get global ID of the fiber reader material
      [[nodiscard]] int fiber_reader_gid() const { return fiber_reader_gid_; };
      //! get yield condition parameter \f[ A \f]
      [[nodiscard]] double yield_cond_a() const { return yield_cond_a_; };
      //! get yield condition parameter \f[ B \f]
      [[nodiscard]] double yield_cond_b() const { return yield_cond_b_; };
      //! get yield condition parameter \f[ F \f]
      [[nodiscard]] double yield_cond_f() const { return yield_cond_f_; };
      //! get material behavior
      [[nodiscard]] InelasticDefgradTransvIsotropElastViscoplastUtils::MatBehavior mat_behavior()
          const
      {
        return mat_behavior_;
      };
      //! get maximum number of times a time step can be halved into smaller and smaller
      //! substeps
      [[nodiscard]] unsigned int max_halve_number() const
      {
        return static_cast<unsigned int>(max_substepping_halve_num_);
      }
      //! get the type of time integration for the evolution equations
      //! of history variables
      [[nodiscard]] InelasticDefgradTransvIsotropElastViscoplastUtils::TimIntType timint_type()
          const
      {
        return timint_type_;
      };
      //! get the type of material linearization used
      [[nodiscard]] InelasticDefgradTransvIsotropElastViscoplastUtils::LinearizationType
      linearization_type() const
      {
        return linearization_type_;
      };
      //! get maximum, numerically evaluable plastic strain increment
      [[nodiscard]] double max_plastic_strain_incr() const { return max_plastic_strain_incr_; };
      //! get maximum, numerically evaluable value for the increment of
      //! the plastic strain derivatives (dt * derivative)
      [[nodiscard]] double max_plastic_strain_deriv_incr() const
      {
        return max_plastic_strain_deriv_incr_;
      }
      //! get computation method for the matrix exponential
      [[nodiscard]] Core::LinAlg::MatrixExpCalcMethod mat_exp_calc_method() const
      {
        return mat_exp_calc_method_;
      }
      //! get computation method for the first derivative of the matrix exponential
      [[nodiscard]] Core::LinAlg::GenMatrixExpFirstDerivCalcMethod mat_exp_deriv_calc_method() const
      {
        return mat_exp_deriv_calc_method_;
      }
      //! get computation method for the matrix logarithm
      [[nodiscard]] Core::LinAlg::MatrixLogCalcMethod mat_log_calc_method() const
      {
        return mat_log_calc_method_;
      }
      //! get computation method for the first derivative of the matrix logarithm
      [[nodiscard]] Core::LinAlg::GenMatrixLogFirstDerivCalcMethod mat_log_deriv_calc_method() const
      {
        return mat_log_deriv_calc_method_;
      }

     private:
      //! ID of the viscoplasticity law
      const int viscoplastic_law_id_;

      //! global ID of the material used for fiber reading (transversely isotropic)
      const int fiber_reader_gid_;

      //! yield condition parameter \f[ A \f]
      const double yield_cond_a_;
      //! yield condition parameter \f[ B \f]
      const double yield_cond_b_;
      //! yield condition parameter \f[ F \f]
      const double yield_cond_f_;

      //! material behavior (transversely isotropic | isotropic )
      const InelasticDefgradTransvIsotropElastViscoplastUtils::MatBehavior mat_behavior_;

      //! computation method for the time integration of the
      //! history variables (standard | logarithmic)
      const InelasticDefgradTransvIsotropElastViscoplastUtils::TimIntType timint_type_;

      //! linearization method (analytic | perturbation based)
      InelasticDefgradTransvIsotropElastViscoplastUtils::LinearizationType linearization_type_;

      //! maximum, numerically evaluable plastic strain increment
      const double max_plastic_strain_incr_;

      //! maximum, numerically evaluable increment of
      //! plastic strain derivatives (time_step * derivative)
      const double max_plastic_strain_deriv_incr_;

      //! maximum number of times the given time step can be halved before reaching the minimum
      //! allowed substep length
      const int max_substepping_halve_num_;

      //! utilized computation method for the matrix exponential
      const Core::LinAlg::MatrixExpCalcMethod mat_exp_calc_method_;

      //! utilized computation method for the first derivative of the matrix exponential
      const Core::LinAlg::GenMatrixExpFirstDerivCalcMethod mat_exp_deriv_calc_method_;

      //! utilized computation method for the matrix logarithm
      const Core::LinAlg::MatrixLogCalcMethod mat_log_calc_method_;

      //! utilized computation method for the first derivative of the matrix logarithm
      const Core::LinAlg::GenMatrixLogFirstDerivCalcMethod mat_log_deriv_calc_method_;
    };
  }  // namespace PAR


  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinearShape
   *
   * This class provides the functionality to be used if the growth law obeys a linear relation
   */
  class InelasticDefgradLinearShape
  {
   public:
    /*!
     * @brief constructor with required parameters
     *
     * @param[in] growth_fac       linear growth factor (slope of linear function)
     * @param[in] reference_value  reference value
     */
    explicit InelasticDefgradLinearShape(double growth_fac, double reference_value);

    /*!
     * @brief evaluation of the linear growth law
     *
     * @param[in] value           value the linear relation shall be evaluated for
     * @return growth factor
     */
    double evaluate_linear_growth(double value) const;

    /// growth factor (needed for linearizations)
    double growth_fac() const { return growth_fac_; }

   private:
    /// growth factor
    const double growth_fac_;

    /// reference value
    const double reference_value_;
  };  // namespace Mat

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolynomialShape
   *
   * This class provides the functionality to be used if the growth law obeys a polynomial relation
   */
  class InelasticDefgradPolynomialShape
  {
   public:
    /*!
     * @brief  constructor with required parameters
     *
     * @param[in] poly_coeffs  coefficients describing the polynomial to be evaluated
     * @param[in] x_min        lower bound of validity of the polynomial
     * @param[in] x_max        upper bound of validity of the polynomial
     */
    explicit InelasticDefgradPolynomialShape(
        std::vector<double> poly_coeffs, double x_min, double x_max);

    /*!
     * @brief checks the bounds of validity of the polynomial and writes a warning to screen if
     * bounds are violated
     *
     * @param[in] x  value the polynomial is evaluated at
     */
    void check_polynomial_bounds(double x) const;

    /*!
     * @brief Evaluate the polynomial defined by #PolyCoeffs_ at the current position X
     *
     * @param[in] x  value the polynomial is evaluated at
     * @return value of the polynomial evaluated at x
     */
    double compute_polynomial(double x);

    /*!
     * @brief Evaluate the first derivative of the polynomial defined by #PolyCoeffs_ at the current
     * position x
     *
     * @param[in] x  value the polynomial is evaluated at
     * @return value the first derivative of the polynomial evaluated at x
     */
    double compute_polynomial_derivative(double x);

   private:
    /// coefficients of the polynomial to be evaluated
    const std::vector<double> poly_coeffs_;
    /// lower bound of validity of polynomial
    const double x_min_;
    /// upper bound of validity of polynomial
    const double x_max_;
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradFactors

      Provides the interface called by the class "MultiplicativeSplitDefgrad_ElastHyper"
      and is needed to evaluate the inelastic deformation gradient and
      their derivatives w.r.t. the primary variables.

      In the material "MultiplicativeSplitDefgrad_ElastHyper" the deformation gradient is split
      multiplicatively in elastic and inelastic deformation gradients (F = F_{el} * F_{in}).
      The inelastic deformation gradient itself can be a product of different inelastic
      deformation gradients, i.e. F_{in} = F_{in,1} * F_{in,2} * ... * F_{in,n}.
      The derived classes below are needed to evaluate the inverse of the j-th inelastic
      deformation gradient F_{in,j}^{-1} and its derivatives w.r.t. the primary variables.
  */
  class InelasticDefgradFactors
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~InelasticDefgradFactors() = default;

    /// construct material with specific material params
    explicit InelasticDefgradFactors(Core::Mat::PAR::Parameter* params);

    /*!
     * @brief create object by input parameter ID
     *
     * @param[in] matnum  material ID
     * @return pointer to material that is defined by material ID
     */
    static std::shared_ptr<InelasticDefgradFactors> factory(int matnum);

    /// provide material type
    virtual Core::Materials::MaterialType material_type() const = 0;

    /*!
     * @brief evaluate the inelastic deformation gradient and its inverse
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] iFin_other Already computed inverse inelastic deformation gradient
     *              (from already computed inelastic factors in the multiplicative split material)
     * @param[out] iFinM   Inverse inelastic deformation gradient
     */
    virtual void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) = 0;

    /*!
     * @brief evaluate additional terms for the elasticity tensor
     *
     * @param[in] defgrad  Deformation gradient
     * @param[in] iFin_other Already computed inverse inelastic deformation gradient
     *              (from already computed inelastic factors in the multiplicative split material)
     * @param[in] iFinjM   Inverse inelastic deformation gradient of current inelastic contribution
     *                     as 3x3 matrix
     * @param[in] iCV      Inverse right Cauchy-Green tensor
     * @param[in] dSdiFinj Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                     deformation gradient of current inelastic contribution
     * @param[in,out] cmatadd  Additional elasticity tensor
     */
    virtual void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) = 0;

    /*!
     * @brief calculate the derivative of the inelastic deformation gradient
     *
     * @param[in] detjacobian  determinant of the deformation gradient
     * @param[out] dFindx      derivative of inelastic deformation gradient w.r.t. primary variable
     *                         of different field
     */
    virtual void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) = 0;

    /*!
     * @brief evaluate off-diagonal stiffness matrix for monolithic systems to get the
     *        cross-linearizations
     *
     * @param[in] defgrad Deformation gradient
     * @param[in] iFinjM  Inverse inelastic deformation gradient of current inelastic contribution
     *                    as 3x3 matrix
     * @param[in] dSdiFinj  Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                      deformation gradient of current inelastic contribution
     * @param[in,out] dstressdx Derivative of 2nd Piola Kirchhoff stresses w.r.t. primary variable
     *                          of different field
     */
    virtual void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) = 0;

    /*!
     * @brief pre-evaluation, intended to be used for stuff that has to be done only once per
     *        evaluate()
     *
     * @param[in] params  parameter list as handed in from the element
     * @param[in] gp      Gauss point
     * @param[in] eleGID  Element ID
     */
    virtual void pre_evaluate(const Teuchos::ParameterList& params,
        const EvaluationContext& context, int gp, int eleGID) = 0;

    /*!
     * @brief set gauss point concentration to parameter class
     *
     * @param[in] concentration  gauss point concentration to be set to internal member of parameter
     *                           class
     *
     * @note This method is used by methods called from the contact algorithm. Since the gauss point
     * ids do not match anyways (volume vs. surface element gauss point ids) and the id is not
     * relevant since the method is only called for one gauss point anyways, we set it to a dummy
     * gauss point id of 0 here
     */
    virtual void set_concentration_gp(double concentration) {};

    /// return material parameters
    virtual Core::Mat::PAR::Parameter* parameter() const { return params_; }

    /// Get type of scalar, that leads to deformation
    virtual PAR::InelasticSource get_inelastic_source() = 0;

    /*!
     * @brief Setup inelastic defgrad factor for the specific element
     *
     * @param[in] numgp Number of Gauss points
     * @param[in] container Input parameter Container
     */
    virtual void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) = 0;

    /// update history variables of the inelastic defgrad factors for next time step
    virtual void update() = 0;

    virtual void pack_inelastic(Core::Communication::PackBuffer& data) const = 0;

    virtual void unpack_inelastic(Core::Communication::UnpackBuffer& data) = 0;

   private:
    /// material parameters
    Core::Mat::PAR::Parameter* params_;
  };

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradNoGrowth

   This class models materials in combination with the multiplicative split material that feature
   no volume changes, i.e. the inelastic deformation gradient is always the identity tensor and
   contributions to the linearizations therefore vanish.
   */
  class InelasticDefgradNoGrowth : public InelasticDefgradFactors
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params           pointer to material specific parameters
     */
    explicit InelasticDefgradNoGrowth(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override;

    PAR::InelasticSource get_inelastic_source() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_no_growth;
    }

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    // identity tensor
    Core::LinAlg::Matrix<3, 3> identity_;
  };

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradTimeFunct

  This class models materials in combination with the multiplicative split material that feature
  isotropic volume changes based on a given time-dependent function for the determinant of the
  inelastic part.
  */
  class InelasticDefgradTimeFunct : public InelasticDefgradFactors
  {
   public:
    explicit InelasticDefgradTimeFunct(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override {};

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override {};

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override {};

    PAR::InelasticSource get_inelastic_source() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_time_funct;
    }

    Mat::PAR::InelasticDefgradTimeFunct* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradTimeFunct*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    //! evaluated function value. Gets filled in pre_evaluate()
    double funct_value_;

    //! identity tensor
    Core::LinAlg::Matrix<3, 3> identity_;
  };

  class InelasticDefgradScalar : public InelasticDefgradFactors
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params           pointer to material specific parameters
     */
    explicit InelasticDefgradScalar(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override = 0;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other,
        Core::LinAlg::Matrix<3, 3>& iFinM) override = 0;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override = 0;

    PAR::InelasticSource get_inelastic_source() override = 0;

    Core::Materials::MaterialType material_type() const override = 0;

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void set_concentration_gp(double concentration) override;

    Mat::PAR::InelasticDefgradScalar* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradScalar*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void update() override = 0;

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override = 0;

    void pack_inelastic(Core::Communication::PackBuffer& data) const override = 0;

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override = 0;

   protected:
    //! Get vector of concentration at current Gauss point
    [[nodiscard]] const std::vector<double>& get_concentration_gp() const
    {
      FOUR_C_ASSERT_ALWAYS(concentrations_ != nullptr, "Concentrations are not set");
      return *concentrations_;
    };

   private:
    /// vector of concentations at the gauss points
    std::shared_ptr<std::vector<double>> concentrations_{};
  };

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolyIntercalFrac

   This class evaluates polynomial and its first derivative w.r.t. intercalation fraction which is
   required in various routines of subclasses for isotropic and anisotropic case. This polynomial
   describes the growth of material with respect to intercalation fraction and it is prescribed by
   user in input file by defining it coefficients.
   */
  class InelasticDefgradPolyIntercalFrac : public InelasticDefgradScalar
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params             pointer to material specific parameters
     * @param[in] polynomial_growth  pointer to object that evaluates the polynomial as prescribed
     *                               in the input file
     */
    explicit InelasticDefgradPolyIntercalFrac(Core::Mat::PAR::Parameter* params);

    /*!
     * @brief evaluate polynomial describing growth of material with regard to intercalation
     * fraction based on the current concentration
     *
     * @param[in] concentration current concentration
     * @param[in] detjacobian   determinant of the deformation gradient
     * @return value of polynomial describing the growth according to current intercalation fraction
     */
    double evaluate_polynomial(double concentration, double detjacobian);

    /*!
     * @brief evaluate the first derivative of the polynomial describing the growth
     *
     * @param[in] concentration current concentration
     * @param[in] detjacobian   determinant of the deformation gradient
     * @return first derivative of the polynomial describing the growth
     */
    double evaluate_polynomial_derivative(double concentration, double detjacobian);

    Core::Materials::MaterialType material_type() const override = 0;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other,
        Core::LinAlg::Matrix<3, 3>& iFinM) override = 0;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override = 0;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override = 0;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdx) override = 0;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Mat::PAR::InelasticDefgradPolyIntercalFrac* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(
          Mat::InelasticDefgradScalar::parameter());
    }

    void update() override = 0;

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override = 0;

    void pack_inelastic(Core::Communication::PackBuffer& data) const override = 0;

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override = 0;

   private:
    /// pointer to class that evaluates the polynomial growth law
    std::shared_ptr<InelasticDefgradPolynomialShape> polynomial_growth_;
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinScalarIso
        This inelastic deformation gradient provides an isotropic growth law. Volumetric change due
        to this law is dependent on the current concentration \f$ c \f$ as follows :
      \f[
      \boldsymbol{F} _\text{in} = \left[1 + \text { scalar1_molar_growth_fac }
      \left(c \det \boldsymbol{F} - \text { Scalar1refconc } \right) \right] ^ { 1 / 3 }
      \boldsymbol{I}
      \f]
      */
  class InelasticDefgradLinScalarIso : public InelasticDefgradScalar
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params          pointer to material specific parameters
     * @param[in] linear_growth   pointer to object that evaluates the linear relation as prescribed
     *                            in the input file
     */
    explicit InelasticDefgradLinScalarIso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_lin_scalar_iso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Mat::PAR::InelasticDefgradLinScalar* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradLinScalar*>(
          Mat::InelasticDefgradScalar::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    /// pointer to class that evaluates the linear growth law
    std::shared_ptr<InelasticDefgradLinearShape> linear_growth_;
  };  // namespace Mat

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinScalarAniso

     This inelastic deformation gradient provides an anisotropic growth law.
     Volumetric change due to this law is dependent on the current concentration \f$ c \f$ as
     follows:
     \f[
     \mathbf{F}_\text{in} = \mathbf{I} + \left[ \text{scalar1_molar_growth_fac}
     \left( c \det\mathbf{F}  - \text{Scalar1refconc} \right) \right] \mathbf{G},
     \f]
     where \f$ \mathbf{G} \f$ (#growthdirmat_) is a matrix providing the information of the
     growth direction, that is constructed as follows:
     \f$ \mathbf{G} = \mathbf{g} \otimes \mathbf{g} \f$,
     where \f$ \mathbf{g} \f$ is the growth direction vector given in the input file.
     \f$ \mathbf{g} \f$ is normalized to length 1 before calculation of \f$ \mathbf{G} \f$.
     \f$ f(\chi) \f$ is defined by the user in the input file.
     */
  class InelasticDefgradLinScalarAniso : public InelasticDefgradScalar
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params          pointer to material specific parameters
     * @param[in] linear_growth   pointer to object that evaluates the linear relation as prescribed
     *                            in the input file
     */
    explicit InelasticDefgradLinScalarAniso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_lin_scalar_aniso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Mat::PAR::InelasticDefgradLinScalarAniso* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradLinScalarAniso*>(
          Mat::InelasticDefgradScalar::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    /// store pointer to class that evaluates the linear growth law
    std::shared_ptr<InelasticDefgradLinearShape> linear_growth_;
  };  // end of InelasticDefgradLinScalarAniso

  /*--------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolyIntercalFracIso

   This inelastic deformation gradient provides an isotropic growth law.
   Volumetric change due to this law is non-linearly dependent on the intercalation fraction
   \f$ \chi \f$ as follows:
   \f[
   \boldsymbol{F}_\text{in} =
   \left[ \frac{f(\chi) + 1 }{f(\chi^0) + 1} \right]^{1/3} \boldsymbol{I},
   \f]
   where \f$ f(\chi) \f$ is defined by the user in the input file.
   */
  class InelasticDefgradPolyIntercalFracIso : public InelasticDefgradPolyIntercalFrac
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params             pointer to material specific parameters
     * @param[in] polynomial_growth  pointer to object that evaluates the polynomial as prescribed
     *                               in the input file
     */
    explicit InelasticDefgradPolyIntercalFracIso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_poly_intercal_frac_iso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticDefgradPolyIntercalFrac* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFrac*>(
          Mat::InelasticDefgradPolyIntercalFrac::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradPolyIntercalFracAniso

   This inelastic deformation gradient provides an anisotropic growth law.
   Volumetric change due to this law is nonlinearly dependent on the intercalation fraction
   \f$ \chi \f$ as follows:
   \f[
   \boldsymbol{F}_\text{in} =
   \boldsymbol{I} + \left[ \frac{f(\chi) - f(\chi^0)}{f(\chi^0) + 1} \right] \boldsymbol{G},
   \f]
   where \f$ \boldsymbol{G} \f$ (#growthdirmat_) is a matrix providing the information of the growth
   direction, that is constructed as follows:
   \f$ \boldsymbol{G} = \boldsymbol{g} \otimes \boldsymbol{g} \f$, where \f$ \boldsymbol{g} \f$ is
   the growth direction vector given in the input file.
   \f$ \boldsymbol{g} \f$ is normalized to length 1 before calculation of \f$ \boldsymbol{G} \f$.
   \f$ f(\chi) \f$ is defined by the user in the input file.
   */
  class InelasticDefgradPolyIntercalFracAniso : public InelasticDefgradPolyIntercalFrac
  {
   public:
    /*!
     * @brief construct material with required inputs
     *
     * @param[in] params             pointer to material specific parameters
     * @param[in] polynomial_growth  pointer to object that evaluates the polynomial as prescribed
     *                               in the input file
     */
    explicit InelasticDefgradPolyIntercalFracAniso(Core::Mat::PAR::Parameter* params);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_poly_intercal_frac_aniso;
    }

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdc) override;

    Mat::PAR::InelasticDefgradPolyIntercalFracAniso* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradPolyIntercalFracAniso*>(
          Mat::InelasticDefgradPolyIntercalFrac::parameter());
    }

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};
  };

  /*----------------------------------------------------------------------*/
  /*! \class InelasticDefgradLinTempIso
   *Volumetric change due to this law depends on the temperature linearly.
   \f$ T \f$ as follows:
   \f[
   \boldsymbol{F}_\text{in} = \boldsymbol{I} \left[ 1 + \beta \left( T - T_\text{ref} \right)
   \right]^\frac{1}{3}, \f]
   */
  class InelasticDefgradLinTempIso : public InelasticDefgradFactors
  {
   public:
    explicit InelasticDefgradLinTempIso(Core::Mat::PAR::Parameter* params);

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override;

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdT) override;

    Mat::PAR::InelasticSource get_inelastic_source() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_lin_temp_iso;
    };

    Mat::PAR::InelasticDefgradLinTempIso* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradLinTempIso*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override {};

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override {};

    void pack_inelastic(Core::Communication::PackBuffer& data) const override {};

    void unpack_inelastic(Core::Communication::UnpackBuffer& data) override {};

   private:
    /// temperature at the gauss point
    double temperature_ = 0.0;
  };

  /*! \class InelasticDefgradTransvIsotropElastViscoplast
   * \brief Finite strain framework for isotropic and transversely isotropic viscoplasticity with
   * arbitrary flow rule and hardening law.
   *
   * This class implements an inelastic deformation gradient, which models viscoplastic material
   * response in a highly adaptable manner, assuming isothermal conditions at a constant
   * temperature.
   * Both isotropic and transversely isotropic material behavior can be modeled. For transversely
   * isotropic materials, both the elastic and viscoplastic deformation components can depend on the
   * preferred material fiber direction. An additive split of isotropic and transversely isotropic
   * components is assumed for the formulated elastic free energy, see Bonet et al. 1998 (below).
   * Furthermore, the model is formulated to allow for an arbitrary choice of the local viscoplastic
   * flow rule and the hardening law, see class ViscoplasticLaws. In this context, "local" refers to
   * the fact that both the flow rule and the hardening are specified independently at each Gauss
   * point, without influence from other Gauss points.
   * Currently, the model only accounts for isotropic hardening.
   * For further information on the model, refer to:
   *   -# Master's Thesis : Dragos-Corneliu Ana, Continuum Modeling and Calibration of
   * Viscoplasticity in the Context of the Lithium Anode in Solid State Batteries, Supervisor:
   * Christoph Schmidt, 2024
   *   -# Mareau et al., A thermodynamically consistent formulation of the Johnson-Cook model,
   *  Mechanics of Materials 143, 2020
   *  -# Aravas, Finite Elastoplastic Transformations of Transversely Isotropic Metals, Int. J.
   * Solids Structures Vol. 29, No. 17, 1992 (Hill 1948 yield condition used in the implemented
   * model, but with notation following Dafalias 1989, see below)
   *  -# Dafalias and Rashid, The Effect of Plastic Spin on Anisotropic Material Behavior, Int. J.
   * Plasticity, Vol. 5, 1989
   *  -# Holzapfel, Nonlinear Solid Mechanics, Wiley & Sons, 2000
   *  -# Bonet et al., A simple orthotropic, transversely isotropic hyperelastic constitutive
   * equation for large strain computations, Comput. Methods Appl. Mech. Engrg. 162, 1998
   */
  class InelasticDefgradTransvIsotropElastViscoplast : public InelasticDefgradFactors
  {
   public:
    /*!
     * @brief construct transversely isotropic material
     *
     * @param[in] params material parameters
     * @param[in] viscoplastic_law viscoplasticity law, determining the flow rule and the hardening
     *                             model
     * @param[in] fiber_reader dummy hyperelastic model utilized to read the fiber direction for
     * transverse isotropy
     * @param[in] pot_sum_el elastic components / potential summands (only isotropic)
     * @param[in] pot_sum_el_transv_iso elastic components / potential summands (only transversely
     * isotropic)
     */

    explicit InelasticDefgradTransvIsotropElastViscoplast(Core::Mat::PAR::Parameter* params,
        std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law,
        Mat::Elastic::CoupTransverselyIsotropic fiber_reader,
        std::vector<std::shared_ptr<Mat::Elastic::Summand>> pot_sum_el,
        std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>>
            pot_sum_el_transv_iso);

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::mfi_transv_isotrop_elast_viscoplast;
    }

    void evaluate_additional_cmat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, const Core::LinAlg::Matrix<3, 3>& iFinjM,
        const Core::LinAlg::Matrix<6, 1>& iCV, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 6>& cmatadd) override;

    void evaluate_inelastic_def_grad_derivative(
        double detjacobian, Core::LinAlg::Tensor<double, 3, 3>& dFindx) override {};

    void evaluate_inverse_inelastic_def_grad(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFin_other, Core::LinAlg::Matrix<3, 3>& iFinM) override;

    void evaluate_od_stiff_mat(const Core::LinAlg::Matrix<3, 3>* defgrad,
        const Core::LinAlg::Matrix<3, 3>& iFinjM, const Core::LinAlg::Matrix<6, 9>& dSdiFinj,
        Core::LinAlg::Matrix<6, 1>& dstressdT) override {};

    Mat::PAR::InelasticSource get_inelastic_source() override { return PAR::InelasticSource::none; }

    Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast* parameter() const override
    {
      return dynamic_cast<Mat::PAR::InelasticDefgradTransvIsotropElastViscoplast*>(
          Mat::InelasticDefgradFactors::parameter());
    }

    void setup(const int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void pre_evaluate(const Teuchos::ParameterList& params, const EvaluationContext& context,
        int gp, int eleGID) override;

    void update() override;

    void pack_inelastic(Core::Communication::PackBuffer& data) const override;

    void unpack_inelastic(Core::Communication::UnpackBuffer& buffer) override;

    /*! @brief Evaluate the current state variables based on a given right Cauchy-Green
     * deformation tensor, given inverse plastic deformation gradient and given equivalent
     * plastic strain
     *
     * @param[in] CM right Cauchy-Green deformation tensor \f[ \boldsymbol{C} \f] in matrix form
     * @param[in] iFinM inverse inelastic deformation gradient
     *                  \f[ \boldsymbol{F}_{\text{in}}^{-1} \f] in matrix form
     * @param[in] plastic_strain plastic strain  \f$ \varepsilon_{\text{p}} \f$
     * @param[out] err_status error status
     * @param[in] dt time step (or substep) length used for time integration
     * @param[in] eval_type evaluation type: full evaluation or only
     * partial evaluation, e.g. stop once the plastic strain rate has
     * been evaluated
     */
    InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities evaluate_state_quantities(
        const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<3, 3>& iFinM,
        const double plastic_strain,
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status, const double dt,
        const InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityEvalType& eval_type);

    /*! @brief Evaluate the current state variable derivatives with respect to the right
     * Cauchy-Green deformation tensor, the inverse plastic deformation gradient and the equivalent
     * plastic strain (for a given/calculated state)
     *
     * @param[in] CM right Cauchy-Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] iFinM inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1}
     *                  \f$ in matrix form
     * @param[in] plastic_strain plastic strain  \f$ \varepsilon_{\text{p}} \f$
     * @param[out] err_status error status
     * @param[in] dt time step length  \f$ \Delta t
     * \f$ (used for the integration)
     * @param[in] eval_state boolean: do we want to also evaluate the current state first (true)
     *                       or is this already available from the
     *                       current state variables (false)
     * @param[in] eval_type evaluation type: full evaluation or only
     * partial evaluation, e.g. stop once the derivatives of the plastic strain rate have
     * been evaluated
     */
    InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
    evaluate_state_quantity_derivatives(const Core::LinAlg::Matrix<3, 3>& CM,
        const Core::LinAlg::Matrix<3, 3>& iFinM, const double plastic_strain,
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status, const double dt,
        const InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivEvalType&
            eval_type,
        const bool eval_state = false);

    //! return the fiber direction of transverse isotropy for the considered element
    Core::LinAlg::Matrix<3, 1> get_fiber_direction() { return m_; }

   private:
    //! constant material-related tensors (isotropic: constant tensors
    //! such
    //! as identity matrices; transversely-isotropic: also contains
    //! tensors associated with the director vector)
    InelasticDefgradTransvIsotropElastViscoplastUtils::ConstMatTensors const_mat_tensors_;

    //! current Gauss Point
    int gp_;
    //! current element ID
    int ele_gid_;

    //! parameter list
    Teuchos::ParameterList params_;

    //! map to elastic materials/potential summands (only isotropic)
    std::vector<std::shared_ptr<Mat::Elastic::Summand>> potsumel_;

    //! map to elastic materials/potential summands (only transversely isotropic)
    std::vector<std::shared_ptr<Mat::Elastic::CoupTransverselyIsotropic>> potsumel_transviso_;

    //! viscoplastic law
    std::shared_ptr<Mat::Viscoplastic::Law> viscoplastic_law_;

    //! fiber reader (hyperelastic transversely isotropic material used for fiber reading)
    Mat::Elastic::CoupTransverselyIsotropic fiber_reader_;

    //! fiber direction (director vector)
    Core::LinAlg::Matrix<3, 1> m_;

    //! utilities for evaluating the matrix exponential and logarithm
    InelasticDefgradTransvIsotropElastViscoplastUtils::MatrixExpLogUtils matrix_exp_log_utils_;

    //! boolean to control whether the history variables should be updated during evaluation
    bool update_hist_var_ = true;

    //! tracker for time step settings and time instants
    InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepTracker time_step_tracker_;

    //! tracker for quantities at the last and current time points (i.e., at \f[ t_n \f] and
    //! \f[ t_{n+1} \f], respectively) for all Gauss points simultaneously
    InelasticDefgradTransvIsotropElastViscoplastUtils::TimeStepQuantities time_step_quantities_;

    //! evaluated state quantities
    InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantities state_quantities_;

    //! evaluated state quantity derivatives
    InelasticDefgradTransvIsotropElastViscoplastUtils::StateQuantityDerivatives
        state_quantity_derivatives_;

    //! tensor interpolator used in the substepping procedure (one-dimensional, of order 1)
    Core::LinAlg::SecondOrderTensorInterpolator<1> tensor_interpolator_;

    //! tensor interpolation: reference matrices
    std::vector<Core::LinAlg::Matrix<3, 3>> ref_matrices_;

    //! tensor interpolation: 1D reference locations (we always interpolate between 0.0 and 1.0
    //! based on the reference matrices of the current time step)
    const std::vector<double> ref_locs_{0.0, 1.0};

    //! tracker object for the local substepping procedure
    InelasticDefgradTransvIsotropElastViscoplastUtils::LocalSubsteppingUtils
        local_substepping_utils_;

    /*!
     * @brief Calculate the Holzapfel gamma and delta values of the isotropic elastic material
     * components
     * @param[in] CeM elastic right Cauchy_Green deformation tensor \f$ \boldsymbol{C}_\text{e}
     * \f$ in matrix form
     * @param[out] gamma stress factors for the isotropic elasticity case, as derived in
     *                   Holzapfel - Nonlinear Solid Mechanics(2000)
     * @param[out] delta constitutive tensor factors for the isotropic elasticity case, as derived
     *                   in Holzapfel - Nonlinear Solid Mechanics(2000)
     */
    void calculate_gamma_delta(const Core::LinAlg::Matrix<3, 3>& CeM,
        Core::LinAlg::Matrix<3, 1>& gamma, Core::LinAlg::Matrix<8, 1>& delta);

    /*!
     * @brief Check if the elastic predictor provides the solution for the current time step,
     * i.e., the deformation in the current time step is purely elastic with no viscoplastic
     * contribution.
     *
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] iFinM_pred predictor of the inverse inelastic deformation gradient \f$
     * \bm{F}_{\text{in, pred}} \f$
     * @param[in] plastic_strain_pred predictor of the plastic strain \f$ \varepsilon_{\text{p,
     * pred}} \f$
     * @param[out] err_status error status
     * @return boolean value: true (predictor = solution), or false (predictor != solution)
     */
    bool check_elastic_predictor(const Core::LinAlg::Matrix<3, 3>& CM,
        const Core::LinAlg::Matrix<3, 3>& iFinM_pred, const double plastic_strain_pred,
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status);

    /*!
     * @brief Calculate the residual for the Local Newton Loop (LNL)
     *
     * @note The state quantities (class variable) are updated in this method, since they
     * are used for the computation of the residual!
     *
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] x vector of Local Newton Loop unknowns, composed of the components of the
     * inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1} \f$ and plastic
     * strain \f$ \varepsilon_{\text{p}} \f$
     * @param[in] last_iFpM last inverse plastic deformation gradient
     *                      \f$ \boldsymbol{F}_{\text{in}, n}^{-1} \f$ in matrix form
     * @param[in] last_plastic_strain last plastic strain \f$ \varepsilon_{\text{p}, n}\f$
     * @param[in] dt time step (or substep) length used for time integration
     * @param[out] err_status error status
     * @return  residual of the LNL equations
     */
    Core::LinAlg::Matrix<10, 1> calculate_local_newton_loop_residual(
        const Core::LinAlg::Matrix<3, 3>& CM, const Core::LinAlg::Matrix<10, 1>& x,
        const Core::LinAlg::Matrix<3, 3>& last_iFinM, const double last_plastic_strain,
        const double dt, InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status);


    /*!
     * @brief For a given right Cauchy_Green tensor and the Local NR Loop unknown vector,
     * compute the 10 x 10 Jacobian matrix required for the Local Newton Loop and the
     * linearization for the Global Newton Loop
     *
     * @note The state quantity derivatives (class variable) are updated in this method.
     * They require the state quantities, which were evaluated and stored
     * previously when calculating the residual.
     *
     * @param[in] CM right Cauchy_Green deformation tensor \f$ \boldsymbol{C} \f$ in matrix form
     * @param[in] x vector of Local Newton Loop unknowns, composed of the components of the
     * inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1} \f$ and plastic
     * strain \f$ \varepsilon_{\text{p}} \f$
     * @param[in] last_iFpM last inverse plastic deformation gradient
     *                      \f$ \boldsymbol{F}_{\text{in}, n}^{-1} \f$ in matrix form
     * @param[in] last_plastic_strain last plastic strain \f$ \varepsilon_{\text{p}, n}\f$
     * @param[in] dt time step (or substep) length used for time integration
     * @param[out] err_status error status
     * @return 10x10 jacobian matrix of the Local Newton Loop and of the linearization
     *         \f$ \boldsymbol{J} \f$
     */
    Core::LinAlg::Matrix<10, 10> calculate_jacobian(const Core::LinAlg::Matrix<3, 3>& CM,
        const Core::LinAlg::Matrix<10, 1>& x, const Core::LinAlg::Matrix<3, 3>& last_iFinM,
        const double last_plastic_strain, const double dt,
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status);

    /*!
     * @brief Local Newton Loop in order to calculate the current inverse plastic deformation
     * gradient and the current plastic strain value
     *
     * @param[in] defgrad deformation gradient \f$ \boldsymbol{F} \f$ in matrix form
     * @param[in] x initial guess of Local Newton Loop, composed of the components of the
     *              inverse inelastic deformation gradient \f$ \boldsymbol{F}_{\text{in}}^{-1} \f$
     *              and plastic strain \f$ \varepsilon_{\text{p}} \f$
     * @param[out] err_status error status
     * @return solution vector of the Local Newton Loop, structured analogously to the initial guess
     * x
     */
    Core::LinAlg::Matrix<10, 1> local_newton_loop(const Core::LinAlg::Matrix<3, 3>& defgrad,
        const Core::LinAlg::Matrix<10, 1>& x,
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status);


    /*!
     * @brief Setup new substep in the Local Newton Loop in case of an encountered evaluation
     * error
     *
     * @param[in,out] sol current solution vector of the Local Newton Loop (reset to the last
     * converged value within this method)
     * @param[in,out] curr_CM current right Cauchy-Green deformation tensor, interpolated using
     * the reference matrices of the time step (interpolated again within this method with the
     * updated new substep length)
     * @return error status for the new substep (true: no errors, false: we have halved the time
     * step too many times)
     *
     */
    bool prepare_new_substep(Core::LinAlg::Matrix<10, 1>& sol, Core::LinAlg::Matrix<3, 3>& curr_CM);

    /*!
     * @brief Evaluate the additional cmat stiffness tensor using a perturbation-based approach,
     if
     * the analytical evaluation fails
     *
     * @note For further information on the procedure, refer to:
     *       -# Master's Thesis : Dragos-Corneliu Ana, Continuum Modeling and Calibration of
     * Viscoplasticity in the Context of the Lithium Anode in Solid State Batteries, Supervisor:
     * Christoph Schmidt, 2024
     *
     * @param[in] FredM reduced deformation gradient \f$ \boldsymbol{F}_{\text{red}} =
     * \boldsymbol{F} \boldsymbol{F_{\text{in,other}}^{-1}} \f$ accounting for all the already
     * computed inelastic defgrad factors
     * @param[out] cmatadd Additional elasticity stiffness
     * @param[in] iFin_other Already computed inverse inelastic deformation gradient
     *              (from already computed inelastic factors in the multiplicative split material)
     * @param[in] dSdiFinj Derivative of 2nd Piola Kirchhoff stresses w.r.t. the inverse inelastic
     *                     deformation gradient of current inelastic contribution
     *
     */
    void evaluate_additional_cmat_perturb_based(const Core::LinAlg::Matrix<3, 3>& FredM,
        Core::LinAlg::Matrix<6, 6>& cmatadd, const Core::LinAlg::Matrix<3, 3>& iFin_other,
        const Core::LinAlg::Matrix<6, 9>& dSdiFinj);

    /*!
     * @brief Get an extensive error message to be displayed when the
     * simulation terminates. This is useful for debugging the time
     * integration in more detail. This message contains a base error
     * message which describes what failed in short form - this is
     * then extended with information on the element ID, the Gauss
     * Point, the last_ values and so on...
     *
     * @param[in] base_error_string base error message to be extended
     * with further information
     */
    std::string get_error_info(const std::string& base_error_string);
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
