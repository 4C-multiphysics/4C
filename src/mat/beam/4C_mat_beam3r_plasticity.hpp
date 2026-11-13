// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_BEAM3R_PLASTICITY_HPP
#define FOUR_C_MAT_BEAM3R_PLASTICITY_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_beam_elasthyper.hpp"
#include "4C_mat_beam_elasthyper_parameter.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


// forward declaration
namespace Discret
{
  class ParObject;
}

namespace Mat
{
  namespace PAR
  {
    /*-------------------------------------------------------------------------------------------*/
    /** constitutive parameters for a Reissner beam formulation (hyperelastic stored energy
     * function)
     */
    class BeamReissnerElastPlasticMaterialParams : public BeamReissnerElastHyperMaterialParams
    {
     public:
      //! standard constructor
      BeamReissnerElastPlasticMaterialParams(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! @name Access to plasticity parameters
      //@{

      //! yield stress for forces
      double get_yield_stress_n() const override { return yield_stress_n_; }

      //! yield stress momentum
      double get_yield_stress_m() const override { return yield_stress_m_; }

      //! hardening rigidity axial direction
      double get_hardening_axial_rigidity() const override
      {
        return isohard_modulus_n_ * get_cross_section_area();
      };

      //! hardening rigidity shear one direction
      double get_hardening_shear_rigidity2() const override
      {
        return get_shear_modulus() * get_cross_section_area() * get_shear_correction_factor();
      };

      //! hardening rigidity shear other direction
      double get_hardening_shear_rigidity3() const override
      {
        return get_shear_modulus() * get_cross_section_area() * get_shear_correction_factor();
      };

      //! hardening rigidity for momentum
      double get_hardening_momental_rigidity() const override
      {
        return isohard_modulus_m_ * get_moment_inertia2();
      }

      //! consider torsion plasticity
      bool get_torsion_plasticity() const override { return torsion_plasticity_; }
      //@}

     private:
      //! @name plasticity parameters
      //@{
      //! Yield stress of forces
      double yield_stress_n_;
      //! Yield stress of moments
      double yield_stress_m_;
      //! Isotropic hardening modulus of forces
      const double isohard_modulus_n_;
      //! Isotropic hardening modulus of moments
      const double isohard_modulus_m_;
      //! defines whether torsional moment contributes to plasticity
      const bool torsion_plasticity_;
      //@}
    };
  }  // namespace PAR

  //! singleton for constitutive law of a beam formulation (hyperelastic stored energy function)
  template <typename T>
  class BeamElastPlasticMaterialType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return typeid(this).name(); }

    //! get instance for beam material
    static BeamElastPlasticMaterialType& instance() { return instance_; };

    //! create material object
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static BeamElastPlasticMaterialType instance_;
  };

  /*---------------------------------------------------------------------------------------------*/
  //! constitutive relations for beam cross-section resultants (hyperelastic stored energy function)
  template <typename T>
  class BeamPlasticMaterial : public BeamElastHyperMaterial<T>
  {
   public:
    //! construct empty material object
    BeamPlasticMaterial() = default;

    //! construct the material object from given material parameters
    explicit BeamPlasticMaterial(Mat::PAR::BeamReissnerElastPlasticMaterialParams* params);

    /**
     * \brief Initialize and setup element specific variables
     *
     */
    void setup(int numgp_force, int numgp_moment) override;

    //! @name Packing and Unpacking
    //@{

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return BeamElastPlasticMaterialType<T>::instance().unique_par_object_id();
    }

    /*!
     * \brief Pack this class so it can be communicated
     *
     * Resizes the vector data and stores all information of a class in it. The first information
     * to be stored in data has to be the unique parobject id delivered by unique_par_object_id()
     * which will then identify the exact class on the receiving processor.
     *
     * @param data (in/out): char vector to store class information
     */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
     * \brief Unpack data from a char vector into this class
     *
     * The vector data contains all information to rebuild the exact copy of an instance of a class
     * on a different processor. The first entry in data hast to be an integer which is the unique
     * parobject id defined at the top of this file and delivered by UniqueParObjectId().
     *
     * @param data (in) : vector storing all data to be unpacked into this instance
     */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! @name Stress contributions
    //@{

    /*!
     * \brief Compute axial stress contributions
     *
     *\param[out] stressN axial stress
     *\param[in] CN constitutive matrix
     *\param[in] Gamma strain
     */
    void evaluate_force_contributions_to_stress(Core::LinAlg::Matrix<3, 1, T>& stressN,
        const Core::LinAlg::Matrix<3, 3, T>& CN, const Core::LinAlg::Matrix<3, 1, T>& Gamma,
        const unsigned int gp) override;

    /*!
     * \brief Compute moment stress contributions
     *
     *\param[out] stressM moment stress
     *\param[in] CM constitutive matrix
     *\param[in] Cur curvature
     */
    void evaluate_moment_contributions_to_stress(Core::LinAlg::Matrix<3, 1, T>& stressM,
        const Core::LinAlg::Matrix<3, 3, T>& CM, const Core::LinAlg::Matrix<3, 1, T>& Cur,
        const unsigned int gp) override;

    /** \brief return copy of this material object
     */
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<BeamPlasticMaterial>(*this);
    }

    //@}

    //! @name Constitutive relations
    //@{

    /** \brief get constitutive matrix relating stress force resultants and translational strain
     *         measures, expressed w.r.t. material frame
     */
    void get_constitutive_matrix_of_forces_material_frame(
        Core::LinAlg::Matrix<3, 3, T>& C_N) const override;

    /** \brief get constitutive matrix relating stress moment resultants and rotational strain
     *         measures, expressed w.r.t. material frame
     *
     */
    void get_constitutive_matrix_of_moments_material_frame(
        Core::LinAlg::Matrix<3, 3, T>& C_M) const override;

    /** \brief get mass inertia factor with respect to translational accelerations
     *         (usually: density * cross-section area)
     *
     */
    double get_translational_mass_inertia_factor() const override;

    /** \brief get the radius of a circular cross-section that is ONLY to be used for evaluation of
     *         any kinds of beam interactions (contact, potentials, viscous drag forces ...)
     *
     */
    double get_interaction_radius() const override;

    /** \brief compute stiffness matrix of moments for plastic regime
     */
    void get_stiffness_matrix_of_moments(Core::LinAlg::Matrix<3, 3, T>& stiffM,
        const Core::LinAlg::Matrix<3, 3, T>& C_M, const int gp) override;

    /** \brief compute stiffness matrix of forces for plastic regime
     */
    void get_stiffness_matrix_of_forces(Core::LinAlg::Matrix<3, 3, T>& stiffN,
        const Core::LinAlg::Matrix<3, 3, T>& C_N, const int gp) override;

    /** \brief update the plastic strain and curvature vectors
     */
    void update() override;

    /** \brief reset the values for current plastic strain and curvature
     */
    void reset() override;

    /** \brief get hardening constitutive parameters depending on the type of plasticity
     */
    void compute_constitutive_parameter(
        Core::LinAlg::Matrix<3, 3, T>& C_N, Core::LinAlg::Matrix<3, 3, T>& C_M) override;

   protected:
    /** \brief get the constitutive matrix of forces during kinematic hardening
     *
     */
    void get_hardening_constitutive_matrix_of_forces_material_frame(
        Core::LinAlg::Matrix<3, 3, T>& CN_eff) const;

    /** \brief get the constitutive matrix of moments during kinematic hardening
     */
    void get_hardening_constitutive_matrix_of_moments_material_frame(
        Core::LinAlg::Matrix<3, 3, T>& CM_eff) const;

    /** \brief returns current effective yield stress of forces depending on plastic deformation
     */
    void get_effective_yield_stress_n(
        T& eff_yieldN, T init_yieldN, T CN_0, T CN_eff_0, const unsigned int gp) const;

    /** \brief returns current effective yield stress of moments depending on plastic deformation
     */
    void get_effective_yield_stress_m(
        T& eff_yieldM, T init_yieldM, T CM_1, T CM_eff_1, const unsigned int gp) const;

   private:
    //! effective constitutive matrices forces
    std::vector<Core::LinAlg::Matrix<3, 3, T>> c_n_eff_;
    //! effective constitutive matrices moments
    std::vector<Core::LinAlg::Matrix<3, 3, T>> c_m_eff_;

    //! converged plastic strain vectors at GPs
    std::vector<Core::LinAlg::Matrix<3, 1, T>> gammaplastconv_;
    //! new plastic strain vectors at GPs
    std::vector<Core::LinAlg::Matrix<3, 1, T>> gammaplastnew_;
    //! accumulated plastic strain vectors at GPs
    std::vector<T> gammaplastaccum_;

    //! converged plastic curvature vectors at GPs
    std::vector<Core::LinAlg::Matrix<3, 1, T>> kappaplastconv_;
    //! new plastic curvature vectors at GPs
    std::vector<Core::LinAlg::Matrix<3, 1, T>> kappaplastnew_;
    //! accumulated plastic curvature vectors at GPs
    std::vector<T> kappaplastaccum_;

    //! effective yield force depending on accumulated plastic strain
    std::vector<T> effyieldstress_n_;
    //! effective yield moment depending on accumulated plastic curvature
    std::vector<T> effyieldstress_m_;

    //! norm of material plastic curvature increment
    std::vector<T> delta_kappaplast_;
    //! norm of the moment vector
    std::vector<T> normstress_m_;
    //! fraction of the norm of the moment vector exceeding the current yield moment
    std::vector<T> deltastress_m_;

    //! material elastic curvature
    std::vector<Core::LinAlg::Matrix<3, 1, T>> kappaelast_;
    /** copy of material elastic curvature needed to determine flow direction when computing the
     * stiffness matrix (first entry is 0 if torsional plasticity is turned off)
     */
    std::vector<Core::LinAlg::Matrix<3, 1, T>> kappaelastflow_;
    //! unit vector in the direction of the elastic curvature
    std::vector<Core::LinAlg::Matrix<3, 1, T>> elastic_curvature_;
    //! material plastic strain increment
    std::vector<Core::LinAlg::Matrix<3, 1, T>> delta_gammaplast_;
    //! fraction of the stress exceeding the current yield force
    std::vector<Core::LinAlg::Matrix<3, 1, T>> deltastress_n_;

    //! axial stress
    std::vector<T> stress_n_;

    /// Number of integration points for forces
    unsigned int numgp_force_;

    /// Number of integration points for moments
    unsigned int numgp_moment_;
    //@}
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
