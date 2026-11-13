// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_BEAM_ELASTHYPER_HPP
#define FOUR_C_MAT_BEAM_ELASTHYPER_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_beam_templated_material_generic.hpp"

#include <memory>
#include <typeinfo>

FOUR_C_NAMESPACE_OPEN


// forward declaration
namespace Discret
{
  class ParObject;
}

namespace Mat
{
  // forward declaration
  namespace PAR
  {
    class BeamElastHyperMaterialParameterGeneric;
  }

  /// singleton for constitutive law of a beam formulation (hyperelastic stored energy function)
  template <typename T>
  class BeamElastHyperMaterialType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return typeid(this).name(); }

    static BeamElastHyperMaterialType& instance()
    {
      static BeamElastHyperMaterialType instance;
      return instance;
    }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;
  };



  /*---------------------------------------------------------------------------------------------*/
  /// constitutive relations for beam cross-section resultants (hyperelastic stored energy function)
  template <typename T>
  class BeamElastHyperMaterial : public BeamMaterialTemplated<T>
  {
   public:
    /// construct empty material object
    BeamElastHyperMaterial();

    /// construct the material object from given material parameters
    explicit BeamElastHyperMaterial(Mat::PAR::BeamElastHyperMaterialParameterGeneric* params);

    /**
     * \brief Initialize and setup element specific variables
     *
     */
    void setup(int numgp_force, int numgp_moment) override {}

    //! @name Packing and Unpacking
    //@{

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return BeamElastHyperMaterialType<T>::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void pack(Core::Communication::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      unique_par_object_id().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! @name derived access methods
    //@{

    /** \brief get type of this material
     *
     */
    Core::Materials::MaterialType material_type() const override
    {
      /* the idea is that we have a generic type of material (this class), but two
       * possible types of material parameter definitions and hence types
       * (BeamReissnerElastHyperMaterialParams and BeamReissnerElastHyperMaterialParamsByMode) */
      return Core::Materials::m_beam_elast_hyper_generic;
    }

    /** \brief return copy of this material object
     *
     */
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<BeamElastHyperMaterial>(*this);
    }

    /** \brief get quick accessible material parameter data
     *
     */
    Core::Mat::PAR::Parameter* parameter() const override;

    //@}


    //! @name Access methods
    //@{

    /** \brief get quick accessible material parameter data
     *
     */
    const Mat::PAR::BeamElastHyperMaterialParameterGeneric& params() const;

    /*
     * \brief Compute axial stress contributions
     *
     *\param[out] stressM axial stress
     *
     *\param[in] CM constitutive matrix
     *
     *\param[in] Cur curvature
     */
    void evaluate_moment_contributions_to_stress(Core::LinAlg::Matrix<3, 1, T>& stressM,
        const Core::LinAlg::Matrix<3, 3, T>& CM, const Core::LinAlg::Matrix<3, 1, T>& Cur,
        const unsigned int gp) override;

    /*
     * \brief Compute axial stress contributions
     *
     *\param[out] stressN axial stress
     *
     *\param[in] CN constitutive matrix
     *
     *\param[in] Gamma triad
     */

    void evaluate_force_contributions_to_stress(Core::LinAlg::Matrix<3, 1, T>& stressN,
        const Core::LinAlg::Matrix<3, 3, T>& CN, const Core::LinAlg::Matrix<3, 1, T>& Gamma,
        const unsigned int gp) override;

    /*
     * \brief Update material-dependent variables
     */
    void compute_constitutive_parameter(
        Core::LinAlg::Matrix<3, 3, T>& C_N, Core::LinAlg::Matrix<3, 3, T>& C_M) override;

    /** \brief get constitutive matrix relating stress force resultants and translational strain
     *         measures, expressed w.r.t. material frame
     *
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
     */
    double get_translational_mass_inertia_factor() const override;

    /** \brief get mass moment of inertia tensor, expressed w.r.t. material frame
     *
     */
    void get_mass_moment_of_inertia_tensor_material_frame(
        Core::LinAlg::Matrix<3, 3>& J) const override;

    /** \brief get mass moment of inertia tensor, expressed w.r.t. material frame
     */
    void get_mass_moment_of_inertia_tensor_material_frame(
        Core::LinAlg::Matrix<3, 3, Sacado::Fad::DFad<double>>& J) const override;



    /** \brief get the radius of a circular cross-section that is ONLY to be used for evaluation of
     *         any kinds of beam interactions (contact, potentials, viscous drag forces ...)
     *
     */
    double get_interaction_radius() const override;

    void get_stiffness_matrix_of_moments(Core::LinAlg::Matrix<3, 3, T>& stiffness_matrix,
        const Core::LinAlg::Matrix<3, 3, T>& C_M, const int gp) override;

    void get_stiffness_matrix_of_forces(Core::LinAlg::Matrix<3, 3, T>& stiffness_matrix,
        const Core::LinAlg::Matrix<3, 3, T>& C_N, const int gp) override;

    void update() override {};

    void reset() override {};

   protected:
    void set_parameter(Mat::PAR::BeamElastHyperMaterialParameterGeneric* parameter)
    {
      params_ = parameter;
    }

   private:
    /// my material parameters
    Mat::PAR::BeamElastHyperMaterialParameterGeneric* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
