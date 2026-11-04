// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_CRYSTAL_PLASTICITY_HPP
#define FOUR_C_MAT_CRYSTAL_PLASTICITY_HPP

/*----------------------------------------------------------------------*
 | headers                                                  			|
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | class definitions                                           			|
 *----------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    /*!
     *  \brief This class processes the material/model parameters provided by the user
     */

    class CrystalPlasticity : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      CrystalPlasticity(const Core::Mat::PAR::Parameter::Data& matdata);


      //! create material instance
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //-----------------------------------------------------------------------------
      /*                                                                           */
      /** \name model parameters                                                   */
      /** @{                                                                       */
      //-----------------------------------------------------------------------------

      //! tolerance for local Newton iteration [--]
      const double tol_;
      //! Young's modulus [MPa]
      const double youngs_;
      //! Poisson's ratio [--]
      const double poisson_;
      //! mass density [ton/mm**3]
      const double density_;
      //! lattice type. Currently 'FCC', 'BCC', 'HCP', 'D019' or 'L10'
      const std::string lattice_;
      //! c to a ratio of the crystal's unit cell [--]
      const double ctoa_;
      //! lattice constant a [mm]
      const double lattice_const_;
      //! number of slip systems
      const int num_slip_sys_;
      //! number of slip system subsets
      const int num_slip_sets_;
      //! vector with num_slip_sys_ entries specifying to which subset the slip systems
      //! belong
      const std::vector<int> slip_set_mem_;
      //! vector with num_slip_sets_ entries for rate sensitivity exponents of slip systems
      const std::vector<int> slip_rate_exp_;
      //! vector with num_slip_sets_ entries for reference slip shear rates
      const std::vector<double> slip_ref_shear_rate_;
      //! vector with num_slip_sets_ entries for initial dislocation densities
      const std::vector<double> dis_dens_init_;
      //! vector with num_slip_sets_ entries for dislocation generation coefficients
      const std::vector<double> dis_gen_coeff_;
      //! vector with num_slip_sets_ entries for dynamic dislocation removal coefficients
      const std::vector<double> dis_dyn_rec_coeff_;
      //! vector with num_slip_sets_ entries for the lattice resistance to slip (Peierl's barrier)
      const std::vector<double> slip_lat_resist_;
      //! microstructural parameters, e.g. grain size; vector with num_slip_sets entries for slip
      //! systems
      const std::vector<double> slip_micro_bound_;
      //! vector with num_slip_sets entries for the Hall-Petch coefficients
      const std::vector<double> slip_hp_coeff_;
      //! vector with num_slip_sets entries for the hardening coefficients of slip systems due to
      //! non-coplanar twinning
      const std::vector<double> slip_by_twin_;
      //! number of twinning systems
      const int num_twin_sys_;
      //! number of twinning system subsets
      const int num_twin_sets_;
      //! vector with num_twin_sys_ entries specifying to which subset the twinning systems
      //! belong
      const std::vector<int> twin_set_mem_;
      //! vector with num_twin_sets_ entries for rate sensitivity exponents of twinning systems
      const std::vector<int> twin_rate_exp_;
      //! vector with num_twin_sets_ entries for reference twinning shear rates
      const std::vector<double> twin_ref_shear_rate_;
      //! vector with num_twin_sets_ entries for the lattice resistance to twinning (Peierl's
      //! barrier)
      const std::vector<double> twin_lat_resist_;
      //! microstructural parameters, e.g. grain size; vector with num_twin_sets entries for
      //! twinning systems
      const std::vector<double> twin_micro_bound_;
      //! vector with num_twin_sets entries for the Hall-Petch coefficients
      const std::vector<double> twin_hp_coeff_;
      //! vector with num_twin_sets entries for the hardening coefficients of twinning systems due
      //! to slip
      const std::vector<double> twin_by_slip_;
      //! vector with num_twin_sets entries for the hardening coefficients of twinning systems due
      //! to non-coplanar twinning
      const std::vector<double> twin_by_twin_;

      //-----------------------------------------------------------------------------
      /** @}                                                                       */
      /*  end of model parameters                                                  */
      /*                                                                           */
      //-----------------------------------------------------------------------------

    };  // class CrystalPlasticity
  }  // namespace PAR

  /*----------------------------------------------------------------------*/

  class CrystalPlasticityType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "CrystalPlasticityType"; }

    static CrystalPlasticityType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static CrystalPlasticityType instance_;

  };  // class CrystalPlasticityType

  /*----------------------------------------------------------------------*/

  /*!
   *  \brief This class introduces the crystal plasticity model
   */

  class CrystalPlasticity : public So3Material
  {
   public:
    //! construct empty material object
    CrystalPlasticity();

    //! construct the material object with the given model parameters
    explicit CrystalPlasticity(Mat::PAR::CrystalPlasticity* params);

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name Packing and Unpacking                                              */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! Return unique ParObject id
    int unique_par_object_id() const override
    {
      return CrystalPlasticityType::instance().unique_par_object_id();
    }

    //! Pack this class so it can be communicated
    void pack(Core::Communication::PackBuffer& data) const override;

    //! Unpack data from a char vector into this class
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of Packing and Unpacking                                             */
    /*                                                                           */
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name access methods                                                     */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! return material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_crystplast;
    }

    //! check whether element kinematics and material kinematics are compatible
    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (!(kinem == Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("Element and material kinematics are not compatible");
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<CrystalPlasticity>(*this);
    }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    //! return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of access methods                                                    */
    /*                                                                           */
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name evaluation methods                                                 */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! setup and initialize internal and variables
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    //! set up the slip/twinning directions and slip/twinning plane normals for the given lattice
    //! type
    void setup_lattice_vectors();

    //! read lattice orientation matrix from an input file
    void setup_lattice_orientation(const Discret::Elements::Fibers& fibers);

    //! update internal variables
    void update() override;

    //! evaluate material law
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    //! transform Miller Bravais index notation of hexagonal lattices to Miller index notation
    void miller_bravais_to_miller(
        const std::vector<Core::LinAlg::Matrix<4, 1>>&
            plane_normal_hex,  //!< [IN] vector of slip/twinning plane
                               //!< normals in Miller-Bravais index notation
        const std::vector<Core::LinAlg::Matrix<4,
            1>>& direction_hex,  //!< [IN] vector of slip/twinning directions in Miller-Bravais
                                 //!< index notation
        std::vector<Core::LinAlg::Matrix<3, 1>>&
            plane_normal,  //!< [OUT] vector of slip/twinning plane normals in Miller index notation
        std::vector<Core::LinAlg::Matrix<3, 1>>&
            Dir  //!< [OUT] vector of slip/twinning directions in Miller index notation
    );

    //! check if two vectors are parallel by checking the angle between them
    bool check_parallel(const Core::LinAlg::Matrix<3, 1>& vector_1,  //!< [IN] vector 1
        const Core::LinAlg::Matrix<3, 1>& vector_2                   //!< [IN] vector 2
    );
    //! check if two vectors are orthogonal by checking the angle between them
    bool check_orthogonal(const Core::LinAlg::Matrix<3, 1>& vector_1,  //!< [IN] vector 1
        const Core::LinAlg::Matrix<3, 1>& vector_2                     //!< [IN] vector 2
    );
    //! local Newton-Raphson iteration
    //! this method identifies the plastic shears gamma_res and defect densities def_dens_res
    //! as well as the stress PK2_res for a given deformation gradient F
    void newton_raphson(Core::LinAlg::Matrix<3, 3>& deform_grad,  //!< [IN] deformation gradient
        std::vector<double>& gamma_res,  //!< [OUT] result vector of plastic shears
        std::vector<double>&
            defect_densites_result,  //!< [OUT] result vector of defect densities (dislocation
                                     //!< densities and twinned volume fractions)
        Core::LinAlg::Matrix<3, 3>& second_pk_stress_result,  //!< [OUT] 2nd Piola-Kirchhoff stress
        Core::LinAlg::Matrix<3, 3>&
            plastic_deform_grad_result  //!< [OUT] plastic deformation gradient
    );

    //! Evaluates the flow rule for a given total deformation gradient F,
    //! and a given vector of plastic shears gamma_trial and
    //! sets up the respective residuals residuals_trial, the 2nd Piola-Kirchhoff stress PK2_trial
    //! and trial defect densities def_dens_trial
    void setup_flow_rule(Core::LinAlg::Matrix<3, 3> deform_grad,  //!< [IN] deformation gradient
        std::vector<double> gamma_trial,  //!< [OUT] trial vector of plastic shears
        Core::LinAlg::Matrix<3, 3>&
            plastic_deform_grad_trial,  //!< [OUT] plastic deformation gradient
        std::vector<double>&
            defect_densities_trial,  //!< [OUT] trial vector of defect densities (dislocation
                                     //!< densities and twinned volume fractions)
        Core::LinAlg::Matrix<3, 3>& second_pk_stress_trial,  //!< [OUT] 2nd Piola-Kirchhoff stress
        std::vector<double>& residuals_trial                 //!< [OUT] vector of slip residuals
    );


    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of evaluation methods                                                */
    /*                                                                           */
    //-----------------------------------------------------------------------------

   private:
    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name General Parameters                                                 */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! model parameters
    Mat::PAR::CrystalPlasticity* params_;

    //! Gauss point number
    int gp_;

    //! time increment
    double dt_;

    //! indicator whether the material model has been initialized already
    bool isinit_ = false;

    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of General Parameters                                                */
    /*                                                                           */
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name User Input                                                         */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! General Properties:
    //!-------------------
    //! tolerance for local Newton Raphson iteration
    double newton_tolerance_;

    //! Elastic Properties:
    //!-------------------------
    //! Young's Modulus
    double youngs_mod_;
    //! Poisson's ratio
    double poisson_ratio_;

    //! Crystal Properties:
    //!-------------------
    //! Lattice type
    std::string lattice_type_;
    //! c to a ratio of the crystal's unit cell
    double c_to_a_ratio_;
    //! Lattice constant a of unit cell
    double lattice_constant_;
    //! number of slip systems
    int slip_system_count_;
    //! number of twinning systems
    int twin_system_count_;
    //! Index to which subset a slip system belongs
    std::vector<int> slip_set_index_;
    //! Index to which subset a twinning system belongs
    std::vector<int> twin_set_index_;

    //! Viscoplastic Properties:
    //!------------------------
    //! reference slip shear rates
    std::vector<double> gamma_dot_0_slip_;
    //! strain rate sensitivity exponents for slip
    std::vector<int> n_slip_;
    //! reference twinning shear rates
    std::vector<double> gamma_dot_0_twin_;
    //! strain rate sensitivity exponents for twinning
    std::vector<int> n_twin_;

    //! Dislocation Generation/Recovery:
    //!--------------------------------
    //! initial dislocation density
    std::vector<double> initial_dislocation_density_;
    //! dislocation generation coefficients
    std::vector<double> dislocation_generation_coeff_;
    //! dynamic dislocation removal coefficients
    std::vector<double> dislocation_dyn_recovery_coeff_;

    //! Initial Slip/Twinning System Strengths:
    //!------------------------------
    //! lattice resistances to slip
    std::vector<double> tau_y_0_;
    //! lattice resistances to twinning
    std::vector<double> tau_t_0_;
    //! microstructural parameters which are relevant for Hall-Petch strengthening, e.g., grain size
    std::vector<double> micro_boundary_distance_slip_;
    //! microstructural parameters which are relevant for Hall-Petch strengthening, e.g., grain size
    std::vector<double> micro_boundary_distance_twin_;
    //! Hall-Petch coefficients corresponding to above microstructural boundaries
    std::vector<double> hall_petch_coeffs_slip_;
    //! Hall-Petch coefficients corresponding to above microstructural boundaries
    std::vector<double> hall_petch_coeffs_twin_;

    //! Work Hardening Interactions:
    //!------------------------------
    //! vector of hardening coefficients of slip systems due to non-coplanar twinning
    std::vector<double> slip_by_twin_hard_;
    //! vector of hardening coefficients of twinning systems due to slip
    std::vector<double> twin_by_slip_hard_;
    //! vector of hardening coefficients of twinning systems due to non-coplanar twinning
    std::vector<double> twin_by_twin_hard_;

    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of User Input                                                      */
    /*                                                                           */
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name Variables Derived from User Input                                  */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! Elastic Properties:
    //!-------------------------
    //! 1st Lame constant
    double lambda_;
    //! Shear modulus / 2nd Lame constant
    double mue_;
    //! Bulk modulus
    double bulk_mod_;

    //! Crystal Properties:
    //!-------------------
    //! total number of slip and twinning systems slip_system_count_ + twin_system_count_
    int def_system_count_;
    //! Switch for mechanical twinning
    bool is_twinning_;
    //! magnitudes of Burgers vectors for slip systems
    std::vector<double> slip_burgers_mag_;
    //! magnitudes of Burgers vectors for twinning systems
    std::vector<double> twin_burgers_mag_;
    //! lattice orientation in terms of rotation matrix with respect to global coordinates
    Core::LinAlg::Matrix<3, 3> lattice_orientation_;
    //! slip plane normals and slip directions
    std::vector<Core::LinAlg::Matrix<3, 1>> slip_plane_normal_;
    std::vector<Core::LinAlg::Matrix<3, 1>> slip_direction_;
    //! twinning plane normals and twinning directions
    std::vector<Core::LinAlg::Matrix<3, 1>> twin_plane_normal_;
    std::vector<Core::LinAlg::Matrix<3, 1>> twin_direction_;
    //! indicator which slip and twinning systems are non-coplanar for work hardening
    std::vector<std::vector<bool>> is_non_coplanar_;
    //! deformation system identifier
    std::vector<std::string> def_system_id_;

    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of Variables Derived from User Input                                 */
    /*                                                                           */
    //-----------------------------------------------------------------------------

    //-----------------------------------------------------------------------------
    /*                                                                           */
    /** \name Internal / history variables                                       */
    /** @{                                                                       */
    //-----------------------------------------------------------------------------

    //! old, i.e. at t=t_n
    //! deformation gradient at each Gauss-point
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> deform_grad_last_;
    //! plastic part of deformation gradient at each Gauss-point
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> plastic_deform_grad_last_;
    //! vector of plastic shears (slip and twinning)
    std::shared_ptr<std::vector<std::vector<double>>> gamma_last_;
    //! vector of dislocation densities (dislocations densities and twinned volume fractions)
    std::shared_ptr<std::vector<std::vector<double>>> defect_densities_last_;

    //! current, i.e. at t=t_n+1
    //!  deformation gradient at each Gauss-point
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> deform_grad_current_;
    //! plastic part of deformation gradient at each Gauss-point
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 3>>> plastic_deform_grad_current_;
    //! vector of plastic shears (slip and twinning)
    std::shared_ptr<std::vector<std::vector<double>>> gamma_current_;
    //! vector of defect densities (dislocations densities and twinned volume fractions)
    std::shared_ptr<std::vector<std::vector<double>>> defect_densities_current_;
    //-----------------------------------------------------------------------------
    /** @}                                                                       */
    /*  end of Internal / history variables                                      */
    /*                                                                           */
    //-----------------------------------------------------------------------------

  };  // class CrystalPlasticity
}  // namespace Mat

/*----------------------------------------------------------------------*/

FOUR_C_NAMESPACE_CLOSE

#endif
