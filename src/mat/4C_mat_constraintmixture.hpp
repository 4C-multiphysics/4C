// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_CONSTRAINTMIXTURE_HPP
#define FOUR_C_MAT_CONSTRAINTMIXTURE_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters
    class ConstraintMixture : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ConstraintMixture(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{
      /// density
      const double density_;
      /// shear modulus
      const double mue_;
      /// Poisson's ratio
      const double nue_;
      /// mass fraction of elastin
      const double phielastin_;
      /// prestretch of elastin
      const double prestretchelastin_;
      /// parameter for linear fiber stiffness of collagen
      const double k1_;
      /// parameter for exponential fiber stiffness of collagen
      const double k2_;
      /// number of homeostatic variables
      const int numhom_;
      /// prestretch of collagen fibers
      const std::vector<double> prestretchcollagen_;
      /// stretch at which collagen fibers are damaged
      const double damagestretch_;
      /// parameter for linear fiber stiffness of smooth muscle
      const double k1muscle_;
      /// parameter for exponential fiber stiffness of smooth muscle
      const double k2muscle_;
      /// mass fraction of smooth muscle
      const double phimuscle_;
      /// prestretch of smooth muscle fibers
      const double prestretchmuscle_;
      /// maximal active stress
      const double Smax_;
      /// dilatation modulus
      const double kappa_;
      /// lifetime of collagen fibers
      const double lifetime_;
      /// growth factor for stress
      // const double growthfactor_;
      /// homeostatic target value of scalar stress measure
      const std::vector<double> homstress_;
      /// growth factor for shear
      const double sheargrowthfactor_;
      /// homeostatic target value of inner radius
      const double homradius_;
      /// at this time turnover of collagen starts
      const double starttime_;
      /// time integration scheme (Explicit,Implicit)
      const std::string integration_;
      /// tolerance for local Newton iteration
      const double abstol_;
      /// driving force of growth (Single,All,ElaCol)
      const std::string growthforce_;
      /// form of elastin  degradation
      const std::string elastindegrad_;
      /// how mass depends on driving force
      const std::string massprodfunc_;
      /// how to set stretches in the beginning
      const std::string initstretch_;
      /// number of timecurve for increase of prestretch in time
      const int timecurve_;
      /// which degradation function
      const std::string degoption_;
      /// maximal factor of mass production
      const double maxmassprodfac_;
      /// store all history variables
      const bool storehistory_;
      /// tolerance for degradation
      const double degtol_;
      //@}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      // !brief enum for mapping between material parameter and entry in the matparams_ vector
      enum Matparamnames
      {
        growthfactor,
        elastin_survival,
        first = growthfactor,
        last = elastin_survival
      };

      double get_parameter(int parametername, const int EleId)
      {
        // check if we have an element based value via size
        if (matparams_[parametername]->global_length() == 1)
        {
          // we have a global value hence we directly return the first entry
          return matparams_[parametername]->local_values_as_span()[0];
        }
        // If someone calls this functions without a valid EleID and we have element based values
        // throw error
        else if (EleId < 0 && matparams_[parametername]->global_length() > 1)
        {
          FOUR_C_THROW("Global mat parameter requested but we have elementwise mat params");
          return 0.0;
        }
        // otherwise just return the element specific value
        else
        {
          // calculate LID here, instead of before each call
          return matparams_[parametername]
              ->local_values_as_span()[matparams_[parametername]->get_map().lid(EleId)];
        }
      }

     private:
      std::vector<std::shared_ptr<Core::LinAlg::Vector<double>>> matparams_;
    };  // class ConstraintMixture

  }  // namespace PAR

  class ConstraintMixtureHistory;

  class ConstraintMixtureType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ConstraintMixtureType"; }

    static ConstraintMixtureType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ConstraintMixtureType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for constraint mixture material
  class ConstraintMixture : public So3Material
  {
   public:
    /// construct empty material object
    ConstraintMixture();

    /// construct the material object given material parameters
    explicit ConstraintMixture(Mat::PAR::ConstraintMixture* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ConstraintMixtureType::instance().unique_par_object_id();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by unique_par_object_id() which will then
      identify the exact class on the receiving processor.
      This material contains history variables, which are packed for restart purposes.

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
      History data is unpacked in restart.

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_constraintmixture;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ConstraintMixture>(*this);
    }

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    /// SetupHistory
    void reset_all(const int numgp);

    /// Update
    void update() override;

    /// Reset time step
    void reset_step() override;

    /// Evaluate material
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    /// Return density
    double density() const override { return params_->density_; }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// Return variables for visualization
    Core::LinAlg::Matrix<3, 1> get_vis(int gp) const { return vismassstress_->at(gp); }
    /// Return actual mass density in reference configuration
    double get_mass_density(int gp) const { return refmassdens_->at(gp); }
    /// Return actual mass density in reference configuration
    Core::LinAlg::Matrix<3, 1> get_mass_density_collagen(int gp) const
    {
      return visrefmassdens_->at(gp);
    }
    /// Return prestretch of collagen fibers
    Core::LinAlg::Matrix<3, 1> get_prestretch(int gp) const
    {
      Core::LinAlg::Matrix<3, 1> visprestretch(Core::LinAlg::Initialization::zero);
      visprestretch(0) = localprestretch_->at(gp)(0);
      visprestretch(1) = localprestretch_->at(gp)(1);
      visprestretch(2) = localprestretch_->at(gp)(2);
      return visprestretch;
    }
    /// Return prestretch of collagen fibers
    Core::LinAlg::Matrix<3, 1> get_homstress(int gp) const
    {
      Core::LinAlg::Matrix<3, 1> visprestretch(Core::LinAlg::Initialization::zero);
      visprestretch(0) = localhomstress_->at(gp)(0);
      visprestretch(1) = localhomstress_->at(gp)(1);
      visprestretch(2) = localhomstress_->at(gp)(2);
      return visprestretch;
    }
    /// Return circumferential fiber direction
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> geta1() const { return a1_; }
    /// Return axial fiber direction
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> geta2() const { return a2_; }

    /// evaluate fiber directions from locsys, pull back
    void evaluate_fiber_vecs(const int gp, const Core::LinAlg::Matrix<3, 3>& locsys,
        const Core::LinAlg::Matrix<3, 3>& defgrd);

    /// Return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    /// Return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

   private:
    /// my material parameters
    Mat::PAR::ConstraintMixture* params_;

    /// temporary for visualization
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> vismassstress_;
    /// actual mass density in reference configuration
    std::shared_ptr<std::vector<double>> refmassdens_;
    /// actual mass density in reference configuration for collagen fibers
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> visrefmassdens_;
    /// basal rate of mass production
    double massprodbasal_;

    /// first fiber vector per gp (reference), circumferential
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> a1_;
    /// second fiber vector per gp (reference), axial
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> a2_;
    /// third fiber vector per gp (reference), diagonal
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> a3_;
    /// fourth fiber vector per gp (reference), diagonal
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> a4_;
    /// homeostatic prestretch of collagen fibers
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<4, 1>>> localprestretch_;
    /// homeostatic stress for growth
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<4, 1>>> localhomstress_;
    /// homeostatic radius
    double homradius_;
    /// list of fibers which have been overstretched and are deleted at the end of the time step
    /// (gp, idpast, idfiber)
    std::shared_ptr<std::vector<Core::LinAlg::Matrix<3, 1>>> deletemass_;
    /// index of oldest fiber still alive, needed if the complete history is stored to avoid
    /// summation of zeros
    int minindex_;
    /// indicates if material is initialized
    bool isinit_;

    /// history
    std::shared_ptr<std::vector<ConstraintMixtureHistory>> history_;

    /// compute stress and cmat
    void evaluate_stress(
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,    ///< green lagrange strain
        const int gp,                                              ///< current Gauss point
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,            ///< 2nd PK-stress
        const int firstiter,  ///< iteration index, different for explicit and implicit integration
        double time,          ///< time
        double elastin_survival  ///< amount of elastin which is still there
    );

    /// elastic response for one collagen fiber family
    void evaluate_fiber_family(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,  ///< Cauchy-Green
        const int gp,                                              ///< current Gauss point
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,            ///< 2nd PK-stress
        Core::LinAlg::Matrix<3, 1> a,                              ///< fiber vector
        double* currmassdens,                                      ///< current massdensity
        const int firstiter,  ///< iteration index, different for explicit and implicit integration
        double time,          ///< time
        const int idfiber     ///< number of fiber family 0,1,2,3
    );

    /// elastic response for one collagen fiber
    void evaluate_single_fiber_scalars(
        double
            I4,  ///< fourth invariant multiplied with (prestretch / stretch at deposition time)^2
        double& fac_cmat,   ///< scalar factor for material stiffness matrix
        double& fac_stress  ///< scalar factor for 2nd PK-stress
    );

    /// elastic response for Elastin
    void evaluate_elastin(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,  ///< Cauchy-Green
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,            ///< 2nd PK-stress
        double time,                                               ///< time
        double* currmassdens,                                      ///< current massdensity
        double elastin_survival  ///< amount of elastin which is still there
    );

    /// elastic response for smooth muscle cells
    void evaluate_muscle(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,  ///< Cauchy-Green
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,            ///< 2nd PK-stress
        const int gp,                                              ///< current Gauss point
        double* currmassdens                                       ///< current massdensity
    );

    /// volumetric part
    void evaluate_volumetric(const Core::LinAlg::Matrix<NUM_STRESS_3D, 1> C,  ///< Cauchy-Green
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,            ///< 2nd PK-stress
        double currMassDens,                                       ///< current mass density
        double refMassDens                                         ///< initial mass density
    );

    /// computes mass production rates for all fiber families
    void mass_production(const int gp,             ///< current Gauss point
        Core::LinAlg::Matrix<3, 3> defgrd,         ///< deformation gradient
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S,  ///< 2nd PK-stress
        Core::LinAlg::Matrix<4, 1>* massstress,    ///< growth stress measure
        double inner_radius,                       ///< inner radius
        Core::LinAlg::Matrix<4, 1>* massprodcomp,  ///< mass production rate
        double growthfactor                        ///< growth factor for stress
    );

    /// computes mass production rate for one fiber family
    void mass_production_single_fiber(const int gp,  ///< current Gauss point
        Core::LinAlg::Matrix<3, 3> defgrd,           ///< deformation gradient
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> S,    ///< 2nd PK-stress
        double* massstress,                          ///< growth stress measure
        double inner_radius,                         ///< inner radius
        double* massprodcomp,                        ///< mass production rate
        Core::LinAlg::Matrix<3, 1> a,                ///< fiber vector
        const int idfiber,                           ///< number of fiber family 0,1,2,3
        double growthfactor                          ///< growth factor for stress
    );

    /// function for massproduction
    void mass_function(double growthfac,  ///< growth factor K
        double delta,                     ///< difference in stress or wall shear stress
        double mmax,                      ///< maximal massproduction fac
        double& massfac                   ///< factor
    );

    /// returns how much collagen has been degraded
    void degradation(double t, double& degr);

    /// function of elastin degradation (initial)
    void elastin_degradation(
        Core::LinAlg::Tensor<double, 3> coord,  ///< gp coordinate in reference configuration
        double& elastin_survival                ///< amount of elastin which is still there
    ) const;

    /// compute stress and cmat for implicit integration with whole stress as driving force
    void evaluate_implicit_all(Core::LinAlg::Matrix<3, 3> defgrd,  ///< deformation gradient
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,    ///< green lagrange strain
        const int gp,                                              ///< current Gauss point
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,  ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,            ///< 2nd PK-stress
        double dt,                                                 ///< delta time
        double time,                                               ///< time
        Core::LinAlg::Matrix<4, 1> massprodcomp,                   ///< mass production rate
        Core::LinAlg::Matrix<4, 1> massstress,                     ///< growth stress measure
        double elastin_survival,  ///< amount of elastin which is still there
        double growthfactor       ///< growth factor for stress
    );

    /// compute stress and cmat for implicit integration with fiber stress as driving force
    void evaluate_implicit_single(Core::LinAlg::Matrix<3, 3> defgrd,  ///< deformation gradient
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,       ///< green lagrange strain
        const int gp,                                                 ///< current Gauss point
        Core::LinAlg::Matrix<NUM_STRESS_3D, NUM_STRESS_3D>* cmat,     ///< material stiffness matrix
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* stress,               ///< 2nd PK-stress
        double dt,                                                    ///< delta time
        double time,                                                  ///< time
        double elastin_survival,  ///< amount of elastin which is still there
        double growthfactor       ///< growth factor for stress
    );

    /// derivative of stress with respect to massproduction of a single fiber family
    void grad_stress_d_mass(
        const Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* glstrain,  ///< green lagrange strain
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* derivative,      ///< result
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv,             ///< inverse cauchy green strain
        Core::LinAlg::Matrix<3, 1> a,                            ///< fiber vector
        double stretch,  ///< prestretch / stretch at deposition time
        double J,        ///< determinant of F
        double dt,       ///< delta time
        bool option      ///< which driving force
    );

    /// derivative of massproduction of a single fiber family with respect to stress
    void grad_mass_d_stress(Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* derivative,  ///< result
        Core::LinAlg::Matrix<3, 3> defgrd,   ///< deformation gradient
        Core::LinAlg::Matrix<3, 3> Smatrix,  ///< 2nd PK-stress in matrix notation
        Core::LinAlg::Matrix<3, 1> a,        ///< fiber vector
        double J,                            ///< determinant of F
        double massstress,                   ///< growth stress measure
        double homstress,                    ///< homeostatic stress
        double actcollstretch,               ///< actual collagen stretch
        double growthfactor                  ///< growth factor for stress
    );

    /// derivative of massproduction of a single fiber family with respect to stretch
    void grad_mass_d_stretch(Core::LinAlg::Matrix<NUM_STRESS_3D, 1>* derivative,  ///< result
        Core::LinAlg::Matrix<3, 3> defgrd,   ///< deformation gradient
        Core::LinAlg::Matrix<3, 3> Smatrix,  ///< 2nd PK-stress in matrix notation
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> Cinv,
        Core::LinAlg::Matrix<3, 1> a,  ///< fiber vector
        double J,                      ///< determinant of F
        double massstress,             ///< growth stress measure
        double homstress,              ///< homeostatic stress
        double actcollstretch,         ///< actual collagen stretch
        double dt,                     ///< delta time
        double growthfactor            ///< growth factor for stress
    );
  };

  /// Debug output to gmsh-file
  /* this needs to be copied to Solid::TimInt::OutputStep() to enable debug output
  {
    discret_->set_state("displacement",Dis());
    Mat::ConstraintMixtureOutputToGmsh(discret_, StepOld(), 1);
  }
  don't forget to include constraintmixture.H */
  void constraint_mixture_output_to_gmsh(
      Core::FE::Discretization& dis,  ///< discretization with displacements
      const int timestep,             ///< index of timestep
      const int iter                  ///< iteration index of newton iteration
  );

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
