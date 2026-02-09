// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_PLASTICLINELAST_HPP
#define FOUR_C_MAT_PLASTICLINELAST_HPP


/*----------------------------------------------------------------------*
 | headers                                                   dano 04/11 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for linear elasto-plastic material
    class PlasticLinElast : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      PlasticLinElast(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! mass density
      const double density_;
      //! initial yield stress (constant)
      const double yield_;
      //! linear isotropic hardening modulus
      const double isohard_;
      //! linear kinematic hardening modulus
      const double kinhard_;
      //! tolerance for local Newton iteration
      const double abstol_;

      //@}

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class PlasticLinElast

  }  // namespace PAR


  class PlasticLinElastType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "PlasticLinElastType"; }

    static PlasticLinElastType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static PlasticLinElastType instance_;

  };  // PlasticLinElastType


  /*----------------------------------------------------------------------*/
  //! wrapper for linear elasto-plastic material
  class PlasticLinElast : public So3Material
  {
   public:
    //! construct empty material object
    PlasticLinElast();

    //! construct the material object given material parameters
    explicit PlasticLinElast(Mat::PAR::PlasticLinElast* params);

    //! @name Packing and Unpacking

    //!  \brief Return unique ParObject id
    //!
    //!  every class implementing ParObject needs a unique id defined at the
    //!  top of parobject.H (this file) and should return it in this method.
    int unique_par_object_id() const override
    {
      return PlasticLinElastType::instance().unique_par_object_id();
    }

    //!  \brief Pack this class so it can be communicated
    //!
    //!  Resizes the vector data and stores all information of a class in it.
    //!  The first information to be stored in data has to be the
    //!  unique parobject id delivered by unique_par_object_id() which will then
    //!  identify the exact class on the receiving processor.
    //!
    //!  \param data (in/out): char vector to store class information
    void pack(Core::Communication::PackBuffer&
            data  //!<  data (i/o): char vector to store class information
    ) const override;

    //!  \brief Unpack data from a char vector into this class
    //!
    //!  The vector data contains all information to rebuild the
    //!  exact copy of an instance of a class on a different processor.
    //!  The first entry in data has to be an integer which is the unique
    //!  parobject id defined at the top of this file and delivered by
    //!  unique_par_object_id().
    //!
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_pllinelast;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::linear))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<PlasticLinElast>(*this);
    }

    //! initialise internal stress variables
    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    //! update internal stress variables
    void update() override;

    //! evaluate material
    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    //! computes isotropic elasticity tensor in matrix notion for 3d
    void setup_cmat(
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  //!< elastic material tangent
        double mu,                                                //!< lame constant mu (=G)
        double lambda                                             //!< lame constant lambda
    ) const;

    //! computes isotropic elastoplastic tensor for 3d
    void setup_cmat_elasto_plastic(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&
                                       cmat,            //!< elasto-plastic material tangent
        double Dgamma,                                  //!< plastic multiplier
        double G,                                       //!< shear modulus
        double q,                                       //!< effective stress
        Core::LinAlg::SymmetricTensor<double, 3, 3> N,  // unit flow vector
        double Hiso,                                    //!< isotropic hardening modulus
        double Hkin                                     //!< kinematic hardening modulus
    ) const;

    void setup_continuum_cmat_elasto_plastic(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&
                                                 cmat,   //!< elasto-plastic material tangent
        double Dgamma,                                   //!< plastic multiplier
        double q,                                        //!< effective stress
        Core::LinAlg::Matrix<NUM_STRESS_3D, 1> unitflow  //!< unit flow vector
    ) const;

    //! return density
    double density() const override { return params_->density_; }

    //! return accumulated strain at Gauss points
    //! use the old vector (last_) for postprocessing
    //! Output is called after(!!) Update, so that newest values are included in
    //! the old history vectors last_, while the current history vectors curr_
    //! are reset
    double accumulated_strain(int gp  //!< current Gauss point
    ) const
    {
      return strainbarpllast_.at(gp);
    }

    //! check if history variables are already initialised
    bool initialized() const { return (isinit_ and !strainplcurr_.empty()); }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    //! return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //! return names of visualization data available for direct VTK output
    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    //! return visualization data for direct VTK output
    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

    //! finite difference check for debugging purposes
    void fd_check(Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,  // updated stress sigma_n+1
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>&
            cmatFD,  // material tangent calculated with FD of stresses
        Core::LinAlg::SymmetricTensor<double, 3, 3>& beta,          // updated back stresses
        double p,                                                   // volumetric stress
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& strain,  // elastic trial strain vector
        double Dgamma,                                              // plastic multiplier
        double G,                                                   // shear modulus
        double qbar,                                    // elastic trial von Mises effective stress
        double kappa,                                   // bulk modulus
        Core::LinAlg::SymmetricTensor<double, 3, 3>& N  // flow vector
    );

   private:
    //! my material parameters
    Mat::PAR::PlasticLinElast* params_;

    //! plastic history vector
    //! old plastic strain at t_n
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
        strainpllast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current plastic strain at t_n+1
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
        strainplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old accumulated plastic strain at t_n
    std::vector<double> strainbarpllast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current accumulated plastic strain at t_n+1
    std::vector<double> strainbarplcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old back stress magnitude at t_n
    std::vector<double> betabarlast_;  //!< \f${\varepsilon}^p_{n}\f$
    //! current back stress magnitude at t_n+1
    std::vector<double> betabarcurr_;  //!< \f${\varepsilon}^p_{n+1}\f$
    //! old back stress at t_n
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
        backstresslast_;  //!< \f${\beta}_{n}\f$
    //! current back stress at t_n+1
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>>
        backstresscurr_;  //!< \f${\beta}_{n+1}\f$

    //! indicator if #Initialize routine has been called
    bool isinit_;
    //! indicator if material has started to be plastic
    bool plastic_step_;

  };  // class PlasticLinElast : public Core::Mat::Material
}  // namespace Mat


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
