// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_STRUCTPORO_HPP
#define FOUR_C_MAT_STRUCTPORO_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  class StructPoro;

  namespace PAR
  {
    class PoroLaw;

    class StructPoro : public Core::Mat::PAR::Parameter
    {
      friend class Mat::StructPoro;

     public:
      //! standard constructor
      StructPoro(const Core::Mat::PAR::Parameter::Data& matdata);

      //! create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      //! @name material parameters
      //!@{

      //! material ID of sub-material
      int matid_;

      //! poro law ID
      int poro_law_ID_;

      //! initial porosity
      double init_porosity_;

      //!@}

      //! implementation of porosity law
      PoroLaw* poro_law_;
    };

  }  // namespace PAR

  class StructPoroType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "StructPoroType"; }

    static StructPoroType& instance() { return instance_; }

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static StructPoroType instance_;
  };

  /*----------------------------------------------------------------------*/
  //! Wrapper for StructPoro material
  //!
  //! This object exists (several times) at every element

  /*!
    The idea is to use any material formulation within the poro framework.
    Therefore, a poro material wraps the 'real' material and holds it as
    a private member. For most evaluation routines it will just call this material.
    In addition it provides poro specific functions, as giving the constitutive law
    for the porosity.

    Main methods of this material are the compute_porosity(...) methods, providing
    the porosity and its derivatives. If the constitutive law is a governing equation
    itself (for poro P1 elements, for instance), the material evaluates the
    consitutive law itself and its derivatives in the ConstituitiveDerivatives(...)
    methods.
    All other evaluate() methods are basically passed through to the underlying
    structure material.

    The poro material can save the porosity gauss point wise. Therefore it
    has an additional setup method, giving the number of gauss points. This is
    only (!) meant for post processing/visualization processes! The gauss point
    wise saved porosity must not be used during simulation as it is not
    guaranteed (and actually not the case) that the gauss point numbering
    is the same for every element (especially for e.g. fluid and solid elements).
   */
  class StructPoro : public So3Material
  {
   public:
    //! construct empty material object
    StructPoro();

    //! construct the material object given material parameters
    explicit StructPoro(Mat::PAR::StructPoro* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return StructPoroType::instance().unique_par_object_id();
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

    //!@}

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_structporo;
    }

    //! poro law type
    virtual Core::Materials::MaterialType poro_law_type() const;

    //! return inverse bulk modulus (=compressibility)
    double inv_bulk_modulus() const;

    //! check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override { mat_->valid_kinematics(kinem); }

    //! return material
    std::shared_ptr<Core::Mat::Material> get_material() const { return mat_; }

    //! return material ID
    int mat_id() const { return params_->matid_; }

    //! return porosity average (for post processing only!)
    double porosity_av() const;

    //! return initial porosity
    double init_porosity() const { return params_->init_porosity_; }

    //! return time derivative of reference porosity (only nonzero with reaction)
    virtual double ref_porosity_time_deriv() const { return 0.0; }

    //! compute current porosity and save it
    virtual void compute_porosity(
        const Teuchos::ParameterList& params,  //!< (i) element parameter list
        double press,                          //!< (i) pressure at gauss point
        double J,                              //!< (i) determinant of jacobian at gauss point
        int gp,                                //!< (i) number of current gauss point
        double& porosity,                      //!< (o) porosity at gauss point
        double* dphi_dp,  //!< (o) first derivative of porosity w.r.t. pressure at gauss point
        double* dphi_dJ,  //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
        double*
            dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss point
        double* dphi_dJJ,  //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
        double* dphi_dpp,  //!< (o) second derivative of porosity w.r.t. pressure at gauss point
        bool save = true);

    //! compute current porosity and save it
    void compute_porosity(const Teuchos::ParameterList& params,  //!< (i) element parameter list
        double press,                                            //!< (i) pressure at gauss point
        double J,          //!< (i) determinant of jacobian at gauss point
        int gp,            //!< (i) number of current gauss point
        double& porosity,  //!< (o) porosity at gauss point
        bool save = true);

    //! compute current surface porosity and save it
    void compute_surf_porosity(
        const Teuchos::ParameterList& params,  //!< (i) element parameter list
        double press,                          //!< (i) pressure at gauss point
        double J,                              //!< (i) determinant of jacobian at gauss point
        const int surfnum,                     //!< (i) number of surface
        int gp,                                //!< (i) number of current gauss point
        double& porosity,                      //!< (o) porosity at gauss point
        double* dphi_dp,  //!< (o) first derivative of porosity w.r.t. pressure at gauss point
        double* dphi_dJ,  //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
        double*
            dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss point
        double* dphi_dJJ,  //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
        double* dphi_dpp,  //!< (o) second derivative of porosity w.r.t. pressure at gauss point
        bool save = true);

    //! compute current surface porosity and save it
    void compute_surf_porosity(
        const Teuchos::ParameterList& params,  //!< (i) element parameter list
        double press,                          //!< (i) pressure at gauss point
        double J,                              //!< (i) determinant of jacobian at gauss point
        const int surfnum,                     //!< (i) number of surface
        int gp,                                //!< (i) number of current gauss point
        double& porosity,                      //!< (o) porosity at gauss point
        bool save = true);

    //! return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<StructPoro>(*this);
    }

    //! Initialize internal variables
    virtual void poro_setup(int numgp,  //!< number of Gauss points
        const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system);

    //! evaluate constitutive relation for porosity and compute derivatives
    virtual void constitutive_derivatives(
        const Teuchos::ParameterList& params,  //!< (i) parameter list
        double press,                          //!< (i) fluid pressure at gauss point
        double J,                              //!< (i) Jacobian determinant at gauss point
        double porosity,                       //!< (i) porosity at gauss point
        double* dW_dp,                         //!< (o) derivative of potential w.r.t. pressure
        double* dW_dphi,                       //!< (o) derivative of potential w.r.t. porosity
        double* dW_dJ,                         //!< (o) derivative of potential w.r.t. jacobian
        double* dW_dphiref,  //!< (o) derivative of potential w.r.t. reference porosity
        double* W            //!< (o) inner potential
    );

    //! evaluate constitutive relation for porosity and compute derivatives using reference porosity
    void constitutive_derivatives(const Teuchos::ParameterList& params,  //!< (i) parameter list
        double press,        //!< (i) fluid pressure at gauss point
        double J,            //!< (i) Jacobian determinant at gauss point
        double porosity,     //!< (i) porosity at gauss point
        double refporosity,  //!< (i) porosity at gauss point
        double* dW_dp,       //!< (o) derivative of potential w.r.t. pressure
        double* dW_dphi,     //!< (o) derivative of potential w.r.t. porosity
        double* dW_dJ,       //!< (o) derivative of potential w.r.t. jacobian
        double* dW_dphiref,  //!< (o) derivative of potential w.r.t. reference porosity
        double* W            //!< (o) inner potential
    );

    //! Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! @name Evaluation methods

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override
    {
      mat_->evaluate(defgrad, glstrain, params, stress, cmat, gp, eleGID);
    }

    [[nodiscard]] double strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        int gp, int eleGID) const override
    {
      return mat_->strain_energy(glstrain, gp, eleGID);
    }

    double evaluate_cauchy_n_dir_and_derivatives(const Core::LinAlg::Tensor<double, 3, 3>& defgrd,
        const Core::LinAlg::Tensor<double, 3>& n, const Core::LinAlg::Tensor<double, 3>& dir,
        Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir,
        Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF, Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
        Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, int gp, int eleGID,
        const double* concentration, const double* temp, double* d_cauchyndir_dT,
        Core::LinAlg::Matrix<9, 1>* d2_cauchyndir_dF_dT) override
    {
      return mat_->evaluate_cauchy_n_dir_and_derivatives(defgrd, n, dir, d_cauchyndir_dn,
          d_cauchyndir_ddir, d_cauchyndir_dF, d2_cauchyndir_dF2, d2_cauchyndir_dF_dn,
          d2_cauchyndir_dF_ddir, gp, eleGID, concentration, temp, d_cauchyndir_dT,
          d2_cauchyndir_dF_dT);
    }

    //!@}

    //! Return material density (if provided by the specific material)
    double density() const override;
    virtual double density_solid_phase() const;

    //! @name Handling of Gauss point data. Here, the poro material just calls the underlying
    //! material

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override
    {
      // setup the underlying material
      // Note: poro material itself is setup when calling poro_setup()
      mat_->setup(numgp, fibers, coord_system);
    }

    void post_setup(const Teuchos::ParameterList& params, const int eleGID) override;

    void update() override { mat_->update(); }

    void reset_step() override { mat_->reset_step(); }

    //!@}

    //! @name Visualization methods

    void vis_names(std::map<std::string, int>& names) const override;

    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    //!@}

   protected:
    //! compute current porosity and save it
    void compute_porosity(
        const double& refporosity,  //!< (i) initial/reference porosity at gauss point
        const double& press,        //!< (i) pressure at gauss point
        const double& J,            //!< (i) determinant of jacobian at gauss point
        const int& gp,              //!< (i) number of current gauss point
        double& porosity,           //!< (o) porosity at gauss point
        double* dphi_dp,  //!< (o) first derivative of porosity w.r.t. pressure at gauss point
        double* dphi_dJ,  //!< (o) first derivative of porosity w.r.t. jacobian at gauss point
        double*
            dphi_dJdp,  //!< (o) derivative of porosity w.r.t. pressure and jacobian at gauss point
        double* dphi_dJJ,      //!< (o) second derivative of porosity w.r.t. jacobian at gauss point
        double* dphi_dpp,      //!< (o) second derivative of porosity w.r.t. pressure at gauss point
        double* dphi_dphiref,  //!< (o) derivative of porosity w.r.t. reference porosity (only
                               //!< nonzero with reaction)
        bool save = true);

    //! my material parameters
    Mat::PAR::StructPoro* params_;

    //! actual material
    std::shared_ptr<Mat::So3Material> mat_;

    //! porosity at gauss points
    std::shared_ptr<std::vector<double>> porosity_;

    //! porosity at gauss points of surface element
    std::shared_ptr<std::map<int, std::vector<double>>> surf_porosity_;

    //! flag indicating initialization of attributes
    bool is_initialized_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
