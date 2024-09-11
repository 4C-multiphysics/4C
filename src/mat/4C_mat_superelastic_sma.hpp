/*----------------------------------------------------------------------*/
/*! \file
\brief Material law for superelastic isotropic material
       following finite strain with linear and exponential
       flow rules.
       The implementation follows
       Auricchio F. and Taylor R.: "Shape-memory alloys: modelling and numerical
       simulations of finite-strain superelastic behavior."
       Computer Methods in Applied Mechanics and Engineering, 1997
           and
       Auricchio F.: A robust integration-algorithm for a finite-strain
       shape-memory-alloy superelastic model."
       International Journal of Plasticity, 2001


       geometrically nonlinear, finite strains, rate-independent, isothermal

       example input line for the exponential model:
       MAT 1 MAT_Struct_SuperElastSMA YOUNG 60000 DENS 1.0 NUE 0.3 EPSILON_L 0.075
       T_AS_s 0 T_AS_f 0 T_SA_s 0 T_SA_f 0 C_AS 0 C_SA 0 SIGMA_AS_s 520 SIGMA_AS_f 750
       SIGMA_SA_s 550 SIGMA_SA_f 200 ALPHA 0.15 MODEL 1 BETA_AS 250 BETA_SA 20

       example input line for the linear model:
       MAT 1 MAT_Struct_SuperElastSMA YOUNG 60000 DENS 1.0 NUE 0.3 EPSILON_L 0.075
       T_AS_s 0 T_AS_f 0 T_SA_s 0 T_SA_f 0 C_AS 0 C_SA 0 SIGMA_AS_s 520 SIGMA_AS_f 600
       SIGMA_SA_s 300 SIGMA_SA_f 200 ALPHA 0.15 MODEL 2 BETA_AS 0 BETA_SA 0

\level 3


*/
/*----------------------------------------------------------------------*
 | definitions                                            hemmler 09/16 |
 *----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_SUPERELASTIC_SMA_HPP
#define FOUR_C_MAT_SUPERELASTIC_SMA_HPP

/*----------------------------------------------------------------------*
 | headers                                                 hemmler 09/16 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    //! material parameters for neo-Hooke
    class SuperElasticSMA : public Core::Mat::PAR::Parameter
    {
     public:
      //! standard constructor
      SuperElasticSMA(const Core::Mat::PAR::Parameter::Data& matdata);

      //! @name material parameters
      //@{

      //! Density
      const double density_;
      //! Young's modulus
      const double youngs_;
      //! Possion's ratio
      const double poissonratio_;
      //! epsilon_tilde_L
      const double epsilon_L_;
      //! start temperature of the phase transformation from austenite to single variant martensite
      const double T_AS_s_;
      //! finish temperature of the phase transformation from austenite to single varinat martensite
      const double T_AS_f_;
      //! start temperature of the phase transformation from single variant martensite to austenite
      const double T_SA_s_;
      //! finish temperature of the phase transformation from single variant martensite to austenite
      const double T_SA_f_;
      //! Pressure dependence of the phase transformation from austenite to single variant
      //! martensite
      const double C_AS_;
      //! Pressure dependence of the phase transformation from single varinat martensite to
      //! austenite
      const double C_SA_;
      //! start stress of the phase transformation from austenite to single variant martensite
      const double sigma_AS_s_;
      //! finish stress of the phase transformation from austenite to single variant martensite
      const double sigma_AS_f_;
      //! start stress of the phase transformation from single variant martensite to austenite
      const double sigma_SA_s_;
      //! finish stress of the phase transformation from single variant martensite to austenite
      const double sigma_SA_f_;
      //! pressure dependency in the drucker-prager-type loading
      const double alpha_;

      //! variable for the model: exponential model (1) or linear model (2)
      const int model_;

      // The following material parameters are only necessary for the exponential model
      //! material parameter measuring the speed of transformation from austenite to single variant
      //! martensite
      const double beta_AS_;
      //! material parameter measuring the speed of transformation from single variant martensite to
      //! austenite
      const double beta_SA_;

      //@}

      //! create material instance of matching type with my parameters
      Teuchos::RCP<Core::Mat::Material> create_material() override;


    };  // class SuperElasticSMA

  }  // namespace PAR


  class SuperElasticSMAType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "SuperElasticSMAType"; }

    static SuperElasticSMAType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static SuperElasticSMAType instance_;

  };  // class SuperElasticSMAType

  /*----------------------------------------------------------------------*/
  //! wrapper for finite strain elasto-plastic material
  class SuperElasticSMA : public So3Material
  {
   public:
    //! construct empty material object
    SuperElasticSMA();

    //! construct the material object given material parameters
    explicit SuperElasticSMA(Mat::PAR::SuperElasticSMA* params);

    //! @name Packing and Unpacking

    /*!
    \brief Return unique ParObject id

    every class implementing ParObject needs a unique id defined at the
    top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return SuperElasticSMAType::instance().unique_par_object_id();
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

    //! @name Access methods

    //! material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_superelast;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! return copy of this material object
    Teuchos::RCP<Core::Mat::Material> clone() const override
    {
      return Teuchos::rcp(new SuperElasticSMA(*this));
    }

    //! density
    double density() const override { return params_->density_; }

    //! return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //! check if history variables are already initialized
    bool initialized() const { return (isinit_ and (xi_s_curr_ != Teuchos::null)); }

    //! return names of visualization data
    void vis_names(std::map<std::string, int>& names) const override;

    //! return visualization data
    bool vis_data(
        const std::string& name, std::vector<double>& data, int numgp, int eleID) const override;

    /// Return whether the material requires the deformation gradient for its evaluation
    bool needs_defgrd() const override { return true; };

    //@}

    //! @name Evaluation methods

    //! initialise internal stress variables
    void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

    //! update internal stress variables
    void update() override;

    //! evaluate material law
    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    /// evaluate strain energy function
    void strain_energy(
        const Core::LinAlg::Matrix<6, 1>& glstrain, double& psi, int gp, int eleGID) const override;

    //@}

   private:
    struct LoadingData;
    struct Material;

    //! my material parameters
    Mat::PAR::SuperElasticSMA* params_;

    //! Drucker-Prager-type loading of last time step
    Teuchos::RCP<std::vector<double>> druckerpragerloadinglast_;
    Teuchos::RCP<std::vector<double>> druckerpragerloadingcurr_;

    //! single variant martensitic fraction
    Teuchos::RCP<std::vector<double>> xi_s_last_;
    Teuchos::RCP<std::vector<double>> xi_s_curr_;

    //! Calculates the kronecker delta. Returns 1 for i==j otherwise 0
    virtual int kron(int i, int j);

    //! Returns the pullback of the 4th order tensor in voigt notation
    virtual void pullback4th_tensor_voigt(const double jacobian,
        const Core::LinAlg::Matrix<3, 3>& defgr, const Core::LinAlg::Matrix<6, 6>& cmatEul,
        Core::LinAlg::Matrix<6, 6>* cmatLag);

    //! Returns the 4th order identity tensor in tensor notation
    virtual double idev(int i, int j, int k, int l);

    //! Returns the residual of the local Newton step
    virtual Core::LinAlg::Matrix<2, 1> compute_local_newton_residual(
        Core::LinAlg::Matrix<2, 1> lambda_s, double xi_s, LoadingData loading, Material mat_data);

    //! Returns the system matrix of the local Newton step
    virtual Core::LinAlg::Matrix<2, 2> compute_local_newton_jacobian(
        Core::LinAlg::Matrix<2, 1> lambda_s, double xi_s, LoadingData loading, Material mat_data);

    //! Returns the loading in the local Newton step, without historical data.
    virtual LoadingData compute_local_newton_loading(
        double xi_s, double log_strain_vol, double log_strain_dev_norm, Material mat_data);

    //! indicator if #Initialize routine has been called
    bool isinit_;

    double strainenergy_;
  };  // class SuperElasticSMA

}  // namespace Mat


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
