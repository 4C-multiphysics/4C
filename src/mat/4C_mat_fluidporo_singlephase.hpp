// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_FLUIDPORO_SINGLEPHASE_HPP
#define FOUR_C_MAT_FLUIDPORO_SINGLEPHASE_HPP



/*---------------------------------------------------------------------*
 | headers                                                              |
 *---------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_mat_fluidporo_relpermeability_law.hpp"
#include "4C_mat_fluidporo_viscosity_law.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_mat_poro_density_law.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


/*---------------------------------------------------------------------*
 *---------------------------------------------------------------------*/
namespace Mat
{
  namespace PAR
  {
    // forward declaration
    class FluidPoroPhaseDof;

    /*----------------------------------------------------------------------*/
    /// material parameters for a single phase of porous multiphase fluid
    ///
    /// This object exists only once for each read fluid.
    class FluidPoroSinglePhase : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      FluidPoroSinglePhase(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// initialize
      void initialize();

      /// @name material parameters
      //@{

      /// density
      const double density_;
      /// type of degree of freedom
      FluidPoroPhaseDof* phasedof_;
      /// density law
      PoroDensityLaw* densitylaw_;
      /// permeability law
      FluidPoroRelPermeabilityLaw* relpermeabilitylaw_;
      /// viscosity law
      FluidPoroViscosityLaw* viscositylaw_;

      //@}

     private:
      bool isinit_;

    };  // class FluidPoroSinglePhase

    /*----------------------------------------------------------------------*/
    /// material parameters for a single volfrac of porous multiphase fluid
    ///
    /// This object exists only once for each read fluid.
    class FluidPoroSingleVolFrac : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      FluidPoroSingleVolFrac(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// initialize
      void initialize();

      /// @name material parameters
      //@{

      /// density
      const double density_;
      /// bulk diffusivity
      const double diffusivity_;
      /// do we have additional scalar dependent flux
      const bool scalardependentflux_;
      /// number of scalars
      const int numscal_;
      /// their diffusivities
      const std::vector<double> scalardiffs_;
      /// constant omega_half for receptor kinetic law
      const std::vector<double> omega_half_;

      //@}

     private:
      bool isinit_;

    };  // class FluidPoroSingleVolFrac

    /*----------------------------------------------------------------------*/
    /// material parameters for a single volfrac pressure of porous multiphase fluid
    ///
    /// This object exists only once for each read fluid.
    class FluidPoroVolFracPressure : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      FluidPoroVolFracPressure(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// initialize
      void initialize();

      /// @name material parameters
      //@{

      /// permeability
      const double permeability_;

      /// minimum volume fraction
      const double min_volfrac_;

      /// viscosity law
      FluidPoroViscosityLaw* viscositylaw_;

      //@}

     private:
      bool isinit_;

    };  // class FluidPoroVolFracPressure

    /*----------------------------------------------------------------------*/
    /// material parameters for a single volfrac pressure material for vascular units in th lungs
    /// modeled as an additional porous network
    ///
    /// This object exists only once for each read fluid.
    class FluidPoroVolFracPressureBloodLung : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      FluidPoroVolFracPressureBloodLung(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// initialize
      void initialize();

      /// @name material parameters
      //@{

      /// density
      const double density_{};

      /// permeability
      const double permeability_{};

      /// viscosity law
      FluidPoroViscosityLaw* viscositylaw_;

      /// initial volume fraction
      const double initial_volfrac_{};

      /// scaling parameter for deformation dependency
      const double scaling_parameter_deformation_{};

      /// scaling parameter for pressure dependency
      const double scaling_parameter_pressure_{};


      //@}

     private:
      bool isinit_;

    };  // class FluidPoroVolFracPressure

    namespace PoroFluidPressureBased
    {
      enum class ClosingRelation
      {
        undefined,
        evolutionequation_homogenized_vasculature_tumor,  //! closing the system of equations
                                                          //! through an evolution equation for
                                                          //! the vasculatur in the tumor
                                                          //! framework
        evolutionequation_blood_lung,  //! closing the system of equations through an evolution
                                       //! equation for the vasculature in the lungs
      };
    }


  }  // namespace PAR

  /*----------------------------------------------------------------------*
   | instance access method                                   vuong 08/16 |
   *----------------------------------------------------------------------*/
  class FluidPoroSinglePhaseType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroSinglePhaseType"; }

    static FluidPoroSinglePhaseType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroSinglePhaseType instance_;
  };

  /*----------------------------------------------------------------------*
   | instance access method                                   vuong 08/16 |
   *----------------------------------------------------------------------*/
  class FluidPoroSingleVolFracType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroSingleVolFracType"; }

    static FluidPoroSingleVolFracType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroSingleVolFracType instance_;
  };

  /*----------------------------------------------------------------------*
   | instance access method                              kremheller 02/18 |
   *----------------------------------------------------------------------*/
  class FluidPoroVolFracPressureType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroVolFracPressureType"; }

    static FluidPoroVolFracPressureType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroVolFracPressureType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Base class for a single porous fluid phase within multiphase porous flow
  ///
  /// This object exists (several times) at every element
  class FluidPoroSinglePhaseBase : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    FluidPoroSinglePhaseBase() {};

    /// initialize
    virtual void initialize() = 0;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a single porous fluid phase within multiphase porous flow
  ///
  /// This object exists (several times) at every element
  class FluidPoroSinglePhase : public FluidPoroSinglePhaseBase
  {
   public:
    /// construct empty material object
    FluidPoroSinglePhase();

    /// construct the material object given material parameters
    explicit FluidPoroSinglePhase(Mat::PAR::FluidPoroSinglePhase* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return FluidPoroSinglePhaseType::instance().unique_par_object_id();
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

    /// initialize
    void initialize() override;

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo_singlephase;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoroSinglePhase>(*this);
    }

    /// return density
    double density() const override { return params_->density_; }

    /// return relative permeability
    double rel_permeability(const double saturation) const
    {
      return params_->relpermeabilitylaw_->get_rel_permeability(saturation);
    }

    // check for constant relative permeability
    bool has_constant_rel_permeability() const
    {
      return params_->relpermeabilitylaw_->has_constant_rel_permeability();
    }

    /// return derivative of relative permeabilty w.r.t. saturation
    double evaluate_deriv_of_rel_permeability_wrt_saturation(const double saturation) const
    {
      return params_->relpermeabilitylaw_->get_deriv_of_rel_permeability_wrt_saturation(saturation);
    }

    // check for constant viscosity
    bool has_constant_viscosity() const { return params_->viscositylaw_->has_constant_viscosity(); }

    /// return viscosity
    double viscosity(const double abspressgrad) const
    {
      return params_->viscositylaw_->get_viscosity(abspressgrad);
    }

    /// return derivative of viscosity w.r.t. to absolute value of pressure gradient
    double viscosity_deriv(const double abspressgrad) const
    {
      return params_->viscositylaw_->get_deriv_of_viscosity_wrt_abs_press_grad(abspressgrad);
    }

    /// return inverse bulk modulus (compressibility)
    double inv_bulkmodulus() const { return params_->densitylaw_->inv_bulkmodulus(); }

    /// return type of degree of freedom
    Core::Materials::MaterialType poro_dof_type() const;

    /// return type of phase law
    Core::Materials::MaterialType poro_phase_law_type() const;

    /// mark dofs associated with this phase in a given row (=numphase) in a matrix
    void fill_do_f_matrix(Core::LinAlg::SerialDenseMatrix& dofmat, int numphase) const;

    /// evaluate saturation of the phase
    double evaluate_saturation(
        int phasenum, const std::vector<double>& state, const std::vector<double>& pressure) const;

    /// evaluate the generalized(!) pressure of this phase
    double evaluate_gen_pressure(int phasenum, const std::vector<double>& state) const;

    //! evaluate derivative of saturation with respect to pressure
    double evaluate_deriv_of_saturation_wrt_pressure(
        int phasenum, int doftoderive, const std::vector<double>& pressure) const;

    //! evaluate 2nd derivative of saturation with respect to pressure
    double evaluate_second_deriv_of_saturation_wrt_pressure(int phasenum, int firstdoftoderive,
        int seconddoftoderive, const std::vector<double>& pressure) const;

    //! evaluate derivative of degree of freedom with respect to pressure
    double evaluate_deriv_of_dof_wrt_pressure(
        int phasenum, int doftoderive, const std::vector<double>& state) const;

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// my material parameters
    Mat::PAR::FluidPoroSinglePhase* params_;
  };


  /*----------------------------------------------------------------------*/
  /// Wrapper for a single volfrac within multiphase porous flow
  ///
  /// This object exists (several times) at every element
  class FluidPoroSingleVolFrac : public FluidPoroSinglePhaseBase
  {
   public:
    /// construct empty material object
    FluidPoroSingleVolFrac();

    /// construct the material object given material parameters
    explicit FluidPoroSingleVolFrac(Mat::PAR::FluidPoroSingleVolFrac* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return FluidPoroSingleVolFracType::instance().unique_par_object_id();
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

    /// initialize
    void initialize() override;

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo_singlevolfrac;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoroSingleVolFrac>(*this);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// return density
    double density() const override { return params_->density_; }

    /// return diffusivity
    double diffusivity() const { return params_->diffusivity_; }

    /// return number of scalars
    int num_scal() const { return params_->numscal_; }

    /// return scalardependentflux_
    bool has_add_scalar_dependent_flux() const { return params_->scalardependentflux_; }

    /// return diffusivities for scalar-dependent flux
    std::vector<double> scalar_diffs() const { return params_->scalardiffs_; }

    /// return omega_half for scalar-dependent flux
    std::vector<double> omega_half() const { return params_->omega_half_; }

   private:
    /// my material parameters
    Mat::PAR::FluidPoroSingleVolFrac* params_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a single volfrac pressure within multiphase porous flow
  ///
  /// This object exists (several times) at every element
  class FluidPoroVolFracPressure : public FluidPoroSinglePhaseBase
  {
   public:
    /// construct empty material object
    FluidPoroVolFracPressure();

    /// construct the material object given material parameters
    explicit FluidPoroVolFracPressure(Mat::PAR::FluidPoroVolFracPressure* params);

    //! @name Packing and Unpacking

    /*!
     \brief Return unique ParObject id

     every class implementing ParObject needs a unique id defined at the
     top of parobject.H (this file) and should return it in this method.
     */
    int unique_par_object_id() const override
    {
      return FluidPoroVolFracPressureType::instance().unique_par_object_id();
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

    /// initialize
    void initialize() override;

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo_volfracpressure;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoroVolFracPressure>(*this);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    /// return permeability
    double permeability() const { return params_->permeability_; }

    /// return minimum volume fraction
    double min_vol_frac() const { return params_->min_volfrac_; }

    // check for constant viscosity
    bool has_constant_viscosity() const { return params_->viscositylaw_->has_constant_viscosity(); }

    /// return viscosity
    double viscosity(const double abspressgrad) const
    {
      return params_->viscositylaw_->get_viscosity(abspressgrad);
    }

    /// return derivative of viscosity w.r.t. to absolute value of pressure gradient
    double viscosity_deriv(const double abspressgrad) const
    {
      return params_->viscositylaw_->get_deriv_of_viscosity_wrt_abs_press_grad(abspressgrad);
    }

   private:
    /// my material parameters
    Mat::PAR::FluidPoroVolFracPressure* params_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a single volfrac pressure material for modeling vascular units in the lungs as an
  /// additional porous network within multiphase porous flow
  ///
  /// This object exists (only once) at every element
  /*----------------------------------------------------------------------*
   *----------------------------------------------------------------------*/
  class FluidPoroVolFracPressureBloodLungType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "FluidPoroVolFracPressureBloodLungType"; }

    static FluidPoroVolFracPressureBloodLungType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static FluidPoroVolFracPressureBloodLungType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a single volfrac pressure material for vascular units in the lungs within
  /// multiphase porous flow
  ///
  /// This object exists (only once) at every element
  class FluidPoroVolFracPressureBloodLung : public FluidPoroSinglePhaseBase
  {
   public:
    //! construct empty material object
    FluidPoroVolFracPressureBloodLung();

    //! construct the material object given material parameters
    explicit FluidPoroVolFracPressureBloodLung(Mat::PAR::FluidPoroVolFracPressureBloodLung* params);


    int unique_par_object_id() const override
    {
      return FluidPoroVolFracPressureBloodLungType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    void initialize() override;

    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_fluidporo_volfrac_pressure_blood_lung;
    }

    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<FluidPoroVolFracPressureBloodLung>(*this);
    }

    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    double density() const override { return params_->density_; }

    //! return permeability
    double permeability() const { return params_->permeability_; }

    //! check for constant viscosity
    bool has_constant_viscosity() const { return params_->viscositylaw_->has_constant_viscosity(); }

    //! return viscosity
    double viscosity(const double abspressgrad) const
    {
      return params_->viscositylaw_->get_viscosity(abspressgrad);
    }

    //! return derivative of viscosity w.r.t. to absolute value of pressure gradient
    double viscosity_deriv(const double abspressgrad) const
    {
      return params_->viscositylaw_->get_deriv_of_viscosity_wrt_abs_press_grad(abspressgrad);
    }

    //! return initial volume fraction (usually end-expiratory state)
    double initial_volfrac() const { return params_->initial_volfrac_; }

    //! return scaling parameter for deformation dependence of the volume fraction
    double scaling_parameter_deformation() const { return params_->scaling_parameter_deformation_; }

    //! return scaling parameter for pressure dependence of the volume fraction
    double scaling_parameter_pressure() const { return params_->scaling_parameter_pressure_; }

   private:
    //! my material parameters
    Mat::PAR::FluidPoroVolFracPressureBloodLung* params_;
  };

}  // namespace Mat


FOUR_C_NAMESPACE_CLOSE

#endif
