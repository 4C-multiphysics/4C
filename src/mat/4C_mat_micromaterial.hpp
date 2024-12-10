// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MICROMATERIAL_HPP
#define FOUR_C_MAT_MICROMATERIAL_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN



namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for micro material
    class MicroMaterial : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      MicroMaterial(const Core::Mat::PAR::Parameter::Data& matdata);

      //! action parameters recognized by ale3
      enum class runtime_output_option
      {
        none,  ///< no output of micromodel
        all,   ///< output micromodel of every gausspoint
        gp1    ///< output micromodel of the first gausspoint
      };
      /// @name material parameters
      //@{

      /// inputfile for microstructure
      const std::string microfile_;
      /// Number of microscale discretization
      const int microdisnum_;
      ///
      const double initvol_;

      //// runtime output option
      runtime_output_option RuntimeOutputOption;
      //! @}

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class MicroMaterial

  }  // namespace PAR

  class MicroMaterialType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "MicroMaterialType"; }

    static MicroMaterialType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MicroMaterialType instance_;
  };

  /*----------------------------------------------------------------------*/
  // forward
  class MicroMaterialGP;

  /*----------------------------------------------------------------------*/
  /// class for handling of micro-macro transitions
  class MicroMaterial : public So3Material
  {
   public:
    /// construct empty material object
    MicroMaterial();

    /// construct the material object given material parameters
    explicit MicroMaterial(Mat::PAR::MicroMaterial* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return MicroMaterialType::instance().unique_par_object_id();
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

    /// material type
    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_struct_multiscale;
    }

    /// check if element kinematics and material kinematics are compatible
    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (!(kinem == Inpar::Solid::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    /// return copy of this material object
    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MicroMaterial>(*this);
    }

    /// evaluate micro material on a processor with macro scale
    void evaluate(const Core::LinAlg::Matrix<3, 3>* defgrd,
        const Core::LinAlg::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        Core::LinAlg::Matrix<6, 1>* stress, Core::LinAlg::Matrix<6, 6>* cmat, int gp,
        int eleGID) override;

    /// evaluate micro material on a processor which only knows about the micro scale (supporting
    /// proc)
    void evaluate(Core::LinAlg::Matrix<3, 3>* defgrd, Core::LinAlg::Matrix<6, 6>* cmat,
        Core::LinAlg::Matrix<6, 1>* stress, const int gp, const int ele_ID, const int microdisnum,
        double V0, bool eleowner);

    double density() const override;

    /// Calculate stresses and strains on the micro-scale
    void prepare_output();

    /// Calculate stresses and strains on the micro-scale
    void prepare_runtime_output() const;

    /// Write output on micro-scale
    void runtime_output_step_state(std::pair<double, int> output_time_and_step) const;

    /// Write output on micro-scale
    void output_step_state();

    /// Update state vectors
    void update() override;

    /// Post setup routine which will be called after the end of the setup
    virtual void post_setup();

    /// Write restart on micro-scale
    void write_restart();

    /// Read restart of micro scale on a processor with macro scale
    void read_restart(const int gp, const int eleID, const bool eleowner);

    /// restart micro material on a processor which only knows about the micro scale (supporting
    /// proc)
    void read_restart(
        const int gp, const int eleID, const bool eleowner, int microdisnum, double V0);

    /// @name Access parameters
    //@{
    std::string micro_input_file_name() const { return params_->microfile_; }
    int micro_dis_num() const { return params_->microdisnum_; }
    double init_vol() const { return params_->initvol_; }
    PAR::MicroMaterial::runtime_output_option runtime_output_option() const
    {
      return params_->RuntimeOutputOption;
    }

    //@}

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    std::map<int, std::shared_ptr<MicroMaterialGP>> matgp_;

    double density_;

    /// my material parameters
    Mat::PAR::MicroMaterial* params_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
