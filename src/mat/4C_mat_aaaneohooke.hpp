/*----------------------------------------------------------------------*/
/*! \file
\brief
This file contains the routines required for aneurysmatic artery wall following
Raghavan and Vorp [2000]

The material is a special case of a generalised pover law neo-Hookean material

the input line should read
  MAT 1 MAT_Struct_AAANeoHooke YOUNG 1.044E7 BETA 188.1E5 DENS 1.0

\level 3


*----------------------------------------------------------------------*/
#ifndef FOUR_C_MAT_AAANEOHOOKE_HPP
#define FOUR_C_MAT_AAANEOHOOKE_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

FOUR_C_NAMESPACE_OPEN


namespace MAT
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for aneurysm wall material
    class AAAneohooke : public CORE::MAT::PAR::Parameter
    {
     public:
      /// standard constructor
      AAAneohooke(Teuchos::RCP<CORE::MAT::PAR::Material> matdata);

      /// create material instance of matching type with my parameters
      Teuchos::RCP<CORE::MAT::Material> create_material() override;

      // !brief enum for mapping between material parameter and entry in the matparams_ vector
      enum Matparamnames
      {
        young,
        nue,
        beta,
        density,
        first = young,
        last = density
      };

    };  // class AAAneohooke

  }  // namespace PAR

  class AAAneohookeType : public CORE::COMM::ParObjectType
  {
   public:
    std::string Name() const override { return "AAAneohookeType"; }

    static AAAneohookeType& Instance() { return instance_; };

    CORE::COMM::ParObject* Create(const std::vector<char>& data) override;

   private:
    static AAAneohookeType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// aneurysm wall material according to Raghavan and Vorp [2000]
  class AAAneohooke : public So3Material
  {
   public:
    // empty constructor
    AAAneohooke();

    // constructor with given material parameters
    AAAneohooke(MAT::PAR::AAAneohooke* params);

    /// check if element kinematics and material kinematics are compatible
    void ValidKinematics(INPAR::STR::KinemType kinem) override
    {
      if (!(kinem == INPAR::STR::KinemType::nonlinearTotLag))
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int UniqueParObjectId() const override
    {
      return AAAneohookeType::Instance().UniqueParObjectId();
    }

    /*!
      \brief Pack this class so it can be communicated

      Resizes the vector data and stores all information of a class in it.
      The first information to be stored in data has to be the
      unique parobject id delivered by UniqueParObjectId() which will then
      identify the exact class on the receiving processor.

      \param data (in/out): char vector to store class information
    */
    void Pack(CORE::COMM::PackBuffer& data) const override;

    /*!
      \brief Unpack data from a char vector into this class

      The vector data contains all information to rebuild the
      exact copy of an instance of a class on a different processor.
      The first entry in data has to be an integer which is the unique
      parobject id defined at the top of this file and delivered by
      UniqueParObjectId().

      \param data (in) : vector storing all data to be unpacked into this
      instance.
    */
    void Unpack(const std::vector<char>& data) override;

    //@}

    /// material mass density
    // virtual double Density() const { return params_->GetDensity(); }
    double Density() const override { return params_->GetParameter(params_->density, -1); }

    /// shear modulus
    // double shear_mod() const { return 0.5*params_->GetYoungs(-1)/(1.0+params_->GetNue()); }
    double shear_mod() const
    {
      return 0.5 * params_->GetParameter(params_->young, -1) /
             (1.0 + params_->GetParameter(params_->nue, -1));
    }

    // material type
    CORE::Materials::MaterialType MaterialType() const override
    {
      return CORE::Materials::m_aaaneohooke;
    }

    /// return copy of this material object
    Teuchos::RCP<CORE::MAT::Material> Clone() const override
    {
      return Teuchos::rcp(new AAAneohooke(*this));
    }

    // THE material routine
    void Evaluate(const CORE::LINALG::Matrix<3, 3>* defgrd,
        const CORE::LINALG::Matrix<6, 1>* glstrain, Teuchos::ParameterList& params,
        CORE::LINALG::Matrix<6, 1>* stress, CORE::LINALG::Matrix<6, 6>* cmat, const int gp,
        const int eleGID) override;

    /// evaluate strain energy function
    void StrainEnergy(const CORE::LINALG::Matrix<6, 1>& glstrain, double& psi, const int gp,
        const int eleGID) override;

    /// Return quick accessible material parameter data
    CORE::MAT::PAR::Parameter* Parameter() const override { return params_; }

    /// Return names of visualization data
    void VisNames(std::map<std::string, int>& names) override;

    /// Return visualization data
    bool VisData(
        const std::string& name, std::vector<double>& data, int numgp, int eleGID) override;

   private:
    /// my material parameters
    MAT::PAR::AAAneohooke* params_;
  };
}  // namespace MAT

FOUR_C_NAMESPACE_CLOSE

#endif