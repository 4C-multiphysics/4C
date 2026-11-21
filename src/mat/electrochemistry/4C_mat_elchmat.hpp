// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELCHMAT_HPP
#define FOUR_C_MAT_ELCHMAT_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for list of materials
    class ElchMat : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      ElchMat(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{

      /// provide ids of the individual phase
      const std::vector<int>& phase_ids() const { return phaseids_; }

      /// provide access to phases by its ID
      std::shared_ptr<Core::Mat::Material> phase_by_id(const int id) const
      {
        if (not local_)
        {
          std::map<int, std::shared_ptr<Core::Mat::Material>>::const_iterator m = mat_.find(id);

          if (m == mat_.end())
          {
            FOUR_C_THROW("Material {} could not be found", id);
            return nullptr;
          }
          else
            return m->second;
        }
        else
          FOUR_C_THROW("This is not allowed");

        return nullptr;
      }

      /// number of degrees of freedom
      const int numdof_;

      /// number of scalar
      const int numscal_;

      /// length of phase list
      const int numphase_;

      /// the list of material IDs
      const std::vector<int> phaseids_;

      /// flag for individual materials or only one at global scope
      bool local_;

     private:
      /// map to materials (only used for local_==true)
      std::map<int, std::shared_ptr<Core::Mat::Material>> mat_;

      //@}

    };  // class MatList

  }  // namespace PAR

  class ElchMatType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "ElchMatType"; }

    static ElchMatType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static ElchMatType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a list of materials
  class ElchMat : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    ElchMat();

    /// construct the material object given material parameters
    explicit ElchMat(Mat::PAR::ElchMat* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return ElchMatType::instance().unique_par_object_id();
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
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_elchmat;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<ElchMat>(*this);
    }

    // return number of Dof used for this problem type
    int num_dof() const { return params_->numdof_; }

    // return number of scalars used for this problem type
    int num_scal() const { return params_->numscal_; }

    /// number of materials
    int num_phase() const { return params_->numphase_; }

    /// material ID by Index
    int phase_id(const unsigned index) const
    {
      if ((int)index < params_->numphase_)
        return params_->phaseids_.at(index);
      else
      {
        FOUR_C_THROW("Index too large");
        return -1;
      }
    }

    /// provide access to material by its ID
    std::shared_ptr<Core::Mat::Material> phase_by_id(const int id) const
    {
      if (params_->local_)
      {
        std::map<int, std::shared_ptr<Core::Mat::Material>>::const_iterator m = mat_.find(id);
        if (m == mat_.end())
        {
          FOUR_C_THROW("Material {} could not be found", id);
          return nullptr;
        }
        else
          return m->second;
      }
      else  // material is global (stored in material parameters)
        return params_->phase_by_id(id);
    }

    /// Return quick accessible material parameter data
    Core::Mat::PAR::Parameter* parameter() const override { return params_; }

   private:
    /// setup of material map
    void setup_mat_map();

    /// clear everything
    void clear();

    /// my material parameters
    Mat::PAR::ElchMat* params_;

    /// map to materials
    std::map<int, std::shared_ptr<Core::Mat::Material>> mat_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
