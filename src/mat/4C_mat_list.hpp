// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_LIST_HPP
#define FOUR_C_MAT_LIST_HPP



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
    class MatList : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      MatList(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{

      /// provide ids of the individual materials
      const std::vector<int>* mat_ids() const { return &matids_; }

      /// provide access to material by its ID
      std::shared_ptr<Core::Mat::Material> material_by_id(const int id) const;

      std::map<int, std::shared_ptr<Core::Mat::Material>>* material_map_write() { return &mat_; }

      /// length of material list
      const int nummat_;

      /// the list of material IDs
      const std::vector<int> matids_;

      /// flag for individual materials or only one at global scope
      bool local_;

     private:
      /// map to materials (only used for local_==true)
      std::map<int, std::shared_ptr<Core::Mat::Material>> mat_;

      //@}

    };  // class MatList

  }  // namespace PAR

  class MatListType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "MatListType"; }

    static MatListType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MatListType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a list of materials
  class MatList : public Core::Mat::Material
  {
   public:
    /// construct empty material object
    MatList();

    /// construct the material object given material parameters
    explicit MatList(Mat::PAR::MatList* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return MatListType::instance().unique_par_object_id();
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
      return Core::Materials::m_matlist;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MatList>(*this);
    }

    /// number of materials
    int num_mat() const { return params_->nummat_; }

    /// material ID by Index
    int mat_id(const unsigned index) const;

    /// provide access to material by its ID
    virtual std::shared_ptr<Core::Mat::Material> material_by_id(const int id) const;

    /// Return quick accessible material parameter data
    Mat::PAR::MatList* parameter() const override { return params_; }

   protected:
    /// return pointer to the materials map, which has read-only access.
    const std::map<int, std::shared_ptr<Core::Mat::Material>>* material_map_read() const
    {
      return &mat_;
    }

    /// return pointer to the materials map, which has read and write access.
    std::map<int, std::shared_ptr<Core::Mat::Material>>* material_map_write() { return &mat_; }

   private:
    /// setup of material map
    void setup_mat_map();

    /// clear everything
    void clear();

    /// my material parameters
    Mat::PAR::MatList* params_;

    /// map to materials
    std::map<int, std::shared_ptr<Core::Mat::Material>> mat_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
