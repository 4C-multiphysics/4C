// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_LIST_REACTIONS_HPP
#define FOUR_C_MAT_LIST_REACTIONS_HPP



#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_list.hpp"
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
    class MatListReactions : public virtual MatList
    {
     public:
      /// standard constructor
      MatListReactions(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create material instance of matching type with my parameters
      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{

      /// provide ids of the individual reaction materials
      const std::vector<int>* reac_ids() const { return &reacids_; }

      /// length of reaction list
      const int numreac_;

      /// the list of reaction IDs
      const std::vector<int> reacids_;

      //@}

    };  // class MatListReactions

  }  // namespace PAR

  class MatListReactionsType : public Core::Communication::ParObjectType
  {
   public:
    std::string name() const override { return "MatListReactionsType"; }

    static MatListReactionsType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static MatListReactionsType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for a list of materials
  class MatListReactions : public virtual MatList
  {
   public:
    /// construct empty material object
    MatListReactions();

    /// construct the material object given material parameters
    explicit MatListReactions(Mat::PAR::MatListReactions* params);

    //! @name Packing and Unpacking

    /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of parobject.H (this file) and should return it in this method.
    */
    int unique_par_object_id() const override
    {
      return MatListReactionsType::instance().unique_par_object_id();
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
    virtual void initialize();

    /// material type
    Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_matlist_reactions;
    }

    /// return copy of this material object
    std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<MatListReactions>(*this);
    }

    /// number of reactions
    int num_reac() const { return paramsreac_->numreac_; }

    /// reaction ID by Index
    int reac_id(const unsigned index) const;

    /// Return quick accessible material parameter data
    Mat::PAR::MatListReactions* parameter() const override { return paramsreac_; }

    /// advanced reaction terms
    double calc_rea_body_force_term(const int k, const std::vector<double>& phinp,
        const double* gpcoord,  //!< current Gauss-point coordinates
        const double time, const double scale = 1.0) const;

    /// calculate advanced reaction term derivatives
    void calc_rea_body_force_deriv_matrix(const int k, std::vector<double>& derivs,
        const std::vector<double>& phinp,
        const double* gpcoord,  //!< current Gauss-point coordinates
        const double time, const double scale = 1.0) const;

    /// advanced reaction terms
    double calc_rea_body_force_term(const int k, const std::vector<double>& phinp,
        const std::vector<std::pair<std::string, double>>& constants,
        const double* gpcoord,  //!< current Gauss-point coordinates
        const double time, const double scale = 1.0) const;

    /// calculate advanced reaction term derivatives
    void calc_rea_body_force_deriv_matrix(const int k, std::vector<double>& derivs,
        const std::vector<double>& phinp,
        const std::vector<std::pair<std::string, double>>& constants,
        const double* gpcoord,  //!< current Gauss-point coordinates
        const double time, const double scale = 1.0) const;

    // add additional variables to the reaction (only for by-function coupling)
    void add_additional_variables(
        const int k, const std::vector<std::pair<std::string, double>>& variables) const;

    /// calculate advanced reaction term derivatives
    void calc_rea_body_force_deriv_matrix_add_variables(const int k, std::vector<double>& derivs,
        const std::vector<double>& phinp,
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants, const double* gpcoord,
        const double time, const double scale = 1.0) const;


   protected:
    /// setup of material map
    virtual void setup_mat_map();

   private:
    /// clear everything
    void clear();

    /// my material parameters
    Mat::PAR::MatListReactions* paramsreac_;
  };

}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
