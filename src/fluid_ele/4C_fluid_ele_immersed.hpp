// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_fluid_ele_immersed_base.hpp"

#ifndef FOUR_C_FLUID_ELE_IMMERSED_HPP
#define FOUR_C_FLUID_ELE_IMMERSED_HPP

FOUR_C_NAMESPACE_OPEN

namespace Discret
{
  namespace Elements
  {
    class FluidTypeImmersed : public FluidTypeImmersedBase
    {
     public:
      std::string name() const override { return "FluidTypeImmersed"; }

      static FluidTypeImmersed& instance();

      Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

      std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

      void setup_element_definition(
          std::map<std::string, std::map<std::string, Core::IO::InputSpec>>& definitions) override;


     private:
      static FluidTypeImmersed instance_;

    };  // class FluidTypeImmersed

    class FluidImmersed : public FluidImmersedBase
    {
     public:
      //@}
      //! @name constructors and destructors and related methods

      /*!
      \brief standard constructor
      */
      FluidImmersed(int id,  ///< A unique global id
          int owner          ///< ???
      );

      /*!
      \brief Copy Constructor

      Makes a deep copy of a Element

      */
      FluidImmersed(const FluidImmersed& old);

      /*!
      \brief Deep copy this instance of fluid and return pointer to the copy

      The clone() method is used from the virtual base class Element in cases
      where the type of the derived class is unknown and a copy-ctor is needed

      */
      Core::Elements::Element* clone() const override;

      /*!
      \brief Return unique ParObject id

      every class implementing ParObject needs a unique id defined at the
      top of this file.
      */
      int unique_par_object_id() const override
      {
        return FluidTypeImmersed::instance().unique_par_object_id();
      }

      /*!
      \brief Each element whose nodes are all covered by an immersed dis. are set IsImmersed
      */
      void set_is_immersed(int isimmersed) override { is_immersed_ = isimmersed; };

      /*!
      \brief Each element which has nodes covered by the immersed dis. but at least one node that is
      not covered is set BoundaryIsImmersed

       An element may also be set BoundaryIsImmersed, if an integration point of the structural
      surface lies in this element.

      */
      void set_boundary_is_immersed(int IsBoundaryImmersed) override
      {
        is_immersed_bdry_ = IsBoundaryImmersed;
      };

      /*!
      \brief Each element, which is either Immersed or BoundaryImmersed is also set
      has_projected_dirichlet
      */
      void set_has_projected_dirichlet(int has_projected_dirichletvalues) override
      {
        has_projected_dirichletvalues_ = has_projected_dirichletvalues;
      };

      /*!
      \brief set if divergence needs to be projected to an integration point
      */
      void set_int_point_has_projected_divergence(
          int gp, int intpoint_has_projected_divergence) override
      {
        intpoint_has_projected_divergence_->at(gp) = intpoint_has_projected_divergence;
      };

      /*!
      \brief store the projected divergence
      */
      void store_projected_int_point_divergence(
          int gp, double projected_intpoint_divergence) override
      {
        stored_projected_intpoint_divergence_->at(gp) = projected_intpoint_divergence;
      };

      /*!
      \brief returns true if element was set IsImmersed
      */
      int is_immersed() override { return is_immersed_; };

      /*!
      \brief returns true if element was set IsBundaryImmersed
      */
      int is_boundary_immersed() override { return is_immersed_bdry_; };

      /*!
      \brief returns true if element needs to get projected Dirichlet values
      */
      int has_projected_dirichlet() override { return has_projected_dirichletvalues_; };

      /*!
      \brief returns true if element needs to get projected divergence at integration point

      */
      int int_point_has_projected_divergence(int gp) override
      {
        return intpoint_has_projected_divergence_->at(gp);
      };

      /*!
      \brief returns projected divergence at integration point
      */
      double projected_int_point_divergence(int gp) override
      {
        return stored_projected_intpoint_divergence_->at(gp);
      };

      /*!
      \brief returns rcp to vector containing gps with projected divergence
      */
      std::shared_ptr<std::vector<int>> get_rcp_int_point_has_projected_divergence() override
      {
        return intpoint_has_projected_divergence_;
      };

      /*!
      \brief returns rcp to vector containing projected divergence values
      */
      std::shared_ptr<std::vector<double>> get_rcp_projected_int_point_divergence() override
      {
        return stored_projected_intpoint_divergence_;
      };

      /*!
      \brief construct rcp to vector for divergence projection handling
      */
      void construct_element_rcp(int size) override
      {
        if (intpoint_has_projected_divergence_ == nullptr)
          intpoint_has_projected_divergence_ = std::make_shared<std::vector<int>>(size, 0);
        else
          intpoint_has_projected_divergence_->resize(size, 0);

        if (stored_projected_intpoint_divergence_ == nullptr)
          stored_projected_intpoint_divergence_ = std::make_shared<std::vector<double>>(size, 0.0);
        else
          stored_projected_intpoint_divergence_->resize(size, 0.0);
      };

      /*!
      \brief Clean up the element rcp
      */
      void destroy_element_rcp() override
      {
        intpoint_has_projected_divergence_->clear();
        stored_projected_intpoint_divergence_->clear();

        if (intpoint_has_projected_divergence_->size() > 0)
          FOUR_C_THROW("intpoint_has_projected_divergence_ not cleared properly");

        if (stored_projected_intpoint_divergence_->size() > 0)
          FOUR_C_THROW("stored_projected_intpoint_divergence_ not cleared properly");
      };

      void vis_names(std::map<std::string, int>& names) override
      {
        names["IsBoundaryImmersed"] = 1;
        names["IsImmersed"] = 1;
      }


      /*!
      \brief Query data to be visualized using BINIO of a given name

      This method is to be overloaded by a derived method.
      The derived method is supposed to call this base method to visualize the owner of
      the element.
      If the derived method recognizes a supported data name, it shall fill it
      with corresponding data.
      If it does NOT recognizes the name, it shall do nothing.

      \warning The method must not change size of variable data

      \param name (in):   Name of data that is currently processed for visualization
      \param data (out):  data to be filled by element if it recognizes the name
      */
      bool vis_data(const std::string& name, std::vector<double>& data) override
      {
        if (name == "Owner")
        {
          if ((int)data.size() < 1) FOUR_C_THROW("Size mismatch");
          data[0] = owner();
          return true;
        }
        if (name == "IsImmersed")
        {
          if ((int)data.size() < 1) FOUR_C_THROW("Size mismatch");
          data[0] = is_immersed();
          return true;
        }
        if (name == "IsBoundaryImmersed")
        {
          if ((int)data.size() < 1) FOUR_C_THROW("Size mismatch");
          data[0] = is_boundary_immersed();
          return true;
        }
        if (name == "EleGId")
        {
          if ((int)data.size() < 1) FOUR_C_THROW("Size mismatch");
          data[0] = id();
          return true;
        }
        return false;
      }

      /*!
      \brief Pack this class so it can be communicated

      \ref pack and \ref unpack are used to communicate this element

      */
      void pack(Core::Communication::PackBuffer& data) const override;

      /*!
      \brief Unpack data from a char vector into this class

      \ref pack and \ref unpack are used to communicate this element
      */
      void unpack(Core::Communication::UnpackBuffer& buffer) override;



     private:
      int is_immersed_;       //!< true if all nodes of this element are covered by immersed
                              //!< discretization
      int is_immersed_bdry_;  //!< true if boundary intersects this element
      int has_projected_dirichletvalues_;  //!< true if Dirichlet values need to be projected to
                                           //!< this element
      std::shared_ptr<std::vector<int>>
          intpoint_has_projected_divergence_;  //!< 1000 max number of gp
      std::shared_ptr<std::vector<double>>
          stored_projected_intpoint_divergence_;  //!< 1000 max number of gp
    };
  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
