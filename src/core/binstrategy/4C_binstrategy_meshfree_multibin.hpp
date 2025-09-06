// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BINSTRATEGY_MESHFREE_MULTIBIN_HPP
#define FOUR_C_BINSTRATEGY_MESHFREE_MULTIBIN_HPP

#include "4C_config.hpp"

#include "4C_binstrategy_meshfree_bin.hpp"
#include "4C_binstrategy_utils.hpp"
#include "4C_fem_general_elementtype.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_linalg_serialdensematrix.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FE::MeshFree
{
  /*--------------------------------------------------------------------------*/
  /*!
   * \brief The element type class of meshfree multibin, creating the same
   *
   *

   */
  /*--------------------------------------------------------------------------*/
  class MeshfreeMultiBinType : public Core::Elements::ElementType
  {
   public:
    //!< name of specific element type
    std::string name() const override { return "MeshfreeMultiBinType"; }

    //!< returning instance of specific element type
    static MeshfreeMultiBinType& instance();

    //!< create parallel object of this element type
    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

    //!< create element of this element type
    std::shared_ptr<Core::Elements::Element> create(const std::string eletype,
        const std::string eledistype, const int id, const int owner) override;

    //!< create element of this element type
    std::shared_ptr<Core::Elements::Element> create(const int id, const int owner) override;

    void nodal_block_information(Core::Elements::Element* dwele, int& numdf, int& dimns) override {}

    Core::LinAlg::SerialDenseMatrix compute_null_space(
        Core::Nodes::Node& node, const double* x0, const int numdof, const int dimnsp) override
    {
      Core::LinAlg::SerialDenseMatrix nullspace;
      FOUR_C_THROW("method ComputeNullSpace not implemented!");
      return nullspace;
    }

   private:
    //!< static instance of this element type (for self-instantiation)
    static MeshfreeMultiBinType instance_;
  };

  /*--------------------------------------------------------------------------*/
  /*!
   * \brief A meshfree multibin holds information about associated fluid and wall
   *        elements which can be added and deleted dynamically
   *
   *

   */
  /*--------------------------------------------------------------------------*/
  class MeshfreeMultiBin : public MeshfreeBin<Core::Elements::Element>
  {
   public:
    /*========================================================================*/
    //! @name Constructors and destructors and related methods
    /*========================================================================*/

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Standard Constructor
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    MeshfreeMultiBin(int id,  //!< (in): A globally unique bin id
        int owner             //!< (in): owner processor of the meshfree bin
    );

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Copy Constructor
     *
     * Makes a deep copy of a meshfree multibin
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    MeshfreeMultiBin(const Core::FE::MeshFree::MeshfreeMultiBin& old);

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Deep copy the derived class and return pointer to it
     *
     * This method is sort of a copy constructor for a class derived from
     * Core::Elements::Element. It allows to copy construct the derived class without
     * knowing what it actually is using the base class Element.
     *
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    Core::Elements::Element* clone() const override;

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Return unique ParObject id
     *
     * Every class implementing ParObject needs a unique id defined at the
     * top of parobject.H
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    int unique_par_object_id() const override
    {
      return MeshfreeMultiBinType::instance().unique_par_object_id();
    };

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Pack this class so it can be communicated
     *
     * \ref pack and \ref unpack are used to communicate this meshfree multibin
     *
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    void pack(Core::Communication::PackBuffer& data) const override;

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Unpack data from a char vector into this class
     *
     * \ref pack and \ref unpack are used to communicate this meshfree multibin
     *
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    // return meshfree bin type instance
    Core::Elements::ElementType& element_type() const override
    {
      return MeshfreeMultiBinType::instance();
    }

    /*!
    \brief Get number of degrees of freedom of a certain node
    */
    int num_dof_per_node(const Core::Nodes::Node& node) const override { return 3; }

    /*------------------------------------------------------------------------*/
    /*!
    \brief Get number of degrees of freedom per element
           (implements pure virtual Core::Elements::Element)
     *
     * Meshfree bins do not have dofs
     *///
    /*------------------------------------------------------------------------*/
    int num_dof_per_element() const override { return 0; }

    /*========================================================================*/
    //! @name Query methods
    /*========================================================================*/

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Get vector of ptrs to associated  elements
     *
     * \return Ptr to pointers to nodes of the element in local nodal ordering.
     *      Returns nullptr if pointers to not exist.
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    inline const std::vector<Core::Elements::Element*>& associated_eles(
        Core::Binstrategy::Utils::BinContentType bin_content)
    {
      return associated_ele_[bin_content];
    }

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Add an associated element to this bin
     *///                                                  (public) ghamm 04/13
    /*------------------------------------------------------------------------*/
    inline virtual void add_associated_ele(Core::Binstrategy::Utils::BinContentType
                                               bin_content,  //!< (in): type of element to be added
        Core::Elements::Element* eleptr  //!< (in): pointer to element to be added
    )
    {
      associated_ele_[bin_content].emplace_back(eleptr);
    }

    /*------------------------------------------------------------------------*/
    /*!
     * \brief Remove all associated elements of all kind from this bin

     *///                                                  (public) ghamm 09/13
    /*------------------------------------------------------------------------*/
    void remove_all_associated_eles();

    /*!
    \brief Evaluate a Neumann boundary condition dummy

    An element derived from this class uses the evaluate_neumann method to receive commands
    and parameters from some control routine in params and evaluates a Neumann boundary condition
    given in condition

    \note This class implements a dummy of this method that prints a warning and
          returns false.

    \param params (in/out)    : ParameterList for communication between control routine
                                and elements
    \param discretization (in): A reference to the underlying discretization
    \param condition (in)     : The condition to be evaluated
    \param lm (in)            : location vector of this element
    \param elevec1 (out)      : Force vector to be filled by element

    \return 0 if successful, negative otherwise
    */
    int evaluate_neumann(Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
        const Core::Conditions::Condition& condition, std::vector<int>& lm,
        Core::LinAlg::SerialDenseVector& elevec1,
        Core::LinAlg::SerialDenseMatrix* elemat1 = nullptr) override
    {
      return 0;
    }

    /*!
    \brief Get shape type of element
    */
    Core::FE::CellType shape() const override { return Core::FE::CellType::dis_none; };

   private:
    std::map<Core::Binstrategy::Utils::BinContentType, std::vector<Core::Elements::Element*>>
        associated_ele_;

  };  // class MeshfreeMultiBin

}  // namespace Core::FE::MeshFree


FOUR_C_NAMESPACE_CLOSE

#endif
