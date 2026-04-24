// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FLUID_ELE_INTFACES_CALC_HPP
#define FOUR_C_FLUID_ELE_INTFACES_CALC_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_mat_material_factory.hpp"
#include "4C_material_base.hpp"
#include "4C_utils_singleton_owner.hpp"
#include "4C_xfem_input.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Core::LinAlg
{
  class SparseMatrix;
}

namespace Core::FE
{
  class Discretization;
  class DiscretizationFaces;
}  // namespace Core::FE

namespace Discret
{
  namespace Elements
  {
    class FluidIntFace;
    class FluidEleParameter;
    class FluidEleParameterTimInt;
    class FluidEleParameterIntFace;

    /// Interface base class for FluidIntFaceImpl
    /*!
      This class exists to provide a common interface for all template
      versions of FluidIntFaceImpl. The only function
      this class actually defines is Impl, which returns a pointer to
      the appropriate version of FluidIntFaceImpl.
     */
    class FluidIntFaceImplInterface
    {
     public:
      /// Empty constructor
      FluidIntFaceImplInterface() {}

      /// Empty destructor
      virtual ~FluidIntFaceImplInterface() = default;
      //! Assemble internal faces integrals using data from both parent elements
      virtual void assemble_internal_faces_using_neighbor_data(
          Discret::Elements::FluidIntFace* intface,        ///< internal face element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          std::vector<int>& nds_master,                    ///< nodal dofset w.r.t. master element
          std::vector<int>& nds_slave,                     ///< nodal dofset w.r.t. slave element
          const XFEM::FaceType& face_type,  ///< which type of face std, ghost, ghost-penalty
          Teuchos::ParameterList& params,   ///< parameter list
          Core::FE::DiscretizationFaces& discretization,              ///< faces discretization
          std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,   ///< systemmatrix
          std::shared_ptr<Core::LinAlg::Vector<double>> systemvector  ///< systemvector
          ) = 0;

      //! Evaluate internal faces
      virtual int evaluate_internal_faces(
          Discret::Elements::FluidIntFace* intface,        ///< internal face element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          Teuchos::ParameterList& params,                  ///< parameter list
          Core::FE::Discretization& discretization,        ///< discretization
          std::vector<int>& patchlm,                       ///< patch local map
          std::vector<int>& lm_masterToPatch,  ///< local map between master dofs and patchlm
          std::vector<int>& lm_slaveToPatch,   ///< local map between slave dofs and patchlm
          std::vector<int>& lm_faceToPatch,    ///< local map between face dofs and patchlm
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch,  ///< local map between slave nodes and nodes in patch
          std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  ///< element matrix blocks
          std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   ///< element vector blocks
          ) = 0;


      /// Internal implementation class for FluidIntFace elements (the first object is created in
      /// Discret::Elements::FluidIntFace::Evaluate)
      static FluidIntFaceImplInterface* impl(const Core::Elements::Element* ele);
    };

    /// Internal FluidIntFace element implementation
    /*!
      This internal class keeps all the working arrays needed to
      calculate the FluidIntFace element.

      <h3>Purpose</h3>

      The FluidIntFace element will allocate exactly one object of this class
      for all FluidIntFace elements with the same number of nodes in the mesh.
      This allows us to use exactly matching working arrays (and keep them
      around.)

      The code is meant to be as clean as possible. This is the only way
      to keep it fast. The number of working arrays has to be reduced to
      a minimum so that the element fits into the cache. (There might be
      room for improvements.)


    */
    template <Core::FE::CellType distype>
    class FluidIntFaceImpl : public FluidIntFaceImplInterface
    {
     public:
      /// Singleton access method
      static FluidIntFaceImpl<distype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      /// Constructor
      FluidIntFaceImpl();


      //! Assemble internal faces integrals using data from both parent elements
      void assemble_internal_faces_using_neighbor_data(
          Discret::Elements::FluidIntFace* intface,        ///< internal face element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          std::vector<int>& nds_master,                    ///< nodal dofset w.r.t. master element
          std::vector<int>& nds_slave,                     ///< nodal dofset w.r.t. slave element
          const XFEM::FaceType& face_type,  ///< which type of face std, ghost, ghost-penalty
          Teuchos::ParameterList& params,   ///< parameter list
          Core::FE::DiscretizationFaces& discretization,              ///< faces discretization
          std::shared_ptr<Core::LinAlg::SparseMatrix> systemmatrix,   ///< systemmatrix
          std::shared_ptr<Core::LinAlg::Vector<double>> systemvector  ///< systemvector
          ) override;

      //! Evaluate internal faces
      int evaluate_internal_faces(
          Discret::Elements::FluidIntFace* intface,        ///< internal face element
          std::shared_ptr<Core::Mat::Material>& material,  ///< material associated with the faces
          Teuchos::ParameterList& params,                  ///< parameter list
          Core::FE::Discretization& discretization,        ///< discretization
          std::vector<int>& patchlm,                       ///< patch local map
          std::vector<int>& lm_masterToPatch,  ///< local map between master dofs and patchlm
          std::vector<int>& lm_slaveToPatch,   ///< local map between slave dofs and patchlm
          std::vector<int>& lm_faceToPatch,    ///< local map between face dofs and patchlm
          std::vector<int>&
              lm_masterNodeToPatch,  ///< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch,  ///< local map between slave nodes and nodes in patch
          std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  ///< element matrix blocks
          std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   ///< element vector blocks
          ) override;


     private:
      //! pointer to parameter list for time integration
      Discret::Elements::FluidEleParameterTimInt* fldparatimint_;
      //! pointer to parameter list for internal faces
      Discret::Elements::FluidEleParameterIntFace* fldpara_intface_;


    };  // end class FluidIntFaceImpl

  }  // namespace Elements
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
