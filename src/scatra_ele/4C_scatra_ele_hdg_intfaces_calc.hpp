#ifndef FOUR_C_SCATRA_ELE_HDG_INTFACES_CALC_HPP
#define FOUR_C_SCATRA_ELE_HDG_INTFACES_CALC_HPP


#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

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
  namespace ELEMENTS
  {
    class ScaTraHDGIntFace;
    class ScaTraHDGEleParameter;
    class ScaTraHDGEleParameterTimInt;

    //! Interface base class for ScaTraHDGIntFaceImpl
    /*!
      This class exists to provide a common interface for all template
      versions of ScaTraHDGIntFaceImpl. The only function
      this class actually defines is Impl, which returns a pointer to
      the appropriate version of ScaTraHDGIntFaceImpl.
     */
    class ScaTraHDGIntFaceImplInterface
    {
     public:
      //! Empty constructor
      ScaTraHDGIntFaceImplInterface() {}

      //! Empty destructor
      virtual ~ScaTraHDGIntFaceImplInterface() = default;
      //! Assemble internal faces integrals using data from both parent elements
      virtual void assemble_internal_faces_using_neighbor_data(
          Discret::ELEMENTS::ScaTraHDGIntFace* intface,   //!< internal face element
          std::vector<int>& nds_master,                   //!< nodal dofset w.r.t. master element
          std::vector<int>& nds_slave,                    //!< nodal dofset w.r.t. slave element
          Teuchos::ParameterList& params,                 //!< parameter list
          Core::FE::DiscretizationFaces& discretization,  //!< faces discretization
          Teuchos::RCP<Core::LinAlg::SparseMatrix> systemmatrix,   //!< systemmatrix
          Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector  //!< systemvector
          ) = 0;

      //! Evaluate internal faces
      virtual int evaluate_internal_faces(
          Discret::ELEMENTS::ScaTraHDGIntFace* intface,  //!< internal face element
          Teuchos::ParameterList& params,                //!< parameter list
          Core::FE::Discretization& discretization,      //!< discretization
          std::vector<int>& patchlm,                     //!< patch local map
          std::vector<int>& lm_masterToPatch,  //!< local map between master dofs and patchlm
          std::vector<int>& lm_slaveToPatch,   //!< local map between slave dofs and patchlm
          std::vector<int>& lm_faceToPatch,    //!< local map between face dofs and patchlm
          std::vector<int>&
              lm_masterNodeToPatch,  //!< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch,  //!< local map between slave nodes and nodes in patch
          std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  //!< element matrix blocks
          std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   //!< element vector blocks
          ) = 0;


      //! Internal implementation class for ScaTraHDGIntFace elements (the first object is created
      //! in Discret::ELEMENTS::ScaTraHDGIntFace::Evaluate)
      static ScaTraHDGIntFaceImplInterface* impl(const Core::Elements::Element* ele);
    };

    //! Internal ScaTraHDGIntFace element implementation
    /*!
      This internal class keeps all the working arrays needed to
      calculate the ScaTraHDGIntFace element.

      <h3>Purpose</h3>

      The ScaTraHDGIntFace element will allocate exactly one object of this class
      for all ScaTraHDGIntFace elements with the same number of nodes in the mesh.
      This allows us to use exactly matching working arrays (and keep them
      around.)

      The code is meant to be as clean as possible. This is the only way
      to keep it fast. The number of working arrays has to be reduced to
      a minimum so that the element fits into the cache. (There might be
      room for improvements.)

      (see fluid_ele_intfaces_calc.H)

    */
    template <Core::FE::CellType distype>
    class ScaTraHDGIntFaceImpl : public ScaTraHDGIntFaceImplInterface
    {
      friend class ScaTraHDGEleParameterTimInt;
      friend class ScaTraHDGEleParameterStd;

     public:
      //! Singleton access method
      static ScaTraHDGIntFaceImpl<distype>* instance(
          Core::Utils::SingletonAction action = Core::Utils::SingletonAction::create);

      //! Constructor
      ScaTraHDGIntFaceImpl();


      //! Assemble internal faces integrals using data from both parent elements
      void assemble_internal_faces_using_neighbor_data(
          Discret::ELEMENTS::ScaTraHDGIntFace* intface,   //!< internal face element
          std::vector<int>& nds_master,                   //!< nodal dofset w.r.t. master element
          std::vector<int>& nds_slave,                    //!< nodal dofset w.r.t. slave element
          Teuchos::ParameterList& params,                 //!< parameter list
          Core::FE::DiscretizationFaces& discretization,  //!< faces discretization
          Teuchos::RCP<Core::LinAlg::SparseMatrix> systemmatrix,   //!< systemmatrix
          Teuchos::RCP<Core::LinAlg::Vector<double>> systemvector  //!< systemvector
          ) override;

      //! Evaluate internal faces
      int evaluate_internal_faces(
          Discret::ELEMENTS::ScaTraHDGIntFace* intface,  //!< internal face element
          Teuchos::ParameterList& params,                //!< parameter list
          Core::FE::Discretization& discretization,      //!< discretization
          std::vector<int>& patchlm,                     //!< patch local map
          std::vector<int>& lm_masterToPatch,  //!< local map between master dofs and patchlm
          std::vector<int>& lm_slaveToPatch,   //!< local map between slave dofs and patchlm
          std::vector<int>& lm_faceToPatch,    //!< local map between face dofs and patchlm
          std::vector<int>&
              lm_masterNodeToPatch,  //!< local map between master nodes and nodes in patch
          std::vector<int>&
              lm_slaveNodeToPatch,  //!< local map between slave nodes and nodes in patch
          std::vector<Core::LinAlg::SerialDenseMatrix>& elemat_blocks,  //!< element matrix blocks
          std::vector<Core::LinAlg::SerialDenseVector>& elevec_blocks   //!< element vector blocks
          ) override;

      //! decide which terms have to be assembled and decide the assembly pattern, return if no
      //! assembly required
      bool prepare_assemble(Teuchos::ParameterList& stabparams, Teuchos::ParameterList& faceparams);


    };  // end class ScaTraHDGIntFaceImpl

  }  // namespace ELEMENTS
}  // namespace Discret

FOUR_C_NAMESPACE_CLOSE

#endif
