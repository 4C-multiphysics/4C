/*-----------------------------------------------------------*/
/*! \file
\brief calc utils for beam interaction framework


\level 3
*/
/*-----------------------------------------------------------*/

#ifndef FOUR_C_BEAMINTERACTION_CALC_UTILS_HPP
#define FOUR_C_BEAMINTERACTION_CALC_UTILS_HPP

#include "baci_config.hpp"

#include "baci_inpar_beaminteraction.hpp"
#include "baci_linalg_fixedsizematrix.hpp"
#include "baci_linalg_mapextractor.hpp"

#include <Teuchos_RCP.hpp>

// forward declaration
class Epetra_FEVector;
class Epetra_Vector;

BACI_NAMESPACE_OPEN

namespace CORE::LINALG
{
  class SerialDenseVector;
  class SerialDenseMatrix;
  class SparseMatrix;
}  // namespace CORE::LINALG

namespace DRT
{
  class Node;
  class Element;
  class Discretization;
}  // namespace DRT

namespace CORE::GEO
{
  namespace MESHFREE
  {
    class BoundingBox;
  }
}  // namespace CORE::GEO

namespace BEAMINTERACTION
{
  class CrosslinkingParams;
  class BeamLink;
  namespace UTILS
  {
    /// specific MultiMapExtractor to handle different types of element during beam interaction
    class MapExtractor : public CORE::LINALG::MultiMapExtractor
    {
     public:
      enum
      {
        beam = 0,
        sphere = 1,
        solid = 2
      };

      MAP_EXTRACTOR_VECTOR_METHODS(Beam, beam)
      MAP_EXTRACTOR_VECTOR_METHODS(Sphere, sphere)
      MAP_EXTRACTOR_VECTOR_METHODS(Solid, solid)
    };

    /// class for comparing DRT::Element* (and DRT::Node*) in a std::set
    /*! -------------------------------------------------------------------------
     * overwrites standard < for pointers, this is necessary to ensure same order
     * of neighboring elements for crosslink check and therefore for random numbers
     * independent of pointer addresses. Without this,
     * simulation with crosslinker is not wrong, but depends randomly on memory
     * allocation, i.e. pointer addresses. Without random numbers, everything is fine
     * with default compare operator
    *  \author J. Eichinger March 2017
     -------------------------------------------------------------------------*/
    class Less
    {
     public:
      template <typename ELEMENT>
      bool operator()(ELEMENT const* first, ELEMENT const* second) const
      {
        return first->Id() < second->Id();
      }
    };

    /*! -------------------------------------------------------------------------
     * class for comparing std::set< std::pair < int, int > >
     *  \author J. Eichinger March 2017
     -------------------------------------------------------------------------*/
    class StdPairComparatorOrderCounts
    {
     public:
      bool operator()(std::pair<int, int> const& lhs, std::pair<int, int> const& rhs) const
      {
        return (lhs.first == rhs.first) ? (lhs.second < rhs.second) : (lhs.first < rhs.first);
      }
    };


    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool IsBeamElement(DRT::Element const& element);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool IsRigidSphereElement(DRT::Element const& element);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool IsBeamNode(DRT::Node const& node);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool IsRigidSphereNode(DRT::Node const& node);

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    bool IsBeamCenterlineNode(DRT::Node const& node);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void PeriodicBoundaryConsistentDisVector(Teuchos::RCP<Epetra_Vector> dis,
        Teuchos::RCP<const CORE::GEO::MESHFREE::BoundingBox> const& pbb,
        Teuchos::RCP<const DRT::Discretization> const& discret);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    inline int CalculateNumberOfBeamElementsFromNumberOfNodesOnFilament(
        int const numnodes, int const numnodesperele)
    {
      // from: nodesperfil = nodesperele + ( numele - 1 ) * ( nodesperele - 1 )
      return ((numnodes - numnodesperele) / (numnodesperele - 1.0)) + 1.0;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    std::vector<int> Permutation(int number);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void GetCurrentElementDis(DRT::Discretization const& discret, DRT::Element const* ele,
        Teuchos::RCP<const Epetra_Vector> const& ia_discolnp, std::vector<double>& eledisp);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void GetCurrentUnshiftedElementDis(DRT::Discretization const& discret, DRT::Element const* ele,
        Teuchos::RCP<const Epetra_Vector> const& ia_discolnp,
        CORE::GEO::MESHFREE::BoundingBox const& pbb, std::vector<double>& eledisp);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    template <typename T>
    void SetFilamentBindingSpotPositions(
        Teuchos::RCP<DRT::Discretization> discret, Teuchos::RCP<T> params);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void ExtendGhostingForFilamentBspotSetup(
        std::set<int>& relevantfilaments, Teuchos::RCP<DRT::Discretization> discret);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void DetermineOffMyRankNodesWithRelevantEleCloudForFilamentBspotSetup(
        std::set<int>& relevantfilaments, std::set<int>& setofrequirednodes,
        Teuchos::RCP<DRT::Discretization> discret);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void ComputeFilamentLengthAndSortItsElements(std::vector<DRT::Element*>& sortedfilamenteles,
        std::vector<int> const* nodeids, double& filreflength,
        Teuchos::RCP<DRT::Discretization> discret);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void SetBindingSpotsPositionsOnFilament(std::vector<DRT::Element*>& sortedfilamenteles,
        double start, INPAR::BEAMINTERACTION::CrosslinkerType linkertype, int numbspot,
        double filamentbspotinterval, double tol);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void GetPosAndTriadOfBindingSpot(DRT::Element* ele,
        Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox> const& pbb,
        INPAR::BEAMINTERACTION::CrosslinkerType linkertype, int locbspotnum,
        CORE::LINALG::Matrix<3, 1>& bspotpos, CORE::LINALG::Matrix<3, 3>& bspottriad,
        std::vector<double>& eledisp);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void GetPosAndTriadOfBindingSpot(DRT::Discretization const& discret, DRT::Element* ele,
        Teuchos::RCP<Epetra_Vector> const& ia_discolnp,
        Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox> const& pbb,
        INPAR::BEAMINTERACTION::CrosslinkerType linkertype, int locbspotnum,
        CORE::LINALG::Matrix<3, 1>& bspotpos, CORE::LINALG::Matrix<3, 3>& bspottriad);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    bool IsDistanceOutOfRange(CORE::LINALG::Matrix<3, 1> const& pos1,
        CORE::LINALG::Matrix<3, 1> const& pos2, double const lowerbound, double const upperbound);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    bool IsEnclosedAngleOutOfRange(CORE::LINALG::Matrix<3, 1> const& direction1,
        CORE::LINALG::Matrix<3, 1> const& direction2, double const lowerbound,
        double const upperbound);

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    bool DoBeamElementsShareNodes(DRT::Element const* const beam, DRT::Element const* const nbbeam);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void FEAssembleEleForceStiffIntoSystemVectorMatrix(const DRT::Discretization& discret,
        std::vector<int> const& elegid, std::vector<CORE::LINALG::SerialDenseVector> const& elevec,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> const& elemat,
        Teuchos::RCP<Epetra_FEVector> fe_sysvec,
        Teuchos::RCP<CORE::LINALG::SparseMatrix> fe_sysmat);


    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/

    /**
     * \brief Get the number of centerline DOFs for a given beam element.
     * @param ele (in) Pointer to the element.
     */
    unsigned int GetNumberOfElementCenterlineDof(const DRT::Element* elerline_gid);

    /**
     * \brief Get the global indices of the centerline DOFs of a beam element.
     * @param discret (in) Pointer to the discretization.
     * @param ele (in) Pointer to the element.
     * @param ele_centerline_dof_indices (out) Vector with global indices of centerline DOFs in the
     * element.
     */
    template <unsigned int n_centerline_dof>
    void GetElementCenterlineGIDIndices(DRT::Discretization const& discret, const DRT::Element* ele,
        CORE::LINALG::Matrix<n_centerline_dof, 1, int>& centerline_gid);

    /**
     * \brief Get the local indices of the centerline DOFs of an element.
     * @param discret (in) Pointer to the discretization.
     * @param ele (in) Pointer to the element.
     * @param ele_centerline_dof_indices (out) Vector with local indices of centerline DOFs in the
     * element.
     * @param num_dof (out) Number total DOFs on the element.
     */
    void GetElementCenterlineDOFIndices(DRT::Discretization const& discret, const DRT::Element* ele,
        std::vector<unsigned int>& ele_centerline_dof_indices, unsigned int& num_dof);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void AssembleCenterlineDofForceStiffIntoElementForceStiff(DRT::Discretization const& discret,
        std::vector<int> const& elegid,
        std::vector<CORE::LINALG::SerialDenseVector> const& eleforce_centerlineDOFs,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> const& elestiff_centerlineDOFs,
        std::vector<CORE::LINALG::SerialDenseVector>* eleforce,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>>* elestiff);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/

    /**
     * \brief Assemble a matrix with columns based on centerline DOFs of an element into a matrix
     * with columns based on all DOFs of the element. Example: Mortar coupling matrices as the rows
     * correspond the Lagrange multipliers and the columns correspond to the centerline DOFs.
     * @param discret (in) Pointer to the discretization.
     * @param element (in) Pointer to the element.
     * @param row_matrix_centerlineDOFs (in) Matrix where the columns correspond to the centerline
     * DOFs.
     * @param row_matrix_elementDOFs (out) Matrix where the columns correspond to all Element DOFs
     * (the rest will be 0).
     */
    void AssembleCenterlineDofColMatrixIntoElementColMatrix(DRT::Discretization const& discret,
        const DRT::Element* element,
        CORE::LINALG::SerialDenseMatrix const& row_matrix_centerlineDOFs,
        CORE::LINALG::SerialDenseMatrix& row_matrix_elementDOFs);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void ExtractPosDofVecAbsoluteValues(DRT::Discretization const& discret, DRT::Element const* ele,
        Teuchos::RCP<const Epetra_Vector> const& ia_discolnp,
        std::vector<double>& element_posdofvec_absolutevalues);
    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void ExtractPosDofVecValues(DRT::Discretization const& discret, DRT::Element const* ele,
        Teuchos::RCP<const Epetra_Vector> const& ia_discolnp,
        std::vector<double>& element_posdofvec_values);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    template <class T1, class T2>
    void ApplyBindingSpotForceToParentElements(DRT::Discretization const& discret,
        Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox> const& pbb,
        const Teuchos::RCP<Epetra_Vector> disp_np_col,
        const Teuchos::RCP<BEAMINTERACTION::BeamLink> elepairptr,
        std::vector<CORE::LINALG::SerialDenseVector> const& bspotforce,
        std::vector<CORE::LINALG::SerialDenseVector>& eleforce);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    template <class T1, class T2>
    void ApplyBindingSpotStiffToParentElements(DRT::Discretization const& discret,
        Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox> const& pbb,
        const Teuchos::RCP<Epetra_Vector> disp_np_col,
        const Teuchos::RCP<BEAMINTERACTION::BeamLink> elepairptr,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> const& bspotstiff,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>>& elestiff);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    template <class T1, class T2>
    void ApplyBindingSpotForceStiffToParentElements(DRT::Discretization const& discret,
        Teuchos::RCP<CORE::GEO::MESHFREE::BoundingBox> const& pbb,
        const Teuchos::RCP<Epetra_Vector> disp_np_col,
        const Teuchos::RCP<BEAMINTERACTION::BeamLink> elepairptr,
        std::vector<CORE::LINALG::SerialDenseVector> const& bspotforce,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>> const& bspotstiff,
        std::vector<CORE::LINALG::SerialDenseVector>& eleforce,
        std::vector<std::vector<CORE::LINALG::SerialDenseMatrix>>& elestiff);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void SetupEleTypeMapExtractor(Teuchos::RCP<const DRT::Discretization> const& discret,
        Teuchos::RCP<CORE::LINALG::MultiMapExtractor>& eletypeextractor);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void UpdateDofMapOfVector(Teuchos::RCP<DRT::Discretization> discret,
        Teuchos::RCP<Epetra_Vector>& dofmapvec, Teuchos::RCP<Epetra_Vector> old = Teuchos::null);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    long long CantorPairing(std::pair<int, int> const& pair);

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    std::pair<int, int> CantorDePairing(long long z);


  }  // namespace UTILS
}  // namespace BEAMINTERACTION

BACI_NAMESPACE_CLOSE

#endif
