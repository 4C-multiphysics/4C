/*----------------------------------------------------------------------*/
/*! \file
 \brief helper methods/classes for multiphase porous fluid flow problems

   \level 3

 *----------------------------------------------------------------------*/

#ifndef FOUR_C_POROFLUIDMULTIPHASE_UTILS_HPP
#define FOUR_C_POROFLUIDMULTIPHASE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_inpar_bio.hpp"
#include "4C_inpar_porofluidmultiphase.hpp"
#include "4C_io.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

#include <Epetra_Vector.h>
#include <Teuchos_RCP.hpp>

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Adapter
{
  class PoroFluidMultiphase;
}

namespace POROFLUIDMULTIPHASE
{
  /// POROFLUIDMULTIPHASE::UTILS: Random stuff that might be helpful when dealing with
  /// poromultiphase problems
  namespace UTILS
  {
    /// setup second materials for porosity evaluation within solid phase
    void setup_material(const Epetra_Comm& comm, const std::string& struct_disname,
        const std::string& fluid_disname);


    /// convert a dof based vector to a node based multi vector
    /*!
      For postprocessing, only vectors based on the primary dof set
      of the discretization can be used. Hence, all other EpetraVectors
      based on secondary dof sets are copied to EpetraMultiVectors with one
      node based vector for each component.

      This method can be deleted, if the post processors would be adapted to
      handle secondary dof sets.

      \param dis              : discretization, the vector is based on
      \param vector           : vector to convert
      \param nds              : number of the dof set the map of the vector corresponds to
      \param numdofpernode    : number of dofs per node of the vector (assumed to be equal for all
      nodes)
     */
    Teuchos::RCP<Epetra_MultiVector> convert_dof_vector_to_node_based_multi_vector(
        const Core::FE::Discretization& dis, const Epetra_Vector& vector, const int nds,
        const int numdofpernode);

    /// create solution algorithm depending on input file
    Teuchos::RCP<Adapter::PoroFluidMultiphase> create_algorithm(
        Inpar::POROFLUIDMULTIPHASE::TimeIntegrationScheme
            timintscheme,                                    //!< time discretization scheme
        Teuchos::RCP<Core::FE::Discretization> dis,          //!< discretization
        const int linsolvernumber,                           //!< number of linear solver
        const Teuchos::ParameterList& probparams,            //!< parameter list of global problem
        const Teuchos::ParameterList& poroparams,            //!< paramter list of poro problem
        Teuchos::RCP<Core::IO::DiscretizationWriter> output  //!< output writer
    );

    /**
     * \brief extend ghosting for artery discretization
     * @param[in] contdis  discretization of 2D/3D domain
     * @param[in] artdis   discretization of 1D domain
     * @param[in] evaluate_on_lateral_surface   is coupling evaluated on lateral surface?
     * @return             set of nearby element pairs as seen from the artery discretization, each
     *                     artery element with a vector of close 3D elements
     */
    std::map<int, std::set<int>> extended_ghosting_artery_discretization(
        Teuchos::RCP<Core::FE::Discretization> contdis,
        Teuchos::RCP<Core::FE::Discretization> artdis, const bool evaluate_on_lateral_surface,
        const Inpar::ArteryNetwork::ArteryPoroMultiphaseScatraCouplingMethod couplingmethod);

    /**
     * \brief get axis-aligned bounding box of element
     * @param[in] ele        compute AABB for this element
     * @param[in] positions  nodal positions of discretization
     * @param[in] evaluate_on_lateral_surface   is coupling evaluated on lateral surface?
     * @return               AABB of element as 3x2 Core::LinAlg::Matrix
     */
    Core::LinAlg::Matrix<3, 2> get_aabb(Core::Elements::Element* ele,
        std::map<int, Core::LinAlg::Matrix<3, 1>>& positions,
        const bool evaluate_on_lateral_surface);

    /// maximum distance between two nodes of an element
    double get_max_nodal_distance(
        Core::Elements::Element* ele, Teuchos::RCP<Core::FE::Discretization> dis);

    /**
     * \brief perform octtree search for NTP coupling
     * @param[in] contdis       discretization of 2D/3D domain
     * @param[in] artdis        discretization of 1D domain
     * @param[in] artsearchdis  fully overlapping discretization of 1D domain on which search is
     * performed
     * @param[in] evaluate_on_lateral_surface   is coupling evaluated on lateral surface?
     * @param[in] artEleGIDs   vector of artery coupling element Ids
     * @param[out] elecolset    additional elements that need to be ghosted are filled into this set
     * @param[out] nodecolset   additional nodes that need to be ghosted are filled into this set
     * @return                  set of nearby element pairs as seen from the artery discretization,
     *                          each artery element with a vector of close 3D elements
     */
    std::map<int, std::set<int>> oct_tree_search(Teuchos::RCP<Core::FE::Discretization> contdis,
        Teuchos::RCP<Core::FE::Discretization> artdis,
        Teuchos::RCP<Core::FE::Discretization> artsearchdis, const bool evaluate_on_lateral_surface,
        const std::vector<int> artEleGIDs, std::set<int>& elecolset, std::set<int>& nodecolset);

    /*!
     * \brief get nodal positions of discretization as std::map
     * @param[in] dis      discretization for which nodal positions will be returned
     * @param[in] nodemap  node-map of the discretization (can be either in row or column format)
     * @return             nodal position as a std::map<int, Core::LinAlg::Matrix<3, 1>>
     */
    std::map<int, Core::LinAlg::Matrix<3, 1>> get_nodal_positions(
        Teuchos::RCP<Core::FE::Discretization> dis, const Epetra_Map* nodemap);

    //! Determine norm of vector
    double calculate_vector_norm(
        const enum Inpar::POROFLUIDMULTIPHASE::VectorNorm norm,  //!< norm to use
        const Teuchos::RCP<const Epetra_Vector> vect             //!< the vector of interest
    );

    /*!
     * create (fully overlapping) search discretization
     * @param artdis  1D artery discretization
     * @param disname  name of fully-overlapping artery discretization
     * @param doboundaryconditions  also do boundary conditions in fill-complete call
     * @return  fully-overlapping artery discretization
     */
    Teuchos::RCP<Core::FE::Discretization> create_fully_overlapping_artery_discretization(
        Teuchos::RCP<Core::FE::Discretization> artdis, std::string disname,
        bool doboundaryconditions);

  }  // namespace UTILS
  // Print the logo
  void print_logo();
}  // namespace POROFLUIDMULTIPHASE


FOUR_C_NAMESPACE_CLOSE

#endif
