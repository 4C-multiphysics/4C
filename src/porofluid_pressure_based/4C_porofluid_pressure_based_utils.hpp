// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_POROFLUID_PRESSURE_BASED_UTILS_HPP
#define FOUR_C_POROFLUID_PRESSURE_BASED_UTILS_HPP

#include "4C_config.hpp"

#include "4C_inpar_bio.hpp"
#include "4C_io.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_porofluid_pressure_based_input.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <memory>
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

namespace PoroPressureBased
{
  /// setup second materials for porosity evaluation within solid phase
  void setup_material(
      MPI_Comm comm, const std::string& struct_disname, const std::string& fluid_disname);


  /// convert a dof based vector to a node based multi vector
  /*!
    For postprocessing, only vectors based on the primary dof set
    of the discretization can be used. Hence, all other vectors
    based on secondary dof sets are copied to multi vectors with one
    node based vector for each component.

    This method can be deleted, if the post processors would be adapted to
    handle secondary dof sets.

    \param dis              : discretization, the vector is based on
    \param vector           : vector to convert
    \param nds              : number of the dof set the map of the vector corresponds to
    \param numdofpernode    : number of dofs per node of the vector (assumed to be equal for all
    nodes)
   */
  std::shared_ptr<Core::LinAlg::MultiVector<double>> convert_dof_vector_to_node_based_multi_vector(
      const Core::FE::Discretization& dis, const Core::LinAlg::Vector<double>& vector,
      const int nds, const int numdofpernode);

  /// create solution algorithm depending on input file
  std::shared_ptr<Adapter::PoroFluidMultiphase> create_algorithm(
      PoroPressureBased::TimeIntegrationScheme timintscheme,  //!< time discretization scheme
      std::shared_ptr<Core::FE::Discretization> dis,          //!< discretization
      const int linsolvernumber,                              //!< number of linear solver
      const Teuchos::ParameterList& probparams,               //!< parameter list of global problem
      const Teuchos::ParameterList& poroparams,               //!< parameter list of poro problem
      std::shared_ptr<Core::IO::DiscretizationWriter> output  //!< output writer
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
      Core::FE::Discretization& contdis, std::shared_ptr<Core::FE::Discretization> artdis,
      const bool evaluate_on_lateral_surface,
      const Inpar::ArteryNetwork::ArteryPorofluidElastScatraCouplingMethod couplingmethod);

  /**
   * \brief get axis-aligned bounding box of element
   * @param[in] ele        compute AABB for this element
   * @param[in] positions  nodal positions of discretization
   * @param[in] evaluate_on_lateral_surface   is coupling evaluated on lateral surface?
   * @return               AABB of element as 3x2 Core::LinAlg::Matrix
   */
  Core::LinAlg::Matrix<3, 2> get_aabb(Core::Elements::Element* ele,
      std::map<int, Core::LinAlg::Matrix<3, 1>>& positions, const bool evaluate_on_lateral_surface);

  /// maximum distance between two nodes of an element
  double get_max_nodal_distance(Core::Elements::Element* ele, Core::FE::Discretization& dis);

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
  std::map<int, std::set<int>> oct_tree_search(Core::FE::Discretization& contdis,
      Core::FE::Discretization& artdis, Core::FE::Discretization& artsearchdis,
      const bool evaluate_on_lateral_surface, const std::vector<int> artEleGIDs,
      std::set<int>& elecolset, std::set<int>& nodecolset);

  /*!
   * \brief get nodal positions of discretization as std::map
   * @param[in] dis      discretization for which nodal positions will be returned
   * @param[in] nodemap  node-map of the discretization (can be either in row or column format)
   * @return             nodal position as a std::map<int, Core::LinAlg::Matrix<3, 1>>
   */
  std::map<int, Core::LinAlg::Matrix<3, 1>> get_nodal_positions(
      Core::FE::Discretization& dis, const Core::LinAlg::Map* nodemap);

  //! Determine norm of vector
  double calculate_vector_norm(const enum PoroPressureBased::VectorNorm norm,  //!< norm to use
      const Core::LinAlg::Vector<double>& vect  //!< the vector of interest
  );

  /*!
   * create (fully overlapping) search discretization
   * @param artdis  1D artery discretization
   * @param disname  name of fully-overlapping artery discretization
   * @param doboundaryconditions  also do boundary conditions in fill-complete call
   * @return  fully-overlapping artery discretization
   */
  std::shared_ptr<Core::FE::Discretization> create_fully_overlapping_artery_discretization(
      Core::FE::Discretization& artdis, std::string disname, bool doboundaryconditions);

}  // namespace PoroPressureBased


FOUR_C_NAMESPACE_CLOSE

#endif
