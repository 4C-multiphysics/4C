// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BINSTRATEGY_HPP
#define FOUR_C_BINSTRATEGY_HPP

/*----------------------------------------------------------------------*
 | headers                                                  ghamm 09/12 |
 *----------------------------------------------------------------------*/
#include "4C_config.hpp"

#include "4C_binstrategy_utils.hpp"
#include "4C_io_visualization_parameters.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_parameter_list.fwd.hpp"

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::FE
{
  class Discretization;
}  // namespace Core::FE

namespace Core::Elements
{
  class Element;
}

namespace Core::Nodes
{
  class Node;
}


namespace Core::Geo::MeshFree
{
  class BoundingBox;
}

namespace Core::IO
{
  class OutputControl;
  class DiscretizationVisualizationWriterMesh;
}  // namespace Core::IO

/*----------------------------------------------------------------------*
 | binning strategy                                         ghamm 11/13 |
 *----------------------------------------------------------------------*/
namespace Core::Binstrategy
{
  /// forward declaration
  class Less;

  /**
   * Write either no, row or column bins for visualization.
   */
  enum WriteBins
  {
    none,
    rows,
    cols
  };

  /**
   * Helper functor to determine relevant points for binning. By default, these are the element
   * nodes. This behavior may be modified by setting the correct_node() function to select different
   * nodes.
   */
  struct DefaultRelevantPoints
  {
    /**
     * Determine relevant points for binning.
     *
     * @param discret Discretization to consider.
     * @param ele Element to consider.
     * @param disnp Displacement state.
     * @return Vector of relevant points stored as array of length 3.
     */
    std::vector<std::array<double, 3>> operator()(const Core::FE::Discretization& discret,
        const Core::Elements::Element& ele,
        std::shared_ptr<const Core::LinAlg::Vector<double>> disnp);

    /**
     * Additional function to determine select a different node. By default, the node is returned.
     */
    std::function<const Core::Nodes::Node&(const Core::Nodes::Node& node)> correct_node =
        [](const auto& node) -> decltype(auto) { return node; };
  };

  /*!
   *  \brief strategy for sorting data (e.g. finite elements) into spatial bins
   *  to enable tracking of interaction (e.g. contact) between them fully in
   *  parallel without fully overlapping maps
   *
   * Binning strategy is based on a geometrical decomposition of the computational domain in
   * box-like shaped cells. Each element/particle is assigned to a cell and the size of the
   * cells is chosen such that all possible interaction partners are found within one layer of
   * cells. The cells are denoted as bins. Hence, interaction evaluation for a single element
   * involves gathering the content residing in its own bin and in the 26 neighboring bins around.
   *
   * Therefore, a binning approach with Cartesian boxes is chosen. Like this, each bin can be
   * addressed using three indices i, j and k each ranging from zero to the maximal number of
   * bins in the respective spatial direction. The assigning procedure of content to bin needs
   * to be updated to always provide the correct neighborhood when interaction is evaluated.
   * Depending on the chosen time step size and bin size, it might be possible to reuse the
   * neighborhood information for several time steps.
   *
   * The assignment of elements that might be of arbitrary shape (e.g. hexahedral or tetrahedral),
   * additionally distorted and potentially partly overlapping covering multiple bins as the
   * standard case is done using an axis aligned bounding box (aabb) that is computed for each
   * element. The element is assigned to each bin which is touched by the aabb. A nice side effect
   * of this approach is that there is no restriction on the size ratio between element and bin. The
   * present approach does not rely on fully redundant information because only elements in one
   * additional layer of bins are ghosted. A crucial assumption in this approach is the restriction
   * of the interaction range to one layer of bins. This constraint ensures that no event is missed
   * in case of parallel computations when one additional layer of ghost bins is provided at
   * processor boundaries.
   *
   * Details on the approach for particle problems can be found e.g. in:
   * - M. P. Allen and D. J. Tildesley, Computer simulation of liquids, Oxford University Press, New
   *   York, 1991.
   * - Allen S. Plimpton, Fast parallel algorithms for short-range molecular dynamics, Journal of
   *   Computational Physics 117, 1-19, 1995.
   * - T. Poschel and T. Schwager, Computational Granular Dynamics, Springer-Verlag,
   *   Berlin,Heidelberg, 2005.

   *
   *
   */
  class BinningStrategy
  {
   public:
    /*!
     * \brief construct binning strategy
     *
     * \param[in] binning_params binning parameters from input
     * \param[in] output_control output control file
     * \param[in] comm MPI Communicator
     * \param[in] my_rank id of this process
     * \param[in] bin_output_visualization_parameters visualization parameters used to write the
     * optional bin output
     * \param[in] correct_node should a node be included into binning? If this function is not
     * provided, all nodes are considered relevant.
     * \param[in] determine_relevant_points function that determines the relevant points for binning
     * \param[in] discret vector of discretizations as basis for build up of binning domain
     * (optional, only needed if binning domain is not described via input file)
     * \param[in] disnp vector of column displacement states (belonging to input discrets) so that
     * current positions of elements and nodes can be considered for build up of binning domain
     */
    BinningStrategy(const Teuchos::ParameterList& binning_params,
        std::shared_ptr<Core::IO::OutputControl> output_control, MPI_Comm comm, const int my_rank,
        std::optional<Core::IO::VisualizationParameters> bin_output_visualization_parameters =
            std::nullopt,
        std::function<const Core::Nodes::Node&(const Core::Nodes::Node& node)> correct_node = {},
        std::function<std::vector<std::array<double, 3>>(const Core::FE::Discretization&,
            const Core::Elements::Element&,
            std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)>
            determine_relevant_points = DefaultRelevantPoints{},
        const std::vector<std::shared_ptr<Core::FE::Discretization>>& discret = {},
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> disnp = {});

    //! \name Read access functions
    //! \{

    /*!
     * \brief get const binning discretization pointer (elements = bins)
     *
     * \return const pointer to binning discretization
     */
    inline std::shared_ptr<Core::FE::Discretization> const& bin_discret() const { return bindis_; }

    /*!
     * \brief get const list of pointer to boundary row bins
     *
     * \return reference to const list of pointer to boundary row bins
     */
    inline std::list<Core::Elements::Element*> const& boundary_row_bins() const
    {
      return boundaryrowbins_;
    }

    /*!
     * \brief get const set of boundary col bins ids
     *
     * \return const set of boundary col bins ids
     */
    inline std::set<int> const& boundary_col_bins_ids() const { return boundarycolbins_; }

    /*!
     * \brief get lower bound for bin size
     *
     * \return const ref to bin_size_lower_bound_
     */
    inline double const& bin_size_lower_bound() const { return bin_size_lower_bound_; }

    /*!
     * \brief get bin size in all three directions (does not have to be equal to
     * bin_size_lower_bound_)
     *
     * \return const pointer to array containing bin sizes in all tree directions
     */
    inline std::array<double, 3> bin_size() const { return bin_size_; }

    /*!
     * \brief get number of bins in all three directions
     *
     * \return const pointer to array containing number of bins in all directions
     */
    inline std::array<int, 3> bin_per_dir() const { return bin_per_dir_; }

    /*!
     * \brief check if periodic boundary conditions are applied in at least one direction
     *
     * \return true if periodic boundary conditions are applied in at least one direction
     */
    inline bool have_periodic_boundary_conditions_applied() const { return havepbc_; };

    /*!
     * \brief check if periodic boundary conditions are applied in a specific direction
     *
     * \param[in] dim direction were application of periodic boundary condition should be checked
     *
     * \return true if periodic boundary conditions are applied in respective direction
     */
    inline bool have_periodic_boundary_conditions_applied_in_spatial_direction(const int dim) const
    {
      return pbconoff_[dim];
    };
    /*!
     * \brief get length of entire binning domain in a certain direction
     *
     * \param[in] dim spatial direction
     *
     * \return length of binning domain in certain direction
     */
    inline double length_of_binning_domain_in_a_spatial_direction(const int dim) const
    {
      return edge_length_binning_domain_[dim];
    };

    /*!
     * \brief get const dimensions of entire binning domain (resembled by a A_xis A_ligned
     * B_ounding B_ox)
     *
     * \return const bounding box dimension of binning domain
     */
    inline Core::LinAlg::Matrix<3, 2> const& domain_bounding_box_corner_positions() const
    {
      return domain_bounding_box_corner_positions_;
    }

    /*!
     * \brief set binning
     *
     * \param[in] pbb dimension for binning domain
     */
    inline void set_deforming_binning_domain_handler(
        std::shared_ptr<Core::Geo::MeshFree::BoundingBox> const pbb)
    {
      deforming_simulation_domain_handler_ = pbb;
    };

    //! \}

    //! \name Helper functions
    //! \{

    /*!
     * \brief get all bin ids for given range of ijk
     *
     * \param[in] ijk_range given range of ijk
     * \param[out] binIds all bin ids in the specified range
     * \param[in] checkexistence check can be added whether the gids are woned from myrank
     */
    void gids_in_ijk_range(const int* ijk_range, std::set<int>& binIds, bool checkexistence) const;
    /*!
     * \brief get all bin ids for given range of ijk
     *
     * \param[in] ijk_range given range of ijk
     * \param[out] binIds all bin ids in the specified range
     * \param[in] checkexistence check can be added whether gids are owned from myrank
     */
    void gids_in_ijk_range(
        const int* ijk_range, std::vector<int>& binIds, bool checkexistence) const;

    /*!
     * \brief get number of bins in ijk range
     *
     * \param[in] ijk_range ijk range in which number ob bins is counted
     *
     * \return number of bins in requested ijk range
     */
    int get_number_of_bins_in_ijk_range(int const ijk_range[6]) const;

    /*!
     * \brief convert ijk reference to bin to its gid
     *
     * \param[in] ijk i,j,k to be converted into a bin id
     *
     * \return gid of requested bin
     */
    int convert_ijk_to_gid(int* ijk) const;

    /*!
     * \brief convert bin id into ijk specification of bin
     *
     * \param[in] gid bin id to be converted in ijk
     * \param[out] ijk  resulting ijk
     */
    void convert_gid_to_ijk(int gid, int* ijk) const;

    /*!
     * \brief convert a position to its corresponding bin
     *
     * \param[in] pos position that is checked in which bin it resides
     *
     * \return global id of corresponding bin
     */
    int convert_pos_to_gid(const double* pos) const;

    /*!
     * \brief convert a position to its corresponding ijk
     *
     * \param[in] pos  position to be converted into ijk
     * \param[out] ijk resulting ijk
     */
    void convert_pos_to_ijk(const double* pos, int* ijk) const;

    /*!
     * \brief convert a position to its corresponding ijk
     *
     * \param[in] pos  position to be converted into ijk
     * \param[out] ijk resulting ijk
     */
    void convert_pos_to_ijk(const Core::LinAlg::Matrix<3, 1>& pos, int* ijk) const;

    /*!
     * \brief convert position to bin id
     *
     * \param[in] pos convert position to bin id
     *
     * \return position of which the corresponding bin id is asked for
     */
    int convert_pos_to_gid(const Core::LinAlg::Matrix<3, 1>& pos) const;

    /*!
     * \brief get 26 neighboring bin ids (one bin layer) to binID (if existing)
     *
     * \param[in] binId bin id whose connectivity is asked for
     * \param[out] binIds all neighboring bins on axes
     */
    void get_neighbor_bin_ids(const int binId, std::vector<int>& binIds) const;

    /*!
     * \brief  27 neighboring bin ids to binId and myself
     *
     * \param[in] binId bin id whose connectivity is asked for
     * \param[out] binIds all neighboring bins on axes
     */
    void get_neighbor_and_own_bin_ids(const int binId, std::vector<int>& binIds) const;

    /*!
     * \brief get nodal coordinates of bin with given binId
     *
     * \param[in] binId bin id of which corners are calculated
     * \param[out] bincorners position of corners of given bin
     */
    void get_bin_corners(
        const int binId, std::vector<Core::LinAlg::Matrix<3, 1>>& bincorners) const;

    /*!
     * \brief get all bin centers (needed for repartitioning)
     *
     * \param[in] binrowmap bin row map
     * \param[out] bincenters centers of all row bins
     */
    void get_all_bin_centers(
        Core::LinAlg::Map& binrowmap, Core::LinAlg::MultiVector<double>& bincenters) const;

    /*!
     * \brief centroid position for given bin id
     *
     * \param[in] binId bin id of which centroid is calculated
     *
     * \return centroid of requested bin
     */
    Core::LinAlg::Matrix<3, 1> get_bin_centroid(const int binId) const;

    /*!
     * \brief get minimal size of bins
     *
     * \return minimal bin size
     */
    double get_min_bin_size() const;

    /*!
     * \brief get maximal size of bins
     *
     * \return maximal bin size
     */
    double get_max_bin_size() const;

    /*!
     * \brief build periodic boundary conditions
     */
    void build_periodic_bc(const Teuchos::ParameterList& binning_params);

    /*!
     * \brief determine boundary row bins
     */
    void determine_boundary_row_bins();

    /*!
     * \brief determine boundary column bins
     */
    void determine_boundary_col_bins();

    /*!
     * \brief create linear map with bin ids
     *
     * \param[in] comm MPI communicator
     *
     * \return linear map linear map based on bin ids
     */
    std::shared_ptr<Core::LinAlg::Map> create_linear_map_for_numbin(MPI_Comm comm) const;

    /*!
     * \brief write binning domain and its parallel distribution as output
     *       \note: This is a debug feature only to visualize the binning domain and the parallel
     *        distribution of bins to procs. This is computationally very expensive.
     *
     * \param[in] step current step
     * \param[out] time current time
     */
    void write_bin_output(int const step, double const time);

    /*!
     * \brief distribute bins via recursive coordinate bisection
     *
     * \param[in] binrowmap bin row map
     * \param[in] bincenters positions of centers of bins
     * \param[in] binweights weights that is assigned to each bin
     */
    void distribute_bins_recurs_coord_bisection(std::shared_ptr<Core::LinAlg::Map>& binrowmap,
        std::shared_ptr<Core::LinAlg::MultiVector<double>>& bincenters,
        std::shared_ptr<Core::LinAlg::MultiVector<double>>& binweights) const;

    /*!
     * \brief fill bins into bin discretization
     *
     * \param[in] rowbins row bins distribution
     */
    void fill_bins_into_bin_discretization(const Core::LinAlg::Map& rowbins);

    //! \}

    //! \name Functions related to parallel distribution of a discretization using binning
    //! \{

    /*!
     *\brief add ijk to a given axis aligned ijk range of an beam element (this may need special
     *treatment if it is cut by a periodic boundary
     *
     * \param[in] ijk ijk to be added to ijk range
     * \param[out] ijk_range extended ijk range
     */
    void add_ijk_to_axis_aligned_ijk_range(int const ijk[3], int ijk_range[6]) const;

    /**
     * Distribute all elements in the @p element_range to bins by constructing axis-aligned bounding
     * boxes.
     *
     * \param[in] discret discretization containing the elements that are distributed to bins
     * \param[in] element_range range of elements that are distributed to bins
     * \param[out] bin_to_ele_map map of bins and assigned row elements
     * \param[in] disnp current col displacement state
     */
    template <typename Range>
    void distribute_elements_to_bins_using_ele_aabb(const Core::FE::Discretization& discret,
        Range element_range, std::map<int, std::set<int>>& bin_to_ele_map,
        std::shared_ptr<const Core::LinAlg::Vector<double>> disnp = nullptr) const;

    /*!
     * \brief distribute single element to bins using its axis aligned bounding box idea
     *
     * \param[in] discret discretization containing the considered element
     * \param[in] eleptr element thats d
     * \param[out] binIds ids of bins tuuched by aabb of current element
     * \param[in] disnp current col displacement state
     */
    void distribute_single_element_to_bins_using_ele_aabb(const Core::FE::Discretization& discret,
        const Core::Elements::Element& eleptr, std::vector<int>& binIds,
        std::shared_ptr<const Core::LinAlg::Vector<double>> const& disnp) const;

    /*!
     * \brief elements of input discretization are assigned to bins
     *
     * \param[in] discret input discretization
     * \param[in] extended_bin_to_row_ele_map map containing bins [key] and elements[std::set]
     * that belong to it
     */
    void assign_eles_to_bins(Core::FE::Discretization& discret,
        std::map<int, std::set<int>> const& extended_bin_to_row_ele_map,
        const std::function<Utils::BinContentType(const Core::Elements::Element* element)>&) const;

    /*!
     * \brief get element of type bincontent in requested bins
     *
     * \param[out] eles elements belonging to requested bins
     * \param[in] bincontent type of elements you want to have
     * \param[in] binIds bins you want to look for your element type
     * \param[in] roweles flag indicating to just consider elements owned by myrank
     */
    void get_bin_content(std::set<Core::Elements::Element*>& eles,
        const std::vector<Core::Binstrategy::Utils::BinContentType>& bincontent,
        std::vector<int>& binIds, bool roweles = false) const;

    /*!
     * \brief remove all eles from bins
     */
    void remove_all_eles_from_bins();

    /*!
     * \brief assign node to bins
     *
     * \param[in] discret discretization
     * \param[out] bin_to_rownodes_map bin to row nodes assignment map
     * \param[in] disnp current column displacement state
     */
    void distribute_row_nodes_to_bins(const Core::FE::Discretization& discret,
        std::map<int, std::vector<int>>& bin_to_rownodes_map,
        std::shared_ptr<const Core::LinAlg::Vector<double>> disnp = nullptr) const;


    /*!
     * \brief Weighted Partitioning of bins to procs and extended ghosting of discretization to one
     * bin layer
     *
     * 1) create bins, weight them according to number of nodes (of the handed over
     * discretizations) they contain. By also considering the bin connectivity (i.e. a
     * puzzle like distribution of bins is not desirable) an optimal distribution of
     * bins (and therefore elements) to procs is obtained with regard to computation costs.
     *
     * 2) Now we have to apply the new parallel bin distribution by rebuilding the input discrets,
     * i.e. we need to change the ownership of the nodes/elements according to the bin they belong
     * to (each proc then owns the nodes/eles laying in its bins.
     *
     * 3) Make sure to have a correct standard ghosting (owner of element needs to at least ghost
     * all nodes of it, although it might not reside in a bin that myrank owns)
     *
     * 4) ghosting is extended to one layer (two layer ghosting is excluded as it is not needed,
     * this case is covered by other procs then) around bins that actually contain elements. discret
     * than contains all elements that need to be owned or ghosted to ensure correct interaction
     * handling of the elements in the range of one layer
     *
     * \param[in] discret vector of discretizations
     * \param[out] stdelecolmap element column map based on standard ghosting
     * \param[out] stdnodecolmap node column map based on standard ghosting
     *
     * \return row bin distribution
     */
    std::shared_ptr<Core::LinAlg::Map>
    do_weighted_partitioning_of_bins_and_extend_ghosting_of_discret_to_one_bin_layer(
        std::vector<std::shared_ptr<Core::FE::Discretization>> discret,
        std::vector<std::shared_ptr<Core::LinAlg::Map>>& stdelecolmap,
        std::vector<std::shared_ptr<Core::LinAlg::Map>>& stdnodecolmap);

    /*!
     * \brief weighted distribution of bins to procs according to number of nodes they contain
     *
     * \param[in] discret input discretization
     * \param[in] disnp current column displacement state vector
     * \param[in] row_nodes_to_bin_map row nodes belonging to which bin
     * \param[in] weight weight for a node
     * \param[in] repartition whether old distribution exists or not
     *
     * \return new row bin distribution
     */
    std::shared_ptr<Core::LinAlg::Map> weighted_distribution_of_bins_to_procs(
        std::vector<std::shared_ptr<Core::FE::Discretization>>& discret,
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>>& disnp,
        std::vector<std::map<int, std::vector<int>>>& row_nodes_to_bin_map, double const& weight,
        bool repartition = false) const;

    /*!
     * \brief ghosting is extended to one layer
     *
     * ghosting is extended to one layer (two layer ghosting is excluded as it
     * is not needed, this case is covered by other procs then) around bins that
     * actually contain elements. discret than contains all elements that need to be owned or
     * ghosted to ensure correct interaction handling of the elements in the range of one layer.
     *
     * \param[in] bin_to_row_ele_map bin to row element map
     * \param[in] bin_to_row_ele_map_to_lookup_requests bin to row element map to look-up requested
     *            elements from other procs
     * \param[in] ext_bin_to_ele_map one layer extended bin to element map
     * \param[in] bin_colmap bin colmun map
     * \param[in] bin_rowmap bin row map
     * \param[in] ele_colmap_from_standardghosting element column map based on standard ghosting
     *
     * \return extended element column map
     */
    std::shared_ptr<Core::LinAlg::Map> extend_element_col_map(
        const std::map<int, std::set<int>>& bin_to_row_ele_map,
        const std::map<int, std::set<int>>& bin_to_row_ele_map_to_lookup_requests,
        std::map<int, std::set<int>>& ext_bin_to_ele_map,
        std::shared_ptr<const Core::LinAlg::Map> bin_colmap = nullptr,
        std::shared_ptr<const Core::LinAlg::Map> bin_rowmap = nullptr,
        const Core::LinAlg::Map* ele_colmap_from_standardghosting = nullptr) const;

    /*!
     * \brief extend ghosting of binning discretization
     *
     * \param[in] rowbins set containing row bins
     * \param[in] colbins set containing desired col bins
     * \param[in] assigndegreesoffreedom flag indicating if degrees of freedom should be assigned
     *
     */
    void extend_ghosting_of_binning_discretization(const Core::LinAlg::Map& rowbins,
        std::set<int> const& colbins, bool assigndegreesoffreedom = true);

    /*!
     * \brief do standard ghosting for input discretization
     *
     * \param[in] discret discretization
     * \param[in] rowbins row bin distribution
     * \param[in] disnp current column displacement state
     * \param[out] stdelecolmap standard element column map
     * \param[out] stdnodecolmap standard node column map
     */
    void standard_discretization_ghosting(std::shared_ptr<Core::FE::Discretization>& discret,
        const Core::LinAlg::Map& rowbins, std::shared_ptr<Core::LinAlg::Vector<double>>& disnp,
        std::shared_ptr<Core::LinAlg::Map>& stdelecolmap,
        std::shared_ptr<Core::LinAlg::Map>& stdnodecolmap) const;

    /*!
     * \brief collect information about content of bins from other procs via round robin loop
     *
     * \param[in] rowbins row bins
     * \param[in] mynodesinbins my row nodes in bins
     * \param[out] allnodesinmybins all nodes in my row bins
     */
    void collect_information_about_content_of_bins_from_other_procs_via_round_robin(
        const Core::LinAlg::Map& rowbins, std::map<int, std::vector<int>>& mynodesinbins,
        std::map<int, std::vector<int>>& allnodesinmybins) const;

    /*!
     * \brief revert extended ghosting
     *
     * \param[in] dis discretization
     * \param[in] stdelecolmap element column map based on standard ghosting
     * \param[in] stdnodecolmap node column map based on standard ghosting
     */
    void revert_extended_ghosting(std::vector<std::shared_ptr<Core::FE::Discretization>> dis,
        std::vector<std::shared_ptr<Core::LinAlg::Map>>& stdelecolmap,
        std::vector<std::shared_ptr<Core::LinAlg::Map>>& stdnodecolmap) const;

    /*!
     * \brief create axis aligned binning domain for discrets and compute lower bound for bin size
     * as largest element in discret
     *
     * \param[in] discret discretization
     * \param[in] disnp current col displacement state
     * \param[out] domain_bounding_box_corner_positions axis aligned binning domain
     * \param[in] set_bin_size_lower_bound_ flag indicating to set lower bound for bin size
     */
    void compute_min_binning_domain_containing_all_elements_of_multiple_discrets(
        std::vector<std::shared_ptr<Core::FE::Discretization>> discret,
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> disnp,
        Core::LinAlg::Matrix<3, 2>& domain_bounding_box_corner_positions,
        bool set_bin_size_lower_bound_);

    /*!
     * \brief compute lower bound for bin size as largest element in discret
     *
     * \param[in] discret discretization
     * \param[in] disnp current col displacement state
     *
     * \return lower bound for bin size based on size of elements in discret
     */
    double compute_lower_bound_for_bin_size_as_max_edge_length_of_aabb_of_largest_ele(
        std::vector<std::shared_ptr<Core::FE::Discretization>> discret,
        std::vector<std::shared_ptr<const Core::LinAlg::Vector<double>>> disnp);

    /*!
     * \brief create bins based on AABB and lower bound for bin size
     *
     * \param[in] dis current column displacement state
     */
    void create_bins_based_on_bin_size_lower_bound_and_binning_domain_dimensions(
        std::shared_ptr<Core::FE::Discretization> dis = nullptr);

    /*!
     * \brief create binning domain dimensions based on discretization and compute lower bound for
     * bin size as largest element in discret
     *
     * \param[in] discret discretization
     * \param[out] XAABB axis aligned binning domain
     * \param[in] disnp current col displacement state
     * \param[in] set_bin_size_lower_bound_ flag indicating to set bin size lower bound
     */
    void compute_min_binning_domain_containing_all_elements_of_single_discret(
        Core::FE::Discretization& discret, Core::LinAlg::Matrix<3, 2>& XAABB,
        std::shared_ptr<const Core::LinAlg::Vector<double>> disnp = nullptr,
        bool set_bin_size_lower_bound_ = false);

    /*!
     * \brief transfer nodes and elements of input discretization in case they have moved to new
     * bins
     *
     * \param[in] discret discretization
     * \param[in] disnp current column displacement state
     * \param[in] bintorowelemap bin to row element map
     */
    void transfer_nodes_and_elements(Core::FE::Discretization& discret,
        std::shared_ptr<const Core::LinAlg::Vector<double>> disnp,
        std::map<int, std::set<int>>& bintorowelemap);

    //! \}

   private:
    /*!
     * \brief binning discretization with bins as elements
     */
    std::shared_ptr<Core::FE::Discretization> bindis_;

    /*!
     * \brief visualization discretization for bins
     */
    std::shared_ptr<Core::FE::Discretization> visbindis_;

    /*!
     * \brief list of boundary row bins
     */
    std::list<Core::Elements::Element*> boundaryrowbins_;

    /*!
     * \brief list of boundary col bins (one full bin layer)
     */
    std::set<int> boundarycolbins_;

    /*!
     * \brief lower bound for bin size (\note: smallest existing bin might be larger)
     */
    double bin_size_lower_bound_;

    /*!
     * \brief binning domain edges
     */
    Core::LinAlg::Matrix<3, 2> domain_bounding_box_corner_positions_;

    /*!
     * \brief if simulation domain is deforming, e.g. under shear, this handler takes care of this
     */
    std::shared_ptr<Core::Geo::MeshFree::BoundingBox> deforming_simulation_domain_handler_;

    /*!
     * \brief type of bin visualization
     */
    int writebinstype_;

    /*!
     * \brief size of each bin in Cartesian coordinates
     */
    std::array<double, 3> bin_size_ = {0.0, 0.0, 0.0};

    /*!
     * \brief inverse of size of each bin in Cartesian coordinates
     */
    std::array<double, 3> inv_bin_size_ = {0.0, 0.0, 0.0};

    /*!
     * \brief number of bins per direction
     */
    std::array<int, 3> bin_per_dir_ = {0, 0, 0};

    /*!
     * \brief number of bins per direction for bin id calculation
     */
    std::array<int, 3> id_calc_bin_per_dir_ = {0, 0, 0};

    /*!
     * \brief exponent 2^x = number of bins per direction for bin id calculation
     */
    std::array<int, 3> id_calc_exp_bin_per_dir_ = {0, 0, 0};

    /*!
     * \brief global flag whether periodic boundary conditions are specified
     */
    bool havepbc_ = false;

    /*!
     * \brief flags for existence of pbcs in x, y, z direction
     */
    std::array<bool, 3> pbconoff_ = {false, false, false};

    /*!
     * \brief edge length of binning domain in x, y, z direction
     */
    std::array<double, 3> edge_length_binning_domain_ = {0, 0, 0};

    /*!
     * \brief my rank
     */
    int myrank_;

    /*!
     * \brief local communicator
     */
    MPI_Comm comm_;


    //! Function that computes the points to consider as the bounding box of an element. May be
    //! user-supplied for special elements. By default, the nodes of the element are used.
    const std::function<std::vector<std::array<double, 3>>(const Core::FE::Discretization&,
        const Core::Elements::Element&, std::shared_ptr<const Core::LinAlg::Vector<double>> disnp)>
        determine_relevant_points_;

    //! Function that allows selecting a different node to be used in binning.
    const std::function<const Core::Nodes::Node&(const Core::Nodes::Node& node)> correct_node_;

    //! Visualization parameters for the optional bin output
    std::optional<Core::IO::VisualizationParameters> bin_output_visualization_parameters_;

    //! VTK writer used for the optional bin output
    std::shared_ptr<Core::IO::DiscretizationVisualizationWriterMesh> bin_output_writer_;
  };  // namespace Core::Binstrategy

  /*!
   *\brief Class for comparing std::shared_ptr<Core::Nodes::Node> in a std::set
   */
  class Less
  {
   public:
    template <typename ELEMENT>
    bool operator()(
        const std::shared_ptr<ELEMENT>& first, const std::shared_ptr<ELEMENT>& second) const
    {
      return first->Id() < second->Id();
    }
  };


  // --- template and inline functions --- //
  template <typename Range>
  void BinningStrategy::distribute_elements_to_bins_using_ele_aabb(
      const Core::FE::Discretization& discret, Range element_range,
      std::map<int, std::set<int>>& bin_to_ele_map,
      std::shared_ptr<const Core::LinAlg::Vector<double>> disnp) const
  {
    bin_to_ele_map.clear();

    for (auto ele : element_range)
    {
      // get corresponding bin ids in ijk range
      std::vector<int> bin_ids;
      distribute_single_element_to_bins_using_ele_aabb(
          discret, *ele.user_element(), bin_ids, disnp);

      for (const int b : bin_ids) bin_to_ele_map[b].insert(ele.global_id());
    }
  }

}  // namespace Core::Binstrategy


/*----------------------------------------------------------------------*/
FOUR_C_NAMESPACE_CLOSE

#endif
