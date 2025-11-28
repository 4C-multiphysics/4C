// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_interface.hpp"

#include "4C_binstrategy.hpp"
#include "4C_contact_interpolator.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_mortar_binarytree.hpp"
#include "4C_mortar_coupling2d.hpp"
#include "4C_mortar_coupling3d.hpp"
#include "4C_mortar_coupling3d_classes.hpp"
#include "4C_mortar_defines.hpp"
#include "4C_mortar_dofset.hpp"
#include "4C_mortar_element.hpp"
#include "4C_mortar_integrator.hpp"
#include "4C_mortar_interface_utils.hpp"
#include "4C_mortar_node.hpp"
#include "4C_mortar_utils.hpp"
#include "4C_poroelast_scatra_utils.hpp"
#include "4C_rebalance_binning_based.hpp"
#include "4C_rebalance_graph_based.hpp"
#include "4C_utils_parameter_list.hpp"

#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <utility>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Mortar::InterfaceDataContainer::InterfaceDataContainer()
    : id_(-1),
      comm_(MPI_COMM_NULL),
      redistributed_(false),
      idiscret_(nullptr),
      dim_(-1),
      imortar_(Teuchos::ParameterList()),
      shapefcn_(Inpar::Mortar::shape_undefined),
      quadslave_(false),
      extendghosting_(Inpar::Mortar::ExtendGhosting::redundant_master),
      oldnodecolmap_(nullptr),
      oldelecolmap_(nullptr),
      snoderowmap_(nullptr),
      snodecolmap_(nullptr),
      mnoderowmap_(nullptr),
      snoderowmapbound_(nullptr),
      snodecolmapbound_(nullptr),
      mnoderowmapnobound_(nullptr),
      mnodecolmapnobound_(nullptr),
      selerowmap_(nullptr),
      selecolmap_(nullptr),
      melerowmap_(nullptr),
      melecolmap_(nullptr),
      sdofrowmap_(nullptr),
      sdofcolmap_(nullptr),
      mdofrowmap_(nullptr),
      mdofcolmap_(nullptr),
      psdofrowmap_(nullptr),
      plmdofmap_(nullptr),
      lmdofmap_(nullptr),
      maxdofglobal_(-1),
      searchalgo_(Inpar::Mortar::search_binarytree),
      binarytree_(nullptr),
      searchparam_(-1.0),
      searchuseauxpos_(false),
      inttime_interface_(0.0),
      nurbs_(false),
      poro_(false),
      porotype_(Inpar::Mortar::other),
      ehl_(false),
      isinit_(false)
{ /* empty */
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Mortar::Interface::Interface(std::shared_ptr<Mortar::InterfaceDataContainer> interfaceData)
    : interface_data_(std::move(interfaceData)),
      id_(interface_data_->id()),
      comm_(interface_data_->comm_ptr()),
      procmap_(interface_data_->proc_map()),
      redistributed_(interface_data_->is_redistributed()),
      idiscret_(interface_data_->i_discret()),
      dim_(interface_data_->n_dim()),
      imortar_(interface_data_->i_mortar()),
      shapefcn_(interface_data_->shape_fcn()),
      quadslave_(interface_data_->is_quad_slave()),
      oldnodecolmap_(interface_data_->old_node_col_map()),
      oldelecolmap_(interface_data_->old_ele_col_map()),
      snoderowmap_(interface_data_->s_node_row_map()),
      snodecolmap_(interface_data_->slave_node_col_map()),
      mnoderowmap_(interface_data_->master_node_row_map()),
      mnodecolmap_(interface_data_->master_node_col_map()),
      snoderowmapbound_(interface_data_->slave_node_row_map_bound()),
      snodecolmapbound_(interface_data_->slave_node_col_map_bound()),
      mnoderowmapnobound_(interface_data_->master_node_row_map_no_bound()),
      mnodecolmapnobound_(interface_data_->master_node_col_map_no_bound()),
      selerowmap_(interface_data_->slave_element_row_map()),
      selecolmap_(interface_data_->slave_element_col_map()),
      melerowmap_(interface_data_->master_element_row_map()),
      melecolmap_(interface_data_->master_element_col_map()),
      sdofrowmap_(interface_data_->slave_dof_row_map()),
      sdofcolmap_(interface_data_->slave_dof_col_map()),
      mdofrowmap_(interface_data_->master_dof_row_map()),
      mdofcolmap_(interface_data_->master_dof_col_map()),
      psdofrowmap_(interface_data_->non_redist_slave_dof_row_map()),
      plmdofmap_(interface_data_->non_redist_lm_dof_row_map()),
      lmdofmap_(interface_data_->lm_dof_row_map()),
      maxdofglobal_(interface_data_->max_dof_global()),
      searchalgo_(interface_data_->search_algorithm()),
      binarytree_(interface_data_->binary_tree()),
      searchparam_(interface_data_->search_param()),
      searchuseauxpos_(interface_data_->search_use_aux_pos()),
      inttime_interface_(interface_data_->int_time_interface()),
      nurbs_(interface_data_->is_nurbs()),
      ehl_(interface_data_->is_ehl())
{
  if (not interface_data_->is_init())
  {
    FOUR_C_THROW(
        "This constructor is only allowed for already initialized "
        "interface data containers!");
  }
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Mortar::Interface> Mortar::Interface::create(const int id, MPI_Comm comm,
    const int spatialDim, const Teuchos::ParameterList& imortar,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type)
{
  std::shared_ptr<Mortar::InterfaceDataContainer> interfaceData =
      std::make_shared<Mortar::InterfaceDataContainer>();

  return std::make_shared<Mortar::Interface>(
      interfaceData, id, comm, spatialDim, imortar, output_control, spatial_approximation_type);
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/07|
 *----------------------------------------------------------------------*/
Mortar::Interface::Interface(std::shared_ptr<InterfaceDataContainer> interfaceData, const int id,
    MPI_Comm comm, const int spatialDim, const Teuchos::ParameterList& imortar,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type)
    : interface_data_(std::move(interfaceData)),
      id_(interface_data_->id()),
      comm_(interface_data_->comm_ptr()),
      procmap_(interface_data_->proc_map()),
      redistributed_(interface_data_->is_redistributed()),
      idiscret_(interface_data_->i_discret()),
      dim_(interface_data_->n_dim()),
      imortar_(interface_data_->i_mortar()),
      shapefcn_(interface_data_->shape_fcn()),
      quadslave_(interface_data_->is_quad_slave()),
      oldnodecolmap_(interface_data_->old_node_col_map()),
      oldelecolmap_(interface_data_->old_ele_col_map()),
      snoderowmap_(interface_data_->s_node_row_map()),
      snodecolmap_(interface_data_->slave_node_col_map()),
      mnoderowmap_(interface_data_->master_node_row_map()),
      mnodecolmap_(interface_data_->master_node_col_map()),
      snoderowmapbound_(interface_data_->slave_node_row_map_bound()),
      snodecolmapbound_(interface_data_->slave_node_col_map_bound()),
      mnoderowmapnobound_(interface_data_->master_node_row_map_no_bound()),
      mnodecolmapnobound_(interface_data_->master_node_col_map_no_bound()),
      selerowmap_(interface_data_->slave_element_row_map()),
      selecolmap_(interface_data_->slave_element_col_map()),
      melerowmap_(interface_data_->master_element_row_map()),
      melecolmap_(interface_data_->master_element_col_map()),
      sdofrowmap_(interface_data_->slave_dof_row_map()),
      sdofcolmap_(interface_data_->slave_dof_col_map()),
      mdofrowmap_(interface_data_->master_dof_row_map()),
      mdofcolmap_(interface_data_->master_dof_col_map()),
      psdofrowmap_(interface_data_->non_redist_slave_dof_row_map()),
      plmdofmap_(interface_data_->non_redist_lm_dof_row_map()),
      lmdofmap_(interface_data_->lm_dof_row_map()),
      maxdofglobal_(interface_data_->max_dof_global()),
      searchalgo_(interface_data_->search_algorithm()),
      binarytree_(interface_data_->binary_tree()),
      searchparam_(interface_data_->search_param()),
      searchuseauxpos_(interface_data_->search_use_aux_pos()),
      inttime_interface_(interface_data_->int_time_interface()),
      nurbs_(interface_data_->is_nurbs()),
      ehl_(interface_data_->is_ehl())
{
  interface_data_->set_is_init(true);
  id_ = id;
  comm_ = comm;
  dim_ = spatialDim;
  imortar_.setParameters(imortar);
  quadslave_ = false;
  interface_data_->set_extend_ghosting(Teuchos::getIntegralValue<Inpar::Mortar::ExtendGhosting>(
      imortar.sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY"));
  searchalgo_ =
      Teuchos::getIntegralValue<Inpar::Mortar::SearchAlgorithm>(imortar, "SEARCH_ALGORITHM");
  searchparam_ = imortar.get<double>("SEARCH_PARAM");
  searchuseauxpos_ = imortar.get<bool>("SEARCH_USE_AUX_POS");
  nurbs_ = imortar.get<bool>("NURBS");

  if (n_dim() != 2 && n_dim() != 3) FOUR_C_THROW("Mortar problem must be 2D or 3D.");

  procmap_.clear();

  create_interface_discretization(output_control, spatial_approximation_type);
  set_shape_function_type();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::create_interface_discretization(
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type)
{
  MPI_Comm comm(get_comm());

  // Create name for mortar interface discretization
  std::stringstream dis_name;
  dis_name << "mortar_interface_" << id_;

  // Create the required type of discretization
  if (nurbs_)
  {
    idiscret_ = std::make_shared<Core::FE::Nurbs::NurbsDiscretization>(dis_name.str(), comm, dim_);

    /*
    Note: The NurbsDiscretization needs a Knotvector to be able to write output. This is probably
    the place, where the Knotvector of the mortar coupling surface needs to be inserted into the
    NurbsDiscretization. However, it's not clear how to compute that Knotvector, so we can't do it
    right now. In the end, it should be sufficient to extract the portion from the underlying volume
    discretization's knot vector that corresponds to the mortar interface.

    As a NurbsDiscretization can't write output for now, we don't do it and rather use the 'old'
    output style, where interface output is written by the underlying volume discretization.
    */
  }
  else
  {
    idiscret_ = std::make_shared<Core::FE::Discretization>(dis_name.str(), comm, dim_);
  }

  // Prepare discretization writer
  idiscret_->set_writer(std::make_shared<Core::IO::DiscretizationWriter>(
      *idiscret_, *output_control, spatial_approximation_type));
  FOUR_C_ASSERT(idiscret_->writer(), "Setup of discretization writer failed.");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::set_shape_function_type()
{
  auto shapefcn =
      Teuchos::getIntegralValue<Inpar::Mortar::ShapeFcn>(interface_params(), "LM_SHAPEFCN");
  switch (shapefcn)
  {
    case Inpar::Mortar::shape_dual:
    {
      shapefcn_ = Inpar::Mortar::shape_dual;
      break;
    }
    case Inpar::Mortar::shape_petrovgalerkin:
    {
      shapefcn_ = Inpar::Mortar::shape_petrovgalerkin;
      break;
    }
    case Inpar::Mortar::shape_standard:
    {
      shapefcn_ = Inpar::Mortar::shape_standard;
      break;
    }
    default:
    {
      FOUR_C_THROW(
          "Invalid shape function type. Interface must either have dual or standard shape "
          "functions.");
      break;
    }
  }
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/07|
 *----------------------------------------------------------------------*/
std::ostream& operator<<(std::ostream& os, const Mortar::Interface& interface)
{
  interface.print(os);
  return os;
}

/*----------------------------------------------------------------------*
 |  print interface (public)                                 mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::print(std::ostream& os) const
{
  if (Core::Communication::my_mpi_rank(get_comm()) == 0)
  {
    os << "\nMortar Interface Id " << id_ << std::endl;
    os << "Mortar Interface discretization:" << std::endl;
  }
  os << discret();
}

/*----------------------------------------------------------------------*
 |  check if interface is fill_complete (public)              mwgee 10/07|
 *----------------------------------------------------------------------*/
bool Mortar::Interface::filled() const { return idiscret_->filled(); }

/*----------------------------------------------------------------------*
 |  print parallel distribution (public)                      popp 06/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::print_parallel_distribution() const
{
  // how many processors
  const int numproc = Core::Communication::num_mpi_ranks(discret().get_comm());

  // only print parallel distribution if numproc > 1
  if (numproc > 1)
  {
    const int myrank = Core::Communication::my_mpi_rank(discret().get_comm());

    std::vector<int> my_n_nodes(numproc, 0);
    std::vector<int> n_nodes(numproc, 0);
    std::vector<int> my_n_ghostnodes(numproc, 0);
    std::vector<int> n_ghostnodes(numproc, 0);
    std::vector<int> my_n_elements(numproc, 0);
    std::vector<int> n_elements(numproc, 0);
    std::vector<int> my_n_ghostele(numproc, 0);
    std::vector<int> n_ghostele(numproc, 0);
    std::vector<int> my_s_nodes(numproc, 0);
    std::vector<int> s_nodes(numproc, 0);
    std::vector<int> my_s_ghostnodes(numproc, 0);
    std::vector<int> s_ghostnodes(numproc, 0);
    std::vector<int> my_s_elements(numproc, 0);
    std::vector<int> s_elements(numproc, 0);
    std::vector<int> my_s_ghostele(numproc, 0);
    std::vector<int> s_ghostele(numproc, 0);
    std::vector<int> my_m_nodes(numproc, 0);
    std::vector<int> m_nodes(numproc, 0);
    std::vector<int> my_m_elements(numproc, 0);
    std::vector<int> m_elements(numproc, 0);
    std::vector<int> my_m_ghostnodes(numproc, 0);
    std::vector<int> m_ghostnodes(numproc, 0);
    std::vector<int> my_m_ghostele(numproc, 0);
    std::vector<int> m_ghostele(numproc, 0);

    my_n_nodes[myrank] = discret().num_my_row_nodes();
    my_n_ghostnodes[myrank] = discret().num_my_col_nodes() - my_n_nodes[myrank];
    my_n_elements[myrank] = discret().num_my_row_elements();
    my_n_ghostele[myrank] = discret().num_my_col_elements() - my_n_elements[myrank];

    my_s_nodes[myrank] = snoderowmap_->num_my_elements();
    my_s_ghostnodes[myrank] = snodecolmap_->num_my_elements() - my_s_nodes[myrank];
    my_s_elements[myrank] = selerowmap_->num_my_elements();
    my_s_ghostele[myrank] = selecolmap_->num_my_elements() - my_s_elements[myrank];

    my_m_nodes[myrank] = mnoderowmap_->num_my_elements();
    my_m_ghostnodes[myrank] = mnodecolmap_->num_my_elements() - my_m_nodes[myrank];
    my_m_elements[myrank] = melerowmap_->num_my_elements();
    my_m_ghostele[myrank] = melecolmap_->num_my_elements() - my_m_elements[myrank];

    // adapt output for redundant master or all redundant case
    if (interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::redundant_master)
    {
      my_m_ghostnodes[myrank] = mnoderowmap_->num_global_elements() - my_m_nodes[myrank];
      my_m_ghostele[myrank] = melerowmap_->num_global_elements() - my_m_elements[myrank];
    }
    else if (interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::redundant_all)
    {
      my_m_ghostnodes[myrank] = mnoderowmap_->num_global_elements() - my_m_nodes[myrank];
      my_m_ghostele[myrank] = melerowmap_->num_global_elements() - my_m_elements[myrank];
      my_s_ghostnodes[myrank] = snoderowmap_->num_global_elements() - my_s_nodes[myrank];
      my_s_ghostele[myrank] = selerowmap_->num_global_elements() - my_s_elements[myrank];
    }

    n_nodes = Core::Communication::sum_all(my_n_nodes, discret().get_comm());
    n_ghostnodes = Core::Communication::sum_all(my_n_ghostnodes, discret().get_comm());
    n_elements = Core::Communication::sum_all(my_n_elements, discret().get_comm());
    n_ghostele = Core::Communication::sum_all(my_n_ghostele, discret().get_comm());

    s_nodes = Core::Communication::sum_all(my_s_nodes, discret().get_comm());
    s_ghostnodes = Core::Communication::sum_all(my_s_ghostnodes, discret().get_comm());
    s_elements = Core::Communication::sum_all(my_s_elements, discret().get_comm());
    s_ghostele = Core::Communication::sum_all(my_s_ghostele, discret().get_comm());

    m_nodes = Core::Communication::sum_all(my_m_nodes, discret().get_comm());
    m_ghostnodes = Core::Communication::sum_all(my_m_ghostnodes, discret().get_comm());
    m_elements = Core::Communication::sum_all(my_m_elements, discret().get_comm());
    m_ghostele = Core::Communication::sum_all(my_m_ghostele, discret().get_comm());

    if (myrank == 0)
    {
      std::cout << std::endl;
      std::cout << "  discretization: " << discret().name() << std::endl;

      // Compute and print statistics
      {
        std::cout << "\n"
                  << "    Statistics of parallel distribution across " << numproc
                  << " ranks:" << std::endl;
        printf(
            "    "
            "+----------------------+----------------+----------------+-----------------+----------"
            "--------+\n");
        printf(
            "    | Type                 | min over procs | max over procs | mean over procs | "
            "max-to-min ratio |\n");
        printf(
            "    "
            "+----------------------+----------------+----------------+-----------------+----------"
            "--------+\n");
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "nodes (s+m)", n_nodes, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "ghost nodes (s+m)", n_ghostnodes, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "elements (s+m)", n_elements, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "ghost elements (s+m)", n_ghostele, myrank == 0);
        printf(
            "    "
            "+----------------------+----------------+----------------+-----------------+----------"
            "--------+\n");
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "nodes (s)", s_nodes, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "ghost nodes (s)", s_ghostnodes, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "elements (s)", s_elements, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "ghost elements (s)", s_ghostele, myrank == 0);
        printf(
            "    "
            "+----------------------+----------------+----------------+-----------------+----------"
            "--------+\n");
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "nodes (m)", m_nodes, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "ghost nodes (m)", m_ghostnodes, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "elements (m)", m_elements, myrank == 0);
        Mortar::InterfaceUtils::compute_and_print_row_of_parallel_distribution_statisctics(
            "ghost elements (m)", m_ghostele, myrank == 0);
        printf(
            "    "
            "+----------------------+----------------+----------------+-----------------+----------"
            "--------+\n");
      }

      // Print details of parallel distribution for each proc if requested by the user
      const bool printDetails =
          interface_params().sublist("PARALLEL REDISTRIBUTION").get<bool>("PRINT_DISTRIBUTION");
      if (printDetails)
      {
        std::cout << std::endl;
        std::cout << "    Detailed distribution:" << std::endl;
        printf("    +-----+-----------------+--------------+-----------------+--------------+\n");
        printf("    | PID |   n_rownodes    | n_ghostnodes |  n_rowelements  |  n_ghostele  |\n");
        printf("    +-----+-----------------+--------------+-----------------+--------------+\n");
        for (int npid = 0; npid < numproc; ++npid)
        {
          printf("    | %3d | Total %9d | %12d | Total %9d | %12d |\n", npid, n_nodes[npid],
              n_ghostnodes[npid], n_elements[npid], n_ghostele[npid]);
          printf("    |     | Slave %9d | %12d | Slave %9d | %12d |\n", s_nodes[npid],
              s_ghostnodes[npid], s_elements[npid], s_ghostele[npid]);
          printf("    |     | Master %8d | %12d | Master %8d | %12d |\n", m_nodes[npid],
              m_ghostnodes[npid], m_elements[npid], m_ghostele[npid]);
          printf("    +-----+-----------------+--------------+-----------------+--------------+\n");
        }
      }
    }
  }
}

/*----------------------------------------------------------------------*
 |  add mortar node (public)                                 mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::add_mortar_node(std::shared_ptr<Mortar::Node> mrtrnode)
{
  idiscret_->add_node(mrtrnode->x(), mrtrnode->id(), mrtrnode);
}

/*----------------------------------------------------------------------*
 |  add mortar element (public)                              mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::add_mortar_element(std::shared_ptr<Mortar::Element> mrtrele)
{
  // check for quadratic 2d slave elements to be modified
  if (mrtrele->is_slave() && (mrtrele->shape() == Core::FE::CellType::line3 ||
                                 mrtrele->shape() == Core::FE::CellType::nurbs3))
    quadslave_ = true;

  // check for quadratic 3d slave elements to be modified
  if (mrtrele->is_slave() && (mrtrele->shape() == Core::FE::CellType::quad9 ||
                                 mrtrele->shape() == Core::FE::CellType::quad8 ||
                                 mrtrele->shape() == Core::FE::CellType::tri6 ||
                                 mrtrele->shape() == Core::FE::CellType::nurbs8 ||
                                 mrtrele->shape() == Core::FE::CellType::nurbs9))
    quadslave_ = true;

  idiscret_->add_element(mrtrele);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::fill_complete_new(const bool isFinalParallelDistribution, const int maxdof)
{
  FOUR_C_THROW("Not implemented for meshtying.");
}

/*----------------------------------------------------------------------*
 |  finalize construction of interface (public)              mwgee 10/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::fill_complete(
    const std::map<std::string, std::shared_ptr<Core::FE::Discretization>>& discretization_map,
    const Teuchos::ParameterList& binning_params,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type,
    const bool isFinalParallelDistribution, const int maxdof, const double meanVelocity)
{
  TEUCHOS_FUNC_TIME_MONITOR("Mortar::Interface::fill_complete");

  // store maximum global dof ID handed in
  // this ID is later needed when setting up the Lagrange multiplier
  // dof map, which of course must not overlap with existing dof ranges
  maxdofglobal_ = maxdof;

  // we'd like to call idiscret_.fill_complete(true,false,false) but this
  // will assign all nodes new degrees of freedom which we don't want.
  // We would like to use the degrees of freedom that were stored in the
  // mortar nodes. To do so, we have to create and set our own
  // version of a DofSet class before we call fill_complete on the
  // interface discretization.
  // Our special dofset class will not assign new dofs but will assign the
  // dofs stored in the nodes.
  {
    std::shared_ptr<Mortar::DofSet> mrtrdofset = std::make_shared<Mortar::DofSet>();
    discret().replace_dof_set(mrtrdofset);
    // do not assign dofs yet, we'll do this below after
    // shuffling around of nodes and elements (saves time)
    discret().fill_complete(Core::FE::OptionsFillComplete::none());
  }

  // check whether crosspoints / edge nodes shall be considered or not
  initialize_cross_points();

  // check for const/linear interpolation of 2D/3D quadratic Lagrange multipliers
  initialize_lag_mult_const();
  initialize_lag_mult_lin();

  // check/init corner/edge modification
  initialize_corner_edge();

  // later we might export node and element column map to extended or even FULL overlap,
  // thus store the standard column maps first
  // get standard nodal column map (overlap=1)
  oldnodecolmap_ = std::make_shared<Core::LinAlg::Map>(*(discret().node_col_map()));
  // get standard element column map (overlap=1)
  oldelecolmap_ = std::make_shared<Core::LinAlg::Map>(*(discret().element_col_map()));

  extend_interface_ghosting(isFinalParallelDistribution, meanVelocity, binning_params,
      output_control, spatial_approximation_type);

  // make sure discretization is complete
  discret().fill_complete({
      .assign_degrees_of_freedom = isFinalParallelDistribution,
      .init_elements = false,
      .do_boundary_conditions = false,
  });

  // ghost also parent elements according to the ghosting strategy of the interface (atm just for
  // poro)
  if (interface_data_->is_poro())
  {
    if (interface_data_->poro_type() == Inpar::Mortar::poroscatra)
      PoroElastScaTra::Utils::create_volume_ghosting(discret());
    else
      PoroElast::Utils::create_volume_ghosting(discret());
  }
  else if (imortar_.isParameter("STRATEGY"))
  {
    if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(imortar_, "ALGORITHM") ==
        Inpar::Mortar::algorithm_gpts)
      create_volume_ghosting(discretization_map);
  }

  // need row and column maps of slave and master nodes / elements / dofs
  // separately so we can easily address them
  update_master_slave_sets();

  // initialize node and element data container
  initialize_data_container();

  // Communicate quadslave status among ALL processors
  communicate_quad_slave_status_among_all_procs();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::communicate_quad_slave_status_among_all_procs()
{
  int localstatus = static_cast<int>(quadslave_);
  int globalstatus = 0;
  globalstatus = Core::Communication::sum_all(localstatus, get_comm());
  quadslave_ = static_cast<bool>(globalstatus);
}

/*----------------------------------------------------------------------*
 |  Check and initialize corner/edge contact                 farah 07/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::initialize_corner_edge()
{
  // if linear LM for quad displacements return!
  // TODO: this case needs a special treatment
  bool lagmultlin = (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                         interface_params(), "LM_QUAD") == Inpar::Mortar::lagmult_lin);

  if (lagmultlin) return;

  for (int i = 0; i < (discret().node_row_map())->num_my_elements(); ++i)
  {
    // static_cast to the corresponding mortar/contact/friction/... node
    // or element would be enough in all places
    // performance loss is negligible when using a dynamic_cast instead
    // but safety is increased enormously
    auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_row_node(i));

    // remove bound/corner/edge flag for master nodes!
    if (!node->is_slave() && node->is_on_corner()) node->set_on_corner() = false;
    if (!node->is_slave() && node->is_on_edge()) node->set_on_edge() = false;

    // candidates are slave nodes with only 1 adjacent Mortar::Element
    if (node->is_slave() && node->is_on_corner_edge())
    {
      node->set_slave() = false;
    }
  }
}


/*----------------------------------------------------------------------*
 |  Check and initialize cross points                        farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::initialize_cross_points()
{
  // check whether crosspoints / edge nodes shall be considered or not
  bool crosspoints = interface_params().get<bool>("CROSSPOINTS");

  // modify crosspoints / edge nodes
  if (crosspoints)
  {
    // only applicable for 2D problems up to now
    if (n_dim() == 3)
      FOUR_C_THROW("Crosspoint / edge node modification not yet implemented for 3D problems.");

    // Detect relevant nodes on slave side
    const int numRowNodes = discret().node_row_map()->num_my_elements();
    for (int i = 0; i < numRowNodes; ++i)
    {
      // static_cast to the corresponding mortar/contact/friction/... node
      // or element would be enough in all places
      // performance loss is negligible when using a dynamic_cast instead
      // but safety is increased enormously
      auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_row_node(i));

      // candidates are slave nodes with only 1 adjacent Mortar::Element
      if (node->is_slave() && node->num_element() == 1)
      {
        // case1: linear shape functions, boundary nodes already found
        if ((node->adjacent_elements()[0].user_element())->num_node() == 2)
        {
          node->set_bound() = true;
          node->set_slave() = false;
        }
        // case2: quad. shape functions, middle nodes must be sorted out
        else if (node->id() != (node->adjacent_elements()[0].user_element())->node_ids()[2])
        {
          node->set_bound() = true;
          node->set_slave() = false;
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Check and initialize for lin lagmult interpolation       farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::initialize_lag_mult_lin()
{
  // check for linear interpolation of 2D/3D quadratic Lagrange multipliers
  bool lagmultlin = (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                         interface_params(), "LM_QUAD") == Inpar::Mortar::lagmult_lin);

  // modify nodes accordingly
  if (lagmultlin)
  {
    // modified treatment of vertex nodes and edge nodes
    // detect middle nodes (quadratic nodes) on slave side
    // set status of middle nodes -> MASTER
    // set status of vertex nodes -> SLAVE

    // loop over all elements
    for (int i = 0; i < discret().node_row_map()->num_my_elements(); ++i)
    {
      // get node and cast to cnode
      auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_row_node(i));

      // candidates are slave nodes with shape line3 (2D), tri6 and quad8/9 (3D)
      if (node->is_slave())
      {
        // search the first adjacent element
        Core::FE::CellType shape = (node->adjacent_elements()[0].user_element())->shape();

        // which discretization type
        switch (shape)
        {
          // line3 contact elements (= quad8/9 or tri6 discretizations)
          case Core::FE::CellType::line3:
          {
            // case1: vertex nodes remain SLAVE
            if (node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[0] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[1])
            {
              // do nothing
            }

            // case2: middle nodes must be set to MASTER
            else
            {
              node->set_bound() = true;
              node->set_slave() = false;
            }

            break;
          }

            // tri6 contact elements (= tet10 discretizations)
          case Core::FE::CellType::tri6:
          {
            // case1: vertex nodes remain SLAVE
            if (node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[0] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[1] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[2])
            {
              // do nothing
            }

            // case2: middle nodes must be set to MASTER
            else
            {
              node->set_bound() = true;
              node->set_slave() = false;
            }

            break;
          }

            // quad8 contact elements (= hex20 discretizations)
          case Core::FE::CellType::quad8:
            // quad9 contact elements (= hex27 discretizations)
          case Core::FE::CellType::quad9:
          {
            // case1: vertex nodes remain SLAVE
            if (node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[0] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[1] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[2] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[3])
            {
              // do nothing
            }

            // case2: middle nodes must be set to MASTER
            else
            {
              node->set_bound() = true;
              node->set_slave() = false;
            }

            break;
          }

            // other cases
          default:
          {
            FOUR_C_THROW(
                "Lin/Lin interpolation of LM only for line3/tri6/quad8/quad9 mortar elements");
            break;
          }
        }  // switch(Shape)
      }  // if (IsSlave())
    }  // for-loop
  }
}


/*----------------------------------------------------------------------*
 |  Check and initialize for const lagmult interpolation     seitz 09/17|
 *----------------------------------------------------------------------*/
void Mortar::Interface::initialize_lag_mult_const()
{
  if ((Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(interface_params(), "LM_QUAD") ==
          Inpar::Mortar::lagmult_const))
  {
    // modified treatment slave side nodes:
    // only the center-node carries LM

    // loop over all elements
    for (int i = 0; i < discret().node_row_map()->num_my_elements(); ++i)
    {
      // get node and cast to cnode
      auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_row_node(i));

      // candidates are slave nodes with shape line3 (2D), tri6 and quad8/9 (3D)
      if (node->is_slave())
      {
        // search the first adjacent element
        Core::FE::CellType shape = (node->adjacent_elements()[0].user_element())->shape();

        // which discretization type
        switch (shape)
        {
          // line3 contact elements (= quad8/9 or tri6 discretizations)
          case Core::FE::CellType::line3:
          {
            // case1: vertex nodes must be set to MASTER
            if (node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[0] ||
                node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[1])
            {
              node->set_bound() = true;
              node->set_slave() = false;
            }

            // case2: middle nodes remain SLAVE
            else
            {
              // do nothing
            }

            break;
          }
          case Core::FE::CellType::quad9:
            if (node->id() == (node->adjacent_elements()[0].user_element())->node_ids()[8])
            {
              // do nothing
            }
            else
            {
              node->set_bound() = true;
              node->set_slave() = false;
            }
            break;

            // other cases
          default:
          {
            FOUR_C_THROW(
                "Lin/Lin interpolation of LM only for line3/tri6/quad8/quad9 mortar elements");
            break;
          }
        }  // switch(Shape)
      }  // if (IsSlave())
    }  // for-loop
  }
}


/*----------------------------------------------------------------------*
 |  Initialize Data Container for nodes and elements         farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::initialize_data_container()
{
  // initialize node data container
  // (include slave side boundary nodes / crosspoints)
  const int numMySlaveColumnNodes = slave_col_nodes_bound()->num_my_elements();
  for (int i = 0; i < numMySlaveColumnNodes; ++i)
  {
    int gid = slave_col_nodes_bound()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid {}", gid);
    auto* mnode = dynamic_cast<Node*>(node);

    //********************************************************
    // NOTE: depending on which kind of node this really is,
    // i.e. mortar, contact or friction node, several derived
    // versions of the initialize_data_container() methods will
    // be called here, apart from the base class version.
    //********************************************************

    // initialize container if not yet initialized before
    mnode->initialize_data_container();
    if (interface_data_->is_poro())  // initialize just for poro contact case!
      mnode->initialize_poro_data_container();
    if (ehl_) mnode->initialize_ehl_data_container();
  }
  if (interface_data_
          ->is_poro())  // as velocities of structure and fluid exist also on master nodes!!!
  {
    const std::shared_ptr<Core::LinAlg::Map> masternodes =
        Core::LinAlg::allreduce_e_map(*(master_row_nodes()));
    // initialize poro node data container for master nodes!!!
    for (int i = 0; i < masternodes->num_my_elements(); ++i)
    {
      int gid = masternodes->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid {}", gid);
      auto* mnode = dynamic_cast<Node*>(node);

      // ATM just implemented for ContactNode ... otherwise error!!!

      // initialize container if not yet initialized before
      mnode->initialize_poro_data_container();
    }
  }

  // initialize element data container
  const int numMySlaveColumnElements = slave_col_elements()->num_my_elements();
  for (int i = 0; i < numMySlaveColumnElements; ++i)
  {
    int gid = slave_col_elements()->gid(i);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
    auto* mele = dynamic_cast<Mortar::Element*>(ele);

    // initialize container if not yet initialized before
    mele->initialize_data_container();
  }

  if (interface_data_->is_poro())
  {
    // initialize master element data container
    for (int i = 0; i < master_col_elements()->num_my_elements(); ++i)
    {
      int gid = master_col_elements()->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
      auto* mele = dynamic_cast<Mortar::Element*>(ele);

      // initialize container if not yet initialized before
      mele->initialize_data_container();
    }
  }

  if (interface_params().isParameter("ALGORITHM"))
  {
    if (Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(interface_params(), "ALGORITHM") ==
        Inpar::Mortar::algorithm_gpts)
    {
      const int numMyMasterColumnElements = master_col_elements()->num_my_elements();
      for (int i = 0; i < numMyMasterColumnElements; ++i)
        dynamic_cast<Mortar::Element*>(discret().g_element(master_col_elements()->gid(i)))
            ->initialize_data_container();
    }
  }
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::Binstrategy::BinningStrategy> Mortar::Interface::setup_binning_strategy(
    Teuchos::ParameterList binning_params, const double meanVelocity,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type)
{
  // Initialize eXtendedAxisAlignedBoundingBox (XAABB)
  Core::LinAlg::Matrix<3, 2> XAABB(Core::LinAlg::Initialization::uninitialized);
  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    XAABB(dim, 0) = +1.0e12;
    XAABB(dim, 1) = -1.0e12;
  }

  // loop all slave nodes and merge XAABB with their eXtendedAxisAlignedBoundingBox
  for (int lid = 0; lid < slave_col_nodes()->num_my_elements(); ++lid)
  {
    int gid = slave_col_nodes()->gid(lid);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node)
      FOUR_C_THROW("Cannot find node with gid {} in discretization '{}'.", gid, discret().name());
    auto* mtrnode = dynamic_cast<Mortar::Node*>(node);

    for (unsigned int dim = 0; dim < 3; ++dim)
    {
      XAABB(dim, 0) = std::min(XAABB(dim, 0), mtrnode->xspatial()[dim] - MORTARPROJLIM);
      XAABB(dim, 1) = std::max(XAABB(dim, 1), mtrnode->xspatial()[dim] + MORTARPROJLIM);
    }
  }

  // local bounding box
  std::array<double, 3> locmin = {XAABB(0, 0), XAABB(1, 0), XAABB(2, 0)};
  std::array<double, 3> locmax = {XAABB(0, 1), XAABB(1, 1), XAABB(2, 1)};

  // global bounding box
  std::array<double, 3> globmin{};
  std::array<double, 3> globmax{};

  // do the necessary communication
  globmin = Core::Communication::min_all(locmin, get_comm());
  globmax = Core::Communication::max_all(locmax, get_comm());

  // compute cutoff radius:
  double global_slave_max_edge_size = -1.0;
  for (int lid = 0; lid < slave_col_elements()->num_my_elements(); ++lid)
  {
    int gid = slave_col_elements()->gid(lid);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele)
      FOUR_C_THROW(
          "Cannot find element with gid {} in discretization '{}'.", gid, discret().name());
    auto* mtrele = dynamic_cast<Mortar::Element*>(ele);

    // to be thought about, whether this is enough (safety = 2??)
    double slave_max_edge_size = mtrele->max_edge_size();
    global_slave_max_edge_size = std::max(slave_max_edge_size, global_slave_max_edge_size);
  }

  double cutoff = -1.0;
  cutoff = Core::Communication::max_all(global_slave_max_edge_size, get_comm());

  // extend cutoff based on problem interface velocity
  // --> only for contact problems
  if (meanVelocity >= 1e-12)
  {
    const double dt = interface_params().get<double>("TIMESTEP");
    cutoff = cutoff + 2 * dt * meanVelocity;
  }

  // increase XAABB by 2x cutoff radius
  std::stringstream domain_bounding_box_stream;
  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    domain_bounding_box_stream << globmin[dim] - cutoff << " ";
  }
  for (unsigned int dim = 0; dim < 3; ++dim)
  {
    domain_bounding_box_stream << globmax[dim] + cutoff << " ";
  }

  binning_params.set<double>("BIN_SIZE_LOWER_BOUND", cutoff);
  binning_params.set<std::string>("DOMAINBOUNDINGBOX", domain_bounding_box_stream.str());
  Core::Utils::add_enum_class_to_parameter_list<Core::FE::ShapeFunctionType>(
      "spatial_approximation_type", spatial_approximation_type, binning_params);

  // Special case for mortar nodes that store their displacements themselves
  const auto determine_relevant_points =
      [](const Core::FE::Discretization& discret, const Core::Elements::Element& ele,
          std::shared_ptr<const Core::LinAlg::Vector<double>>) -> std::vector<std::array<double, 3>>
  {
    std::vector<std::array<double, 3>> relevant_points;
    const auto* mortar_ele = dynamic_cast<const Mortar::Element*>(&ele);
    FOUR_C_ASSERT_ALWAYS(mortar_ele, "Element is not a Mortar::Element.");

    for (int j = 0; j < mortar_ele->num_node(); ++j)
    {
      const Core::Nodes::Node* node = mortar_ele->nodes()[j];
      const double* coords = dynamic_cast<const Mortar::Node*>(node)->xspatial();
      relevant_points.push_back({coords[0], coords[1], coords[2]});
    }
    return relevant_points;
  };

  std::shared_ptr<Core::Binstrategy::BinningStrategy> binningstrategy =
      std::make_shared<Core::Binstrategy::BinningStrategy>(binning_params, output_control,
          get_comm(), Core::Communication::my_mpi_rank(get_comm()), nullptr,
          determine_relevant_points);
  return binningstrategy;
}


/*----------------------------------------------------------------------*
 |  redistribute interface (public)                           popp 08/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::redistribute()
{
  std::stringstream ss;
  ss << "Mortar::Interface::redistribute of '" << discret().name() << "'";
  TEUCHOS_FUNC_TIME_MONITOR(ss.str());

  const Teuchos::ParameterList& mortarParallelRedistParams =
      interface_params().sublist("PARALLEL REDISTRIBUTION");

  // make sure we are supposed to be here
  if (Teuchos::getIntegralValue<Inpar::Mortar::ParallelRedist>(mortarParallelRedistParams,
          "PARALLEL_REDIST") == Inpar::Mortar::ParallelRedist::redist_none)
    FOUR_C_THROW("You are not supposed to be here...");

  // some local variables
  MPI_Comm comm(get_comm());
  const int myrank = Core::Communication::my_mpi_rank(comm);
  const int numproc = Core::Communication::num_mpi_ranks(comm);
  Teuchos::Time time("", true);

  // vector containing all proc ids
  std::vector<int> allproc(numproc);
  for (int i = 0; i < numproc; ++i) allproc[i] = i;

  // we need an arbitrary preliminary element row map
  Core::LinAlg::Map sroweles(*slave_row_elements());
  Core::LinAlg::Map mroweles(*master_row_elements());

  //**********************************************************************
  // (1) PREPARATIONS decide how many procs are used
  //**********************************************************************
  // first we assume that all procs will be used
  int sproc = numproc;
  int mproc = numproc;

  // minimum number of elements per proc
  int minele = mortarParallelRedistParams.get<int>("MIN_ELEPROC");

  // Max. relative imbalance between subdomain sizes
  const double imbalance_tol = mortarParallelRedistParams.get<double>("IMBALANCE_TOL");

  // calculate real number of procs to be used
  if (minele > 0)
  {
    sproc = static_cast<int>((sroweles.num_global_elements()) / minele);
    mproc = static_cast<int>((mroweles.num_global_elements()) / minele);
    if (sroweles.num_global_elements() < 2 * minele) sproc = 1;
    if (mroweles.num_global_elements() < 2 * minele) mproc = 1;
    if (sproc > numproc) sproc = numproc;
    if (mproc > numproc) mproc = numproc;
  }

  // print message
  if (!myrank)
  {
    std::cout << "\nRedistributing interface '" << discret().name() << "' .........\n";
    std::cout << "Procs used for redistribution: " << sproc << " / " << mproc << " (S / M)\n";
  }

  //**********************************************************************
  // (2) SLAVE redistribution
  //**********************************************************************
  std::shared_ptr<Core::LinAlg::Map> srownodes = nullptr;
  std::shared_ptr<Core::LinAlg::Map> scolnodes = nullptr;

  {
    std::stringstream ss_slave;
    ss_slave << "Mortar::Interface::redistribute of '" << discret().name() << "' (slave)";
    TEUCHOS_FUNC_TIME_MONITOR(ss_slave.str());

    std::shared_ptr<const Core::LinAlg::Graph> snodegraph =
        Core::Rebalance::build_graph(*idiscret_, sroweles);

    Teuchos::ParameterList rebalanceParams;
    rebalanceParams.set<std::string>("num_global_parts", std::to_string(sproc));
    rebalanceParams.set<std::string>("imbalance_tolerance", std::to_string(imbalance_tol));
    rebalanceParams.set<std::string>("algorithm", "parmetis");

    std::tie(srownodes, scolnodes) =
        Core::Rebalance::rebalance_node_maps(*snodegraph, rebalanceParams);
  }

  //**********************************************************************
  // (3) MASTER redistribution
  //**********************************************************************
  std::shared_ptr<Core::LinAlg::Map> mrownodes = nullptr;
  std::shared_ptr<Core::LinAlg::Map> mcolnodes = nullptr;

  {
    std::stringstream ss_master;
    ss_master << "Mortar::Interface::redistribute of '" << discret().name() << "' (master)";
    TEUCHOS_FUNC_TIME_MONITOR(ss_master.str());

    redistribute_master_side(mrownodes, mcolnodes, mroweles, comm, mproc, imbalance_tol);
  }

  //**********************************************************************
  // (4) Merge global interface node row and column map
  //**********************************************************************
  // merge node maps from slave and master parts
  std::shared_ptr<Core::LinAlg::Map> rownodes =
      Core::LinAlg::merge_map(srownodes, mrownodes, false);
  std::shared_ptr<Core::LinAlg::Map> colnodes =
      Core::LinAlg::merge_map(scolnodes, mcolnodes, false);

  //**********************************************************************
  // (5) Get partitioning information into discretization
  //**********************************************************************
  {
    std::stringstream ss_comm;
    ss_comm << "Mortar::Interface::redistribute of '" << discret().name() << "' (communicate)";
    TEUCHOS_FUNC_TIME_MONITOR(ss_comm.str());

    // build reasonable element maps from the already valid and final node maps
    // (note that nothing is actually redistributed in here)
    auto const& [roweles, coleles] = discret().build_element_row_column(*rownodes, *colnodes);

    discret().redistribute(
        {*rownodes, *colnodes}, {*roweles, *coleles}, {.fill_complete = std::nullopt});
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::Interface::redistribute_master_side(std::shared_ptr<Core::LinAlg::Map>& rownodes,
    std::shared_ptr<Core::LinAlg::Map>& colnodes, Core::LinAlg::Map& roweles, MPI_Comm comm,
    const int parts, const double imbalance) const
{
  // call parallel redistribution
  std::shared_ptr<const Core::LinAlg::Graph> nodegraph =
      Core::Rebalance::build_graph(*idiscret_, roweles);

  Teuchos::ParameterList rebalanceParams;

  rebalanceParams.set<std::string>("num_global_parts", std::to_string(parts));
  rebalanceParams.set<std::string>("imbalance_tolerance", std::to_string(imbalance));
  rebalanceParams.set<std::string>("algorithm", "parmetis");

  std::tie(rownodes, colnodes) = Core::Rebalance::rebalance_node_maps(*nodegraph, rebalanceParams);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_parallel_layout_and_data_structures(const bool perform_rebalancing,
    const bool enforce_ghosting_update, const int maxdof, const double meanVelocity)
{
  FOUR_C_THROW("Not implemented for meshtying.");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::extend_interface_ghosting_safely(const double meanVelocity)
{
  FOUR_C_THROW("Not implemented for meshtying.");
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::extend_interface_ghosting(const bool isFinalParallelDistribution,
    const double meanVelocity, const Teuchos::ParameterList& binning_params,
    std::shared_ptr<Core::IO::OutputControl> output_control,
    const Core::FE::ShapeFunctionType spatial_approximation_type)
{
  //*****REDUNDANT SLAVE AND MASTER STORAGE*****
  if (interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::redundant_all)
  {
    // to ease our search algorithms we'll afford the luxury to ghost all nodes
    // on all processors. To do so, we'll take the node row map and export it to
    // full overlap. Then we export the discretization to full overlap column map.
    // This way, also all mortar elements will be fully ghosted on all processors.
    // Note that we'll do ghosting NOT ONLY on procs that do own or ghost any of the
    // nodes in the natural distribution of idiscret_, but really on ALL procs.
    // This makes dynamic redistribution easier!

    //**********************************************************************
    // IMPORTANT NOTE:
    // In an older code version, we only did ghosting on procs that own or ghost
    // any of the interface nodes or elements  in the natural distr. of idiscret_.
    // The corresponding code lines for creating this proc list are:
    //
    // std::vector<int> stproc;
    // if (oldnodecolmap_->NumMyElements() || oldelecolmap_->NumMyElements())
    //   stproc.push_back(Core::Communication::my_mpi_rank(Comm()));
    // std::vector<int> rtproc;
    // Core::LinAlg::gather<int>(stproc,rtproc,Comm().NumProc(),allproc.data(),Comm());
    //
    // In this case, we use "rtproc" instead of "allproc" afterwards, i.e. when
    // the node gids and element gids are gathered among procs.
    //**********************************************************************

    // we want to do full ghosting on all procs
    std::vector<int> allproc(Core::Communication::num_mpi_ranks(get_comm()));
    for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) allproc[i] = i;

    // fill my own row node ids
    const Core::LinAlg::Map* noderowmap = discret().node_row_map();
    std::vector<int> sdata(noderowmap->num_my_elements());
    for (int i = 0; i < noderowmap->num_my_elements(); ++i) sdata[i] = noderowmap->gid(i);

    // gather all gids of nodes redundantly
    std::vector<int> rdata;
    Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

    // build completely overlapping map of nodes (on ALL processors)
    Core::LinAlg::Map newnodecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
    sdata.clear();
    rdata.clear();

    // fill my own row element ids
    const Core::LinAlg::Map* elerowmap = discret().element_row_map();
    sdata.resize(elerowmap->num_my_elements());
    for (int i = 0; i < elerowmap->num_my_elements(); ++i) sdata[i] = elerowmap->gid(i);

    // gather all gids of elements redundantly
    rdata.resize(0);
    Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

    // build complete overlapping map of elements (on ALL processors)
    Core::LinAlg::Map newelecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
    sdata.clear();
    rdata.clear();
    allproc.clear();

    // redistribute the discretization of the interface according to the
    // new column layout
    discret().export_column_nodes(newnodecolmap);
    discret().export_column_elements(newelecolmap);
  }

  //*****ONLY REDUNDANT MASTER STORAGE*****
  else if (interface_data_->get_extend_ghosting() ==
           Inpar::Mortar::ExtendGhosting::redundant_master)
  {
    // to ease our search algorithms we'll afford the luxury to ghost all master
    // nodes on all processors. To do so, we'll take the master node row map and
    // export it to full overlap. Then we export the discretization to partially
    // full overlap column map. This way, also all master elements will be fully
    // ghosted on all processors. Note that we'll do ghosting NOT ONLY on procs
    // that do own or ghost any nodes in the natural distribution of idiscret_,
    // but really on ALL procs. This makes dynamic redistribution easier!

    //**********************************************************************
    // IMPORTANT NOTE:
    // In an older code version, we only did ghosting on procs that own or ghost
    // any of the interface nodes or elements  in the natural distr. of idiscret_.
    // The corresponding code lines for creating this proc list are:
    //
    // std::vector<int> stproc;
    // if (oldnodecolmap_->NumMyElements() || oldelecolmap_->NumMyElements())
    //   stproc.push_back(Core::Communication::my_mpi_rank(Comm()));
    // std::vector<int> rtproc;
    // Core::LinAlg::gather<int>(stproc,rtproc,Comm().NumProc(),allproc.data(),Comm());
    //
    // In this case, we use "rtproc" instead of "allproc" afterwards, i.e. when
    // the node gids and element gids are gathered among procs.
    //**********************************************************************

    // at least for master, we want to do full ghosting on all procs
    std::vector<int> allproc(Core::Communication::num_mpi_ranks(get_comm()));
    for (int i = 0; i < Core::Communication::num_mpi_ranks(get_comm()); ++i) allproc[i] = i;

    // fill my own master row node ids
    const Core::LinAlg::Map* noderowmap = discret().node_row_map();
    std::vector<int> sdata;
    for (int i = 0; i < noderowmap->num_my_elements(); ++i)
    {
      int gid = noderowmap->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      auto* mrtrnode = dynamic_cast<Node*>(node);
      if (!mrtrnode->is_slave()) sdata.push_back(gid);
    }

    // gather all master row node gids redundantly
    std::vector<int> rdata;
    Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

    // add my own slave column node ids (non-redundant, standard overlap)
    const Core::LinAlg::Map* nodecolmap = discret().node_col_map();
    for (int i = 0; i < nodecolmap->num_my_elements(); ++i)
    {
      int gid = nodecolmap->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      auto* mrtrnode = dynamic_cast<Node*>(node);
      if (mrtrnode->is_slave()) rdata.push_back(gid);
    }

    // build new node column map (on ALL processors)
    Core::LinAlg::Map newnodecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
    sdata.clear();
    rdata.clear();

    // fill my own master row element ids
    const Core::LinAlg::Map* elerowmap = discret().element_row_map();
    sdata.resize(0);
    for (int i = 0; i < elerowmap->num_my_elements(); ++i)
    {
      int gid = elerowmap->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      auto* mrtrele = dynamic_cast<Mortar::Element*>(ele);
      if (!mrtrele->is_slave()) sdata.push_back(gid);
    }

    // gather all gids of elements redundantly
    rdata.resize(0);
    Core::LinAlg::gather<int>(sdata, rdata, (int)allproc.size(), allproc.data(), get_comm());

    // add my own slave column node ids (non-redundant, standard overlap)
    const Core::LinAlg::Map* elecolmap = discret().element_col_map();
    for (int i = 0; i < elecolmap->num_my_elements(); ++i)
    {
      int gid = elecolmap->gid(i);
      Core::Elements::Element* ele = discret().g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      auto* mrtrele = dynamic_cast<Mortar::Element*>(ele);
      if (mrtrele->is_slave()) rdata.push_back(gid);
    }

    // build new element column map (on ALL processors)
    Core::LinAlg::Map newelecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
    sdata.clear();
    rdata.clear();
    allproc.clear();

    // redistribute the discretization of the interface according to the
    // new node / element column layout (i.e. master = full overlap)
    discret().export_column_nodes(newnodecolmap);
    discret().export_column_elements(newelecolmap);
  }

  //*****NON-REDUNDANT STORAGE*****
  else if (interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::roundrobin ||
           interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::binning)
  {
    // nothing to do here, we work with the given non-redundant distribution
    // of both slave and master nodes to the individual processors. However
    // we want ALL procs to be part of the interface discretization, not only
    // the ones that do own or ghost any nodes in the natural distribution of
    // idiscret_. This makes dynamic redistribution easier!

    //**********************************************************************
    // IMPORTANT NOTE:
    // In an older code version, we only did ghosting on procs that own or ghost
    // any of the interface nodes or elements  in the natural distr. of idiscret_.
    // The corresponding code lines for creating this proc list are:
    //
    // std::vector<int> stproc;
    // if (oldnodecolmap_->NumMyElements() || oldelecolmap_->NumMyElements())
    //   stproc.push_back(Core::Communication::my_mpi_rank(Comm()));
    // std::vector<int> rtproc;
    // Core::LinAlg::gather<int>(stproc,rtproc,Comm().NumProc(),allproc.data(),Comm());
    //
    // In this case, we use "rtproc" instead of "allproc" afterwards, i.e. when
    // the node gids and element gids are gathered among procs.
    //**********************************************************************

    // we keep the current ghosting, but we want to (formally) include
    // all processors, even if they do not own or ghost a single node or
    // element in the natural distribution of idiscret_
    std::vector<int> rdata;

    // fill my own slave and master column node ids (non-redundant)
    const Core::LinAlg::Map* nodecolmap = discret().node_col_map();
    for (int i = 0; i < nodecolmap->num_my_elements(); ++i)
    {
      int gid = nodecolmap->gid(i);
      rdata.push_back(gid);
    }

    // re-build node column map (now formally on ALL processors)
    Core::LinAlg::Map newnodecolmap(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
    rdata.clear();

    // fill my own slave and master column element ids (non-redundant)
    const Core::LinAlg::Map* elecolmap = discret().element_col_map();
    for (int i = 0; i < elecolmap->num_my_elements(); ++i)
    {
      int gid = elecolmap->gid(i);
      rdata.push_back(gid);
    }

    // re-build element column map (now formally on ALL processors)
    std::shared_ptr<Core::LinAlg::Map> newelecolmap =
        std::make_shared<Core::LinAlg::Map>(-1, (int)rdata.size(), rdata.data(), 0, get_comm());
    rdata.clear();

    // redistribute the discretization of the interface according to the
    // new (=old) node / element column layout
    discret().export_column_nodes(newnodecolmap);
    discret().export_column_elements(*newelecolmap);

    if (interface_data_->get_extend_ghosting() == Inpar::Mortar::ExtendGhosting::binning)
    {
      /* We have to update the row/column maps split into master/slave. We start from the new
       * node/element column maps. Since we don't have row maps at this point, we can/have to pass
       * the column map as row map.
       */
      update_master_slave_element_maps(*newelecolmap, *newelecolmap);
      update_master_slave_node_maps(newnodecolmap, newnodecolmap);

      /* Ask the discretization to initialize the elements. We need this, since the setup of the
       * binning strategy relies on some element information.
       * Note: This should be cheap, since we don't assign new degrees of freedom. Still, we skip
       * initialization of elements, if we know, that the discretization will be redistributed
       * again.
       */
      discret().fill_complete({
          .assign_degrees_of_freedom = false,
          .init_elements = isFinalParallelDistribution,
          .do_boundary_conditions = false,
      });

      // Create the binning strategy
      std::shared_ptr<Core::Binstrategy::BinningStrategy> binningstrategy = setup_binning_strategy(
          binning_params, meanVelocity, output_control, spatial_approximation_type);

      // fill master and slave elements into bins
      std::map<int, std::set<int>> slavebinelemap;
      binningstrategy->distribute_elements_to_bins_using_ele_aabb(discret(),
          std::views::filter(discret().my_col_element_range(), [](auto ele)
              { return dynamic_cast<const Mortar::Element*>(ele.user_element())->is_slave(); }),
          slavebinelemap);

      std::map<int, std::set<int>> masterbinelemap;
      binningstrategy->distribute_elements_to_bins_using_ele_aabb(discret(),
          std::views::filter(discret().my_col_element_range(), [](auto ele)
              { return !dynamic_cast<const Mortar::Element*>(ele.user_element())->is_slave(); }),
          masterbinelemap);

      // Extend ghosting of the master elements
      std::map<int, std::set<int>> ext_bin_to_ele_map;
      std::shared_ptr<const Core::LinAlg::Map> extendedmastercolmap =
          binningstrategy->extend_element_col_map(slavebinelemap, masterbinelemap,
              ext_bin_to_ele_map, nullptr, nullptr, newelecolmap.get());

      // adapt layout to extended ghosting in the discretization
      // first export the elements according to the processor local element column maps
      discret().export_column_elements(*extendedmastercolmap);

      // get the node ids of the elements that are to be ghosted and create a proper node column map
      // for their export
      std::set<int> nodes;
      for (int lid = 0; lid < extendedmastercolmap->num_my_elements(); ++lid)
      {
        Core::Elements::Element* ele = discret().g_element(extendedmastercolmap->gid(lid));
        const int* nodeids = ele->node_ids();
        for (int inode = 0; inode < ele->num_node(); ++inode) nodes.insert(nodeids[inode]);
      }

      std::vector<int> colnodes(nodes.begin(), nodes.end());
      Core::LinAlg::Map nodecolmap(-1, (int)colnodes.size(), colnodes.data(), 0, get_comm());

      // now ghost the nodes
      discret().export_column_nodes(nodecolmap);
    }
  }

  //*****INVALID CASES*****
  else
  {
    FOUR_C_THROW("Invalid redundancy type.");
  }
}

/*----------------------------------------------------------------------*
 |  create search tree (public)                               popp 01/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::create_search_tree()
{
  // binary tree search
  if (search_alg() == Inpar::Mortar::search_binarytree)
  {
    // create fully overlapping map of all master elements
    // for non-redundant storage (RRloop) we handle the master elements
    // like the slave elements --> melecolmap_
    auto strategy = Teuchos::getIntegralValue<Inpar::Mortar::ExtendGhosting>(
        interface_params().sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY");

    // get update type of binary tree
    auto updatetype = Teuchos::getIntegralValue<Inpar::Mortar::BinaryTreeUpdateType>(
        interface_params(), "BINARYTREE_UPDATETYPE");

    std::shared_ptr<Core::LinAlg::Map> melefullmap = nullptr;
    switch (strategy)
    {
      case Inpar::Mortar::ExtendGhosting::roundrobin:
      case Inpar::Mortar::ExtendGhosting::binning:
      {
        melefullmap = melecolmap_;
        break;
      }
      case Inpar::Mortar::ExtendGhosting::redundant_all:
      case Inpar::Mortar::ExtendGhosting::redundant_master:
      {
        melefullmap = Core::LinAlg::allreduce_e_map(*melerowmap_);
        break;
      }
      default:
      {
        FOUR_C_THROW("Unknown strategy to deal with interface ghosting.");
        break;
      }
    }

    // create binary tree object for search and setup tree
    binarytree_ = std::make_shared<Mortar::BinaryTree>(discret(), selecolmap_, melefullmap, n_dim(),
        search_param(), updatetype, search_use_aux_pos());
    // initialize the binary tree
    binarytree_->init();
  }
}

/*----------------------------------------------------------------------*
 |  update master and slave sets (nodes etc.)                 popp 11/09|
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_master_slave_sets()
{
  update_master_slave_node_maps();
  update_master_slave_element_maps();
  update_master_slave_dof_maps();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_master_slave_dof_maps()
{
  // Vectors to collect GIDs to build maps
  std::vector<int> sc;  // slave column map
  std::vector<int> sr;  // slave row map
  std::vector<int> mc;  // master column map
  std::vector<int> mr;  // master row map

  const int numMyColumnDofs = discret().node_col_map()->num_my_elements();
  for (int i = 0; i < numMyColumnDofs; ++i)
  {
    int gid = discret().node_col_map()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);
    bool isslave = mrtrnode->is_slave();
    const int numdof = mrtrnode->num_dof();

    if (isslave)
      for (int j = 0; j < numdof; ++j) sc.push_back(mrtrnode->dofs()[j]);
    else
      for (int j = 0; j < numdof; ++j) mc.push_back(mrtrnode->dofs()[j]);

    if (discret().node_row_map()->my_gid(gid))
    {
      if (isslave)
        for (int j = 0; j < numdof; ++j) sr.push_back(mrtrnode->dofs()[j]);
      else
        for (int j = 0; j < numdof; ++j) mr.push_back(mrtrnode->dofs()[j]);
    }
  }

  sdofrowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sr.size(), sr.data(), 0, get_comm());
  sdofcolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sc.size(), sc.data(), 0, get_comm());
  mdofrowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)mr.size(), mr.data(), 0, get_comm());
  mdofcolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)mc.size(), mc.data(), 0, get_comm());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_master_slave_element_maps()
{
  update_master_slave_element_maps(*discret().element_row_map(), *discret().element_col_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_master_slave_element_maps(
    const Core::LinAlg::Map& elementRowMap, const Core::LinAlg::Map& elementColumnMap)
{
  // Vectors to collect GIDs to build maps
  std::vector<int> sc;  // slave column map
  std::vector<int> sr;  // slave row map
  std::vector<int> mc;  // master column map
  std::vector<int> mr;  // master row map

  const int numMyColumnElements = elementColumnMap.num_my_elements();
  for (int i = 0; i < numMyColumnElements; ++i)
  {
    int gid = elementColumnMap.gid(i);
    bool isslave = dynamic_cast<Mortar::Element*>(discret().g_element(gid))->is_slave();

    if (isslave)
      sc.push_back(gid);
    else
      mc.push_back(gid);

    if (elementRowMap.my_gid(gid))
    {
      if (isslave)
        sr.push_back(gid);
      else
        mr.push_back(gid);
    }
  }

  selerowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sr.size(), sr.data(), 0, get_comm());
  selecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sc.size(), sc.data(), 0, get_comm());
  melerowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)mr.size(), mr.data(), 0, get_comm());
  melecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)mc.size(), mc.data(), 0, get_comm());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_master_slave_node_maps()
{
  update_master_slave_node_maps(*discret().node_row_map(), *discret().node_col_map());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::update_master_slave_node_maps(
    const Core::LinAlg::Map& nodeRowMap, const Core::LinAlg::Map& nodeColumnMap)
{
  // Vectors to collect GIDs to build maps
  std::vector<int> sc;   // slave column map
  std::vector<int> sr;   // slave row map
  std::vector<int> mc;   // master column map
  std::vector<int> mr;   // master row map
  std::vector<int> srb;  // slave row map + boundary nodes
  std::vector<int> scb;  // slave column map + boundary nodes
  std::vector<int> mrb;  // master row map - boundary nodes
  std::vector<int> mcb;  // master column map - boundary nodes

  const int numMyColumnNodes = nodeColumnMap.num_my_elements();
  for (int i = 0; i < numMyColumnNodes; ++i)
  {
    int gid = nodeColumnMap.gid(i);
    bool isslave = dynamic_cast<Mortar::Node*>(discret().g_node(gid))->is_slave();
    bool isonbound = dynamic_cast<Mortar::Node*>(discret().g_node(gid))->is_on_boundor_ce();

    if (isslave || isonbound)
      scb.push_back(gid);
    else
      mcb.push_back(gid);
    if (isslave)
      sc.push_back(gid);
    else
      mc.push_back(gid);

    if (nodeRowMap.my_gid(gid))
    {
      if (isslave || isonbound)
        srb.push_back(gid);
      else
        mrb.push_back(gid);
      if (isslave)
        sr.push_back(gid);
      else
        mr.push_back(gid);
    }
  }

  snoderowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sr.size(), sr.data(), 0, get_comm());
  snodecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sc.size(), sc.data(), 0, get_comm());
  mnoderowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)mr.size(), mr.data(), 0, get_comm());
  mnodecolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)mc.size(), mc.data(), 0, get_comm());

  snoderowmapbound_ =
      std::make_shared<Core::LinAlg::Map>(-1, (int)srb.size(), srb.data(), 0, get_comm());
  snodecolmapbound_ =
      std::make_shared<Core::LinAlg::Map>(-1, (int)scb.size(), scb.data(), 0, get_comm());
  mnoderowmapnobound_ =
      std::make_shared<Core::LinAlg::Map>(-1, (int)mrb.size(), mrb.data(), 0, get_comm());
  mnodecolmapnobound_ =
      std::make_shared<Core::LinAlg::Map>(-1, (int)mcb.size(), mcb.data(), 0, get_comm());

  // build exporter
  interface_data_->sl_exporter_ptr() = std::make_shared<Core::Communication::Exporter>(
      *snoderowmapbound_, *snodecolmapbound_, get_comm());
}

/*----------------------------------------------------------------------*
 |  restrict slave sets to actual meshtying zone              popp 08/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::restrict_slave_sets()
{
  //********************************************************************
  // NODES
  //********************************************************************
  {
    std::vector<int> sc;      // slave column map
    std::vector<int> sr;      // slave row map
    std::vector<int> scfull;  // slave full map

    for (int i = 0; i < snodecolmap_->num_my_elements(); ++i)
    {
      int gid = snodecolmap_->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      auto* mrtrnode = dynamic_cast<Node*>(node);
      int istied = (int)mrtrnode->is_tied_slave();

      if (istied && snodecolmap_->my_gid(gid)) sc.push_back(gid);
      if (istied && snoderowmap_->my_gid(gid)) sr.push_back(gid);
    }

    snoderowmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sr.size(), sr.data(), 0, get_comm());
    snodecolmap_ =
        std::make_shared<Core::LinAlg::Map>(-1, (int)sc.size(), sc.data(), 0, get_comm());
  }

  //********************************************************************
  // ELEMENTS
  //********************************************************************
  // no need to do this for elements, because all mortar quantities
  // are defined with respect to node or dof maps (D,M,...). As all
  // mortar stuff has already been evaluated, it would not matter if
  // we adapted the element maps as well, but we just skip it.

  //********************************************************************
  // DOFS
  //********************************************************************
  {
    std::vector<int> sc;      // slave column map
    std::vector<int> sr;      // slave row map
    std::vector<int> scfull;  // slave full map

    for (int i = 0; i < snodecolmap_->num_my_elements(); ++i)
    {
      int gid = snodecolmap_->gid(i);
      Core::Nodes::Node* node = discret().g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      auto* mrtrnode = dynamic_cast<Node*>(node);
      const bool istied = mrtrnode->is_tied_slave();
      const int numdof = mrtrnode->num_dof();

      if (istied && snodecolmap_->my_gid(gid))
        for (int j = 0; j < numdof; ++j) sc.push_back(mrtrnode->dofs()[j]);

      if (istied && snoderowmap_->my_gid(gid))
        for (int j = 0; j < numdof; ++j) sr.push_back(mrtrnode->dofs()[j]);
    }

    sdofrowmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sr.size(), sr.data(), 0, get_comm());
    sdofcolmap_ = std::make_shared<Core::LinAlg::Map>(-1, (int)sc.size(), sc.data(), 0, get_comm());
  }
}

/*----------------------------------------------------------------------*
 |  update Lagrange multiplier set (dofs)                     popp 08/10|
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Mortar::Interface::update_lag_mult_sets(
    int offset_if, const bool& redistributed, const Core::LinAlg::Map& ref_map) const
{
  if (redistributed)
  {
    return redistribute_lag_mult_sets();
  }
  //********************************************************************
  // LAGRANGE MULTIPLIER DOFS
  //********************************************************************
  // NOTE: we want no gap between the displacement dofs and the newly
  // defined Lagrange multiplier dofs!! Thus, if the maximum displacement
  // dof is 12.345, we want the LM dofs to start with 12.346. This can
  // be readily achieved, because we know that the lmdofmap will have
  // the same parallel distribution as the slavedofrowmap. The only
  // thing we need to take care of is to avoid overlapping of the LM
  // dofs among different processors. Therefore, the total number of
  // slave nodes (and thus LM nodes) of each processor is communicated
  // to ALL other processors and an offset is then determined for each
  // processor based on this information.
  //********************************************************************
  // temporary vector of LM dofs
  std::vector<int> lmdof;

  // gather information over all procs
  std::vector<int> localnumlmdof(Core::Communication::num_mpi_ranks(get_comm()));
  std::vector<int> globalnumlmdof(Core::Communication::num_mpi_ranks(get_comm()));
  localnumlmdof[Core::Communication::my_mpi_rank(get_comm())] = ref_map.num_my_elements();
  globalnumlmdof = Core::Communication::sum_all(localnumlmdof, get_comm());

  // compute offset for LM dof initialization for all procs
  int offset = 0;
  for (int k = 0; k < Core::Communication::my_mpi_rank(get_comm()); ++k)
    offset += globalnumlmdof[k];

  // loop over all slave dofs and initialize LM dofs
  lmdof.reserve(ref_map.num_my_elements());
  for (int i = 0; i < ref_map.num_my_elements(); ++i)
    lmdof.push_back(max_dof_global() + 1 + offset_if + offset + i);

  // create interface LM map
  // (if maxdofglobal_ == 0, we do not want / need this)
  if (max_dof_global() > 0)
    return std::make_shared<Core::LinAlg::Map>(-1, (int)lmdof.size(), lmdof.data(), 0, get_comm());

  return nullptr;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Interface::store_unredistributed_maps()
{
  psdofrowmap_ = std::make_shared<Core::LinAlg::Map>(*sdofrowmap_);
  interface_data_->non_redist_master_dof_row_map() =
      std::make_shared<Core::LinAlg::Map>(*mdofrowmap_);
  plmdofmap_ = std::make_shared<Core::LinAlg::Map>(*lmdofmap_);

  interface_data_->non_redist_slave_node_row_map() =
      std::make_shared<Core::LinAlg::Map>(*snoderowmap_);
  interface_data_->non_redist_master_node_row_map() =
      std::make_shared<Core::LinAlg::Map>(*mnoderowmap_);
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Map> Mortar::Interface::redistribute_lag_mult_sets() const
{
  if (!plmdofmap_) FOUR_C_THROW("The plmdofmap_ is not yet initialized!");
  if (!psdofrowmap_) FOUR_C_THROW("The psdofrowmap_ is not yet initialized!");

  // new lm dofs
  std::vector<int> lmdof(sdofrowmap_->num_my_elements());

  /* get the initial correlation between the slave dofs
   * and the lagrange multiplier dofs
   *
   * There is always a tuple of two values which correlate
   * with each other. The first one is the lm-dof, the second
   * one is the slave dof. */
  std::vector<int> my_related_gids(2 * plmdofmap_->num_my_elements(), -1);
  for (int i = 0; i < plmdofmap_->num_my_elements(); ++i)
  {
    my_related_gids[2 * i] = plmdofmap_->gid(i);
    my_related_gids[2 * i + 1] = psdofrowmap_->gid(i);
  }

  std::vector<int> g_related_gids;
  Core::Communication::barrier(get_comm());
  for (int p = 0; p < Core::Communication::num_mpi_ranks(get_comm()); ++p)
  {
    int num_mygids = plmdofmap_->num_my_elements();

    Core::Communication::broadcast(&num_mygids, 1, p, get_comm());
    // skip processors which hold no correlation info
    if (num_mygids == 0) continue;
    g_related_gids.resize(2 * num_mygids);

    /* communicate the correlation list of proc p
     * to all procs */
    if (p == Core::Communication::my_mpi_rank(get_comm()))
      for (std::size_t i = 0; i < my_related_gids.size(); ++i)
        g_related_gids[i] = my_related_gids[i];
    Core::Communication::broadcast(g_related_gids.data(), 2 * num_mygids, p, get_comm());

    for (int i = 0; i < num_mygids; ++i)
    {
      /* check in the already redistributed sdofrowmap on
       * each processor which one holds the current gid */
      int my_sllid = sdofrowmap_->lid(g_related_gids[2 * i + 1]);
      /* on the proc holding the gid, we store the corresponding
       * lm-dof-gid as well at the same lid. */
      if (my_sllid != -1) lmdof[my_sllid] = g_related_gids[2 * i];
    }
    // wait for the arrival of all procs
    Core::Communication::barrier(get_comm());
  }

  // create deterministic interface LM map
  return std::make_shared<Core::LinAlg::Map>(-1, (int)lmdof.size(), lmdof.data(), 0, get_comm());
}

/*----------------------------------------------------------------------*
 |  initialize / reset mortar interface                       popp 01/08|
 *----------------------------------------------------------------------*/
void Mortar::Interface::initialize()
{
  // loop over all nodes to reset stuff (fully overlapping column map)
  for (int i = 0; i < idiscret_->num_my_col_nodes(); ++i)
  {
    auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_col_node(i));

    // reset feasible projection and segmentation status
    node->has_proj() = false;
    node->has_segment() = false;
  }

  // loop over all slave nodes to reset stuff (standard column map)
  // (include slave side boundary nodes / crosspoints)
  for (int i = 0; i < slave_col_nodes_bound()->num_my_elements(); ++i)
  {
    int gid = slave_col_nodes_bound()->gid(i);
    Core::Nodes::Node* node = discret().g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* monode = dynamic_cast<Mortar::Node*>(node);

    // reset nodal normal
    for (int j = 0; j < 3; ++j) monode->mo_data().n()[j] = 0.0;

    // reset nodal Mortar maps
    monode->mo_data().get_d().clear();
    monode->mo_data().get_m().clear();
    monode->mo_data().get_mmod().clear();
  }

  // loop over all elements to reset candidates / search lists
  // (use standard slave column map)
  for (int i = 0; i < slave_col_elements()->num_my_elements(); ++i)
  {
    int gid = slave_col_elements()->gid(i);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
    auto* mele = dynamic_cast<Mortar::Element*>(ele);

    mele->mo_data().search_elements().resize(0);
  }
}


/*----------------------------------------------------------------------*
 |  set current and old deformation state                      popp 12/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::set_state(
    const StateType& statetype, const Core::LinAlg::Vector<double>& vec)
{
  switch (statetype)
  {
    case state_new_displacement:
    {
      // alternative method to get vec to full overlap
      Core::LinAlg::Vector<double> global(*idiscret_->dof_col_map(), false);
      Core::LinAlg::export_to(vec, global);

      // set displacements in interface discretization
      idiscret_->set_state(state_type_to_string(statetype), global);

      // loop over all nodes to set current displacement
      // (use fully overlapping column map)
      for (int i = 0; i < idiscret_->num_my_col_nodes(); ++i)
      {
        auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_col_node(i));
        const int numdof = node->num_dof();
        std::vector<int> lm(numdof);

        for (int j = 0; j < numdof; ++j) lm[j] = node->dofs()[j];

        std::vector<double> mydisp = Core::FE::extract_values(global, lm);

        // add mydisp[2]=0 for 2D problems
        if (mydisp.size() < 3) mydisp.resize(3);

        // set current configuration
        auto x = node->x();
        for (size_t j = 0; j < x.size(); ++j) node->xspatial()[j] = x[j] + mydisp[j];
        for (size_t j = x.size(); j < 3; ++j) node->xspatial()[j] = 0.0;
      }

      // compute element areas
      set_element_areas();
      break;
    }
    case state_lagrange_multiplier:
    {
      // alternative method to get vec to full overlap
      Core::LinAlg::Vector<double> global(*idiscret_->dof_col_map(), false);
      Core::LinAlg::export_to(vec, global);

      // loop over all nodes to set current displacement
      // (use fully overlapping column map)
      for (int i = 0; i < slave_col_nodes()->num_my_elements(); ++i)
      {
        auto* node = dynamic_cast<Mortar::Node*>(idiscret_->g_node(slave_col_nodes()->gid(i)));
        const int numdof = node->num_dof();
        std::vector<int> lm(numdof);

        for (int j = 0; j < numdof; ++j) lm[j] = node->dofs()[j];

        std::vector<double> mydisp = Core::FE::extract_values(global, lm);

        // add mydisp[2]=0 for 2D problems
        if (mydisp.size() < 3) mydisp.resize(3);

        // set current configuration
        for (int j = 0; j < 3; ++j) node->mo_data().lm()[j] = mydisp[j];
      }
      break;
    }
    case state_old_displacement:
    {
      // alternative method to get vec to full overlap
      Core::LinAlg::Vector<double> global(*idiscret_->dof_col_map(), false);
      Core::LinAlg::export_to(vec, global);

      // set displacements in interface discretization
      idiscret_->set_state(state_type_to_string(statetype), global);

      // loop over all nodes to set current displacement
      // (use fully overlapping column map)
      for (int i = 0; i < idiscret_->num_my_col_nodes(); ++i)
      {
        auto* node = dynamic_cast<Mortar::Node*>(idiscret_->l_col_node(i));
        const int numdof = node->num_dof();
        std::vector<int> lm(numdof);

        for (int j = 0; j < numdof; ++j) lm[j] = node->dofs()[j];

        std::vector<double> myolddisp = Core::FE::extract_values(global, lm);

        // add mydisp[2]=0 for 2D problems
        if (myolddisp.size() < 3) myolddisp.resize(3);

        // set old displacement
        for (int j = 0; j < 3; ++j) node->uold()[j] = myolddisp[j];
      }

      break;
    }
    default:
    {
      FOUR_C_THROW(
          "The given state type is unsupported! (type = {})", state_type_to_string(statetype));
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 |  compute element areas (public)                            popp 11/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::set_element_areas()
{
  // loop over all elements to set current element length / area
  // (use standard slave column map)
  for (int i = 0; i < slave_col_elements()->num_my_elements(); ++i)
  {
    int gid = slave_col_elements()->gid(i);
    Core::Elements::Element* ele = discret().g_element(gid);
    if (!ele) FOUR_C_THROW("Cannot find ele with gid {}", gid);
    auto* mele = dynamic_cast<Mortar::Element*>(ele);

    mele->mo_data().area() = mele->compute_area();
  }
}


/*----------------------------------------------------------------------*
 |  evaluate geometric setting (create integration cells)    farah 01/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_geometry(std::vector<std::shared_ptr<Mortar::IntCell>>& intcells)
{
  // time measurement
  Core::Communication::barrier(get_comm());
  const double t_start = Teuchos::Time::wallTime();

  // check
  if (n_dim() == 2) FOUR_C_THROW("Geometry evaluation for mortar interface only for 3D problems!");

  auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  if (algo == Inpar::Mortar::algorithm_nts)
    FOUR_C_THROW("Geometry evaluation only for mortar problems!");

  // interface needs to be complete
  if (!filled() && Core::Communication::my_mpi_rank(get_comm()) == 0)
    FOUR_C_THROW("fill_complete() not called on interface %", id_);

  // clear vector
  intcells.clear();

  //**********************************************************************
  // search algorithm
  //**********************************************************************
  if (search_alg() == Inpar::Mortar::search_bfele)
    evaluate_search_brute_force(search_param());
  else if (search_alg() == Inpar::Mortar::search_binarytree)
    evaluate_search_binarytree();
  else
    FOUR_C_THROW("Invalid search algorithm");

  // create normals
  evaluate_nodal_normals();

  // export nodal normals to slave node column map
  // this call is very expensive and the computation
  // time scales directly with the proc number !
  export_nodal_normals();

  // loop over proc's slave elements of the interface for integration
  // use standard column map to include processor's ghosted elements
  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    int gid1 = selecolmap_->gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid %", gid1);
    auto* selement = dynamic_cast<Mortar::Element*>(ele1);

    // skip zero-sized nurbs elements (slave)
    if (selement->zero_sized()) continue;

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int j = 0; j < selement->mo_data().num_search_elements(); ++j)
    {
      int gid2 = selement->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      auto* melement = dynamic_cast<Mortar::Element*>(ele2);

      // skip zero-sized nurbs elements (master)
      if (melement->zero_sized()) continue;

      //********************************************************************
      // 1) perform coupling (projection + overlap detection for sl/m pairs)
      //********************************************************************
      if (selement->is_quad())
      {
        FOUR_C_THROW("Geometry evaluation only implemented for first order elements!");
      }
      // noquad!
      else
      {
        Mortar::Coupling3d coup(*idiscret_, dim_, false, imortar_, *selement, *melement);

        // do coupling
        coup.evaluate_coupling();

        // set sele and mele id and push into global vector
        for (auto& coupcell : coup.cells())
        {
          coupcell->set_slave_id(selement->id());
          coupcell->set_master_id(melement->id());
          intcells.push_back(coupcell);
        }
      }
    }
  }  // end sele loop

  // time measurement
  Core::Communication::barrier(get_comm());
  const double evaltime = Teuchos::Time::wallTime() - t_start;

  // time output
  std::cout << "Required time for geometry evaluation: " << evaltime << std::endl;
}


/*----------------------------------------------------------------------*
 |  evaluate mortar coupling (public)                         popp 11/07|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate(int rriter, const int& step, const int& iter,
    std::shared_ptr<Mortar::ParamsInterface> mparams_ptr)
{
  TEUCHOS_FUNC_TIME_MONITOR("Mortar::Interface::Evaluate");

  // interface needs to be complete
  if (!filled()) FOUR_C_THROW("fill_complete() not called on interface %", id_);

  //******************************************
  // Start basic evaluation part of interface
  //******************************************
  // start time measurement
  const double t_start = Teuchos::Time::wallTime();

  // evaluate nodal normals and decide in contact case if
  // this is a nonsmooth or smooth contact
  pre_evaluate(step, iter);

  // evaluation routine for coupling
  evaluate_coupling(*selecolmap_, snoderowmap_.get(), mparams_ptr);

  // do some post operations. nothing happens for standard cases...
  post_evaluate(step, iter);

  // end time on this proc
  const double inttime = Teuchos::Time::wallTime() - t_start;
  //******************************************
  // End basic evaluation part of interface
  //******************************************

  // store integrationtime
  inttime_interface_ = inttime;
}


/*----------------------------------------------------------------------*
 |  protected evaluate routine                               farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_coupling(const Core::LinAlg::Map& selecolmap,
    const Core::LinAlg::Map* snoderowmap,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  // decide which type of coupling should be evaluated
  auto algo = Teuchos::getIntegralValue<Inpar::Mortar::AlgorithmType>(imortar_, "ALGORITHM");

  // smooth contact
  switch (algo)
  {
    //*********************************
    // Mortar Coupling (STS)    (2D/3D)
    // Gauss-Point-To-Segment (GPTS)
    //*********************************
    case Inpar::Mortar::algorithm_mortar:
    case Inpar::Mortar::algorithm_gpts:
    {
      //********************************************************************
      // 1) perform coupling (projection + overlap detection for sl/m pairs)
      // 2) integrate Mortar matrix M and weighted gap g
      // 3) compute directional derivative of M and g and store into nodes
      //    (only for contact setting)
      //********************************************************************
      Mortar::Interface::evaluate_sts(selecolmap, mparams_ptr);
      break;
    }
    //*********************************
    // Segment-to-Line Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_stl:
    {
      //********************************************************************
      // 1) perform coupling (projection + line clipping edge surface pairs)
      // 2) integrate Mortar matrices D + M and weighted gap g
      // 3) compute directional derivative of D + M and g and store into nodes
      //    (only for contact setting)
      //********************************************************************
      evaluate_stl();
      break;
    }
    //*********************************
    // Line-to-Segment Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_lts:
    {
      //********************************************************************
      // 1) perform coupling (projection + line clipping edge surface pairs)
      // 2) integrate Mortar matrices D + M and weighted gap g
      // 3) compute directional derivative of D + M and g and store into nodes
      //    (only for contact setting)
      //********************************************************************
      evaluate_lts();
      break;
    }
    //*********************************
    // line-to-line Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_ltl:
    {
      //********************************************************************
      // 1) perform coupling (find closest point between to lines)
      // 2) evaluate gap and shape functions at this point
      // 3) compute directional derivative of entries and store into nodes
      //    (only for contact setting)
      //********************************************************************
      evaluate_ltl();
      break;
    }
    //*********************************
    // Node-to-Segment Coupling (2D/3D)
    //*********************************
    case Inpar::Mortar::algorithm_nts:
    {
      //********************************************************************
      // 1) try to project slave nodes onto master elements
      // 2) evaluate shape functions at projected positions
      // 3) compute directional derivative of M and g and store into nodes
      //    (only for contact setting)
      //********************************************************************
      evaluate_nts();
      break;
    }
    //*********************************
    // Node-to-Line Coupling (3D)
    //*********************************
    case Inpar::Mortar::algorithm_ntl:
    {
      FOUR_C_THROW("not yet implemented!");
      break;
    }
    //*********************************
    // Default case
    //*********************************
    default:
    {
      FOUR_C_THROW("Unknown discretization type for constraints!");
      break;
    }
  }
}


/*----------------------------------------------------------------------*
 |  evaluate coupling type segment-to-segment coupl          farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_sts(const Core::LinAlg::Map& selecolmap,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  TEUCHOS_FUNC_TIME_MONITOR("Mortar::Interface::EvaluateSTS");

  // loop over all slave col elements
  for (int i = 0; i < selecolmap.num_my_elements(); ++i)
  {
    const int gid1 = selecolmap.gid(i);
    Core::Elements::Element* ele1 = idiscret_->g_element(gid1);
    if (!ele1) FOUR_C_THROW("Cannot find slave element with gid {}", gid1);

    auto* selement = dynamic_cast<Mortar::Element*>(ele1);

    // skip zero-sized nurbs elements (slave)
    if (selement->zero_sized()) continue;

    // empty vector of master element pointers
    std::vector<Mortar::Element*> melements;

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int j = 0; j < selement->mo_data().num_search_elements(); ++j)
    {
      int gid2 = selement->mo_data().search_elements()[j];
      Core::Elements::Element* ele2 = idiscret_->g_element(gid2);
      if (!ele2) FOUR_C_THROW("Cannot find master element with gid {}", gid2);
      auto* melement = dynamic_cast<Mortar::Element*>(ele2);

      // skip zero-sized nurbs elements (master)
      if (melement->zero_sized()) continue;

      melements.push_back(melement);
    }

    // concrete coupling evaluation routine
    mortar_coupling(selement, melements, mparams_ptr);
  }
}


/*----------------------------------------------------------------------*
 |  evaluate coupling type node-to-segment coupl             farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_nts()
{
  // loop over slave nodes
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Mortar::Node*>(node);

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    // vector with possible contacting master eles
    std::vector<Mortar::Element*> meles;

    // fill vector with possibly contacting meles
    find_master_elements(*mrtrnode, meles);

    // skip calculation if no meles vector is empty
    if (meles.size() < 1) continue;

    // call interpolation functions
    NTS::MTInterpolator::impl(meles)->interpolate(*mrtrnode, meles);
  }
}


/*----------------------------------------------------------------------*
 |  evaluate coupling type line-to-segment coupl             farah 07/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_lts()
{
  FOUR_C_THROW("Line -to-segment is not available for meshtying problems.");
}

/*----------------------------------------------------------------------*
 |  evaluate coupling type line-to-line coupl                farah 07/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_ltl()
{
  FOUR_C_THROW("Line-to-line is not available for meshtying problems.");
}

/*----------------------------------------------------------------------*
 |  evaluate coupling type segment-to-line coupl             farah 07/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_stl()
{
  FOUR_C_THROW("Segment-to-line is not available for meshtying problems.");
}

/*----------------------------------------------------------------------*
 |  evaluate nodal normals (public)                           popp 10/11|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_nodal_normals() const
{
  // loop over proc's slave nodes of the interface
  // use row map and export to column map later
  // (use boundary map to include slave side boundary nodes)
  for (int i = 0; i < snoderowmapbound_->num_my_elements(); ++i)
  {
    int gid = snoderowmapbound_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // build averaged normal at each slave node
    mrtrnode->build_averaged_normal();
  }
}

/*----------------------------------------------------------------------*
 |  pre evaluate to calc normals                            farah 02/16 |
 *----------------------------------------------------------------------*/
void Mortar::Interface::pre_evaluate(const int& step, const int& iter)
{
  //**********************************************************************
  // search algorithm
  //**********************************************************************
  if (search_alg() == Inpar::Mortar::search_bfele)
    evaluate_search_brute_force(search_param());
  else if (search_alg() == Inpar::Mortar::search_binarytree)
    evaluate_search_binarytree();
  else
    FOUR_C_THROW("Invalid search algorithm");

  // evaluate averaged nodal normals on slave side
  evaluate_nodal_normals();

  // export nodal normals to slave node column map
  // this call is very expensive and the computation
  // time scales directly with the proc number !
  export_nodal_normals();
}


/*----------------------------------------------------------------------*
 |  post evaluate                                           farah 02/16 |
 *----------------------------------------------------------------------*/
void Mortar::Interface::post_evaluate(const int step, const int iter)
{
  // nothing to do...
}


/*----------------------------------------------------------------------*
 |  find meles to snode                                     farah 01/16 |
 *----------------------------------------------------------------------*/
void Mortar::Interface::find_master_elements(
    const Node& mrtrnode, std::vector<Mortar::Element*>& meles) const
{
  // clear vector
  meles.clear();

  // empty vector of master element pointers
  std::set<int> donebefore;

  for (auto ele : mrtrnode.adjacent_elements())
  {
    auto* adjcele = dynamic_cast<const Mortar::Element*>(ele.user_element());

    // skip zero-sized nurbs elements (slave)
    if (adjcele->zero_sized()) continue;

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int k = 0; k < adjcele->mo_data().num_search_elements(); ++k)
    {
      int gid2 = adjcele->mo_data().search_elements()[k];
      Core::Elements::Element* mele = idiscret_->g_element(gid2);
      if (!mele) FOUR_C_THROW("Cannot find master element with gid %", gid2);
      auto* melement = dynamic_cast<Mortar::Element*>(mele);

      // skip zero-sized nurbs elements (master)
      if (melement->zero_sized()) continue;

      // check uniqueness
      auto iter = donebefore.find(melement->id());
      if (iter != donebefore.end()) continue;

      donebefore.insert(melement->id());

      // fill vector
      meles.push_back(melement);
    }  // found eles
  }  // loop over adjacent slave elements
}


/*----------------------------------------------------------------------*
 |  find mnodes to snode                                    farah 01/16 |
 *----------------------------------------------------------------------*/
void Mortar::Interface::find_master_nodes(
    const Node& mrtrnode, std::vector<Mortar::Element*>& meles, std::vector<Node*>& mnodes) const
{
  // clear vector
  mnodes.clear();

  // check meles
  if (meles.size() < 1) return;

  // set object to guarantee uniqueness of found mnodes
  std::set<int> donebefore;

  for (auto& mele : meles)
  {
    // skip zero-sized nurbs elements (master)
    if (mele->zero_sized()) continue;

    // loop over the candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int k = 0; k < mele->num_node(); ++k)
    {
      Core::Nodes::Node* node = mele->nodes()[k];
      if (!node) FOUR_C_THROW("Cannot find master node");
      auto* mnode = dynamic_cast<Node*>(node);

      // check uniqueness
      auto iter = donebefore.find(mnode->id());
      if (iter != donebefore.end()) continue;

      donebefore.insert(mnode->id());

      // fill vector
      mnodes.push_back(mnode);
    }  // found eles
  }  // loop over adjacent slave elements
}


/*----------------------------------------------------------------------*
 |  evaluate nodal normals and store them in map (public)      jb 07/14 |
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_nodal_normals(std::map<int, std::vector<double>>& mynormals)
{
  // loop over proc's slave nodes of the interface
  // use row map and export to column map later
  // (use boundary map to include slave side boundary nodes)
  for (int i = 0; i < snoderowmapbound_->num_my_elements(); ++i)
  {
    int gid = snoderowmapbound_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // build averaged normal at each slave node
    mrtrnode->build_averaged_normal();

    int numdofs = mrtrnode->num_dof();
    std::vector<double> temp(numdofs, 0.0);
    for (int j = 0; j < numdofs; j++)
    {
      temp[j] = mrtrnode->mo_data().n()[j];
    }
    mynormals.insert(std::pair<int, std::vector<double>>(gid, temp));
  }
}

/*----------------------------------------------------------------------*
 |  export nodal normals (public)                             popp 11/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::export_nodal_normals() const
{
  // create empty data objects
  std::map<int, std::shared_ptr<Core::LinAlg::SerialDenseMatrix>> triad;

  // build info on row map
  for (int i = 0; i < snoderowmapbound_->num_my_elements(); ++i)
  {
    int gid = snoderowmapbound_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // fill nodal matrix
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc =
        std::make_shared<Core::LinAlg::SerialDenseMatrix>(3, 1);
    (*loc)(0, 0) = mrtrnode->mo_data().n()[0];
    (*loc)(1, 0) = mrtrnode->mo_data().n()[1];
    (*loc)(2, 0) = mrtrnode->mo_data().n()[2];

    triad[gid] = loc;
  }

  // communicate from slave node row to column map

  interface_data_->exporter().do_export(triad);

  // extract info on column map
  for (int i = 0; i < snodecolmapbound_->num_my_elements(); ++i)
  {
    // only do something for ghosted nodes
    int gid = snodecolmapbound_->gid(i);
    if (snoderowmapbound_->my_gid(gid)) continue;

    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // extract info
    std::shared_ptr<Core::LinAlg::SerialDenseMatrix> loc = triad[gid];
    mrtrnode->mo_data().n()[0] = (*loc)(0, 0);
    mrtrnode->mo_data().n()[1] = (*loc)(1, 0);
    mrtrnode->mo_data().n()[2] = (*loc)(2, 0);
  }
}

/*----------------------------------------------------------------------*
 |  Search element-based "brute force" (public)               popp 10/08|
 *----------------------------------------------------------------------*/
void Mortar::Interface::evaluate_search_brute_force(const double& eps)
{
  /**********************************************************************/
  /* SEARCH ALGORITHM:                                                  */
  /* The idea of the search is to reduce the number of master / slave   */
  /* element pairs that are checked for overlap and coupling by intro-  */
  /* ducing information about proximity and maybe history!              */
  /* This old version is already based on bounding volumes, but still   */
  /* brute force for finding the overlap of these bounding volumes,     */
  /* so it has been replaced by a more efficient approach (binary tree).*/
  /**********************************************************************/

  // calculate minimal element length
  double lmin = 1.0e12;
  double enlarge = 0.0;

  // create fully overlapping map of all master elements
  // for non-redundant storage (RRloop) we handle the master elements
  // like the slave elements --> melecolmap_
  auto strategy = Teuchos::getIntegralValue<Inpar::Mortar::ExtendGhosting>(
      interface_params().sublist("PARALLEL REDISTRIBUTION"), "GHOSTING_STRATEGY");
  std::shared_ptr<Core::LinAlg::Map> melefullmap = nullptr;

  switch (strategy)
  {
    case Inpar::Mortar::ExtendGhosting::redundant_all:
    case Inpar::Mortar::ExtendGhosting::redundant_master:
    {
      melefullmap = Core::LinAlg::allreduce_e_map(*melerowmap_);
      break;
    }
    case Inpar::Mortar::ExtendGhosting::roundrobin:
    {
      melefullmap = melerowmap_;
      break;
    }
    case Inpar::Mortar::ExtendGhosting::binning:
    {
      melefullmap = melecolmap_;
      break;
    }
    default:
    {
      FOUR_C_THROW("Unknown strategy to deal with interface ghosting.");
      break;
    }
  }

  // loop over all slave elements on this proc.
  for (int i = 0; i < selecolmap_->num_my_elements(); ++i)
  {
    Core::Elements::Element* element = idiscret_->g_element(selecolmap_->gid(i));
    if (!element) FOUR_C_THROW("Cannot find element with gid {}", selecolmap_->gid(i));
    auto* mrtrelement = dynamic_cast<Mortar::Element*>(element);
    if (mrtrelement->min_edge_size() < lmin) lmin = mrtrelement->min_edge_size();
  }

  // loop over all master elements on this proc.
  for (int i = 0; i < melefullmap->num_my_elements(); ++i)
  {
    Core::Elements::Element* element = idiscret_->g_element(melefullmap->gid(i));
    if (!element) FOUR_C_THROW("Cannot find element with gid {}", melefullmap->gid(i));
    auto* mrtrelement = dynamic_cast<Mortar::Element*>(element);
    if (mrtrelement->min_edge_size() < lmin) lmin = mrtrelement->min_edge_size();
  }

  // compute DOP inflation length
  enlarge = eps * lmin;

  // define dopnormals
  Core::LinAlg::SerialDenseMatrix dopnormals;
  int kdop = 0;

  if (dim_ == 2)
  {
    kdop = 8;

    // setup normals for DOP
    dopnormals.reshape(4, 3);
    dopnormals(0, 0) = 1;
    dopnormals(0, 1) = 0;
    dopnormals(0, 2) = 0;
    dopnormals(1, 0) = 0;
    dopnormals(1, 1) = 1;
    dopnormals(1, 2) = 0;
    dopnormals(2, 0) = 1;
    dopnormals(2, 1) = 1;
    dopnormals(2, 2) = 0;
    dopnormals(3, 0) = -1;
    dopnormals(3, 1) = 1;
    dopnormals(3, 2) = 0;
  }
  else if (dim_ == 3)
  {
    kdop = 18;

    // setup normals for DOP
    dopnormals.reshape(9, 3);
    dopnormals(0, 0) = 1;
    dopnormals(0, 1) = 0;
    dopnormals(0, 2) = 0;
    dopnormals(1, 0) = 0;
    dopnormals(1, 1) = 1;
    dopnormals(1, 2) = 0;
    dopnormals(2, 0) = 0;
    dopnormals(2, 1) = 0;
    dopnormals(2, 2) = 1;
    dopnormals(3, 0) = 1;
    dopnormals(3, 1) = 1;
    dopnormals(3, 2) = 0;
    dopnormals(4, 0) = 1;
    dopnormals(4, 1) = 0;
    dopnormals(4, 2) = 1;
    dopnormals(5, 0) = 0;
    dopnormals(5, 1) = 1;
    dopnormals(5, 2) = 1;
    dopnormals(6, 0) = 1;
    dopnormals(6, 1) = 0;
    dopnormals(6, 2) = -1;
    dopnormals(7, 0) = -1;
    dopnormals(7, 1) = 1;
    dopnormals(7, 2) = 0;
    dopnormals(8, 0) = 0;
    dopnormals(8, 1) = -1;
    dopnormals(8, 2) = 1;
  }
  else
    FOUR_C_THROW("Problem dimension must be either 2D or 3D.");

  // decide whether auxiliary positions are used when computing dops
  const bool useauxpos = search_use_aux_pos();

  // define slave and master slabs
  Core::LinAlg::SerialDenseMatrix sslabs(kdop / 2, 2);
  Core::LinAlg::SerialDenseMatrix mslabs(kdop / 2, 2);

  //**********************************************************************
  // perform brute-force search (element-based)
  //**********************************************************************
  // for every slave element
  for (int i = 0; i < selecolmap_->num_my_elements(); i++)
  {
    // calculate slabs
    double dcurrent = 0.0;

    // initialize slabs with first node
    int sgid = selecolmap_->gid(i);
    Core::Elements::Element* element = idiscret_->g_element(sgid);
    if (!element) FOUR_C_THROW("Cannot find element with gid {}", sgid);
    Core::Nodes::Node** node = element->nodes();
    auto* mrtrnode = dynamic_cast<Node*>(node[0]);
    const double* posnode = mrtrnode->xspatial();

    // calculate slabs initialization
    for (int j = 0; j < kdop / 2; j++)
    {
      //= ax+by+cz=d/sqrt(aa+bb+cc)
      sslabs(j, 0) = sslabs(j, 1) =
          (dopnormals(j, 0) * posnode[0] + dopnormals(j, 1) * posnode[1] +
              dopnormals(j, 2) * posnode[2]) /
          std::sqrt((dopnormals(j, 0) * dopnormals(j, 0)) + (dopnormals(j, 1) * dopnormals(j, 1)) +
                    (dopnormals(j, 2) * dopnormals(j, 2)));
    }

    // for int j=1, because of initialization done before
    for (int j = 1; j < element->num_node(); j++)
    {
      auto* mrtrnode = dynamic_cast<Node*>(node[j]);
      posnode = mrtrnode->xspatial();

      for (int k = 0; k < kdop / 2; k++)
      {
        //= ax+by+cz=d/sqrt(aa+bb+cc)
        dcurrent = (dopnormals(k, 0) * posnode[0] + dopnormals(k, 1) * posnode[1] +
                       dopnormals(k, 2) * posnode[2]) /
                   std::sqrt((dopnormals(k, 0) * dopnormals(k, 0)) +
                             (dopnormals(k, 1) * dopnormals(k, 1)) +
                             (dopnormals(k, 2) * dopnormals(k, 2)));
        if (dcurrent > sslabs(k, 1)) sslabs(k, 1) = dcurrent;
        if (dcurrent < sslabs(k, 0)) sslabs(k, 0) = dcurrent;
      }
    }

    // add auxiliary positions
    // (last converged positions for all slave nodes)
    if (useauxpos)
    {
      for (int j = 0; j < element->num_node(); j++)
      {
        // get pointer to slave node
        auto* mrtrnode = dynamic_cast<Node*>(node[j]);

        std::array<double, 3> auxpos = {0.0, 0.0, 0.0};
        double scalar = 0.0;
        for (int k = 0; k < dim_; k++)
          scalar += (mrtrnode->x()[k] + mrtrnode->uold()[k] - mrtrnode->xspatial()[k]) *
                    mrtrnode->mo_data().n()[k];
        for (int k = 0; k < dim_; k++)
          auxpos[k] = mrtrnode->xspatial()[k] + scalar * mrtrnode->mo_data().n()[k];

        for (int l = 0; l < kdop / 2; l++)
        {
          //= ax+by+cz=d/sqrt(aa+bb+cc)
          dcurrent = (dopnormals(l, 0) * auxpos[0] + dopnormals(l, 1) * auxpos[1] +
                         dopnormals(l, 2) * auxpos[2]) /
                     std::sqrt((dopnormals(l, 0) * dopnormals(l, 0)) +
                               (dopnormals(l, 1) * dopnormals(l, 1)) +
                               (dopnormals(l, 2) * dopnormals(l, 2)));
          if (dcurrent > sslabs(l, 1)) sslabs(l, 1) = dcurrent;
          if (dcurrent < sslabs(l, 0)) sslabs(l, 0) = dcurrent;
        }
      }
    }

    // enlarge slabs with scalar factor
    for (int j = 0; j < kdop / 2; j++)
    {
      sslabs(j, 0) = sslabs(j, 0) - enlarge;
      sslabs(j, 1) = sslabs(j, 1) + enlarge;
    }

    // for every master element
    for (int j = 0; j < melefullmap->num_my_elements(); j++)
    {
      // calculate slabs
      double dcurrent = 0.0;

      // initialize slabs with first node
      int mgid = melefullmap->gid(j);
      Core::Elements::Element* element = idiscret_->g_element(mgid);
      if (!element) FOUR_C_THROW("Cannot find element with gid {}", mgid);
      Core::Nodes::Node** node = element->nodes();
      auto* mrtrnode = dynamic_cast<Node*>(node[0]);
      const double* posnode = mrtrnode->xspatial();

      // calculate slabs initialization
      for (int k = 0; k < kdop / 2; k++)
      {
        //= ax+by+cz=d/sqrt(aa+bb+cc)
        mslabs(k, 0) = mslabs(k, 1) =
            (dopnormals(k, 0) * posnode[0] + dopnormals(k, 1) * posnode[1] +
                dopnormals(k, 2) * posnode[2]) /
            std::sqrt((dopnormals(k, 0) * dopnormals(k, 0)) +
                      (dopnormals(k, 1) * dopnormals(k, 1)) +
                      (dopnormals(k, 2) * dopnormals(k, 2)));
      }

      // for int k=1, because of initialization done before
      for (int k = 1; k < element->num_node(); k++)
      {
        auto* mrtrnode = dynamic_cast<Node*>(node[k]);
        posnode = mrtrnode->xspatial();

        for (int l = 0; l < kdop / 2; l++)
        {
          //= d=ax+by+cz/sqrt(aa+bb+cc)
          dcurrent = (dopnormals(l, 0) * posnode[0] + dopnormals(l, 1) * posnode[1] +
                         dopnormals(l, 2) * posnode[2]) /
                     std::sqrt((dopnormals(l, 0) * dopnormals(l, 0)) +
                               (dopnormals(l, 1) * dopnormals(l, 1)) +
                               (dopnormals(l, 2) * dopnormals(l, 2)));
          if (dcurrent > mslabs(l, 1)) mslabs(l, 1) = dcurrent;
          if (dcurrent < mslabs(l, 0)) mslabs(l, 0) = dcurrent;
        }
      }

      // enlarge slabs with scalar factor
      for (int k = 0; k < kdop / 2; k++)
      {
        mslabs(k, 0) = mslabs(k, 0) - enlarge;
        mslabs(k, 1) = mslabs(k, 1) + enlarge;
      }

      // check if slabs of current master and slave element intercept
      int nintercepts = 0;
      for (int k = 0; k < kdop / 2; k++)
      {
        if ((sslabs(k, 0) <= mslabs(k, 0) && sslabs(k, 1) >= mslabs(k, 0)) ||
            (mslabs(k, 1) >= sslabs(k, 0) && mslabs(k, 0) <= sslabs(k, 0)) ||
            (sslabs(k, 0) <= mslabs(k, 0) && sslabs(k, 1) >= mslabs(k, 1)) ||
            (sslabs(k, 0) >= mslabs(k, 0) && mslabs(k, 1) >= sslabs(k, 1)))
        {
          nintercepts++;
        }
      }

      // std::cout <<"\n"<< Core::Communication::my_mpi_rank(Comm()) << " Number of intercepts
      // found: " << nintercepts ;

      // slabs of current master and slave element do intercept
      if (nintercepts == kdop / 2)
      {
        // std::cout << Core::Communication::my_mpi_rank(Comm()) << " Coupling found between slave
        // element: " << sgid <<" and master element: "<< mgid << std::endl;
        Core::Elements::Element* element = idiscret_->g_element(sgid);
        auto* selement = dynamic_cast<Mortar::Element*>(element);
        selement->add_search_elements(mgid);
      }
    }  // for all master elements
  }  // for all slave elements
}

/*----------------------------------------------------------------------*
 |  Search for potentially coupling sl/ma pairs (public)      popp 10/08|
 *----------------------------------------------------------------------*/
bool Mortar::Interface::evaluate_search_binarytree()
{
  binarytree_->evaluate_search();

  return true;
}

/*----------------------------------------------------------------------*
 |  Integrate matrix M and gap g on slave/master overlap      popp 11/08|
 *----------------------------------------------------------------------*/
bool Mortar::Interface::mortar_coupling(Mortar::Element* sele, std::vector<Mortar::Element*> mele,
    const std::shared_ptr<Mortar::ParamsInterface>& mparams_ptr)
{
  pre_mortar_coupling(sele, mele, mparams_ptr);

  // check if quadratic interpolation is involved
  bool quadratic = false;
  if (sele->is_quad()) quadratic = true;
  for (auto& m : mele)
    if (m->is_quad()) quadratic = true;

  // *********************************************************************
  // do interface coupling within a new class
  // (projection slave and master, overlap detection, integration and
  // linearization of the Mortar matrix M)
  // ************************************************************** 2D ***
  if (n_dim() == 2)
  {
    // *************************************************** linear 2D ***
    // ************************************************ quadratic 2D ***
    // neither quadratic interpolation nor mixed linear and quadratic
    // interpolation need any special treatment in the 2d case

    // create Coupling2dManager and evaluate
    Mortar::Coupling2dManager(discret(), n_dim(), quadratic, interface_params(), sele, mele)
        .evaluate_coupling(mparams_ptr);
  }
  // ************************************************************** 3D ***
  else if (n_dim() == 3)
  {
    // *************************************************** linear 3D ***
    if (!quadratic)
    {
      // create Coupling3dManager and evaluate
      Mortar::Coupling3dManager(discret(), n_dim(), false, interface_params(), sele, mele)
          .evaluate_coupling(mparams_ptr);
    }

    // ************************************************** quadratic 3D ***
    else
    {
      // create Coupling3dQuadManager and evaluate
      Mortar::Coupling3dQuadManager(discret(), n_dim(), false, interface_params(), sele, mele)
          .evaluate_coupling(mparams_ptr);
    }  // quadratic
  }  // 3D
  else
    FOUR_C_THROW("Dimension for Mortar coupling must be either 2D or 3D.");
  // *********************************************************************

  post_mortar_coupling(sele, mele, mparams_ptr);

  return true;
}

/*----------------------------------------------------------------------*
 | Split Mortar::Elements->IntElements for 3D quad. coupling    popp 03/09|
 *----------------------------------------------------------------------*/
bool Mortar::Interface::split_int_elements(
    Mortar::Element& ele, std::vector<std::shared_ptr<Mortar::IntElement>>& auxele)
{
  // *********************************************************************
  // do splitting for given element
  // *********************************************************** quad9 ***
  if (ele.shape() == Core::FE::CellType::quad9)
  {
    // split into for quad4 elements
    int numnode = 4;
    Core::FE::CellType dt = Core::FE::CellType::quad4;

    // first integration element
    // containing parent nodes 0,4,8,7
    int nodeids[4] = {0, 0, 0, 0};
    nodeids[0] = ele.node_ids()[0];
    nodeids[1] = ele.node_ids()[4];
    nodeids[2] = ele.node_ids()[8];
    nodeids[3] = ele.node_ids()[7];

    std::vector<Core::Nodes::Node*> nodes(4);
    nodes[0] = ele.nodes()[0];
    nodes[1] = ele.nodes()[4];
    nodes[2] = ele.nodes()[8];
    nodes[3] = ele.nodes()[7];

    auxele.push_back(std::make_shared<IntElement>(
        0, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));

    // second integration element
    // containing parent nodes 4,1,5,8
    nodeids[0] = ele.node_ids()[4];
    nodeids[1] = ele.node_ids()[1];
    nodeids[2] = ele.node_ids()[5];
    nodeids[3] = ele.node_ids()[8];

    nodes[0] = ele.nodes()[4];
    nodes[1] = ele.nodes()[1];
    nodes[2] = ele.nodes()[5];
    nodes[3] = ele.nodes()[8];

    auxele.push_back(std::make_shared<IntElement>(
        1, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));

    // third integration element
    // containing parent nodes 8,5,2,6
    nodeids[0] = ele.node_ids()[8];
    nodeids[1] = ele.node_ids()[5];
    nodeids[2] = ele.node_ids()[2];
    nodeids[3] = ele.node_ids()[6];

    nodes[0] = ele.nodes()[8];
    nodes[1] = ele.nodes()[5];
    nodes[2] = ele.nodes()[2];
    nodes[3] = ele.nodes()[6];

    auxele.push_back(std::make_shared<IntElement>(
        2, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));

    // fourth integration element
    // containing parent nodes 7,8,6,3
    nodeids[0] = ele.node_ids()[7];
    nodeids[1] = ele.node_ids()[8];
    nodeids[2] = ele.node_ids()[6];
    nodeids[3] = ele.node_ids()[3];

    nodes[0] = ele.nodes()[7];
    nodes[1] = ele.nodes()[8];
    nodes[2] = ele.nodes()[6];
    nodes[3] = ele.nodes()[3];

    auxele.push_back(std::make_shared<IntElement>(
        3, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));
  }

  // *********************************************************** quad8 ***
  else if (ele.shape() == Core::FE::CellType::quad8)
  {
    // split into four tri3 elements and one quad4 element
    int numnodetri = 3;
    int numnodequad = 4;
    Core::FE::CellType dttri = Core::FE::CellType::tri3;
    Core::FE::CellType dtquad = Core::FE::CellType::quad4;

    // first integration element
    // containing parent nodes 0,4,7
    int nodeids[3] = {0, 0, 0};
    nodeids[0] = ele.node_ids()[0];
    nodeids[1] = ele.node_ids()[4];
    nodeids[2] = ele.node_ids()[7];

    std::vector<Core::Nodes::Node*> nodes(3);
    nodes[0] = ele.nodes()[0];
    nodes[1] = ele.nodes()[4];
    nodes[2] = ele.nodes()[7];

    auxele.push_back(std::make_shared<IntElement>(
        0, ele.id(), ele.owner(), &ele, dttri, numnodetri, nodeids, nodes, ele.is_slave(), false));

    // second integration element
    // containing parent nodes 1,5,4
    nodeids[0] = ele.node_ids()[1];
    nodeids[1] = ele.node_ids()[5];
    nodeids[2] = ele.node_ids()[4];

    nodes[0] = ele.nodes()[1];
    nodes[1] = ele.nodes()[5];
    nodes[2] = ele.nodes()[4];

    auxele.push_back(std::make_shared<IntElement>(
        1, ele.id(), ele.owner(), &ele, dttri, numnodetri, nodeids, nodes, ele.is_slave(), false));

    // third integration element
    // containing parent nodes 2,6,5
    nodeids[0] = ele.node_ids()[2];
    nodeids[1] = ele.node_ids()[6];
    nodeids[2] = ele.node_ids()[5];

    nodes[0] = ele.nodes()[2];
    nodes[1] = ele.nodes()[6];
    nodes[2] = ele.nodes()[5];

    auxele.push_back(std::make_shared<IntElement>(
        2, ele.id(), ele.owner(), &ele, dttri, numnodetri, nodeids, nodes, ele.is_slave(), false));

    // fourth integration element
    // containing parent nodes 3,7,6
    nodeids[0] = ele.node_ids()[3];
    nodeids[1] = ele.node_ids()[7];
    nodeids[2] = ele.node_ids()[6];

    nodes[0] = ele.nodes()[3];
    nodes[1] = ele.nodes()[7];
    nodes[2] = ele.nodes()[6];

    auxele.push_back(std::make_shared<IntElement>(
        3, ele.id(), ele.owner(), &ele, dttri, numnodetri, nodeids, nodes, ele.is_slave(), false));

    // fifth integration element
    // containing parent nodes 4,5,6,7
    int nodeidsquad[4] = {0, 0, 0, 0};
    nodeidsquad[0] = ele.node_ids()[4];
    nodeidsquad[1] = ele.node_ids()[5];
    nodeidsquad[2] = ele.node_ids()[6];
    nodeidsquad[3] = ele.node_ids()[7];

    std::vector<Core::Nodes::Node*> nodesquad(4);
    nodesquad[0] = ele.nodes()[4];
    nodesquad[1] = ele.nodes()[5];
    nodesquad[2] = ele.nodes()[6];
    nodesquad[3] = ele.nodes()[7];

    auxele.push_back(std::make_shared<IntElement>(4, ele.id(), ele.owner(), &ele, dtquad,
        numnodequad, nodeidsquad, nodesquad, ele.is_slave(), false));
  }

  // ************************************************************ tri6 ***
  else if (ele.shape() == Core::FE::CellType::tri6)
  {
    // split into four tri3 elements
    int numnode = 3;
    Core::FE::CellType dt = Core::FE::CellType::tri3;

    // first integration element
    // containing parent nodes 0,3,5
    int nodeids[3] = {0, 0, 0};
    nodeids[0] = ele.node_ids()[0];
    nodeids[1] = ele.node_ids()[3];
    nodeids[2] = ele.node_ids()[5];

    std::vector<Core::Nodes::Node*> nodes(3);
    nodes[0] = ele.nodes()[0];
    nodes[1] = ele.nodes()[3];
    nodes[2] = ele.nodes()[5];

    auxele.push_back(std::make_shared<IntElement>(
        0, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));

    // second integration element
    // containing parent nodes 3,1,4
    nodeids[0] = ele.node_ids()[3];
    nodeids[1] = ele.node_ids()[1];
    nodeids[2] = ele.node_ids()[4];

    nodes[0] = ele.nodes()[3];
    nodes[1] = ele.nodes()[1];
    nodes[2] = ele.nodes()[4];

    auxele.push_back(std::make_shared<IntElement>(
        1, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));

    // third integration element
    // containing parent nodes 5,4,2
    nodeids[0] = ele.node_ids()[5];
    nodeids[1] = ele.node_ids()[4];
    nodeids[2] = ele.node_ids()[2];

    nodes[0] = ele.nodes()[5];
    nodes[1] = ele.nodes()[4];
    nodes[2] = ele.nodes()[2];

    auxele.push_back(std::make_shared<IntElement>(
        2, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));

    // fourth integration element
    // containing parent nodes 4,5,3
    nodeids[0] = ele.node_ids()[4];
    nodeids[1] = ele.node_ids()[5];
    nodeids[2] = ele.node_ids()[3];

    nodes[0] = ele.nodes()[4];
    nodes[1] = ele.nodes()[5];
    nodes[2] = ele.nodes()[3];

    auxele.push_back(std::make_shared<IntElement>(
        3, ele.id(), ele.owner(), &ele, dt, numnode, nodeids, nodes, ele.is_slave(), false));
  }

  // *********************************************************** quad4 ***
  else if (ele.shape() == Core::FE::CellType::quad4)
  {
    // 1:1 conversion to IntElement
    std::vector<Core::Nodes::Node*> nodes(4);
    nodes[0] = ele.nodes()[0];
    nodes[1] = ele.nodes()[1];
    nodes[2] = ele.nodes()[2];
    nodes[3] = ele.nodes()[3];

    auxele.push_back(std::make_shared<IntElement>(0, ele.id(), ele.owner(), &ele, ele.shape(),
        ele.num_node(), ele.node_ids(), nodes, ele.is_slave(), false));
  }

  // ************************************************************ tri3 ***
  else if (ele.shape() == Core::FE::CellType::tri3)
  {
    // 1:1 conversion to IntElement
    std::vector<Core::Nodes::Node*> nodes(3);
    nodes[0] = ele.nodes()[0];
    nodes[1] = ele.nodes()[1];
    nodes[2] = ele.nodes()[2];

    auxele.push_back(std::make_shared<IntElement>(0, ele.id(), ele.owner(), &ele, ele.shape(),
        ele.num_node(), ele.node_ids(), nodes, ele.is_slave(), false));
  }

  // ********************************************************* invalid ***
  else
    FOUR_C_THROW("split_int_elements called for unknown element shape!");

  // *********************************************************************

  return true;
}

/*----------------------------------------------------------------------*
 |  Assemble geometry-dependent lagrange multipliers (global)      popp 05/09|
 *----------------------------------------------------------------------*/
void Mortar::Interface::assemble_lm(Core::LinAlg::Vector<double>& zglobal)
{
  // loop over all slave nodes
  for (int j = 0; j < snoderowmap_->num_my_elements(); ++j)
  {
    int gid = snoderowmap_->gid(j);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    int dim = mrtrnode->num_dof();
    double* lm = mrtrnode->mo_data().lm();

    Core::LinAlg::SerialDenseVector lmnode(dim);
    std::vector<int> lmdof(dim);
    std::vector<int> lmowner(dim);

    for (int k = 0; k < dim; ++k)
    {
      lmnode(k) = lm[k];
      lmdof[k] = mrtrnode->dofs()[k];
      lmowner[k] = mrtrnode->owner();
    }

    // do assembly
    Core::LinAlg::assemble(zglobal, lmnode, lmdof, lmowner);
  }
}


/*----------------------------------------------------------------------*
 |  Assemble Mortar D matrix                                  popp 01/08|
 *----------------------------------------------------------------------*/
void Mortar::Interface::assemble_d(Core::LinAlg::SparseMatrix& dglobal)
{
  const bool nonsmooth = interface_params().get<bool>("NONSMOOTH_GEOMETRIES");
  const bool lagmultlin = (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                               interface_params(), "LM_QUAD") == Inpar::Mortar::lagmult_lin);

  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    /**************************************************** D-matrix ******/
    if ((mrtrnode->mo_data().get_d()).size() > 0)
    {
      const Core::Gen::Pairedvector<int, double>& dmap = mrtrnode->mo_data().get_d();
      int rowsize = mrtrnode->num_dof();

      Core::Gen::Pairedvector<int, double>::const_iterator colcurr;

      for (colcurr = dmap.begin(); colcurr != dmap.end(); ++colcurr)
      {
        double val = colcurr->second;

        Core::Nodes::Node* knode = discret().g_node(colcurr->first);
        if (!knode) FOUR_C_THROW("node not found");
        auto* kcnode = dynamic_cast<Node*>(knode);
        if (!kcnode) FOUR_C_THROW("node not found");

        for (int j = 0; j < rowsize; ++j)
        {
          int row = mrtrnode->dofs()[j];
          int col = kcnode->dofs()[j];

          // do the assembly into global D matrix
          if (!nonsmooth and (shapefcn_ == Inpar::Mortar::shape_dual or
                                 shapefcn_ == Inpar::Mortar::shape_petrovgalerkin))
          {
            if (lagmultlin)
            {
              // do lumping of D-matrix
              // create an explicitly diagonal d matrix
              dglobal.assemble(val, row, row);
            }
            else
            {
              // check for diagonality
              if (row != col && abs(val) > 1.0e-12) FOUR_C_THROW("D-Matrix is not diagonal!");

              // create an explicitly diagonal d matrix
              if (row == col) dglobal.assemble(val, row, col);
            }
          }
          else if (nonsmooth or shapefcn_ == Inpar::Mortar::shape_standard)
          {
            // don't check for diagonality
            // since for standard shape functions, as in general when using
            // arbitrary shape function types, this is not the case

            // create the d matrix, do not assemble zeros
            dglobal.assemble(val, row, col);
          }
        }
      }
    }
  }
}


/*----------------------------------------------------------------------*
 |  Assemble Mortar M matrix                                  popp 01/08|
 *----------------------------------------------------------------------*/
void Mortar::Interface::assemble_m(Core::LinAlg::SparseMatrix& mglobal)
{
  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    /**************************************************** M-matrix ******/
    if ((mrtrnode->mo_data().get_m()).size() > 0)
    {
      const std::map<int, double>& mmap = mrtrnode->mo_data().get_m();
      int rowsize = mrtrnode->num_dof();

      std::map<int, double>::const_iterator colcurr;

      for (colcurr = mmap.begin(); colcurr != mmap.end(); ++colcurr)
      {
        Core::Nodes::Node* knode = discret().g_node(colcurr->first);
        if (!knode) FOUR_C_THROW("node not found");
        auto* kcnode = dynamic_cast<Node*>(knode);
        if (!kcnode) FOUR_C_THROW("node not found");

        double val = colcurr->second;

        for (int j = 0; j < rowsize; ++j)
        {
          int row = mrtrnode->dofs()[j];
          int col = kcnode->dofs()[j];

          // do not assemble zeros into m matrix
          //          if (abs(val) > 1.0e-12)
          mglobal.assemble(val, row, col);
        }
      }
    }

    /************************************************* Mmod-matrix ******/
    if ((mrtrnode->mo_data().get_mmod()).size() > 0)
    {
      std::map<int, double>& mmap = mrtrnode->mo_data().get_mmod();
      int rowsize = mrtrnode->num_dof();
      int colsize = static_cast<int>(mmap.size()) * rowsize;

      Core::LinAlg::SerialDenseMatrix Mnode(rowsize, colsize);
      std::vector<int> lmrow(rowsize);
      std::vector<int> lmcol(colsize);
      std::vector<int> lmrowowner(rowsize);
      std::map<int, double>::const_iterator colcurr;
      int k = 0;

      for (colcurr = mmap.begin(); colcurr != mmap.end(); ++colcurr)
      {
        Core::Nodes::Node* knode = discret().g_node(colcurr->first);
        if (!knode) FOUR_C_THROW("node not found");
        auto* kcnode = dynamic_cast<Node*>(knode);
        if (!kcnode) FOUR_C_THROW("node not found");

        for (int j = 0; j < rowsize; ++j)
        {
          int row = mrtrnode->dofs()[j];
          lmrow[j] = row;
          lmrowowner[j] = mrtrnode->owner();

          int col = kcnode->dofs()[j];
          double val = colcurr->second;
          lmcol[k] = col;

          Mnode(j, k) = val;
          ++k;
        }
      }

      mglobal.assemble(-1, Mnode, lmrow, lmrowowner, lmcol);
    }
  }
}


/*----------------------------------------------------------------------*
 |  Assemble Mortar matrices                                 farah 02/16|
 *----------------------------------------------------------------------*/
void Mortar::Interface::assemble_dm(
    Core::LinAlg::SparseMatrix& dglobal, Core::LinAlg::SparseMatrix& mglobal)
{
  // call subroutines:

  // assemble mortar matrix D (slave side)
  assemble_d(dglobal);

  // assemble mortar matrix M (master side)
  assemble_m(mglobal);
}


/*----------------------------------------------------------------------*
 |  Assemble matrix of normals                                popp 10/11|
 *----------------------------------------------------------------------*/
void Mortar::Interface::assemble_normals(Core::LinAlg::SparseMatrix& nglobal)
{
  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    // nodal normal
    double* nodalnormal = mrtrnode->mo_data().n();

    // add normal to corresponding row in global matrix
    for (int k = 0; k < mrtrnode->num_dof(); ++k)
      nglobal.assemble(nodalnormal[k], gid, mrtrnode->dofs()[k]);
  }
}

/*----------------------------------------------------------------------*
 |  Assemble interface displacement trafo matrices            popp 06/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::assemble_trafo(Core::LinAlg::SparseMatrix& trafo,
    Core::LinAlg::SparseMatrix& invtrafo, std::set<int>& donebefore)
{
  // check for dual shape functions and quadratic slave elements
  if (shapefcn_ == Inpar::Mortar::shape_standard || !quadslave_)
    FOUR_C_THROW("AssembleTrafo -> you should not be here...");

  // check whether locally linear LM interpolation is used
  const bool lagmultlin = (Teuchos::getIntegralValue<Inpar::Mortar::LagMultQuad>(
                               interface_params(), "LM_QUAD") == Inpar::Mortar::lagmult_lin);

  //********************************************************************
  //********************************************************************
  // LOOP OVER ALL SLAVE NODES
  //********************************************************************
  //********************************************************************
  // loop over proc's slave nodes of the interface for assembly
  // use standard row map to assemble each node only once
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
      FOUR_C_THROW("Node ownership inconsistency!");

    // find out whether this is a corner node (no trafo), an edge
    // node (trafo of displacement DOFs) or a center node (only for
    // quad9, theoretically trafo of displacement DOFs but not used)
    // and also store transformation factor theta
    enum NodeType
    {
      corner,
      edge,
      center,
      undefined
    };
    NodeType nt = undefined;
    double theta = 0.0;

    // search within the first adjacent element
    auto* mrtrele = dynamic_cast<Mortar::Element*>(mrtrnode->adjacent_elements()[0].user_element());
    Core::FE::CellType shape = mrtrele->shape();

    // which discretization type
    switch (shape)
    {
      // line3 contact elements (= tri6||quad8||quad9 discretizations)
      case Core::FE::CellType::line3:
      {
        // modification factor
        if (Inpar::Mortar::LagMultQuad() == Inpar::Mortar::lagmult_lin)
          theta = 1.0 / 2.0;
        else
          theta = 1.0 / 5.0;

        // corner nodes
        if (mrtrnode->id() == mrtrele->node_ids()[0] || mrtrnode->id() == mrtrele->node_ids()[1])
        {
          nt = corner;
        }

        // edge nodes
        else if (mrtrnode->id() == mrtrele->node_ids()[2])
        {
          nt = edge;
        }

        break;
      }

      // tri6 contact elements (= tet10 discretizations)
      case Core::FE::CellType::tri6:
      {
        // modification factor
        theta = 1.0 / 5.0;

        // corner nodes
        if (mrtrnode->id() == mrtrele->node_ids()[0] || mrtrnode->id() == mrtrele->node_ids()[1] ||
            mrtrnode->id() == mrtrele->node_ids()[2])
        {
          nt = corner;
        }

        // edge nodes
        else if (mrtrnode->id() == mrtrele->node_ids()[3] ||
                 mrtrnode->id() == mrtrele->node_ids()[4] ||
                 mrtrnode->id() == mrtrele->node_ids()[5])
        {
          nt = edge;
        }

        break;
      }

        // quad8 contact elements (= hex20 discretizations)
      case Core::FE::CellType::quad8:
      {
        // modification factor
        theta = 1.0 / 5.0;

        // corner nodes
        if (mrtrnode->id() == mrtrele->node_ids()[0] || mrtrnode->id() == mrtrele->node_ids()[1] ||
            mrtrnode->id() == mrtrele->node_ids()[2] || mrtrnode->id() == mrtrele->node_ids()[3])
        {
          nt = corner;
        }

        // edge nodes
        else if (mrtrnode->id() == mrtrele->node_ids()[4] ||
                 mrtrnode->id() == mrtrele->node_ids()[5] ||
                 mrtrnode->id() == mrtrele->node_ids()[6] ||
                 mrtrnode->id() == mrtrele->node_ids()[7])
        {
          nt = edge;
        }

        break;
      }

        // quad9 contact elements (= hex27 discretizations)
        // *************************************************
        // ** currently we only use this modification for **
        // ** tri6 and quad8 surfaces, but NOT for quad9  **
        // ** as in this case, there is no real need!     **
        // ** (positivity of shape function integrals)    **
        // ** thus, we simply want to assemble the        **
        // ** identity matrix here, which we achieve by   **
        // ** setting the trafo factor theta = 0.0!       **
        // *************************************************
      case Core::FE::CellType::quad9:
      {
        // modification factor
        theta = 0.0;

        // corner nodes
        if (mrtrnode->id() == mrtrele->node_ids()[0] || mrtrnode->id() == mrtrele->node_ids()[1] ||
            mrtrnode->id() == mrtrele->node_ids()[2] || mrtrnode->id() == mrtrele->node_ids()[3])
        {
          nt = corner;
        }

        // edge nodes
        else if (mrtrnode->id() == mrtrele->node_ids()[4] ||
                 mrtrnode->id() == mrtrele->node_ids()[5] ||
                 mrtrnode->id() == mrtrele->node_ids()[6] ||
                 mrtrnode->id() == mrtrele->node_ids()[7])
        {
          nt = edge;
        }

        // center node
        else if (mrtrnode->id() == mrtrele->node_ids()[8])
        {
          nt = center;
        }

        break;
      }

        // other cases
      default:
      {
        FOUR_C_THROW("Trafo matrix only for line3/tri6/quad8/quad9 contact elements");
        break;
      }
    }  // switch(Shape)

    //********************************************************************
    // CASE 1: CORNER NODES AND CENTER NODE
    //********************************************************************
    if (nt == corner || nt == center)
    {
      // check if processed before
      auto iter = donebefore.find(gid);

      // if not then assemble trafo matrix block
      if (iter == donebefore.end())
      {
        // add to set of processed nodes
        donebefore.insert(gid);

        // add transformation matrix block (unity block!)
        for (int k = 0; k < mrtrnode->num_dof(); ++k)
        {
          // assemble diagonal values
          trafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
          invtrafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
        }
      }
    }

    //********************************************************************
    // CASE 2: EDGE NODES
    //********************************************************************
    else if (nt == edge)
    {
      // check if processed before
      auto iter = donebefore.find(gid);

      // if not then assemble trafo matrix block
      if (iter == donebefore.end())
      {
        // add to set of processed nodes
        donebefore.insert(gid);

        // find adjacent corner nodes locally
        int index1 = 0;
        int index2 = 0;
        int hoindex = mrtrele->get_local_node_id(gid);
        Core::FE::get_corner_node_indices(index1, index2, hoindex, shape);

        // find adjacent corner nodes globally
        int gindex1 = mrtrele->node_ids()[index1];
        int gindex2 = mrtrele->node_ids()[index2];
        // std::cout << "-> adjacent corner nodes: " << gindex1 << " " << gindex2 << std::endl;
        Core::Nodes::Node* adjnode1 = idiscret_->g_node(gindex1);
        if (!adjnode1) FOUR_C_THROW("Cannot find node with gid %", gindex1);
        auto* adjmrtrnode1 = dynamic_cast<Node*>(adjnode1);
        Core::Nodes::Node* adjnode2 = idiscret_->g_node(gindex2);
        if (!adjnode2) FOUR_C_THROW("Cannot find node with gid %", gindex2);
        auto* adjmrtrnode2 = dynamic_cast<Node*>(adjnode2);

        // add transformation matrix block
        for (int k = 0; k < mrtrnode->num_dof(); ++k)
        {
          // assemble diagonal values
          trafo.assemble(1.0 - 2 * theta, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
          invtrafo.assemble(1.0 / (1.0 - 2 * theta), mrtrnode->dofs()[k], mrtrnode->dofs()[k]);

          // assemble off-diagonal values
          trafo.assemble(theta, mrtrnode->dofs()[k], adjmrtrnode1->dofs()[k]);
          trafo.assemble(theta, mrtrnode->dofs()[k], adjmrtrnode2->dofs()[k]);
          invtrafo.assemble(
              -theta / (1.0 - 2 * theta), mrtrnode->dofs()[k], adjmrtrnode1->dofs()[k]);
          invtrafo.assemble(
              -theta / (1.0 - 2 * theta), mrtrnode->dofs()[k], adjmrtrnode2->dofs()[k]);
        }
      }
    }

    //********************************************************************
    // CASE 3: UNDEFINED NODES
    //********************************************************************
    else
    {
      FOUR_C_THROW("Undefined node type (corner, edge, center)");
    }
  }

  // assembly for locally linear LM interpolation
  if (lagmultlin)
  {
    //********************************************************************
    //********************************************************************
    // LOOP OVER ALL MASTER NODES
    //********************************************************************
    //********************************************************************
    // loop over proc's master nodes of the interface for assembly
    // use standard row map to assemble each node only once
    for (int i = 0; i < mnoderowmap_->num_my_elements(); ++i)
    {
      int gid = mnoderowmap_->gid(i);
      Core::Nodes::Node* node = idiscret_->g_node(gid);
      if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
      auto* mrtrnode = dynamic_cast<Node*>(node);

      if (mrtrnode->owner() != Core::Communication::my_mpi_rank(get_comm()))
        FOUR_C_THROW("AssembleTrafo: Node ownership inconsistency!");

      // find out whether this is a "real" master node (no trafo), a former
      // slave edge node (trafo of displacement DOFs) or a former slave
      // center node (only for quadd9, trafo of displacement DOFs)
      enum NodeType
      {
        master,
        slaveedge,
        slavecenter,
        undefined
      };
      NodeType nt = undefined;

      // search within the first adjacent element
      auto* mrtrele =
          dynamic_cast<Mortar::Element*>(mrtrnode->adjacent_elements()[0].user_element());
      Core::FE::CellType shape = mrtrele->shape();

      // real master nodes are easily identified
      if (!mrtrnode->is_on_bound()) nt = master;
      // former slave node type depends on discretization
      else
      {
        switch (shape)
        {
          case Core::FE::CellType::line3:
          {
            // edge node
            if (mrtrnode->id() == mrtrele->node_ids()[2]) nt = slaveedge;

            break;
          }

          // tri6 contact elements (= tet10 discretizations)
          case Core::FE::CellType::tri6:
          {
            // edge nodes
            if (mrtrnode->id() == mrtrele->node_ids()[3] ||
                mrtrnode->id() == mrtrele->node_ids()[4] ||
                mrtrnode->id() == mrtrele->node_ids()[5])
              nt = slaveedge;

            break;
          }

          // quad8 contact elements (= hex20 discretizations)
          case Core::FE::CellType::quad8:
          {
            // edge nodes
            if (mrtrnode->id() == mrtrele->node_ids()[4] ||
                mrtrnode->id() == mrtrele->node_ids()[5] ||
                mrtrnode->id() == mrtrele->node_ids()[6] ||
                mrtrnode->id() == mrtrele->node_ids()[7])
              nt = slaveedge;

            break;
          }

          // quad9 contact elements (= hex27 discretizations)
          case Core::FE::CellType::quad9:
          {
            // edge nodes
            if (mrtrnode->id() == mrtrele->node_ids()[4] ||
                mrtrnode->id() == mrtrele->node_ids()[5] ||
                mrtrnode->id() == mrtrele->node_ids()[6] ||
                mrtrnode->id() == mrtrele->node_ids()[7])
            {
              nt = slaveedge;
            }

            // center node
            else if (mrtrnode->id() == mrtrele->node_ids()[8])
              nt = slavecenter;

            break;
          }

          // other cases
          default:
          {
            FOUR_C_THROW("Trafo matrix only for line3/tri6/quad8/quad9 contact elements");
            break;
          }
        }  // switch(Shape)
      }

      //********************************************************************
      // CASE 1: REAL MASTER NODE
      //********************************************************************
      if (nt == master)
      {
        // check if processed before
        auto iter = donebefore.find(gid);

        // if not then assemble trafo matrix block
        if (iter == donebefore.end())
        {
          // add to set of processed nodes
          donebefore.insert(gid);

          // add transformation matrix block (unity block!)
          for (int k = 0; k < mrtrnode->num_dof(); ++k)
          {
            // assemble diagonal values
            trafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
            invtrafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
          }
        }
      }

      //********************************************************************
      // CASE 2: FORMER SLAVE EDGE NODE
      // (for linear LM interpolation -> full distribution of edge nodes)
      // (nevertheless, we keep the 1.0 on the main diagonal -> no PoU!)
      //********************************************************************
      else if (nt == slaveedge)
      {
        // check if processed before
        auto iter = donebefore.find(gid);

        // if not then assemble trafo matrix block
        if (iter == donebefore.end())
        {
          // add to set of processed nodes
          donebefore.insert(gid);

          // find adjacent corner nodes locally
          int index1 = 0;
          int index2 = 0;
          int hoindex = mrtrele->get_local_node_id(gid);
          Core::FE::get_corner_node_indices(index1, index2, hoindex, shape);

          // find adjacent corner nodes globally
          int gindex1 = mrtrele->node_ids()[index1];
          int gindex2 = mrtrele->node_ids()[index2];
          // std::cout << "-> adjacent corner nodes: " << gindex1 << " " << gindex2 << std::endl;
          Core::Nodes::Node* adjnode1 = idiscret_->g_node(gindex1);
          if (!adjnode1) FOUR_C_THROW("Cannot find node with gid %", gindex1);
          auto* adjmrtrnode1 = dynamic_cast<Node*>(adjnode1);
          Core::Nodes::Node* adjnode2 = idiscret_->g_node(gindex2);
          if (!adjnode2) FOUR_C_THROW("Cannot find node with gid %", gindex2);
          auto* adjmrtrnode2 = dynamic_cast<Node*>(adjnode2);

          // add transformation matrix block
          for (int k = 0; k < mrtrnode->num_dof(); ++k)
          {
            // assemble diagonal values
            trafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
            invtrafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);

            // assemble off-diagonal values
            trafo.assemble(0.5, mrtrnode->dofs()[k], adjmrtrnode1->dofs()[k]);
            trafo.assemble(0.5, mrtrnode->dofs()[k], adjmrtrnode2->dofs()[k]);
            invtrafo.assemble(-0.5, mrtrnode->dofs()[k], adjmrtrnode1->dofs()[k]);
            invtrafo.assemble(-0.5, mrtrnode->dofs()[k], adjmrtrnode2->dofs()[k]);
          }
        }
      }

      //********************************************************************
      // CASE 3: FORMER SLAVE CENTER NODE (QUAD9)
      // (for linear LM interpolation -> full distribution of corner nodes)
      // (nevertheless, we keep the 1.0 on the main diagonal -> no PoU!)
      //********************************************************************
      else if (nt == slavecenter)
      {
        // check if processed before
        auto iter = donebefore.find(gid);

        // if not then assemble trafo matrix block
        if (iter == donebefore.end())
        {
          // add to set of processed nodes
          donebefore.insert(gid);

          // find adjacent corner nodes globally
          int gindex1 = mrtrele->node_ids()[0];
          int gindex2 = mrtrele->node_ids()[1];
          int gindex3 = mrtrele->node_ids()[2];
          int gindex4 = mrtrele->node_ids()[3];
          // std::cout << "-> adjacent corner nodes: " << gindex1 << " " << gindex2 << std::endl;
          // std::cout << "-> adjacent corner nodes: " << gindex3 << " " << gindex4 << std::endl;
          Core::Nodes::Node* adjnode1 = idiscret_->g_node(gindex1);
          if (!adjnode1) FOUR_C_THROW("Cannot find node with gid %", gindex1);
          auto* adjmrtrnode1 = dynamic_cast<Node*>(adjnode1);
          Core::Nodes::Node* adjnode2 = idiscret_->g_node(gindex2);
          if (!adjnode2) FOUR_C_THROW("Cannot find node with gid %", gindex2);
          auto* adjmrtrnode2 = dynamic_cast<Node*>(adjnode2);
          Core::Nodes::Node* adjnode3 = idiscret_->g_node(gindex3);
          if (!adjnode3) FOUR_C_THROW("Cannot find node with gid %", gindex3);
          auto* adjmrtrnode3 = dynamic_cast<Node*>(adjnode3);
          Core::Nodes::Node* adjnode4 = idiscret_->g_node(gindex4);
          if (!adjnode4) FOUR_C_THROW("Cannot find node with gid %", gindex4);
          auto* adjmrtrnode4 = dynamic_cast<Node*>(adjnode4);

          // add transformation matrix block
          for (int k = 0; k < mrtrnode->num_dof(); ++k)
          {
            // assemble diagonal values
            trafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);
            invtrafo.assemble(1.0, mrtrnode->dofs()[k], mrtrnode->dofs()[k]);

            // assemble off-diagonal values
            trafo.assemble(0.25, mrtrnode->dofs()[k], adjmrtrnode1->dofs()[k]);
            trafo.assemble(0.25, mrtrnode->dofs()[k], adjmrtrnode2->dofs()[k]);
            trafo.assemble(0.25, mrtrnode->dofs()[k], adjmrtrnode3->dofs()[k]);
            trafo.assemble(0.25, mrtrnode->dofs()[k], adjmrtrnode4->dofs()[k]);
            invtrafo.assemble(-0.25, mrtrnode->dofs()[k], adjmrtrnode1->dofs()[k]);
            invtrafo.assemble(-0.25, mrtrnode->dofs()[k], adjmrtrnode2->dofs()[k]);
            invtrafo.assemble(-0.25, mrtrnode->dofs()[k], adjmrtrnode3->dofs()[k]);
            invtrafo.assemble(-0.25, mrtrnode->dofs()[k], adjmrtrnode4->dofs()[k]);
          }
        }
      }

      //********************************************************************
      // CASE 4: UNDEFINED NODES
      //********************************************************************
      else
        FOUR_C_THROW("Undefined node type (corner, edge, center)");
    }
  }  // end of assembly for locally linear LM interpolation
}

/*----------------------------------------------------------------------*
 |  Detect actual meshtying zone (node by node)               popp 08/10|
 *----------------------------------------------------------------------*/
void Mortar::Interface::detect_tied_slave_nodes(int& founduntied)
{
  //**********************************************************************
  // STEP 1: Build tying info for slave node row map (locally+globally)
  //**********************************************************************
  // global vector for tying info
  Core::LinAlg::Vector<double> rowtied(*snoderowmap_);

  // loop over proc's slave row nodes of the interface for detection
  for (int i = 0; i < snoderowmap_->num_my_elements(); ++i)
  {
    int gid = snoderowmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // perform detection
    const Core::Gen::Pairedvector<int, double>& dmap = mrtrnode->mo_data().get_d();
    const std::map<int, double>& mmap = mrtrnode->mo_data().get_m();
    int sized = dmap.size();
    int sizem = mmap.size();

    // found untied node
    if (sized == 0 && sizem == 0)
    {
      // increase counter
      founduntied += 1;

      // set node status to untied slave
      mrtrnode->set_tied_slave() = false;

      // set vector entry (tiedtoggle)
      (rowtied).get_values()[i] = 1.0;
    }

    // found tied node
    else if (sized > 0 && sizem > 0)
    {
      // do nothing
    }

    // found inconsistency
    else
    {
      FOUR_C_THROW("Inconsistency in tied/untied node detection");
    }
  }

  //**********************************************************************
  // STEP 2: Communicate tying info to slave node column map (globally)
  //**********************************************************************
  // communicate tying information to standard column map
  Core::LinAlg::Vector<double> coltied(*snodecolmap_);
  Core::LinAlg::export_to(rowtied, coltied);

  //**********************************************************************
  // STEP 3: Extract tying info for slave node column map (locally)
  //**********************************************************************
  // loop over proc's slave col nodes of the interface for storage
  for (int i = 0; i < snodecolmap_->num_my_elements(); ++i)
  {
    int gid = snodecolmap_->gid(i);
    Core::Nodes::Node* node = idiscret_->g_node(gid);
    if (!node) FOUR_C_THROW("Cannot find node with gid %", gid);
    auto* mrtrnode = dynamic_cast<Node*>(node);

    // check if this node is untied
    if (coltied.local_values_as_span()[i] == 1.0) mrtrnode->set_tied_slave() = false;
  }
}

/*----------------------------------------------------------------------*
 | create volume ghosting (public)                            ager 06/15|
 *----------------------------------------------------------------------*/
void Mortar::Interface::create_volume_ghosting(
    const std::map<std::string, std::shared_ptr<Core::FE::Discretization>>& discretization_map)
{
  const CONTACT::Problemtype prb =
      interface_params().get<CONTACT::Problemtype>("PROBTYPE", CONTACT::Problemtype::other);

  switch (prb)
  {
    case CONTACT::Problemtype::ssi:
    case CONTACT::Problemtype::ssi_elch:
    {
      std::vector<std::shared_ptr<Core::FE::Discretization>> tar_dis;
      FOUR_C_ASSERT(discretization_map.find("structure") != discretization_map.end(),
          "Could not find discretization 'structure'");
      FOUR_C_ASSERT(discretization_map.find("scatra") != discretization_map.end(),
          "Could not find discretization 'scatra'");
      tar_dis.emplace_back(discretization_map.at("structure"));
      tar_dis.emplace_back(discretization_map.at("scatra"));
      std::vector<std::pair<int, int>> material_map;
      material_map.emplace_back(std::pair<int, int>(0, 1));
      material_map.emplace_back(std::pair<int, int>(1, 0));

      Mortar::Utils::create_volume_ghosting(discret(), tar_dis, material_map);

      // we need to redistribute the scalar field since distribution has changed during setup
      const auto& structure_dis = discretization_map.at("structure");

      if (structure_dis->has_state(1, "scalarfield"))
      {
        // get the state and export it to the rowmap to be able to reset the state
        auto statevec = structure_dis->get_state(1, "scalarfield");
        auto statevecrowmap =
            std::make_shared<Core::LinAlg::Vector<double>>(*structure_dis->dof_row_map(1), true);
        Core::LinAlg::export_to(*statevec, *statevecrowmap);

        // now set the state again
        structure_dis->set_state(1, "scalarfield", *statevecrowmap);
      }

      break;
    }
    case CONTACT::Problemtype::tsi:
    {
      std::vector<std::shared_ptr<Core::FE::Discretization>> tar_dis;
      FOUR_C_ASSERT(discretization_map.find("structure") != discretization_map.end(),
          "Could not find discretization 'structure'");
      FOUR_C_ASSERT(discretization_map.find("thermo") != discretization_map.end(),
          "Could not find discretization 'thermo'");
      tar_dis.emplace_back(discretization_map.at("structure"));
      tar_dis.emplace_back(discretization_map.at("thermo"));
      std::vector<std::pair<int, int>> material_map;
      material_map.emplace_back(std::pair<int, int>(0, 1));
      material_map.emplace_back(std::pair<int, int>(1, 0));

      Mortar::Utils::create_volume_ghosting(discret(), tar_dis, material_map);
      break;
    }
    default:
    {
      std::vector<std::shared_ptr<Core::FE::Discretization>> tar_dis;
      FOUR_C_ASSERT(discretization_map.find("structure") != discretization_map.end(),
          "Could not find discretization 'structure'");
      tar_dis.emplace_back(discretization_map.at("structure"));
      Mortar::Utils::create_volume_ghosting(
          discret(), tar_dis, std::vector<std::pair<int, int>>(0));

      break;
    }
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Mortar::Interface::has_ma_sharing_ref_interface() const
{
  return (interface_data_->get_ma_sharing_ref_interface_ptr() != nullptr);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
const Mortar::Interface* Mortar::Interface::get_ma_sharing_ref_interface_ptr() const
{
  return interface_data_->get_ma_sharing_ref_interface_ptr();
}


/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::Interface::add_ma_sharing_ref_interface(const Interface* ref_interface)
{
  // avoid non-uniqueness and closed loops
  if (ref_interface->has_ma_sharing_ref_interface())
  {
    if (ref_interface->get_ma_sharing_ref_interface_ptr() == this) return;
  }

  /* The following test is valid, since this interface must be a FULL subset of
   * the reference interface, i.e. all master elements of this interface must
   * be also contained in the reference master element map. Therefore, if a new
   * reference interface candidate is supposed to replace the current one, it
   * must have more global entries in its master element map. Otherwise, it is
   * as well a sub-set of the current reference interface.
   *
   * Again: The last assumption holds only if no partial overlaps are allowed.
   *                                                          hiermeier 01/18 */
  if (has_ma_sharing_ref_interface())
  {
    const int size_curr_ref_interface =
        get_ma_sharing_ref_interface_ptr()->master_row_elements()->num_global_elements();
    const int size_new_ref_interface = ref_interface->master_row_elements()->num_global_elements();

    if (size_curr_ref_interface >= size_new_ref_interface) return;
  }

  interface_data_->set_ma_sharing_ref_interface_ptr(ref_interface);
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Mortar::Interface::postprocess_quantities(const Teuchos::ParameterList& outputParams) const
{
  using std::shared_ptr;

  // Check if the given parameter list contains all required data to be written to output
  {
    // Vector with names of all required parameter entries
    std::vector<std::string> requiredEntries;
    requiredEntries.emplace_back("step");
    requiredEntries.emplace_back("time");
    requiredEntries.emplace_back("displacement");
    requiredEntries.emplace_back("interface traction");
    requiredEntries.emplace_back("slave forces");
    requiredEntries.emplace_back("master forces");

    check_output_list(outputParams, requiredEntries);
  }

  // Get the discretization writer and get ready for writing
  std::shared_ptr<Core::IO::DiscretizationWriter> writer = idiscret_->writer();

  // Get output for this time step started
  {
    const int step = outputParams.get<int>("step");
    const double time = outputParams.get<double>("time");

    writer->clear_map_cache();
    writer->write_mesh(step, time);
    writer->new_step(step, time);
  }

  /* Write interface displacement
   *
   * The interface displacement has been handed in via the parameter list outParams.
   * Grab it from there, then use Core::LinAlg::export_to() to extract the interface
   * portion from the global displacement vector. Finally, write the interface
   * portion using this interfaces' discretization writer.
   */
  {
    // Get full displacement vector and extract interface displacement
    std::shared_ptr<const Core::LinAlg::Vector<double>> disp =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("displacement");
    std::shared_ptr<Core::LinAlg::Vector<double>> iDisp =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*disp, *iDisp);

    // Write the interface displacement field
    writer->write_vector("displacement", iDisp, Core::IO::VectorType::dofvector);
  }

  // Write Lagrange multiplier field
  {
    // Get full Lagrange multiplier vector and extract values of this interface
    std::shared_ptr<const Core::LinAlg::Vector<double>> lagMult =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("interface traction");
    std::shared_ptr<Core::LinAlg::Vector<double>> iLagMult =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*lagMult, *iLagMult);

    // Write this interface's Lagrange multiplier field
    writer->write_vector("interfacetraction", iLagMult, Core::IO::VectorType::dofvector);
  }

  // Write nodal forces of slave side
  {
    // Get nodal forces
    std::shared_ptr<const Core::LinAlg::Vector<double>> slaveforces =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("slave forces");
    std::shared_ptr<Core::LinAlg::Vector<double>> forces =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*slaveforces, *forces);

    // Write to output
    writer->write_vector("slaveforces", forces, Core::IO::VectorType::dofvector);
  }

  // Write nodal forces of master side
  {
    // Get nodal forces
    std::shared_ptr<const Core::LinAlg::Vector<double>> masterforces =
        outputParams.get<std::shared_ptr<const Core::LinAlg::Vector<double>>>("master forces");
    std::shared_ptr<Core::LinAlg::Vector<double>> forces =
        std::make_shared<Core::LinAlg::Vector<double>>(*idiscret_->dof_row_map());
    Core::LinAlg::export_to(*masterforces, *forces);

    // Write to output
    writer->write_vector("masterforces", forces, Core::IO::VectorType::dofvector);
  }


  // Nodes: node-based vector with '0' at slave nodes and '1' at master nodes
  {
    Core::LinAlg::Vector<double> masterVec(*mnoderowmap_);
    masterVec.put_scalar(1.0);

    std::shared_ptr<const Core::LinAlg::Map> nodeRowMap =
        Core::LinAlg::merge_map(snoderowmap_, mnoderowmap_, false);
    std::shared_ptr<Core::LinAlg::Vector<double>> masterSlaveVec =
        std::make_shared<Core::LinAlg::Vector<double>>(*nodeRowMap, true);
    Core::LinAlg::export_to(masterVec, *masterSlaveVec);

    writer->write_vector("slavemasternodes", masterSlaveVec, Core::IO::VectorType::nodevector);
  }

  // Elements: element-based vector with '0' at slave elements and '1' at master elements
  {
    Core::LinAlg::Vector<double> masterVec(*melerowmap_);
    masterVec.put_scalar(1.0);

    std::shared_ptr<const Core::LinAlg::Map> eleRowMap =
        Core::LinAlg::merge_map(selerowmap_, melerowmap_, false);
    std::shared_ptr<Core::LinAlg::Vector<double>> masterSlaveVec =
        std::make_shared<Core::LinAlg::Vector<double>>(*eleRowMap, true);
    Core::LinAlg::export_to(masterVec, *masterSlaveVec);

    writer->write_vector(
        "slavemasterelements", masterSlaveVec, Core::IO::VectorType::elementvector);
  }

  // Write element owners
  {
    std::shared_ptr<const Core::LinAlg::Map> eleRowMap =
        Core::LinAlg::merge_map(selerowmap_, melerowmap_, false);
    std::shared_ptr<Core::LinAlg::Vector<double>> owner =
        std::make_shared<Core::LinAlg::Vector<double>>(*eleRowMap);

    for (int i = 0; i < idiscret_->element_row_map()->num_my_elements(); ++i)
      (*owner).get_values()[i] = idiscret_->l_row_element(i)->owner();

    writer->write_vector("Owner", owner, Core::IO::VectorType::elementvector);
  }
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
bool Mortar::Interface::check_output_list(
    const Teuchos::ParameterList& outParams, const std::vector<std::string>& requiredEntries) const
{
  // Check for each required parameter entry if it exists
  for (const auto& requiredEntry : requiredEntries)
  {
    if (not outParams.isParameter(requiredEntry))
    {
      FOUR_C_THROW("Parameter list is missing the required entry '{}'.", (requiredEntry));
      return false;
    }
  }

  // We only make it to here, if all checks passed. So it's safe to return 'true'.
  return true;
}

FOUR_C_NAMESPACE_CLOSE
