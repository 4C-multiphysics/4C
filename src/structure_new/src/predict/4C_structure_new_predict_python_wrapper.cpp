// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_predict_python_wrapper.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_utils.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_global_data.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_linalg_vector_numpy_utils.hpp"
#include "4C_structure_new_impl_generic.hpp"
#include "4C_structure_new_model_evaluator_manager.hpp"
#include "4C_structure_new_predict_factory.hpp"
#include "4C_structure_new_predict_python_wrapper_utils.hpp"
#include "4C_structure_new_timint_base.hpp"

#ifdef FOUR_C_WITH_PYBIND11

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------------*/
/*  Implementation of the Python interface                                    */
/*----------------------------------------------------------------------------*/
/*!
 * \brief Wrapper class for encapsulating the interaction with python.
 *
 * The interaction with python is designed in a way that communication between
 * C++ and python happens exclusively through the process with rank 0.
 * If all processes were to talk to python, by default each would have their own
 * python interpreter and own python context. For reasons of simplicity and
 * maintainability, the parallel communication has thus been restricted to the
 * C++ side. However, this results in parallel communication that needs to be
 * performed in every load / time step which might pose some limitations on the
 * practical applicability of this class.
 */
class FOUR_C_HIDDEN Solid::Predict::PythonWrapper::Implementation
{
 public:
  /*!
   * \brief Constructor
   *
   * Imports the provided python file and checks that it provides the required interface, i.e.,
   * mandatory compute() function that will be called in the PythonWrapper::compute() step, and
   * optionally a setup() function that will be called once per simulation that -- if provided --
   * will be called in the PythonWrapper::setup() step.
   *
   * @param python_filename[in] The filename of the python script which provides the predictor
   * implementation.
   * @param comm[in] The MPI communicator for parallel communication.
   */
  explicit Implementation(const std::filesystem::path& python_filename, MPI_Comm comm);

  /*!
   * \brief Wrapper function for performing communication with python that happens once per
   * simulation.
   *
   * This function wraps the call to the setup() function defined in the accompanying python script.
   * It uses gather_mesh_to_rank_zero() for extracting a lightweight mesh representation and
   * build_python_context() to create the context dictionary with information that should be passed
   * to the python interpreter exactly once.
   *
   * \param[in] global_state_ptr Pointer to the global data of the time integration routine.
   * \param[in] sdyn Reference to the structural dynamic data container.
   */
  void setup(const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
      const Solid::TimeInt::BaseDataSDyn& sdyn);

  //! Returns a flag indicating whether the accompanying python script requires a setup step.
  bool requires_setup() const;

  /*!
   * \brief Wrapper function for performing communication with python that happens every load / time
   * step.
   *
   * This function wraps the call to the compute() function in the accompanying python script.
   * Prior to invoking the python interpreter, it uses gather_state_to_rank_zero() and
   * build_python_state() to assemble a dictionary with data that might be needed in python to
   * perform the actual predictor step. Once the prediction step is completed on the python side, it
   * uses scatter_state_from_rank_zero() to convert the apply the resulting updated global state
   * vectors to the 4C internal data structures.
   *
   * \param[out] global_state_ptr Pointer to the global data of the time integration routine.
   */
  void compute(std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr);

 private:
  // Helper structs for data encapsulation

  //! Convenience struct for encapsulating the gathered mesh information in one place.
  struct GatheredMesh
  {
    std::map<int, std::vector<double>> node_coords;
    std::map<int, std::vector<int>> node_dofs;
    std::map<int, std::vector<int>> element_conn;
  };
  //! Convenience struct for encapsulating the gathered global state vectors in one place.
  struct GatheredState
  {
    std::unique_ptr<Core::LinAlg::Vector<double>> dis_n;
    std::unique_ptr<Core::LinAlg::Vector<double>> dis_np;
    std::unique_ptr<Core::LinAlg::Vector<double>> vel_n;
    std::unique_ptr<Core::LinAlg::Vector<double>> vel_np;
    std::unique_ptr<Core::LinAlg::Vector<double>> acc_n;
    std::unique_ptr<Core::LinAlg::Vector<double>> acc_np;
  };

  // Helper functions

  //! Returns a flag indicating whether the executing processor has rank zero.
  bool is_rank_zero() const;

  /*!
   * \brief Builds a dictionary of information that will be passed to python interpreter exactly
   * once when `PythonWrapper::setup()` is called.
   *
   * Under the hood, this function calls convert_gathered_mesh_to_python_mesh() to convert the
   * lightweight representation of the computational mesh to a python-conforming dictionary.
   *
   * \param[in] global_state_ptr Pointer to the global data of the time integration routine
   * providing additional data that might be needed in python to perform the once-per-simulation
   * setup tasks.
   * \param[in] sdyn Reference to the structural dynamic data container providing additional data
   * that might be needed in python to perform the once-per-simulation setup tasks.
   * \param[in] gathered_mesh The lightweight mesh representation that should be included in the
   * data passed to python.
   * \return A Solid::Predict::PythonWrapperUtils::PythonContext containing all information that
   * will be passed to python once.
   */
  Solid::Predict::PythonWrapperUtils::PythonContext build_python_context(
      const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
      const Solid::TimeInt::BaseDataSDyn& sdyn, const GatheredMesh& gathered_mesh) const;

  /*!
   * \brief This function converts a lightweight representation of the computational mesh located on
   * process with rank 0 to a python-conforming dictionary.
   *
   * \param[in] gathered_mesh The mesh representation data that should be passed to python.
   * \param[in] problem_dim Spatial dimensionality of the problem.
   * \return A Solid::Predict::PythonWrapperUtils::PythonMesh containing the data stored in
   * gathered_mesh, as well as additional arrays with the node and element indices. once.
   */
  Solid::Predict::PythonWrapperUtils::PythonMesh convert_gathered_mesh_to_python_mesh(
      const GatheredMesh& gathered_mesh, unsigned int problem_dim) const;

  /*!
   * \brief This function builds a python-compatible dictionary with information that will be passed
   * to the python interpreter everytime that `PythonWrapper::compute()` is called.
   *
   * For now, the dictionary mostly contains the global displacement, velocity, and acceleration
   * vectors, as well as some iteration depending information such as the current time/load step.
   *
   * \param[in] global_state_ptr Pointer to the global data of the time integration routine.
   * \param[in] gathered_state The global state vectors gathered on process with rank 0.
   * \return A Solid::Predict::PythonWrapperUtils::PythonState containing the states that will be
   * passed to python in each predictor. step.
   */
  Solid::Predict::PythonWrapperUtils::PythonState build_python_state(
      const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
      const GatheredState& gathered_state) const;

  /*!
   * \brief This function receives updated global state vectors, gathered on rank 0, and scatters
   * them back to each processor.
   *
   * \param[in] gathered The updated global state vectors gathered on process with rank 0.
   * \param[out] global_state_ptr Pointer to the global data of the time integration routine
   * containing the distributed state vectors.
   */
  void scatter_state_from_rank_zero(const GatheredState& gathered,
      std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr) const;

  /*!
   * \brief Convenience function for gathering a distributed Core::LinAlg::Vector<double> on rank
   * zero.
   *
   * \param[in] distributed The distributed vector that should be gathered on the processor with
   * rank zero.
   * \return A pointer to the gathered, i.e., global vector.
   */
  std::unique_ptr<Core::LinAlg::Vector<double>> gather_vector_to_rank_zero(
      const Core::LinAlg::Vector<double>& distributed) const;

  /*!
   * \brief Convenience function for distributing a Core::LinAlg::Vector<double> from rank zero to
   * all other processes.
   *
   * \param[in] gathered The global vector located on processor with rank 0.
   * \param[out] distributed The distributed vector which only holds processor n's share of the
   * data.
   */
  void scatter_vector_from_rank_zero(const Core::LinAlg::Vector<double>& gathered,
      Core::LinAlg::Vector<double>& distributed) const;

  /*!
   * \brief Convenience function for gathering all global state vectors (displacements, velocities,
   * accelerations) on processor with rank zero.
   *
   * \param[in] global_state_ptr Pointer to the global data of the time integration routine.
   * \return The global state vectors (displacements, velocities, accelerations), all gathered on
   * process with rank 0.
   */
  GatheredState gather_state_to_rank_zero(
      const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr) const;

  /*! \brief Convenience function for creating a lightweight mesh snapshot and gathering it on
   * processor with rank 0.
   *
   * Currently, the lightweight mesh representation consists of nodal coordinates, connectivity
   * information as well as the ordering of the global DOF vector. Since this information is
   * generally stored in a distributed-memory fashion, this function involves parallel
   * communication, where the mesh information is gathered on the process with rank 0.
   *
   * \param[in] discret_ptr Pointer to the underlying discretization.
   * \param[in] problem_dim Spatial dimensionality of the problem.
   * \return The lightweight mesh snapshot, gathered on process with rank 0.
   */
  GatheredMesh gather_mesh_to_rank_zero(
      const std::shared_ptr<Core::FE::Discretization>& discret_ptr, unsigned int problem_dim) const;

 private:
  //! The C++ wrapper instance for the setup() python method.
  pybind11::object setup_;
  //! The C++ wrapper instance for the compute() python method.
  pybind11::object compute_;
  //! MPI communicator for parallel communication.
  MPI_Comm comm_;
  //! Flag indicating whether the python file provides a setup() method.
  bool has_setup_ = false;
};

Solid::Predict::PythonWrapper::Implementation::Implementation(
    const std::filesystem::path& python_filename, MPI_Comm comm)
    : comm_(comm)
{
  // Only process with rank 0 interacts with python.
  if (is_rank_zero())
  {
    // Initialize python interpreter once per process.
    static pybind11::scoped_interpreter py_guard;

    // Add the directory where the python file is located to the PATH so that python finds it
    pybind11::module sys = pybind11::module::import("sys");
    sys.attr("path").attr("insert")(0, python_filename.parent_path().string());

    // Register the helper-classes with python
    Solid::Predict::PythonWrapperUtils::ensure_python_wrapper_module();

    // import the provided python file as module
    pybind11::module model = pybind11::module::import(python_filename.stem().string().c_str());

    // An implementation of a setup() function is optional in the python script
    has_setup_ = pybind11::hasattr(model, "setup");
    if (has_setup_) setup_ = model.attr("setup");

    // But the python script needs to implement a compute() function
    if (!pybind11::hasattr(model, "compute"))
      FOUR_C_THROW("Python surrogate module must define 'compute(state)'.");

    compute_ = model.attr("compute");
  }

  // Broadcast the flag indicating whether a setup function exists to all processes
  Core::Communication::broadcast(has_setup_, 0, comm_);
}

void Solid::Predict::PythonWrapper::Implementation::setup(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    const Solid::TimeInt::BaseDataSDyn& sdyn)
{
  // all processes contribute to assembling the mesh snapshot that will be passed to python
  GatheredMesh gathered_mesh =
      gather_mesh_to_rank_zero(global_state_ptr->get_discret(), global_state_ptr->get_dim());

  // If a setup() method exists and you are processor 0, communicate with python
  if (has_setup_ && is_rank_zero())
  {
    // make it safe to touch python objects for the rest of this function
    pybind11::gil_scoped_acquire gil;
    // build the context dictionary that will be passed to python
    const Solid::Predict::PythonWrapperUtils::PythonContext context =
        build_python_context(global_state_ptr, sdyn, gathered_mesh);
    // perform the once-per-simulation setup tasks in python
    setup_(pybind11::cast(context));
  }
}

bool Solid::Predict::PythonWrapper::Implementation::requires_setup() const { return has_setup_; }

void Solid::Predict::PythonWrapper::Implementation::compute(
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr)
{
  // Gather all global state information on process with rank 0
  GatheredState gathered = gather_state_to_rank_zero(global_state_ptr);

  // Only process with rank 0 communicates with python
  if (is_rank_zero())
  {
    // make it safe to touch python objects for the rest of this function
    pybind11::gil_scoped_acquire gil;
    // convert the gathered data to a python-compatible dictionary that will be passed to the python
    // interpreter
    const Solid::Predict::PythonWrapperUtils::PythonState state =
        build_python_state(global_state_ptr, gathered);
    // create local references to the mutable entries of the state dictionary to be able to ensure
    // after the python call that the arrays in the dict entries have actually been updated instead
    // of been replaced by new ones
    pybind11::object dis_np_before = state.dis_np;
    pybind11::object vel_np_before = state.vel_np;
    pybind11::object acc_np_before = state.acc_np;
    // invoke the python predictor step implemented in the python script which is supposed to update
    // the data contained in the state dictionary
    pybind11::object result = compute_(pybind11::cast(state));
    // the we do not expect a return value so check whether the user complied with this contract:
    if (!result.is_none())
    {
      FOUR_C_THROW(
          "PythonWrapper::compute(state) is supposed to update 'dis_np', 'vel_np', and 'acc_np' in "
          "place and must not return a value.");
    }
    // perform a sanity check to validate that the arrays in the state dictionary have been updated
    // and not replaced by local copies
    if (state.dis_np.ptr() != dis_np_before.ptr() || state.vel_np.ptr() != vel_np_before.ptr() ||
        state.acc_np.ptr() != acc_np_before.ptr())
    {
      FOUR_C_THROW(
          "The PythonWrapper predictor requires the python script to modify state['dis_np'], "
          "state['vel_np'], and state['acc_np'] in-place.");
    }
  }

  // the following statement contains an implicit barrier since all processes need to wait until
  // process with rank 0 has completed the python call and the conversion of the result data before
  // they can receive the updated global state vectors in the scatter step below

  // Once python is done, the updated global state vectors will be communicated back to each
  // processor.
  scatter_state_from_rank_zero(gathered, global_state_ptr);
}

bool Solid::Predict::PythonWrapper::Implementation::is_rank_zero() const
{
  // check whether the current process is rank 0
  return Core::Communication::my_mpi_rank(comm_) == 0;
}

Solid::Predict::PythonWrapperUtils::PythonContext
Solid::Predict::PythonWrapper::Implementation::build_python_context(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    const Solid::TimeInt::BaseDataSDyn& sdyn, const GatheredMesh& gathered_mesh) const
{
  Solid::Predict::PythonWrapperUtils::PythonContext context;
  auto output_control = Global::Problem::instance()->output_control_file();
  // sanity check to make sure that the input file exists
  if (!output_control)
    FOUR_C_THROW(
        "PythonWrapper requires an output control file to access the input file name, but none "
        "is available.");
  context.input_file = output_control->input_file_name();
  context.mesh = convert_gathered_mesh_to_python_mesh(gathered_mesh, global_state_ptr->get_dim());
  context.problem_dim = global_state_ptr->get_dim();
  context.step_max = sdyn.get_step_max();
  context.time_max = sdyn.get_time_max();
  return context;
}

Solid::Predict::PythonWrapperUtils::PythonMesh
Solid::Predict::PythonWrapper::Implementation::convert_gathered_mesh_to_python_mesh(
    const GatheredMesh& gathered_mesh, unsigned int problem_dim) const
{
  // Allocate the containers that will be moved to python
  const pybind11::ssize_t num_nodes = gathered_mesh.node_coords.size();
  const pybind11::ssize_t num_elements = gathered_mesh.element_conn.size();
  pybind11::array_t<int> node_ids(num_nodes);
  pybind11::array_t<double> coordinates(std::vector<pybind11::ssize_t>{num_nodes, 3});
  pybind11::array_t<int> disp_dof_ids(
      std::vector<pybind11::ssize_t>{num_nodes, static_cast<pybind11::ssize_t>(problem_dim)});
  pybind11::array_t<int> element_ids(num_elements);
  pybind11::list connectivity;
  // create views on elements of the above containers
  auto node_ids_view = node_ids.mutable_unchecked<1>();
  auto coordinates_view = coordinates.mutable_unchecked<2>();
  auto disp_dof_ids_view = disp_dof_ids.mutable_unchecked<2>();
  auto element_ids_view = element_ids.mutable_unchecked<1>();

  // copy data from C++ into python data structures

  // iterate over all nodes
  pybind11::ssize_t inode = 0;
  for (const auto& [node_id, coords] : gathered_mesh.node_coords)
  {
    // copy the node IDs
    node_ids_view(inode) = node_id;

    // copy the actual coordinate values
    if (coords.size() != 3)
      FOUR_C_THROW("Node {} has {} coordinate entries, expected 3.", node_id, coords.size());
    for (size_t dim = 0; dim < 3; ++dim) coordinates_view(inode, dim) = coords[dim];

    // copy the actual (displacement) DOFs as well as their association with the mesh nodes
    const auto dof_it = gathered_mesh.node_dofs.find(node_id);
    if (dof_it == gathered_mesh.node_dofs.end())
      FOUR_C_THROW("Missing displacement DOFs for node {} on rank 0.", node_id);
    if (dof_it->second.size() != problem_dim)
      FOUR_C_THROW("Node {} has {} displacement DOFs, expected {}.", node_id, dof_it->second.size(),
          problem_dim);
    for (size_t dim = 0; dim < problem_dim; ++dim)
      disp_dof_ids_view(inode, dim) = dof_it->second[dim];

    ++inode;
  }

  // iterate over all elements
  pybind11::ssize_t ielem = 0;
  for (const auto& [element_id, conn] : gathered_mesh.element_conn)
  {
    // copy the element IDs
    element_ids_view(ielem) = element_id;

    // copy the connectivity
    pybind11::list element_connectivity;
    for (const int node_id : conn) element_connectivity.append(node_id);
    connectivity.append(element_connectivity);

    ++ielem;
  }

  // combine the converted data into a dictionary that will be sent to python
  Solid::Predict::PythonWrapperUtils::PythonMesh mesh;
  mesh.node_ids = node_ids;
  mesh.coordinates = coordinates;
  mesh.disp_dof_ids = disp_dof_ids;
  mesh.element_ids = element_ids;
  mesh.connectivity = connectivity;

  return mesh;
}

Solid::Predict::PythonWrapperUtils::PythonState
Solid::Predict::PythonWrapper::Implementation::build_python_state(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr,
    const GatheredState& gathered) const
{
  // From here on only process with rank 0 acts

  // copy C++ data structures into python data structures
  Solid::Predict::PythonWrapperUtils::PythonState state;
  state.step = global_state_ptr->get_step_np();
  state.time = global_state_ptr->get_time_np();
  state.dt = global_state_ptr->get_delta_time()[0];
  state.num_global_dofs = gathered.dis_n->get_map().num_global_elements();
  state.dis_n = Core::LinAlg::make_const_numpy_view(*gathered.dis_n);
  state.dis_np = Core::LinAlg::make_numpy_view(*gathered.dis_np);
  state.vel_n = Core::LinAlg::make_const_numpy_view(*gathered.vel_n);
  state.vel_np = Core::LinAlg::make_numpy_view(*gathered.vel_np);
  state.acc_n = Core::LinAlg::make_const_numpy_view(*gathered.acc_n);
  state.acc_np = Core::LinAlg::make_numpy_view(*gathered.acc_np);

  return state;
}

std::unique_ptr<Core::LinAlg::Vector<double>>
Solid::Predict::PythonWrapper::Implementation::gather_vector_to_rank_zero(
    const Core::LinAlg::Vector<double>& distributed) const
{
  // construct a map that maps all data to process 0
  auto rank_zero_map = Core::LinAlg::allreduce_e_map(distributed.get_map(), 0);
  auto gathered = std::make_unique<Core::LinAlg::Vector<double>>(*rank_zero_map, true);

  // communicate data from source vector "distributed" to target vector "gathered"
  Core::LinAlg::export_to(distributed, *gathered);

  return gathered;
}

void Solid::Predict::PythonWrapper::Implementation::scatter_vector_from_rank_zero(
    const Core::LinAlg::Vector<double>& gathered, Core::LinAlg::Vector<double>& distributed) const
{
  Core::LinAlg::export_to(gathered, distributed);
}

void Solid::Predict::PythonWrapper::Implementation::scatter_state_from_rank_zero(
    const GatheredState& gathered,
    std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr) const
{
  // all processes participate in the parallel communication (send local data back to each process)
  scatter_vector_from_rank_zero(*gathered.dis_np, *global_state_ptr->get_dis_np());
  scatter_vector_from_rank_zero(*gathered.vel_np, *global_state_ptr->get_vel_np());
  scatter_vector_from_rank_zero(*gathered.acc_np, *global_state_ptr->get_acc_np());
}

Solid::Predict::PythonWrapper::Implementation::GatheredState
Solid::Predict::PythonWrapper::Implementation::gather_state_to_rank_zero(
    const std::shared_ptr<Solid::TimeInt::BaseDataGlobalState>& global_state_ptr) const
{
  // collect the state vectors of all processes on process with rank 0
  GatheredState gathered_state;
  gathered_state.dis_n = gather_vector_to_rank_zero(*global_state_ptr->get_dis_n());
  gathered_state.dis_np = gather_vector_to_rank_zero(*global_state_ptr->get_dis_np());
  gathered_state.vel_n = gather_vector_to_rank_zero(*global_state_ptr->get_vel_n());
  gathered_state.vel_np = gather_vector_to_rank_zero(*global_state_ptr->get_vel_np());
  gathered_state.acc_n = gather_vector_to_rank_zero(*global_state_ptr->get_acc_n());
  gathered_state.acc_np = gather_vector_to_rank_zero(*global_state_ptr->get_acc_np());

  return gathered_state;
}

Solid::Predict::PythonWrapper::Implementation::GatheredMesh
Solid::Predict::PythonWrapper::Implementation::gather_mesh_to_rank_zero(
    const std::shared_ptr<Core::FE::Discretization>& discret_ptr, unsigned int problem_dim) const
{
  // STL containers for a lightweight per-processor mesh representation
  std::map<int, std::vector<double>> local_node_coords;
  std::map<int, std::vector<int>> local_node_dofs;
  std::map<int, std::vector<int>> local_element_conn;

  // iterate over all nodes on the current processor and fill node-based data
  for (auto node : discret_ptr->my_row_node_range())
  {
    // get the global node ID
    const int node_id = node.global_id();

    // copy the node's coordinates into a simple STL container
    std::vector<double> coords(3, 0.0);
    const auto x = node.x();
    for (size_t dim = 0; dim < problem_dim; ++dim) coords[dim] = x[dim];
    local_node_coords[node_id] = std::move(coords);

    // copy the node's DOF ids into a simple STL container
    // assume structure has problem_id = 0 --> displacement DOFs are contained in first DOF set
    const std::vector<int> dofs = discret_ptr->dof(0, node);
    if (dofs.size() < problem_dim)
      FOUR_C_THROW("Node {} has only {} DOFs, expected at least {} displacement DOFs.", node_id,
          dofs.size(), problem_dim);
    // copy only problem_dim entries of the dofs
    local_node_dofs[node_id] = std::vector<int>(dofs.begin(), dofs.begin() + problem_dim);
  }

  // iterate over all elements and fill element-based data
  for (auto element : discret_ptr->my_row_element_range())
  {
    std::vector<int> conn;

    // copy the global node ids that comprise each element into a simple STL container
    for (auto node : element.nodes()) conn.push_back(node.global_id());
    local_element_conn[element.global_id()] = std::move(conn);
  }

  // communicate all information to process 0 which will communicate with python
  MPI_Comm comm = discret_ptr->get_comm();
  const int target_proc = 0;
  std::map<int, std::vector<double>> gathered_node_coords;
  std::map<int, std::vector<int>> gathered_node_dofs;
  std::map<int, std::vector<int>> gathered_element_conn;
  Core::LinAlg::gather(local_node_coords, gathered_node_coords, 1, &target_proc, comm);
  Core::LinAlg::gather(local_node_dofs, gathered_node_dofs, 1, &target_proc, comm);
  Core::LinAlg::gather(local_element_conn, gathered_element_conn, 1, &target_proc, comm);

  // combine the converted data into the struct that will be sent to python
  GatheredMesh mesh;
  mesh.node_coords = gathered_node_coords;
  mesh.node_dofs = gathered_node_dofs;
  mesh.element_conn = gathered_element_conn;

  return mesh;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Predict::PythonWrapper::PythonWrapper() : python_filename_(), python_implementation_(nullptr)
{
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
Solid::Predict::PythonWrapper::~PythonWrapper() = default;

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::PythonWrapper::setup()
{
  check_init();

  const auto sdyn_params = Global::Problem::instance()->structural_dynamic_params();
  // perform a sanity check and warn the user if they try to use this class
  // outside of a static context
  const auto type = sdyn_params.get<Solid::DynamicType>("DYNAMICTYPE");
  if (type != Solid::DynamicType::Statics)
    FOUR_C_THROW("PythonWrapper is currently only supported for DYNAMICTYPE = Statics.");

  // initialize the python predictor
  // retrieve the python file containing the implementation from the input file parameter
  python_filename_ = sdyn_params.get<std::filesystem::path>("PYTHON_PREDICTOR_FILE");

  if (python_filename_.empty())
    FOUR_C_THROW("PREDICT = PythonWrapper requires PYTHON_PREDICTOR_FILE to be set.");

  // initialize the pointer to the python implementation
  python_implementation_ = std::make_unique<Implementation>(
      python_filename_, global_state_ptr()->get_discret()->get_comm());

  // initialize the python implementation if it needs initialization
  if (python_implementation_->requires_setup())
    python_implementation_->setup(global_state_ptr(), impl_int_ptr()->get_data_sdyn());

  issetup_ = true;
}

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
void Solid::Predict::PythonWrapper::compute(::NOX::Abstract::Group& grp)
{
  check_init_setup();

  // Compute new disnp_ptr, velnp_ptr, accnp_ptr via call to python script
  python_implementation_->compute(global_state_ptr());

  impl_int().model_eval().predict();
}


FOUR_C_NAMESPACE_CLOSE

#endif