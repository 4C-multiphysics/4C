// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_beaminteraction_calc_utils.hpp"

#include "4C_beam3_base.hpp"
#include "4C_beam3_reissner.hpp"
#include "4C_beaminteraction_crosslinking_params.hpp"
#include "4C_beaminteraction_link.hpp"
#include "4C_beaminteraction_spherebeamlinking_params.hpp"
#include "4C_binstrategy.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_geometry_intersection_math.hpp"
#include "4C_fem_geometry_periodic_boundingbox.hpp"
#include "4C_global_data.hpp"
#include "4C_io_input_parameter_container.hpp"
#include "4C_io_pstream.hpp"
#include "4C_linalg_fevector.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_rigidsphere.hpp"
#include "4C_shell_kl_nurbs.hpp"
#include "4C_solid_3D_ele.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  namespace Utils
  {
    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    bool is_beam_element(Core::Elements::Element const& element)
    {
      return (dynamic_cast<const Discret::Elements::Beam3Base*>(&element) != nullptr) ? true
                                                                                      : false;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    bool is_rigid_sphere_element(Core::Elements::Element const& element)
    {
      return (dynamic_cast<const Discret::Elements::Rigidsphere*>(&element) != nullptr) ? true
                                                                                        : false;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    bool is_beam_node(Core::Nodes::Node const& node)
    {
      bool beameles = false;
      bool othereles = false;

      for (auto ele : node.adjacent_elements())
      {
        if (is_beam_element(*ele.user_element()))
          beameles = true;
        else
          othereles = true;
      }

      if (beameles and othereles)
        FOUR_C_THROW(
            "Beam elements and other (solid, rigid sphere) elements sharing the same node is "
            "currently not allowed in 4C!");

      return beameles;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    bool is_beam_centerline_node(Core::Nodes::Node const& node)
    {
      bool beamclnode = false;

      // TODO: actually we would have to check all elements of all processors!!! Gather?
      for (auto ele : node.adjacent_elements())
      {
        const Discret::Elements::Beam3Base* beamele =
            dynamic_cast<const Discret::Elements::Beam3Base*>(ele.user_element());

        if (beamele != nullptr and beamele->is_centerline_node(node)) beamclnode = true;
      }
      return beamclnode;
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    bool is_rigid_sphere_node(Core::Nodes::Node const& node)
    {
      bool sphereele = false;
      bool othereles = false;

      // TODO: actually we would have to check all elements of all processors!!! Gather?
      for (auto ele : node.adjacent_elements())
      {
        if (is_rigid_sphere_element(*ele.user_element()))
          sphereele = true;
        else
          othereles = true;
      }

      if (sphereele and othereles)
        FOUR_C_THROW(
            "Rigid sphere elements and other (solid, beam) elements sharing "
            "the same node is currently not allowed in 4C!");

      return sphereele;
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void periodic_boundary_consistent_dis_vector(Core::LinAlg::Vector<double>& dis,
        const Core::Geo::MeshFree::BoundingBox& pbb, const Core::FE::Discretization& discret)
    {
      Core::LinAlg::Matrix<3, 1> d;
      Core::LinAlg::Matrix<3, 1> X;
      int doflid[3];

      for (int i = 0; i < discret.num_my_row_nodes(); ++i)
      {
        d.clear();
        X.clear();

        // get a pointer at i-th row node
        Core::Nodes::Node* node = discret.l_row_node(i);

        /* Hermite Interpolation: Check whether node is a beam node which is NOT
         * used for centerline interpolation if so, we simply skip it because
         * it does not have position DoFs */
        if (is_beam_node(*node) and not is_beam_centerline_node(*node)) continue;

        // get GIDs of this node's degrees of freedom
        std::vector<int> dofnode = discret.dof(node);

        for (int dim = 0; dim < 3; ++dim)
        {
          doflid[dim] = dis.get_map().lid(dofnode[dim]);
          d(dim) = (dis).local_values_as_span()[doflid[dim]];
          X(dim) = node->x()[dim];
        }
        // shift
        pbb.shift_3d(d, X);

        for (int dim = 0; dim < 3; ++dim)
        {
          (dis).get_values()[doflid[dim]] = d(dim);
        }
      }
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    std::vector<int> permutation(int number)
    {
      // auxiliary variable
      int j = 0;

      Global::Problem::instance()->random()->set_rand_range(0.0, 1.0);

      // result vector initialized with ordered numbers from 0 to N-1
      std::vector<int> randorder(number, 0);
      for (int i = 0; i < (int)randorder.size(); i++) randorder[i] = i;

      for (int i = 0; i < number; ++i)
      {
        // generate random number between 0 and i
        j = (int)floor((i + 1.0) * Global::Problem::instance()->random()->uni());

        /*exchange values at positions i and j (note: value at position i is i due to above
         *initialization and because so far only positions <=i have been changed*/
        randorder[i] = randorder[j];
        randorder[j] = i;
      }

      return randorder;
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void get_current_element_dis(Core::FE::Discretization const& discret,
        Core::Elements::Element const* ele, const Core::LinAlg::Vector<double>& ia_discolnp,
        std::vector<double>& eledisp)
    {
      // clear
      eledisp.clear();

      std::vector<int> lm, lmowner, lmstride;

      ele->location_vector(discret, lm, lmowner, lmstride);
      eledisp = Core::FE::extract_values(ia_discolnp, lm);
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void get_current_unshifted_element_dis(Core::FE::Discretization const& discret,
        Core::Elements::Element const* ele, const Core::LinAlg::Vector<double>& ia_discolnp,
        Core::Geo::MeshFree::BoundingBox const& pbb, std::vector<double>& eledisp)
    {
      get_current_element_dis(discret, ele, ia_discolnp, eledisp);

      // cast to beambase element
      Discret::Elements::Beam3Base const* beamele =
          dynamic_cast<Discret::Elements::Beam3Base const*>(ele);

      // so far, only beam elements can be cut by a periodic boundary
      if (beamele == nullptr) return;

      beamele->un_shift_node_position(eledisp, pbb);
    }

    /*----------------------------------------------------------------------*
     *----------------------------------------------------------------------*/
    template <typename T>
    void set_filament_binding_spot_positions(
        std::shared_ptr<Core::FE::Discretization> discret, T& params)
    {
      // todo: set somewhere else
      double const tol = Core::Geo::TOL7;

      // temporarily extend ghosting, should be reverted somewhere later on as it is needed
      // in this fashion only here
      std::set<int> relevantfilaments;
      extend_ghosting_for_filament_bspot_setup(relevantfilaments, *discret);

      // get pointers to all filament number conditions set
      std::vector<const Core::Conditions::Condition*> filamentconditions;
      discret->get_condition("BeamLineFilamentCondition", filamentconditions);

      // compute number of linker types
      std::vector<Inpar::BeamInteraction::CrosslinkerType> linkertypes = params.linker_types();

      // loop over all relevant (on myrank) filaments
      for (auto const& filiter : relevantfilaments)
      {
        // loop over all nodes of current filament, sort elements and calculate total filament
        // length
        std::vector<int> const* nodeids = filamentconditions[filiter]->get_nodes();
        std::vector<Core::Elements::Element*> sortedfilamenteles;
        double filreflength = 0.0;
        compute_filament_length_and_sort_its_elements(
            sortedfilamenteles, nodeids, filreflength, *discret);

        // loop over all linking types
        for (unsigned int linkertype_i = 0; linkertype_i < linkertypes.size(); ++linkertype_i)
        {
          // set start and end arc length parameter for filament binding spot
          double start = 0.0;
          double end = filreflength;
          double filamentbspotinterval =
              params.filament_bspot_interval_global(linkertypes[linkertype_i]);
          std::pair<double, double> filamentbspotrangelocal =
              params.filament_bspot_range_local(linkertypes[linkertype_i]);
          std::pair<double, double> filamentbspotrangeglobal =
              params.filament_bspot_range_global(linkertypes[linkertype_i]);

          // in case certain range of filament was specified
          if (filamentbspotrangelocal.first > 0.0 and filamentbspotrangelocal.first < 1.0)
            start = filamentbspotrangelocal.first * filreflength;
          if (filamentbspotrangelocal.second > 0.0 and filamentbspotrangelocal.second < 1.0)
            end *= filamentbspotrangelocal.second;

          // in case binding spot interval is constant on every filament
          if (filamentbspotrangeglobal.first > 0.0)
          {
            // get arc parameter range for binding spot positions for current filament
            start = (filreflength < filamentbspotrangeglobal.first)
                        ? (filreflength + 1.0)
                        : (filamentbspotrangeglobal.first);
            end = (filreflength < filamentbspotrangeglobal.second or
                      filamentbspotrangeglobal.second < 0.0)
                      ? filreflength
                      : filamentbspotrangeglobal.second;
          }

          // set different filament binding spot interval for each filament dependent on its
          // reference length
          if (params.filament_bspot_interval_local(linkertypes[linkertype_i]) > 0.0)
            filamentbspotinterval =
                (end - start) * params.filament_bspot_interval_local(linkertypes[linkertype_i]);

          // get number of binding spots for current filament in bonds
          int numbspot = (start < filreflength and ((end - start) > 0.0))
                             ? (std::floor((end + tol - start) / filamentbspotinterval) + 1)
                             : 0;

          // in case current filament has no binding spots, we go to next filament
          if (numbspot == 0) continue;

          // print number of binding spots for current filament
          Core::IO::cout(Core::IO::debug)
              << "\n---------------------------------------------------------------"
              << Core::IO::endl;
          Core::IO::cout(Core::IO::debug)
              << numbspot << " binding spots of type "
              << Inpar::BeamInteraction::crosslinker_type_to_string(linkertypes[linkertype_i])
              << " on filament " << filiter << " (consists of "
              << static_cast<int>(sortedfilamenteles.size()) << " elements)"
              << " with: " << Core::IO::endl;

          // set xis on element level
          set_binding_spots_positions_on_filament(sortedfilamenteles, start,
              linkertypes[linkertype_i], numbspot, filamentbspotinterval, tol);

          Core::IO::cout(Core::IO::debug)
              << "---------------------------------------------------------------\n"
              << Core::IO::endl;
        }
      }
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void extend_ghosting_for_filament_bspot_setup(
        std::set<int>& relevantfilaments, Core::FE::Discretization& discret)
    {
      std::set<int> setofnodegidswithrequiredelecloud;
      determine_off_my_rank_nodes_with_relevant_ele_cloud_for_filament_bspot_setup(
          relevantfilaments, setofnodegidswithrequiredelecloud, discret);

      // collect element gids that need to be ghosted so that each proc ghosts all
      // elements of a filament containing at least one node of myrank
      // do communication to gather all elements to temporarily extend ghosting
      std::set<int> coleleset;
      for (int iproc = 0; iproc < Core::Communication::num_mpi_ranks(discret.get_comm()); ++iproc)
      {
        // myrank == iproc: copy set to vector in order to broadcast data
        std::vector<int> requirednodes;
        if (iproc == Core::Communication::my_mpi_rank(discret.get_comm()))
          requirednodes.insert(requirednodes.begin(), setofnodegidswithrequiredelecloud.begin(),
              setofnodegidswithrequiredelecloud.end());

        // proc i tells all procs how many nodegids it has
        int numnodes = requirednodes.size();
        Core::Communication::broadcast(&numnodes, 1, iproc, discret.get_comm());

        // proc i actually sends nodegids
        requirednodes.resize(numnodes);
        Core::Communication::broadcast(requirednodes.data(), numnodes, iproc, discret.get_comm());

        std::set<int> sdata;
        std::set<int> rdata;

        // each proc looks in node row map and adds element clouds of owned nodes to sdata
        for (int i = 0; i < numnodes; ++i)
        {
          // only if myrank is owner of requested node
          if (discret.node_row_map()->lid(requirednodes[i]) < 0) continue;

          // insert element cloud of current node
          Core::Nodes::Node* currnode = discret.g_node(requirednodes[i]);
          for (int j = 0; j < currnode->num_element(); ++j)
            sdata.insert(currnode->adjacent_elements()[j].global_id());
        }

        // gather and store information on iproc
        Core::LinAlg::gather<int>(sdata, rdata, 1, &iproc, discret.get_comm());
        if (iproc == Core::Communication::my_mpi_rank(discret.get_comm())) coleleset = rdata;
      }

      // insert previous ghosting
      for (int lid = 0; lid < discret.num_my_col_elements(); ++lid)
        coleleset.insert(discret.element_col_map()->gid(lid));
      std::vector<int> colgids(coleleset.begin(), coleleset.end());

      // create new ele col map
      Core::LinAlg::Map newelecolmap(
          -1, static_cast<int>(colgids.size()), colgids.data(), 0, discret.get_comm());

      // temporarily extend ghosting
      Core::Binstrategy::Utils::extend_discretization_ghosting(
          discret, newelecolmap, true, false, true);
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void determine_off_my_rank_nodes_with_relevant_ele_cloud_for_filament_bspot_setup(
        std::set<int>& relevantfilaments, std::set<int>& setofrequirednodes,
        Core::FE::Discretization& discret)
    {
      std::vector<const Core::Conditions::Condition*> filament_conditions;
      discret.get_condition("BeamLineFilamentCondition", filament_conditions);
      auto filament_nodes_and_conditions =
          Core::Conditions::find_conditioned_node_ids_and_conditions(
              discret, filament_conditions, Core::Conditions::LookFor::locally_owned);

      // loop over all row nodes
      for (int rown = 0; rown < discret.num_my_row_nodes(); ++rown)
      {
        Core::Nodes::Node* currnode = discret.l_row_node(rown);
        if (!filament_nodes_and_conditions.contains(currnode->id())) continue;

        const Core::Conditions::Condition* cond =
            filament_nodes_and_conditions.find(currnode->id())->second;

        // get filament number
        int const currfilnum = cond->parameters().get<int>("ID");

        // if a filament has already been examined --> continue with next node
        if (relevantfilaments.find(currfilnum) != relevantfilaments.end()) continue;
        // filament is examined for the first time --> new entry for relevant filaments on myrank
        else
          relevantfilaments.insert(currfilnum);

        // loop over all nodes of current filament and store gids of nodes not owned by myrank
        std::vector<int> const* nodeids = cond->get_nodes();
        for (int i = 0; i < static_cast<int>(nodeids->size()); ++i)
          if (discret.node_row_map()->lid((*nodeids)[i]) < 0)
            setofrequirednodes.insert((*nodeids)[i]);
      }
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void compute_filament_length_and_sort_its_elements(
        std::vector<Core::Elements::Element*>& sortedfilamenteles, std::vector<int> const* nodeids,
        double& filreflength, Core::FE::Discretization& discret)
    {
      // loop over all nodes associated with current filament
      for (int nodei = 0; nodei < static_cast<int>(nodeids->size()); ++nodei)
      {
        // insert element cloud of current node
        Core::Nodes::Node* node = discret.g_node((*nodeids)[nodei]);
        for (auto ele : node->adjacent_elements())
        {
          // only if element has not yet been added to the filaments elements
          if (std::find(sortedfilamenteles.begin(), sortedfilamenteles.end(), ele.user_element()) !=
              sortedfilamenteles.end())
            continue;

          Discret::Elements::Beam3Base* currbeamele =
              dynamic_cast<Discret::Elements::Beam3Base*>(ele.user_element());

#ifdef FOUR_C_ENABLE_ASSERTIONS
          if (currbeamele == nullptr)
            FOUR_C_THROW("DESIGN LINE BEAM FILAMENT CONDITIONS only applicable to beam elements.");
#endif

          // add element reference length of new element to filament reference length
          filreflength += currbeamele->ref_length();
          // add element
          sortedfilamenteles.push_back(ele.user_element());
        }
      }
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void set_binding_spots_positions_on_filament(
        std::vector<Core::Elements::Element*>& sortedfilamenteles, double start,
        Inpar::BeamInteraction::CrosslinkerType linkertype, int numbspot,
        double filamentbspotinterval, double tol)
    {
      // default no occupied binding spots
      std::pair<double, double> elearcinterval = std::make_pair(0.0, 0.0);
      int bspotcounter = 0;
      double currbspotarcparam = start;
      // loop over all elements in sorted order
      for (unsigned int se_iter = 0; se_iter < sortedfilamenteles.size(); ++se_iter)
      {
        Discret::Elements::Beam3Base* beamele =
            dynamic_cast<Discret::Elements::Beam3Base*>(sortedfilamenteles[se_iter]);

        // init variables set in beam eles
        std::vector<double> bspotposxi;

        // get arc parameter range of current element
        elearcinterval =
            std::make_pair(elearcinterval.second, elearcinterval.second + beamele->ref_length());

        while ((currbspotarcparam > (elearcinterval.first - tol) and
                   currbspotarcparam <= (elearcinterval.second + tol)) and
               bspotcounter < numbspot)
        {
          // linear mapping from arc pos to xi (local element system)
          double xi = (2.0 * currbspotarcparam - elearcinterval.first - elearcinterval.second) /
                      beamele->ref_length();
          xi = (abs(round(xi) - xi) < tol) ? round(xi) : xi;
          bspotposxi.push_back(xi);

          // print to screen
          Core::IO::cout(Core::IO::debug) << bspotcounter + 1 << ". binding spot: "
                                          << "xi = " << xi << " on element " << se_iter << " (gid "
                                          << beamele->id() << ")" << Core::IO::endl;

          ++bspotcounter;
          currbspotarcparam += filamentbspotinterval;
        }

        // set element variables for binding spot position and status
        beamele->set_positions_of_binding_spot_type(linkertype, bspotposxi);
      }
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void get_pos_and_triad_of_binding_spot(Core::Elements::Element* ele,
        Core::Geo::MeshFree::BoundingBox& pbb, Inpar::BeamInteraction::CrosslinkerType linkertype,
        int locbspotnum, Core::LinAlg::Matrix<3, 1>& bspotpos,
        Core::LinAlg::Matrix<3, 3>& bspottriad, std::vector<double>& eledisp)
    {
      // cast to beambase element
      Discret::Elements::Beam3Base* beamele = dynamic_cast<Discret::Elements::Beam3Base*>(ele);

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (beamele == nullptr) FOUR_C_THROW("Dynamic cast to beam3base failed");
#endif

      // get current position at binding spot xi
      beamele->get_pos_of_binding_spot(bspotpos, eledisp, linkertype, locbspotnum, pbb);

      // get current triad at binding spot xi
      beamele->get_triad_of_binding_spot(bspottriad, eledisp, linkertype, locbspotnum);
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void get_pos_and_triad_of_binding_spot(Core::FE::Discretization const& discret,
        Core::Elements::Element* ele, Core::LinAlg::Vector<double>& ia_discolnp,
        Core::Geo::MeshFree::BoundingBox& pbb, Inpar::BeamInteraction::CrosslinkerType linkertype,
        int locbspotnum, Core::LinAlg::Matrix<3, 1>& bspotpos,
        Core::LinAlg::Matrix<3, 3>& bspottriad)
    {
      std::vector<double> eledisp;
      get_current_unshifted_element_dis(discret, ele, ia_discolnp, pbb, eledisp);

      get_pos_and_triad_of_binding_spot(
          ele, pbb, linkertype, locbspotnum, bspotpos, bspottriad, eledisp);
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool is_distance_out_of_range(Core::LinAlg::Matrix<3, 1> const& pos1,
        Core::LinAlg::Matrix<3, 1> const& pos2, double const lowerbound, double const upperbound)
    {
      Core::LinAlg::Matrix<3, 1> dist_vec(Core::LinAlg::Initialization::zero);
      dist_vec.update(1.0, pos1, -1.0, pos2);

      const double distance = dist_vec.norm2();

      return (distance < lowerbound or distance > upperbound) ? true : false;
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool is_enclosed_angle_out_of_range(Core::LinAlg::Matrix<3, 1> const& direction1,
        Core::LinAlg::Matrix<3, 1> const& direction2, double const lowerbound,
        double const upperbound)
    {
      // cosine of angle is scalar product of vectors divided by their norms
      // direction vectors should be unit vectors since they come from triads, but anyway ...
      double cos_angle = direction1.dot(direction2) / direction1.norm2() / direction2.norm2();

      // to enable parallel directions that are numerically slightly > 1 ( < -1)
      // ( would lead to NaN in std::acos )
      if (abs(abs(cos_angle) - 1.0) < Core::Geo::TOL12) cos_angle = std::round(cos_angle);

      double angle = std::acos(cos_angle);

      // acos returns angle \in [0,\pi] but we always want the acute angle here
      if (angle > 0.5 * std::numbers::pi) angle = std::numbers::pi - angle;

      return (angle < lowerbound or angle > upperbound) ? true : false;
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    bool do_beam_elements_share_nodes(
        Core::Elements::Element const* const beam, Core::Elements::Element const* const nbbeam)
    {
      // check if two considered eles share nodes
      for (unsigned int i = 0; i < 2; ++i)
      {
        // node 0 and 1 are always first and last node, respectively
        if (beam->node_ids()[i] == nbbeam->node_ids()[0] ||
            beam->node_ids()[i] == nbbeam->node_ids()[1])
          return true;
      }

      return false;
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void fe_assemble_ele_force_stiff_into_system_vector_matrix(
        const Core::FE::Discretization& discret, std::vector<int> const& elegid,
        std::vector<Core::LinAlg::SerialDenseVector> const& elevec,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& elemat,
        std::shared_ptr<Core::LinAlg::FEVector<double>> fe_sysvec,
        std::shared_ptr<Core::LinAlg::SparseMatrix> fe_sysmat)
    {
      // the entries of elevecX  belong to the Dofs of the element with GID elegidX
      // the rows    of elematXY belong to the Dofs of the element with GID elegidX
      // the columns of elematXY belong to the Dofs of the element with GID elegidY
      const Core::Elements::Element* ele1 = discret.g_element(elegid[0]);
      const Core::Elements::Element* ele2 = discret.g_element(elegid[1]);

      // get element location vector and ownerships
      std::vector<int> lmrow1;
      std::vector<int> lmrow2;
      std::vector<int> lmrowowner1;
      std::vector<int> lmrowowner2;
      std::vector<int> lmstride;

      ele1->location_vector(discret, lmrow1, lmrowowner1, lmstride);
      ele2->location_vector(discret, lmrow2, lmrowowner2, lmstride);

      // assemble both element vectors into global system vector
      if (fe_sysvec != nullptr)
      {
        fe_sysvec->sum_into_global_values(elevec[0].length(), lmrow1.data(), elevec[0].values());
        fe_sysvec->sum_into_global_values(elevec[1].length(), lmrow2.data(), elevec[1].values());
      }

      // and finally also assemble stiffness contributions
      if (fe_sysmat != nullptr)
      {
        fe_sysmat->fe_assemble(elemat[0][0], lmrow1, lmrow1);
        fe_sysmat->fe_assemble(elemat[0][1], lmrow1, lmrow2);
        fe_sysmat->fe_assemble(elemat[1][0], lmrow2, lmrow1);
        fe_sysmat->fe_assemble(elemat[1][1], lmrow2, lmrow2);
      }
    }
    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/

    /**
     *
     */
    unsigned int get_number_of_element_centerline_dof(const Core::Elements::Element* ele)
    {
      std::vector<unsigned int> local_centerline_dof_indices;
      auto beam_element = dynamic_cast<const Discret::Elements::Beam3Base*>(ele);
      if (beam_element != nullptr)
      {
        beam_element->centerline_dof_indices_of_element(local_centerline_dof_indices);
        return local_centerline_dof_indices.size();
      }
      else
        FOUR_C_THROW(
            "GetNumberOfElementCenterlineDof: The given element has to be a beam element.");
    }


    /**
     *
     */
    template <unsigned int n_centerline_dof>
    void get_element_centerline_gid_indices(Core::FE::Discretization const& discret,
        const Core::Elements::Element* ele,
        Core::LinAlg::Matrix<n_centerline_dof, 1, int>& centerline_gid)
    {
      // First we need the local IDs of the centerline DOFs.
      std::vector<unsigned int> local_centerline_dof_indices;
      auto beam_element = dynamic_cast<const Discret::Elements::Beam3Base*>(ele);
      if (beam_element != nullptr)
      {
        beam_element->centerline_dof_indices_of_element(local_centerline_dof_indices);
        if (local_centerline_dof_indices.size() != n_centerline_dof)
          FOUR_C_THROW(
              "get_element_centerline_gid_indices: The size of the local_centerline_dof_indices "
              "({}) "
              "does not match the number of beam dofs ({}).",
              local_centerline_dof_indices.size(), n_centerline_dof);
      }
      else
        FOUR_C_THROW(
            "get_element_centerline_gid_indices: The given element has to be a beam element.");

      // Get the GIDs of the whole beam element (including rotational DOFs).
      std::vector<int> lmrow;
      std::vector<int> lmrowowner;
      std::vector<int> lmstride;
      ele->location_vector(discret, lmrow, lmrowowner, lmstride);

      // Get the GIDs of only the centerline DOFs.
      for (unsigned int i_dof = 0; i_dof < n_centerline_dof; i_dof++)
        centerline_gid(i_dof) = lmrow[local_centerline_dof_indices[i_dof]];
    }

    /**
     *
     */
    void get_element_centerline_dof_indices(Core::FE::Discretization const& discret,
        const Core::Elements::Element* ele, std::vector<unsigned int>& ele_centerline_dof_indices,
        unsigned int& num_dof)
    {
      // Todo implement method in Core::Elements::Element or find alternative way of doing this
      // find out the elements' number of Dofs (=dimension of element vector/matrices)
      std::vector<int> lmrow;
      std::vector<int> dummy1, dummy2;

      ele->location_vector(discret, lmrow, dummy1, dummy2);
      num_dof = lmrow.size();

      const Discret::Elements::Beam3Base* beamele =
          dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

      if (beamele != nullptr)
      {
        beamele->centerline_dof_indices_of_element(ele_centerline_dof_indices);
      }
      else
      {
        ele_centerline_dof_indices.resize(num_dof);
        for (unsigned int i = 0; i < num_dof; ++i) ele_centerline_dof_indices[i] = i;
      }
    }

    /**
     *
     */
    std::vector<int> get_element_rot_gid_indices(
        const Core::FE::Discretization& discret, const Core::Elements::Element* element)
    {
      const auto* beam_sr_element = dynamic_cast<const Discret::Elements::Beam3r*>(element);
      if (beam_sr_element == nullptr)
        FOUR_C_THROW(
            "The function get_element_rot_gid_indices is only implemented for Simo-Reissner beam "
            "elements!");

      if (not(beam_sr_element->num_node() == 3 and
              beam_sr_element->hermite_centerline_interpolation()))
        FOUR_C_THROW(
            "The function get_element_rot_gid_indices is only implemented for Simo-Reissner beam "
            "elements with hermite3line2 interpolation!");

      // Get all GID of the element
      std::vector<int> lm_beam, gid_solid, lmowner, lmstride;
      beam_sr_element->location_vector(discret, lm_beam, lmowner, lmstride);

      // Local indices of the rotational DOFs for the Simo-Reissner beam element.
      constexpr auto n_dof_rot = 9;
      std::array<int, n_dof_rot> rot_dof_indices{3, 4, 5, 12, 13, 14, 18, 19, 20};

      // Gather the GID of the rotational DOF
      std::vector<int> element_rot_gid_indices(9, -1);
      for (unsigned int i = 0; i < n_dof_rot; i++)
        element_rot_gid_indices[i] = lm_beam[rot_dof_indices[i]];
      return element_rot_gid_indices;
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void assemble_centerline_dof_force_stiff_into_element_force_stiff(
        Core::FE::Discretization const& discret, std::vector<int> const& elegid,
        std::vector<Core::LinAlg::SerialDenseVector> const& eleforce_centerlineDOFs,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& elestiff_centerlineDOFs,
        std::vector<Core::LinAlg::SerialDenseVector>* eleforce,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>* elestiff)
    {
      std::vector<unsigned int> numdof_ele(2);
      std::vector<std::vector<unsigned int>> ele_centerlinedofindices(2);

      for (unsigned int iele = 0; iele < 2; ++iele)
      {
        Core::Elements::Element* ele = discret.g_element(elegid[iele]);
        get_element_centerline_dof_indices(
            discret, ele, ele_centerlinedofindices[iele], numdof_ele[iele]);
      }


      // assemble centerline DOF values correctly into element DOFvec vectors/matrices
      if (eleforce != nullptr)
      {
        for (unsigned int iele = 0; iele < 2; ++iele)
        {
          // resize and clear variable
          ((*eleforce)[iele]).size(numdof_ele[iele]);

          // safety check: dimensions
          if ((unsigned int)eleforce_centerlineDOFs[iele].numRows() !=
              ele_centerlinedofindices[iele].size())
            FOUR_C_THROW(
                "size mismatch! need to assemble {} values of centerline-Dof based "
                "force vector into element vector but only got {} element-local Dof indices",
                eleforce_centerlineDOFs[iele].numRows(), ele_centerlinedofindices[iele].size());

          // Todo maybe use a more general 'SerialDenseAssemble' method here
          for (unsigned int idof = 0; idof < ele_centerlinedofindices[iele].size(); ++idof)
            ((*eleforce)[iele])(ele_centerlinedofindices[iele][idof]) =
                eleforce_centerlineDOFs[iele](idof);
        }
      }

      if (elestiff != nullptr)
      {
        for (unsigned int iele = 0; iele < 2; ++iele)
        {
          for (unsigned int jele = 0; jele < 2; ++jele)
          {
            // resize and clear variable
            ((*elestiff)[iele][jele]).shape(numdof_ele[iele], numdof_ele[jele]);

            // safety check: dimensions
            if ((unsigned int)elestiff_centerlineDOFs[iele][jele].numRows() !=
                ele_centerlinedofindices[iele].size())
              FOUR_C_THROW(
                  "size mismatch! need to assemble {} row values of centerline-Dof based "
                  "stiffness matrix into element matrix but only got {} element-local Dof indices",
                  elestiff_centerlineDOFs[iele][jele].numRows(),
                  ele_centerlinedofindices[iele].size());

            if ((unsigned int)elestiff_centerlineDOFs[iele][jele].numCols() !=
                ele_centerlinedofindices[jele].size())
              FOUR_C_THROW(
                  "size mismatch! need to assemble {} column values of centerline-Dof based "
                  "stiffness matrix into element matrix but only got {} element-local Dof indices",
                  elestiff_centerlineDOFs[iele][jele].numCols(),
                  ele_centerlinedofindices[jele].size());

            for (unsigned int idof = 0; idof < ele_centerlinedofindices[iele].size(); ++idof)
              for (unsigned int jdof = 0; jdof < ele_centerlinedofindices[jele].size(); ++jdof)
                ((*elestiff)[iele][jele])(
                    ele_centerlinedofindices[iele][idof], ele_centerlinedofindices[jele][jdof]) =
                    elestiff_centerlineDOFs[iele][jele](idof, jdof);
          }
        }
      }
    }

    /**
     *
     */
    void assemble_centerline_dof_col_matrix_into_element_col_matrix(
        Core::FE::Discretization const& discret, const Core::Elements::Element* element,
        Core::LinAlg::SerialDenseMatrix const& row_matrix_centerlineDOFs,
        Core::LinAlg::SerialDenseMatrix& row_matrix_elementDOFs)
    {
      // Get the centerline DOFs of the element.
      unsigned int numdof_ele;
      std::vector<unsigned int> ele_centerlinedofindices;
      get_element_centerline_dof_indices(discret, element, ele_centerlinedofindices, numdof_ele);

      // Fill in the centerline matrix into the full element matrix.
      // Resize and clear output matrix variable.
      row_matrix_elementDOFs.shape(row_matrix_centerlineDOFs.numRows(), numdof_ele);

      // Safety check: dimensions.
      if ((unsigned int)row_matrix_centerlineDOFs.numCols() != ele_centerlinedofindices.size())
        FOUR_C_THROW(
            "Size mismatch! Need to assemble {} col values of centerline-Dof based "
            "stiffness matrix into element matrix but only got {} element-local Dof indices",
            row_matrix_centerlineDOFs.numCols(), ele_centerlinedofindices.size());

      // Fill in the values.
      for (unsigned int i_row = 0; i_row < (unsigned int)row_matrix_elementDOFs.numRows(); ++i_row)
        for (unsigned int i_col = 0; i_col < ele_centerlinedofindices.size(); ++i_col)
          row_matrix_elementDOFs(i_row, ele_centerlinedofindices[i_col]) =
              row_matrix_centerlineDOFs(i_row, i_col);
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void extract_pos_dof_vec_absolute_values(Core::FE::Discretization const& discret,
        Core::Elements::Element const* ele, const Core::LinAlg::Vector<double>& ia_discolnp,
        std::vector<double>& element_posdofvec_absolutevalues)
    {
      std::vector<double> eledispvec;

      // extract the Dof values of this element from displacement vector
      get_current_element_dis(discret, ele, ia_discolnp, eledispvec);

      Discret::Elements::Beam3Base const* beam_element_ptr =
          dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

      if (beam_element_ptr != nullptr)
      {
        // get the current absolute values for those Dofs relevant for centerline interpolation
        // initial values are added by element itself
        beam_element_ptr->extract_centerline_dof_values_from_element_state_vector(
            eledispvec, element_posdofvec_absolutevalues, true);
      }
      else
      {
        element_posdofvec_absolutevalues = eledispvec;
        for (unsigned int dim = 0; dim < 3; ++dim)
          for (int node = 0; node < ele->num_node(); ++node)
            element_posdofvec_absolutevalues[3 * node + dim] += ele->nodes()[node]->x()[dim];
      }
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    void extract_pos_dof_vec_values(Core::FE::Discretization const& discret,
        Core::Elements::Element const* ele, const Core::LinAlg::Vector<double>& ia_discolnp,
        std::vector<double>& element_posdofvec_values)
    {
      std::vector<double> eledispvec;

      // extract the Dof values of this element from displacement vector
      get_current_element_dis(discret, ele, ia_discolnp, eledispvec);

      Discret::Elements::Beam3Base const* beam_element_ptr =
          dynamic_cast<const Discret::Elements::Beam3Base*>(ele);

      if (beam_element_ptr != nullptr)
      {
        // get the current  values for those Dofs relevant for centerline interpolation
        beam_element_ptr->extract_centerline_dof_values_from_element_state_vector(
            eledispvec, element_posdofvec_values, false);
      }
      else
      {
        // for non beam elements, just return the element Dof vector
        element_posdofvec_values = eledispvec;
      }
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    template <class T1, class T2>
    void apply_binding_spot_force_to_parent_elements(Core::FE::Discretization const& discret,
        Core::Geo::MeshFree::BoundingBox& pbb, Core::LinAlg::Vector<double>& disp_np_col,
        BeamInteraction::BeamLink& elepairptr,
        std::vector<Core::LinAlg::SerialDenseVector> const& bspotforce,
        std::vector<Core::LinAlg::SerialDenseVector>& eleforce)
    {
      // auxiliary transformation matrix, will be resized and reused
      Core::LinAlg::SerialDenseMatrix trafomatrix;

      T1* cast_ele1 = dynamic_cast<T1*>(discret.g_element(elepairptr.get_ele_gid(0)));
      T2* cast_ele2 = dynamic_cast<T2*>(discret.g_element(elepairptr.get_ele_gid(1)));

      for (unsigned int elei = 0; elei < 2; ++elei)
      {
        // get current element displacements
        std::vector<double> eledisp;
        get_current_unshifted_element_dis(
            discret, discret.g_element(elepairptr.get_ele_gid(elei)), disp_np_col, pbb, eledisp);
        const int numdof_ele = eledisp.size();

        // zero out and set correct size of transformation matrix
        trafomatrix.shape(6, numdof_ele);

        // I_variations
        if (elei == 0)
          cast_ele1->get_generalized_interpolation_matrix_variations_at_xi(trafomatrix,
              cast_ele1->get_binding_spot_xi(
                  elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(elei)),
              eledisp);
        else
          cast_ele2->get_generalized_interpolation_matrix_variations_at_xi(trafomatrix,
              cast_ele2->get_binding_spot_xi(
                  elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(elei)),
              eledisp);

        eleforce[elei].size(numdof_ele);
        Core::LinAlg::multiply_tn(eleforce[elei], trafomatrix, bspotforce[elei]);
      }
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    template <class T1, class T2>
    void apply_binding_spot_stiff_to_parent_elements(Core::FE::Discretization const& discret,
        Core::Geo::MeshFree::BoundingBox& pbb, Core::LinAlg::Vector<double>& disp_np_col,
        BeamInteraction::BeamLink& elepairptr,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& bspotstiff,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>& elestiff)
    {
      // Todo grill 02/17
      FOUR_C_THROW(
          "we can not evaluate the tangent stiffness matrix without evaluating "
          "the residual in case of Kirchhoff beam Beam3k. This is because we have a coupled "
          "term and the residual vector is required to compute the full linearization. "
          "Use the combined evaluation method instead!");

      // todo: put this in a loop
      Core::Elements::Element* ele1 = discret.g_element(elepairptr.get_ele_gid(0));
      Core::Elements::Element* ele2 = discret.g_element(elepairptr.get_ele_gid(1));

      // get current element displacements
      std::vector<double> ele1disp;
      get_current_unshifted_element_dis(discret, ele1, disp_np_col, pbb, ele1disp);
      const int numdof_ele1 = ele1disp.size();

      std::vector<double> ele2disp;
      get_current_unshifted_element_dis(discret, ele2, disp_np_col, pbb, ele2disp);
      const int numdof_ele2 = ele2disp.size();

      // transformation matrix, will be resized and reused for various needs
      Core::LinAlg::SerialDenseMatrix trafomatrix;

      // auxiliary matrices required to store intermediate results after first of two
      // consecutive matrix-matrix products for each of four stiffness matrices
      std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> auxmat(
          2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

      T1* cast_ele1 = dynamic_cast<T1*>(ele1);
      T2* cast_ele2 = dynamic_cast<T2*>(ele2);

      // element 1:
      {
        // zero out and set correct size of transformation matrix
        trafomatrix.shape(6, numdof_ele1);

        // i) I_variations
        cast_ele1->get_generalized_interpolation_matrix_variations_at_xi(trafomatrix,
            cast_ele1->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(0)),
            ele1disp);

        auxmat[0][0].shape(numdof_ele1, 6);
        Core::LinAlg::multiply_tn(auxmat[0][0], trafomatrix, bspotstiff[0][0]);

        auxmat[0][1].shape(numdof_ele1, 6);
        Core::LinAlg::multiply_tn(auxmat[0][1], trafomatrix, bspotstiff[0][1]);

        // ii) I_increments
        trafomatrix.shape(6, numdof_ele1);

        cast_ele1->get_generalized_interpolation_matrix_increments_at_xi(trafomatrix,
            cast_ele1->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(0)),
            ele1disp);

        elestiff[0][0].shape(numdof_ele1, numdof_ele1);
        Core::LinAlg::multiply(elestiff[0][0], auxmat[0][0], trafomatrix);

        auxmat[1][0].shape(6, numdof_ele1);
        Core::LinAlg::multiply(auxmat[1][0], bspotstiff[1][0], trafomatrix);
      }

      // element 2:
      {
        // i) I_variations
        trafomatrix.shape(6, numdof_ele2);

        cast_ele2->get_generalized_interpolation_matrix_variations_at_xi(trafomatrix,
            cast_ele2->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(1)),
            ele2disp);

        elestiff[1][0].shape(numdof_ele2, numdof_ele1);
        Core::LinAlg::multiply_tn(elestiff[1][0], trafomatrix, auxmat[1][0]);

        auxmat[1][1].shape(numdof_ele2, 6);
        Core::LinAlg::multiply_tn(auxmat[1][1], trafomatrix, bspotstiff[1][1]);

        // ii) I_increments
        trafomatrix.shape(6, numdof_ele2);

        cast_ele2->get_generalized_interpolation_matrix_increments_at_xi(trafomatrix,
            cast_ele2->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(1)),
            ele2disp);

        elestiff[0][1].shape(numdof_ele1, numdof_ele2);
        Core::LinAlg::multiply(elestiff[0][1], auxmat[1][0], trafomatrix);

        elestiff[1][1].shape(numdof_ele2, numdof_ele2);
        Core::LinAlg::multiply(elestiff[1][1], auxmat[1][1], trafomatrix);
      }
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    template <class T1, class T2>
    void apply_binding_spot_force_stiff_to_parent_elements(Core::FE::Discretization const& discret,
        Core::Geo::MeshFree::BoundingBox& pbb, Core::LinAlg::Vector<double>& disp_np_col,
        BeamInteraction::BeamLink& elepairptr,
        std::vector<Core::LinAlg::SerialDenseVector> const& bspotforce,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const& bspotstiff,
        std::vector<Core::LinAlg::SerialDenseVector>& eleforce,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>& elestiff)
    {
      // apply force on binding spots and corresponding linearizations to parent elements
      // and get discrete element force vectors and stiffness matrices

      /* Of course, the underlying idea is much more general:
       * Consider some kind of abstract interaction between two points \xi_1,\xi_2 \in [-1,1]
       * on two beam elements (or the same beam element).
       * We computed a discrete force/moment exerted on each of the two points; in addition,
       * we computed the linearizations with respect to the position/rotation vector of
       * this point and also the linearizations with respect to the position/rotation vector
       * of the other point (in general, the forces/moments depend on position/rotation
       * vector of both points).
       * Starting from here, we calculate the generalized interpolation matrices for
       * variations and increments of the position/rotation vectors at the two points
       * by expressing their dependency on the primary nodal DoFs of the corresponding
       * element (this is element specific, of course). Finally, we can 'transform' the
       * force vectors and stiffness matrices to discrete element force vectors and
       * stiffness matrices */

      // todo @grill: put this in a loop
      Core::Elements::Element* ele1 = discret.g_element(elepairptr.get_ele_gid(0));
      Core::Elements::Element* ele2 = discret.g_element(elepairptr.get_ele_gid(1));

      // get current element displacements
      std::vector<double> ele1disp;
      get_current_unshifted_element_dis(discret, ele1, disp_np_col, pbb, ele1disp);
      const int numdof_ele1 = ele1disp.size();

      std::vector<double> ele2disp;
      get_current_unshifted_element_dis(discret, ele2, disp_np_col, pbb, ele2disp);
      const int numdof_ele2 = ele2disp.size();

      // transformation matrix, will be resized and reused for various needs
      Core::LinAlg::SerialDenseMatrix trafomatrix;

      // auxiliary matrices required to store intermediate results after first of two
      // consecutive matrix-matrix products for each of four stiffness matrices
      std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> auxmat(
          2, std::vector<Core::LinAlg::SerialDenseMatrix>(2));

      // contribution to stiffmat from linearization of generalized interpolation matrix for
      // variations
      Core::LinAlg::SerialDenseMatrix stiffmat_lin_Ivar;

      // zero out and set correct size of transformation matrix
      trafomatrix.shape(6, numdof_ele1);

      T1* cast_ele1 = dynamic_cast<T1*>(ele1);
      T2* cast_ele2 = dynamic_cast<T2*>(ele2);

      // todo:
      // element 1:
      {
        // i) I_variations
        cast_ele1->get_generalized_interpolation_matrix_variations_at_xi(trafomatrix,
            cast_ele1->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(0)),
            ele1disp);

        eleforce[0].size(numdof_ele1);
        Core::LinAlg::multiply_tn(eleforce[0], trafomatrix, bspotforce[0]);

        auxmat[0][0].shape(numdof_ele1, 6);
        Core::LinAlg::multiply_tn(auxmat[0][0], trafomatrix, bspotstiff[0][0]);

        auxmat[0][1].shape(numdof_ele1, 6);
        Core::LinAlg::multiply_tn(auxmat[0][1], trafomatrix, bspotstiff[0][1]);

        // ii) I_increments
        trafomatrix.shape(6, numdof_ele1);

        cast_ele1->get_generalized_interpolation_matrix_increments_at_xi(trafomatrix,
            cast_ele1->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(0)),
            ele1disp);

        elestiff[0][0].shape(numdof_ele1, numdof_ele1);
        Core::LinAlg::multiply(elestiff[0][0], auxmat[0][0], trafomatrix);

        auxmat[1][0].shape(6, numdof_ele1);
        Core::LinAlg::multiply(auxmat[1][0], bspotstiff[1][0], trafomatrix);


        // additional contribution from linearization of generalized interpolation matrix for
        // variations
        stiffmat_lin_Ivar.shape(numdof_ele1, numdof_ele1);

        cast_ele1->get_stiffmat_resulting_from_generalized_interpolation_matrix_at_xi(
            stiffmat_lin_Ivar,
            cast_ele1->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(0)),
            ele1disp, bspotforce[0]);

        Core::LinAlg::update(1.0, stiffmat_lin_Ivar, 1.0, elestiff[0][0]);
      }

      // element 2
      {
        // i) I_variations
        trafomatrix.shape(6, numdof_ele2);

        cast_ele2->get_generalized_interpolation_matrix_variations_at_xi(trafomatrix,
            cast_ele2->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(1)),
            ele2disp);

        eleforce[1].size(numdof_ele2);
        Core::LinAlg::multiply_tn(eleforce[1], trafomatrix, bspotforce[1]);

        elestiff[1][0].shape(numdof_ele2, numdof_ele1);
        Core::LinAlg::multiply_tn(elestiff[1][0], trafomatrix, auxmat[1][0]);

        auxmat[1][1].shape(numdof_ele2, 6);
        Core::LinAlg::multiply_tn(auxmat[1][1], trafomatrix, bspotstiff[1][1]);

        // ii) I_increments
        trafomatrix.shape(6, numdof_ele2);

        cast_ele2->get_generalized_interpolation_matrix_increments_at_xi(trafomatrix,
            cast_ele2->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(1)),
            ele2disp);

        elestiff[0][1].shape(numdof_ele1, numdof_ele2);
        Core::LinAlg::multiply(elestiff[0][1], auxmat[0][1], trafomatrix);

        elestiff[1][1].shape(numdof_ele2, numdof_ele2);
        Core::LinAlg::multiply(elestiff[1][1], auxmat[1][1], trafomatrix);


        // additional contribution from linearization of generalized interpolation matrix for
        // variations
        stiffmat_lin_Ivar.shape(numdof_ele2, numdof_ele2);

        cast_ele2->get_stiffmat_resulting_from_generalized_interpolation_matrix_at_xi(
            stiffmat_lin_Ivar,
            cast_ele2->get_binding_spot_xi(
                elepairptr.get_linker_type(), elepairptr.get_loc_b_spot_num(1)),
            ele2disp, bspotforce[1]);

        Core::LinAlg::update(1.0, stiffmat_lin_Ivar, 1.0, elestiff[1][1]);
      }
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void setup_ele_type_map_extractor(
        const Core::FE::Discretization& discret, Core::LinAlg::MultiMapExtractor& eletypeextractor)
    {
      std::vector<std::set<int>> eletypeset(3);

      for (int i = 0; i < discret.num_my_row_elements(); ++i)
      {
        // get ele pointer
        Core::Elements::Element* eleptr = discret.l_row_element(i);

        if (dynamic_cast<Discret::Elements::Beam3Base const*>(eleptr) != nullptr)
        {
          eletypeset[0].insert(eleptr->id());
        }
        else if (dynamic_cast<Discret::Elements::Rigidsphere const*>(eleptr) != nullptr)
        {
          eletypeset[1].insert(eleptr->id());
        }
        else if (dynamic_cast<Discret::Elements::Solid const*>(eleptr) != nullptr ||
                 dynamic_cast<Discret::Elements::KirchhoffLoveShellNurbs const*>(eleptr) != nullptr)
        {
          eletypeset[2].insert(eleptr->id());
        }
        else
        {
          FOUR_C_THROW("eletype multi map extractor cannot yet handle current element type.");
        }
      }

      std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps(eletypeset.size());
      for (int i = 0; i < static_cast<int>(eletypeset.size()); ++i)
      {
        std::vector<int> mapvec(eletypeset[i].begin(), eletypeset[i].end());
        eletypeset[i].clear();
        maps[i] = std::make_shared<Core::LinAlg::Map>(
            -1, mapvec.size(), mapvec.data(), 0, discret.get_comm());
      }

      eletypeextractor.setup(*discret.element_row_map(), maps);
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    void update_dof_map_of_vector(Core::FE::Discretization& discret,
        std::shared_ptr<Core::LinAlg::Vector<double>>& dofmapvec,
        std::shared_ptr<Core::LinAlg::Vector<double>> old)
    {
      if (dofmapvec != nullptr)
      {
        if (old == nullptr) old = dofmapvec;
        dofmapvec = Core::LinAlg::create_vector(*discret.dof_row_map(), true);
        Core::LinAlg::export_to(*old, *dofmapvec);
      }
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    long long cantor_pairing(std::pair<int, int> const& pair)
    {
      long long z = 0.5 * (pair.first + pair.second) * (pair.first + pair.second + 1) + pair.second;

#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (z > std::numeric_limits<long long>::max())
        FOUR_C_THROW(" Your cantor paired value exceeds limit of data type int.");
      if (pair != cantor_de_pairing(z))
        FOUR_C_THROW(
            " {} and {} cannot be paired using Cantor pairing function", pair.first, pair.second);
#endif

      return z;
    }

    /*----------------------------------------------------------------------------*
     *----------------------------------------------------------------------------*/
    std::pair<int, int> cantor_de_pairing(long long z)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (z > std::numeric_limits<long long>::max() / 8)
        FOUR_C_THROW(" Your cantor paired value will exceed limit of data type int.");
#endif

      long long w = std::floor((std::sqrt(8.0 * z + 1.0) - 1.0) * 0.5);
      long long t = (w + 1) * w * 0.5;

      std::pair<int, int> pair;
      pair.second = z - t;
      pair.first = w - pair.second;

      return pair;
    }

    /*-----------------------------------------------------------------------------*
     *-----------------------------------------------------------------------------*/
    Core::Binstrategy::Utils::BinContentType convert_element_to_bin_content_type(
        const Core::Elements::Element* ele)
    {
      if (dynamic_cast<Discret::Elements::Beam3Base const*>(ele) != nullptr)
      {
        return Core::Binstrategy::Utils::BinContentType::Beam;
      }
      else if (dynamic_cast<Discret::Elements::Rigidsphere const*>(ele) != nullptr)
      {
        return Core::Binstrategy::Utils::BinContentType::RigidSphere;
      }
      else if (dynamic_cast<Discret::Elements::Solid const*>(ele) != nullptr ||
               dynamic_cast<Discret::Elements::KirchhoffLoveShellNurbs const*>(ele) != nullptr)
      {
        return Core::Binstrategy::Utils::BinContentType::Solid;
      }
      else
      {
        FOUR_C_THROW(
            " Element you are about to assign to a bin could not be converted"
            " to a valid bin content type. ");
      }
    }


    //-----------------------------------------------------------------------------
    // explicit template instantiation (to please every compiler)
    //-----------------------------------------------------------------------------
    template void set_filament_binding_spot_positions(
        std::shared_ptr<Core::FE::Discretization>, BeamInteraction::CrosslinkingParams&);
    template void set_filament_binding_spot_positions(
        std::shared_ptr<Core::FE::Discretization>, BeamInteraction::SphereBeamLinkingParams&);

    template void apply_binding_spot_force_to_parent_elements<Discret::Elements::Beam3Base,
        Discret::Elements::Beam3Base>(Core::FE::Discretization const&,
        Core::Geo::MeshFree::BoundingBox&, Core::LinAlg::Vector<double>&,
        BeamInteraction::BeamLink&, std::vector<Core::LinAlg::SerialDenseVector> const&,
        std::vector<Core::LinAlg::SerialDenseVector>&);
    template void apply_binding_spot_force_to_parent_elements<Discret::Elements::Rigidsphere,
        Discret::Elements::Beam3Base>(Core::FE::Discretization const&,
        Core::Geo::MeshFree::BoundingBox&, Core::LinAlg::Vector<double>&,
        BeamInteraction::BeamLink&, std::vector<Core::LinAlg::SerialDenseVector> const&,
        std::vector<Core::LinAlg::SerialDenseVector>&);

    template void apply_binding_spot_stiff_to_parent_elements<Discret::Elements::Beam3Base,
        Discret::Elements::Beam3Base>(Core::FE::Discretization const&,
        Core::Geo::MeshFree::BoundingBox&, Core::LinAlg::Vector<double>&,
        BeamInteraction::BeamLink&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>&);
    template void apply_binding_spot_stiff_to_parent_elements<Discret::Elements::Rigidsphere,
        Discret::Elements::Beam3Base>(Core::FE::Discretization const&,
        Core::Geo::MeshFree::BoundingBox&, Core::LinAlg::Vector<double>&,
        BeamInteraction::BeamLink&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>&);

    template void apply_binding_spot_force_stiff_to_parent_elements<Discret::Elements::Beam3Base,
        Discret::Elements::Beam3Base>(Core::FE::Discretization const&,
        Core::Geo::MeshFree::BoundingBox&, Core::LinAlg::Vector<double>&,
        BeamInteraction::BeamLink&, std::vector<Core::LinAlg::SerialDenseVector> const&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const&,
        std::vector<Core::LinAlg::SerialDenseVector>&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>&);
    template void apply_binding_spot_force_stiff_to_parent_elements<Discret::Elements::Rigidsphere,
        Discret::Elements::Beam3Base>(Core::FE::Discretization const&,
        Core::Geo::MeshFree::BoundingBox&, Core::LinAlg::Vector<double>&,
        BeamInteraction::BeamLink&, std::vector<Core::LinAlg::SerialDenseVector> const&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>> const&,
        std::vector<Core::LinAlg::SerialDenseVector>&,
        std::vector<std::vector<Core::LinAlg::SerialDenseMatrix>>&);

    template void get_element_centerline_gid_indices<6>(Core::FE::Discretization const&,
        const Core::Elements::Element*, Core::LinAlg::Matrix<6, 1, int>&);
    template void get_element_centerline_gid_indices<12>(Core::FE::Discretization const&,
        const Core::Elements::Element*, Core::LinAlg::Matrix<12, 1, int>&);

  }  // namespace Utils
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE
