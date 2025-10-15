// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fsi_utils.hpp"

#include "4C_adapter_ale_fsi.hpp"
#include "4C_adapter_str_fsiwrapper.hpp"
#include "4C_coupling_adapter.hpp"
#include "4C_coupling_adapter_mortar.hpp"
#include "4C_fem_condition_utils.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_geometry_searchtree.hpp"
#include "4C_fem_geometry_searchtree_service.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_solid_3D_ele_line.hpp"
#include "4C_solid_3D_ele_surface.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::Utils::fluid_ale_nodes_disjoint(
    Core::FE::Discretization& fluiddis, Core::FE::Discretization& aledis)
{
  // flag indicating whether fluid and ALE node numbers have are non-overlapping or not
  bool isdisjoint = false;

  // try a simple check that should work for most cases
  if (fluiddis.node_row_map()->max_all_gid() < aledis.node_row_map()->min_all_gid() or
      fluiddis.node_row_map()->min_all_gid() > aledis.node_row_map()->max_all_gid())
  {
    // no overlap of node numbers
    isdisjoint = true;
  }
  else  // do a more sophisticated check
  {
    // get node row maps
    std::shared_ptr<const Core::LinAlg::Map> fluidmap =
        std::make_shared<Core::LinAlg::Map>(*fluiddis.node_row_map());
    std::shared_ptr<const Core::LinAlg::Map> alemap =
        std::make_shared<Core::LinAlg::Map>(*aledis.node_row_map());

    // Create intersection of fluid and ALE map
    std::vector<std::shared_ptr<const Core::LinAlg::Map>> intersectionmaps;
    intersectionmaps.push_back(fluidmap);
    intersectionmaps.push_back(alemap);
    std::shared_ptr<Core::LinAlg::Map> intersectionmap =
        Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

    if (intersectionmap->num_global_elements() == 0) isdisjoint = true;
  }

  return isdisjoint;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
// class SlideAleUtils
/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
FSI::Utils::SlideAleUtils::SlideAleUtils(std::shared_ptr<Core::FE::Discretization> structdis,
    std::shared_ptr<Core::FE::Discretization> fluiddis, Coupling::Adapter::CouplingMortar& coupsf,
    bool structcoupmaster, Inpar::FSI::SlideALEProj aleproj)
    : aletype_(aleproj)
{
  structcoupmaster_ = structcoupmaster;

  coupff_ = std::make_shared<Coupling::Adapter::CouplingMortar>(
      Global::Problem::instance()->n_dim(), Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type());

  // declare struct objects in interface
  std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>> structelements;
  std::map<int, std::shared_ptr<Core::Elements::Element>> structmelements;
  std::map<int, std::shared_ptr<Core::Elements::Element>> structdelements;
  std::map<int, Core::Nodes::Node*> dummy1;                 // dummy map
  std::map<int, std::map<int, Core::Nodes::Node*>> dummy2;  // dummy map
  std::map<int, Core::Nodes::Node*> structmnodes;  // partial map of sticking structure nodes
  std::map<int, Core::Nodes::Node*> structdnodes;  // partial map of centerdisp structure nodes
  std::map<int, std::map<int, Core::Nodes::Node*>> structgnodes;  // complete map of structure nodes

  // initialize struct objects in interface
  Core::Conditions::find_condition_objects(
      *structdis, dummy2, structgnodes, structelements, "FSICoupling");
  Core::Conditions::find_condition_objects(
      *structdis, dummy1, structmnodes, structmelements, "FSICouplingNoSlide");
  Core::Conditions::find_condition_objects(
      *structdis, dummy1, structdnodes, structdelements, "FSICouplingCenterDisp");
  istructdispnodes_ = structdnodes;
  istructdispeles_ = structdelements;
  istructslideles_ = structelements;

  std::vector<int> slideeleidvector;

  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator it;
  std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>::iterator meit;

  for (it = structmelements.begin(); it != structmelements.end(); it++)
  {
    int err = 0;
    for (meit = istructslideles_.begin(); meit != istructslideles_.end(); meit++)
      err += meit->second.erase((*it).first);
    if (!err) FOUR_C_THROW("Non sliding interface has to be a subset of FSI-interface or empty");
  }

  int max_id = 0;
  // find max FSI condition ID
  for (meit = istructslideles_.begin(); meit != istructslideles_.end(); meit++)
  {
    if (meit->first > max_id)
    {
      max_id = meit->first;
    }
  }

  maxid_ = Core::Communication::max_all(max_id, structdis->get_comm());

  // declare fluid objects in interface
  std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>> fluidelements;
  std::map<int, std::shared_ptr<Core::Elements::Element>> fluidmelements;
  std::map<int, std::map<int, Core::Nodes::Node*>> fluidnodes;  // complete map of fluid nodes
  std::map<int, Core::Nodes::Node*> fluidmnodes;  // partial map of sticking fluid nodes

  // initialize struct objects in interface
  Core::Conditions::find_condition_objects(
      *fluiddis, fluidnodes, dummy2, fluidelements, "FSICoupling");
  Core::Conditions::find_condition_objects(
      *fluiddis, fluidmnodes, dummy1, fluidmelements, "FSICouplingNoSlide");
  ifluidconfnodes_ = fluidmnodes;
  ifluidslidnodes_ = fluidnodes;
  ifluidslideles_ = fluidelements;

  for (it = fluidmelements.begin(); it != fluidmelements.end(); it++)
  {
    int err = 0;
    for (meit = ifluidslideles_.begin(); meit != ifluidslideles_.end(); meit++)
      err += meit->second.erase((*it).first);
    if (!err) FOUR_C_THROW("Non sliding interface has to be a subset of FSI-interface or empty");
  }

  std::map<int, Core::Nodes::Node*>::iterator nit;
  std::map<int, std::map<int, Core::Nodes::Node*>>::iterator mnit;
  for (nit = ifluidconfnodes_.begin(); nit != ifluidconfnodes_.end(); nit++)
  {
    int err = 0;
    for (mnit = ifluidslidnodes_.begin(); mnit != ifluidslidnodes_.end(); mnit++)
      err += mnit->second.erase((*nit).first);
    if (!err) FOUR_C_THROW("Non sliding interface has to be a subset of FSI-interface or empty");
  }

  std::shared_ptr<Core::LinAlg::Map> structdofrowmap;
  std::shared_ptr<Core::LinAlg::Map> fluiddofrowmap;


  // useful displacement vectors
  if (structcoupmaster_)
  {
    structdofrowmap_ = coupsf.master_dof_map();
    fluiddofrowmap_ = coupsf.slave_dof_map();
  }
  else
  {
    structdofrowmap_ = coupsf.slave_dof_map();
    fluiddofrowmap_ = coupsf.master_dof_map();
  }

  std::shared_ptr<Core::LinAlg::Map> dofrowmap =
      Core::LinAlg::merge_map(*structdofrowmap_, *fluiddofrowmap_, true);
  idispms_ = Core::LinAlg::create_vector(*dofrowmap, true);

  iprojhist_ = std::make_shared<Core::LinAlg::Vector<double>>(*fluiddofrowmap_, true);


  centerdisptotal_.resize(Global::Problem::instance()->n_dim());

  redundant_elements(coupsf, structdis->get_comm());

  maxmindist_ = 1.0e-1;

  // coupling condition at the fsi interface: displacements (=number spacial dimensions) are
  // coupled) e.g.: 3D: coupleddof = [1, 1, 1]
  std::vector<int> coupleddof(Global::Problem::instance()->n_dim(), 1);

  // this setup only initialize two sets of identical mortar elements (master and slave)
  // -> projection matrix is a unity matrix
  coupff_->setup(fluiddis, fluiddis, nullptr, coupleddof, "FSICoupling", fluiddis->get_comm(),
      Global::Problem::instance()->function_manager(),
      Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->discretization_map(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), false, true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Utils::SlideAleUtils::remeshing(Adapter::FSIStructureWrapper& structure,
    Core::FE::Discretization& fluiddis, Core::LinAlg::Vector<double>& idispale,
    Core::LinAlg::Vector<double>& iprojdispale, Coupling::Adapter::CouplingMortar& coupsf,
    MPI_Comm comm)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> idisptotal = structure.extract_interface_dispnp();
  const int dim = Global::Problem::instance()->n_dim();

  // project sliding fluid nodes onto struct interface surface
  slide_projection(structure, fluiddis, idispale, iprojdispale, coupsf, comm);

  // For the NON sliding ALE Nodes, use standard ALE displacements

  std::map<int, Core::Nodes::Node*>::const_iterator nodeiter;
  for (nodeiter = ifluidconfnodes_.begin(); nodeiter != ifluidconfnodes_.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;
    std::vector<int> gids = fluiddis.dof(node);

    // current coord of ale node = ref coord + ifluid_
    std::vector<double> finaldxyz(dim);

    for (int p = 0; p < dim; p++) finaldxyz[p] = idispale[fluiddofrowmap_->lid(gids[p])];

    iprojdispale.replace_global_values(dim, finaldxyz.data(), gids.data());
  }

  // merge displacement values of interface nodes (struct+fluid) into idispms_ for mortar
  idispms_->put_scalar(0.0);

  std::shared_ptr<Core::LinAlg::Map> dofrowmap =
      Core::LinAlg::merge_map(*structdofrowmap_, *fluiddofrowmap_, true);
  Core::LinAlg::Import msimpo(*dofrowmap, *structdofrowmap_);
  Core::LinAlg::Import slimpo(*dofrowmap, *fluiddofrowmap_);

  idispms_->import(*idisptotal, msimpo, Add);
  idispms_->import(iprojdispale, slimpo, Add);

  iprojhist_->update(1.0, iprojdispale, 0.0);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Utils::SlideAleUtils::evaluate_mortar(Core::LinAlg::Vector<double>& idispstruct,
    Core::LinAlg::Vector<double>& idispfluid, Coupling::Adapter::CouplingMortar& coupsf)
{
  // merge displacement values of interface nodes (struct+fluid) into idispms_ for mortar
  idispms_->put_scalar(0.0);

  std::shared_ptr<Core::LinAlg::Map> dofrowmap =
      Core::LinAlg::merge_map(*structdofrowmap_, *fluiddofrowmap_, true);
  Core::LinAlg::Import master_importer(*dofrowmap, *structdofrowmap_);
  Core::LinAlg::Import slave_importer(*dofrowmap, *fluiddofrowmap_);

  if (idispms_->import(idispstruct, master_importer, Add)) FOUR_C_THROW("Import operation failed.");
  if (idispms_->import(idispfluid, slave_importer, Add)) FOUR_C_THROW("Import operation failed.");

  // new D,M,Dinv out of disp of struct and fluid side
  coupsf.evaluate(idispms_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Utils::SlideAleUtils::evaluate_fluid_mortar(
    std::shared_ptr<Core::LinAlg::Vector<double>> ima,
    std::shared_ptr<Core::LinAlg::Vector<double>> isl)
{
  // new D,M,Dinv out of fluid disp before and after sliding
  coupff_->evaluate(ima, isl);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<Core::LinAlg::Vector<double>> FSI::Utils::SlideAleUtils::interpolate_fluid(
    const Core::LinAlg::Vector<double>& uold)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> unew = coupff_->master_to_slave(uold);
  unew->replace_map(uold.get_map());

  return unew;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> FSI::Utils::SlideAleUtils::centerdisp(
    Adapter::FSIStructureWrapper& structure, MPI_Comm comm)
{
  std::shared_ptr<Core::FE::Discretization> structdis = structure.discretization();

  std::shared_ptr<Core::LinAlg::Vector<double>> idispn = structure.extract_interface_dispn();
  std::shared_ptr<Core::LinAlg::Vector<double>> idisptotal = structure.extract_interface_dispnp();
  std::shared_ptr<Core::LinAlg::Vector<double>> idispstep = structure.extract_interface_dispnp();

  idispstep->update(-1.0, *idispn, 1.0);

  const int dim = Global::Problem::instance()->n_dim();
  // get structure and fluid discretizations  and set stated for element evaluation
  Core::LinAlg::Vector<double> idisptotalcol(*structdis->dof_col_map(), true);
  export_to(*idisptotal, idisptotalcol);
  Core::LinAlg::Vector<double> idispstepcol(*structdis->dof_col_map(), true);
  export_to(*idispstep, idispstepcol);

  structdis->set_state("displacementtotal", idisptotalcol);
  structdis->set_state("displacementincr", idispstepcol);

  // define stuff needed by the elements
  Teuchos::ParameterList params;
  Core::LinAlg::SerialDenseMatrix elematrix1;
  Core::LinAlg::SerialDenseMatrix elematrix2;
  Core::LinAlg::SerialDenseVector elevector1;
  Core::LinAlg::SerialDenseVector elevector2;
  Core::LinAlg::SerialDenseVector elevector3;

  // prepare variables for length (2D) or area (3D) of the interface
  std::vector<double> mycenterdisp(dim);
  std::vector<double> centerdisp(dim);
  double mylengthcirc = 0.0;
  double lengthcirc = 0.0;

  // calculating the center displacement by evaluating structure interface elements
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator elemiter;
  for (elemiter = istructdispeles_.begin(); elemiter != istructdispeles_.end(); ++elemiter)
  {
    std::shared_ptr<Core::Elements::Element> iele = elemiter->second;
    std::vector<int> lm;
    std::vector<int> lmowner;
    std::vector<int> lmstride;
    iele->location_vector(*structdis, lm, lmowner, lmstride);
    elevector2.size(1);    // length of circ with gaussinteg
    elevector3.size(dim);  // centerdisp part of ele

    params.set<std::string>("action", "calc_struct_centerdisp");
    int err = iele->evaluate(
        params, *structdis, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
    if (err) FOUR_C_THROW("error while evaluating elements");
    mylengthcirc += elevector2[0];

    // disp of the interface
    for (int i = 0; i < dim; i++)
    {
      mycenterdisp[i] += elevector3[i];
    }
  }  // end of ele loop
  structdis->clear_state();

  // Communicate to 'assemble' length and center displacements
  lengthcirc = Core::Communication::sum_all(mylengthcirc, comm);
  centerdisp = Core::Communication::sum_all(mycenterdisp, comm);

  if (lengthcirc <= 1.0E-6) FOUR_C_THROW("Zero interface length!");

  // calculating the final disp of the interface and summation over all time steps
  for (int i = 0; i < dim; i++)
  {
    centerdisp[i] = centerdisp[i] / lengthcirc;
    centerdisptotal_[i] += centerdisp[i];
  }

  return centerdisp;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::map<int, Core::LinAlg::Matrix<3, 1>> FSI::Utils::SlideAleUtils::current_struct_pos(
    Core::LinAlg::Vector<double>& reddisp, Core::FE::Discretization& interfacedis,
    std::map<int, double>& maxcoord)
{
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions;
  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator eleiter;
  std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>::const_iterator meleiter;

  // map with fully reduced struct element distribution
  for (meleiter = structreduelements_.begin(); meleiter != structreduelements_.end(); meleiter++)
  {
    maxcoord[meleiter->first] = 0.0;
    for (eleiter = meleiter->second.begin(); eleiter != meleiter->second.end(); eleiter++)
    {
      std::shared_ptr<Core::Elements::Element> tmpele = eleiter->second;

      const int* n = tmpele->node_ids();

      // fill currentpositions
      for (int j = 0; j < tmpele->num_node(); j++)
      {
        const int gid = n[j];
        const Core::Nodes::Node* node = interfacedis.g_node(gid);
        std::vector<int> lm;
        lm.reserve(3);
        // extract global dof ids
        interfacedis.dof(node, lm);
        Core::LinAlg::Matrix<3, 1> currpos;

        std::vector<double> mydisp = Core::FE::extract_values(reddisp, lm);

        for (int a = 0; a < 3; a++)
        {
          currpos(a, 0) = node->x()[a] + mydisp[a];
        }
        if (abs(currpos(2, 0)) > maxcoord[meleiter->first])
          maxcoord[meleiter->first] = abs(currpos(2, 0));
        currentpositions[node->id()] = currpos;
      }
    }
  }

  return currentpositions;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::Utils::SlideAleUtils::slide_projection(
    Adapter::FSIStructureWrapper& structure, Core::FE::Discretization& fluiddis,
    Core::LinAlg::Vector<double>& idispale, Core::LinAlg::Vector<double>& iprojdispale,
    Coupling::Adapter::CouplingMortar& coupsf, MPI_Comm comm

)
{
  const int dim = Global::Problem::instance()->n_dim();

  std::shared_ptr<Core::LinAlg::Vector<double>> idispnp = structure.extract_interface_dispnp();

  // Redistribute displacement of structnodes on the interface to all processors.
  Core::LinAlg::Import interimpo(*structfullnodemap_, *structdofrowmap_);
  std::shared_ptr<Core::LinAlg::Vector<double>> reddisp =
      Core::LinAlg::create_vector(*structfullnodemap_, true);
  reddisp->import(*idispnp, interimpo, Add);

  Core::FE::Discretization& interfacedis = coupsf.interface()->discret();
  std::map<int, double> rotrat;
  // currentpositions of struct nodes for the search tree (always 3 coordinates)
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions =
      current_struct_pos(*reddisp, interfacedis, rotrat);

  // calculate structural interface center of gravity
  std::vector<double> centerdisp_v = centerdisp(structure, comm);

  std::shared_ptr<Core::LinAlg::Vector<double>> frotfull =
      Core::LinAlg::create_vector(*fluiddofrowmap_, true);
  if (aletype_ == Inpar::FSI::ALEprojection_rot_z ||
      aletype_ == Inpar::FSI::ALEprojection_rot_zsphere)
  {
    rotation(coupsf.interface()->discret(), idispale, comm, rotrat, *frotfull);
  }


  std::map<int, std::map<int, Core::Nodes::Node*>>::iterator mnit;
  for (mnit = ifluidslidnodes_.begin(); mnit != ifluidslidnodes_.end(); ++mnit)
  {
    // translation + projection
    std::map<int, Core::Nodes::Node*>::const_iterator nodeiter;
    for (nodeiter = mnit->second.begin(); nodeiter != mnit->second.end(); ++nodeiter)
    {
      // Project fluid nodes onto the struct interface
      // init of search tree
      Core::Geo::SearchTree searchTree(5);
      const Core::LinAlg::Matrix<3, 2> rootBox =
          Core::Geo::get_xaab_bof_eles(structreduelements_[mnit->first], currentpositions);

      if (dim == 2)
        searchTree.initialize_tree_slide_ale(
            rootBox, structreduelements_[mnit->first], Core::Geo::TreeType(Core::Geo::QUADTREE));
      else if (dim == 3)
        searchTree.initialize_tree_slide_ale(
            rootBox, structreduelements_[mnit->first], Core::Geo::TreeType(Core::Geo::OCTTREE));
      else
        FOUR_C_THROW("wrong dimension");


      Core::Nodes::Node* node = nodeiter->second;
      std::vector<int> gids = fluiddis.dof(node);
      std::vector<int> lids(dim);
      for (int p = 0; p < dim; p++)
        // lids of gids of node
        lids[p] = fluiddofrowmap_->lid(gids[p]);

      // current coord of ale node.
      // Initialize as coordinates of current node, which is extremely important for 2D!
      Core::LinAlg::Matrix<3, 1> alenodecurr(node->x().data());

      // compute ALE position to project from
      if (aletype_ == Inpar::FSI::ALEprojection_curr)
      {
        // current coord of ale node = ref + centerdispincr + history
        for (int p = 0; p < dim; p++)
          alenodecurr(p, 0) = (node->x()[p]) + centerdisp_v[p] + 1.0 * (*iprojhist_)[(lids[p])];
      }
      else if (aletype_ == Inpar::FSI::ALEprojection_ref)
      {
        // current coord of ale node = ref + centerdisp
        for (int p = 0; p < dim; p++) alenodecurr(p, 0) = node->x()[p] + centerdisptotal_[p];
      }
      else if (aletype_ == Inpar::FSI::ALEprojection_rot_z ||
               aletype_ == Inpar::FSI::ALEprojection_rot_zsphere)
      {
        // current coord of ale node = ref + centerdisp
        for (int p = 0; p < dim; p++)
        {
          alenodecurr(p, 0) = node->x()[p] + (idispale)[(lids[p])] -
                              1.0 * rotrat[mnit->first] * (*frotfull)[(lids[p])];
        }
      }
      else
        FOUR_C_THROW("you should not turn up here!");


      // final displacement of projection
      std::vector<double> finaldxyz(dim);

      // search for near elements next to the query point (ie within a radius of 2x maxmindist)
      std::map<int, std::set<int>> closeeles = searchTree.search_elements_in_radius(
          interfacedis, currentpositions, alenodecurr, maxmindist_, 0);
      // if no close elements could be found, try with a much larger radius and print a warning
      if (closeeles.empty())
      {
        const double enlarge_factor = 100;
        std::cout << "WARNING: no elements found in radius r=" << maxmindist_
                  << ". Will try once with a " << static_cast<int>(enlarge_factor)
                  << "-times bigger radius!" << std::endl;
        closeeles = searchTree.search_elements_in_radius(
            interfacedis, currentpositions, alenodecurr, enlarge_factor * maxmindist_, 0);
        maxmindist_ *= 10.0;

        // if still no element is found, complain about it!
        if (closeeles.empty()) FOUR_C_THROW("No elements in a large radius! Should not happen!");
      }
      // search for the nearest point to project on
      Core::LinAlg::Matrix<3, 1> minDistCoords;
      if (dim == 2)
      {
        Core::Geo::nearest_2d_object_in_node(interfacedis, structreduelements_[mnit->first],
            currentpositions, closeeles, alenodecurr, minDistCoords);
        finaldxyz[0] = minDistCoords(0, 0) - node->x()[0];
        finaldxyz[1] = minDistCoords(1, 0) - node->x()[1];
      }
      else
      {
        Core::Geo::nearest_3d_object_in_node(interfacedis, structreduelements_[mnit->first],
            currentpositions, closeeles, alenodecurr, minDistCoords);
        finaldxyz[0] = minDistCoords(0, 0) - node->x()[0];
        finaldxyz[1] = minDistCoords(1, 0) - node->x()[1];
        finaldxyz[2] = minDistCoords(2, 0) - node->x()[2];
      }

      // store displacement into parallel vector
      iprojdispale.replace_global_values(dim, finaldxyz.data(), gids.data());
    }
  }
}

void FSI::Utils::SlideAleUtils::redundant_elements(
    Coupling::Adapter::CouplingMortar& coupsf, MPI_Comm comm)
{
  // We need the structure elements (NOT THE MORTAR-ELEMENTS!) on every processor for the projection
  // of the fluid nodes. Furthermore we need the current position of the structnodes on every
  // processor. Elements provided by interface discretization, necessary maps provided by interface.

  int soffset = 0;
  int foffset = 0;
  if (structcoupmaster_)
  {
    structfullnodemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->master_row_dofs()));
    structfullelemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->master_row_elements()));
    fluidfullnodemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->slave_row_dofs()));
    fluidfullelemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->slave_row_elements()));
    soffset = 0;
    foffset = fluidfullelemap_->min_my_gid();
  }
  else
  {
    fluidfullnodemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->master_row_dofs()));
    fluidfullelemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->master_row_elements()));
    structfullnodemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->slave_row_dofs()));
    structfullelemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->slave_row_elements()));
    soffset = structfullelemap_->min_my_gid();
    foffset = 0;
  }

  Core::FE::Discretization& interfacedis = coupsf.interface()->discret();

  std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>::iterator mapit;
  // build redundant version istructslideles_;
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator it;
  int dim = Global::Problem::instance()->n_dim();

  for (int i = 0; i <= maxid_; ++i)
  {
    std::vector<int> vstruslideleids;  // vector for ele ids
    if (istructslideles_.find(i) != istructslideles_.end())
    {
      for (it = istructslideles_[i].begin(); it != istructslideles_[i].end(); it++)
      {
        if (Core::Communication::my_mpi_rank(interfacedis.get_comm()) == (*it).second->owner())
          vstruslideleids.push_back(it->first);
      }
    }
    int globsum = 0;
    int partsum = (vstruslideleids.size());

    globsum = Core::Communication::sum_all(partsum, comm);
    // map with ele ids
    Core::LinAlg::Map mstruslideleids(
        globsum, vstruslideleids.size(), vstruslideleids.data(), 0, comm);
    // redundant version of it
    Core::LinAlg::Map redmstruslideleids(*Core::LinAlg::allreduce_e_map(mstruslideleids));

    for (int eleind = 0; eleind < redmstruslideleids.num_my_elements(); eleind++)
    {
      {
        Core::Elements::Element* tmpele =
            interfacedis.g_element(redmstruslideleids.gid(eleind) + soffset);
        if (dim == 3)
        {
          structreduelements_[i][tmpele->id()] =
              std::make_shared<Discret::Elements::SolidSurface>(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0);
        }
        else if (dim == 2)
        {
          structreduelements_[i][tmpele->id()] =
              std::make_shared<Discret::Elements::SolidLine<3>>(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0);
        }
      }
    }

    if (ifluidslideles_.find(i) != ifluidslideles_.end())
    {
      for (it = ifluidslideles_[i].begin(); it != ifluidslideles_[i].end(); it++)
      {
        Core::Elements::Element* tmpele = interfacedis.g_element(it->first + foffset);
        if (dim == 3)
        {
          ifluidslidstructeles_[i][tmpele->id()] =
              std::make_shared<Discret::Elements::SolidSurface>(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0);
        }
        else if (dim == 2)
        {
          ifluidslidstructeles_[i][tmpele->id()] =
              std::make_shared<Discret::Elements::SolidLine<3>>(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0);
        }
      }
    }
  }
}


void FSI::Utils::SlideAleUtils::rotation(
    Core::FE::Discretization& mtrdis,        ///< fluid discretization
    Core::LinAlg::Vector<double>& idispale,  ///< vector of ALE displacements
    MPI_Comm comm,                           ///< communicator
    std::map<int, double>& rotrat,           ///< rotation ratio of tangential displacements
    Core::LinAlg::Vector<double>&
        rotfull  ///< vector of full displacements in tangential directions
)
{
  std::shared_ptr<Core::LinAlg::Vector<double>> idispstep =
      Core::LinAlg::create_vector(*fluiddofrowmap_, false);
  idispstep->update(1.0, idispale, -1.0, *iprojhist_, 0.0);

  // get structure and fluid discretizations  and set state for element evaluation
  Core::LinAlg::Vector<double> idispstepcol(*mtrdis.dof_col_map(), false);
  export_to(*idispstep, idispstepcol);
  Core::LinAlg::Vector<double> idispnpcol(*mtrdis.dof_col_map(), false);
  Core::LinAlg::export_to(idispale, idispnpcol);

  mtrdis.set_state("displacementnp", idispnpcol);
  mtrdis.set_state("displacementincr", idispstepcol);

  std::map<int, std::map<int, std::shared_ptr<Core::Elements::Element>>>::iterator melit;
  for (int i = 0; i <= maxid_; ++i)
  {
    // prepare variables for length (2D) or area (3D) of the interface
    double myrotation = 0.0;
    double rotation = 0.0;
    double mylengthcirc = 0.0;
    double lengthcirc = 0.0;
    double maxcoord = 0.0;
    if (ifluidslidstructeles_.find(i) != ifluidslidstructeles_.end()) maxcoord = rotrat[i];

    std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator elemiter;
    for (elemiter = ifluidslidstructeles_[i].begin(); elemiter != ifluidslidstructeles_[i].end();
        elemiter++)
    {
      // define stuff needed by the elements
      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;
      Teuchos::ParameterList params;

      std::shared_ptr<Core::Elements::Element> iele = elemiter->second;
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      iele->location_vector(mtrdis, lm, lmowner, lmstride);
      elevector2.size(1);  // circumference (2D) or surface area (3D) of the considered elements
      elevector3.size(1);  // normalized displacement in tangential direction ('rotation')

      params.set<std::string>("action", "calc_struct_rotation");
      params.set<double>("maxcoord", maxcoord);
      params.set<Inpar::FSI::SlideALEProj>("aletype", aletype_);
      int err = iele->evaluate(
          params, mtrdis, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      mylengthcirc += elevector2[0];
      // disp of the interface
      myrotation += elevector3[0];
    }  // end of ele loop

    // Communicate to 'assemble' length and center displacements
    lengthcirc = Core::Communication::sum_all(mylengthcirc, comm);
    rotation = Core::Communication::sum_all(myrotation, comm);

    if (lengthcirc >= 1.0E-6)
    {
      // calculating the final disp of the interface and summation over all time steps
      rotrat[i] = rotation / lengthcirc;
    }

    // second round!
    // compute correction displacement to account for rotation
    for (elemiter = ifluidslidstructeles_[i].begin(); elemiter != ifluidslidstructeles_[i].end();
        elemiter++)
    {
      // define stuff needed by the elements
      Core::LinAlg::SerialDenseMatrix elematrix1;
      Core::LinAlg::SerialDenseMatrix elematrix2;
      Core::LinAlg::SerialDenseVector elevector1;
      Core::LinAlg::SerialDenseVector elevector2;
      Core::LinAlg::SerialDenseVector elevector3;
      Teuchos::ParameterList params;

      std::shared_ptr<Core::Elements::Element> iele = elemiter->second;
      std::vector<int> lm;
      std::vector<int> lmowner;
      std::vector<int> lmstride;
      iele->location_vector(mtrdis, lm, lmowner, lmstride);
      elevector1.size(lm.size());

      params.set<std::string>("action", "calc_undo_struct_rotation");
      params.set<double>("maxcoord", maxcoord);
      params.set<Inpar::FSI::SlideALEProj>("aletype", aletype_);
      int err = iele->evaluate(
          params, mtrdis, lm, elematrix1, elematrix2, elevector1, elevector2, elevector3);
      if (err) FOUR_C_THROW("error while evaluating elements");

      Core::LinAlg::assemble(rotfull, elevector1, lm, lmowner);
    }
  }
  mtrdis.clear_state();

  return;
}

void FSI::Utils::SlideAleUtils::output_restart(Core::IO::DiscretizationWriter& output)
{
  output.write_vector("projhist", iprojhist_);

  return;
}

void FSI::Utils::SlideAleUtils::read_restart(Core::IO::DiscretizationReader& reader)
{
  reader.read_vector(iprojhist_, "projhist");
}

FOUR_C_NAMESPACE_CLOSE
