/*----------------------------------------------------------------------------*/
/*! \file

\brief Utilities for FSI problems

\level 1

*/

/*----------------------------------------------------------------------------*/

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
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_assemble.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_mortar_interface.hpp"
#include "4C_so3_line.hpp"
#include "4C_so3_surface.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool FSI::UTILS::fluid_ale_nodes_disjoint(
    Teuchos::RCP<Core::FE::Discretization> fluiddis, Teuchos::RCP<Core::FE::Discretization> aledis)
{
  // flag indicating whether fluid and ALE node numbers have are non-overlapping or not
  bool isdisjoint = false;

  // try a simple check that should work for most cases
  if (fluiddis->node_row_map()->MaxAllGID() < aledis->node_row_map()->MinAllGID() or
      fluiddis->node_row_map()->MinAllGID() > aledis->node_row_map()->MaxAllGID())
  {
    // no overlap of node numbers
    isdisjoint = true;
  }
  else  // do a more sophisticated check
  {
    // get node row maps
    Teuchos::RCP<const Epetra_Map> fluidmap =
        Teuchos::rcp(new const Epetra_Map(*fluiddis->node_row_map()));
    Teuchos::RCP<const Epetra_Map> alemap =
        Teuchos::rcp(new const Epetra_Map(*aledis->node_row_map()));

    // Create intersection of fluid and ALE map
    std::vector<Teuchos::RCP<const Epetra_Map>> intersectionmaps;
    intersectionmaps.push_back(fluidmap);
    intersectionmaps.push_back(alemap);
    Teuchos::RCP<Epetra_Map> intersectionmap =
        Core::LinAlg::MultiMapExtractor::intersect_maps(intersectionmaps);

    if (intersectionmap->NumGlobalElements() == 0) isdisjoint = true;
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
FSI::UTILS::SlideAleUtils::SlideAleUtils(Teuchos::RCP<Core::FE::Discretization> structdis,
    Teuchos::RCP<Core::FE::Discretization> fluiddis, Coupling::Adapter::CouplingMortar& coupsf,
    bool structcoupmaster, Inpar::FSI::SlideALEProj aleproj)
    : aletype_(aleproj)
{
  structcoupmaster_ = structcoupmaster;

  coupff_ = Teuchos::rcp(new Coupling::Adapter::CouplingMortar(Global::Problem::instance()->n_dim(),
      Global::Problem::instance()->mortar_coupling_params(),
      Global::Problem::instance()->contact_dynamic_params(),
      Global::Problem::instance()->spatial_approximation_type()));

  // declare struct objects in interface
  std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>> structelements;
  std::map<int, Teuchos::RCP<Core::Elements::Element>> structmelements;
  std::map<int, Teuchos::RCP<Core::Elements::Element>> structdelements;
  std::map<int, Core::Nodes::Node*> dummy1;                 // dummy map
  std::map<int, std::map<int, Core::Nodes::Node*>> dummy2;  // dummy map
  std::map<int, Core::Nodes::Node*> structmnodes;  // partial map of sticking structure nodes
  std::map<int, Core::Nodes::Node*> structdnodes;  // partial map of centerdisp structure nodes
  std::map<int, std::map<int, Core::Nodes::Node*>> structgnodes;  // complete map of strucutre nodes

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

  std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator eit;
  std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>>::iterator meit;

  for (eit = structmelements.begin(); eit != structmelements.end(); eit++)
  {
    int err = 0;
    for (meit = istructslideles_.begin(); meit != istructslideles_.end(); meit++)
      err += meit->second.erase((*eit).first);
    if (!err) FOUR_C_THROW("Non sliding interface has to be a subset of FSI-interface or empty");
  }

  int max_id = 0;
  // find max FSI condition ID
  for (meit = istructslideles_.begin(); meit != istructslideles_.end(); meit++)
  {
    //    for ( eit=meit->second.begin(); eit != meit->second.end(); eit++ )
    //    {
    //      //build slideeleidvector with unique distribution. Otherwise, AllreduceEMap() will
    //      complain in DEBUG if (structdis->Comm().MyPID()==(*eit).second->Owner())
    //        slideeleidvector.push_back((*eit).first);
    //    }
    //    const Epetra_Map slideelemap (-1, slideeleidvector.size(), slideeleidvector.data(), 0,
    //    structdis->Comm()); slideeleredmap_[meit->first] =
    //    Core::LinAlg::AllreduceEMap(slideelemap);
    if (meit->first > max_id) max_id = meit->first;
  }

  structdis->get_comm().MaxAll(&max_id, &maxid_, 1);

  // declare fluid objects in interface
  std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>> fluidelements;
  std::map<int, Teuchos::RCP<Core::Elements::Element>> fluidmelements;
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

  for (eit = fluidmelements.begin(); eit != fluidmelements.end(); eit++)
  {
    int err = 0;
    for (meit = ifluidslideles_.begin(); meit != ifluidslideles_.end(); meit++)
      err += meit->second.erase((*eit).first);
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

  Teuchos::RCP<Epetra_Map> structdofrowmap;
  Teuchos::RCP<Epetra_Map> fluiddofrowmap;


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

  Teuchos::RCP<Epetra_Map> dofrowmap =
      Core::LinAlg::merge_map(*structdofrowmap_, *fluiddofrowmap_, true);
  idispms_ = Core::LinAlg::create_vector(*dofrowmap, true);

  iprojhist_ = Teuchos::rcp(new Epetra_Vector(*fluiddofrowmap_, true));


  centerdisptotal_.resize(Global::Problem::instance()->n_dim());

  redundant_elements(coupsf, structdis->get_comm());

  maxmindist_ = 1.0e-1;

  // coupling condition at the fsi interface: displacements (=number spacial dimensions) are
  // coupled) e.g.: 3D: coupleddof = [1, 1, 1]
  std::vector<int> coupleddof(Global::Problem::instance()->n_dim(), 1);

  // this setup only initialize two sets of identical mortar elements (master and slave)
  // -> projection matrix is a unity matrix
  coupff_->setup(fluiddis, fluiddis, Teuchos::null, coupleddof, "FSICoupling", fluiddis->get_comm(),
      Global::Problem::instance()->function_manager(),
      Global::Problem::instance()->binning_strategy_params(),
      Global::Problem::instance()->discretization_map(),
      Global::Problem::instance()->output_control_file(),
      Global::Problem::instance()->spatial_approximation_type(), false, true);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::UTILS::SlideAleUtils::remeshing(Adapter::FSIStructureWrapper& structure,
    Teuchos::RCP<Core::FE::Discretization> fluiddis, Teuchos::RCP<Epetra_Vector> idispale,
    Teuchos::RCP<Epetra_Vector> iprojdispale, Coupling::Adapter::CouplingMortar& coupsf,
    const Epetra_Comm& comm)
{
  Teuchos::RCP<Epetra_Vector> idisptotal = structure.extract_interface_dispnp();
  const int dim = Global::Problem::instance()->n_dim();

  // project sliding fluid nodes onto struct interface surface
  slide_projection(structure, fluiddis, idispale, iprojdispale, coupsf, comm);

  // For the NON sliding ALE Nodes, use standard ALE displacements

  std::map<int, Core::Nodes::Node*>::const_iterator nodeiter;
  for (nodeiter = ifluidconfnodes_.begin(); nodeiter != ifluidconfnodes_.end(); ++nodeiter)
  {
    Core::Nodes::Node* node = nodeiter->second;
    std::vector<int> lids(dim);
    for (int p = 0; p < dim; p++)
      // lids of gids of node
      lids[p] = fluiddofrowmap_->LID((fluiddis->dof(node))[p]);

    // current coord of ale node = ref coord + ifluid_
    std::vector<double> finaldxyz(dim);

    for (int p = 0; p < dim; p++) finaldxyz[p] = (*idispale)[(lids[p])];

    int err = iprojdispale->ReplaceMyValues(dim, finaldxyz.data(), lids.data());
    if (err == 1) FOUR_C_THROW("error while replacing values");
  }

  // merge displacement values of interface nodes (struct+fluid) into idispms_ for mortar
  idispms_->PutScalar(0.0);

  Teuchos::RCP<Epetra_Map> dofrowmap =
      Core::LinAlg::merge_map(*structdofrowmap_, *fluiddofrowmap_, true);
  Teuchos::RCP<Epetra_Import> msimpo =
      Teuchos::rcp(new Epetra_Import(*dofrowmap, *structdofrowmap_));
  Teuchos::RCP<Epetra_Import> slimpo =
      Teuchos::rcp(new Epetra_Import(*dofrowmap, *fluiddofrowmap_));

  idispms_->Import(*idisptotal, *msimpo, Add);
  idispms_->Import(*iprojdispale, *slimpo, Add);

  iprojhist_->Update(1.0, *iprojdispale, 0.0);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::UTILS::SlideAleUtils::evaluate_mortar(Teuchos::RCP<Epetra_Vector> idispstruct,
    Teuchos::RCP<Epetra_Vector> idispfluid, Coupling::Adapter::CouplingMortar& coupsf)
{
  // merge displacement values of interface nodes (struct+fluid) into idispms_ for mortar
  idispms_->PutScalar(0.0);

  Teuchos::RCP<Epetra_Map> dofrowmap =
      Core::LinAlg::merge_map(*structdofrowmap_, *fluiddofrowmap_, true);
  Teuchos::RCP<Epetra_Import> master_importer =
      Teuchos::rcp(new Epetra_Import(*dofrowmap, *structdofrowmap_));
  Teuchos::RCP<Epetra_Import> slave_importer =
      Teuchos::rcp(new Epetra_Import(*dofrowmap, *fluiddofrowmap_));

  if (idispms_->Import(*idispstruct, *master_importer, Add))
    FOUR_C_THROW("Import operation failed.");
  if (idispms_->Import(*idispfluid, *slave_importer, Add)) FOUR_C_THROW("Import operation failed.");

  // new D,M,Dinv out of disp of struct and fluid side
  coupsf.evaluate(idispms_);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FSI::UTILS::SlideAleUtils::evaluate_fluid_mortar(
    Teuchos::RCP<Epetra_Vector> ima, Teuchos::RCP<Epetra_Vector> isl)
{
  // new D,M,Dinv out of fluid disp before and after sliding
  coupff_->evaluate(ima, isl);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
Teuchos::RCP<Epetra_Vector> FSI::UTILS::SlideAleUtils::interpolate_fluid(
    Teuchos::RCP<const Epetra_Vector> uold)
{
  Teuchos::RCP<Epetra_Vector> unew = coupff_->master_to_slave(uold);
  unew->ReplaceMap(uold->Map());

  return unew;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::vector<double> FSI::UTILS::SlideAleUtils::centerdisp(
    Adapter::FSIStructureWrapper& structure, const Epetra_Comm& comm)
{
  Teuchos::RCP<Core::FE::Discretization> structdis = structure.discretization();

  Teuchos::RCP<Epetra_Vector> idispn = structure.extract_interface_dispn();
  Teuchos::RCP<Epetra_Vector> idisptotal = structure.extract_interface_dispnp();
  Teuchos::RCP<Epetra_Vector> idispstep = structure.extract_interface_dispnp();

  int err = idispstep->Update(-1.0, *idispn, 1.0);
  if (err != 0) FOUR_C_THROW("ERROR");

  const int dim = Global::Problem::instance()->n_dim();
  // get structure and fluid discretizations  and set stated for element evaluation
  const Teuchos::RCP<Epetra_Vector> idisptotalcol =
      Core::LinAlg::create_vector(*structdis->dof_col_map(), true);
  Core::LinAlg::export_to(*idisptotal, *idisptotalcol);
  const Teuchos::RCP<Epetra_Vector> idispstepcol =
      Core::LinAlg::create_vector(*structdis->dof_col_map(), true);
  Core::LinAlg::export_to(*idispstep, *idispstepcol);

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
  std::map<int, Teuchos::RCP<Core::Elements::Element>>::const_iterator elemiter;
  for (elemiter = istructdispeles_.begin(); elemiter != istructdispeles_.end(); ++elemiter)
  {
    Teuchos::RCP<Core::Elements::Element> iele = elemiter->second;
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
  comm.SumAll(&mylengthcirc, &lengthcirc, 1);
  comm.SumAll(mycenterdisp.data(), centerdisp.data(), dim);

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
std::map<int, Core::LinAlg::Matrix<3, 1>> FSI::UTILS::SlideAleUtils::current_struct_pos(
    Teuchos::RCP<Epetra_Vector> reddisp, Core::FE::Discretization& interfacedis,
    std::map<int, double>& maxcoord)
{
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions;
  std::map<int, Teuchos::RCP<Core::Elements::Element>>::const_iterator eleiter;
  std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>>::const_iterator meleiter;

  // map with fully reduced struct element distribution
  for (meleiter = structreduelements_.begin(); meleiter != structreduelements_.end(); meleiter++)
  {
    maxcoord[meleiter->first] = 0.0;
    for (eleiter = meleiter->second.begin(); eleiter != meleiter->second.end(); eleiter++)
    {
      Teuchos::RCP<Core::Elements::Element> tmpele = eleiter->second;

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
        std::vector<double> mydisp(3);
        Core::LinAlg::Matrix<3, 1> currpos;

        Core::FE::extract_my_values(*reddisp, mydisp, lm);

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
void FSI::UTILS::SlideAleUtils::slide_projection(
    Adapter::FSIStructureWrapper& structure, Teuchos::RCP<Core::FE::Discretization> fluiddis,
    Teuchos::RCP<Epetra_Vector> idispale, Teuchos::RCP<Epetra_Vector> iprojdispale,
    Coupling::Adapter::CouplingMortar& coupsf, const Epetra_Comm& comm

)
{
  const int dim = Global::Problem::instance()->n_dim();

  Teuchos::RCP<Epetra_Vector> idispnp = structure.extract_interface_dispnp();

  // Redistribute displacement of structnodes on the interface to all processors.
  Teuchos::RCP<Epetra_Import> interimpo =
      Teuchos::rcp(new Epetra_Import(*structfullnodemap_, *structdofrowmap_));
  Teuchos::RCP<Epetra_Vector> reddisp = Core::LinAlg::create_vector(*structfullnodemap_, true);
  reddisp->Import(*idispnp, *interimpo, Add);

  Core::FE::Discretization& interfacedis = coupsf.interface()->discret();
  std::map<int, double> rotrat;
  // currentpositions of struct nodes for the search tree (always 3 coordinates)
  std::map<int, Core::LinAlg::Matrix<3, 1>> currentpositions =
      current_struct_pos(reddisp, interfacedis, rotrat);

  // calculate structural interface center of gravity
  std::vector<double> centerdisp_v = centerdisp(structure, comm);

  Teuchos::RCP<Epetra_Vector> frotfull = Core::LinAlg::create_vector(*fluiddofrowmap_, true);
  if (aletype_ == Inpar::FSI::ALEprojection_rot_z ||
      aletype_ == Inpar::FSI::ALEprojection_rot_zsphere)
  {
    rotation(coupsf.interface()->discret(), idispale, comm, rotrat, frotfull);
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
      Teuchos::RCP<Core::Geo::SearchTree> searchTree = Teuchos::rcp(new Core::Geo::SearchTree(5));
      const Core::LinAlg::Matrix<3, 2> rootBox =
          Core::Geo::get_xaab_bof_eles(structreduelements_[mnit->first], currentpositions);

      if (dim == 2)
        searchTree->initialize_tree_slide_ale(
            rootBox, structreduelements_[mnit->first], Core::Geo::TreeType(Core::Geo::QUADTREE));
      else if (dim == 3)
        searchTree->initialize_tree_slide_ale(
            rootBox, structreduelements_[mnit->first], Core::Geo::TreeType(Core::Geo::OCTTREE));
      else
        FOUR_C_THROW("wrong dimension");


      Core::Nodes::Node* node = nodeiter->second;
      std::vector<int> lids(dim);
      for (int p = 0; p < dim; p++)
        // lids of gids of node
        lids[p] = (fluiddofrowmap_)->LID((fluiddis->dof(node))[p]);

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
          alenodecurr(p, 0) = node->x()[p] + (*idispale)[(lids[p])] -
                              1.0 * rotrat[mnit->first] * (*frotfull)[(lids[p])];
        }
      }
      else
        FOUR_C_THROW("you should not turn up here!");


      // final displacement of projection
      std::vector<double> finaldxyz(dim);

      // search for near elements next to the query point (ie within a radius of 2x maxmindist)
      std::map<int, std::set<int>> closeeles = searchTree->search_elements_in_radius(
          interfacedis, currentpositions, alenodecurr, maxmindist_, 0);
      // if no close elements could be found, try with a much larger radius and print a warning
      if (closeeles.empty())
      {
        const double enlarge_factor = 100;
        std::cout << "WARNING: no elements found in radius r=" << maxmindist_
                  << ". Will try once with a " << static_cast<int>(enlarge_factor)
                  << "-times bigger radius!" << std::endl;
        closeeles = searchTree->search_elements_in_radius(
            interfacedis, currentpositions, alenodecurr, enlarge_factor * maxmindist_, 0);
        maxmindist_ *= 10.0;

        // if still no element is found, complain about it!
        if (closeeles.empty()) FOUR_C_THROW("No elements in a large radius! Should not happen!");
      }
      // search for the nearest point to project on
      Core::LinAlg::Matrix<3, 1> minDistCoords;
      if (dim == 2)
      {
        Core::Geo::nearest_2d_object_in_node(Teuchos::rcp(&interfacedis, false),
            structreduelements_[mnit->first], currentpositions, closeeles, alenodecurr,
            minDistCoords);
        finaldxyz[0] = minDistCoords(0, 0) - node->x()[0];
        finaldxyz[1] = minDistCoords(1, 0) - node->x()[1];
      }
      else
      {
        Core::Geo::nearest_3d_object_in_node(Teuchos::rcp(&interfacedis, false),
            structreduelements_[mnit->first], currentpositions, closeeles, alenodecurr,
            minDistCoords);
        finaldxyz[0] = minDistCoords(0, 0) - node->x()[0];
        finaldxyz[1] = minDistCoords(1, 0) - node->x()[1];
        finaldxyz[2] = minDistCoords(2, 0) - node->x()[2];
      }

      // store displacement into parallel vector
      int err = iprojdispale->ReplaceMyValues(dim, finaldxyz.data(), lids.data());
      if (err == 1) FOUR_C_THROW("error while replacing values");
    }
  }
}

void FSI::UTILS::SlideAleUtils::redundant_elements(
    Coupling::Adapter::CouplingMortar& coupsf, const Epetra_Comm& comm)
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
    foffset = fluidfullelemap_->MinMyGID();
  }
  else
  {
    fluidfullnodemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->master_row_dofs()));
    fluidfullelemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->master_row_elements()));
    structfullnodemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->slave_row_dofs()));
    structfullelemap_ = Core::LinAlg::allreduce_e_map(*(coupsf.interface()->slave_row_elements()));
    soffset = structfullelemap_->MinMyGID();
    foffset = 0;
  }

  Core::FE::Discretization& interfacedis = coupsf.interface()->discret();

  std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>>::iterator mapit;
  // build redundant version istructslideles_;
  std::map<int, Teuchos::RCP<Core::Elements::Element>>::iterator eit;
  int dim = Global::Problem::instance()->n_dim();

  for (int i = 0; i <= maxid_; ++i)
  {
    std::vector<int> vstruslideleids;  // vector for ele ids
    if (istructslideles_.find(i) != istructslideles_.end())
    {
      for (eit = istructslideles_[i].begin(); eit != istructslideles_[i].end(); eit++)
      {
        if (interfacedis.get_comm().MyPID() == (*eit).second->owner())
          vstruslideleids.push_back(eit->first);
      }
    }
    int globsum = 0;
    int partsum = (vstruslideleids.size());

    comm.SumAll(&partsum, &globsum, 1);
    // map with ele ids
    Epetra_Map mstruslideleids(globsum, vstruslideleids.size(), vstruslideleids.data(), 0, comm);
    // redundant version of it
    Epetra_Map redmstruslideleids(*Core::LinAlg::allreduce_e_map(mstruslideleids));

    for (int eleind = 0; eleind < redmstruslideleids.NumMyElements(); eleind++)
    {
      {
        Core::Elements::Element* tmpele =
            interfacedis.g_element(redmstruslideleids.GID(eleind) + soffset);
        if (dim == 3)
        {
          structreduelements_[i][tmpele->id()] =
              Teuchos::rcp(new Discret::ELEMENTS::StructuralSurface(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0));
        }
        else if (dim == 2)
        {
          structreduelements_[i][tmpele->id()] =
              Teuchos::rcp(new Discret::ELEMENTS::StructuralLine(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0));
        }
      }
    }

    if (ifluidslideles_.find(i) != ifluidslideles_.end())
    {
      for (eit = ifluidslideles_[i].begin(); eit != ifluidslideles_[i].end(); eit++)
      {
        Core::Elements::Element* tmpele = interfacedis.g_element(eit->first + foffset);
        if (dim == 3)
        {
          ifluidslidstructeles_[i][tmpele->id()] =
              Teuchos::rcp(new Discret::ELEMENTS::StructuralSurface(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0));
        }
        else if (dim == 2)
        {
          ifluidslidstructeles_[i][tmpele->id()] =
              Teuchos::rcp(new Discret::ELEMENTS::StructuralLine(tmpele->id(), tmpele->owner(),
                  tmpele->num_node(), tmpele->node_ids(), tmpele->nodes(), &(*tmpele), 0));
        }
      }
    }
  }
}


void FSI::UTILS::SlideAleUtils::rotation(
    Core::FE::Discretization& mtrdis,      ///< fluid discretization
    Teuchos::RCP<Epetra_Vector> idispale,  ///< vector of ALE displacements
    const Epetra_Comm& comm,               ///< communicator
    std::map<int, double>& rotrat,         ///< rotation ratio of tangential displacements
    Teuchos::RCP<Epetra_Vector> rotfull  ///< vector of full displacements in tangential directions
)
{
  Teuchos::RCP<Epetra_Vector> idispstep = Core::LinAlg::create_vector(*fluiddofrowmap_, false);
  idispstep->Update(1.0, *idispale, -1.0, *iprojhist_, 0.0);

  // get structure and fluid discretizations  and set state for element evaluation
  const Teuchos::RCP<Epetra_Vector> idispstepcol =
      Core::LinAlg::create_vector(*mtrdis.dof_col_map(), false);
  Core::LinAlg::export_to(*idispstep, *idispstepcol);
  const Teuchos::RCP<Epetra_Vector> idispnpcol =
      Core::LinAlg::create_vector(*mtrdis.dof_col_map(), false);
  Core::LinAlg::export_to(*idispale, *idispnpcol);

  mtrdis.set_state("displacementnp", idispnpcol);
  mtrdis.set_state("displacementincr", idispstepcol);

  std::map<int, std::map<int, Teuchos::RCP<Core::Elements::Element>>>::iterator melit;
  for (int i = 0; i <= maxid_; ++i)
  {
    // prepare variables for length (2D) or area (3D) of the interface
    double myrotation = 0.0;
    double rotation = 0.0;
    double mylengthcirc = 0.0;
    double lengthcirc = 0.0;
    double maxcoord = 0.0;
    if (ifluidslidstructeles_.find(i) != ifluidslidstructeles_.end()) maxcoord = rotrat[i];

    std::map<int, Teuchos::RCP<Core::Elements::Element>>::const_iterator elemiter;
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

      Teuchos::RCP<Core::Elements::Element> iele = elemiter->second;
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
    comm.SumAll(&mylengthcirc, &lengthcirc, 1);
    comm.SumAll(&myrotation, &rotation, 1);

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

      Teuchos::RCP<Core::Elements::Element> iele = elemiter->second;
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

      Core::LinAlg::assemble(*rotfull, elevector1, lm, lmowner);
    }
  }
  mtrdis.clear_state();

  return;
}

void FSI::UTILS::SlideAleUtils::output_restart(Core::IO::DiscretizationWriter& output)
{
  output.write_vector("projhist", iprojhist_);

  return;
}

void FSI::UTILS::SlideAleUtils::read_restart(Core::IO::DiscretizationReader& reader)
{
  reader.read_vector(iprojhist_, "projhist");
}

FOUR_C_NAMESPACE_CLOSE
