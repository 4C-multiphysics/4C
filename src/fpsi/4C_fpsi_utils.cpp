// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fpsi_utils.hpp"

#include "4C_ale_utils_clonestrategy.hpp"
#include "4C_fem_condition_selector.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_fpsi_monolithic_plain.hpp"
#include "4C_fsi_utils.hpp"
#include "4C_global_data.hpp"
#include "4C_inpar_fpsi.hpp"
#include "4C_poroelast_scatra_utils_clonestrategy.hpp"
#include "4C_poroelast_scatra_utils_setup.hpp"
#include "4C_poroelast_utils_clonestrategy.hpp"
#include "4C_poroelast_utils_setup.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------/
| Instance method for singleton class                            rauch  |
/----------------------------------------------------------------------*/
FPSI::InterfaceUtils* FPSI::InterfaceUtils::instance()
{
  static auto singleton_owner = Core::Utils::make_singleton_owner(
      []() { return std::unique_ptr<FPSI::InterfaceUtils>(new FPSI::InterfaceUtils); });

  return singleton_owner.instance(Core::Utils::SingletonAction::create);
}

/*----------------------------------------------------------------------/
| Setup discretization                                           rauch  |
/----------------------------------------------------------------------*/
std::shared_ptr<FPSI::FpsiBase> FPSI::InterfaceUtils::setup_discretizations(MPI_Comm comm,
    const Teuchos::ParameterList& fpsidynparams, const Teuchos::ParameterList& poroelastdynparams)
{
  Global::Problem* problem = Global::Problem::instance();

  fluid_poro_fluid_interface_map_ = std::make_shared<std::map<int, int>>();
  poro_fluid_fluid_interface_map_ = std::make_shared<std::map<int, int>>();

  // 1.-Initialization.
  std::shared_ptr<Core::FE::Discretization> structdis = problem->get_dis("structure");
  if (structdis == nullptr) FOUR_C_THROW(" !!! structdis empty !!! Awwww MAAAAN !!!");
  std::shared_ptr<Core::FE::Discretization> porofluiddis = problem->get_dis("porofluid");
  if (porofluiddis == nullptr) FOUR_C_THROW(" !!! porofluiddis empty !!! Awwww MAAAAN !!!");
  std::shared_ptr<Core::FE::Discretization> fluiddis = problem->get_dis("fluid");
  if (fluiddis == nullptr) FOUR_C_THROW(" !!! fluiddis empty !!! Awwww MAAAAN !!!");
  std::shared_ptr<Core::FE::Discretization> aledis = problem->get_dis("ale");
  if (aledis == nullptr) FOUR_C_THROW(" !!! aledis empty !!! Awwww MAAAAN !!!");

  /*
   Ok Dude, listen carefully for the following is very important. The stuff below is
   actually quite basic but the emphasis is put on the right order of doing it. In
   general the filling order of the discretizations determines their
   individual dof numbers and thus the mapping of the local indices to the global ones.
   The rule is simple: First filled first mapped. So have a look at what we do below.
   First we fill the structural discretization which means that its local indices equal
   its global indices beginning from 0 and ending with the number of structural dofs
   depending on the specific problem size under consideration. Now the porofluiddis is
   filled. Note that this discretization doesn't exist at that moment because in our
   input file we only have a structural discretization for the domain of the porous
   medium ("structure") and a fluid discretization for the domain containing the free
   fluid ("fluid"). To bring the porofluiddis to life we clone it from the structdis.
   Since it was filled right after the structdis the dof numbering of the porous structure
   and the fluid contained by it are adjacent. Now we fill the fluiddis (which exists) and
   the aledis (which is nonexistent just like the porofluiddis before the cloning procedure).
   We expect the fluiddis to have smaller dof numbers than the ale discretization. Finally
   cloning the aledis from the fluiddis makes this dream come true. Building the monolithic
   systemmatrix based on the DofMaps built before, results in the following order of the
   diagonal blocks (from northwest to southeast): porostructure - porofluid - free fluid - ale.

   Thats nice....

   This would be corrupted however if we were doing the following:

   structdis    -> fill_complete()
   porofluiddis -> fill_complete()
   fluiddis     -> fill_complete()
   aledis       -> fill_complete()

   Clone(structdis,porofluiddis)
   Clone(fluiddis,aledis)

   Thats not so nice because then the fluiddis would have the same DofMap like the porofluiddis.
   The reason for this awkward mess is that when filling the fluiddis after the porofluiddis
   the DofNumbering of the structuredis and the fluiddis will be adjacent for there exists no
   porofluiddis to append the fluiddofmap to. The resulting monolithic blockmatrix will assemble
   into the same positions like the porofluiddis. Consequently not even the matrix dimension
   will be correct. That's because when the porofluiddis is cloned from the structdis in the end the
   DofMap of the porofluiddis will also be adjacent to the structdis DofMap.

   You get it??

   Thats why it's important to immediately clone the nonexistent discretizations after filling
   them.

   */
  // setup of the discretizations, including clone strategy

  // choose cloning strategy depending on poroelast or scatra poroelast problem type
  if (problem->get_problem_type() == Core::ProblemType::fps3i)
  {
    PoroElast::Utils::setup_poro<PoroElastScaTra::Utils::PoroelastCloneStrategyforScatraElements>();
  }
  else
  {
    PoroElast::Utils::setup_poro<PoroElast::Utils::PoroelastCloneStrategy>();
  }


  fluiddis->fill_complete();
  aledis->fill_complete();

  // 3.- Create ALE elements if the ale discretization is empty
  if (aledis->num_global_nodes() == 0)  // ALE discretization still empty
  {
    Core::FE::clone_discretization<ALE::Utils::AleCloneStrategy>(
        *fluiddis, *aledis, Global::Problem::instance()->cloning_material_map());
    aledis->fill_complete();
    // setup material in every ALE element
    Teuchos::ParameterList params;
    params.set<std::string>("action", "setup_material");
    aledis->evaluate(params);
  }
  else  // ALE discretization already filled
  {
    if (!FSI::Utils::fluid_ale_nodes_disjoint(*fluiddis, *aledis))
      FOUR_C_THROW(
          "Fluid and ALE nodes have the same node numbers. "
          "This it not allowed since it causes problems with Dirichlet BCs. "
          "Use the ALE cloning functionality or ensure non-overlapping node numbering!");
  }

  setup_interface_map(comm, *structdis, porofluiddis, fluiddis, *aledis);

  // 4.- get coupling algorithm
  std::shared_ptr<FPSI::FpsiBase> fpsi_algo = nullptr;
  const auto coupling = Teuchos::getIntegralValue<FpsiCouplingType>(fpsidynparams, "COUPALGO");
  switch (coupling)
  {
    case fpsi_monolithic_plain:
    {
      fpsi_algo = std::make_shared<FPSI::MonolithicPlain>(comm, fpsidynparams, poroelastdynparams);
      break;
    }  // case monolithic
    case partitioned:
    {
      FOUR_C_THROW(
          "Partitioned solution scheme not implemented for FPSI, yet. "
          "Make sure that the parameter COUPALGO is set to 'fpsi_monolithic_plain', "
          "and the parameter PARITIONED is set to 'monolithic'. ");
    }
  }

  return fpsi_algo;
}  // SetupDiscretization()


/*---------------------------------------------------------------------------/
| Setup Local Interface Facing Element Map (for parallel distr.)      rauch  |
/---------------------------------------------------------------------------*/
void FPSI::InterfaceUtils::setup_local_interface_facing_element_map(
    Core::FE::Discretization& masterdis, const Core::FE::Discretization& slavedis,
    const std::string& condname, std::map<int, int>& interfacefacingelementmap)
{
  Global::Problem* problem = Global::Problem::instance();
  MPI_Comm mastercomm = problem->get_dis(masterdis.name())->get_comm();

  bool condition_exists = true;

  if (!slavedis.has_condition(condname))
  {
    condition_exists = false;
    std::cout << "WARNING: Condition <" << condname << "> does not exist on discretisation <"
              << slavedis.name() << ">! (OK if no " << condname << "-Interface is wanted)"
              << std::endl;
  }

  if (!masterdis.has_condition(condname))
  {
    condition_exists = false;
    std::cout << "WARNING: Condition <" << condname << "> does not exist on discretisation <"
              << masterdis.name() << ">! (OK if no " << condname << "-Interface is wanted)"
              << std::endl;
  }

  if (!condition_exists) return;

  std::vector<const Core::Conditions::Condition*> conds;
  slavedis.get_condition(condname, conds);
  FOUR_C_ASSERT_ALWAYS(conds.size() == 1, "Expected exactly one condition of type '{}'", condname);
  const std::map<int, std::shared_ptr<Core::Elements::Element>>& slavegeom =
      conds.front()->geometry();

  masterdis.get_condition(condname, conds);
  FOUR_C_ASSERT_ALWAYS(conds.size() == 1, "Expected exactly one condition of type '{}'", condname);
  const std::map<int, std::shared_ptr<Core::Elements::Element>>& mastergeom =
      conds.front()->geometry();

  std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator curr;
  std::multimap<int, double> slaveinterfaceelementidentificationmap;

  ///////////////////////////////////////////////////////////////////////////////////
  ////////////                                                           ////////////
  ////////////                         SLAVE                             ////////////
  ////////////                       Interface                           ////////////
  ///////////////////////////////////////////////////////////////////////////////////
  int slavegeomsize = slavegeom.size();
  int globalslavegeomsize;
  globalslavegeomsize = Core::Communication::sum_all(slavegeomsize, mastercomm);

  // do for every slave interface element
  for (curr = slavegeom.begin(); curr != slavegeom.end(); ++curr)
  {
    // std::cout<<"Proc "<<masterCore::Communication::my_mpi_rank(comm)<<" owns slavegeom
    // "<<curr->first<<endl;

    std::vector<double> slaveloc;
    slaveloc.assign(3, 0.0);

    //       std::cout<<"Current Slave Interface Element ID: "<<curr->second->Id()<<endl;

    if (slavedis.have_dofs() == false)
    {
      FOUR_C_THROW("No degrees of freedom have been assigned to discretization");
    }

    // do for every interface slave node
    for (int nodenum = 0; nodenum < curr->second->num_node(); nodenum++)
    {
      const Core::Nodes::Node* const* currslavenode = curr->second->nodes();

      std::vector<double> temploc(3);
      auto x = currslavenode[nodenum]->x();
      temploc[0] = x[0];
      temploc[1] = x[1];
      if (x.size() > 2) temploc[2] = x[2];
      for (int dim = 0; dim < 3; dim++)
      {
        slaveloc[dim] = slaveloc[dim] + temploc[dim];
      }
    }  // for every slave node

    slaveinterfaceelementidentificationmap.insert(
        std::pair<int, double>(curr->second->id(), slaveloc[0]));
    slaveinterfaceelementidentificationmap.insert(
        std::pair<int, double>(curr->second->id(), slaveloc[1]));
    slaveinterfaceelementidentificationmap.insert(
        std::pair<int, double>(curr->second->id(), slaveloc[2]));

  }  // do for every global slave element

  Core::Communication::barrier(mastercomm);  // wait for procs

  //  // printout
  //  for(int proc = 0; proc < (mastercomm.NumProc()+1); proc++)
  //  {
  //    if(masterCore::Communication::my_mpi_rank(comm) == proc)
  //    {
  //      std::cout<<"\n slaveinterfaceelementidentificationmap on proc "<<proc<<" :\n"<<endl;
  //      for (curr=slavegeom.begin(); curr!=slavegeom.end(); ++curr)
  //      {
  //        std::pair <std::multimap<int,double>::iterator, std::multimap<int,double>::iterator>
  //        range; range = slaveinterfaceelementidentificationmap.equal_range(curr->second->Id());
  //          std::cout<<curr->second->Id()<<" :";
  //          for (std::multimap<int,double>::iterator it=range.first; it!=range.second; ++it)
  //          {
  //            std::cout << ' ' << it->second;
  //          }
  //          std::cout<<"\n"<<endl;
  //      }
  //    }
  //    else
  //      Core::Communication::barrier(mastercomm);
  //  }

  //  int count=0;
  //  for (curr=slavegeom.begin(); curr!=slavegeom.end(); ++curr)
  //  {
  //    count++;
  //  }
  //  std::cout<<"count = "<< count << " "<<"on
  //  proc"<<masterCore::Communication::my_mpi_rank(comm)<<endl;
  ///////////////////////////////////////////////////////////////////////////////////
  ////////////                                                           ////////////
  ////////////                         MASTER                            ////////////
  ////////////                          Bulk                             ////////////
  ///////////////////////////////////////////////////////////////////////////////////

  // loop over procs
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(mastercomm); proc++)
  {
    curr = mastergeom.begin();

    int parenteleid = -1;
    int parenteleowner = -1;
    int match = 0;
    int mastergeomsize = mastergeom.size();
    std::vector<int> sizelist(Core::Communication::num_mpi_ranks(
        mastercomm));  // how many master interface elements has each processor
    Core::Communication::gather_all(&mastergeomsize, sizelist.data(), 1, mastercomm);
    Core::Communication::barrier(mastercomm);  // wait for procs

    bool done;
    if (sizelist[proc])
      done = false;
    else
      done = true;  // no master interface elements on proc

    int counter = 0;
    while (not done)  // do until every master element on proc has been matched
    {
      match = 0;
      std::vector<double> masterloc;
      masterloc.assign(3, 0.0);

      if (masterdis.have_dofs() == false)
      {
        FOUR_C_THROW("No degrees of freedom have been assigned to discretization, DaFUQ?!?!?");
      }

      // do for every master node
      if (proc == Core::Communication::my_mpi_rank(mastercomm))
      {
        const int numnode = curr->second->num_node();

        for (int nodenum = 0; nodenum < numnode; nodenum++)
        {
          const Core::Nodes::Node* const* currmasternode = curr->second->nodes();

          std::vector<double> temploc(3);
          auto x = currmasternode[nodenum]->x();
          temploc[0] = x[0];
          temploc[1] = x[1];
          if (x.size() > 2) temploc[2] = x[2];
          for (int dim = 0; dim < 3; dim++)
          {
            masterloc[dim] = masterloc[dim] + temploc[dim];
          }

        }  // for every master node

        std::shared_ptr<Core::Elements::FaceElement> bele =
            std::dynamic_pointer_cast<Core::Elements::FaceElement>(curr->second);
        parenteleid = bele->parent_element()->id();
        if (parenteleid == -1) FOUR_C_THROW("Couldn't get master parent element Id() ...");
        parenteleowner = bele->parent_element()->owner();
        if (parenteleowner == -1) FOUR_C_THROW("Couldn't get master parent element Owner() ...");
      }

      Core::Communication::broadcast(&parenteleid, 1, proc, mastercomm);
      Core::Communication::broadcast(&parenteleowner, 1, proc, mastercomm);
      Core::Communication::broadcast(masterloc.data(), masterloc.size(), proc, mastercomm);

      Core::Communication::barrier(mastercomm);
      // match current master element
      // compare position to every element on interface slave side, every processor compares
      // masterloc of current master element of processor[proc]
      std::shared_ptr<Core::Elements::Element> matchcurr = nullptr;

      for (std::map<int, std::shared_ptr<Core::Elements::Element>>::const_iterator scurr =
               slavegeom.begin();
          scurr != slavegeom.end(); ++scurr)
      {
        std::pair<std::multimap<int, double>::iterator, std::multimap<int, double>::iterator> range;
        range = slaveinterfaceelementidentificationmap.equal_range(scurr->second->id());

        int dim = 0;
        match = 0;
        for (std::multimap<int, double>::iterator it = range.first; it != range.second; ++it)
        {
          double slaveloccomponent = it->second;
          double masterloccomponent = masterloc[dim];
          if (abs(slaveloccomponent - masterloccomponent) < 10e-3)
          {
            match += 1;
          }
          else
          {
            match += 0;
          }
          dim++;
        }

        if (match == 3)
        {
          matchcurr = scurr->second;
          break;
        }
      }  // end matching

      Core::Communication::barrier(mastercomm);

      for (int p = 0; p < Core::Communication::num_mpi_ranks(mastercomm); ++p)
      {
        Core::Communication::barrier(mastercomm);
        if (Core::Communication::my_mpi_rank(mastercomm) == p)
        {
          if (matchcurr != nullptr)
          {
            // slave interface element is unique => key
            interfacefacingelementmap.insert(std::pair<int, int>(matchcurr->id(), parenteleid));
          }
        }
      }

      Core::Communication::barrier(mastercomm);

      if (counter < (sizelist[Core::Communication::my_mpi_rank(mastercomm)] - 1))
        curr++;  // increment iterator

      if (counter != sizelist[proc]) counter++;
      if (counter == sizelist[proc])
        done = true;  // true on every proc
      else if (counter > sizelist[proc])
        FOUR_C_THROW("tried to match more master elements as are on proc ... ");

    }  // while iterating master elements on proc
    Core::Communication::barrier(mastercomm);
  }  // loop over procs

  int mymatchedelements = interfacefacingelementmap.size();
  int globalmatchedelements = 0;

  globalmatchedelements = Core::Communication::sum_all(mymatchedelements, mastercomm);

  if (Core::Communication::my_mpi_rank(mastercomm) == 0)
    std::cout << "Could match " << globalmatchedelements << " " << slavedis.name()
              << " interface elements to " << masterdis.name() << " bulk elements." << std::endl;

  Core::Communication::barrier(mastercomm);  // wait for procs

  if (abs(globalslavegeomsize - globalmatchedelements) < 1e-6 and
      Core::Communication::my_mpi_rank(mastercomm) == 0)
  {
    std::cout << "Setting up local interfacefacingelementmaps was successful. \n" << std::endl;
  }
  else if (abs(globalslavegeomsize - globalmatchedelements) > 1e-3 and
           Core::Communication::my_mpi_rank(mastercomm) == 0)
  {
    FOUR_C_THROW("ERROR: globalslavegeomsize != globalmatchedelements ({},{})", globalslavegeomsize,
        globalmatchedelements);
  }

  return;
}

/*---------------------------------------------------------------------------/
| Redistribute Interface (for parallel distr.)                        rauch  |
/---------------------------------------------------------------------------*/
void FPSI::InterfaceUtils::redistribute_interface(Core::FE::Discretization& masterdis,
    const std::string& condname, std::map<int, int>& interfacefacingelementmap)
{
  int printid = -1;

  std::map<int, int>::iterator mapcurr;
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator slaveelecurr;
  std::map<int, std::shared_ptr<Core::Elements::Element>>::iterator masterelecurr;
  Core::Elements::Element* masterele = nullptr;

  Global::Problem* problem = Global::Problem::instance();
  MPI_Comm comm = problem->get_dis(masterdis.name())->get_comm();

  int mymapsize = interfacefacingelementmap.size();
  int globalmapsize;
  std::vector<int> mapsizearray(Core::Communication::num_mpi_ranks(comm));
  Core::Communication::gather_all(&mymapsize, mapsizearray.data(), 1, comm);
  globalmapsize = Core::Communication::sum_all(mymapsize, comm);

  int counter = 0;
  for (int proc = 0; proc < Core::Communication::num_mpi_ranks(comm); ++proc)
  {
    mapcurr = interfacefacingelementmap.begin();

    int done = 0;

    if (mapsizearray[proc] > 0)
      done = 0;
    else if (mapsizearray[proc] == 0)
      done = 1;
    else
      FOUR_C_THROW("weird size of processor local interfacefacingelementmap ... ");

    int HasMasterEle = 0;

    while (done == 0)
    {
      int slaveeleid = -1;
      int mastereleowner = -1;
      int mastereleid = -1;
      std::vector<int> slaveeleowners(Core::Communication::num_mpi_ranks(comm));
      std::vector<int> mastereleowners(Core::Communication::num_mpi_ranks(comm));

      // get master id
      if (Core::Communication::my_mpi_rank(comm) == proc)
      {
        mastereleid = mapcurr->second;
        slaveeleid = mapcurr->first;
      }
      Core::Communication::broadcast(&mastereleid, 1, proc, comm);
      Core::Communication::broadcast(&slaveeleid, 1, proc, comm);

      if (masterdis.have_global_element(mastereleid))
      {
        masterele = masterdis.g_element(mastereleid);
        mastereleowner = masterele->owner();

        if (masterele->owner() != Core::Communication::my_mpi_rank(comm))
        {
          masterele = nullptr;
          mastereleowner = -1;
        }
      }  // only the owner of the masterele has a pointer != nullptr and mastereleowner != -1

      Core::Communication::gather_all(&mastereleowner, mastereleowners.data(), 1, comm);

      for (int i = 0; i < Core::Communication::num_mpi_ranks(comm); i++)
      {
        if (mastereleowners[i] != -1) mastereleowner = i;
      }  // now every processor knows the mastereleowner

      std::vector<int> procHasMasterEle(Core::Communication::num_mpi_ranks(comm));
      HasMasterEle = masterdis.have_global_element(mastereleid);
      Core::Communication::gather_all(&HasMasterEle, procHasMasterEle.data(), 1, comm);

      // ghost parent master element on master discretization of proc owning the matching slave
      // interface element
      const Core::LinAlg::Map colcopy = *(masterdis.element_col_map());
      int myglobalelementsize = colcopy.num_my_elements();
      std::vector<int> myglobalelements(myglobalelementsize);
      colcopy.my_global_elements(myglobalelements.data());

      if (Core::Communication::my_mpi_rank(comm) == proc and
          mastereleowner != proc)  // ghost master ele on owner of slave ele, but only if this proc
                                   // doesn't already own the masterele
      {
        if (colcopy.lid(mastereleid) == -1)  // if element not already in ElementColMap()
        {
          myglobalelements.push_back(mastereleid);
          myglobalelementsize = myglobalelementsize + 1;
        }
      }

      int globalsize;
      globalsize = Core::Communication::sum_all(myglobalelementsize, comm);
      Core::LinAlg::Map newelecolmap(
          globalsize, myglobalelementsize, myglobalelements.data(), 0, comm);

      if (mastereleid == printid)
      {
        std::cout << "mastereleowner" << mastereleowner << std::endl;
        std::cout << "slaveeleid" << slaveeleid << std::endl;
        std::cout << "counter" << counter << std::endl;
      }

      // Do the Redistribution
      int before = 0;
      int after = 0;
      if (procHasMasterEle[proc] == 0)
      {
        if (Core::Communication::my_mpi_rank(comm) == proc)
        {
          // std::cout<<counter<<" --Before: Have GID "<<mastereleid<<" =
          // "<<masterdis->HaveGlobalElement(mastereleid)<<" on proc "<<slaveeleowner<<endl;
          before = masterdis.have_global_element(mastereleid);
        }
        Core::Communication::barrier(comm);
        masterdis.extended_ghosting(newelecolmap, true, false, true, true);
        if (Core::Communication::my_mpi_rank(comm) == proc)
        {
          // std::cout<<counter<<" --After: Have GID "<<mastereleid<<" =
          // "<<masterdis->HaveGlobalElement(mastereleid)<<" on proc "<<slaveeleowner<<endl;
          after = masterdis.have_global_element(mastereleid);
          if (after == 0 and before == 0)
            FOUR_C_THROW("Element with gid={} has not been redistributed ! ", mastereleid);
        }
      }

      if (Core::Communication::my_mpi_rank(comm) == proc and
          mapcurr != interfacefacingelementmap.end())
        ++mapcurr;
      if ((Core::Communication::my_mpi_rank(comm) == proc and
              mapcurr == interfacefacingelementmap.end()))
        done = 1;

      if (counter == globalmapsize) done = 1;
      Core::Communication::broadcast(&done, 1, proc, comm);

      if (done == 0) counter++;

    }  // for all elements of interfacefacingelementmap
    Core::Communication::barrier(comm);
  }  // for all procs

  if (Core::Communication::my_mpi_rank(comm) == 0)
    std::cout << "\n EXTENDEDGHOSTING: checked " << counter
              << " elements in interfacefacingelementmap ... \n"
              << std::endl;
  return;
}

/*---------------------------------------------------------------------------/
| Setup Interface Map (for parallel distr.)                           rauch  |
/---------------------------------------------------------------------------*/
void FPSI::InterfaceUtils::setup_interface_map(MPI_Comm comm, Core::FE::Discretization& structdis,
    std::shared_ptr<Core::FE::Discretization> porofluiddis,
    std::shared_ptr<Core::FE::Discretization> fluiddis, Core::FE::Discretization& aledis)
{
  poro_fluid_fluid_interface_map_ = std::make_shared<std::map<int, int>>();
  fluid_poro_fluid_interface_map_ = std::make_shared<std::map<int, int>>();

  setup_local_interface_facing_element_map(
      *fluiddis, *porofluiddis, "fpsi_coupling", *poro_fluid_fluid_interface_map_);
  setup_local_interface_facing_element_map(
      *porofluiddis, *fluiddis, "fpsi_coupling", *fluid_poro_fluid_interface_map_);

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Utils::MapExtractor::setup(
    const Core::FE::Discretization& dis, bool withpressure, bool overlapping)
{
  const int ndim = Global::Problem::instance()->n_dim();
  Core::Conditions::setup_extractor(dis, *this,
      {
          Core::Conditions::Selector("FSICoupling", 0, ndim + withpressure),
          Core::Conditions::Selector("fpsi_coupling", 0, ndim + withpressure),
      },
      overlapping);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void FPSI::Utils::MapExtractor::setup(std::shared_ptr<const Core::LinAlg::Map>& additionalothermap,
    const FPSI::Utils::MapExtractor& extractor)
{
  // build the new othermap
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> othermaps;
  othermaps.push_back(additionalothermap);
  othermaps.push_back(extractor.other_map());

  if (Core::LinAlg::MultiMapExtractor::intersect_maps(othermaps)->num_global_elements() > 0)
    FOUR_C_THROW("Failed to add dofmap of foreign discretization to other_map. Detected overlap.");

  std::shared_ptr<const Core::LinAlg::Map> mergedothermap =
      Core::LinAlg::MultiMapExtractor::merge_maps(othermaps);

  // the vector of maps for the new map extractor consists of othermap at position 0
  // followed by the maps of conditioned DOF
  std::vector<std::shared_ptr<const Core::LinAlg::Map>> maps;
  // append the merged other map at first position
  maps.push_back(mergedothermap);

  // append the condition maps subsequently
  for (int i = 1; i < extractor.num_maps(); ++i) maps.push_back(extractor.map(i));

  // merge
  std::shared_ptr<const Core::LinAlg::Map> fullmap =
      Core::LinAlg::MultiMapExtractor::merge_maps(maps);

  Core::LinAlg::MultiMapExtractor::setup(*fullmap, maps);
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
std::shared_ptr<std::set<int>> FPSI::Utils::MapExtractor::conditioned_element_map(
    const Core::FE::Discretization& dis) const
{
  std::shared_ptr<std::set<int>> condelements =
      Core::Conditions::conditioned_element_map(dis, "FSICoupling");
  std::shared_ptr<std::set<int>> condelements2 =
      Core::Conditions::conditioned_element_map(dis, "fpsi_coupling");
  std::copy(condelements2->begin(), condelements2->end(),
      std::inserter(*condelements, condelements->begin()));
  return condelements;
}

FOUR_C_NAMESPACE_CLOSE
