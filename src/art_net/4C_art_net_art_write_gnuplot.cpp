/*----------------------------------------------------------------------*/
/*! \file
\brief Method to print the arteries in a way that could be displayed by
\gnuplot

\level 3


*----------------------------------------------------------------------*/


#include "4C_art_net_art_write_gnuplot.hpp"

#include "4C_fem_condition.hpp"
#include "4C_fem_general_extract_values.hpp"

#include <Teuchos_ParameterList.hpp>

#include <fstream>
#include <sstream>

FOUR_C_NAMESPACE_OPEN

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 08/09|
 |                                                                      |
 |                                                                      |
 |       ------> (direction of the flow)                                |
 |       1                 2                 3                 4        |
 |       +-----------------o-----------------o-----------------+        |
 |       ^        ^                 ^                 ^        ^        |
 |    ___|____    |                 |                 |     ___|____    |
 |   [DPOINT 1]   |                 |                 |    [DPOINT 2]   |
 |             ___|___           ___|___           ___|___              |
 |            [DLINE 1]         [DLINE 1]         [DLINE 1]             |
 |                                                                      |
 | ...................................................................  |
 |                                                                      |
 | The gnuplot format exporter will export the results (DOFs) of each   |
 | artery in a different file.                                          |
 | Each artery is defined as a set of elements that belong to a similar |
 | design line (DLINE)                                                  |
 |                                                                      |
 | Therefore, ArtWriteGnuplotWrapper will check how many arteries are   |
 | there to export and generate the associated condition which will     |
 | export it.                                                           |
 |                                                                      |
 | For now we will consider that each artery must have a ascending      |
 | node numbering in the direction of the flow. This could be improved  |
 | later! ;)                                                            |
 |                                                                      |
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//

Arteries::UTILS::ArtWriteGnuplotWrapper::ArtWriteGnuplotWrapper(
    Teuchos::RCP<Core::FE::Discretization> actdis, Teuchos::ParameterList& params)
    : discret_(actdis)
{
  // -------------------------------------------------------------------
  // Get all gnuplot export conditions
  // -------------------------------------------------------------------
  std::vector<Core::Conditions::Condition*> myConditions;
  discret_->get_condition("ArtWriteGnuplotCond", myConditions);
  int numofcond = myConditions.size();

  // -------------------------------------------------------------------
  // if gnuplot export conditions exist then create the classes
  // which will export the files
  // -------------------------------------------------------------------
  if (numofcond > 0 && discret_->get_comm().MyPID() == 0)
  {
    // Start by creating a map of classes that will export the wanted arteries
    for (unsigned int i = 0; i < myConditions.size(); i++)
    {
      // ---------------------------------------------------------------
      // Read in the artery number and the nodes assosiated with the
      // condition
      // ---------------------------------------------------------------
      const int Artery_Number = myConditions[i]->parameters().get<int>("ArteryNumber");
      const std::vector<int>* nodes = myConditions[i]->get_nodes();

      // ---------------------------------------------------------------
      // Sort all nodes so such that inlet node is the first and outlet
      // node is the last
      // ---------------------------------------------------------------

      // step (1) find both inlet and outlet nodes
      Core::Nodes::Node* ndi = nullptr;  // ith node
      Core::Nodes::Node* ndl = nullptr;  // last node

      for (unsigned int n = 0; n < nodes->size(); n++)
      {
        Core::Nodes::Node* nd = actdis->g_node((*nodes)[n]);
        if (nd->get_condition("ArtInOutCond"))
        {
          std::string TerminalType =
              (nd->get_condition("ArtInOutCond")->parameters().get<std::string>("terminaltype"));
          if (TerminalType == "inlet")
            ndi = nd;
          else
            ndl = nd;
        }
      }

      if (ndl == nullptr) FOUR_C_THROW("artery %d has no outlet node!", Artery_Number);
      if (ndi == nullptr) FOUR_C_THROW("artery %d has no inlet node!", Artery_Number);


      // loop over all nodes
      std::vector<int>* sorted_nodes = new std::vector<int>;
      Core::Elements::Element** Elements = ndi->elements();

      Core::Elements::Element* Elem_i;
      if (ndi->num_element() != 1)
        FOUR_C_THROW("artery %d must have one element connected to the inlet node!", Artery_Number);

      Elem_i = Elements[0];

      sorted_nodes->push_back(ndi->id());

      for (unsigned int n = 0; n < nodes->size() - 2; n++)
      {
        // find the next node!
        if (Elem_i->nodes()[0]->id() != ndi->id())
          ndi = Elem_i->nodes()[0];
        else
          ndi = Elem_i->nodes()[1];
        if (ndi->num_element() != 2)
          FOUR_C_THROW(
              "artery %d must have two elements connected to any internal node!", Artery_Number);

        // find the next element
        Elements = ndi->elements();

        if (Elements[0][0].id() != Elem_i->id())
          Elem_i = Elements[0];
        else
          Elem_i = Elements[1];
        sorted_nodes->push_back(ndi->id());
      }

      sorted_nodes->push_back(ndl->id());

      // ---------------------------------------------------------------
      // Allocate the gnuplot export condition
      // ---------------------------------------------------------------
      Teuchos::RCP<ArtWriteGnuplot> artgnu_c = Teuchos::make_rcp<ArtWriteGnuplot>(Artery_Number);


      // ---------------------------------------------------------------
      // Sort the export ondition in a map and check whether the
      // condition exists more than once, which shouldn't be allowed
      // ---------------------------------------------------------------
      bool inserted = agmap_.insert(std::make_pair(Artery_Number, artgnu_c)).second;
      bool inserted2 = agnode_map_.insert(std::make_pair(Artery_Number, sorted_nodes)).second;

      if (!inserted || !inserted2)
        FOUR_C_THROW(
            "Each artery must have a unique artery number, please correct your input file\n");

      std::cout << "----------------------------------------------------------" << std::endl;
      std::cout << "Artery[" << Artery_Number << "] has the following sorted nodes" << std::endl;
      for (unsigned int n = 0; n < sorted_nodes->size(); n++)
      {
        std::cout << (*sorted_nodes)[n] << "\t";
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------------------" << std::endl;
    }
  }
  // throw;
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Write (public)                                          ismail 08/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//

void Arteries::UTILS::ArtWriteGnuplotWrapper::write(Teuchos::ParameterList& params)
{
  //----------------------------------------------------------------------
  // Exit if the function accessed by a non-master processor
  //----------------------------------------------------------------------
  if (discret_->get_comm().MyPID() == 0)
  {
    // -------------------------------------------------------------------
    // loop over all conditions and export the arteries values
    // -------------------------------------------------------------------
    std::map<const int, Teuchos::RCP<class ArtWriteGnuplot>>::iterator mapiter;

    // defining a constant that will have the artery number
    int art_num;
    for (mapiter = agmap_.begin(); mapiter != agmap_.end(); mapiter++)
    {
      art_num = mapiter->first;
      mapiter->second->ArtWriteGnuplot::write(discret_, params, agnode_map_[art_num]);
    }
  }
}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 08/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
Arteries::UTILS::ArtWriteGnuplot::ArtWriteGnuplot(int ArteryNum) : artery_num_(ArteryNum)
{
  // -------------------------------------------------------------------
  // Create the file with the following name
  // artery[ArteryNum]_.art
  // -------------------------------------------------------------------
  std::stringstream out;
  std::string str, Numb_str;
  char* cstr;
  out << ArteryNum;
  Numb_str = out.str();
  str.clear();
  str = "xxx";
  str += Numb_str;
  str += "_";
  str += ".art";
  cstr = new char[str.size() + 1];
  strcpy(cstr, str.c_str());
  fout_ = Teuchos::make_rcp<std::ofstream>(cstr);
  delete[] cstr;

  // Avoid warning on unused variable
  (void)artery_num_;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 08/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
Arteries::UTILS::ArtWriteGnuplot::ArtWriteGnuplot() {}


//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
/*----------------------------------------------------------------------*
 |  Constructor (public)                                    ismail 08/09|
 *----------------------------------------------------------------------*/
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
//<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>//
void Arteries::UTILS::ArtWriteGnuplot::write(Teuchos::RCP<Core::FE::Discretization> discret,
    Teuchos::ParameterList& params, const std::vector<int>* nodes)
{
  // defining the Length
  double L = 0.0;
  double dL, time;
  int ElemNum;

  for (unsigned int i = 0; i < nodes->size() - 1; i++)
  {
    // get the elements connected to the node
    if (!discret->have_global_node((*nodes)[i]))
    {
      int proc = discret->get_comm().MyPID();
      FOUR_C_THROW("Global Node (%d) doesn't exist on processor (%d)\n", (*nodes)[i], proc);
      exit(1);
    }

    //    Core::Nodes::Node * nd = discret->lColNode((*nodes)[i]);
    Core::Nodes::Node* nd = discret->g_node((*nodes)[i]);
    Core::Elements::Element** ele = nd->elements();

    // get element location vector, dirichlet flags and ownerships
    std::vector<int> lm;
    std::vector<int> lmstride;
    Teuchos::RCP<std::vector<int>> lmowner = Teuchos::make_rcp<std::vector<int>>();
    const int* ele_nodes = ele[0][0].node_ids();

    if (ele_nodes[0] == (*nodes)[i])
      ElemNum = 0;
    else
      ElemNum = 1;

    ele[ElemNum][0].location_vector(*discret, lm, *lmowner, lmstride);

    // get node coordinates and number of elements per node
    Core::LinAlg::Matrix<3, 2> xyze;
    for (int inode = 0; inode < 2; inode++)
    {
      const auto& x = discret->g_node((*nodes)[i + inode])->x();
      xyze(0, inode) = x[0];
      xyze(1, inode) = x[1];
      xyze(2, inode) = x[2];
    }
    // calculate Length of the element
    dL = sqrt(pow(xyze(0, 0) - xyze(0, 1), 2) + pow(xyze(1, 0) - xyze(1, 1), 2) +
              pow(xyze(2, 0) - xyze(2, 1), 2));

    // get the degrees of freedom
    Teuchos::RCP<const Core::LinAlg::Vector<double>> qanp = discret->get_state("qanp");
    std::vector<double> myqanp(lm.size());
    Core::FE::extract_my_values(*qanp, myqanp, lm);

    // get the current simulation time
    time = params.get<double>("total time");

    // export the degrees of freedom
    (*fout_) << time << "\t" << L << "\t";
    for (unsigned int j = 0; j < lm.size() / 2; j++)
    {
      (*fout_) << myqanp[j] << "\t";
    }
    (*fout_) << nd->id() << std::endl;
    // Update L
    L += dL;
    // export the dof of the final node
    if (i == nodes->size() - 2)
    {
      (*fout_) << time << "\t" << L << "\t";
      for (unsigned int j = lm.size() / 2; j < lm.size(); j++)
      {
        (*fout_) << myqanp[j] << "\t";
      }
      (*fout_) << nd->id() << std::endl;
    }
  }
  (*fout_) << std::endl;
}

FOUR_C_NAMESPACE_CLOSE
