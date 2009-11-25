/*!----------------------------------------------------------------------
\file art_write_gnuplot.cpp
\brief Method to print the arteries in a way that could be displayed by
\gnuplot

<pre>
Maintainer: Mahmoud Ismail
            ismail@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15268
</pre>

*----------------------------------------------------------------------*/

#ifdef CCADISCRET

#include "art_write_gnuplot.H"
#include <sstream>

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

ART::UTILS::ArtWriteGnuplotWrapper::ArtWriteGnuplotWrapper( RCP<DRT::Discretization>  actdis,
                                                            ParameterList & params):
  discret_(actdis)
{


  // -------------------------------------------------------------------
  // Get all gnuplot export conditions
  // -------------------------------------------------------------------
  vector<DRT::Condition*> myConditions;
  discret_->GetCondition("ArtWriteGnuplotCond",myConditions);
  int numofcond = myConditions.size();

  // -------------------------------------------------------------------
  // if gnuplot export conditions exist then create the classes
  // which will export the files
  // -------------------------------------------------------------------
  if(numofcond>0)
  {
    // Start by creating a map of classes that will export the wanted arteries
    int Artery_Number;
    for(unsigned int i=0; i<myConditions.size(); i++)
    {
      // ---------------------------------------------------------------
      // Read in the artery number and the nodes assosiated with the
      // condition 
      // ---------------------------------------------------------------
      Artery_Number = myConditions[i]->GetInt("ArteryNumber");
      const vector<int> * nodes = myConditions[i]->Nodes();

      // ---------------------------------------------------------------
      // Sort all nodes so such that inlet node is the first and outlet
      // node is the last
      // ---------------------------------------------------------------
      
      // step (1) find both inlet and outlet nodes
      DRT::Node* nd;
      DRT::Node* ndi; // ith node
      DRT::Node* ndl; // last node
      //      cout<<"finding nodes"<<endl;
      for(unsigned int n=0; n<nodes->size();n++)
      {
        nd = actdis->gNode((*nodes)[n]);
        if (nd->GetCondition("ArtInOutCond"))
        {
          string TerminalType = *(nd->GetCondition("ArtInOutCond")->Get<string>("terminaltype"));
          if(TerminalType == "inlet")
            ndi = nd;
          else
            ndl = nd;
        }
      }
      //      cout<<"nodes found"<<endl;
      if(ndl == NULL)
        dserror("artery %d has no outlet node!",Artery_Number);
      if(ndi == NULL)
        dserror("artery %d has no inlet node!",Artery_Number);

      //    cout<<"Before defs"<<endl;
      // loop over all nodes
      vector<int> * sorted_nodes = new vector<int>;
      DRT::Element ** Elements = ndi->Elements();
      //    cout<<"ndi Elements extracted"<<endl;
      DRT::Element * Elem_i;
      if (ndi->NumElement()!=1)
        dserror("artery %d must have one element connected to the inlet node!",Artery_Number);
      //    cout<<"H1"<<endl;
      Elem_i = & Elements[0][0];
      //    cout<<"H2"<<endl;

      sorted_nodes->push_back(ndi->Id());
      //    cout<<"After defs"<<endl;
      for(unsigned int n=0; n<nodes->size()-2;n++)
      {
        //      cout<<"LOOPING: "<<n<<endl;
        //      cout<<*ndi<<endl;
        //      cout<<"pushed back"<<endl;
        // find the next node!
        nd = Elem_i->Nodes()[0];
        //      cout<<"this is funny"<<endl;
        //      cout<<"nd * "<<nd;
        //      cout<<"nd "<<*nd<<endl;
        if (Elem_i->Nodes()[0]->Id() != ndi->Id())
        ndi = Elem_i->Nodes()[0];
        else
        ndi = Elem_i->Nodes()[1];
        if (ndi->NumElement()!=2)
          dserror("artery %d must have two elements connected to any internal node!",Artery_Number);
        //      cout<<"node found"<<endl;
        // find the next element
        Elements = ndi->Elements();
        //      cout<<"Elements * "<<Elements<<endl;
        //      cout<<Elements[0][0]<<endl;
        //      cout<<Elements[1][0]<<endl;
        //      cout<<*Elem_i<<endl;
        if (Elements[0][0].Id()!= Elem_i->Id())
        Elem_i = Elements[0];
        else
        Elem_i = Elements[1];
        sorted_nodes->push_back(ndi->Id());        
      }
      
      sorted_nodes->push_back(ndl->Id());

      // ---------------------------------------------------------------
      // Allocate the gnuplot export condition
      // ---------------------------------------------------------------
      RCP<ArtWriteGnuplot> artgnu_c = rcp(new ArtWriteGnuplot(Artery_Number));    


      // ---------------------------------------------------------------
      // Sort the export ondition in a map and check whether the
      // condition exists more than once, which shouldn't be allowed
      // ---------------------------------------------------------------
      bool inserted  = agmap_.insert( make_pair( Artery_Number, artgnu_c ) ).second;
      bool inserted2 = agnode_map_.insert(make_pair(Artery_Number, sorted_nodes)).second;

      if(!inserted || !inserted2)
        dserror("Each artery must have a unique artery number, please correct your input file\n");

      cout<<"----------------------------------------------------------"<<endl;
      cout<<"Artery["<<Artery_Number<<"] has the following sorted nodes"<<endl;
      for(unsigned int n=0; n<sorted_nodes->size();n++)
      {
        cout<<(*sorted_nodes)[n]<<"\t";
      }
      cout<<"----------------------------------------------------------"<<endl;
    }
  }
  //throw;
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

void ART::UTILS::ArtWriteGnuplotWrapper::Write(ParameterList & params)
{

  //----------------------------------------------------------------------
  // Exit if the function accessed by a non-master processor
  //----------------------------------------------------------------------
  if (discret_->Comm().MyPID()==0)
  {
  
    // -------------------------------------------------------------------
    // loop over all conditions and export the arteries values
    // -------------------------------------------------------------------
    map<const int, RCP<class ArtWriteGnuplot> >::iterator mapiter;
    
    // defining a constant that will have the artery number
    int art_num;
    for (mapiter = agmap_.begin(); mapiter != agmap_.end(); mapiter++ )
    {
      art_num = mapiter->first;
      mapiter->second->ArtWriteGnuplot::Write(discret_, params, agnode_map_[art_num]);
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
ART::UTILS::ArtWriteGnuplot:: ArtWriteGnuplot(int ArteryNum):
  ArteryNum_(ArteryNum)
{

  // -------------------------------------------------------------------
  // Create the file with the following name 
  // artery[ArteryNum]_.art
  // -------------------------------------------------------------------
  stringstream out;
  string str, Numb_str;
  char *cstr;
  out << ArteryNum;
  Numb_str = out.str();
  str.clear();
  str = "artery";
  str+= Numb_str;
  str+= "_";
  str+= ".art";
  cstr  = new char [str.size()+1];
  strcpy (cstr, str.c_str());
  fout_ = rcp(new ofstream(cstr));
  delete [] cstr;

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
ART::UTILS::ArtWriteGnuplot::ArtWriteGnuplot()
{

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
void ART::UTILS::ArtWriteGnuplot::Write(RCP<DRT::Discretization>  discret,
                                        ParameterList&            params,
                                        const vector<int>*        nodes)
{

  // defining the Length
  double L = 0.0;
  double dL, time;
  int ElemNum;

  for(unsigned int i =0; i<nodes->size()-1; i++)
  {  

    // get the elements connected to the node
    DRT::Node * nd = discret->lColNode((*nodes)[i]);
    DRT::Element** ele = nd->Elements();

    // get element location vector, dirichlet flags and ownerships
    vector<int> lm;
    RCP<vector<int> > lmowner = rcp(new vector<int>);
    const int* ele_nodes = ele[0][0].NodeIds();

    if(ele_nodes[0] == (*nodes)[i])
       ElemNum = 0;
    else
       ElemNum = 1;

    ele[ElemNum][0].LocationVector(*discret,lm,*lmowner);

    // get node coordinates and number of elements per node
    LINALG::Matrix<3,2> xyze;
    for (int inode= 0; inode<2; inode++)
    {
      const double* x = discret->lColNode((*nodes)[i+inode])->X();
      xyze(0,inode) = x[0];
      xyze(1,inode) = x[1];
      xyze(2,inode) = x[2];
    }
    // calculate Length of the element
    dL = sqrt(  pow(xyze(0,0) - xyze(0,1),2)
              + pow(xyze(1,0) - xyze(1,1),2)
              + pow(xyze(2,0) - xyze(2,1),2));

    // get the degrees of freedom
    RefCountPtr<const Epetra_Vector> qanp  = discret->GetState("qanp");
    vector<double> myqanp(lm.size());
    DRT::UTILS::ExtractMyValues(*qanp,myqanp,lm);

    // get the current simulation time
    time = params.get<double>("total time");

    // export the degrees of freedom
    (*fout_)<<time<<"\t"<<L<<"\t";
    for (unsigned int j =0; j <lm.size()/2; j++)
    {
      (*fout_)<<myqanp[j]<<"\t";
    }
    (*fout_)<<nd->Id()<<endl;
    // Update L
    L+=dL;
    // export the dof of the final node
    if(i==nodes->size()-2)
    {
      (*fout_)<<time<<"\t"<<L<<"\t";
      for (unsigned int j =lm.size()/2; j <lm.size(); j++)
      {
        (*fout_)<<myqanp[j]<<"\t";
      }
      (*fout_)<<nd->Id()<<endl;
    }
  }
  (*fout_)<<endl;
}

#endif //CCADISCRET
