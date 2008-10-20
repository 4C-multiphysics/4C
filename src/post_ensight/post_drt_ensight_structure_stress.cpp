/*!
  \file post_drt_ensight_structure_stress.cpp

  \brief postprocessing of structural stresses

  <pre>
  Maintainer: Lena Wiechert
              wiechert@lnm.mw.tum.de
              http://www.lnm.mw.tum.de/Members/wiechert
              089-289-15303
  </pre>

*/

#ifdef CCADISCRET

#include "post_drt_ensight_writer.H"
#include <string>
#include "post_drt_ensight_single_field_writers.H"
#include "../drt_lib/drt_utils.H"

using namespace std;


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::PostStress(const string groupname, const string stresstype)
{
  PostResult result = PostResult(field_);
  result.next_result();

  if (!map_has_map(result.group(), groupname.c_str()))
    return;

  //--------------------------------------------------------------------
  // calculation and output of nodal stresses in xyz-reference frame
  //--------------------------------------------------------------------

  if (stresstype == "ndxyz")
  {
    WriteNodalStress(groupname, result);
  }

  //-------------------------------------------------------------------------
  // calculation and output of element center stresses in xyz-reference frame
  //-------------------------------------------------------------------------

  else if (stresstype == "cxyz")
  {
    WriteElementCenterStress(groupname, result);
  }

  //-----------------------------------------------------------------------------------
  // calculation and output of nodal and element center stresses in xyz-reference frame
  //-----------------------------------------------------------------------------------

  else if (stresstype == "cxyz_ndxyz")
  {
    WriteNodalStress(groupname, result);

    // reset result for postprocessing and output of element center stresses
    PostResult resultelestress = PostResult(field_);
    resultelestress.next_result();
    WriteElementCenterStress(groupname,resultelestress);
  }

  else if (stresstype == "nd123")
  {
    WriteNodalEigenStress(groupname, result);
  }

  else if (stresstype == "c123")
  {
    WriteElementCenterEigenStress(groupname, result);
  }

  else if (stresstype == "c123_nd123")
  {
    WriteNodalEigenStress(groupname, result);

    // reset result for postprocessing and output of element center stresses
    PostResult resultelestress = PostResult(field_);
    resultelestress.next_result();
    WriteElementCenterEigenStress(groupname,resultelestress);
  }

  else
  {
    dserror("Unknown stress/strain type");
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteNodalStress(const string groupname,
                                              PostResult& result)
{
  string name;
  string out;

  if (groupname=="gauss_2PK_stresses_xyz")
  {
    name="nodal_2PK_stresses_xyz";
    out="2nd Piola-Kirchhoff stresses";
  }
  else if (groupname=="gauss_cauchy_stresses_xyz")
  {
    name="nodal_cauchy_stresses_xyz";
    out="Cauchy stresses";
  }
  else if (groupname=="gauss_GL_strains_xyz")
  {
    name="nodal_GL_strains_xyz";
    out="Green-Lagrange strains";
  }
  else if (groupname=="gauss_EA_strains_xyz")
  {
    name="nodal_EA_strains_xyz";
    out="Euler-Almansi strains";
  }
  else
  {
    dserror("trying to write something that is not a stress or a strain");
    exit(1);
  }

  // new for file continuation
  bool multiple_files = false;

  // open file
  const string filename = filename_ + "_"+ field_->name() + "."+ name;
  ofstream file;
  int startfilepos = 0;
  if (myrank_==0)
  {
    file.open(filename.c_str());
    startfilepos = file.tellp(); // file position should be zero, but we stay flexible
  }

  map<string, vector<ofstream::pos_type> > resultfilepos;
  int stepsize = 0;

  if (myrank_==0)
    cout<<"writing node-based " << out << endl;

  // store information for later case file creation
  variableresulttypemap_[name] = "node";

  int numdf = 6;
  if (field_->problem()->num_dim()==2) numdf = 3;
  WriteNodalStressStep(file,result,resultfilepos,groupname,name,numdf);
  // how many bits are necessary per time step (we assume a fixed size)?
  if (myrank_==0)
  {
    stepsize = ((int) file.tellp())-startfilepos;
    if (stepsize <= 0) dserror("found invalid step size for result file");
  }
  else
    stepsize = 1; //use dummy value on other procs

  while (result.next_result())
  {
    const int indexsize = 80+2*sizeof(int)+(file.tellp()/stepsize+2)*sizeof(long);
    if (static_cast<long unsigned int>(file.tellp())+stepsize+indexsize>= FILE_SIZE_LIMIT_)
    {
      FileSwitcher(file, multiple_files, filesetmap_, resultfilepos, stepsize, name, filename);
    }

    WriteNodalStressStep(file,result,resultfilepos,groupname,name,numdf);
  }
  // store information for later case file creation
  filesetmap_[name].push_back(file.tellp()/stepsize);// has to be done BEFORE writing the index table
  variablenumdfmap_[name] = numdf;
  variablefilenamemap_[name] = filename;
  // store solution times vector for later case file creation
  {
    PostResult res = PostResult(field_); // this is needed!
    vector<double> restimes = res.get_result_times(field_->name(),groupname);
    timesetmap_[name] = restimes;
  }

  // append index table
  WriteIndexTable(file, resultfilepos[name]);
  resultfilepos[name].clear();

  // close result file
  if (file.is_open())
    file.close();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteNodalStressStep(ofstream& file,
                                                  PostResult& result,
                                                  map<string, vector<ofstream::pos_type> >& resultfilepos,
                                                  const string groupname,
                                                  const string name,
                                                  const int numdf) const
{
  //--------------------------------------------------------------------
  // calculate nodal stresses from gauss point stresses
  //--------------------------------------------------------------------

  const RCP<std::map<int,RCP<Epetra_SerialDenseMatrix> > > data =
    result.read_result_serialdensematrix(groupname);

  const RCP<DRT::Discretization> dis = field_->discretization();

  ParameterList p;
  p.set("action","postprocess_stress");
  p.set("stresstype","ndxyz");
  p.set("gpstressmap", data);
  RCP<Epetra_Vector> normal_stresses = LINALG::CreateVector(*(dis->DofRowMap()),true);
  RCP<Epetra_Vector> shear_stresses = LINALG::CreateVector(*(dis->DofRowMap()),true);
  dis->Evaluate(p,null,null,normal_stresses,shear_stresses,null);

  const Epetra_Map* nodemap = dis->NodeRowMap();
  RCP<Epetra_MultiVector> nodal_stresses = rcp(new Epetra_MultiVector(*nodemap, numdf));

  const int numnodes = dis->NumMyRowNodes();

  if (numdf==6)
  {
    for (int i=0;i<numnodes;++i)
    {
      (*((*nodal_stresses)(0)))[i] = (*normal_stresses)[3*i];
      (*((*nodal_stresses)(1)))[i] = (*normal_stresses)[3*i+1];
      (*((*nodal_stresses)(2)))[i] = (*normal_stresses)[3*i+2];
      (*((*nodal_stresses)(3)))[i] = (*shear_stresses)[3*i];
      (*((*nodal_stresses)(4)))[i] = (*shear_stresses)[3*i+1];
      (*((*nodal_stresses)(5)))[i] = (*shear_stresses)[3*i+2];
    }
  }
  if (numdf==3)
  {
    for (int i=0;i<numnodes;++i)
    {
      (*((*nodal_stresses)(0)))[i] = (*normal_stresses)[2*i];
      (*((*nodal_stresses)(1)))[i] = (*normal_stresses)[2*i+1];
      (*((*nodal_stresses)(2)))[i] = (*shear_stresses)[2*i];
    }
  }

  const Epetra_BlockMap& datamap = nodal_stresses->Map();

  // contract Epetra_MultiVector on proc0 (proc0 gets everything, other procs empty)
  RCP<Epetra_MultiVector> data_proc0 = rcp(new Epetra_MultiVector(*proc0map_,numdf));
  Epetra_Import proc0dofimporter(*proc0map_,datamap);
  int err = data_proc0->Import(*nodal_stresses,proc0dofimporter,Insert);
  if (err>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err);


  //--------------------------------------------------------------------
  // write some key words
  //--------------------------------------------------------------------

  vector<ofstream::pos_type>& filepos = resultfilepos[name];
  Write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());
  Write(file, "description");
  Write(file, "part");
  Write(file, field_->field_pos()+1);
  Write(file, "coordinates");

  //--------------------------------------------------------------------
  // write results
  //--------------------------------------------------------------------

  const int finalnumnode = proc0map_->NumGlobalElements();

  if (myrank_==0) // ensures pointer dofgids is valid
  {
    for (int idf=0; idf<numdf; ++idf)
    {
      for (int inode=0; inode<finalnumnode; inode++) // inode == lid of node because we use proc0map_
      {
        Write(file, static_cast<float>((*((*data_proc0)(idf)))[inode]));
      }
    }
  } // if (myrank_==0)

  Write(file, "END TIME STEP");
  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteElementCenterStress(const string groupname,
                                                      PostResult& result)
{
  string name;
  string out;

  if (groupname=="gauss_2PK_stresses_xyz")
  {
    name="element_2PK_stresses_xyz";
    out="2nd Piola-Kirchhoff stresses";
  }
  else if (groupname=="gauss_cauchy_stresses_xyz")
  {
    name="element_cauchy_stresses_xyz";
    out="Cauchy stresses";
  }
  else if (groupname=="gauss_GL_strains_xyz")
  {
    name="element_GL_strains_xyz";
    out="Green-Lagrange strains";
  }
  else if (groupname=="gauss_EA_strains_xyz")
  {
    name="element_EA_strains_xyz";
    out="Euler-Almansi strains";
  }
  else
  {
    dserror("trying to write something that is not a stress or a strain");
    exit(1);
  }

  // new for file continuation
  bool multiple_files = false;

  // open file
  const string filename = filename_ + "_"+ field_->name() + "."+ name;
  ofstream file;
  int startfilepos = 0;
  if (myrank_==0)
  {
    file.open(filename.c_str());
    startfilepos = file.tellp(); // file position should be zero, but we stay flexible
  }

  map<string, vector<ofstream::pos_type> > resultfilepos;
  int stepsize = 0;

  if (myrank_==0)
    cout<<"writing element-based center " << out << endl;

  // store information for later case file creation
  variableresulttypemap_[name] = "element";

  int numdf = 6;
  if (field_->problem()->num_dim()==2) numdf = 3;
  WriteElementCenterStressStep(file,result,resultfilepos,groupname,name,numdf);

  // how many bits are necessary per time step (we assume a fixed size)?
  if (myrank_==0)
  {
    stepsize = ((int) file.tellp())-startfilepos;
    if (stepsize <= 0) dserror("found invalid step size for result file");
  }
  else
    stepsize = 1; //use dummy value on other procs

  while (result.next_result())
  {
    const int indexsize = 80+2*sizeof(int)+(file.tellp()/stepsize+2)*sizeof(long);
    if (static_cast<long unsigned int>(file.tellp())+stepsize+indexsize>= FILE_SIZE_LIMIT_)
    {
      FileSwitcher(file, multiple_files, filesetmap_, resultfilepos, stepsize, name, filename);
    }
    WriteElementCenterStressStep(file,result,resultfilepos,groupname,name,numdf);
  }
  // store information for later case file creation
  filesetmap_[name].push_back(file.tellp()/stepsize);// has to be done BEFORE writing the index table
  variablenumdfmap_[name] = numdf;
  variablefilenamemap_[name] = filename;
  // store solution times vector for later case file creation
  {
    PostResult res = PostResult(field_); // this is needed!
    vector<double> restimes = res.get_result_times(field_->name(),groupname);
    timesetmap_[name] = restimes;
  }

  // append index table
  WriteIndexTable(file, resultfilepos[name]);
  resultfilepos[name].clear();

  // close result file
  if (file.is_open())
    file.close();

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteElementCenterStressStep(ofstream& file,
                                                          PostResult& result,
                                                          map<string, vector<ofstream::pos_type> >& resultfilepos,
                                                          const string groupname,
                                                          const string name,
                                                          const int numdf) const
{
  //--------------------------------------------------------------------
  // calculate element center stresses from gauss point stresses
  //--------------------------------------------------------------------

  const RefCountPtr<DRT::Discretization> dis = field_->discretization();
  const RefCountPtr<std::map<int,RefCountPtr<Epetra_SerialDenseMatrix> > > data =
    result.read_result_serialdensematrix(groupname);
  ParameterList p;
  p.set("action","postprocess_stress");
  p.set("stresstype","cxyz");
  p.set("gpstressmap", data);
  RCP<Epetra_MultiVector> elestress = rcp(new Epetra_MultiVector(*(dis->ElementRowMap()),numdf));
  p.set("elestress",elestress);
  dis->Evaluate(p,null,null,null,null,null);
  if (elestress==null)
  {
    dserror("vector containing element center stresses/strains not available");
  }

  //--------------------------------------------------------------------
  // write some key words
  //--------------------------------------------------------------------

  vector<ofstream::pos_type>& filepos = resultfilepos[name];
  Write(file, "BEGIN TIME STEP");
  filepos.push_back(file.tellp());
  Write(file, "description");
  Write(file, "part");
  Write(file, field_->field_pos()+1);

  const Epetra_BlockMap& datamap = elestress->Map();

  // do stupid conversion into Epetra map
  RefCountPtr<Epetra_Map> epetradatamap;
  epetradatamap = rcp(new Epetra_Map(datamap.NumGlobalElements(),
                                     datamap.NumMyElements(),
                                     datamap.MyGlobalElements(),
                                     0,
                                     datamap.Comm()));

  RefCountPtr<Epetra_Map> proc0datamap;
  proc0datamap = LINALG::AllreduceEMap(*epetradatamap,0);
  // sort proc0datamap so that we can loop it and get nodes in ascending order.
  std::vector<int> sortmap;
  sortmap.reserve(proc0datamap->NumMyElements());
  sortmap.assign(proc0datamap->MyGlobalElements(), proc0datamap->MyGlobalElements()+proc0datamap->NumMyElements());
  std::sort(sortmap.begin(), sortmap.end());
  proc0datamap = Teuchos::rcp(new Epetra_Map(-1, sortmap.size(), &sortmap[0], 0, proc0datamap->Comm()));

  // contract Epetra_MultiVector on proc0 (proc0 gets everything, other procs empty)
  RefCountPtr<Epetra_MultiVector> data_proc0 = rcp(new Epetra_MultiVector(*proc0datamap,numdf));
  Epetra_Import proc0dofimporter(*proc0datamap,datamap);
  int err = data_proc0->Import(*elestress,proc0dofimporter,Insert);
  if (err>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err);

  //--------------------------------------------------------------------
  // specify the element type
  //--------------------------------------------------------------------
  // loop over the different element types present
  if (myrank_==0)
  {
    if (eleGidPerDisType_.empty()==true) dserror("no element types available");
  }
  EleGidPerDisType::const_iterator iter;
  for (iter=eleGidPerDisType_.begin(); iter != eleGidPerDisType_.end(); ++iter)
  {
    const string ensighteleString = GetEnsightString(iter->first);
    const int numelepertype = (iter->second).size();
    vector<int> actelegids(numelepertype);
    actelegids = iter->second;
    // write element type
    Write(file, ensighteleString);

    //------------------------------------------------------------------
    // write results
    //------------------------------------------------------------------

    if (myrank_==0) // ensures pointer dofgids is valid
    {
      for (int idf=0; idf<numdf; ++idf)
      {
        for (int iele=0; iele<numelepertype; iele++) // inode == lid of node because we use proc0map_
        {
          // extract element global id
          const int gid = actelegids[iele];
          // get the dof local id w.r.t. the final datamap
          int lid = proc0datamap->LID(gid);
          if (lid > -1)
          {
            Write(file, static_cast<float>((*((*data_proc0)(idf)))[lid]));
          }
        }
      }
    } // if (myrank_==0)
  }
  Write(file, "END TIME STEP");
  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteNodalEigenStress(const string groupname,
                                                   PostResult& result)
{
  int numdf = 6;
  int numfiles = 6;
  if (field_->problem()->num_dim()==2)
  {
    numdf = 3;
    numfiles = 4;
  }

  vector<string> name(numfiles);
  string out;

  if (numdf == 6)
  {
    if (groupname=="gauss_2PK_stresses_xyz")
    {
      name[0]="nodal_2PK_stresses_eigenval1";
      name[1]="nodal_2PK_stresses_eigenval2";
      name[2]="nodal_2PK_stresses_eigenval3";
      name[3]="nodal_2PK_stresses_eigenvec1";
      name[4]="nodal_2PK_stresses_eigenvec2";
      name[5]="nodal_2PK_stresses_eigenvec3";
      out="principal 2nd Piola-Kirchhoff stresses";
    }
    else if (groupname=="gauss_cauchy_stresses_xyz")
    {
      name[0]="nodal_cauchy_stresses_eigenval1";
      name[1]="nodal_cauchy_stresses_eigenval2";
      name[2]="nodal_cauchy_stresses_eigenval3";
      name[3]="nodal_cauchy_stresses_eigenvec1";
      name[4]="nodal_cauchy_stresses_eigenvec2";
      name[5]="nodal_cauchy_stresses_eigenvec3";
      out="principal Cauchy stresses";
    }
    else if (groupname=="gauss_GL_strains_xyz")
    {
      name[0]="nodal_GL_strains_eigenval1";
      name[1]="nodal_GL_strains_eigenval2";
      name[2]="nodal_GL_strains_eigenval3";
      name[3]="nodal_GL_strains_eigenvec1";
      name[4]="nodal_GL_strains_eigenvec2";
      name[5]="nodal_GL_strains_eigenvec3";
      out="principal Green-Lagrange strains";
    }
    else if (groupname=="gauss_EA_strains_xyz")
    {
      name[0]="nodal_EA_strains_eigenval1";
      name[1]="nodal_EA_strains_eigenval2";
      name[2]="nodal_EA_strains_eigenval3";
      name[3]="nodal_EA_strains_eigenvec1";
      name[4]="nodal_EA_strains_eigenvec2";
      name[5]="nodal_EA_strains_eigenvec3";
      out="principal Euler-Almansi strains";
    }
    else
    {
      dserror("trying to write something that is not a stress or a strain");
      exit(1);
    }
  }
  else
  {
    if (groupname=="gauss_2PK_stresses_xyz")
    {
      name[0]="nodal_2PK_stresses_eigenval1";
      name[1]="nodal_2PK_stresses_eigenval2";
      name[2]="nodal_2PK_stresses_eigenvec1";
      name[3]="nodal_2PK_stresses_eigenvec2";
      out="principal 2nd Piola-Kirchhoff stresses";
    }
    else if (groupname=="gauss_cauchy_stresses_xyz")
    {
      name[0]="nodal_cauchy_stresses_eigenval1";
      name[1]="nodal_cauchy_stresses_eigenval2";
      name[2]="nodal_cauchy_stresses_eigenvec1";
      name[3]="nodal_cauchy_stresses_eigenvec2";
      out="principal Cauchy stresses";
    }
    else if (groupname=="gauss_GL_strains_xyz")
    {
      name[0]="nodal_GL_strains_eigenval1";
      name[1]="nodal_GL_strains_eigenval2";
      name[2]="nodal_GL_strains_eigenvec1";
      name[3]="nodal_GL_strains_eigenvec2";
      out="principal Green-Lagrange strains";
    }
    else if (groupname=="gauss_EA_strains_xyz")
    {
      name[0]="nodal_EA_strains_eigenval1";
      name[1]="nodal_EA_strains_eigenval2";
      name[2]="nodal_EA_strains_eigenvec1";
      name[3]="nodal_EA_strains_eigenvec2";
      out="principal Euler-Almansi strains";
    }
    else
    {
      dserror("trying to write something that is not a stress or a strain");
      exit(1);
    }
  }

  // new for file continuation
  vector<bool> multiple_files(numfiles);
  for (int i=0;i<numfiles;++i)
  {
    multiple_files[i] = false;
  }

  // open file
  vector<string> filenames(numfiles);
  for (int i=0;i<numfiles;++i)
  {
    filenames[i] = filename_ + "_"+ field_->name() + "."+ name[i];
  }

  std::vector<RCP<ofstream> > files(numfiles);
  vector<int> startfilepos(numfiles);
  for (int i=0;i<numfiles;++i)
    startfilepos[i] = 0;
  for (int i=0;i<numfiles;++i)
  {
    files[i] = rcp(new ofstream);

    if (myrank_==0)
    {
      files[i]->open(filenames[i].c_str());
      startfilepos[i] = files[i]->tellp(); // file position should be zero, but we stay flexible
    }
  }

  map<string, vector<ofstream::pos_type> > resultfilepos;
  vector<int> stepsize(numfiles);
  for (int i=0;i<numfiles;++i)
  {
    stepsize[i]=0;
  }

  if (myrank_==0)
    cout << "writing node-based " << out << endl;

  // store information for later case file creation
  for (int i=0;i<numfiles;++i)
  {
    variableresulttypemap_[name[i]] = "node";
  }

  WriteNodalEigenStressStep(files,result,resultfilepos,groupname,name,numdf);


  // how many bits are necessary per time step (we assume a fixed size)?
  if (myrank_==0)
  {
    for (int i=0;i<numfiles;++i)
    {
      stepsize[i] = ((int) files[i]->tellp())-startfilepos[i];
      if (stepsize[i] <= 0) dserror("found invalid step size for result file");
    }
  }
  else
  {
    for (int i=0;i<numfiles;++i)
    {
      stepsize[i] = 1; //use dummy value on other procs
    }
  }

  while (result.next_result())
  {
    for (int i=0;i<numfiles;++i)
    {
      const int indexsize = 80+2*sizeof(int)+(files[i]->tellp()/stepsize[i]+2)*sizeof(long);
      if (static_cast<long unsigned int>(files[i]->tellp())+stepsize[i]+indexsize>= FILE_SIZE_LIMIT_)
      {
        bool mf = multiple_files[i];
        FileSwitcher(*(files[i]),mf,filesetmap_,resultfilepos,stepsize[i],name[i],filenames[i]);
      }
    }

    WriteNodalEigenStressStep(files,result,resultfilepos,groupname,name,numdf);
  }
  // store information for later case file creation

  if (numfiles==6)
  {
    for (int i=0;i<numfiles;++i)
    {
      filesetmap_[name[i]].push_back(files[i]->tellp()/stepsize[i]);// has to be done BEFORE writing the index table
      if (i<3) variablenumdfmap_[name[i]] = 1;
      else     variablenumdfmap_[name[i]] = 3;
      variablefilenamemap_[name[i]] = filenames[i];
    }
  }
  else
  {
    for (int i=0;i<numfiles;++i)
    {
      filesetmap_[name[i]].push_back(files[i]->tellp()/stepsize[i]);// has to be done BEFORE writing the index table
      if (i<2) variablenumdfmap_[name[i]] = 1;
      else     variablenumdfmap_[name[i]] = 3;
      variablefilenamemap_[name[i]] = filenames[i];
    }
  }

  // store solution times vector for later case file creation
  for (int i=0;i<numfiles;++i)
  {
    PostResult res = PostResult(field_); // this is needed!
    vector<double> restimes = res.get_result_times(field_->name(),groupname);
    timesetmap_[name[i]] = restimes;
  }

  //append index table
  for (int i=0;i<numfiles;++i)
  {
    WriteIndexTable(*(files[i]), resultfilepos[name[i]]);
    resultfilepos[name[i]].clear();
    if (files[i]->is_open()) files[i]->close();
  }

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteNodalEigenStressStep(std::vector<RCP<ofstream> > files,
                                                       PostResult& result,
                                                       map<string, vector<ofstream::pos_type> >& resultfilepos,
                                                       const string groupname,
                                                       vector<string> name,
                                                       const int numdf)
{
  //--------------------------------------------------------------------
  // calculate nodal stresses from gauss point stresses
  //--------------------------------------------------------------------

  const RefCountPtr<std::map<int,RefCountPtr<Epetra_SerialDenseMatrix> > > data =
    result.read_result_serialdensematrix(groupname);

  const RefCountPtr<DRT::Discretization> dis = field_->discretization();

  ParameterList p;
  p.set("action","postprocess_stress");
  p.set("stresstype","ndxyz");
  p.set("gpstressmap", data);
  RefCountPtr<Epetra_Vector> normal_stresses = LINALG::CreateVector(*(dis->DofRowMap()),true);
  RefCountPtr<Epetra_Vector> shear_stresses = LINALG::CreateVector(*(dis->DofRowMap()),true);
  dis->Evaluate(p,null,null,normal_stresses,shear_stresses,null);

  const Epetra_BlockMap& datamap = normal_stresses->Map();

  // do stupid conversion into Epetra map
  RefCountPtr<Epetra_Map> epetradatamap;
  epetradatamap = rcp(new Epetra_Map(datamap.NumGlobalElements(),
                                     datamap.NumMyElements(),
                                     datamap.MyGlobalElements(),
                                     0,
                                     datamap.Comm()));

  RefCountPtr<Epetra_Map> proc0datamap;
  proc0datamap = LINALG::AllreduceEMap(*epetradatamap,0);

  // contract Epetra_MultiVector on proc0 (proc0 gets everything, other procs empty)
  RefCountPtr<Epetra_Vector> normal_data_proc0 = rcp(new Epetra_Vector(*proc0datamap));
  RefCountPtr<Epetra_Vector> shear_data_proc0 = rcp(new Epetra_Vector(*proc0datamap));
  Epetra_Import proc0dofimporter(*proc0datamap,*epetradatamap);
  int err1 = normal_data_proc0->Import(*normal_stresses,proc0dofimporter,Insert);
  if (err1>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err1);
  int err2 = shear_data_proc0->Import(*shear_stresses,proc0dofimporter,Insert);
  if (err2>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err2);

  //--------------------------------------------------------------------
  // write some key words
  //--------------------------------------------------------------------

  int numfiles=6;
  if (numdf==3) numfiles=4;

  for (int i=0;i<numfiles;++i)
  {
    vector<ofstream::pos_type>& filepos = resultfilepos[name[i]];
    Write(*(files[i]), "BEGIN TIME STEP");
    filepos.push_back(files[i]->tellp());
    Write(*(files[i]), "description");
    Write(*(files[i]), "part");
    Write(*(files[i]), field_->field_pos()+1);
    Write(*(files[i]), "coordinates");
  }

  //--------------------------------------------------------------------
  // write results
  //--------------------------------------------------------------------

  const int finalnumnode = proc0map_->NumGlobalElements();

  if (myrank_==0) // ensures pointer dofgids is valid
  {
    if (numdf==6)
    {
      vector<Epetra_SerialDenseMatrix> eigenvec(finalnumnode, Epetra_SerialDenseMatrix(3,3));
      vector<Epetra_SerialDenseVector> eigenval(finalnumnode, Epetra_SerialDenseVector(3));

      for (int i=0;i<finalnumnode;++i)
      {
        (eigenvec[i])(0,0) = (*normal_data_proc0)[3*i];
        (eigenvec[i])(0,1) = (*shear_data_proc0)[3*i];
        (eigenvec[i])(0,2) = (*shear_data_proc0)[3*i+2];
        (eigenvec[i])(1,0) = (eigenvec[i])(0,1);
        (eigenvec[i])(1,1) = (*normal_data_proc0)[3*i+1];
        (eigenvec[i])(1,2) = (*shear_data_proc0)[3*i+1];
        (eigenvec[i])(2,0) = (eigenvec[i])(0,2);
        (eigenvec[i])(2,1) = (eigenvec[i])(1,2);
        (eigenvec[i])(2,2) = (*normal_data_proc0)[3*i+2];

        LINALG::SymmetricEigenProblem((eigenvec[i]), eigenval[i], true);
      }

      for (int inode=0; inode<finalnumnode; inode++) // inode == lid of node because we use proc0datamap
      {
        for (int i=0;i<3;++i) Write(*(files[i]), static_cast<float>((eigenval[inode])[i]));
      }

      for (int idf=0; idf<3; ++idf)
      {
        for (int inode=0; inode<finalnumnode; inode++) // inode == lid of node because we use proc0datamap
        {
          for (int i=0;i<3;++i) Write(*(files[i+3]), static_cast<float>((eigenvec[inode])(idf,i)));
        }
      }
    }
    else
    {
      vector<Epetra_SerialDenseMatrix> eigenvec(finalnumnode, Epetra_SerialDenseMatrix(2,2));
      vector<Epetra_SerialDenseVector> eigenval(finalnumnode, Epetra_SerialDenseVector(2));

      for (int i=0;i<finalnumnode;++i)
      {
        (eigenvec[i])(0,0) = (*normal_data_proc0)[2*i];
        (eigenvec[i])(0,1) = (*shear_data_proc0)[2*i];
        (eigenvec[i])(1,0) = (eigenvec[i])(0,1);
        (eigenvec[i])(1,1) = (*normal_data_proc0)[2*i+1];

        LINALG::SymmetricEigenProblem((eigenvec[i]), eigenval[i], true);
      }

      for (int inode=0; inode<finalnumnode; inode++) // inode == lid of node because we use proc0datamap
      {
        for (int i=0;i<2;++i) Write(*(files[i]), static_cast<float>((eigenval[inode])[i]));
      }
      for (int idf=0; idf<2; ++idf)
      {
        for (int inode=0; inode<finalnumnode; inode++) // inode == lid of node because we use proc0datamap
        {
          for (int i=0;i<2;++i) Write(*(files[i+2]), static_cast<float>((eigenvec[inode])(idf,i)));
        }
      }
      // 2D vector for eigenproblem needed, 3D vector for paraview -> append 0.
      for (int inode=0; inode<finalnumnode; inode++) // inode == lid of node because we use proc0datamap
      {
        for (int i=0;i<2;++i) Write(*(files[i+2]), static_cast<float>(0.));
      }
    }
  } // if (myrank_==0)

  for (int i=0;i<numfiles;++i) Write(*(files[i]), "END TIME STEP");

  return;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteElementCenterEigenStress(const string groupname,
                                                           PostResult& result)
{
  int numdf = 6;
  int numfiles = 6;
  if (field_->problem()->num_dim()==2)
  {
    numdf = 3;
    numfiles = 4;
  }

  vector<string> name(numfiles);
  string out;

  if (numdf == 6)
  {
    if (groupname=="gauss_2PK_stresses_xyz")
    {
      name[0]="element_2PK_stresses_eigenval1";
      name[1]="element_2PK_stresses_eigenval2";
      name[2]="element_2PK_stresses_eigenval3";
      name[3]="element_2PK_stresses_eigenvec1";
      name[4]="element_2PK_stresses_eigenvec2";
      name[5]="element_2PK_stresses_eigenvec3";
      out="principal 2nd Piola-Kirchhoff stresses";
    }
    else if (groupname=="gauss_cauchy_stresses_xyz")
    {
      name[0]="element_cauchy_stresses_eigenval1";
      name[1]="element_cauchy_stresses_eigenval2";
      name[2]="element_cauchy_stresses_eigenval3";
      name[3]="element_cauchy_stresses_eigenvec1";
      name[4]="element_cauchy_stresses_eigenvec2";
      name[5]="element_cauchy_stresses_eigenvec3";
      out="principal Cauchy stresses";
    }
    else if (groupname=="gauss_GL_strains_xyz")
    {
      name[0]="element_GL_strains_eigenval1";
      name[1]="element_GL_strains_eigenval2";
      name[2]="element_GL_strains_eigenval3";
      name[3]="element_GL_strains_eigenvec1";
      name[4]="element_GL_strains_eigenvec2";
      name[5]="element_GL_strains_eigenvec3";
      out="principal Green-Lagrange strains";
    }
    else if (groupname=="gauss_EA_strains_xyz")
    {
      name[0]="element_EA_strains_eigenval1";
      name[1]="element_EA_strains_eigenval2";
      name[2]="element_EA_strains_eigenval3";
      name[3]="element_EA_strains_eigenvec1";
      name[4]="element_EA_strains_eigenvec2";
      name[5]="element_EA_strains_eigenvec3";
      out="principal Euler-Almansi strains";
    }
    else
    {
      dserror("trying to write something that is not a stress or a strain");
      exit(1);
    }
  }
  else
  {
    if (groupname=="gauss_2PK_stresses_xyz")
    {
      name[0]="element_2PK_stresses_eigenval1";
      name[1]="element_2PK_stresses_eigenval2";
      name[2]="element_2PK_stresses_eigenvec1";
      name[3]="element_2PK_stresses_eigenvec2";
      out="principal 2nd Piola-Kirchhoff stresses";
    }
    else if (groupname=="gauss_cauchy_stresses_xyz")
    {
      name[0]="element_cauchy_stresses_eigenval1";
      name[1]="element_cauchy_stresses_eigenval2";
      name[2]="element_cauchy_stresses_eigenvec1";
      name[3]="element_cauchy_stresses_eigenvec2";
      out="principal Cauchy stresses";
    }
    else if (groupname=="gauss_GL_strains_xyz")
    {
      name[0]="element_GL_strains_eigenval1";
      name[1]="element_GL_strains_eigenval2";
      name[2]="element_GL_strains_eigenvec1";
      name[3]="element_GL_strains_eigenvec2";
      out="principal Green-Lagrange strains";
    }
    else if (groupname=="gauss_EA_strains_xyz")
    {
      name[0]="element_EA_strains_eigenval1";
      name[1]="element_EA_strains_eigenval2";
      name[2]="element_EA_strains_eigenvec1";
      name[3]="element_EA_strains_eigenvec2";
      out="principal Euler-Almansi strains";
    }
    else
    {
      dserror("trying to write something that is not a stress or a strain");
      exit(1);
    }
  }

  // new for file continuation
  vector<bool> multiple_files(numfiles);
  for (int i=0;i<numfiles;++i)
  {
    multiple_files[i] = false;
  }

  // open file
  vector<string> filenames(numfiles);
  for (int i=0;i<numfiles;++i)
  {
    filenames[i] = filename_ + "_"+ field_->name() + "."+ name[i];
  }

  std::vector<RCP<ofstream> > files(numfiles);
  vector<int> startfilepos(numfiles);
  for (int i=0;i<numfiles;++i)
    startfilepos[i] = 0;

  for (int i=0;i<numfiles;++i)
  {
    files[i] = rcp(new ofstream);

    if (myrank_==0)
    {
      files[i]->open(filenames[i].c_str());
      startfilepos[i] = files[i]->tellp(); // file position should be zero, but we stay flexible
    }
  }

  map<string, vector<ofstream::pos_type> > resultfilepos;
  vector<int> stepsize(numfiles);
  for (int i=0;i<numfiles;++i)
  {
    stepsize[i]=0;
  }

  if (myrank_==0)
    cout << "writing element-based center " << out << endl;

  // store information for later case file creation
  for (int i=0;i<numfiles;++i)
  {
    variableresulttypemap_[name[i]] = "element";
  }

  WriteElementCenterEigenStressStep(files,result,resultfilepos,groupname,name,numdf);

  // how many bits are necessary per time step (we assume a fixed size)?
  if (myrank_==0)
  {
    for (int i=0;i<numfiles;++i)
    {
      stepsize[i] = ((int) files[i]->tellp())-startfilepos[i];
      if (stepsize[i] <= 0) dserror("found invalid step size for result file");
    }
  }
  else
  {
    for (int i=0;i<numfiles;++i)
    {
      stepsize[i] = 1; //use dummy value on other procs
    }
  }

  while (result.next_result())
  {
    for (int i=0;i<numfiles;++i)
    {
      const int indexsize = 80+2*sizeof(int)+(files[i]->tellp()/stepsize[i]+2)*sizeof(long);
      if (static_cast<long unsigned int>(files[i]->tellp())+stepsize[i]+indexsize>= FILE_SIZE_LIMIT_)
      {
        bool mf = multiple_files[i];
        FileSwitcher(*(files[i]),mf,filesetmap_,resultfilepos,stepsize[i],name[i],filenames[i]);
      }
    }

    WriteElementCenterEigenStressStep(files,result,resultfilepos,groupname,name,numdf);
  }
  // store information for later case file creation

  if (numfiles==6)
  {
    for (int i=0;i<numfiles;++i)
    {
      filesetmap_[name[i]].push_back(files[i]->tellp()/stepsize[i]);// has to be done BEFORE writing the index table
      if (i<3) variablenumdfmap_[name[i]] = 1;
      else     variablenumdfmap_[name[i]] = 3;
      variablefilenamemap_[name[i]] = filenames[i];
    }
  }
  else
  {
    for (int i=0;i<numfiles;++i)
    {
      filesetmap_[name[i]].push_back(files[i]->tellp()/stepsize[i]);// has to be done BEFORE writing the index table
      if (i<2) variablenumdfmap_[name[i]] = 1;
      else     variablenumdfmap_[name[i]] = 3;
      variablefilenamemap_[name[i]] = filenames[i];
    }
  }

  // store solution times vector for later case file creation
  for (int i=0;i<numfiles;++i)
  {
    PostResult res = PostResult(field_); // this is needed!
    vector<double> restimes = res.get_result_times(field_->name(),groupname);
    timesetmap_[name[i]] = restimes;
  }

  //append index table
  for (int i=0;i<numfiles;++i)
  {
    WriteIndexTable(*(files[i]), resultfilepos[name[i]]);
    resultfilepos[name[i]].clear();
    if (files[i]->is_open()) files[i]->close();
  }

  return;
}


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void StructureEnsightWriter::WriteElementCenterEigenStressStep(std::vector<RCP<ofstream> > files,
                                                               PostResult& result,
                                                               map<string, vector<ofstream::pos_type> >& resultfilepos,
                                                               const string groupname,
                                                               vector<string> name,
                                                               const int numdf)
{
  //--------------------------------------------------------------------
  // calculate nodal stresses from gauss point stresses
  //--------------------------------------------------------------------

  const RefCountPtr<std::map<int,RefCountPtr<Epetra_SerialDenseMatrix> > > data =
    result.read_result_serialdensematrix(groupname);

  const RefCountPtr<DRT::Discretization> dis = field_->discretization();

  ParameterList p;
  p.set("action","postprocess_stress");
  p.set("stresstype","cxyz");
  p.set("gpstressmap", data);
  RCP<Epetra_MultiVector> elestress = rcp(new Epetra_MultiVector(*(dis->ElementRowMap()),numdf));
  p.set("elestress",elestress);
  dis->Evaluate(p,null,null,null,null,null);
  if (elestress==null)
  {
    dserror("vector containing element center stresses/strains not available");
  }

  const Epetra_BlockMap& datamap = elestress->Map();

  // do stupid conversion into Epetra map
  RefCountPtr<Epetra_Map> epetradatamap;
  epetradatamap = rcp(new Epetra_Map(datamap.NumGlobalElements(),
                                     datamap.NumMyElements(),
                                     datamap.MyGlobalElements(),
                                     0,
                                     datamap.Comm()));

  RefCountPtr<Epetra_Map> proc0datamap;
  proc0datamap = LINALG::AllreduceEMap(*epetradatamap,0);
  // sort proc0datamap so that we can loop it and get nodes in ascending order.
  std::vector<int> sortmap;
  sortmap.reserve(proc0datamap->NumMyElements());
  sortmap.assign(proc0datamap->MyGlobalElements(), proc0datamap->MyGlobalElements()+proc0datamap->NumMyElements());
  std::sort(sortmap.begin(), sortmap.end());
  proc0datamap = Teuchos::rcp(new Epetra_Map(-1, sortmap.size(), &sortmap[0], 0, proc0datamap->Comm()));

  // contract Epetra_MultiVector on proc0 (proc0 gets everything, other procs empty)
  RefCountPtr<Epetra_MultiVector> data_proc0 = rcp(new Epetra_MultiVector(*proc0datamap,numdf));
  Epetra_Import proc0dofimporter(*proc0datamap,datamap);
  int err = data_proc0->Import(*elestress,proc0dofimporter,Insert);
  if (err>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err);

  //--------------------------------------------------------------------
  // write some key words
  //--------------------------------------------------------------------

  int numfiles=6;
  if (numdf==3) numfiles=4;

  for (int i=0;i<numfiles;++i)
  {
    vector<ofstream::pos_type>& filepos = resultfilepos[name[i]];
    Write(*(files[i]), "BEGIN TIME STEP");
    filepos.push_back(files[i]->tellp());
    Write(*(files[i]), "description");
    Write(*(files[i]), "part");
    Write(*(files[i]), field_->field_pos()+1);
  }

  //--------------------------------------------------------------------
  // specify the element type
  //--------------------------------------------------------------------
  // loop over the different element types present
  if (myrank_==0)
  {
    if (eleGidPerDisType_.empty()==true) dserror("no element types available");
  }
  EleGidPerDisType::const_iterator iter;
  for (iter=eleGidPerDisType_.begin(); iter != eleGidPerDisType_.end(); ++iter)
  {
    const string ensighteleString = GetEnsightString(iter->first);
    const int numelepertype = (iter->second).size();
    vector<int> actelegids(numelepertype);
    actelegids = iter->second;
    // write element type
    for (int i=0;i<numfiles;++i) Write(*(files[i]), ensighteleString);

    //--------------------------------------------------------------------
    // write results
    //--------------------------------------------------------------------

    if (myrank_==0) // ensures pointer dofgids is valid
    {
      if (numdf==6)
      {
        vector<Epetra_SerialDenseMatrix> eigenvec(numelepertype, Epetra_SerialDenseMatrix(3,3));
        vector<Epetra_SerialDenseVector> eigenval(numelepertype, Epetra_SerialDenseVector(3));

        for (int i=0;i<numelepertype;++i)
        {
          // extract element global id
          const int gid = actelegids[i];
          // get the dof local id w.r.t. the final datamap
          int lid = proc0datamap->LID(gid);

          (eigenvec[i])(0,0) = (*(*data_proc0)(0))[lid];
          (eigenvec[i])(0,1) = (*(*data_proc0)(3))[lid];
          (eigenvec[i])(0,2) = (*(*data_proc0)(5))[lid];
          (eigenvec[i])(1,0) = (eigenvec[i])(0,1);
          (eigenvec[i])(1,1) = (*(*data_proc0)(1))[lid];
          (eigenvec[i])(1,2) = (*(*data_proc0)(4))[lid];
          (eigenvec[i])(2,0) = (eigenvec[i])(0,2);
          (eigenvec[i])(2,1) = (eigenvec[i])(1,2);
          (eigenvec[i])(2,2) = (*(*data_proc0)(2))[lid];

          LINALG::SymmetricEigenProblem((eigenvec[i]), eigenval[i], true);
        }

        for (int iele=0; iele<numelepertype; iele++)
        {
          for (int i=0;i<3;++i) Write(*(files[i]), static_cast<float>((eigenval[iele])[i]));
        }

        for (int idf=0; idf<3; ++idf)
        {
          for (int iele=0; iele<numelepertype; iele++)
          {
            for (int i=0;i<3;++i) Write(*(files[i+3]), static_cast<float>((eigenvec[iele])(idf,i)));
          }
        }
      }
      else
      {
        vector<Epetra_SerialDenseMatrix> eigenvec(numelepertype, Epetra_SerialDenseMatrix(2,2));
        vector<Epetra_SerialDenseVector> eigenval(numelepertype, Epetra_SerialDenseVector(2));

        for (int i=0;i<numelepertype;++i)
        {
          // extract element global id
          const int gid = actelegids[i];
          // get the dof local id w.r.t. the final datamap
          int lid = proc0datamap->LID(gid);

          (eigenvec[i])(0,0) = (*(*data_proc0)(0))[lid];
          (eigenvec[i])(0,1) = (*(*data_proc0)(2))[lid];
          (eigenvec[i])(1,0) = (eigenvec[i])(0,1);
          (eigenvec[i])(1,1) = (*(*data_proc0)(1))[lid];

          LINALG::SymmetricEigenProblem((eigenvec[i]), eigenval[i], true);
        }

        for (int iele=0; iele<numelepertype; iele++)
        {
          for (int i=0;i<2;++i) Write(*(files[i]), static_cast<float>((eigenval[iele])[i]));
        }

        for (int idf=0; idf<2; ++idf)
        {
          for (int iele=0; iele<numelepertype; iele++)
          {
            for (int i=0;i<2;++i) Write(*(files[i+2]), static_cast<float>((eigenvec[iele])(idf,i)));
          }
        }
        // 2D vector for eigenproblem needed, 3D vector for paraview -> append 0.
        for (int inode=0; inode<numelepertype; inode++) // inode == lid of node because we use proc0datamap
        {
          for (int i=0;i<2;++i) Write(*(files[i+2]), static_cast<float>(0.));
        }
      }
    } // if (myrank_==0)

    for (int i=0;i<numfiles;++i) Write(*(files[i]), "END TIME STEP");
  }

  return;
}

#endif
