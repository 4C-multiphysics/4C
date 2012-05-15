/*!----------------------------------------------------------------------
\file shell8_input.cpp
\brief

<pre>
Maintainer: Michael Gee
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef D_SHELL8

#include "shell8.H"
#include "../drt_lib/drt_linedefinition.H"


/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Shell8::ReadElement(const std::string& eletype,
                                        const std::string& distype,
                                        DRT::INPUT::LineDefinition* linedef)
{
  // read number of material model
  int material = 0;
  linedef->ExtractInt("MAT",material);
  SetMaterial(material);

  linedef->ExtractDouble("THICK",thickness_);

  std::vector<int> gp;
  linedef->ExtractIntVector("GP",gp);
  std::copy(gp.begin(),gp.end(),ngp_);

  linedef->ExtractInt("GP_TRI",ngptri_);

  std::string buffer;
  linedef->ExtractString("FORCES",buffer);

  if      (buffer=="XYZ")       forcetype_ = s8_xyz;
  else if (buffer=="RST")       forcetype_ = s8_rst;
  else if (buffer=="RST_ortho") forcetype_ = s8_rst_ortho;
  else dserror("Reading of SHELL8 element failed");

  linedef->ExtractString("EAS",buffer);
  if      (buffer=="none")  eas_[0]=0;
  else if (buffer=="N4_1")  eas_[0]=1;
  else if (buffer=="N4_2")  eas_[0]=2;
  else if (buffer=="N4_3")  eas_[0]=3;
  else if (buffer=="N4_4")  eas_[0]=4;
  else if (buffer=="N4_5")  eas_[0]=5;
  else if (buffer=="N4_7")  eas_[0]=7;
  else if (buffer=="N9_7")  eas_[0]=7;
  else if (buffer=="N9_9")  eas_[0]=9;
  else if (buffer=="N9_11") eas_[0]=11;
  else dserror("Illegal eas parameter '%s'", buffer.c_str());

  linedef->ExtractString("EAS2",buffer);
  if      (buffer=="none")  eas_[1]=0;
  else if (buffer=="N4_4")  eas_[1]=4;
  else if (buffer=="N4_5")  eas_[1]=5;
  else if (buffer=="N4_6")  eas_[1]=6;
  else if (buffer=="N4_7")  eas_[1]=7;
  else if (buffer=="N9_9")  eas_[1]=9;
  else if (buffer=="N9_11") eas_[1]=11;
  else dserror("Illegal eas parameter '%s'", buffer.c_str());

  linedef->ExtractString("EAS3",buffer);
  if      (buffer=="none")  eas_[2]=0;
  else if (buffer=="N_1")   eas_[2]=1;
  else if (buffer=="N_3")   eas_[2]=3;
  else if (buffer=="N_4")   eas_[2]=4;
  else if (buffer=="N_6")   eas_[2]=6;
  else if (buffer=="N_8")   eas_[2]=8;
  else if (buffer=="N_9")   eas_[2]=9;
  else dserror("Illegal eas parameter '%s'", buffer.c_str());

  linedef->ExtractString("EAS4",buffer);
  if      (buffer=="none")  eas_[3]=0;
  else if (buffer=="N4_2")  eas_[3]=2;
  else if (buffer=="N4_4")  eas_[3]=4;
  else if (buffer=="N9_2")  eas_[3]=2;
  else if (buffer=="N9_4")  eas_[3]=4;
  else if (buffer=="N9_6")  eas_[3]=6;
  else dserror("Illegal eas parameter '%s'", buffer.c_str());

  linedef->ExtractString("EAS5",buffer);
  if      (buffer=="none")  eas_[4]=0;
  else if (buffer=="N4_2")  eas_[4]=2;
  else if (buffer=="N4_4")  eas_[4]=4;
  else if (buffer=="N9_2")  eas_[4]=2;
  else if (buffer=="N9_4")  eas_[4]=4;
  else if (buffer=="N9_6")  eas_[4]=6;
  else dserror("Illegal eas parameter '%s'", buffer.c_str());

  // count no. eas parameters
  nhyb_ = 0;
  for (int i=0; i<5; ++i) nhyb_ += eas_[i];

  // create arrays alfa, Dtildinv, Lt, Rtild in data_
  vector<double> alfa(nhyb_);
  vector<double> alfao(nhyb_);
  vector<double> Rtild(nhyb_);
  std::fill(alfa.begin(), alfa.end(), 0);
  std::fill(alfao.begin(),alfao.end(),0);
  std::fill(Rtild.begin(),Rtild.end(),0);

  Epetra_SerialDenseMatrix Dtildinv;
  Epetra_SerialDenseMatrix Lt;
  Dtildinv.Shape(nhyb_,nhyb_);
  Lt.Shape(nhyb_,NumNode()*6);

  data_.Add("alfa",alfa);
  data_.Add("alfao",alfao);
  data_.Add("Rtild",Rtild);
  data_.Add("Dtildinv",Dtildinv);
  data_.Add("Lt",Lt);

  // read ANS
  linedef->ExtractString("ANS",buffer);
  if      (buffer=="none") ans_=0;
  else if (buffer=="Q")    ans_=1;
  else if (buffer=="T")    ans_=2;
  else if (buffer=="QT")   ans_=3;
  else if (buffer=="TQ")   ans_=3;
  else dserror("Illegal ans parameter '%s'", buffer.c_str());

  // read SDC
  linedef->ExtractDouble("SDC",sdc_);

  return true;
}


#if 0
/*----------------------------------------------------------------------*
 |  read element input (public)                              mwgee 11/06|
 *----------------------------------------------------------------------*/
bool DRT::ELEMENTS::Shell8::ReadElement()
{
  // read element's nodes
  int ierr=0;
  int nnode=0;
  int nodes[9];
  frchk("QUAD4",&ierr);
  if (ierr==1)
  {
    nnode = 4;
    frint_n("QUAD4",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("QUAD8",&ierr);
  if (ierr==1)
  {
    nnode = 8;
    frint_n("QUAD8",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("QUAD9",&ierr);
  if (ierr==1)
  {
    nnode = 9;
    frint_n("QUAD9",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("TRI3",&ierr);
  if (ierr==1)
  {
    nnode = 3;
    frint_n("TRI3",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }
  frchk("TRI6",&ierr);
  if (ierr==1)
  {
    nnode = 6;
    frint_n("TRI6",nodes,nnode,&ierr);
    if (ierr != 1) dserror("Reading of ELEMENT Topology failed");
  }

  // reduce node numbers by one
  for (int i=0; i<nnode; ++i) nodes[i]--;

  SetNodeIds(nnode,nodes);

  // read number of material model
  material_ = 0;
  frint("MAT",&material_,&ierr);
  if (ierr!=1) dserror("Reading of SHELL8 element failed");
  SetMaterial(material_);

  // read shell thickness
  thickness_ = 1.0;
  frdouble("THICK",&thickness_,&ierr);
  if (ierr!=1) dserror("Reading of SHELL8 element failed");

  // read gaussian points
  frint_n("GP",ngp_,3,&ierr);
  if (ierr!=1) dserror("Reading of SHELL8 element failed");

  // read gaussian points for triangle element
  frint("GP_TRI",&ngptri_,&ierr);
  if (ierr!=1) dserror("Reading of SHELL8 element failed");

  // read local or global forces
  char buffer[50];
  frchar("FORCES",buffer,&ierr);
  if (ierr)
  {
   if      (strncmp(buffer,"XYZ",3)==0)       forcetype_ = s8_xyz;
   else if (strncmp(buffer,"RST",3)==0)       forcetype_ = s8_rst;
   else if (strncmp(buffer,"RST_ortho",9)==0) forcetype_ = s8_rst_ortho;
   else dserror("Reading of SHELL8 element failed");
  }

  // read EAS parameters
  for (int i=0; i<5; ++i) eas_[i] = 0;
  char* colpointer = strstr(fractplace(),"EAS");
  colpointer+=3;
  colpointer = strpbrk(colpointer,"Nn");
  ierr = sscanf(colpointer," %s ",buffer);
  if (ierr!=1) dserror("Reading of shell8 eas failed");
  if (strncmp(buffer,"none",4)==0)  eas_[0]=0;
  if (strncmp(buffer,"N4_1",4)==0)  eas_[0]=1;
  if (strncmp(buffer,"N4_2",4)==0)  eas_[0]=2;
  if (strncmp(buffer,"N4_3",4)==0)  eas_[0]=3;
  if (strncmp(buffer,"N4_4",4)==0)  eas_[0]=4;
  if (strncmp(buffer,"N4_5",4)==0)  eas_[0]=5;
  if (strncmp(buffer,"N4_7",4)==0)  eas_[0]=7;
  if (strncmp(buffer,"N9_7",4)==0)  eas_[0]=7;
  if (strncmp(buffer,"N9_9",4)==0)  eas_[0]=9;
  if (strncmp(buffer,"N9_11",4)==0) eas_[0]=11;
  colpointer += strlen(buffer);

  colpointer = strpbrk(colpointer,"Nn");
  ierr = sscanf(colpointer," %s ",buffer);
  if (ierr!=1) dserror("Reading of shell8 eas failed");
  if (strncmp(buffer,"none",4)==0)  eas_[1]=0;
  if (strncmp(buffer,"N4_4",4)==0)  eas_[1]=4;
  if (strncmp(buffer,"N4_5",4)==0)  eas_[1]=5;
  if (strncmp(buffer,"N4_6",4)==0)  eas_[1]=6;
  if (strncmp(buffer,"N4_7",4)==0)  eas_[1]=7;
  if (strncmp(buffer,"N9_9",4)==0)  eas_[1]=9;
  if (strncmp(buffer,"N9_11",4)==0) eas_[1]=11;
  colpointer += strlen(buffer);

  colpointer = strpbrk(colpointer,"Nn");
  ierr = sscanf(colpointer," %s ",buffer);
  if (ierr!=1) dserror("Reading of shell8 eas failed");
  if (strncmp(buffer,"none",4)==0)  eas_[2]=0;
  if (strncmp(buffer,"N_1",4)==0)   eas_[2]=1;
  if (strncmp(buffer,"N_3",4)==0)   eas_[2]=3;
  if (strncmp(buffer,"N_4",4)==0)   eas_[2]=4;
  if (strncmp(buffer,"N_6",4)==0)   eas_[2]=6;
  if (strncmp(buffer,"N_8",4)==0)   eas_[2]=8;
  if (strncmp(buffer,"N_9",4)==0)   eas_[2]=9;
  colpointer += strlen(buffer);

  colpointer = strpbrk(colpointer,"Nn");
  ierr = sscanf(colpointer," %s ",buffer);
  if (ierr!=1) dserror("Reading of shell8 eas failed");
  if (strncmp(buffer,"none",4)==0)  eas_[3]=0;
  if (strncmp(buffer,"N4_2",4)==0)  eas_[3]=2;
  if (strncmp(buffer,"N4_4",4)==0)  eas_[3]=4;
  if (strncmp(buffer,"N9_2",4)==0)  eas_[3]=2;
  if (strncmp(buffer,"N9_4",4)==0)  eas_[3]=4;
  if (strncmp(buffer,"N9_6",4)==0)  eas_[3]=6;
  colpointer += strlen(buffer);

  colpointer = strpbrk(colpointer,"Nn");
  ierr = sscanf(colpointer," %s ",buffer);
  if (ierr!=1) dserror("Reading of shell8 eas failed");
  if (strncmp(buffer,"none",4)==0)  eas_[4]=0;
  if (strncmp(buffer,"N4_2",4)==0)  eas_[4]=2;
  if (strncmp(buffer,"N4_4",4)==0)  eas_[4]=4;
  if (strncmp(buffer,"N9_2",4)==0)  eas_[4]=2;
  if (strncmp(buffer,"N9_4",4)==0)  eas_[4]=4;
  if (strncmp(buffer,"N9_6",4)==0)  eas_[4]=6;

  // count no. eas parameters
  nhyb_ = 0;
  for (int i=0; i<5; ++i) nhyb_ += eas_[i];
  // create arrays alfa, Dtildinv, Lt, Rtild in data_
  vector<double> alfa(nhyb_);
  vector<double> alfao(nhyb_);
  vector<double> Rtild(nhyb_);
  for (int i=0; i<nhyb_; ++i)
  {
    alfa[i] = 0.0;
    alfao[i] = 0.0;
    Rtild[i] = 0.0;
  }
  Epetra_SerialDenseMatrix Dtildinv;
  Epetra_SerialDenseMatrix Lt;
  Dtildinv.Shape(nhyb_,nhyb_);
  Lt.Shape(nhyb_,nnode*6);
  data_.Add("alfa",alfa);
  data_.Add("alfao",alfao);
  data_.Add("Rtild",Rtild);
  data_.Add("Dtildinv",Dtildinv);
  data_.Add("Lt",Lt);

  // read ANS
  ans_ = 0;
  frchar("ANS",buffer,&ierr);
  if (ierr!=1) dserror("reading of shell8 ans failed");
  if (strncmp(buffer,"none",4)==0) ans_=0;
  if (strncmp(buffer,"Q",4)==0)    ans_=1;
  if (strncmp(buffer,"T",4)==0)    ans_=2;
  if (strncmp(buffer,"QT",4)==0)   ans_=3;
  if (strncmp(buffer,"TQ",4)==0)   ans_=3;

  // read SDC
  sdc_ = 1.0;
  frdouble("SDC",&sdc_,&ierr);
  if (ierr!=1) dserror("Reading of shell8 sdc failed");

  return true;
} // Shell8::ReadElement()
#endif


#endif  // #ifdef D_SHELL8
