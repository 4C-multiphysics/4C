/*!----------------------------------------------------------------------
\file contact_interface_tools.cpp
\brief

<pre>
-------------------------------------------------------------------------
                        BACI Contact library
            Copyright (2008) Technical University of Munich

Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed,
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de)
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de

-------------------------------------------------------------------------
</pre>

<pre>
Maintainer: Alexander Popp
            gee@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15239
</pre>

*----------------------------------------------------------------------*/
#ifdef CCADISCRET

#include "contact_interface.H"
#include "contact_integrator.H"
#include "contact_defines.H"
#include "friction_node.H"
#include "selfcontact_binarytree.H"
#include "../drt_mortar/mortar_element.H"
#include "../drt_mortar/mortar_dofset.H"
#include "../drt_mortar/mortar_integrator.H"
#include "../drt_mortar/mortar_defines.H"
#include "../linalg/linalg_utils.H"
#include "../drt_lib/drt_globalproblem.H"
#include "../drt_inpar/inpar_contact.H"
#include "../drt_io/io_gmsh.H"
#include "../drt_io/io_control.H"



/*----------------------------------------------------------------------*
 |  Visualize contact stuff with gmsh                         popp 08/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::VisualizeGmsh(const int step, const int iter)
{
  // get out of here if not participating in interface
  if (!lComm()) return;
  
  //**********************************************************************
  // GMSH output of all interface elements
  //**********************************************************************
  // construct unique filename for gmsh output
  // first index = time step index
  std::ostringstream filename;
  const std::string filebase = DRT::Problem::Instance()->OutputControlFile()->FileName();
  filename << "o/gmsh_output/" << filebase << "_";
  if (step<10)
    filename << 0 << 0 << 0 << 0;
  else if (step<100)
    filename << 0 << 0 << 0;
  else if (step<1000)
    filename << 0 << 0;
  else if (step<10000)
    filename << 0;
  else if (step>99999)
    dserror("Gmsh output implemented for a maximum of 99.999 time steps");
  filename << step;

  // construct unique filename for gmsh output
  // second index = Newton iteration index
#ifdef MORTARGMSH2
  filename << "_";
  if (iter<10)
    filename << 0;
  else if (iter>99)
    dserror("Gmsh output implemented for a maximum of 99 iterations");
  filename << iter;
#endif // #ifdef MORTARGMSH2

#ifdef MORTARGMSH3
  filename << "_";
  if (iter<10)
    filename << 0;
  else if (iter>99)
    dserror("Gmsh output implemented for a maximum of 99 iterations");
  filename << iter;
#endif // #ifdef MORTARGMSH3

  // create three files (slave, master and whole interface)
  std::ostringstream filenameslave;
  std::ostringstream filenamemaster;
  filenameslave << filename.str();
  filenamemaster << filename.str();
  filename << "_iface.pos";
  filenameslave << "_slave.pos";
  filenamemaster << "_master.pos";

  // do output to file in c-style
  FILE* fp = NULL;
  FILE* fps = NULL;
  FILE* fpm = NULL;

  //**********************************************************************
  // Start GMSH output
  //**********************************************************************
  for (int proc=0;proc<lComm()->NumProc();++proc)
  {
    if (proc==lComm()->MyPID())
    {
      // open files (overwrite if proc==0, else append)
      if (proc==0)
      {
        fp = fopen(filename.str().c_str(), "w");
        fps = fopen(filenameslave.str().c_str(), "w");
        fpm = fopen(filenamemaster.str().c_str(), "w");
      }
      else 
      {
        fp = fopen(filename.str().c_str(), "a");
        fps = fopen(filenameslave.str().c_str(), "a");
        fpm = fopen(filenamemaster.str().c_str(), "a");
      }

      // write output to temporary stringstream
      std::stringstream gmshfilecontent;
      std::stringstream gmshfilecontentslave;
      std::stringstream gmshfilecontentmaster;
      if (proc==0)
      {
        gmshfilecontent << "View \" Step " << step << " Iter " << iter << " Iface\" {" << endl;
        gmshfilecontentslave << "View \" Step " << step << " Iter " << iter << " Slave\" {" << endl;
        gmshfilecontentmaster << "View \" Step " << step << " Iter " << iter << " Master\" {" << endl;
      }

      //******************************************************************
      // plot elements
      //******************************************************************
      for (int i=0; i<idiscret_->NumMyRowElements(); ++i)
      {
        MORTAR::MortarElement* element = static_cast<MORTAR::MortarElement*>(idiscret_->lRowElement(i));
        int nnodes = element->NumNode();
        LINALG::SerialDenseMatrix coord(3,nnodes);
        element->GetNodalCoords(coord);
        double color = (double)element->Owner();

        //local center
        double xi[2] = {0.0, 0.0};

        // 2D linear case (2noded line elements)
        if (element->Shape()==DRT::Element::line2)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "SL(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "};" << endl;
            gmshfilecontentslave << "SL(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "};" << endl;
          }
          else
          {
            gmshfilecontent << "SL(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "SL(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "};" << endl;
          }
          
        }

        // 2D quadratic case (3noded line elements)
        if (element->Shape()==DRT::Element::line3)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "SL2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "SL2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "};" << endl;
          }
          else
          {
            gmshfilecontent << "SL2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "SL2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "};" << endl;
          }
        }

        // 3D linear case (3noded triangular elements)
        if (element->Shape()==DRT::Element::tri3)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "};" << endl;
          }
          else
          {
            gmshfilecontent << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "};" << endl;
          }
          xi[0] = 1.0/3; xi[1] = 1.0/3;
        }

        // 3D bilinear case (4noded quadrilateral elements)
        if (element->Shape()==DRT::Element::quad4)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "SQ(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "SQ(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
          }
          else
          {
            gmshfilecontent << "SQ(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "SQ(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
          }
        }

        // 3D quadratic case (6noded triangular elements)
        if (element->Shape()==DRT::Element::tri6)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "ST2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color <<"};" << endl;
            gmshfilecontentslave << "ST2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color <<"};" << endl;
          }
          else
          {
            gmshfilecontent << "ST2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color <<"};" << endl;
            gmshfilecontentmaster << "ST2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color <<"};" << endl;
          }
          xi[0] = 1.0/3; xi[1] = 1.0/3;
        }

        // 3D serendipity case (8noded quadrilateral elements)
        if (element->Shape()==DRT::Element::quad8)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "ST(" << scientific << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "ST(" << scientific << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "ST(" << scientific << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "SQ(" << scientific << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "ST(" << scientific << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "ST(" << scientific << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "ST(" << scientific << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "SQ(" << scientific << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
          }
          else
          {
            gmshfilecontent << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "ST(" << scientific << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "ST(" << scientific << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "ST(" << scientific << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontent << "SQ(" << scientific << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "ST(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "ST(" << scientific << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "ST(" << scientific << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "ST(" << scientific << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "SQ(" << scientific << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "," << color << "};" << endl;
          }
        }

        // 3D biquadratic case (9noded quadrilateral elements)
        if (element->Shape()==DRT::Element::quad9)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "SQ2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,8) << "," << coord(1,8) << ","
                                << coord(2,8) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentslave << "SQ2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,8) << "," << coord(1,8) << ","
                                << coord(2,8) << ")";
            gmshfilecontentslave << "{" << scientific << color << "," << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color << "," << color << "," << color << "};" << endl;
          }
          else
          {
            gmshfilecontent << "SQ2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,8) << "," << coord(1,8) << ","
                                << coord(2,8) << ")";
            gmshfilecontent << "{" << scientific << color << "," << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color << "," << color << "," << color << "};" << endl;
            gmshfilecontentmaster << "SQ2(" << scientific << coord(0,0) << "," << coord(1,0) << ","
                                << coord(2,0) << "," << coord(0,1) << "," << coord(1,1) << ","
                                << coord(2,1) << "," << coord(0,2) << "," << coord(1,2) << ","
                                << coord(2,2) << "," << coord(0,3) << "," << coord(1,3) << ","
                                << coord(2,3) << "," << coord(0,4) << "," << coord(1,4) << ","
                                << coord(2,4) << "," << coord(0,5) << "," << coord(1,5) << ","
                                << coord(2,5) << "," << coord(0,6) << "," << coord(1,6) << ","
                                << coord(2,6) << "," << coord(0,7) << "," << coord(1,7) << ","
                                << coord(2,7) << "," << coord(0,8) << "," << coord(1,8) << ","
                                << coord(2,8) << ")";
            gmshfilecontentmaster << "{" << scientific << color << "," << color << "," << color << "," << color << ","
                            << color << "," << color << "," << color << "," << color << "," << color << "};" << endl;
          }
        }

        // plot element number in element center
        double elec[3];
        element->LocalToGlobal(xi,elec,0);
        
        if (element->IsSlave())
        {
          gmshfilecontent << "T3(" << scientific << elec[0] << "," << elec[1] << "," << elec[2] << "," << 17 << ")";
          gmshfilecontent << "{" << "S" << element->Id() << "};" << endl;
          gmshfilecontentslave << "T3(" << scientific << elec[0] << "," << elec[1] << "," << elec[2] << "," << 17 << ")";
          gmshfilecontentslave << "{" << "S" << element->Id() << "};" << endl;
        }
        else
        {
          gmshfilecontent << "T3(" << scientific << elec[0] << "," << elec[1] << "," << elec[2] << "," << 17 << ")";
          gmshfilecontent << "{" << "M" << element->Id() << "};" << endl;
          gmshfilecontentmaster << "T3(" << scientific << elec[0] << "," << elec[1] << "," << elec[2] << "," << 17 << ")";
          gmshfilecontentmaster << "{" << "M" << element->Id() << "};" << endl;
        }

        // plot node numbers at the nodes
        for (int j=0;j<nnodes;++j)
        {
          if (element->IsSlave())
          {
            gmshfilecontent << "T3(" << scientific << coord(0,j) << "," << coord(1,j) << "," << coord(2,j) << "," << 17 << ")";
            gmshfilecontent << "{" << "SN" << element->NodeIds()[j] << "};" << endl;
            gmshfilecontentslave << "T3(" << scientific << coord(0,j) << "," << coord(1,j) << "," << coord(2,j) << "," << 17 << ")";
            gmshfilecontentslave << "{" << "SN" << element->NodeIds()[j] << "};" << endl;
          }
          else
          {
            gmshfilecontent << "T3(" << scientific << coord(0,j) << "," << coord(1,j) << "," << coord(2,j) << "," << 17 << ")";
            gmshfilecontent << "{" << "MN" << element->NodeIds()[j] << "};" << endl;
            gmshfilecontentmaster << "T3(" << scientific << coord(0,j) << "," << coord(1,j) << "," << coord(2,j) << "," << 17 << ")";
            gmshfilecontentmaster << "{" << "MN" << element->NodeIds()[j] << "};" << endl;
          }
        }
      }

      //******************************************************************
      // plot normal vector, tangent vectors and contact status
      //******************************************************************
      for (int i=0; i<snoderowmap_->NumMyElements(); ++i)
      {
        int gid = snoderowmap_->GID(i);
        DRT::Node* node = idiscret_->gNode(gid);
        if (!node) dserror("ERROR: Cannot find node with gid %",gid);
        CoNode* cnode = static_cast<CoNode*>(node);
        if (!cnode) dserror("ERROR: Static Cast to CoNode* failed");

        double nc[3];
        double nn[3];
        double nt1[3];
        double nt2[3];

        for (int j=0;j<3;++j)
        {
          nc[j]=cnode->xspatial()[j];
          nn[j]=cnode->MoData().n()[j];
          nt1[j]=cnode->CoData().txi()[j];
          nt2[j]=cnode->CoData().teta()[j];
        }

        //******************************************************************
        // plot normal and tangent vectors
        //******************************************************************
        gmshfilecontentslave << "VP(" << scientific << nc[0] << "," << nc[1] << "," << nc[2] << ")";
        gmshfilecontentslave << "{" << scientific << nn[0] << "," << nn[1] << "," << nn[2] << "};" << endl;

        if (friction_)
        {
          gmshfilecontentslave << "VP(" << scientific << nc[0] << "," << nc[1] << "," << nc[2] << ")";
          gmshfilecontentslave << "{" << scientific << nt1[0] << "," << nt1[1] << "," << nt1[2] << "};" << endl;
          gmshfilecontentslave << "VP(" << scientific << nc[0] << "," << nc[1] << "," << nc[2] << ")";
          gmshfilecontentslave << "{" << scientific << nt2[0] << "," << nt2[1] << "," << nt2[2] << "};" << endl;
          
        }

        //******************************************************************
        // plot contact status of slave nodes (inactive, active, stick, slip)
        //******************************************************************
        // frictionless contact, active node = {A}
        if (!friction_ && cnode->Active())
        {
          gmshfilecontentslave << "T3(" << scientific << nc[0] << "," << nc[1] << "," << nc[2] << "," << 17 << ")";
          gmshfilecontentslave << "{" << "A" << "};" << endl;
        }

        // frictionless contact, inactive node = { }
        else if (!friction_ && !cnode->Active())
        {
          //do nothing
        }

        // frictional contact, slip node = {G}
        else if (friction_ && cnode->Active())
        {
          if (static_cast<FriNode*>(cnode)->FriData().Slip())
          {  
            gmshfilecontentslave << "T3(" << scientific << nc[0] << "," << nc[1] << "," << nc[2] << "," << 17 << ")";
            gmshfilecontentslave << "{" << "G" << "};" << endl;
          }
          else 
          {
            gmshfilecontentslave << "T3(" << scientific << nc[0] << "," << nc[1] << "," << nc[2] << "," << 17 << ")";
            gmshfilecontentslave << "{" << "H" << "};" << endl;
          }
        }
      }

      // end GMSH output section in all files
      if (proc==lComm()->NumProc()-1)
      {
        gmshfilecontent << "};" << endl;
        gmshfilecontentslave << "};" << endl;
        gmshfilecontentmaster << "};" << endl;
      }

      // move everything to gmsh post-processing files and close them
      fprintf(fp,gmshfilecontent.str().c_str());
      fprintf(fps,gmshfilecontentslave.str().c_str());
      fprintf(fpm,gmshfilecontentmaster.str().c_str());
      fclose(fp);
      fclose(fps);
      fclose(fpm);
    }
    lComm()->Barrier();
  }

  
  //**********************************************************************
  // GMSH output of all treenodes (DOPs) on all layers
  //**********************************************************************
#ifdef MORTARGMSHTN
  // get max. number of layers for every proc.
  // (master elements are equal on each proc)
  int lnslayers=binarytree_->Streenodesmap().size();
  int gnmlayers=binarytree_->Mtreenodesmap().size();
  int gnslayers = 0;
  lComm()->MaxAll(&lnslayers, &gnslayers,1);

  // create files for visualization of slave dops for every layer
  std::ostringstream filenametn;
  const std::string filebasetn = DRT::Problem::Instance()->OutputControlFile()->FileName();
  filenametn << "o/gmsh_output/" << filebasetn << "_";

  if (step<10)
    filenametn << 0 << 0 << 0 << 0;
  else if (step<100)
    filenametn << 0 << 0 << 0;
  else if (step<1000)
    filenametn << 0 << 0;
  else if (step<10000)
    filenametn << 0;
  else if (step>99999)
    dserror("Gmsh output implemented for a maximum of 99.999 time steps");
  filenametn << step;

  // construct unique filename for gmsh output
  // second index = Newton iteration index
#ifdef MORTARGMSH2
  filenametn << "_";
  if (iter<10)
    filenametn << 0;
  else if (iter>99)
    dserror("Gmsh output implemented for a maximum of 99 iterations");
  filenametn << iter;
#endif // #ifdef MORTARGMSH2

#ifdef MORTARGMSH3
  filenametn << "_";
  if (iter<10)
    filenametn << 0;
  else if (iter>99)
    dserror("Gmsh output implemented for a maximum of 99 iterations");
  filenametn << iter;
#endif // #ifdef MORTARGMSH3
  
  if (lComm()->MyPID()==0)
  {
    for (int i=0; i<gnslayers;i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_s_tnlayer_" <<  i << ".pos";
      //cout << endl << lComm()->MyPID()<< "filename: " << currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "w");
      std::stringstream gmshfile;
      gmshfile << "View \" Step " << step << " Iter " << iter << " stl " << i << " \" {" << endl;
      fprintf(fp,gmshfile.str().c_str());
      fclose(fp);
    }
  }

  lComm()->Barrier();

  // for every proc, one after another, put data of slabs into files
  for (int i=0;i<lComm()->NumProc();i++)
  {
    if ((i==lComm()->MyPID())&&(binarytree_->Sroot()->Type()!=4))
    {
      //print full tree with treenodesmap
      for (int j=0;j < (int)binarytree_->Streenodesmap().size();j++)
      {
        for (int k=0;k < (int)binarytree_->Streenodesmap()[j].size();k++)
        {
          //if proc !=0 and first treenode to plot->create new sheet in gmsh
          if (i!=0 && k==0)
          {
            //create new sheet "Treenode" in gmsh
            std::ostringstream currentfilename;
            currentfilename << filenametn.str().c_str() << "_s_tnlayer_" <<  j << ".pos";
            fp = fopen(currentfilename.str().c_str(), "a");
            std::stringstream gmshfile;
            gmshfile << "};" << endl << "View \" Treenode \" { " << endl;
            fprintf(fp,gmshfile.str().c_str());
            fclose(fp);
          }
          //cout << endl << "plot streenode level: " << j << "treenode: " << k;
          std::ostringstream currentfilename;
          currentfilename << filenametn.str().c_str() << "_s_tnlayer_" <<  j << ".pos";
          binarytree_->Streenodesmap()[j][k]->PrintDopsForGmsh(currentfilename.str().c_str());

          //if there is another treenode to plot
          if (k<((int)binarytree_->Streenodesmap()[j].size()-1))
          {
            //create new sheet "Treenode" in gmsh
            std::ostringstream currentfilename;
            currentfilename << filenametn.str().c_str() << "_s_tnlayer_" <<  j << ".pos";
            fp = fopen(currentfilename.str().c_str(), "a");
            std::stringstream gmshfile;
            gmshfile << "};" << endl << "View \" Treenode \" { " << endl;
            fprintf(fp,gmshfile.str().c_str());
            fclose(fp);
          }
        }
      }
    }

    lComm()->Barrier();
  }

  lComm()->Barrier();
  //close all slave-gmsh files
  if (lComm()->MyPID()==0)
  {
    for (int i=0; i<gnslayers;i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_s_tnlayer_" << i << ".pos";
      //cout << endl << lComm()->MyPID()<< "current filename: " << currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "a");
      std::stringstream gmshfilecontent;
      gmshfilecontent  << "};" ;
      fprintf(fp,gmshfilecontent.str().c_str());
      fclose(fp);
    }
  }
  lComm()->Barrier();

  // create master slabs
  if (lComm()->MyPID()==0)
  {
    for (int i=0; i<gnmlayers;i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_m_tnlayer_" <<  i << ".pos";
      //cout << endl << lComm()->MyPID()<< "filename: " << currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "w");
      std::stringstream gmshfile;
      gmshfile << "View \" Step " << step << " Iter " << iter << " mtl " << i << " \" {" << endl;
      fprintf(fp,gmshfile.str().c_str());
      fclose(fp);
    }

    //print full tree with treenodesmap
    for (int j=0;j < (int)binarytree_->Mtreenodesmap().size();j++)
    {
      for (int k=0;k < (int)binarytree_->Mtreenodesmap()[j].size();k++)
      {
        std::ostringstream currentfilename;
        currentfilename << filenametn.str().c_str() << "_m_tnlayer_" <<  j << ".pos";
        binarytree_->Mtreenodesmap()[j][k]->PrintDopsForGmsh(currentfilename.str().c_str());

        //if there is another treenode to plot
        if (k<((int)binarytree_->Mtreenodesmap()[j].size()-1))
        {
          //create new sheet "Treenode" in gmsh
          std::ostringstream currentfilename;
          currentfilename << filenametn.str().c_str() << "_m_tnlayer_" <<  j << ".pos";
          fp = fopen(currentfilename.str().c_str(), "a");
          std::stringstream gmshfile;
          gmshfile << "};" << endl << "View \" Treenode \" { " << endl;
          fprintf(fp,gmshfile.str().c_str());
          fclose(fp);
        }
      }
    }

    //close all master files
    for (int i=0; i<gnmlayers;i++)
    {
      std::ostringstream currentfilename;
      currentfilename << filenametn.str().c_str() << "_m_tnlayer_" << i << ".pos";
      fp = fopen(currentfilename.str().c_str(), "a");
      std::stringstream gmshfilecontent;
      gmshfilecontent << endl << "};";
      fprintf(fp,gmshfilecontent.str().c_str());
      fclose(fp);
    }
  }
#endif // ifdef MORTARGMSHTN

  
  //**********************************************************************
  // GMSH output of all active treenodes (DOPs) on leaf level
  //**********************************************************************
#ifdef MORTARGMSHCTN
  std::ostringstream filenamectn;
  const std::string filebasectn = DRT::Problem::Instance()->OutputControlFile()->FileName();
  filenamectn << "o/gmsh_output/" << filebasectn << "_";
  if (step<10)
    filenamectn << 0 << 0 << 0 << 0;
  else if (step<100)
    filenamectn << 0 << 0 << 0;
  else if (step<1000)
    filenamectn << 0 << 0;
  else if (step<10000)
    filenamectn << 0;
  else if (step>99999)
    dserror("Gmsh output implemented for a maximum of 99.999 time steps");
  filenamectn << step;

  // construct unique filename for gmsh output
  // second index = Newton iteration index
#ifdef MORTARGMSH2
  filenamectn << "_";
  if (iter<10)
    filenamectn << 0;
  else if (iter>99)
    dserror("Gmsh output implemented for a maximum of 99 iterations");
  filenamectn << iter;
#endif // #ifdef MORTARGMSH2

  int lcontactmapsize=(int)(binarytree_->CouplingMap()[0].size());
  int gcontactmapsize;

  lComm()->MaxAll(&lcontactmapsize, &gcontactmapsize,1);

  if (gcontactmapsize>0)
  {
    // open/create new file
    if (lComm()->MyPID()==0)
    {
      std::ostringstream currentfilename;
      currentfilename << filenamectn.str().c_str() << "_ct.pos";
      //cout << endl << lComm()->MyPID()<< "filename: " << currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "w");
      std::stringstream gmshfile;
      gmshfile << "View \" Step " << step << " Iter " << iter << " contacttn  \" {" << endl;
      fprintf(fp,gmshfile.str().c_str());
      fclose(fp);
    }

    // every proc should plot its contacting treenodes!
    for (int i=0;i<lComm()->NumProc();i++)
    {
      if (lComm()->MyPID()==i)
      {
        if ( (int)(binarytree_->CouplingMap()[0]).size() != (int)(binarytree_->CouplingMap()[1]).size() )
        dserror("ERROR: Binarytree CouplingMap does not have right size!");

        for (int j=0; j<(int)((binarytree_->CouplingMap()[0]).size());j++)
        {
          std::ostringstream currentfilename;
          std::stringstream gmshfile;
          std::stringstream newgmshfile;
          
          // create new sheet for slave
          if (lComm()->MyPID()==0 && j==0)
          {
            currentfilename << filenamectn.str().c_str() << "_ct.pos";
            fp = fopen(currentfilename.str().c_str(), "w");
            gmshfile << "View \" Step " << step << " Iter " << iter << " CS  \" {" << endl;
            fprintf(fp,gmshfile.str().c_str());
            fclose(fp);     
            (binarytree_->CouplingMap()[0][j])->PrintDopsForGmsh(currentfilename.str().c_str());
          }
          else
          {
            currentfilename << filenamectn.str().c_str() << "_ct.pos";
            fp = fopen(currentfilename.str().c_str(), "a");
            gmshfile << "};" << endl << "View \" Step " << step << " Iter " << iter << " CS  \" {" << endl;
            fprintf(fp,gmshfile.str().c_str());
            fclose(fp);     
            (binarytree_->CouplingMap()[0][j])->PrintDopsForGmsh(currentfilename.str().c_str());
          }
          
          // create new sheet for master
          fp = fopen(currentfilename.str().c_str(), "a");
          newgmshfile << "};" << endl << "View \" Step " << step << " Iter " << iter << " CM  \" {" << endl;
          fprintf(fp,newgmshfile.str().c_str());
          fclose(fp);
          (binarytree_->CouplingMap()[1][j])->PrintDopsForGmsh(currentfilename.str().c_str());
        }
      }
      lComm()->Barrier();
    }

    //close file
    if (lComm()->MyPID()==0)
    {
      std::ostringstream currentfilename;
      currentfilename << filenamectn.str().c_str() << "_ct.pos";
      //cout << endl << lComm()->MyPID()<< "filename: " << currentfilename.str().c_str();
      fp = fopen(currentfilename.str().c_str(), "a");
      std::stringstream gmshfile;
      gmshfile  << "};" ;
      fprintf(fp,gmshfile.str().c_str());
      fclose(fp);
    }
  }
#endif //MORTARGMSHCTN

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for normal/tangent deriv.          popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckNormalDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // create storage for normals / tangents
  vector<double> refnx(int(snodecolmapbound_->NumMyElements()));
  vector<double> refny(int(snodecolmapbound_->NumMyElements()));
  vector<double> refnz(int(snodecolmapbound_->NumMyElements()));
  vector<double> newnx(int(snodecolmapbound_->NumMyElements()));
  vector<double> newny(int(snodecolmapbound_->NumMyElements()));
  vector<double> newnz(int(snodecolmapbound_->NumMyElements()));

  vector<double> reftxix(int(snodecolmapbound_->NumMyElements()));
  vector<double> reftxiy(int(snodecolmapbound_->NumMyElements()));
  vector<double> reftxiz(int(snodecolmapbound_->NumMyElements()));
  vector<double> newtxix(int(snodecolmapbound_->NumMyElements()));
  vector<double> newtxiy(int(snodecolmapbound_->NumMyElements()));
  vector<double> newtxiz(int(snodecolmapbound_->NumMyElements()));

  vector<double> reftetax(int(snodecolmapbound_->NumMyElements()));
  vector<double> reftetay(int(snodecolmapbound_->NumMyElements()));
  vector<double> reftetaz(int(snodecolmapbound_->NumMyElements()));
  vector<double> newtetax(int(snodecolmapbound_->NumMyElements()));
  vector<double> newtetay(int(snodecolmapbound_->NumMyElements()));
  vector<double> newtetaz(int(snodecolmapbound_->NumMyElements()));

  // compute and print all nodal normals / derivatives (reference)
  for(int j=0; j<snodecolmapbound_->NumMyElements();++j)
  {
    int jgid = snodecolmapbound_->GID(j);
    DRT::Node* jnode = idiscret_->gNode(jgid);
    if (!jnode) dserror("ERROR: Cannot find node with gid %",jgid);
    CoNode* jcnode = static_cast<CoNode*>(jnode);

    typedef map<int,double>::const_iterator CI;

    // print reference data
    cout << endl << "Node: " << jcnode->Id() << "  Owner: " << jcnode->Owner() << endl;

    cout << "Normal-derivative-maps: " << endl;
    cout << "Row dof id: " << jcnode->Dofs()[0] << endl;
    for (CI p=(jcnode->CoData().GetDerivN()[0]).begin();p!=(jcnode->CoData().GetDerivN()[0]).end();++p)
      cout << p->first << '\t' << p->second << endl;
    cout << "Row dof id: " << jcnode->Dofs()[1] << endl;
    for (CI p=(jcnode->CoData().GetDerivN()[1]).begin();p!=(jcnode->CoData().GetDerivN()[1]).end();++p)
      cout << p->first << '\t' << p->second << endl;
    cout << "Row dof id: " << jcnode->Dofs()[2] << endl;
    for (CI p=(jcnode->CoData().GetDerivN()[2]).begin();p!=(jcnode->CoData().GetDerivN()[2]).end();++p)
      cout << p->first << '\t' << p->second << endl;

    cout << "Tangent txi-derivative-maps: " << endl;
    cout << "Row dof id: " << jcnode->Dofs()[0] << endl;
    for (CI p=(jcnode->CoData().GetDerivTxi()[0]).begin();p!=(jcnode->CoData().GetDerivTxi()[0]).end();++p)
      cout << p->first << '\t' << p->second << endl;
    cout << "Row dof id: " << jcnode->Dofs()[1] << endl;
    for (CI p=(jcnode->CoData().GetDerivTxi()[1]).begin();p!=(jcnode->CoData().GetDerivTxi()[1]).end();++p)
      cout << p->first << '\t' << p->second << endl;
    cout << "Row dof id: " << jcnode->Dofs()[2] << endl;
    for (CI p=(jcnode->CoData().GetDerivTxi()[2]).begin();p!=(jcnode->CoData().GetDerivTxi()[2]).end();++p)
      cout << p->first << '\t' << p->second << endl;

    cout << "Tangent teta-derivative-maps: " << endl;
    cout << "Row dof id: " << jcnode->Dofs()[0] << endl;
    for (CI p=(jcnode->CoData().GetDerivTeta()[0]).begin();p!=(jcnode->CoData().GetDerivTeta()[0]).end();++p)
      cout << p->first << '\t' << p->second << endl;
    cout << "Row dof id: " << jcnode->Dofs()[1] << endl;
    for (CI p=(jcnode->CoData().GetDerivTeta()[1]).begin();p!=(jcnode->CoData().GetDerivTeta()[1]).end();++p)
      cout << p->first << '\t' << p->second << endl;
    cout << "Row dof id: " << jcnode->Dofs()[2] << endl;
    for (CI p=(jcnode->CoData().GetDerivTeta()[2]).begin();p!=(jcnode->CoData().GetDerivTeta()[2]).end();++p)
      cout << p->first << '\t' << p->second << endl;

    // store reference normals / tangents
    refnx[j] = jcnode->MoData().n()[0];
    refny[j] = jcnode->MoData().n()[1];
    refnz[j] = jcnode->MoData().n()[2];
    reftxix[j] = jcnode->CoData().txi()[0];
    reftxiy[j] = jcnode->CoData().txi()[1];
    reftxiz[j] = jcnode->CoData().txi()[2];
    reftetax[j] = jcnode->CoData().teta()[0];
    reftetay[j] = jcnode->CoData().teta()[1];
    reftetaz[j] = jcnode->CoData().teta()[2];
  }
  
  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  for (int i=0; i<3*snodefullmap->NumMyElements();++i)
  {
    // reset normal etc.
    Initialize();

    // compute element areas
    SetElementAreas();

    // now finally get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(i/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof(l): " << i%3
         << " Dof(g): " << snode->Dofs()[i%3] << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (i%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (i%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // compute finite difference derivative
    for(int k=0; k<snodecolmapbound_->NumMyElements();++k)
    {
      int kgid = snodecolmapbound_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      // build NEW averaged normal at each slave node
      kcnode->BuildAveragedNormal();

      newnx[k] = kcnode->MoData().n()[0];
      newny[k] = kcnode->MoData().n()[1];
      newnz[k] = kcnode->MoData().n()[2];
      newtxix[k] = kcnode->CoData().txi()[0];
      newtxiy[k] = kcnode->CoData().txi()[1];
      newtxiz[k] = kcnode->CoData().txi()[2];
      newtetax[k] = kcnode->CoData().teta()[0];
      newtetay[k] = kcnode->CoData().teta()[1];
      newtetaz[k] = kcnode->CoData().teta()[2];

      // get reference normal / tangent
      double refn[3] = {0.0, 0.0, 0.0};
      double reftxi[3] = {0.0, 0.0, 0.0};
      double refteta[3] = {0.0, 0.0, 0.0};
      refn[0] = refnx[k];
      refn[1] = refny[k];
      refn[2] = refnz[k];
      reftxi[0] = reftxix[k];
      reftxi[1] = reftxiy[k];
      reftxi[2] = reftxiz[k];
      refteta[0] = reftetax[k];
      refteta[1] = reftetay[k];
      refteta[2] = reftetaz[k];

      // get modified normal / tangent
      double newn[3] = {0.0, 0.0, 0.0};
      double newtxi[3] = {0.0, 0.0, 0.0};
      double newteta[3] = {0.0, 0.0, 0.0};
      newn[0] = newnx[k];
      newn[1] = newny[k];
      newn[2] = newnz[k];
      newtxi[0] = newtxix[k];
      newtxi[1] = newtxiy[k];
      newtxi[2] = newtxiz[k];
      newteta[0] = newtetax[k];
      newteta[1] = newtetay[k];
      newteta[2] = newtetaz[k];

      // print results (derivatives) to screen
      if (abs(newn[0]-refn[0])>1e-12 || abs(newn[1]-refn[1])>1e-12 || abs(newn[2]-refn[2])>1e-12)
      {
        cout << "Node: " << kcnode->Id() << "  Owner: " << kcnode->Owner() << endl;
        cout << "Normal derivative (FD):" << endl;
        if (abs(newn[0]-refn[0])>1e-12)
        {
          double val = (newn[0]-refn[0])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[0] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }

        if (abs(newn[1]-refn[1])>1e-12)
        {
          double val = (newn[1]-refn[1])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[1] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }

        if (abs(newn[2]-refn[2])>1e-12)
        {
          double val = (newn[2]-refn[2])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[2] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }
      }

      if (abs(newtxi[0]-reftxi[0])>1e-12 || abs(newtxi[1]-reftxi[1])>1e-12 | abs(newtxi[2]-reftxi[2])>1e-12)
      {
        cout << "Node: " << kcnode->Id() << "  Owner: " << kcnode->Owner() << endl;
        cout << "Tangent txi derivative (FD):" << endl;
        if (abs(newtxi[0]-reftxi[0])>1e-12)
        {
          double val = (newtxi[0]-reftxi[0])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[0] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }

        if (abs(newtxi[1]-reftxi[1])>1e-12)
        {
          double val = (newtxi[1]-reftxi[1])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[1] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }

        if (abs(newtxi[2]-reftxi[2])>1e-12)
        {
          double val = (newtxi[2]-reftxi[2])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[2] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }
      }

      if (abs(newteta[0]-refteta[0])>1e-12 || abs(newteta[1]-refteta[1])>1e-12 | abs(newteta[2]-refteta[2])>1e-12)
      {
        cout << "Node: " << kcnode->Id() << "  Owner: " << kcnode->Owner() << endl;
        cout << "Tangent teta derivative (FD):" << endl;
        if (abs(newteta[0]-refteta[0])>1e-12)
        {
          double val = (newteta[0]-refteta[0])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[0] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }

        if (abs(newteta[1]-refteta[1])>1e-12)
        {
          double val = (newteta[1]-refteta[1])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[1] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }

        if (abs(newteta[2]-refteta[2])>1e-12)
        {
          double val = (newteta[2]-refteta[2])/delta;
          cout << "Row dof id: " << kcnode->Dofs()[2] << endl;
          cout << snode->Dofs()[i%3] << '\t' << val << endl;
        }
      }
    }

    // undo finite difference modification
    if (i%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (i%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;

    }
  }

  // back to normal...
  // reset normal etc.
  Initialize();

  // compute element areas
  SetElementAreas();
  
  // contents of Evaluate()
  Evaluate();
  
  return;
}


/*----------------------------------------------------------------------*
 | Finite difference check for D-Mortar derivatives           popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckMortarDDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // create storage for D-Matrix entries
  map<int, map<int,double> > refD; // stores dof-wise the entries of D
  map<int, map<int,double> > newD;

  map<int, map<int, map<int,double> > > refDerivD; // stores old derivm for every node

  // print reference to screen (D-derivative-maps) and store them for later comparison
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements(); ++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find node with gid %",gid);
    CoNode* cnode = static_cast<CoNode*>(node);

    int dim = cnode->NumDof();

    typedef map<int,map<int,double> >::const_iterator CID;
    typedef map<int,double>::const_iterator CI;

    if ((int)(cnode->MoData().GetD().size())==0)
      continue;

    for( int d=0; d<dim; d++ )
    {
      int dof = cnode->Dofs()[d];
      refD[dof] = cnode->MoData().GetD()[d];
    }

    refDerivD[gid] = cnode->CoData().GetDerivD();
  }

  // global loop to apply FD scheme to all SLAVE dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements(); ++fd)
  {
    // store warnings for this finite difference
    int w=0;

    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    int sdof = snode->Dofs()[fd%3];

    cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements(); ++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode)
        dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      int dim = kcnode->NumDof();

      if ((int)(kcnode->MoData().GetD().size())==0)
        continue;

      typedef map<int,double>::const_iterator CI;

      for( int d=0; d<dim; d++ )
      {
        int dof = kcnode->Dofs()[d];

        // store D-values into refD
        newD[dof] = kcnode->MoData().GetD()[d];

        // print results (derivatives) to screen
        for (CI p=newD[dof].begin(); p!=newD[dof].end(); ++p)
        {
          if (abs(newD[dof][p->first]-refD[dof][p->first]) > 1e-12)
          {
            double finit = (newD[dof][p->first]-refD[dof][p->first])/delta;
            double analy = ((refDerivD[kgid])[(p->first)/Dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            cout << "(" << dof << "," << (p->first)/Dim() << "," << sdof << ") : fd=" << finit << " derivd=" << analy << " DEVIATION " << dev;

            if( abs(dev) > 1e-4 )
            {
              cout << " ***** WARNING ***** ";
              w++;
            }
            else if( abs(dev) > 1e-5 )
            {
              cout << " ***** warning ***** ";
              w++;
            }

            cout << endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    cout << " ******************** GENERATED " << w << " WARNINGS ***************** " << endl;
  }

  // global loop to apply FD scheme to all MASTER dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements(); ++fd)
  {
    // store warnings for this finite difference
    int w=0;

    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* mnode = static_cast<CoNode*>(node);

    int mdof = mnode->Dofs()[fd%3];

    cout << "\nDERIVATIVE FOR M-NODE # " << gid << " DOF: " << mdof << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements(); ++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode)
        dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      int dim = kcnode->NumDof();

      if ((int)(kcnode->MoData().GetD().size())==0)
        continue;

      typedef map<int,double>::const_iterator CI;

      for( int d=0; d<dim; d++ )
      {
        int dof = kcnode->Dofs()[d];

        // store D-values into refD
        newD[dof] = kcnode->MoData().GetD()[d];

        // print results (derivatives) to screen
        for (CI p=newD[dof].begin(); p!=newD[dof].end(); ++p)
        {
          if (abs(newD[dof][p->first]-refD[dof][p->first]) > 1e-12)
          {
            double finit = (newD[dof][p->first]-refD[dof][p->first])/delta;
            double analy = ((refDerivD[kgid])[(p->first)/Dim()])[mdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            cout << "(" << dof << "," << (p->first)/Dim() << "," << mdof << ") : fd=" << finit << " derivd=" << analy << " DEVIATION " << dev;

            if( abs(dev) > 1e-4 )
            {
              cout << " ***** WARNING ***** ";
              w++;
            }
            else if( abs(dev) > 1e-5 )
            {
              cout << " ***** warning ***** ";
              w++;
            }

            cout << endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }

    cout << " ******************** GENERATED " << w << " WARNINGS ***************** " << endl;
  }

  // back to normal...

  // Initialize
  Initialize();

  // compute element areas
  SetElementAreas();

  // *******************************************************************
  // contents of Evaluate()
  // *******************************************************************
  Evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for M-Mortar derivatives           popp 05/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckMortarMDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // create storage for M-Matrix entries
  map<int, map<int,double> > refM; // stores dof-wise the entries of M
  map<int, map<int,double> > newM;

  map<int, map<int, map<int,double> > > refDerivM; // stores old derivm for every node

  // print reference to screen (M-derivative-maps) and store them for later comparison
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements(); ++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find node with gid %",gid);
    CoNode* cnode = static_cast<CoNode*>(node);

    int dim = cnode->NumDof();

    typedef map<int,map<int,double> >::const_iterator CIM;
    typedef map<int,double>::const_iterator CI;

    if ((int)(cnode->MoData().GetM().size())==0)
      continue;

    for( int d=0; d<dim; d++ )
    {
      int dof = cnode->Dofs()[d];
      refM[dof] = cnode->MoData().GetM()[d];
    }

    refDerivM[gid] = cnode->CoData().GetDerivM();
  }

  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements(); ++fd)
  {
    // store warnings for this finite difference
    int w=0;

    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    int sdof = snode->Dofs()[fd%3];

    cout << "\nDERIVATIVE FOR S-NODE # " << gid << " DOF: " << sdof << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements(); ++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode)
        dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      int dim = kcnode->NumDof();

      if ((int)(kcnode->MoData().GetM().size())==0)
        continue;

      typedef map<int,double>::const_iterator CI;

      for( int d=0; d<dim; d++ )
      {
        int dof = kcnode->Dofs()[d];

        // store M-values into refM
        newM[dof] = kcnode->MoData().GetM()[d];

        // print results (derivatives) to screen
        for (CI p=newM[dof].begin(); p!=newM[dof].end(); ++p)
        {
          if (abs(newM[dof][p->first]-refM[dof][p->first]) > 1e-12)
          {
            double finit = (newM[dof][p->first]-refM[dof][p->first])/delta;
            double analy = ((refDerivM[kgid])[(p->first)/Dim()])[sdof];
            double dev = finit - analy;

            // kgid: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // sdof: currently modified slave dof
            cout << "(" << dof << "," << (p->first)/Dim() << "," << sdof << ") : fd=" << finit << " derivm=" << analy << " DEVIATION " << dev;

            if( abs(dev) > 1e-4 )
            {
              cout << " ***** WARNING ***** ";
              w++;
            }
            else if( abs(dev) > 1e-5 )
            {
              cout << " ***** warning ***** ";
              w++;
            }

            cout << endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }

    cout << " ******************** GENERATED " << w << " WARNINGS ***************** " << endl;
  }

  // global loop to apply FD scheme to all MASTER dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements(); ++fd)
  {
    // store warnings for this finite difference
    int w=0;

    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* mnode = static_cast<CoNode*>(node);

    int mdof = mnode->Dofs()[fd%3];

    cout << "\nDEVIATION FOR M-NODE # " << gid << " DOF: " << mdof << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements(); ++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode)
        dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      int dim = kcnode->NumDof();

      if ((int)(kcnode->MoData().GetM().size())==0)
        continue;

      typedef map<int,double>::const_iterator CI;

      for( int d=0; d<dim; d++ )
      {
        int dof = kcnode->Dofs()[d];

        // store M-values into refM
        newM[dof] = kcnode->MoData().GetM()[d];

        // print results (derivatives) to screen
        for (CI p=newM[dof].begin(); p!=newM[dof].end(); ++p)
        {
          if (abs(newM[dof][p->first]-refM[dof][p->first]) > 1e-12)
          {
            double finit = (newM[dof][p->first]-refM[dof][p->first])/delta;
            double analy = ((refDerivM[kgid])[(p->first)/Dim()])[mdof];
            double dev = finit - analy;

            // dof: currently tested dof of slave node kgid
            // (p->first)/Dim(): paired master
            // mdof: currently modified master dof
            cout << "(" << dof << "," << (p->first)/Dim() << "," << mdof << ") : fd=" << finit << " derivm=" << analy << " DEVIATION " << dev;

            if( abs(dev) > 1e-4 )
            {
              cout << " ***** WARNING ***** ";
              w++;
            }
            else if( abs(dev) > 1e-5 )
            {
              cout << " ***** warning ***** ";
              w++;
            }

            cout << endl;
          }
        }
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }

    cout << " ******************** GENERATED " << w << " WARNINGS ***************** " << endl;
  }

  // back to normal...

  // Initialize
  Initialize();

  // compute element areas
  SetElementAreas();

  // *******************************************************************
  // contents of Evaluate()
  // *******************************************************************
  Evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for normal gap derivatives         popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckGapDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // create storage for gap values
  int nrow = snoderowmap_->NumMyElements();
  vector<double> refG(nrow);
  vector<double> newG(nrow);

  // store reference
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements();++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    CoNode* cnode = static_cast<CoNode*>(node);

    if (cnode->Active())
    {
      // check two versions of weighted gap
      double defgap = 0.0;
      double wii = (cnode->MoData().GetD()[0])[cnode->Dofs()[0]];

      for (int j=0;j<3;++j)
        defgap-= (cnode->MoData().n()[j])*wii*(cnode->xspatial()[j]);

      vector<map<int,double> > mmap = cnode->MoData().GetM();
      map<int,double>::iterator mcurr;

      for (int m=0;m<mnodefullmap->NumMyElements();++m)
      {
        int gid = mnodefullmap->GID(m);
        DRT::Node* mnode = idiscret_->gNode(gid);
        if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
        CoNode* cmnode = static_cast<CoNode*>(mnode);
        const int* mdofs = cmnode->Dofs();
        bool hasentry = false;

        // look for this master node in M-map of the active slave node
        for (mcurr=mmap[0].begin();mcurr!=mmap[0].end();++mcurr)
          if ((mcurr->first)==mdofs[0])
          {
            hasentry=true;
            break;
          }

        double mik = (mmap[0])[mdofs[0]];
        double* mxi = cmnode->xspatial();

        // get out of here, if master node not adjacent or coupling very weak
        if (!hasentry || abs(mik)<1.0e-12) continue;

        for (int j=0;j<3;++j)
          defgap+= (cnode->MoData().n()[j]) * mik * mxi[j];
      }

      //cout << "SNode: " << cnode->Id() << " IntGap: " << cnode->CoData().Getg() << " DefGap: " << defgap << endl;
      //cnode->CoData().Getg = defgap;
    }

    // store gap-values into refG
    refG[i]=cnode->CoData().Getg();
  }

  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
  {
    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==snode->Owner())
    {
      cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof(l): " << fd%3
           << " Dof(g): " << snode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0;k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      if (kcnode->Active())
      {
        // check two versions of weighted gap
        double defgap = 0.0;
        double wii = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];

        for (int j=0;j<3;++j)
          defgap-= (kcnode->MoData().n()[j])*wii*(kcnode->xspatial()[j]);

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        map<int,double>::iterator mcurr;

        for (int m=0;m<mnodefullmap->NumMyElements();++m)
        {
          int gid = mnodefullmap->GID(m);
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          CoNode* cmnode = static_cast<CoNode*>(mnode);
          const int* mdofs = cmnode->Dofs();
          bool hasentry = false;

          // look for this master node in M-map of the active slave node
          for (mcurr=mmap[0].begin();mcurr!=mmap[0].end();++mcurr)
            if ((mcurr->first)==mdofs[0])
            {
              hasentry=true;
              break;
            }

          double mik = (mmap[0])[mdofs[0]];
          double* mxi = cmnode->xspatial();

          // get out of here, if master node not adjacent or coupling very weak
          if (!hasentry || abs(mik)<1.0e-12) continue;

          for (int j=0;j<3;++j)
            defgap+= (kcnode->MoData().n()[j]) * mik * mxi[j];
        }

        //cout << "SNode: " << kcnode->Id() << " IntGap: " << kcnode->CoData().Getg << " DefGap: " << defgap << endl;
        //kcnode->CoData().Getg = defgap;
      }

      // store gap-values into newG
      newG[k]=kcnode->CoData().Getg();

      // print results (derivatives) to screen
      if (abs(newG[k]-refG[k]) > 1e-12)
      {
        cout << "G-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newG[k]-refG[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }
    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }

  // global loop to apply FD scheme to all master dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements();++fd)
  {
    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find master node with gid %",gid);
    CoNode* mnode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==mnode->Owner())
    {
      cout << "\nBuilding FD for Master Node: " << mnode->Id() << " Dof(l): " << fd%3
           << " Dof(g): " << mnode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0;k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      if (kcnode->Active())
      {
        // check two versions of weighted gap
        double defgap = 0.0;
        double wii = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];

        for (int j=0;j<3;++j)
          defgap-= (kcnode->MoData().n()[j])*wii*(kcnode->xspatial()[j]);

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        map<int,double>::iterator mcurr;

        for (int m=0;m<mnodefullmap->NumMyElements();++m)
        {
          int gid = mnodefullmap->GID(m);
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          CoNode* cmnode = static_cast<CoNode*>(mnode);
          const int* mdofs = cmnode->Dofs();
          bool hasentry = false;

          // look for this master node in M-map of the active slave node
          for (mcurr=mmap[0].begin();mcurr!=mmap[0].end();++mcurr)
            if ((mcurr->first)==mdofs[0])
            {
              hasentry=true;
              break;
            }

          double mik = (mmap[0])[mdofs[0]];
          double* mxi = cmnode->xspatial();

          // get out of here, if master node not adjacent or coupling very weak
          if (!hasentry || abs(mik)<1.0e-12) continue;

          for (int j=0;j<3;++j)
            defgap+= (kcnode->MoData().n()[j]) * mik * mxi[j];
        }

        //cout << "SNode: " << kcnode->Id() << " IntGap: " << kcnode->CoData().Getg << " DefGap: " << defgap << endl;
        //kcnode->CoData().Getg = defgap;
      }

      // store gap-values into newG
      newG[k]=kcnode->CoData().Getg();

      // print results (derivatives) to screen
      if (abs(newG[k]-refG[k]) > 1e-12)
      {
        cout << "G-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << mnode->Dofs()[fd%3] << " " << (newG[k]-refG[k])/delta << endl;
        //cout << "Analytical: " << mnode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[mnode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[mnode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }
  }

  // back to normal...
  Initialize();
  Evaluate();

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check for tang. LM derivatives           popp 06/08|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckTangLMDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // create storage for tangential LM values
  int nrow = snoderowmap_->NumMyElements();
  vector<double> refTLMxi(nrow);
  vector<double> newTLMxi(nrow);
  vector<double> refTLMeta(nrow);
  vector<double> newTLMeta(nrow);

  // store reference
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements();++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    CoNode* cnode = static_cast<CoNode*>(node);

    double valxi = 0.0;
    double valeta = 0.0;
    for (int dim=0;dim<3;++dim)
    {
      valxi  += (cnode->CoData().txi()[dim])*(cnode->MoData().lm()[dim]);
      valeta += (cnode->CoData().teta()[dim])*(cnode->MoData().lm()[dim]);
    }

    // store gap-values into refTLM
    refTLMxi[i]=valxi;
    refTLMeta[i]=valeta;
  }

  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
  {
    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    for (int i=0;i<idiscret_->NumMyColNodes();++i)
    {
      CONTACT::CoNode* node = static_cast<CONTACT::CoNode*>(idiscret_->lColNode(i));

      //reset nodal normal vector
      for (int j=0;j<3;++j)
      {
        node->MoData().n()[j]=0.0;
        node->CoData().txi()[j]=0.0;
        node->CoData().teta()[j]=0.0;
      }

      // reset derivative maps of normal vector
      for (int j=0;j<(int)((node->CoData().GetDerivN()).size());++j)
        (node->CoData().GetDerivN())[j].clear();
      (node->CoData().GetDerivN()).resize(0);

      // reset derivative maps of tangent vectors
      for (int j=0;j<(int)((node->CoData().GetDerivTxi()).size());++j)
        (node->CoData().GetDerivTxi())[j].clear();
      (node->CoData().GetDerivTxi()).resize(0);
      for (int j=0;j<(int)((node->CoData().GetDerivTeta()).size());++j)
        (node->CoData().GetDerivTeta())[j].clear();
      (node->CoData().GetDerivTeta()).resize(0);

      // reset nodal Mortar maps
      for (int j=0;j<(int)((node->MoData().GetD()).size());++j)
        (node->MoData().GetD())[j].clear();
      for (int j=0;j<(int)((node->MoData().GetM()).size());++j)
        (node->MoData().GetM())[j].clear();
      for (int j=0;j<(int)((node->MoData().GetMmod()).size());++j)
        (node->MoData().GetMmod())[j].clear();

      (node->MoData().GetD()).resize(0);
      (node->MoData().GetM()).resize(0);
      (node->MoData().GetMmod()).resize(0);

      // reset derivative map of Mortar matrices
      (node->CoData().GetDerivD()).clear();
      (node->CoData().GetDerivM()).clear();

      // reset nodal weighted gap
      node->CoData().Getg() = 1.0e12;
      (node->CoData().GetDerivG()).clear();

      // reset feasible projection status
      node->HasProj() = false;
    }

    // loop over all elements to reset candidates / search lists
    // (use standard slave column map)
    for (int i=0;i<SlaveColElements()->NumMyElements();++i)
    {
    	int gid = SlaveColElements()->GID(i);
  		DRT::Element* ele = Discret().gElement(gid);
  		if (!ele) dserror("ERROR: Cannot find ele with gid %i",gid);
  		MORTAR::MortarElement* mele = static_cast<MORTAR::MortarElement*>(ele);

      mele->MoData().SearchElements().resize(0);
    }

    // reset matrix containing interface contact segments (gmsh)
    //CSegs().Shape(0,0);

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==snode->Owner())
    {
      cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof(l): " << fd%3
           << " Dof(g): " << snode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    // loop over proc's slave nodes of the interface
    // use standard column map to include processor's ghosted nodes
    // use boundary map to include slave side boundary nodes
    for(int i=0; i<snodecolmapbound_->NumMyElements();++i)
    {
      int gid1 = snodecolmapbound_->GID(i);
      DRT::Node* node = idiscret_->gNode(gid1);
      if (!node) dserror("ERROR: Cannot find node with gid %",gid1);
      CoNode* cnode = static_cast<CoNode*>(node);

      // build averaged normal at each slave node
      cnode->BuildAveragedNormal();
    }

    // contact search algorithm
    EvaluateSearchBinarytree();

    // loop over proc's slave elements of the interface for integration
    // use standard column map to include processor's ghosted elements
    for (int i=0; i<selecolmap_->NumMyElements();++i)
    {
      int gid1 = selecolmap_->GID(i);
      DRT::Element* ele1 = idiscret_->gElement(gid1);
      if (!ele1) dserror("ERROR: Cannot find slave element with gid %",gid1);
      MORTAR::MortarElement* selement = static_cast<MORTAR::MortarElement*>(ele1);

      // loop over the contact candidate master elements of sele_
      // use slave element's candidate list SearchElements !!!
      for (int j=0;j<selement->MoData().NumSearchElements();++j)
      {
        int gid2 = selement->MoData().SearchElements()[j];
        DRT::Element* ele2 = idiscret_->gElement(gid2);
        if (!ele2) dserror("ERROR: Cannot find master element with gid %",gid2);
        MORTAR::MortarElement* melement = static_cast<MORTAR::MortarElement*>(ele2);

        //********************************************************************
        // 1) perform coupling (projection + overlap detection for sl/m pair)
        // 2) integrate Mortar matrix M and weighted gap g
        // 3) compute directional derivative of M and g and store into nodes
        //********************************************************************
        IntegrateCoupling(*selement,*melement);
      }
    }
    // *******************************************************************

    // compute finite difference derivative
    for (int k=0;k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      double valxi = 0.0;
      double valeta = 0.0;
      for (int dim=0;dim<3;++dim)
      {
        valxi  += (kcnode->CoData().txi()[dim])*(kcnode->MoData().lm()[dim]);
        valeta += (kcnode->CoData().teta()[dim])*(kcnode->MoData().lm()[dim]);
      }

      // store gap-values into newTLM
      newTLMxi[k]=valxi;
      newTLMeta[k]=valeta;

      // print results (derivatives) to screen
      if (abs(newTLMxi[k]-refTLMxi[k]) > 1e-12)
      {
        cout << "Xi-TLM-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-Xi-TLM: " << refTLMxi[k] << endl;
        //cout << "New-Xi-TLM: " << newTLMxi[k] << endl;
        cout << "Deriv: " << snode->Dofs()[fd%3] << " " << (newTLMxi[k]-refTLMxi[k])/delta << endl;
      }
      // print results (derivatives) to screen
      if (abs(newTLMeta[k]-refTLMeta[k]) > 1e-12)
      {
        cout << "Eta-TLM-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-TLM: " << refTLMeta[k] << endl;
        //cout << "New-TLM: " << newTLMeta[k] << endl;
        cout << "Deriv: " << snode->Dofs()[fd%3] << " " << (newTLMeta[k]-refTLMeta[k])/delta << endl;
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }

  // global loop to apply FD scheme to all master dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements();++fd)
  {
    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    for (int i=0;i<idiscret_->NumMyColNodes();++i)
    {
      CONTACT::CoNode* node = static_cast<CONTACT::CoNode*>(idiscret_->lColNode(i));

      //reset nodal normal vector
      for (int j=0;j<3;++j)
      {
        node->MoData().n()[j]=0.0;
        node->CoData().txi()[j]=0.0;
        node->CoData().teta()[j]=0.0;
      }

      // reset derivative maps of normal vector
      for (int j=0;j<(int)((node->CoData().GetDerivN()).size());++j)
        (node->CoData().GetDerivN())[j].clear();
      (node->CoData().GetDerivN()).resize(0);

      // reset derivative maps of tangent vectors
      for (int j=0;j<(int)((node->CoData().GetDerivTxi()).size());++j)
        (node->CoData().GetDerivTxi())[j].clear();
      (node->CoData().GetDerivTxi()).resize(0);
      for (int j=0;j<(int)((node->CoData().GetDerivTeta()).size());++j)
        (node->CoData().GetDerivTeta())[j].clear();
      (node->CoData().GetDerivTeta()).resize(0);

      // reset nodal Mortar maps
      for (int j=0;j<(int)((node->MoData().GetD()).size());++j)
        (node->MoData().GetD())[j].clear();
      for (int j=0;j<(int)((node->MoData().GetM()).size());++j)
        (node->MoData().GetM())[j].clear();
      for (int j=0;j<(int)((node->MoData().GetMmod()).size());++j)
        (node->MoData().GetMmod())[j].clear();

      (node->MoData().GetD()).resize(0);
      (node->MoData().GetM()).resize(0);
      (node->MoData().GetMmod()).resize(0);

      // reset derivative map of Mortar matrices
      (node->CoData().GetDerivD()).clear();
      (node->CoData().GetDerivM()).clear();

      // reset nodal weighted gap
      node->CoData().Getg() = 1.0e12;
      (node->CoData().GetDerivG()).clear();

      // reset feasible projection status
      node->HasProj() = false;
    }

    // loop over all elements to reset candidates / search lists
    // (use standard slave column map)
    for (int i=0;i<SlaveColElements()->NumMyElements();++i)
    {
    	int gid = SlaveColElements()->GID(i);
  		DRT::Element* ele = Discret().gElement(gid);
  		if (!ele) dserror("ERROR: Cannot find ele with gid %i",gid);
  		MORTAR::MortarElement* mele = static_cast<MORTAR::MortarElement*>(ele);

      mele->MoData().SearchElements().resize(0);
    }

    // reset matrix containing interface contact segments (gmsh)
    //CSegs().Shape(0,0);

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find master node with gid %",gid);
    CoNode* mnode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==mnode->Owner())
    {
      cout << "\nBuilding FD for Master Node: " << mnode->Id() << " Dof(l): " << fd%3
           << " Dof(g): " << mnode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    // loop over proc's slave nodes of the interface
    // use standard column map to include processor's ghosted nodes
    // use boundary map to include slave side boundary nodes
    for(int i=0; i<snodecolmapbound_->NumMyElements();++i)
    {
      int gid1 = snodecolmapbound_->GID(i);
      DRT::Node* node = idiscret_->gNode(gid1);
      if (!node) dserror("ERROR: Cannot find node with gid %",gid1);
      CoNode* cnode = static_cast<CoNode*>(node);

      // build averaged normal at each slave node
      cnode->BuildAveragedNormal();
    }

    // contact search algorithm
    EvaluateSearchBinarytree();

    // loop over proc's slave elements of the interface for integration
    // use standard column map to include processor's ghosted elements
    for (int i=0; i<selecolmap_->NumMyElements();++i)
    {
      int gid1 = selecolmap_->GID(i);
      DRT::Element* ele1 = idiscret_->gElement(gid1);
      if (!ele1) dserror("ERROR: Cannot find slave element with gid %",gid1);
      MORTAR::MortarElement* selement = static_cast<MORTAR::MortarElement*>(ele1);

      // loop over the contact candidate master elements of sele_
      // use slave element's candidate list SearchElements !!!
      for (int j=0;j<selement->MoData().NumSearchElements();++j)
      {
        int gid2 = selement->MoData().SearchElements()[j];
        DRT::Element* ele2 = idiscret_->gElement(gid2);
        if (!ele2) dserror("ERROR: Cannot find master element with gid %",gid2);
        MORTAR::MortarElement* melement = static_cast<MORTAR::MortarElement*>(ele2);

        //********************************************************************
        // 1) perform coupling (projection + overlap detection for sl/m pair)
        // 2) integrate Mortar matrix M and weighted gap g
        // 3) compute directional derivative of M and g and store into nodes
        //********************************************************************
        IntegrateCoupling(*selement,*melement);
      }
    }
    // *******************************************************************

    // compute finite difference derivative
    for (int k=0;k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      double valxi = 0.0;
      double valeta = 0.0;
      for (int dim=0;dim<3;++dim)
      {
        valxi  += (kcnode->CoData().txi()[dim])*(kcnode->MoData().lm()[dim]);
        valeta += (kcnode->CoData().teta()[dim])*(kcnode->MoData().lm()[dim]);
      }

      // store gap-values into newTLM
      newTLMxi[k]=valxi;
      newTLMeta[k]=valeta;

      // print results (derivatives) to screen
      if (abs(newTLMxi[k]-refTLMxi[k]) > 1e-12)
      {
        cout << "Xi-TLM-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-TLM: " << refTLMxi[k] << endl;
        //cout << "New-TLM: " << newTLMxi[k] << endl;
        cout << "Deriv: " << mnode->Dofs()[fd%3] << " " << (newTLMxi[k]-refTLMxi[k])/delta << endl;
      }
      // print results (derivatives) to screen
      if (abs(newTLMeta[k]-refTLMeta[k]) > 1e-12)
      {
        cout << "Eta-TLM-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-TLM: " << refTLMeta[k] << endl;
        //cout << "New-TLM: " << newTLMeta[k] << endl;
        cout << "Deriv: " << mnode->Dofs()[fd%3] << " " << (newTLMeta[k]-refTLMeta[k])/delta << endl;
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }
  }

  // back to normal...

  // Initialize
  // loop over all nodes to reset normals, closestnode and Mortar maps
  // (use fully overlapping column map)
  for (int i=0;i<idiscret_->NumMyColNodes();++i)
  {
    CONTACT::CoNode* node = static_cast<CONTACT::CoNode*>(idiscret_->lColNode(i));

    //reset nodal normal vector
    for (int j=0;j<3;++j)
    {
      node->MoData().n()[j]=0.0;
      node->CoData().txi()[j]=0.0;
      node->CoData().teta()[j]=0.0;
    }

    // reset derivative maps of normal vector
    for (int j=0;j<(int)((node->CoData().GetDerivN()).size());++j)
      (node->CoData().GetDerivN())[j].clear();
    (node->CoData().GetDerivN()).resize(0);

    // reset derivative maps of tangent vectors
    for (int j=0;j<(int)((node->CoData().GetDerivTxi()).size());++j)
      (node->CoData().GetDerivTxi())[j].clear();
    (node->CoData().GetDerivTxi()).resize(0);
    for (int j=0;j<(int)((node->CoData().GetDerivTeta()).size());++j)
      (node->CoData().GetDerivTeta())[j].clear();
    (node->CoData().GetDerivTeta()).resize(0);

    // reset nodal Mortar maps
    for (int j=0;j<(int)((node->MoData().GetD()).size());++j)
      (node->MoData().GetD())[j].clear();
    for (int j=0;j<(int)((node->MoData().GetM()).size());++j)
      (node->MoData().GetM())[j].clear();
    for (int j=0;j<(int)((node->MoData().GetMmod()).size());++j)
      (node->MoData().GetMmod())[j].clear();

    (node->MoData().GetD()).resize(0);
    (node->MoData().GetM()).resize(0);
    (node->MoData().GetMmod()).resize(0);

    // reset derivative map of Mortar matrices
    (node->CoData().GetDerivD()).clear();
    (node->CoData().GetDerivM()).clear();

    // reset nodal weighted gap
    node->CoData().Getg() = 1.0e12;
    (node->CoData().GetDerivG()).clear();

    // reset feasible projection status
    node->HasProj() = false;
  }

  // loop over all elements to reset candidates / search lists
  // (use standard slave column map)
  for (int i=0;i<SlaveColElements()->NumMyElements();++i)
  {
  	int gid = SlaveColElements()->GID(i);
		DRT::Element* ele = Discret().gElement(gid);
		if (!ele) dserror("ERROR: Cannot find ele with gid %i",gid);
		MORTAR::MortarElement* mele = static_cast<MORTAR::MortarElement*>(ele);

    mele->MoData().SearchElements().resize(0);
  }

  // reset matrix containing interface contact segments (gmsh)
  //CSegs().Shape(0,0);

  // compute element areas
  SetElementAreas();

  // *******************************************************************
  // contents of Evaluate()
  // *******************************************************************
  // loop over proc's slave nodes of the interface
  // use standard column map to include processor's ghosted nodes
  // use boundary map to include slave side boundary nodes
  for(int i=0; i<snodecolmapbound_->NumMyElements();++i)
  {
    int gid1 = snodecolmapbound_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid1);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid1);
    CoNode* cnode = static_cast<CoNode*>(node);

    // build averaged normal at each slave node
    cnode->BuildAveragedNormal();
  }

  // contact search algorithm
  EvaluateSearchBinarytree();

  // loop over proc's slave elements of the interface for integration
  // use standard column map to include processor's ghosted elements
  for (int i=0; i<selecolmap_->NumMyElements();++i)
  {
    int gid1 = selecolmap_->GID(i);
    DRT::Element* ele1 = idiscret_->gElement(gid1);
    if (!ele1) dserror("ERROR: Cannot find slave element with gid %",gid1);
    MORTAR::MortarElement* selement = static_cast<MORTAR::MortarElement*>(ele1);

    // loop over the contact candidate master elements of sele_
    // use slave element's candidate list SearchElements !!!
    for (int j=0;j<selement->MoData().NumSearchElements();++j)
    {
      int gid2 = selement->MoData().SearchElements()[j];
      DRT::Element* ele2 = idiscret_->gElement(gid2);
      if (!ele2) dserror("ERROR: Cannot find master element with gid %",gid2);
      MORTAR::MortarElement* melement = static_cast<MORTAR::MortarElement*>(ele2);

      //********************************************************************
      // 1) perform coupling (projection + overlap detection for sl/m pair)
      // 2) integrate Mortar matrix M and weighted gap g
      // 3) compute directional derivative of M and g and store into nodes
      //********************************************************************
      IntegrateCoupling(*selement,*melement);
    }
  }
  // *******************************************************************

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of stick condition derivatives      mgit 03/09|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckStickDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // create storage for values of complementary function C
  int nrow = snoderowmap_->NumMyElements();
  vector<double> refCtxi(nrow);
  vector<double> refCteta(nrow);
  vector<double> newCtxi(nrow);
  vector<double> newCteta(nrow);

  // store reference
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements();++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    FriNode* cnode = static_cast<FriNode*>(node);

    double jumptxi = 0;
    double jumpteta = 0;

    if (cnode->Active() and !(cnode->FriData().Slip()))
    {
      // calculate value of C-function
      double D = (cnode->MoData().GetD()[0])[cnode->Dofs()[0]];
      double Dold = (cnode->FriData().GetDOld()[0])[cnode->Dofs()[0]];

      for (int dim=0;dim<cnode->NumDof();++dim)
      {
        jumptxi -= (cnode->CoData().txi()[dim])*(D-Dold)*(cnode->xspatial()[dim]);
        jumpteta -= (cnode->CoData().teta()[dim])*(D-Dold)*(cnode->xspatial()[dim]);
      }

      vector<map<int,double> > mmap = cnode->MoData().GetM();
      vector<map<int,double> > mmapold = cnode->FriData().GetMOld();

      map<int,double>::iterator colcurr;
      set <int> mnodes;

      for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
        mnodes.insert((colcurr->first)/Dim());

      for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
        mnodes.insert((colcurr->first)/Dim());

      set<int>::iterator mcurr;

      // loop over all master nodes (find adjacent ones to this slip node)
      for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
      {
        int gid = *mcurr;
        DRT::Node* mnode = idiscret_->gNode(gid);
        if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
        FriNode* cmnode = static_cast<FriNode*>(mnode);
        const int* mdofs = cmnode->Dofs();

        double mik = (mmap[0])[mdofs[0]];
        double mikold = (mmapold[0])[mdofs[0]];

        map<int,double>::iterator mcurr;

        for (int dim=0;dim<cnode->NumDof();++dim)
        {
          jumptxi+= (cnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
          jumpteta+= (cnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
        }
      } //  loop over master nodes
    } // if cnode == Stick

    // store C in vector
    refCtxi[i] = jumptxi;
    refCteta[i] = jumpteta;
  } // loop over procs slave nodes


//  FD CHECK for Lagrange Multipliers (will be needed when formulating
//  stick condition according to Hueber)
//  // global loop to apply FD scheme for LM to all slave dofs (=3*nodes)
//  for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
//  {
//
//   // now get the node we want to apply the FD scheme to
//   int gid = snodefullmap->GID(fd/3);
//   DRT::Node* node = idiscret_->gNode(gid);
//   if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
//   FriNode* snode = static_cast<FriNode*>(node);
//
//   // apply finite difference scheme
//   if (Comm().MyPID()==snode->Owner())
//   {
//     cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof: " << fd%3
//          << " Dof: " << snode->Dofs()[fd%3] << endl;
//   }
//
//   // do step forward (modify nodal lagrange multiplier)
//   double delta = 1e-8;
//   if (fd%3==0)
//   {
//     snode->MoData().lm()[0] += delta;
//   }
//   else if (fd%3==1)
//   {
//     snode->MoData().lm()[1] += delta;
//   }
//   else
//   {
//     snode->MoData().lm()[2] += delta;
//   }
//
//   // compute finite difference derivative
//   for (int k=0; k<snoderowmap_->NumMyElements();++k)
//   {
//    int kgid = snoderowmap_->GID(k);
//     DRT::Node* knode = idiscret_->gNode(kgid);
//     if (!node) dserror("ERROR: Cannot find node with gid %",kgid);
//     FriNode* kcnode = static_cast<FriNode*>(knode);
//
//     double jumptan = 0;
//     double ztan = 0;
//     double znor = 0;
//
//     if (kcnode->Active() and !(kcnode->FriData().Slip()))
//     {
//       // check two versions of weighted gap
//       double D = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];
//       double Dold = (kcnode->FriData().GetDOld()[0])[kcnode->Dofs()[0]];
//
//       for (int dim=0;dim<kcnode->NumDof();++dim)
//       {
//        jumptan -= (kcnode->CoData().txi()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
//        ztan += (kcnode->CoData().txi()[dim])*(kcnode->MoData().lm()[dim]);
//        znor += (kcnode->MoData().n()[dim])*(kcnode->MoData().lm()[dim]);
//       }
//
//       vector<map<int,double> > mmap = kcnode->MoData().GetM();
//       vector<map<int,double> > mmapold = kcnode->FriData().GetMOld();
//
//       map<int,double>::iterator colcurr;
//       set <int> mnodes;
//
//       for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
//         mnodes.insert((colcurr->first)/Dim());
//
//       for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
//         mnodes.insert((colcurr->first)/Dim());
//
//       set<int>::iterator mcurr;
//
//       // loop over all master nodes (find adjacent ones to this stick node)
//       for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
//       {
//         int gid = *mcurr;
//         DRT::Node* mnode = idiscret_->gNode(gid);
//         if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
//         FriNode* cmnode = static_cast<FriNode*>(mnode);
//         const int* mdofs = cmnode->Dofs();
//
//         double mik = (mmap[0])[mdofs[0]];
//         double mikold = (mmapold[0])[mdofs[0]];
//         map<int,double>::iterator mcurr;
//
//         for (int dim=0;dim<kcnode->NumDof();++dim)
//         {
//            jumptan+= (kcnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
//         }
//       } //  loop over master nodes
//     } // if cnode == Slip
//
//       // store C in vector
//#ifdef CONTACTCOMPHUEBER
//         newC[k] = -frcoeff*(znor-cn*kcnode->CoData().Getg)*ct*jumptan;
//#else
//         newC[k] = jumptan;
//#endif
//
//       // print results (derivatives) to screen
//       if (abs(newC[k]-refC[k]) > 1e-12)
//       {
//         cout << "SlipCon-FD-derivative for LM for node S " << kcnode->Id() << endl;
//         //cout << "Ref-G: " << refG[k] << endl;
//         //cout << "New-G: " << newG[k] << endl;
//         cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newC[k]-refC[k])/delta << endl;
//         //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
//         //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
//         //  cout << "***WARNING*****************************************************************************" << endl;
//       }
//     }
//     // undo finite difference modification
//     if (fd%3==0)
//     {
//       snode->MoData().lm()[0] -= delta;
//     }
//     else if (fd%3==1)
//     {
//       snode->MoData().lm()[1] -= delta;
//     }
//     else
//     {
//       snode->MoData().lm()[2] -= delta;
//     }
//   } // loop over procs slave nodes


  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
  {
    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    FriNode* snode = static_cast<FriNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==snode->Owner())
    {
      cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof: " << fd%3
           << " Dof: " << snode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!node) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;

      if (kcnode->Active() and !(kcnode->FriData().Slip()))
      {
        // check two versions of weighted gap
        double D = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];
        double Dold = (kcnode->FriData().GetDOld()[0])[kcnode->Dofs()[0]];

        for (int dim=0;dim<kcnode->NumDof();++dim)
        {
          jumptxi -= (kcnode->CoData().txi()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->CoData().teta()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
        }

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        vector<map<int,double> > mmapold = kcnode->FriData().GetMOld();

        map<int,double>::iterator colcurr;
        set <int> mnodes;

        for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          FriNode* cmnode = static_cast<FriNode*>(mnode);
          const int* mdofs = cmnode->Dofs();

          double mik = (mmap[0])[mdofs[0]];
          double mikold = (mmapold[0])[mdofs[0]];

          map<int,double>::iterator mcurr;

          for (int dim=0;dim<kcnode->NumDof();++dim)
          {
            jumptxi+= (kcnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
            jumpteta+= (kcnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
          }
        } //  loop over master nodes
      } // if cnode == Slip

    newCtxi[k] = jumptxi;
    newCteta[k] = jumpteta;

      // print results (derivatives) to screen
      if (abs(newCtxi[k]-refCtxi[k]) > 1e-12)
      {
        cout << "StickCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newCtxi[k]-refCtxi[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newCteta[k]-refCteta[k]) > 1e-12)
      {
        cout << "StickCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newCteta[k]-refCteta[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }
    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  } // loop over procs slave nodes

  // global loop to apply FD scheme to all master dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements();++fd)
  {
    // Initialize
    // loop over all nodes to reset normals, closestnode and Mortar maps
    // (use fully overlapping column map)
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find master node with gid %",gid);
    FriNode* mnode = static_cast<FriNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==mnode->Owner())
    {
      cout << "\nBuilding FD for Master Node: " << mnode->Id() << " Dof: " << fd%3
           << " Dof: " << mnode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;

      if (kcnode->Active() and !(kcnode->FriData().Slip()))
      {
        // check two versions of weighted gap
        double D = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];
        double Dold = (kcnode->FriData().GetDOld()[0])[kcnode->Dofs()[0]];

        for (int dim=0;dim<kcnode->NumDof();++dim)
        {
          jumptxi -= (kcnode->CoData().txi()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->CoData().teta()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
        }

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        vector<map<int,double> > mmapold = kcnode->FriData().GetMOld();

        map<int,double>::iterator colcurr;
        set <int> mnodes;

        for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          FriNode* cmnode = static_cast<FriNode*>(mnode);
          const int* mdofs = cmnode->Dofs();

          double mik = (mmap[0])[mdofs[0]];
          double mikold = (mmapold[0])[mdofs[0]];

          map<int,double>::iterator mcurr;

          for (int dim=0;dim<kcnode->NumDof();++dim)
          {
            jumptxi+= (kcnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
            jumpteta+= (kcnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
          }
        } //  loop over master nodes
      } // if cnode == Slip

    newCtxi[k] = jumptxi;
    newCteta[k] = jumpteta;

      // print results (derivatives) to screen
      if (abs(newCtxi[k]-refCtxi[k]) > 1e-12)
      {
        cout << "StickCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << mnode->Dofs()[fd%3] << " " << (newCtxi[k]-refCtxi[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newCteta[k]-refCteta[k]) > 1e-12)
      {
        cout << "StickCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << mnode->Dofs()[fd%3] << " " << (newCteta[k]-refCteta[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }
  }

  // back to normal...
  Initialize();
  Evaluate();

  return;

} // FDCheckStickDeriv

/*----------------------------------------------------------------------*
 | Finite difference check of slip condition derivatives mgit      03/09|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckSlipDeriv()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // information from interface contact parameter list
  INPAR::CONTACT::FrictionType ftype =
    Teuchos::getIntegralValue<INPAR::CONTACT::FrictionType>(IParams(),"FRICTION");
  double frbound = IParams().get<double>("FRBOUND");
  double frcoeff = IParams().get<double>("FRCOEFF");
  double ct = IParams().get<double>("SEMI_SMOOTH_CT");
  double cn = IParams().get<double>("SEMI_SMOOTH_CN");

   // create storage for values of complementary function C
  int nrow = snoderowmap_->NumMyElements();
  vector<double> refCtxi(nrow);
  vector<double> refCteta(nrow);
  vector<double> newCtxi(nrow);
  vector<double> newCteta(nrow);

  // store reference
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements();++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    FriNode* cnode = static_cast<FriNode*>(node);

    double jumptxi = 0;
    double jumpteta = 0;
    double ztxi = 0;
    double zteta = 0;
    double znor = 0;
    double euclidean = 0;

    if (cnode->FriData().Slip())
    {
      // calculate value of C-function
      double D = (cnode->MoData().GetD()[0])[cnode->Dofs()[0]];
      double Dold = (cnode->FriData().GetDOld()[0])[cnode->Dofs()[0]];

      for (int dim=0;dim<cnode->NumDof();++dim)
      {
        jumptxi -= (cnode->CoData().txi()[dim])*(D-Dold)*(cnode->xspatial()[dim]);
        jumpteta -= (cnode->CoData().teta()[dim])*(D-Dold)*(cnode->xspatial()[dim]);
        ztxi += (cnode->CoData().txi()[dim])*(cnode->MoData().lm()[dim]);
        zteta += (cnode->CoData().teta()[dim])*(cnode->MoData().lm()[dim]);
        znor += (cnode->MoData().n()[dim])*(cnode->MoData().lm()[dim]);
      }

      vector<map<int,double> > mmap = cnode->MoData().GetM();
      vector<map<int,double> > mmapold = cnode->FriData().GetMOld();

      map<int,double>::iterator colcurr;
      set <int> mnodes;

      for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
        mnodes.insert((colcurr->first)/Dim());

      for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
        mnodes.insert((colcurr->first)/Dim());

      set<int>::iterator mcurr;

      // loop over all master nodes (find adjacent ones to this slip node)
      for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
      {
        int gid = *mcurr;
        DRT::Node* mnode = idiscret_->gNode(gid);
        if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
        FriNode* cmnode = static_cast<FriNode*>(mnode);
        const int* mdofs = cmnode->Dofs();

        double mik = (mmap[0])[mdofs[0]];
        double mikold = (mmapold[0])[mdofs[0]];

        map<int,double>::iterator mcurr;

        for (int dim=0;dim<cnode->NumDof();++dim)
        {
           jumptxi+= (cnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
           jumpteta+= (cnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
        }
      } //  loop over master nodes

      // evaluate euclidean norm ||vec(zt)+ct*vec(jumpt)||
      vector<double> sum1 (Dim()-1,0);
      sum1[0] =  ztxi+ct*jumptxi;
      if (Dim()==3) sum1[1] = zteta+ct*jumpteta;
      if (Dim()==2) euclidean = abs(sum1[0]);
      if (Dim()==3) euclidean = sqrt(sum1[0]*sum1[0]+sum1[1]*sum1[1]);
    } // if cnode == Slip

    // store C in vector
    if (ftype==INPAR::CONTACT::friction_tresca)
    {
      refCtxi[i] = euclidean*ztxi-frbound*(ztxi+ct*jumptxi);
      refCteta[i] = euclidean*zteta-frbound*(zteta+ct*jumpteta);
    }
    else if (ftype==INPAR::CONTACT::friction_coulomb)
    {
      refCtxi[i] = euclidean*ztxi-(frcoeff*znor)*(ztxi+ct*jumptxi);
      refCteta[i] = euclidean*zteta-(frcoeff*znor)*(zteta+ct*jumpteta);
    }
    else dserror ("ERROR: Friction law is neiter Tresca nor Coulomb");
#ifdef CONTACTCOMPHUEBER
  refCtxi[i] = euclidean*ztxi-(frcoeff*(znor-cn*cnode->CoData().Getg()))*(ztxi+ct*jumptxi);
  refCteta[i] = euclidean*zteta-(frcoeff*(znor-cn*cnode->CoData().Getg()))*(zteta+ct*jumpteta);
#endif

  } // loop over procs slave nodes

  // global loop to apply FD scheme for LM to all slave dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
  {

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    FriNode* snode = static_cast<FriNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==snode->Owner())
    {
      cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof: " << fd%3
           << " Dof: " << snode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->MoData().lm()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->MoData().lm()[1] += delta;
    }
    else
    {
      snode->MoData().lm()[2] += delta;
    }

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!node) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->FriData().Slip())
      {
        // check two versions of weighted gap
        double D = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];
        double Dold = (kcnode->FriData().GetDOld()[0])[kcnode->Dofs()[0]];
        for (int dim=0;dim<kcnode->NumDof();++dim)
        {
          jumptxi -= (kcnode->CoData().txi()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->CoData().teta()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          ztxi += (kcnode->CoData().txi()[dim])*(kcnode->MoData().lm()[dim]);
          zteta += (kcnode->CoData().teta()[dim])*(kcnode->MoData().lm()[dim]);
          znor += (kcnode->MoData().n()[dim])*(kcnode->MoData().lm()[dim]);
        }

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        vector<map<int,double> > mmapold = kcnode->FriData().GetMOld();

        map<int,double>::iterator colcurr;
        set <int> mnodes;

        for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          FriNode* cmnode = static_cast<FriNode*>(mnode);
          const int* mdofs = cmnode->Dofs();
          double mik = (mmap[0])[mdofs[0]];
          double mikold = (mmapold[0])[mdofs[0]];

          map<int,double>::iterator mcurr;

          for (int dim=0;dim<kcnode->NumDof();++dim)
          {
             jumptxi+= (kcnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
             jumpteta+= (kcnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
          }
        } //  loop over master nodes

        // evaluate euclidean norm ||vec(zt)+ct*vec(jumpt)||
        vector<double> sum1 (Dim()-1,0);
        sum1[0] = ztxi+ct*jumptxi;
        if (Dim()==3) sum1[1] = zteta+ct*jumpteta;
        if (Dim()==2) euclidean = abs(sum1[0]);
        if (Dim()==3) euclidean = sqrt(sum1[0]*sum1[0]+sum1[1]*sum1[1]);
      } // if cnode == Slip

      // store C in vector
      if (ftype==INPAR::CONTACT::friction_tresca)
      {
        newCtxi[k] = euclidean*ztxi-frbound*(ztxi+ct*jumptxi);
        newCteta[k] = euclidean*zteta-frbound*(zteta+ct*jumpteta);
      }
      else if (ftype==INPAR::CONTACT::friction_coulomb)
      {
        newCtxi[k] = euclidean*ztxi-(frcoeff*znor)*(ztxi+ct*jumptxi);
        newCteta[k] = euclidean*zteta-(frcoeff*znor)*(zteta+ct*jumpteta);
      }
      else dserror ("ERROR: Friction law is neiter Tresca nor Coulomb");
#ifdef CONTACTCOMPHUEBER
  newCtxi[k] = euclidean*ztxi-(frcoeff*(znor-cn*kcnode->CoData().Getg()))*(ztxi+ct*jumptxi);
  newCteta[k] = euclidean*zteta-(frcoeff*(znor-cn*kcnode->CoData().Getg()))*(zteta+ct*jumpteta);
#endif

      // print results (derivatives) to screen
      if (abs(newCtxi[k]-refCtxi[k]) > 1e-12)
      {
        cout << "SlipCon-FD-derivative for LM for node S " << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newCtxi[k]-refCtxi[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
       if (abs(newCteta[k]-refCteta[k]) > 1e-12)
       {
         cout << "SlipCon-FD-derivative for LM for node S " << kcnode->Id() << endl;
         //cout << "Ref-G: " << refG[k] << endl;
         //cout << "New-G: " << newG[k] << endl;
         cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newCteta[k]-refCteta[k])/delta << endl;
         //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
         //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
         //  cout << "***WARNING*****************************************************************************" << endl;
       }
     }
    // undo finite difference modification
    if (fd%3==0)
    {
      snode->MoData().lm()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->MoData().lm()[1] -= delta;
    }
    else
    {
      snode->MoData().lm()[2] -= delta;
    }
  } // loop over procs slave nodes


  // global loop to apply FD scheme to all slave dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
  {
    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    FriNode* snode = static_cast<FriNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==snode->Owner())
    {
      cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof: " << fd%3
           << " Dof: " << snode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!node) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->FriData().Slip())
      {
        // check two versions of weighted gap
        double D = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];
        double Dold = (kcnode->FriData().GetDOld()[0])[kcnode->Dofs()[0]];

        for (int dim=0;dim<kcnode->NumDof();++dim)
        {
          jumptxi -= (kcnode->CoData().txi()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->CoData().teta()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          ztxi += (kcnode->CoData().txi()[dim])*(kcnode->MoData().lm()[dim]);
          zteta += (kcnode->CoData().teta()[dim])*(kcnode->MoData().lm()[dim]);
          znor += (kcnode->MoData().n()[dim])*(kcnode->MoData().lm()[dim]);
        }

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        vector<map<int,double> > mmapold = kcnode->FriData().GetMOld();

        map<int,double>::iterator colcurr;
        set <int> mnodes;

        for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          FriNode* cmnode = static_cast<FriNode*>(mnode);
          const int* mdofs = cmnode->Dofs();

          double mik = (mmap[0])[mdofs[0]];
          double mikold = (mmapold[0])[mdofs[0]];

          map<int,double>::iterator mcurr;

          for (int dim=0;dim<kcnode->NumDof();++dim)
          {
            jumptxi+= (kcnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
            jumpteta+= (kcnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
          }
        } //  loop over master nodes

        // evaluate euclidean norm ||vec(zt)+ct*vec(jumpt)||
        vector<double> sum1 (Dim()-1,0);
        sum1[0] = ztxi+ct*jumptxi;
        if (Dim()==3) sum1[1] = zteta+ct*jumpteta;
        if (Dim()==2) euclidean = abs(sum1[0]);
        if (Dim()==3) euclidean = sqrt(sum1[0]*sum1[0]+sum1[1]*sum1[1]);

      } // if cnode == Slip

      // store C in vector
      if (ftype==INPAR::CONTACT::friction_tresca)
      {
        newCtxi[k] = euclidean*ztxi-frbound*(ztxi+ct*jumptxi);
        newCteta[k] = euclidean*zteta-frbound*(zteta+ct*jumpteta);
      }
      else if (ftype==INPAR::CONTACT::friction_coulomb)
      {
        newCtxi[k] = euclidean*ztxi-(frcoeff*znor)*(ztxi+ct*jumptxi);
        newCteta[k] = euclidean*zteta-(frcoeff*znor)*(zteta+ct*jumpteta);
      }
      else dserror ("ERROR: Friction law is neiter Tresca nor Coulomb");
#ifdef CONTACTCOMPHUEBER
  newCtxi[k] = euclidean*ztxi-(frcoeff*(znor-cn*kcnode->CoData().Getg()))*(ztxi+ct*jumptxi);
  newCteta[k] = euclidean*zteta-(frcoeff*(znor-cn*kcnode->CoData().Getg()))*(zteta+ct*jumpteta);
#endif

      // print results (derivatives) to screen
      if (abs(newCtxi[k]-refCtxi[k]) > 1e-12)
      {
        cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newCtxi[k]-refCtxi[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newCteta[k]-refCteta[k]) > 1e-12)
      {
        cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << snode->Dofs()[fd%3] << " " << (newCteta[k]-refCteta[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }
    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  } // loop over procs slave nodes

  // global loop to apply FD scheme to all master dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements();++fd)
  {
    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find master node with gid %",gid);
    FriNode* mnode = static_cast<FriNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==mnode->Owner())
    {
      cout << "\nBuilding FD for Master Node: " << mnode->Id() << " Dof: " << fd%3
           << " Dof: " << mnode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      double jumptxi = 0;
      double jumpteta = 0;
      double ztxi = 0;
      double zteta = 0;
      double znor = 0;
      double euclidean = 0;

      if (kcnode->FriData().Slip())
      {
        // check two versions of weighted gap
        double D = (kcnode->MoData().GetD()[0])[kcnode->Dofs()[0]];
        double Dold = (kcnode->FriData().GetDOld()[0])[kcnode->Dofs()[0]];

        for (int dim=0;dim<kcnode->NumDof();++dim)
        {
          jumptxi -= (kcnode->CoData().txi()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          jumpteta -= (kcnode->CoData().teta()[dim])*(D-Dold)*(kcnode->xspatial()[dim]);
          ztxi += (kcnode->CoData().txi()[dim])*(kcnode->MoData().lm()[dim]);
          zteta += (kcnode->CoData().teta()[dim])*(kcnode->MoData().lm()[dim]);
          znor += (kcnode->MoData().n()[dim])*(kcnode->MoData().lm()[dim]);
        }

        vector<map<int,double> > mmap = kcnode->MoData().GetM();
        vector<map<int,double> > mmapold = kcnode->FriData().GetMOld();

        map<int,double>::iterator colcurr;
        set <int> mnodes;

        for (colcurr=mmap[0].begin(); colcurr!=mmap[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        for (colcurr=mmapold[0].begin(); colcurr!=mmapold[0].end(); colcurr++)
          mnodes.insert((colcurr->first)/Dim());

        set<int>::iterator mcurr;

        // loop over all master nodes (find adjacent ones to this stick node)
        for (mcurr=mnodes.begin(); mcurr != mnodes.end(); mcurr++)
        {
          int gid = *mcurr;
          DRT::Node* mnode = idiscret_->gNode(gid);
          if (!mnode) dserror("ERROR: Cannot find node with gid %",gid);
          FriNode* cmnode = static_cast<FriNode*>(mnode);
          const int* mdofs = cmnode->Dofs();

          double mik = (mmap[0])[mdofs[0]];
          double mikold = (mmapold[0])[mdofs[0]];

          map<int,double>::iterator mcurr;

          for (int dim=0;dim<kcnode->NumDof();++dim)
          {
             jumptxi+= (kcnode->CoData().txi()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
             jumpteta+= (kcnode->CoData().teta()[dim])*(mik-mikold)*(cmnode->xspatial()[dim]);
          }
        } //  loop over master nodes

        // evaluate euclidean norm ||vec(zt)+ct*vec(jumpt)||
        vector<double> sum1 (Dim()-1,0);
        sum1[0] = ztxi+ct*jumptxi;
        if (Dim()==3) sum1[1] = zteta+ct*jumpteta;
        if (Dim()==2) euclidean = abs(sum1[0]);
        if (Dim()==3) euclidean = sqrt(sum1[0]*sum1[0]+sum1[1]*sum1[1]);

     } // if cnode == Slip

      // store C in vector
      if (ftype==INPAR::CONTACT::friction_tresca)
      {
        newCtxi[k] = euclidean*ztxi-frbound*(ztxi+ct*jumptxi);
        newCteta[k] = euclidean*zteta-frbound*(zteta+ct*jumpteta);
      }
      else if (ftype==INPAR::CONTACT::friction_coulomb)
      {
        newCtxi[k] = euclidean*ztxi-(frcoeff*znor)*(ztxi+ct*jumptxi);
        newCteta[k] = euclidean*zteta-(frcoeff*znor)*(zteta+ct*jumpteta);
      }
      else dserror ("ERROR: Friction law is neiter Tresca nor Coulomb");
#ifdef CONTACTCOMPHUEBER
  newCtxi[k] = euclidean*ztxi-(frcoeff*(znor-cn*kcnode->CoData().Getg()))*(ztxi+ct*jumptxi);
  newCteta[k] = euclidean*zteta-(frcoeff*(znor-cn*kcnode->CoData().Getg()))*(zteta+ct*jumpteta);
#endif

      // print results (derivatives) to screen
      if (abs(newCtxi[k]-refCtxi[k]) > 1e-12)
      {
        cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << mnode->Dofs()[fd%3] << " " << (newCtxi[k]-refCtxi[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      if (abs(newCteta[k]-refCteta[k]) > 1e-12)
      {
        cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " << mnode->Dofs()[fd%3] << " " << (newCteta[k]-refCteta[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }
  }

  // back to normal...
  Initialize();
  Evaluate();

  return;

} // FDCheckSlipTrescaDeriv

/*----------------------------------------------------------------------*
 | Finite difference check of lagr. mult. derivatives        popp 06/09 |
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckPenaltyTracNor()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  cout << setprecision(14);

  // create storage for lm entries
  map<int, double> reflm;
  map<int, double> newlm;

  map<int, map<int,double> > deltastorage;

  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements(); ++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find node with gid %",gid);
    CoNode* cnode = static_cast<CoNode*>(node);

    int dim = cnode->NumDof();

    for( int d=0; d<dim; d++ )
    {
      int dof = cnode->Dofs()[d];

      if ((int)(cnode->CoData().GetDerivZ()).size()!=0)
      {
        typedef map<int,double>::const_iterator CI;
        map<int,double>& derivzmap = cnode->CoData().GetDerivZ()[d];

        // print derivz-values to screen and store
        for (CI p=derivzmap.begin(); p!=derivzmap.end(); ++p)
        {
          //cout << " (" << cnode->Dofs()[k] << ", " << p->first << ") : \t " << p->second << endl;
          (deltastorage[cnode->Dofs()[d]])[p->first] = p->second;
        }
      }

      // store lm-values into refM
      reflm[dof] = cnode->MoData().lm()[d];
    }
  }

  cout << "FINITE DIFFERENCE SOLUTION\n" << endl;

  int w = 0;

  // global loop to apply FD scheme to all SLAVE dofs (=3*nodes)
  for (int fd=0; fd<3*snodefullmap->NumMyElements(); ++fd)
  {

    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    int sdof = snode->Dofs()[fd%3];

    cout << "DEVIATION FOR S-NODE # " << gid << " DOF: " << sdof << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // Evaluate
    Evaluate();
    bool isincontact, activesetchange = false;
    AssembleRegNormalForces(isincontact, activesetchange);

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements(); ++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode)
        dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      int dim = kcnode->NumDof();

      double fd;
      double dev;

      for( int d=0; d<dim; d++ )
      {
        int dof = kcnode->Dofs()[d];

        newlm[dof] = kcnode->MoData().lm()[d];

        fd = (newlm[dof] - reflm[dof]) / delta;

        dev = deltastorage[dof][sdof] - fd;

        if( dev )
        {
          cout << " (" << dof << ", " << sdof << ") :\t fd=" <<  fd
                    << " derivz=" << deltastorage[dof][sdof] << " DEVIATION: " << dev;

          if( abs(dev) > 1e-4 )
          {
            cout << " **** WARNING ****";
            w++;
          }
          else if( abs(dev) > 1e-5 )
          {
            cout << " **** warning ****";
            w++;
          }
          cout << endl;

          if( (abs(dev) > 1e-2) )
          {
             cout << " *************** ERROR *************** " << endl;
             //dserror("un-tolerable deviation");
          }
        }
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  }
  cout << "\n ******************** GENERATED " << w << " WARNINGS ***************** \n" << endl;

  w = 0;

  // global loop to apply FD scheme to all MASTER dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements(); ++fd)
  {

    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node)
      dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* mnode = static_cast<CoNode*>(node);

    int mdof = mnode->Dofs()[fd%3];

    cout << "DEVIATION FOR M-NODE # " << gid << " DOF: " << mdof << endl;

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // Evaluate
    Evaluate();
    bool isincontact, activesetchange = false;
    AssembleRegNormalForces(isincontact, activesetchange);

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements(); ++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!knode)
        dserror("ERROR: Cannot find node with gid %",kgid);
      CoNode* kcnode = static_cast<CoNode*>(knode);

      int dim = kcnode->NumDof();

      double fd;
      double dev;

      // calculate derivative and deviation for every dof
      for( int d=0; d<dim; d++ )
      {
        int dof = kcnode->Dofs()[d];

        newlm[dof] = kcnode->MoData().lm()[d];

        fd = (newlm[dof] - reflm[dof]) / delta;

        dev = deltastorage[dof][mdof] - fd;

        if( dev )
        {
          cout << " (" << dof << ", " << mdof << ") :\t fd=" <<  fd
                    << " derivz=" << deltastorage[dof][mdof] << " DEVIATION: " << dev;

          if( abs(dev) > 1e-4 )
          {
            cout << " **** WARNING ****";
            w++;
          }
          else if( abs(dev) > 1e-5 )
          {
            cout << " **** warning ****";
            w++;
          }
          cout << endl;

          if( (abs(dev) > 1e-2) )
          {
             cout << " *************** ERROR *************** " << endl;
             //dserror("un-tolerable deviation");
          }
        }
      }
    }

    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }

  }
  cout << "\n ******************** GENERATED " << w << " WARNINGS ***************** \n" << endl;

  // back to normal...

  // Initialize
  Initialize();

  // compute element areas
  SetElementAreas();

  // *******************************************************************
  // contents of Evaluate()
  // *******************************************************************
  Evaluate();
  bool isincontact, activesetchange = false;
  AssembleRegNormalForces(isincontact, activesetchange);

  return;
}

/*----------------------------------------------------------------------*
 | Finite difference check of frictional penalty traction     mgit 11/09|
 *----------------------------------------------------------------------*/
void CONTACT::CoInterface::FDCheckPenaltyTracFric()
{
	// FD checks only for serial case
	RCP<Epetra_Map> snodefullmap = LINALG::AllreduceEMap(*snoderowmap_);
	RCP<Epetra_Map> mnodefullmap = LINALG::AllreduceEMap(*mnoderowmap_);
	if (Comm().NumProc() > 1) dserror("ERROR: FD checks only for serial case");

  // get out of here if not participating in interface
  if (!lComm()) return;

  // information from interface contact parameter list
  double frcoeff = IParams().get<double>("FRCOEFF");
  double ppnor = IParams().get<double>("PENALTYPARAM");
  double pptan = IParams().get<double>("PENALTYPARAMTAN");

   // create storage for values of complementary function C
  int nrow = snoderowmap_->NumMyElements();
  vector<double> reftrac1(nrow);
  vector<double> newtrac1(nrow);
  vector<double> reftrac2(nrow);
  vector<double> newtrac2(nrow);
  vector<double> reftrac3(nrow);
  vector<double> newtrac3(nrow);

  // store reference
  // loop over proc's slave nodes
  for (int i=0; i<snoderowmap_->NumMyElements();++i)
  {
    int gid = snoderowmap_->GID(i);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find node with gid %",gid);
    FriNode* cnode = static_cast<FriNode*>(node);

    // get some informatiom form the node
    double gap = cnode->CoData().Getg();
    int dim = cnode->NumDof();
    double kappa = cnode->CoData().Kappa();
    double* n = cnode->MoData().n();

    // evaluate traction
    Epetra_SerialDenseMatrix jumpvec(dim,1);
    Epetra_SerialDenseMatrix tanplane(dim,dim);
    vector<double> trailtraction(dim);
    vector<double> tractionold(dim);
    double magnitude = 0;

    // fill vectors and matrices
    for (int j=0;j<dim;j++)
    {
      jumpvec(j,0) = cnode->FriData().jump()[j];
      tractionold[j] = cnode->FriData().tractionold()[j];
    }

    if (dim==3)
    {
      tanplane(0,0)= 1-(n[0]*n[0]);
      tanplane(0,1)=  -(n[0]*n[1]);
      tanplane(0,2)=  -(n[0]*n[2]);
      tanplane(1,0)=  -(n[1]*n[0]);
      tanplane(1,1)= 1-(n[1]*n[1]);
      tanplane(1,2)=  -(n[1]*n[2]);

      tanplane(2,0)=  -(n[2]*n[0]);
      tanplane(2,1)=  -(n[2]*n[1]);
      tanplane(2,2)= 1-(n[2]*n[2]);
    }
    else if (dim==2)
    {
      tanplane(0,0)= 1-(n[0]*n[0]);
      tanplane(0,1)=  -(n[0]*n[1]);

      tanplane(1,0)=  -(n[1]*n[0]);
      tanplane(1,1)= 1-(n[1]*n[1]);
    }
    else
      dserror("Error: Unknown dimension.");

    // Evaluate frictional trail traction
    Epetra_SerialDenseMatrix temptrac(dim,1);
    temptrac.Multiply('N','N',kappa*pptan,tanplane,jumpvec,0.0);

    //Lagrange multiplier in normal direction
    double lmuzawan = 0.0;
    for (int j=0;j<dim;++j)
    lmuzawan += cnode->MoData().lmuzawa()[j]*cnode->MoData().n()[j];

    // Lagrange multiplier from Uzawa algorithm
    Epetra_SerialDenseMatrix lmuzawa(dim,1);
    for (int k=0;k<dim;++k)
      lmuzawa(k,0) = cnode->MoData().lmuzawa()[k];

    // Lagrange multiplier in tangential direction
    Epetra_SerialDenseMatrix lmuzawatan(dim,1);
    lmuzawatan.Multiply('N','N',1,tanplane,lmuzawa,0.0);

    if (Teuchos::getIntegralValue<INPAR::CONTACT::SolvingStrategy>(IParams(),"STRATEGY")== INPAR::CONTACT::solution_penalty)
    {
      for (int j=0;j<dim;j++)
      {
        trailtraction[j]=tractionold[j]+temptrac(j,0);
        magnitude += (trailtraction[j]*trailtraction[j]);
      }
    }
    else
    {
      for (int j=0;j<dim;j++)
      {
        trailtraction[j] = lmuzawatan(j,0) + temptrac(j,0);
        magnitude += (trailtraction[j]*trailtraction[j]);
      }
    }

    // evaluate magnitude of trailtraction
    magnitude = sqrt(magnitude);

    // evaluate maximal tangential traction
    double maxtantrac = frcoeff*(lmuzawan - kappa * ppnor * gap);

    if(cnode->Active()==true and cnode->FriData().Slip()==false)
    {
      reftrac1[i] = n[0]*(lmuzawan - kappa * ppnor * gap)+trailtraction[0];
      reftrac2[i] = n[1]*(lmuzawan - kappa * ppnor * gap)+trailtraction[1];
      reftrac3[i] = n[2]*(lmuzawan - kappa * ppnor * gap)+trailtraction[2];
    }
    if(cnode->Active()==true and cnode->FriData().Slip()==true)
    {
      // compute lagrange multipliers and store into node
      reftrac1[i] = n[0]*(lmuzawan - kappa * ppnor * gap)+trailtraction[0]*maxtantrac/magnitude;
      reftrac2[i] = n[1]*(lmuzawan - kappa * ppnor * gap)+trailtraction[1]*maxtantrac/magnitude;
      reftrac3[i] = n[2]*(lmuzawan - kappa * ppnor * gap)+trailtraction[2]*maxtantrac/magnitude;
     }
   } // loop over procs slave nodes

    // global loop to apply FD scheme to all slave dofs (=3*nodes)
    for (int fd=0; fd<3*snodefullmap->NumMyElements();++fd)
    {
    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = snodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find slave node with gid %",gid);
    CoNode* snode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==snode->Owner())
    {
      cout << "\nBuilding FD for Slave Node: " << snode->Id() << " Dof: " << fd%3
           << " Dof: " << snode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      snode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] += delta;
    }
    else
    {
      snode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();
    //EvaluateRelMov();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!node) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      // get some informatiom form the node
      double gap = kcnode->CoData().Getg();
      int dim = kcnode->NumDof();
      double kappa = kcnode->CoData().Kappa();
      double* n = kcnode->MoData().n();

      // evaluate traction
      Epetra_SerialDenseMatrix jumpvec(dim,1);
      Epetra_SerialDenseMatrix tanplane(dim,dim);
      vector<double> trailtraction(dim);
      vector<double> tractionold(dim);
      double magnitude = 0;

      // fill vectors and matrices
      for (int j=0;j<dim;j++)
      {
        jumpvec(j,0) = kcnode->FriData().jump()[j];
        tractionold[j] = kcnode->FriData().tractionold()[j];
      }

      if (dim==3)
      {
        tanplane(0,0)= 1-(n[0]*n[0]);
        tanplane(0,1)=  -(n[0]*n[1]);
        tanplane(0,2)=  -(n[0]*n[2]);
        tanplane(1,0)=  -(n[1]*n[0]);
        tanplane(1,1)= 1-(n[1]*n[1]);
        tanplane(1,2)=  -(n[1]*n[2]);

        tanplane(2,0)=  -(n[2]*n[0]);
        tanplane(2,1)=  -(n[2]*n[1]);
        tanplane(2,2)= 1-(n[2]*n[2]);
      }
      else if (dim==2)
      {
        tanplane(0,0)= 1-(n[0]*n[0]);
        tanplane(0,1)=  -(n[0]*n[1]);

        tanplane(1,0)=  -(n[1]*n[0]);
        tanplane(1,1)= 1-(n[1]*n[1]);
      }
      else
        dserror("Error in AssembleTangentForces: Unknown dimension.");


      // Evaluate frictional trail traction
      Epetra_SerialDenseMatrix temptrac(dim,1);
      temptrac.Multiply('N','N',kappa*pptan,tanplane,jumpvec,0.0);

      //Lagrange multiplier in normal direction
      double lmuzawan = 0.0;
      for (int j=0;j<dim;++j)
        lmuzawan += kcnode->MoData().lmuzawa()[j]*kcnode->MoData().n()[j];

      // Lagrange multiplier from Uzawa algorithm
      Epetra_SerialDenseMatrix lmuzawa(dim,1);
      for (int j=0;j<dim;++j)
        lmuzawa(j,0) = kcnode->MoData().lmuzawa()[j];

      // Lagrange multiplier in tangential direction
      Epetra_SerialDenseMatrix lmuzawatan(dim,1);
      lmuzawatan.Multiply('N','N',1,tanplane,lmuzawa,0.0);

      if (Teuchos::getIntegralValue<INPAR::CONTACT::SolvingStrategy>(IParams(),"STRATEGY")== INPAR::CONTACT::solution_penalty)
      {
        for (int j=0;j<dim;j++)
        {
          trailtraction[j]=tractionold[j]+temptrac(j,0);
          magnitude += (trailtraction[j]*trailtraction[j]);
        }
      }
      else
      {
        for (int j=0;j<dim;j++)
        {
          trailtraction[j] = lmuzawatan(j,0)+temptrac(j,0);
          magnitude += (trailtraction[j]*trailtraction[j]);
        }
      }

      // evaluate magnitude of trailtraction
      magnitude = sqrt(magnitude);

      // evaluate maximal tangential traction
      double maxtantrac = frcoeff*(lmuzawan- kappa * ppnor * gap);

      if(kcnode->Active()==true and kcnode->FriData().Slip()==false)
      {
        newtrac1[k] = n[0]*(lmuzawan - kappa * ppnor * gap)+trailtraction[0];
        newtrac2[k] = n[1]*(lmuzawan - kappa * ppnor * gap)+trailtraction[1];
        newtrac3[k] = n[2]*(lmuzawan - kappa * ppnor * gap)+trailtraction[2];
      }
      if(kcnode->Active()==true and kcnode->FriData().Slip()==true)
      {
        // compute lagrange multipliers and store into node
        newtrac1[k] = n[0]*(lmuzawan - kappa * ppnor * gap)+trailtraction[0]*maxtantrac/magnitude;
        newtrac2[k] = n[1]*(lmuzawan - kappa * ppnor * gap)+trailtraction[1]*maxtantrac/magnitude;
        newtrac3[k] = n[2]*(lmuzawan - kappa * ppnor * gap)+trailtraction[2]*maxtantrac/magnitude;
      }

      // print results (derivatives) to screen
      if (abs(newtrac1[k]-reftrac1[k]) > 1e-12)
      {
        //cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv0:      " <<  kcnode->Dofs()[0] << " " << snode->Dofs()[fd%3] << " " << (newtrac1[k]-reftrac1[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newtrac2[k]-reftrac2[k]) > 1e-12)
      {
        //cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv1:      " <<  kcnode->Dofs()[1] << " "<< snode->Dofs()[fd%3] << " " << (newtrac2[k]-reftrac2[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newtrac3[k]-reftrac3[k]) > 1e-12)
      {
        //cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv2:      " <<  kcnode->Dofs()[2] << " " << snode->Dofs()[fd%3] << " " << (newtrac3[k]-reftrac3[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }
    // undo finite difference modification
    if (fd%3==0)
    {
      snode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      snode->xspatial()[1] -= delta;
    }
    else
    {
      snode->xspatial()[2] -= delta;
    }
  } // loop over procs slave nodes

   // global loop to apply FD scheme to all master dofs (=3*nodes)
  for (int fd=0; fd<3*mnodefullmap->NumMyElements();++fd)
  {
    // Initialize
    Initialize();

    // now get the node we want to apply the FD scheme to
    int gid = mnodefullmap->GID(fd/3);
    DRT::Node* node = idiscret_->gNode(gid);
    if (!node) dserror("ERROR: Cannot find master node with gid %",gid);
    CoNode* mnode = static_cast<CoNode*>(node);

    // apply finite difference scheme
    if (Comm().MyPID()==mnode->Owner())
    {
      cout << "\nBuilding FD for Master Node: " << mnode->Id() << " Dof: " << fd%3
           << " Dof: " << mnode->Dofs()[fd%3] << endl;
    }

    // do step forward (modify nodal displacement)
    double delta = 1e-8;
    if (fd%3==0)
    {
      mnode->xspatial()[0] += delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] += delta;
    }
    else
    {
      mnode->xspatial()[2] += delta;
    }

    // compute element areas
    SetElementAreas();

    // *******************************************************************
    // contents of Evaluate()
    // *******************************************************************
    Evaluate();
    //EvaluateRelMov();

    // compute finite difference derivative
    for (int k=0; k<snoderowmap_->NumMyElements();++k)
    {
      int kgid = snoderowmap_->GID(k);
      DRT::Node* knode = idiscret_->gNode(kgid);
      if (!node) dserror("ERROR: Cannot find node with gid %",kgid);
      FriNode* kcnode = static_cast<FriNode*>(knode);

      // get some informatiom form the node
      double gap = kcnode->CoData().Getg();
      int dim = kcnode->NumDof();
      double kappa = kcnode->CoData().Kappa();
      double* n = kcnode->MoData().n();

      // evaluate traction
      Epetra_SerialDenseMatrix jumpvec(dim,1);
      Epetra_SerialDenseMatrix tanplane(dim,dim);
      vector<double> trailtraction(dim);
      vector<double> tractionold(dim);
      double magnitude = 0;

      // fill vectors and matrices
      for (int j=0;j<dim;j++)
      {
        jumpvec(j,0) = kcnode->FriData().jump()[j];
        tractionold[j] = kcnode->FriData().tractionold()[j];
      }

      if (dim==3)
      {
        tanplane(0,0)= 1-(n[0]*n[0]);
        tanplane(0,1)=  -(n[0]*n[1]);
        tanplane(0,2)=  -(n[0]*n[2]);
        tanplane(1,0)=  -(n[1]*n[0]);
        tanplane(1,1)= 1-(n[1]*n[1]);
        tanplane(1,2)=  -(n[1]*n[2]);

        tanplane(2,0)=  -(n[2]*n[0]);
        tanplane(2,1)=  -(n[2]*n[1]);
        tanplane(2,2)= 1-(n[2]*n[2]);
      }
      else if (dim==2)
      {
        tanplane(0,0)= 1-(n[0]*n[0]);
        tanplane(0,1)=  -(n[0]*n[1]);

        tanplane(1,0)=  -(n[1]*n[0]);
        tanplane(1,1)= 1-(n[1]*n[1]);
      }
      else
        dserror("Error in AssembleTangentForces: Unknown dimension.");

      // Evaluate frictional trail traction
      Epetra_SerialDenseMatrix temptrac(dim,1);
      temptrac.Multiply('N','N',kappa*pptan,tanplane,jumpvec,0.0);

      //Lagrange multiplier in normal direction
      double lmuzawan = 0.0;
      for (int j=0;j<dim;++j)
        lmuzawan += kcnode->MoData().lmuzawa()[j]*kcnode->MoData().n()[j];

      // Lagrange multiplier from Uzawa algorithm
      Epetra_SerialDenseMatrix lmuzawa(dim,1);
      for (int j=0;j<dim;++j)
        lmuzawa(j,0) = kcnode->MoData().lmuzawa()[j];

      // Lagrange multiplier in tangential direction
      Epetra_SerialDenseMatrix lmuzawatan(dim,1);
      lmuzawatan.Multiply('N','N',1,tanplane,lmuzawa,0.0);

      if (Teuchos::getIntegralValue<INPAR::CONTACT::SolvingStrategy>(IParams(),"STRATEGY")== INPAR::CONTACT::solution_penalty)
      {
        for (int j=0;j<dim;j++)
        {
          trailtraction[j]=tractionold[j]+temptrac(j,0);
          magnitude += (trailtraction[j]*trailtraction[j]);
        }
      }
      else
      {
        for (int j=0;j<dim;j++)
        {
          trailtraction[j] = lmuzawatan(j,0)+temptrac(j,0);
          magnitude += (trailtraction[j]*trailtraction[j]);
        }
      }

      // evaluate magnitude of trailtraction
      magnitude = sqrt(magnitude);

      // evaluate maximal tangential traction
      double maxtantrac = frcoeff*(lmuzawan- kappa * ppnor * gap);

      if(kcnode->Active()==true and kcnode->FriData().Slip()==false)
      {
        newtrac1[k] = n[0]*(lmuzawan - kappa * ppnor * gap)+trailtraction[0];
        newtrac2[k] = n[1]*(lmuzawan - kappa * ppnor * gap)+trailtraction[1];
        newtrac3[k] = n[2]*(lmuzawan - kappa * ppnor * gap)+trailtraction[2];
      }
      if(kcnode->Active()==true and kcnode->FriData().Slip()==true)
      {
        // compute lagrange multipliers and store into node
        newtrac1[k] = n[0]*(lmuzawan - kappa * ppnor * gap)+trailtraction[0]*maxtantrac/magnitude;
        newtrac2[k] = n[1]*(lmuzawan - kappa * ppnor * gap)+trailtraction[1]*maxtantrac/magnitude;
        newtrac3[k] = n[2]*(lmuzawan - kappa * ppnor * gap)+trailtraction[2]*maxtantrac/magnitude;
      }

      // print results (derivatives) to screen
      if (abs(newtrac1[k]-reftrac1[k]) > 1e-12)
      {
        //cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " <<  kcnode->Dofs()[0] << " " << mnode->Dofs()[fd%3] << " " << (newtrac1[k]-reftrac1[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newtrac2[k]-reftrac2[k]) > 1e-12)
      {
        //cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " <<  kcnode->Dofs()[1] << " "<< mnode->Dofs()[fd%3] << " " << (newtrac2[k]-reftrac2[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }

      // print results (derivatives) to screen
      if (abs(newtrac3[k]-reftrac3[k]) > 1e-12)
      {
        //cout << "SlipCon-FD-derivative for node S" << kcnode->Id() << endl;
        //cout << "Ref-G: " << refG[k] << endl;
        //cout << "New-G: " << newG[k] << endl;
        cout << "Deriv:      " <<  kcnode->Dofs()[2] << " " << mnode->Dofs()[fd%3] << " " << (newtrac3[k]-reftrac3[k])/delta << endl;
        //cout << "Analytical: " << snode->Dofs()[fd%3] << " " << kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]] << endl;
        //if (abs(kcnode->CoData().GetDerivG()[snode->Dofs()[fd%3]]-(newG[k]-refG[k])/delta)>1.0e-5)
        //  cout << "***WARNING*****************************************************************************" << endl;
      }
    }
    // undo finite difference modification
    if (fd%3==0)
    {
      mnode->xspatial()[0] -= delta;
    }
    else if (fd%3==1)
    {
      mnode->xspatial()[1] -= delta;
    }
    else
    {
      mnode->xspatial()[2] -= delta;
    }
  }

  // back to normal...
  Initialize();
  Evaluate();
  //EvaluateRelMov();

  return;
} // FDCheckPenaltyFricTrac

#endif  // #ifdef CCADISCRET
