/*!
  \file post_drt_ensight_writer_nurbs.cpp

  \brief Nurbs specific helper methods for ensight filter basis class
  Methods are declared in post_drt_ensight_writer header file.

  <pre>
  Maintainer: Peter Gamnitzer
  gamnitzer@lnm.mw.tum.de
  http://www.lnm.mw.tum.de/Members/gammi
  089 - 289-15235
  </pre>

*/

#ifdef CCADISCRET

#include "post_drt_ensight_writer.H"
#include "../drt_nurbs_discret/drt_nurbs_discret.H"
#include "../drt_fem_general/drt_utils_nurbs_shapefunctions.H"
#include "../drt_nurbs_discret/drt_control_point.H"
#include <string>

using namespace std;

/*----------------------------------------------------------------------*/
/*
    Write the coordinates for a Nurbs discretization
    The ccordinates of the vizualisation points (i.e. the corner 
    nodes of elements displayed in paraview) are not the control point 
    coordinates of the nodes in the discretization but the points the
    knot values are mapped to.
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::WriteCoordinatesForNurbsShapefunctions
(
  ofstream&                              geofile ,
  const RefCountPtr<DRT::Discretization> dis     ,
  RefCountPtr<Epetra_Map>&               proc0map
  )
{
  // refcountpointer to vector of all coordinates
  // distributed among all procs
  RefCountPtr<Epetra_MultiVector> nodecoords;
  
  // the ids of the visualisation points on this proc
  vector<int> local_vis_point_ids;
  local_vis_point_ids.clear();

  // the coordinates of the visualisation points on this proc
  // used to construct the multivector nodecoords
  vector<vector<double> > local_vis_point_x;
  local_vis_point_x.clear();

  // cast dis to NurbsDiscretisation
  DRT::NURBS::NurbsDiscretization* nurbsdis
    =
    dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*dis));

  if(nurbsdis==NULL)
  {
    dserror("This probably isn't a NurbsDiscretization\n");
  }

  // get dimension
  int dim = (nurbsdis->Return_nele_x_mele_x_lele(0)).size();

  // get the knotvector itself
  RefCountPtr<DRT::NURBS::Knotvector> knots=nurbsdis->GetKnotVector();

  // detrmine number of patches from knotvector
  int npatches=knots->ReturnNP();

  // get vispoint offsets among patches
  vector<int> vpoff(npatches);

  vpoff[0]=0;

  // loop all patches
  for(int np=1;np<npatches;++np)
  {
    // get nurbs dis' knotvector sizes
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(np-1));

    int numvisp=1;

    for(unsigned rr=0;rr<nele_x_mele_x_lele.size();++rr)
    {
      numvisp*=2*nele_x_mele_x_lele[rr]+1;
    }

    vpoff[np]=vpoff[np-1]+numvisp;
  }

  // get element map
  const Epetra_Map* elementmap = nurbsdis->ElementRowMap();

  // loop all available elements
  for (int iele=0; iele<elementmap->NumMyElements(); ++iele)
  {
    DRT::Element* const actele = nurbsdis->gElement(elementmap->GID(iele));
    DRT::Node**   nodes = actele->Nodes();

    // get gid, location in the patch
    int gid = actele->Id();

    vector<int> ele_cart_id(dim);
    int np=-1;
    knots->ConvertEleGidToKnotIds(gid,np,ele_cart_id);

    // get nurbs dis' element numbers
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(np));

    // want to loop all control points of the element,
    // so get the number of points
    const int numnp = actele->NumNode();

    // access elements knot span
    std::vector<blitz::Array<double,1> > knots(dim);
    (*((*nurbsdis).GetKnotVector())).GetEleKnots(knots,actele->Id());

    // aquire weights from nodes
    blitz::Array<double,1> weights(numnp);

    for (int inode=0; inode<numnp; ++inode)
    {
      DRT::NURBS::ControlPoint* cp
        =
        dynamic_cast<DRT::NURBS::ControlPoint* > (nodes[inode]);

      weights(inode) = cp->W();
    }

    // get shapefunctions, compute all visualisation point positions
    blitz::Array<double,1> nurbs_shape_funct(numnp);

    switch (actele->Shape())
    {
    case DRT::Element::nurbs4:
    {
      // element local point position
      blitz::Array<double,1> uv(2);

      // standard

      // 3           4
      //  X---------X
      //  |         |
      //  |         |
      //  |         |
      //  |         |
      //  |         |
      //  X---------X
      // 1           2
      // append 4 points
      local_vis_point_ids.push_back((2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]  );
      local_vis_point_ids.push_back((2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+1);
      local_vis_point_ids.push_back((2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]  );
      local_vis_point_ids.push_back((2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+1);

      // temporary x vector
      std::vector<double> x(3);
      x[2]=0;

      // point 1
      uv(0)= -1.0;
      uv(1)= -1.0;
      DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                            uv               ,
                                            knots            ,
                                            weights          ,
                                            actele->Shape()  );
      for (int isd=0; isd<2; ++isd)
      {
        double val = 0;
        for (int inode=0; inode<numnp; ++inode)
        {
          val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
        }
        x[isd]=val;
      }
      local_vis_point_x.push_back(x);

      // point 2
      uv(0)=  1.0;
      uv(1)= -1.0;
      DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                            uv               ,
                                            knots            ,
                                            weights          ,
                                            actele->Shape()  );
      for (int isd=0; isd<2; ++isd)
      {
        double val = 0;
        for (int inode=0; inode<numnp; ++inode)
        {
          val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
        }
        x[isd]=val;
      }
      local_vis_point_x.push_back(x);

      // point 3
      uv(0)= -1.0;
      uv(1)=  1.0;
      DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                            uv               ,
                                            knots            ,
                                            weights          ,
                                            actele->Shape()  );
      for (int isd=0; isd<2; ++isd)
      {
        double val = 0;
        for (int inode=0; inode<numnp; ++inode)
        {
          val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
        }
        x[isd]=val;
      }
      local_vis_point_x.push_back(x);

      // point 4
      uv(0)= 1.0;
      uv(1)= 1.0;
      DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                            uv               ,
                                            knots            ,
                                            weights          ,
                                            actele->Shape()  );
      for (int isd=0; isd<2; ++isd)
      {
        double val = 0;
        for (int inode=0; inode<numnp; ++inode)
        {
          val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
        }
        x[isd]=val;
      }
      local_vis_point_x.push_back(x);

      break;
    }
    case DRT::Element::nurbs9:
    {
      // element local point position
      blitz::Array<double,1> uv(2);

      {
        // standard

        //
        //  +---------+
        //  |         |
        //  |         |
        //  X    X    |
        // 3|   4     |
        //  |         |
        //  X----X----+
        // 1    2
        // append 4 points
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]  );
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+1);
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]  );
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+1);

        // temporary x vector
        std::vector<double> x(3);
        x[2]=0;

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 2
        uv(0)=  0.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 3
        uv(0)= -1.0;
        uv(1)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 4
        uv(0)= 0.0;
        uv(1)= 0.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      if(ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {
        // top line

        //
        //  X----X----+
        // 5|   6     |
        //  |         |
        //  X    X    |
        // 3|   4     |
        //  |         |
        //  X----X----+
        // 1    2
        //

        // append points 5 and 6
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]  );
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+1);

        // temporary x vector
        std::vector<double> x(3);
        x[2]=0;

        // point 5
        uv(0)= -1.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );

        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 6
        uv(0)=  0.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0])
      {
        // right line

        //
        //  +---------+
        //  |         |
        //  |         |
        //  X    X    X
        // 4|   5    6|
        //  |         |
        //  X----X----X
        // 1    2    3

        // append points 3 and 6
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+2);
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+2);

        // temporary x vector
        std::vector<double> x(3);
        x[2]=0;

        // point 3
        uv(0)=  1.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 6
        uv(0)=  1.0;
        uv(1)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }
      if(ele_cart_id[1]+1==nele_x_mele_x_lele[1]
         &&
         ele_cart_id[0]+1==nele_x_mele_x_lele[0])
      {
        // top right corner

        //
        //  X----X----X
        // 7|   8    9|
        //  |         |
        //  X    X    X
        // 4|   5    6|
        //  |         |
        //  X----X----X
        // 1    2    3

        // append point 9
        local_vis_point_ids.push_back(vpoff[np]+(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1)+2*ele_cart_id[0]+2);

        // temporary x vector
        std::vector<double> x(3);
        x[2]=0;

        // point 9
        uv(0)=  1.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<2; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }
      break;
    }
    case DRT::Element::nurbs27:
    {
      // element local point position
      blitz::Array<double,1> uv(3);

      int idu;
      int idv;
      int idw;

      // number of visualisation points in u direction
      int nvpu=2*(nurbsdis->Return_nele_x_mele_x_lele(np))[0]+1;

      // number of visualisation points in v direction
      int nvpv=2*(nurbsdis->Return_nele_x_mele_x_lele(np))[1]+1;

      {
        // standard

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /         /  |
        //  +---------+   |
        //  | A----A  |   |
        //  |/|   /|  |   +
        //  A----A |  |  /
        //  | A--|-A  | /
        //  |/   |/   |/
        //  A----A----+ ----->u
        //
        // append 8 points

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);


        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);


        // point 2
        uv(0)=  0.0;
        uv(1)= -1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 3
        uv(0)= -1.0;
        uv(1)=  0.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 4
        uv(0)=  0.0;
        uv(1)=  0.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 5
        uv(0)= -1.0;
        uv(1)= -1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 6
        uv(0)=  0.0;
        uv(1)= -1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 7
        uv(0)= -1.0;
        uv(1)=  0.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 8
        uv(0)=  0.0;
        uv(1)=  0.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0])
      {

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /         /  |
        //  +---------+   |
        //  | X----X--|-A |
        //  |/|   /|  |/| +
        //  X----X----A |/
        //  | X--|-X--|-A
        //  |/   |/   |/
        //  X----X----A ----->u
        //
        // append 4 additional points

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);


        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)= -1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);


        // point 2
        uv(0)=  1.0;
        uv(1)=  0.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 3
        uv(0)=  1.0;
        uv(1)= -1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 4
        uv(0)=  1.0;
        uv(1)=  0.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      if(ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /  A----A /  |
        //  +---------+   |
        //  | X----X ||   |
        //  |/| A-/|-A|   +
        //  X----X |/ |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        // append 4 additional points

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);


        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)=  1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);


        // point 2
        uv(0)=  0.0;
        uv(1)=  1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 3
        uv(0)= -1.0;
        uv(1)=  1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 4
        uv(0)=  0.0;
        uv(1)=  1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0]
         &&
         ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /  X----X-/--A
        //  +---------+  /|
        //  | X----X--|-X |
        //  |/| X-/|-X|/|-A
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        // append 2 additional points

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]  )*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]+1)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)=  1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 2
        uv(0)=  1.0;
        uv(1)=  1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }


      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | A----A    / |
        //   /    /|   /  |
        //  A----A----+   |
        //  | X--|-X  |   |
        //  |/|  |/|  |   +
        //  X----X |  |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        //
        // append 4 additional points

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);


        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);


        // point 2
        uv(0)=  0.0;
        uv(1)= -1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 3
        uv(0)= -1.0;
        uv(1)=  0.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 4
        uv(0)=  0.0;
        uv(1)=  0.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

      }


      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2]
         &&
         ele_cart_id[1]+1==nele_x_mele_x_lele[1]
        )
      {
        //               v
        //              /
        //  w          /
        //  ^   A----A----+
        //  |  /|   /|   /|
        //  | X----X |  / |
        //   /| X /| X /  |
        //  X----X----+   |
        //  | X--|-X ||   |
        //  |/|  |/| X|   +
        //  X----X |/ |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        //
        // append 2 additional points

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);
        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)=  1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 2
        uv(0)=  0.0;
        uv(1)=  1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

      }

      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2]
         &&
         ele_cart_id[0]+1==nele_x_mele_x_lele[0]
        )
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | X----X----A |
        //   /    /|   /| |
        //  X----X----A | |
        //  | X--|-X--|-X |
        //  |/|  |/|  |/| +
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        //
        // append 2 additional points

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]  )*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+1)*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)= -1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);

        // point 2
        uv(0)=  1.0;
        uv(1)=  0.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2]
         &&
         ele_cart_id[1]+1==nele_x_mele_x_lele[1]
         &&
         ele_cart_id[0]+1==nele_x_mele_x_lele[0]
        )
      {

        //               v
        //              /
        //  w          /
        //  ^   X----X----A
        //  |  /|   /    /|
        //  | X----X----X |
        //   /| X-/|-X-/|-X
        //  X----X----X |/|
        //  | X--|-X--|-X |
        //  |/| X|/|-X|/|-X
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        // append 1 additional point


        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+2)*nvpu;
        idw=(2*ele_cart_id[2]+2)*nvpu*nvpv;
        local_vis_point_ids.push_back(vpoff[np]+idu+idv+idw);

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)=  1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              knots            ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<3; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=(((nodes[inode])->X())[isd])*nurbs_shape_funct(inode);
          }
          x[isd]=val;
        }
        local_vis_point_x.push_back(x);
      }

      break;
    }
    default:
      dserror("Unknown distype for nurbs element output\n");
    }
  }

  // construct map for visualisation points. Store it in
  // class variable for access in data interpolation
  int numvispoints = 0;

  // loop all patches
  for(int np=0;np<npatches;++np)
  {
    // get nurbs dis' knotvector sizes
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(np));

    int numvisp=1;

    for(unsigned rr=0;rr<nele_x_mele_x_lele.size();++rr)
    {
      numvisp*=2*(nele_x_mele_x_lele[rr])+1;
    }
    numvispoints+=numvisp;
  }

  vispointmap_ = Teuchos::rcp(new Epetra_Map(numvispoints,
                                             local_vis_point_ids.size(),
                                             &local_vis_point_ids[0],
                                             0,
                                             nurbsdis->Comm()));

  // allocate the coordinates of the vizualisation points
  nodecoords = rcp(new Epetra_MultiVector(*vispointmap_,3));

  // loop over the nodes on this proc and store the coordinate information
  for (int inode=0; inode<(int)local_vis_point_x.size(); inode++)
  {
    for (int isd=0; isd<3; ++isd)
    {
      double val = (local_vis_point_x[inode])[isd];
      nodecoords->ReplaceMyValue(inode, isd, val);
    }
  }

  //new procmap
  proc0map = LINALG::AllreduceEMap(*vispointmap_,0);

  // import my new values (proc0 gets everything, other procs empty)
  Epetra_Import proc0importer(*proc0map,*vispointmap_);
  RefCountPtr<Epetra_MultiVector> allnodecoords = rcp(new Epetra_MultiVector(*proc0map,3));
  int err = allnodecoords->Import(*nodecoords,proc0importer,Insert);
  if (err>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err);

  // write the node coordinates (only proc 0)
  // ensight format requires x_1 .. x_n, y_1 .. y_n, z_1 ... z_n
  // this is fulfilled automatically due to Epetra_MultiVector usage (columnwise writing data)
  if (myrank_==0)
  {
    double* coords = allnodecoords->Values();
    int numentries = (3*(allnodecoords->GlobalLength()));

    if (nodeidgiven_)
    {
      // first write node global ids (default)
      for (int inode=0; inode<proc0map->NumGlobalElements(); ++inode)
      {
        Write(geofile,static_cast<float>(proc0map->GID(inode))+1);
        // gid+1 delivers the node numbering of the *.dat file starting with 1
      }
    }
    // now write the coordinate information
    for (int i=0; i<numentries; ++i)
    {
      Write(geofile, static_cast<float>(coords[i]));
    }
  }

  return;
}

/*----------------------------------------------------------------------
         Write the cells for a Nurbs discretization
         quadratic nurbs split one element in knot space into 
         four(2d)/eight(3d) cells. The global numbering of the 
         vizualisation points (i.e. the corner points of the 
         cells) is computed from the local patch numbering and 
         the patch offset.                             (gammi)
----------------------------------------------------------------------*/
void EnsightWriter::WriteNurbsCell(
  const DRT::Element::DiscretizationType distype   ,
  const int                              gid       ,
  ofstream&                              geofile   ,
  vector<int>&                           nodevector,
  const RefCountPtr<DRT::Discretization> dis       ,
  const RefCountPtr<Epetra_Map>&         proc0map
) const
{
  // cast dis to NurbsDiscretisation
  DRT::NURBS::NurbsDiscretization* nurbsdis
    =
    dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*dis));

  if(nurbsdis==NULL)
  {
    dserror("This probably isn't a NurbsDiscretization\n");
  }

  // get the knotvector itself
  RefCountPtr<DRT::NURBS::Knotvector> knots=nurbsdis->GetKnotVector();

  // detrmine number of patches from knotvector
  int npatches=knots->ReturnNP();

  // get vispoint offsets among patches
  vector<int> vpoff(npatches);

  vpoff[0]=0;

  // loop all patches
  for(int np=1;np<npatches;++np)
  {
    // get nurbs dis' knotvector sizes
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(np-1));

    int numvisp=1;

    for(unsigned rr=0;rr<nele_x_mele_x_lele.size();++rr)
    {
      numvisp*=2*nele_x_mele_x_lele[rr]+1;
    }

    vpoff[np]=vpoff[np-1]+numvisp;
  }

  switch(distype)
  {
  case DRT::Element::nurbs4:
  {
    // get dimension
    const int dim = 2;

    // get the knotvector itself
    RefCountPtr<DRT::NURBS::Knotvector> knots=nurbsdis->GetKnotVector();

    // get location in the patch and the number of the patch
    int npatch  =-1;
    vector<int> ele_cart_id(dim);
    knots->ConvertEleGidToKnotIds(gid,npatch,ele_cart_id);

    // number of visualisation points in u direction
    int nvpu=(nurbsdis->Return_nele_x_mele_x_lele(npatch))[0]+1;

    // 3           4
    //  X---------X
    //  |         |
    //  |         |
    //  |         |
    //  |         |
    //  |         |
    //  X---------X
    // 1           2

    // append 4 elements
    if (myrank_==0) // proc0 can write its elements immediately
    {
      Write(geofile, proc0map->LID(((ele_cart_id[1]  )*(nvpu)+ele_cart_id[0]  ))+1);
      Write(geofile, proc0map->LID(((ele_cart_id[1]  )*(nvpu)+ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(((ele_cart_id[1]+1)*(nvpu)+ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(((ele_cart_id[1]+1)*(nvpu)+ele_cart_id[0]  ))+1);
    }
    else // elements on other procs have to store their global node ids
    {
      nodevector.push_back((ele_cart_id[1]  )*(nvpu)+ele_cart_id[0]  );
      nodevector.push_back((ele_cart_id[1]  )*(nvpu)+ele_cart_id[0]+1);
      nodevector.push_back((ele_cart_id[1]+1)*(nvpu)+ele_cart_id[0]+1);
      nodevector.push_back((ele_cart_id[1]+1)*(nvpu)+ele_cart_id[0]  );
    }
  }
  break;
  case DRT::Element::nurbs9:
  {
    // get dimension
    const int dim = 2;

    // get location in the patch from gid
    int npatch  =-1;
    vector<int> ele_cart_id(dim);
    knots->ConvertEleGidToKnotIds(gid,npatch,ele_cart_id);

    // number of visualisation points in u direction
    int nvpu=2*(nurbsdis->Return_nele_x_mele_x_lele(npatch))[0]+1;

    //
    //  X----X----X
    // 7|   8    9|
    //  |         |
    //  X    X    X
    // 4|   5    6|
    //  |         |
    //  X----X----X
    // 1    2    3

    // append 4 elements
    if (myrank_==0) // proc0 can write its elements immediately
    {
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]  ))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]  ))+1);

      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]  ))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]  ))+1);

      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+2))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+2))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1))+1);

      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+2))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+2))+1);
      Write(geofile, proc0map->LID(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+1))+1);
    }
    else // elements on other procs have to store their global node ids
    {
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]  ));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+1));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]  ));

      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]  ));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+1));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]  ));

      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+1));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+2));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+2));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1));

      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+1));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+2));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+2));
      nodevector.push_back(vpoff[npatch]+((2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+1));
    }
  }
  break;
  case DRT::Element::nurbs27:
  {
    //               v
    //              /
    //  w          /
    //  ^   X----X----A
    //  |  /|   /    /|
    //  | X----X----X |
    //   /| X-/|-X-/|-X
    //  X----X----X |/|
    //  | X--|-X--|-X |
    //  |/| X|/|-X|/|-X
    //  X----X----X |/
    //  | X--|-X--|-X
    //  |/   |/   |/
    //  X----X----X ----->u
    //

    // get dimension
    const int dim = 3;

    // get location in the patch
    int npatch  =-1;
    vector<int> ele_cart_id(dim);
    knots->ConvertEleGidToKnotIds(gid,npatch,ele_cart_id);

    // number of visualisation points in u direction
    int nvpu=2*(nurbsdis->Return_nele_x_mele_x_lele(npatch))[0]+1;
    
    // number of visualisation points in v direction
    int nvpv=2*(nurbsdis->Return_nele_x_mele_x_lele(npatch))[1]+1;

    // vector containing node connectivity for all sub hexes (in blocks of 8)
    vector<int> cellnodes(0);

    // bottom, left front
    AppendNurbsSubHex(cellnodes,0,0,0,ele_cart_id,nvpu,nvpv,npatch);
    // bottom, right front
    AppendNurbsSubHex(cellnodes,1,0,0,ele_cart_id,nvpu,nvpv,npatch);
    // bottom, left rear
    AppendNurbsSubHex(cellnodes,0,1,0,ele_cart_id,nvpu,nvpv,npatch);
    // bottom, right rear
    AppendNurbsSubHex(cellnodes,1,1,0,ele_cart_id,nvpu,nvpv,npatch);
    // top, left front
    AppendNurbsSubHex(cellnodes,0,0,1,ele_cart_id,nvpu,nvpv,npatch);
    // top, right front
    AppendNurbsSubHex(cellnodes,1,0,1,ele_cart_id,nvpu,nvpv,npatch);
    // top, left rear
    AppendNurbsSubHex(cellnodes,0,1,1,ele_cart_id,nvpu,nvpv,npatch);
    // top, right rear
    AppendNurbsSubHex(cellnodes,1,1,1,ele_cart_id,nvpu,nvpv,npatch);

    if(cellnodes.size()!=64)
    {
      dserror("something went wrong with the construction of cellnode connectivity\n");
    }

    if (myrank_==0) // proc0 can write its elements immediately
    {
      for(unsigned id=0;id<cellnodes.size();++id)
      {
        Write(geofile,proc0map->LID(vpoff[npatch]+cellnodes[id])+1);
      }
    }
    else // elements on other procs have to store their global node ids
    {
      for(unsigned id=0;id<cellnodes.size();++id)
      {
        nodevector.push_back(vpoff[npatch]+cellnodes[id]);
      }
    }
  }
  break;
  default:
  {
    dserror("unknown nurbs discretisation type\n");
  }
  } // end switch distype
  return;
}

/*----------------------------------------------------------------------*/
/*
    Write the results for a NURBS discretisation (dof based).
    
    On input, result data for an ndimensional computation
    is provided (from the result file)

    This element data is communicated in such a way that
    all elements have access to their (dof-accessible) data.
    Here we seperate velocity and pressure output, since
    for velocity and pressure different dofs are required.

    Then, all elements are looped and function values are
    evaluated at visualisation points. This is the place 
    where we need the dof data (again, different data for 
    velocity and pressure output)

    The resulting vector is allreduced on proc0 and written.

    gammi
*/
/*----------------------------------------------------------------------*/
void EnsightWriter::WriteDofResultStepForNurbs(
  ofstream&                        file ,
  const int                        numdf,
  const RefCountPtr<Epetra_Vector> data ,
  const string                     name
  ) const
{
  // a multivector for the interpolated data
  Teuchos::RefCountPtr<Epetra_MultiVector> idata;
  idata = Teuchos::rcp(new Epetra_MultiVector(*vispointmap_,numdf));

  DRT::NURBS::NurbsDiscretization* nurbsdis
    =
    dynamic_cast<DRT::NURBS::NurbsDiscretization*>(&(*field_->discretization()));

  if(nurbsdis==NULL)
  {
    dserror("This probably isn't a NurbsDiscretization\n");
  }

  // get number of patches
  int npatches = (nurbsdis->GetKnotVector())->ReturnNP();

  // assuming that dimension of the manifold is
  // equal to spatial dimension
  int dim = (int)(nurbsdis->Return_nele_x_mele_x_lele(0)).size();

  // the number of vizualisation points
  int numvispoints = 0;

  for(int np=0;np<npatches;++np)
  {
    int numvisp=1;

    // get nurbs dis' knotvector sizes
    vector<int> n_x_m_x_l(nurbsdis->Return_n_x_m_x_l(np));

    // get nurbs dis' knotvector sizes
    vector<int> degree(nurbsdis->Return_degree(np));

    for(unsigned rr=0;rr<n_x_m_x_l.size();++rr)
    {
      numvisp*=2*(n_x_m_x_l[rr]-2*degree[rr])-1;
    }
    numvispoints+=numvisp;
  } // end loop over patches

    // get the knotvector itself
  RefCountPtr<DRT::NURBS::Knotvector> knots=nurbsdis->GetKnotVector();

  // get vispoint offsets among patches
  vector<int> vpoff(npatches);

  vpoff[0]=0;

  // loop all patches
  for(int np=1;np<npatches;++np)
  {
    // get nurbs dis' knotvector sizes
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(np-1));

    int numvisp=1;

    for(unsigned rr=0;rr<nele_x_mele_x_lele.size();++rr)
    {
      numvisp*=2*nele_x_mele_x_lele[rr]+1;
    }

    vpoff[np]=vpoff[np-1]+numvisp;
  }

  // get element map
  const Epetra_Map* elementmap = nurbsdis->ElementRowMap();

  // construct a colmap for data to have it available at
  // all elements (the critical ones are the ones at the
  // processor boundary)
  // loop all available elements
  std::set<int> coldofset;
  for (int iele=0; iele<elementmap->NumMyElements(); ++iele)
  {
    DRT::Element* const actele = nurbsdis->gElement(elementmap->GID(iele));

    vector<int> lm;
    vector<int> lmowner;

    // extract local values from the global vectors
    actele->LocationVector(*nurbsdis,lm,lmowner);

    for (int inode=0; inode<actele->NumNode(); ++inode)
    {

      if(name == "velocity")
      {
        for(int rr=0;rr<dim;++rr)
        {
          coldofset.insert(lm[inode*(dim+1)+rr]);
        }
      }
      else if(name == "pressure")
      {
        coldofset.insert(lm[inode*(dim+1)+dim]);
      }
      else
      {
        dserror("up to now, I'm only able to write velocity and pressure\n");
      }
    }
  }

  std::vector<int> coldofmapvec;
  coldofmapvec.reserve(coldofset.size());
  coldofmapvec.assign(coldofset.begin(), coldofset.end());
  coldofset.clear();
  Teuchos::RCP<Epetra_Map> coldofmap =
    Teuchos::rcp(new Epetra_Map(-1,
                                coldofmapvec.size(),
                                &coldofmapvec[0],
                                0,
                                nurbsdis->Comm()));
  coldofmapvec.clear();

  const Epetra_Map* fulldofmap = &(*coldofmap);
  const RefCountPtr<Epetra_Vector> coldata
    = Teuchos::rcp(new Epetra_Vector(*fulldofmap,true));

  // create an importer and import the data
  Epetra_Import importer((*coldata).Map(),(*data).Map());
  int imerr = (*coldata).Import((*data),importer,Insert);
  if(imerr)
  {
    dserror("import falied\n");
  }

  // loop all available elements
  for (int iele=0; iele<elementmap->NumMyElements(); ++iele)
  {
    DRT::Element* const actele = nurbsdis->gElement(elementmap->GID(iele));
    DRT::Node**   nodes = actele->Nodes();

    // get gid, location in the patch and the number of the patch
    int gid = actele->Id();

    int npatch  =-1;
    vector<int> ele_cart_id(dim);
    knots->ConvertEleGidToKnotIds(gid,npatch,ele_cart_id);

    // get nele_x_mele_x_lele array
    vector<int> nele_x_mele_x_lele(nurbsdis->Return_nele_x_mele_x_lele(npatch));

    // number of all control points of the element
    const int numnp = actele->NumNode();

    // access elements knot span
    std::vector<blitz::Array<double,1> > eleknots(dim);
    knots->GetEleKnots(eleknots,actele->Id());

    // aquire weights from nodes
    blitz::Array<double,1> weights(numnp);

    for (int inode=0; inode<numnp; ++inode)
    {
      DRT::NURBS::ControlPoint* cp = dynamic_cast<DRT::NURBS::ControlPoint* > (nodes[inode]);
      weights(inode) = cp->W();
    }

    // get shapefunctions, compute all visualisation point positions
    blitz::Array<double,1> nurbs_shape_funct(numnp);

    // element local visualisation point position
    blitz::Array<double,1> uv(dim);

    // extract local values from the global vectors
    vector<int> lm;
    vector<int> lmowner;

    actele->LocationVector(*nurbsdis,lm,lmowner);

    vector<double> my_data(lm.size());
    if(name == "velocity")
    {
      my_data.resize(dim*numnp);

      for (int inode=0; inode<numnp; ++inode)
      {
        for(int rr=0;rr<dim;++rr)
        {
          my_data[dim*inode+rr]=(*coldata)[(*coldata).Map().LID(lm[inode*(dim+1)+rr])];
        }
      }
    }
    else if(name == "pressure")
    {
      my_data.resize(numnp);

      for (int inode=0; inode<numnp; ++inode)
      {
        my_data[inode]=(*coldata)[(*coldata).Map().LID(lm[inode*(dim+1)+dim])];
      }
    }
    else
    {
      dserror("up to now, I'm only able to write velocity and pressure\n");
    }

    switch (actele->Shape())
    {
    case DRT::Element::nurbs4:
    {

      // number of visualisation points in u direction
      int nvpu=(nurbsdis->Return_nele_x_mele_x_lele(npatch))[0]+1;

      {
        // standard

        // 3           4
        //  X---------X
        //  |         |
        //  |         |
        //  |         |
        //  |         |
        //  |         |
        //  X---------X
        // 1           2

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID((ele_cart_id[1]  )*(nvpu)+ele_cart_id[0]  );
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  1.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID((ele_cart_id[1]  )*(nvpu)+ele_cart_id[0]+1);
          (idata)->ReplaceMyValue(lid,isd,val);
        }


        // point 3
        uv(0)= -1.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID((ele_cart_id[1]+1)*(nvpu)+ele_cart_id[0]  );
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 4
        uv(0)=  1.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID((ele_cart_id[1]+1)*(nvpu)+ele_cart_id[0]+1);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

      }
      break;
    }
    case DRT::Element::nurbs9:
    {

      int idu;
      int idv;

      // number of visualisation points in u direction
      int nvpu=2*(nurbsdis->Return_nele_x_mele_x_lele(npatch))[0]+1;

      {
        // standard

        //
        //  +---------+
        //  |         |
        //  |         |
        //  X    X    |
        // 3|   4     |
        //  |         |
        //  X----X----+
        // 1    2

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        idu=2*ele_cart_id[0];
        idv=2*ele_cart_id[1]*(nvpu);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  0.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        idu=2*ele_cart_id[0]+1;
        idv=2*ele_cart_id[1]*(nvpu);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idv+idu);
          (idata)->ReplaceMyValue(lid,isd,val);
        }


        // point 3
        uv(0)= -1.0;
        uv(1)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        idu=2*ele_cart_id[0];
        idv=(2*ele_cart_id[1]+1)*(nvpu);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 4
        uv(0)=  0.0;
        uv(1)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        idu=2*ele_cart_id[0]+1;
        idv=(2*ele_cart_id[1]+1)*(nvpu);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

      }
      // top line

      //
      //  X----X----+
      // 5|   6     |
      //  |         |
      //  X    X    |
      // 3|   4     |
      //  |         |
      //  X----X----+
      // 1    2
      //
      // two additional points

      if(ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {
        // point 5
        uv(0)= -1.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+(2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]  );
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 6
        uv(0)=  0.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+(2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+1);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }

      // right line
      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0])
      {

        //
        //  +---------+
        //  |         |
        //  |         |
        //  x    x    X
        // 4|   5    6|
        //  |         |
        //  x----x----X
        // 1    2    3

        // two additional points
        // point 5
        uv(0)=  1.0;
        uv(1)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+(2*ele_cart_id[1]  )*(nvpu)+2*ele_cart_id[0]+2);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 6
        uv(0)=  1.0;
        uv(1)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+(2*ele_cart_id[1]+1)*(nvpu)+2*ele_cart_id[0]+2);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }

      // top right corner
      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0]&&ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {
        //
        //  x----x----X
        // 7|   8    9|
        //  |         |
        //  x    x    x
        // 4|   5    6|
        //  |         |
        //  x----x----x
        // 1    2    3

        // point 9
        uv(0)=  1.0;
        uv(1)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_2D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots         ,
                                              weights          ,
                                              actele->Shape()  );
        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+(2*ele_cart_id[1]+2)*(nvpu)+2*ele_cart_id[0]+2);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }
      break;
    }
    case DRT::Element::nurbs27:
    {
      // element local point position
      blitz::Array<double,1> uv(3);

      int idu;
      int idv;
      int idw;

      {
        // standard

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /         /  |
        //  +---------+   |
        //  | A----A  |   |
        //  |/|   /|  |   +
        //  A----A |  |  /
        //  | A--|-A  | /
        //  |/   |/   |/
        //  A----A----+ ----->u
        //
        // append 8 points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  0.0;
        uv(1)= -1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 3
        uv(0)= -1.0;
        uv(1)=  0.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 4
        uv(0)=  0.0;
        uv(1)=  0.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 5
        uv(0)= -1.0;
        uv(1)= -1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 6
        uv(0)=  0.0;
        uv(1)= -1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 7
        uv(0)= -1.0;
        uv(1)=  0.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 8
        uv(0)=  0.0;
        uv(1)=  0.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }

      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0])
      {

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /         /  |
        //  +---------+   |
        //  | X----X--|-A |
        //  |/|   /|  |/| +
        //  X----X----A |/
        //  | X--|-X--|-A
        //  |/   |/   |/
        //  X----X----A ----->u
        //
        // append 4 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)= -1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );
        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  1.0;
        uv(1)=  0.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 3
        uv(0)=  1.0;
        uv(1)= -1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 4
        uv(0)=  1.0;
        uv(1)=  0.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }

      if(ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /  A----A /  |
        //  +---------+   |
        //  | X----X ||   |
        //  |/| A-/|-A|   +
        //  X----X |/ |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        // append 4 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)=  1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }


        // point 2
        uv(0)=  0.0;
        uv(1)=  1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 3
        uv(0)= -1.0;
        uv(1)=  1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 4
        uv(0)=  0.0;
        uv(1)=  1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

      }

      if(ele_cart_id[0]+1==nele_x_mele_x_lele[0]
         &&
         ele_cart_id[1]+1==nele_x_mele_x_lele[1])
      {

        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | /         / |
        //   /  X----X-/--A
        //  +---------+  /|
        //  | X----X--|-X |
        //  |/| X-/|-X|/|-A
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        // append 2 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)=  1.0;
        uv(2)= -1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );
        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]  )*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  1.0;
        uv(1)=  1.0;
        uv(2)=  0.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+1)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }


      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2])
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | A----A    / |
        //   /    /|   /  |
        //  A----A----+   |
        //  | X--|-X  |   |
        //  |/|  |/|  |   +
        //  X----X |  |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        //
        // append 4 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)= -1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  0.0;
        uv(1)= -1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 3
        uv(0)= -1.0;
        uv(1)=  0.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 4
        uv(0)=  0.0;
        uv(1)=  0.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }


      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2]
         &&
         ele_cart_id[1]+1==nele_x_mele_x_lele[1]
        )
      {
        //               v
        //              /
        //  w          /
        //  ^   A----A----+
        //  |  /|   /|   /|
        //  | X----X |  / |
        //   /| X /| X /  |
        //  X----X----+   |
        //  | X--|-X ||   |
        //  |/|  |/| X|   +
        //  X----X |/ |  /
        //  | X--|-X  | /
        //  |/   |/   |/
        //  X----X----+ ----->u
        //
        //
        // append 2 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)= -1.0;
        uv(1)=  1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]  );
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  0.0;
        uv(1)=  1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+1);
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

      }

      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2]
         &&
         ele_cart_id[0]+1==nele_x_mele_x_lele[0]
        )
      {
        //               v
        //              /
        //  w          /
        //  ^   +---------+
        //  |  /         /|
        //  | X----X----A |
        //   /    /|   /| |
        //  X----X----A | |
        //  | X--|-X--|-X |
        //  |/|  |/|  |/| +
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        //
        // append 2 additional points

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)= -1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]  )*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

        // point 2
        uv(0)=  1.0;
        uv(1)=  0.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+1)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }
      }

      if(ele_cart_id[2]+1==nele_x_mele_x_lele[2]
         &&
         ele_cart_id[1]+1==nele_x_mele_x_lele[1]
         &&
         ele_cart_id[0]+1==nele_x_mele_x_lele[0]
        )
      {

        //               v
        //              /
        //  w          /
        //  ^   X----X----A
        //  |  /|   /    /|
        //  | X----X----X |
        //   /| X-/|-X-/|-X
        //  X----X----X |/|
        //  | X--|-X--|-X |
        //  |/| X|/|-X|/|-X
        //  X----X----X |/
        //  | X--|-X--|-X
        //  |/   |/   |/
        //  X----X----X ----->u
        //
        // append 1 additional point

        // temporary x vector
        std::vector<double> x(3);

        // point 1
        uv(0)=  1.0;
        uv(1)=  1.0;
        uv(2)=  1.0;
        DRT::NURBS::UTILS::nurbs_get_3D_funct(nurbs_shape_funct,
                                              uv               ,
                                              eleknots            ,
                                              weights          ,
                                              actele->Shape()  );

        idu=(2*ele_cart_id[0]+2);
        idv=(2*ele_cart_id[1]+2)*(2*nele_x_mele_x_lele[0]+1);
        idw=(2*ele_cart_id[2]+2)*(2*nele_x_mele_x_lele[1]+1)*(2*nele_x_mele_x_lele[0]+1);

        for (int isd=0; isd<numdf; ++isd)
        {
          double val = 0;
          for (int inode=0; inode<numnp; ++inode)
          {
            val+=my_data[numdf*inode+isd]*nurbs_shape_funct(inode);
          }
          int lid = (*vispointmap_).LID(vpoff[npatch]+idu+idv+idw);
          (idata)->ReplaceMyValue(lid,isd,val);
        }

      }

      break;
    }
    default:
      dserror("unable to visualise this as a nurbs discretisation\n");
    }
  }

  // import my new values (proc0 gets everything, other procs empty)
  Epetra_Import proc0importer(*proc0map_,*vispointmap_);
  RefCountPtr<Epetra_MultiVector> allsols = rcp(new Epetra_MultiVector(*proc0map_,numdf));
  int err = allsols->Import(*idata,proc0importer,Insert);
  if (err>0) dserror("Importing everything to proc 0 went wrong. Import returns %d",err);

  // write the node results (only proc 0)
  // ensight format requires u_1 .. u_n, v_1 .. v_n, w_1 ... w_n, as for nodes
  // this is fulfilled automatically due to Epetra_MultiVector usage (columnwise writing data)
  if (myrank_==0)
  {
    double* solvals = allsols->Values();
    int numentries = (numdf*(allsols->GlobalLength()));

    // now write the solution
    for (int i=0; i<numentries; ++i)
    {
      Write(file, static_cast<float>(solvals[i]));
    }

    // 2 component vectors in a 3d problem require a row of zeros.
    // do we really need this?
    if (numdf==2)
    {
      for (int inode=0; inode<numvispoints; inode++)
      {
        Write<float>(file, 0.);
      }
    }
  }
} //EnsightWriter::WriteDofResultStepForNurbs

#endif
