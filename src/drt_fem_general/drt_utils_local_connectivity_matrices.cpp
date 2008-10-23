 /*!
 \file drt_utils_local_connectivity_matrices.cpp

 \brief Provide a node numbering scheme together with a set of shape functions

 <pre>
-------------------------------------------------------------------------
                 BACI finite element library subsystem
            Copyright (2008) Technical University of Munich
              
Under terms of contract T004.008.000 there is a non-exclusive license for use
of this work by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library is proprietary software. It must not be published, distributed, 
copied or altered in any form or any media without written permission
of the copyright holder. It may be used under terms and conditions and of the
above mentioned license by or on behalf of Rolls-Royce Ltd & Co KG, Germany.

This library may solemnly used in conjunction with the BACI contact library
for purposes described in the above mentioned contract.

This library contains and makes use of software copyrighted by Sandia Corporation
and distributed under LGPL licence. Licensing does not apply to this or any
other third party software used here.

Questions? Contact Dr. Michael W. Gee (gee@lnm.mw.tum.de) 
                   or
                   Prof. Dr. Wolfgang A. Wall (wall@lnm.mw.tum.de)

http://www.lnm.mw.tum.de                   

-------------------------------------------------------------------------
<\pre>

Provided are 1D, 2D and 3D shape functions

 The surface mapping gives the node numbers such that the 2D shape functions can be used
 Nodal mappings describe the relation between volume, surface and line node numbering.
 They should be used as the only reference for such relationships.
 The corresponding graphics and a detailed description can be found in the Baci guide in the Convention chapter.
 The numbering of lower order elements is included in the higher order element, such that
 e.g. the hex8 volume element uses only the first 8 nodes of the hex27 mapping

 !!!!
 The corresponding graphics and a detailed description can be found
 in the Baci guide in the Convention chapter.
 !!!!

 \author Axel Gerstenberger
 gerstenberger@lnm.mw.tum.de
 http://www.lnm.mw.tum.de
 089 - 289-15236
 */
#ifdef CCADISCRET

#include "drt_utils_local_connectivity_matrices.H"
#include "../drt_lib/drt_dserror.H"

/*----------------------------------------------------------------------*
 |  returns the number of nodes                              a.ger 11/07|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
int DRT::UTILS::getNumberOfElementNodes(
    const DRT::Element::DiscretizationType&     distype)
{

    int numnodes = 0;

    switch(distype)
    {
    case DRT::Element::dis_none:     return 0;    break;
    case DRT::Element::point1:       return 1;    break;
    case DRT::Element::line2:        return 2;    break;
    case DRT::Element::line3:        return 3;    break;
    case DRT::Element::tri3:         return 3;    break;
    case DRT::Element::tri6:         return 6;    break;
    case DRT::Element::quad4:        return 4;    break;
    case DRT::Element::quad8:        return 8;    break;
    case DRT::Element::quad9:        return 9;    break;
    case DRT::Element::nurbs2:       return 2;    break;
    case DRT::Element::nurbs3:       return 3;    break;
    case DRT::Element::nurbs4:       return 4;    break;
    case DRT::Element::nurbs9:       return 9;    break;
    case DRT::Element::hex8:         return 8;    break;
    case DRT::Element::hex20:        return 20;   break;
    case DRT::Element::hex27:        return 27;   break;
    case DRT::Element::tet4:         return 4;    break;
    case DRT::Element::tet10:        return 10;   break;
    case DRT::Element::wedge6:       return 6;    break;
    case DRT::Element::wedge15:      return 15;   break;
    case DRT::Element::pyramid5:     return 5;    break;
    default:
        cout << DRT::DistypeToString(distype) << endl;
        dserror("discretization type %s not yet implemented", (DRT::DistypeToString(distype)).c_str());
    }

    return numnodes;
}


/*----------------------------------------------------------------------*
 |  returns the number of corner nodes                       u.may 08/07|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
int DRT::UTILS::getNumberOfElementCornerNodes(
    const DRT::Element::DiscretizationType&     distype)
{
    int numCornerNodes = 0;
    switch(distype)
    {
        case DRT::Element::hex8: case DRT::Element::hex20: case DRT::Element::hex27:
        {
            numCornerNodes = 8;
            break;
        }
        case DRT::Element::tet4: case DRT::Element::tet10:
        case DRT::Element::quad9: case DRT::Element::quad8: case DRT::Element::quad4:
        {
            numCornerNodes = 4;
            break;
        }
        case DRT::Element::tri6: case DRT::Element::tri3:
        {
            numCornerNodes = 3;
            break;
        }
        default:
            dserror("discretization type not yet implemented");
    }
    return numCornerNodes;
}



/*----------------------------------------------------------------------*
 |  returns the number of lines                              a.ger 08/07|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
int DRT::UTILS::getNumberOfElementLines(
    const DRT::Element::DiscretizationType&     distype)
{
    int numLines = 0;
    switch(distype)
    {
        case DRT::Element::hex8: case DRT::Element::hex20: case DRT::Element::hex27:
            numLines = 12;
            break;
        case DRT::Element::wedge6: case DRT::Element::wedge15:
            numLines = 9;
            break;
        case DRT::Element::tet4: case DRT::Element::tet10:
            numLines = 6;
            break;
        case DRT::Element::quad4: case DRT::Element::quad8: case DRT::Element::quad9:
            numLines = 4;
            break;
        case DRT::Element::nurbs4: case DRT::Element::nurbs9:
            numLines = 4;
            break;
        case DRT::Element::tri3: case DRT::Element::tri6:
            numLines = 3;
            break;
        default:
            dserror("discretization type not yet implemented");
    }
    return numLines;
}


/*----------------------------------------------------------------------*
 |  returns the number of lines                              a.ger 08/07|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
int DRT::UTILS::getNumberOfElementSurfaces(
    const DRT::Element::DiscretizationType&     distype)
{
    int numSurf = 0;
    switch(distype)
    {
        case DRT::Element::hex8: case DRT::Element::hex20: case DRT::Element::hex27:
            numSurf = 6;
            break;
        case DRT::Element::wedge6: case DRT::Element::wedge15:
            numSurf = 5;
            break;
        case DRT::Element::tet4: case DRT::Element::tet10:
            numSurf = 4;
            break;
        default:
            dserror("discretization type not yet implemented");
    }
    return numSurf;
}

/*----------------------------------------------------------------------*
 |  Fills a vector< vector<int> > with all nodes for         u.may 08/07|
 |  every surface for each discretization type                          |
 *----------------------------------------------------------------------*/
vector< vector<int> > DRT::UTILS::getEleNodeNumberingSurfaces(
    const DRT::Element::DiscretizationType&     distype)
{
    vector< vector<int> >   map;

    switch(distype)
    {
        case DRT::Element::hex8:
        {
            const int nSurf = 6;
            const int nNode = 4;
            vector<int> submap(nNode, 0);
            for(int i = 0; i < nSurf; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_hex27_surfaces[i][j];
            }
            break;
        }
        case DRT::Element::hex20:
        {
            const int nSurf = 6;
            const int nNode = 8;
            vector<int> submap(nNode, 0);
            for(int i = 0; i < nSurf; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_hex27_surfaces[i][j];
            }
            break;
        }
        case DRT::Element::hex27:
        {
            const int nSurf = 6;
            const int nNode = 9;
            vector<int> submap(nNode, 0);
            for(int i = 0; i < nSurf; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_hex27_surfaces[i][j];
            }
            break;
        }
        case DRT::Element::tet4:
        {
            const int nSurf = 4;
            const int nNode = 3;
            vector<int> submap(nNode, 0);
            for(int i = 0; i < nSurf; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_tet10_surfaces[i][j];
            }
            break;
        }
        case DRT::Element::tet10:
        {
            const int nSurf = 4;
            const int nNode = 6;
            vector<int> submap(nNode, 0);
            for(int i = 0; i < nSurf; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_tet10_surfaces[i][j];
            }
            break;
        }
        case DRT::Element::wedge6:
        {
            // quad surfaces
            const int nqSurf = 3;
            const int nqNode = 4;
            vector<int> submapq(nqNode, 0);
            for(int i = 0; i < nqSurf; i++)
            {
                map.push_back(submapq);
                for(int j = 0; j < nqNode; j++)
                    map[i][j] = eleNodeNumbering_wedge15_quadsurfaces[i][j];
            }

            // tri surfaces
            const int ntSurf = 2;
            const int ntNode = 3;
            vector<int> submapt(ntNode, 0);
            for(int i = 0; i < ntSurf; i++)
            {
                map.push_back(submapt);
                for(int j = 0; j < ntNode; j++)
                    map[i+nqSurf][j] = eleNodeNumbering_wedge15_trisurfaces[i][j];
            }
            break;
        }
        case DRT::Element::pyramid5:
        {
          // quad surfaces
          const int nqSurf = 1;
          const int nqNode = 4;
          vector<int> submapq(nqNode, 0);
          for(int i = 0; i < nqSurf; i++)
          {
              map.push_back(submapq);
              for(int j = 0; j < nqNode; j++)
                  map[i][j] = eleNodeNumbering_pyramid5_quadsurfaces[i][j];
          }

          // tri surfaces
          const int ntSurf = 4;
          const int ntNode = 3;
          vector<int> submapt(ntNode, 0);
          for(int i = 0; i < ntSurf; i++)
          {
              map.push_back(submapt);
              for(int j = 0; j < ntNode; j++)
                  map[i+1][j] = eleNodeNumbering_pyramid5_trisurfaces[i][j];
          }
          break;
        }
        default:
            dserror("discretizationtype is not yet implemented");
    }

    return map;
}





/*----------------------------------------------------------------------*
 |  Fills a vector< vector<int> > with all nodes for         u.may 08/07|
 |  every line for each discretization type                             |
 *----------------------------------------------------------------------*/
vector< vector<int> > DRT::UTILS::getEleNodeNumberingLines(
    const DRT::Element::DiscretizationType&     distype)
{
    vector< vector<int> >  map;

    switch(distype)
    {
        case DRT::Element::hex8:
        {
            const int nLine = 12;
            const int nNode = 2;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_hex27_lines[i][j];
            }
            break;
        }
        case DRT::Element::hex20: case DRT::Element::hex27:
        {
            const int nLine = 12;
            const int nNode = 3;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_hex27_lines[i][j];
            }
            break;
        }
        case DRT::Element::tet4:
        {
            const int nLine = 6;
            const int nNode = 2;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_tet10_lines[i][j];
            }
            break;
        }
        case DRT::Element::tet10:
        {
            const int nLine = 6;
            const int nNode = 3;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_tet10_lines[i][j];
            }
            break;
        }
        case DRT::Element::quad9:
        case DRT::Element::quad8:
        {
            const int nLine = 4;
            const int nNode = 3;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_quad9_lines[i][j];
            }
            break;
        }
        case DRT::Element::nurbs9:
        {
            const int nLine = 4;
            const int nNode = 3;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_nurbs9_lines[i][j];
            }
            break;
        }
        case DRT::Element::quad4:
        {
            const int nLine = 4;
            const int nNode = 2;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_quad9_lines[i][j];
            }
            break;
        }
        case DRT::Element::nurbs4:
        {
            const int nLine = 4;
            const int nNode = 2;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_nurbs4_lines[i][j];
            }
            break;
        }
        case DRT::Element::tri6:
        {
            const int nLine = 3;
            const int nNode = 3;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_tri6_lines[i][j];
            }
            break;
        }
        case DRT::Element::tri3:
        {
            const int nLine = 3;
            const int nNode = 2;
            vector<int> submap(nNode, -1);

            for(int i = 0; i < nLine; i++)
            {
                map.push_back(submap);
                for(int j = 0; j < nNode; j++)
                    map[i][j] = eleNodeNumbering_tri6_lines[i][j];
            }
            break;
        }
        default:
            dserror("discretizationtype is not yet implemented");
    }

    return map;
}






/*----------------------------------------------------------------------*
 |  Fills a vector< vector<int> > with all surfaces for      u.may 08/07|
 |  every line for each discretization type                             |
 *----------------------------------------------------------------------*/
vector< vector<int> > DRT::UTILS::getEleNodeNumbering_lines_surfaces(
    const DRT::Element::DiscretizationType&     distype)
{
    int nLine;
    int nSurf;

    vector< vector<int> > map;

    if(distype == DRT::Element::hex8 ||  distype == DRT::Element::hex20 || distype == DRT::Element::hex27)
    {
        nLine = 12;
        nSurf = 2;
        vector<int> submap(nSurf, 0);
        for(int i = 0; i < nLine; i++)
        {
            map.push_back(submap);
            for(int j = 0; j < nSurf; j++)
                map[i][j] = eleNodeNumbering_hex27_lines_surfaces[i][j];
        }
    }
    else if(distype == DRT::Element::tet4 ||  distype == DRT::Element::tet10)
    {
        nLine = 6;
        nSurf = 2;
        vector<int> submap(nSurf, 0);
        for(int i = 0; i < nLine; i++)
        {
            map.push_back(submap);
            for(int j = 0; j < nSurf; j++)
                map[i][j] = eleNodeNumbering_tet10_lines_surfaces[i][j];
        }
    }
    else
        dserror("discretizationtype not yet implemented");


    return map;

}




/*----------------------------------------------------------------------*
 |  Fills a vector< vector<int> > with all lines for         u.may 08/08|
 |  every node for each discretization type                             |
 *----------------------------------------------------------------------*/
vector< vector<int> > DRT::UTILS::getEleNodeNumbering_nodes_lines(
    const DRT::Element::DiscretizationType      distype)
{
    vector< vector<int> >   map;

    const int nCornerNode = getNumberOfElementCornerNodes(distype);

    if(distype == DRT::Element::hex8 ||  distype == DRT::Element::hex20 || distype == DRT::Element::hex27)
    {
        const int nLine = 3;
        vector<int> submap(nLine, 0);
        for(int i = 0; i < nCornerNode; i++)
        {
            map.push_back(submap);
            for(int j = 0; j < nLine; j++)
                map[i][j] = eleNodeNumbering_hex27_nodes_lines[i][j];
        }
    }
    else if(distype == DRT::Element::tet4 ||  distype == DRT::Element::tet10)
    {
        const int nLine = 3;
        vector<int> submap(nLine, 0);
        for(int i = 0; i < nCornerNode; i++)
        {
            map.push_back(submap);
            for(int j = 0; j < nLine; j++)
                map[i][j] = eleNodeNumbering_tet10_nodes_lines[i][j];
        }
    }
    else
        dserror("discretizationtype not yet implemented");

    return map;
}



/*----------------------------------------------------------------------*
 |  Fills a vector< vector<int> > with all surfaces for      u.may 08/07|
 |  every node for each discretization type                             |
 *----------------------------------------------------------------------*/
vector< vector<int> > DRT::UTILS::getEleNodeNumbering_nodes_surfaces(
    const DRT::Element::DiscretizationType      distype)
{
    const int nCornerNode = getNumberOfElementCornerNodes(distype);
    int nSurf;

    vector< vector<int> >   map;

    if(distype == DRT::Element::hex8 ||  distype == DRT::Element::hex20 || distype == DRT::Element::hex27)
    {
        nSurf = 3;
        vector<int> submap(nSurf, 0);
        for(int i = 0; i < nCornerNode; i++)
        {
            map.push_back(submap);
            for(int j = 0; j < nSurf; j++)
                map[i][j] = eleNodeNumbering_hex27_nodes_surfaces[i][j];
        }
    }
    else if(distype == DRT::Element::tet4 ||  distype == DRT::Element::tet10)
    {
        nSurf = 3;
        vector<int> submap(nSurf, 0);
        for(int i = 0; i < nCornerNode; i++)
        {
            map.push_back(submap);
            for(int j = 0; j < nSurf; j++)
                map[i][j] = eleNodeNumbering_tet10_nodes_surfaces[i][j];
        }
    }
    else
        dserror("discretizationtype not yet implemented");

    return map;

}



/*----------------------------------------------------------------------*
 |  Fills a vector< vector<double> > with positions in reference coordinates
 |                                                           u.may 08/07|
 *----------------------------------------------------------------------*/
vector< vector<double> > DRT::UTILS::getEleNodeNumbering_nodes_reference(
    const DRT::Element::DiscretizationType      distype)
{
    const int nNode = getNumberOfElementNodes(distype);
    const int dim = getDimension(distype);
    vector< vector<double> >   map(nNode, vector<double>(dim,0.0));

    switch(distype)
    {
        case DRT::Element::quad4:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_quad9_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::quad8:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_quad9_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::quad9:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_quad9_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::tri3:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_tri6_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::tri6:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_tri6_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::hex8:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_hex27_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::hex20:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_hex27_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::hex27:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_hex27_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::tet4:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_tet10_nodes_reference[inode][isd];
            }
            break;
        }
        case DRT::Element::tet10:
        {
            for(int inode = 0; inode < nNode; inode++)
            {
                for(int isd = 0; isd < dim; isd++)
                    map[inode][isd] = eleNodeNumbering_tet10_nodes_reference[inode][isd];
            }
            break;
        }
        default:
            dserror("discretizationtype not yet implemented");
    }

    return map;
}



/*----------------------------------------------------------------------*
 |  Returns a vector with surface ID s a point is lying on   u.may 08/07|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
vector<int> DRT::UTILS::getSurfaces(
    const blitz::TinyVector<double,3>&          rst,
    const DRT::Element::DiscretizationType      distype)
{
    const double TOL = 1e-7;
    vector<int> surfaces;

    if(distype == DRT::Element::hex8 ||  distype == DRT::Element::hex20 || distype == DRT::Element::hex27)
    {
        if(fabs(rst(0)-1.0) < TOL)      surfaces.push_back(2);
        if(fabs(rst(0)+1.0) < TOL)      surfaces.push_back(4);
        if(fabs(rst(1)-1.0) < TOL)      surfaces.push_back(3);
        if(fabs(rst(1)+1.0) < TOL)      surfaces.push_back(1);
        if(fabs(rst(2)-1.0) < TOL)      surfaces.push_back(5);
        if(fabs(rst(2)+1.0) < TOL)      surfaces.push_back(0);
    }
    else if(distype == DRT::Element::tet4 ||  distype == DRT::Element::tet10 )
    {
        const double tetcoord = rst(0)+rst(1)+rst(2);
        if(fabs(rst(1))         < TOL)  surfaces.push_back(0);
        if(fabs(tetcoord-1.0)   < TOL)  surfaces.push_back(1);
        if(fabs(rst(0))         < TOL)  surfaces.push_back(2);
        if(fabs(rst(2))         < TOL)  surfaces.push_back(3);
    }
    else
        dserror("discretization type not yet implemented");

    return surfaces;
}


/*----------------------------------------------------------------------*
 |  Returns a vector with surface ID s a point is lying on     u.may 07/08|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
vector<int> DRT::UTILS::getLines(
    const blitz::TinyVector<double,3>&          rst,
    const DRT::Element::DiscretizationType      distype)
{

    const double TOL = 1e-7;
    vector<int> lines;

    if(distype == DRT::Element::hex8 ||  distype == DRT::Element::hex20 || distype == DRT::Element::hex27)
    {
        if(fabs(rst(1)+1.0) < TOL && fabs(rst(2)+1.0) < TOL)      lines.push_back(0);  // -s -t
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(2)+1.0) < TOL)      lines.push_back(1);  // +r -t
        if(fabs(rst(1)-1.0) < TOL && fabs(rst(2)+1.0) < TOL)      lines.push_back(2);  // +s -t
        if(fabs(rst(0)+1.0) < TOL && fabs(rst(2)+1.0) < TOL)      lines.push_back(3);  // -r -t

        if(fabs(rst(0)+1.0) < TOL && fabs(rst(1)+1.0) < TOL)      lines.push_back(4);  // -r -s
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(1)+1.0) < TOL)      lines.push_back(5);  // +r -s
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(1)-1.0) < TOL)      lines.push_back(6);  // +r +s
        if(fabs(rst(0)+1.0) < TOL && fabs(rst(1)-1.0) < TOL)      lines.push_back(7);  // -r +s

        if(fabs(rst(1)+1.0) < TOL && fabs(rst(2)-1.0) < TOL)      lines.push_back(8);  // -s +t
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(2)-1.0) < TOL)      lines.push_back(9);  // +r +t
        if(fabs(rst(1)-1.0) < TOL && fabs(rst(2)-1.0) < TOL)      lines.push_back(10); // +s +t
        if(fabs(rst(0)+1.0) < TOL && fabs(rst(2)-1.0) < TOL)      lines.push_back(11); // -r +t
    }
    else if(distype == DRT::Element::tet4 ||  distype == DRT::Element::tet10)
    {
        const double tcoord = 1.0 - rst(0) - rst(1) - rst(2);
        if(fabs(rst(1)) < TOL && fabs(rst(2)) < TOL)      lines.push_back(0);
        if(fabs(rst(2)) < TOL && fabs(tcoord) < TOL)      lines.push_back(1);
        if(fabs(rst(0)) < TOL && fabs(rst(2)) < TOL)      lines.push_back(2);
        if(fabs(rst(0)) < TOL && fabs(rst(1)) < TOL)      lines.push_back(3);
        if(fabs(rst(1)) < TOL && fabs(tcoord) < TOL)      lines.push_back(4);
        if(fabs(rst(0)) < TOL && fabs(tcoord) < TOL)      lines.push_back(5);
    }
    else
        dserror("discretization type not yet implemented");

    return lines;
}



/*----------------------------------------------------------------------*
 |  Returns the node ID a point is lying on                  u.may 07/08|
 |  for each discretization type                                        |
 *----------------------------------------------------------------------*/
int DRT::UTILS::getNode(
    const blitz::TinyVector<double,3>&          rst,
    const DRT::Element::DiscretizationType      distype)
{
    const double TOL = 1e-7;
    int node = -1;

    if(distype == DRT::Element::hex8 ||  distype == DRT::Element::hex20 || distype == DRT::Element::hex27)
    {
        if(fabs(rst(0)+1.0) < TOL && fabs(rst(1)+1.0) < TOL && fabs(rst(2)+1.0) < TOL)      node = 0;  // -r -s -t
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(1)+1.0) < TOL && fabs(rst(2)+1.0) < TOL)      node = 1;  // +r -s -t
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(1)-1.0) < TOL && fabs(rst(2)+1.0) < TOL)      node = 2;  // +r +s -t
        if(fabs(rst(0)+1.0) < TOL && fabs(rst(1)-1.0) < TOL && fabs(rst(2)+1.0) < TOL)      node = 3;  // -r +s -t

        if(fabs(rst(0)+1.0) < TOL && fabs(rst(1)+1.0) < TOL && fabs(rst(2)-1.0) < TOL)      node = 4;  // -r -s +t
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(1)+1.0) < TOL && fabs(rst(2)-1.0) < TOL)      node = 5;  // +r -s +t
        if(fabs(rst(0)-1.0) < TOL && fabs(rst(1)-1.0) < TOL && fabs(rst(2)-1.0) < TOL)      node = 6;  // +r +s +t
        if(fabs(rst(0)+1.0) < TOL && fabs(rst(1)-1.0) < TOL && fabs(rst(2)-1.0) < TOL)      node = 7 ; // -r +s +t

    }
    else
        dserror("discretization type not yet implemented");

    return node;
}




/*----------------------------------------------------------------------*
 |  Returns a vector with coordinates in the reference       u.may 08/07|
 |  system of the cutter element                                        |
 |  according to the node ID for each discretization type               |
 *----------------------------------------------------------------------*/
vector<double> DRT::UTILS::getNodeCoordinates(  const int                                   nodeId,
                                                const DRT::Element::DiscretizationType      distype)
{

    vector<double> coord(3,0.0);

    if(distype == DRT::Element::quad4 ||  distype == DRT::Element::quad8 || distype == DRT::Element::quad9)
    {
        switch(nodeId)
        {
            case 0:
            {
                coord[0] = -1.0;
                coord[1] = -1.0;
                break;
            }
            case 1:
            {
                coord[0] =  1.0;
                coord[1] = -1.0;
                break;
            }
            case 2:
            {
                coord[0] =  1.0;
                coord[1] =  1.0;
                break;
            }
            case 3:
            {
                coord[0] = -1.0;
                coord[1] =  1.0;
                break;
            }
            default:
                dserror("node number not correct");
        }
        coord[2] = 0.0;
    }
    else if(distype == DRT::Element::tri3 ||  distype == DRT::Element::tri6)
    {
        switch(nodeId)
        {
            case 0:
            {
                coord[0] = 0.0;
                coord[1] = 0.0;
                break;
            }
            case 1:
            {
                coord[0] =  1.0;
                coord[1] =  0.0;
                break;
            }
            case 2:
            {
                coord[0] =  0.0;
                coord[1] =  1.0;
                break;
            }
            default:
                dserror("node number not correct");
        }
        coord[2] = 0.0;
    }
    else dserror("discretizationtype is not yet implemented");

    return coord;
}



/*----------------------------------------------------------------------*
 |  Returns a vector with coordinates in the reference       u.may 08/07|
 |  system of the cutter element                                        |
 |  according to the line ID for each discretization type               |
 *----------------------------------------------------------------------*/
vector<double> DRT::UTILS::getLineCoordinates(
    const int                                   lineId,
    const double                                lineCoord,
    const DRT::Element::DiscretizationType      distype)
{

  vector<double> coord(3,0.0);
  if(distype == DRT::Element::quad4 ||  distype == DRT::Element::quad8 || distype == DRT::Element::quad9)
  {
    // change minus sign if you change the line numbering
    switch(lineId)
    {
      case 0:
      {
        coord[0] = lineCoord;
        coord[1] = -1.0;
        break;
      }
      case 1:
      {
        coord[0] = 1.0;
        coord[1] = lineCoord;
        break;
      }
      case 2:
      {
        coord[0] =  -lineCoord;
        coord[1] =  1.0;
        break;
      }
      case 3:
      {
        coord[0] = -1.0;
        coord[1] = -lineCoord;
        break;
      }
      default:
          dserror("node number not correct");
    }
    coord[2] =  0.0;
  }
  else if(distype == DRT::Element::tri3 ||  distype == DRT::Element::tri6)
  {
    // change minus sign if you change the line numbering
    switch(lineId)
    {
      case 0:
      {
        coord[0] = (lineCoord+1)*0.5;
        coord[1] = 0.0;
        break;
      }
      case 1:
      {
        coord[0] = 1.0;
        coord[1] = (lineCoord+1)*0.5;
        break;
      }
      case 2:
      {
        coord[0] =  1.0 - (lineCoord+1)*0.5;
        coord[1] =  (lineCoord+1)*0.5;
        break;
      }
      default:
        dserror("node number not correct");

      }
      coord[2] =  0.0;
  }
  else
    dserror("discretization type not yet implemented");

  return coord;
}



/*----------------------------------------------------------------------*
 |  returns the index of a higher order                      u.may 09/07|
 |  element node index lying between two specified corner               |
 |  node indices for each discretizationtype                            |
 *----------------------------------------------------------------------*/
int DRT::UTILS::getHigherOrderIndex(
    const int                                   index1,
    const int                                   index2,
    const DRT::Element::DiscretizationType      distype )
{

    int higherOrderIndex = 0;

    switch(distype)
    {
        case DRT::Element::tet10:
        {
            if     ( (index1 == 0 && index2 == 1) || (index1 == 1 && index2 == 0) )      higherOrderIndex = 4;
            else if( (index1 == 1 && index2 == 2) || (index1 == 2 && index2 == 1) )      higherOrderIndex = 5;
            else if( (index1 == 2 && index2 == 0) || (index1 == 0 && index2 == 2) )      higherOrderIndex = 6;
            else if( (index1 == 0 && index2 == 3) || (index1 == 3 && index2 == 0) )      higherOrderIndex = 7;
            else if( (index1 == 1 && index2 == 3) || (index1 == 3 && index2 == 1) )      higherOrderIndex = 8;
            else if( (index1 == 2 && index2 == 3) || (index1 == 3 && index2 == 2) )      higherOrderIndex = 9;
            else dserror("no valid tet10 edge found");
            break;
        }
        case DRT::Element::quad9:
        {
            if     ( (index1 == 0 && index2 == 1) || (index1 == 1 && index2 == 0) )      higherOrderIndex = 4;
            else if( (index1 == 1 && index2 == 2) || (index1 == 2 && index2 == 1) )      higherOrderIndex = 5;
            else if( (index1 == 2 && index2 == 3) || (index1 == 3 && index2 == 2) )      higherOrderIndex = 6;
            else if( (index1 == 3 && index2 == 0) || (index1 == 0 && index2 == 3) )      higherOrderIndex = 7;
            else dserror("no valid quad9 edge found");
            break;
        }
        case DRT::Element::tri6:
        {
            if     ( (index1 == 0 && index2 == 1) || (index1 == 1 && index2 == 0) )      higherOrderIndex = 3;
            else if( (index1 == 1 && index2 == 2) || (index1 == 2 && index2 == 1) )      higherOrderIndex = 4;
            else if( (index1 == 2 && index2 == 0) || (index1 == 0 && index2 == 2) )      higherOrderIndex = 5;
            else dserror("no valid tri6 edge found");
            break;
        }
        default:
            dserror("discretizationtype not yet implemented");
    }
    return higherOrderIndex;
}



///*----------------------------------------------------------------------*
// |  returns the dimension of the element parameter space     u.may 10/07|
// *----------------------------------------------------------------------*/
//int DRT::UTILS::getDimension(
//    const DRT::Element*   element)
//{
//    return getDimension(element->Shape());
//}

/*----------------------------------------------------------------------*
 |  returns the dimension of the element-shape                 bos 01/08|
 *----------------------------------------------------------------------*/
int DRT::UTILS::getDimension(const DRT::Element::DiscretizationType distype)
{
    int dim = 0;

    switch(distype)
    {
        case DRT::Element::line2  : dim = DisTypeToDim<DRT::Element::line2>::dim; break;
        case DRT::Element::line3  : dim = DisTypeToDim<DRT::Element::line3>::dim; break;
        case DRT::Element::nurbs2 : dim = DisTypeToDim<DRT::Element::nurbs2>::dim; break;
        case DRT::Element::nurbs3 : dim = DisTypeToDim<DRT::Element::nurbs3>::dim; break;
        case DRT::Element::quad4  : dim = DisTypeToDim<DRT::Element::quad4>::dim; break;
        case DRT::Element::quad8  : dim = DisTypeToDim<DRT::Element::quad8>::dim; break;
        case DRT::Element::quad9  : dim = DisTypeToDim<DRT::Element::quad9>::dim; break;
        case DRT::Element::tri3   : dim = DisTypeToDim<DRT::Element::tri3>::dim; break;
        case DRT::Element::tri6   : dim = DisTypeToDim<DRT::Element::tri6>::dim; break;
        case DRT::Element::nurbs4 : dim = DisTypeToDim<DRT::Element::nurbs4>::dim; break;
        case DRT::Element::nurbs9 : dim = DisTypeToDim<DRT::Element::nurbs9>::dim; break;
        case DRT::Element::hex8   : dim = DisTypeToDim<DRT::Element::hex8>::dim; break;
        case DRT::Element::hex20  : dim = DisTypeToDim<DRT::Element::hex20>::dim; break;
        case DRT::Element::hex27  : dim = DisTypeToDim<DRT::Element::hex27>::dim; break;
        case DRT::Element::tet4   : dim = DisTypeToDim<DRT::Element::tet4>::dim; break;
        case DRT::Element::tet10  : dim = DisTypeToDim<DRT::Element::tet10>::dim; break;
        default:
            dserror("discretization type is not yet implemented");
    }
    return dim;
}


/*----------------------------------------------------------------------*
 |  returns the order of the element-shape                   u.may 06/08|
 *----------------------------------------------------------------------*/
int DRT::UTILS::getOrder(const DRT::Element::DiscretizationType distype)
{
    int order = 0;

    switch(distype)
    {
        case DRT::Element::line2  : order = DisTypeToEdgeOrder<DRT::Element::line2>::order; break;
        case DRT::Element::line3  : order = DisTypeToEdgeOrder<DRT::Element::line3>::order; break;
        case DRT::Element::nurbs2 : order = DisTypeToEdgeOrder<DRT::Element::nurbs2>::order; break;
        case DRT::Element::nurbs3 : order = DisTypeToEdgeOrder<DRT::Element::nurbs3>::order; break;
        case DRT::Element::quad4  : order = DisTypeToEdgeOrder<DRT::Element::quad4>::order; break;
        case DRT::Element::quad8  : order = DisTypeToEdgeOrder<DRT::Element::quad8>::order; break;
        case DRT::Element::quad9  : order = DisTypeToEdgeOrder<DRT::Element::quad9>::order; break;
        case DRT::Element::tri3   : order = DisTypeToEdgeOrder<DRT::Element::tri3>::order; break;
        case DRT::Element::tri6   : order = DisTypeToEdgeOrder<DRT::Element::tri6>::order; break;
        case DRT::Element::nurbs4 : order = DisTypeToEdgeOrder<DRT::Element::nurbs4>::order; break;
        case DRT::Element::nurbs9 : order = DisTypeToEdgeOrder<DRT::Element::nurbs9>::order; break;
        case DRT::Element::hex8 :   order = DisTypeToEdgeOrder<DRT::Element::hex8>::order; break;
        case DRT::Element::hex20 :  order = DisTypeToEdgeOrder<DRT::Element::hex20>::order; break;
        case DRT::Element::hex27 :  order = DisTypeToEdgeOrder<DRT::Element::hex27>::order; break;
        case DRT::Element::tet4 :   order = DisTypeToEdgeOrder<DRT::Element::tet4>::order; break;
        case DRT::Element::tet10 :  order = DisTypeToEdgeOrder<DRT::Element::tet10>::order; break;
        default:
            dserror("discretization type is not yet implemented");
    }
    return order;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double DRT::UTILS::getSizeInLocalCoordinates(
    const DRT::Element::DiscretizationType     distype)
{
    double size = 0.0;
    switch(distype)
    {
        case DRT::Element::hex8:
        case DRT::Element::hex20:
        case DRT::Element::hex27:
            size = 8.0;
            break;
        case DRT::Element::tet4:
        case DRT::Element::tet10:
            size = 1.0/6.0;
            break;
        case DRT::Element::quad4:
        case DRT::Element::quad8:
        case DRT::Element::quad9:
            size = 4.0;
            break;
        case DRT::Element::tri3:
        case DRT::Element::tri6:
            size = 0.5;
            break;
        case DRT::Element::line2:
        case DRT::Element::line3:
            size = 2.0;
            break;
        default:
            dserror("discretization type not yet implemented");
    };

    return size;
}

#endif  // #ifdef CCADISCRET
