/*!
\file integrationcell.cpp

\brief integration cell classes for domain and boundary integration

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
*/

#ifdef CCADISCRET

#include "integrationcell.H"
#include <string>
#include <sstream>
#include <blitz/array.h>
#include "../drt_lib/drt_node.H"
#include "../drt_lib/drt_utils_integration.H"
#include "../drt_lib/drt_utils_fem_shapefunctions.H"
#include "../drt_lib/drt_utils_local_connectivity_matrices.H"

using namespace std;
using namespace XFEM;


//! little helper function
static BlitzMat ConvertPosArrayToBlitz(
        const vector<vector<double> >&         pos_array,
        const DRT::Element::DiscretizationType distype,
        const int                              dim
        )
{
    const int numnode = DRT::UTILS::getNumberOfElementNodes(distype);
    BlitzMat pos_array_blitz(dim,numnode,blitz::ColumnMajorArray<2>());
    for (int inode=0; inode<numnode; ++inode)
    {
        for (int isd=0; isd<dim; ++isd)
        {
            pos_array_blitz(isd,inode) = pos_array[inode][isd];
        }
    }    
    return pos_array_blitz;
}


//
//  ctor
//
IntCell::IntCell(
        const DRT::Element::DiscretizationType distype) :
            distype_(distype)
{
    return;
}

/*----------------------------------------------------------------------*
 |  copy-ctor                                                mwgee 11/06|
 *----------------------------------------------------------------------*/
IntCell::IntCell(
        const IntCell& old) : 
            distype_(old.distype_)
{
    return;   
}
 
/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 11/06|
 *----------------------------------------------------------------------*/
IntCell::~IntCell()
{
  return;
}

////
////  get coordinates
////
//vector< vector<double> > IntCell::NodalPosXiDomain() const
//{
//    dserror("no default implementation is given");
//    vector<vector<double> > dummy;
//    return dummy;
//}

////
////  get coordinates in physical space
////
//vector< vector<double> > IntCell::NodalPosXYZ(const DRT::Element& ele) const
//{
//    dserror("no default implementation is given");
//    vector<vector<double> > dummy;
//    return dummy;
//}

//
// Print method
//
std::string IntCell::toString() const
{
  return "";
}


//
//  ctor
//
DomainIntCell::DomainIntCell(
        const DRT::Element::DiscretizationType distype,
        const vector< vector<double> >& domainCoordinates) :
            IntCell(distype),
            nodalpos_xi_domain_blitz_(ConvertPosArrayToBlitz(domainCoordinates, distype, 3))
{
    return;
}


//
//  ctor
//
DomainIntCell::DomainIntCell(
        const DRT::Element::DiscretizationType distype,
        const BlitzMat&                        domainCoordinates) :
            IntCell(distype),
            nodalpos_xi_domain_blitz_(domainCoordinates)
{
    return;
}

        
//
//  ctor for dummy cells
//
DomainIntCell::DomainIntCell(
        const DRT::Element::DiscretizationType distype) :
            IntCell(distype),
            nodalpos_xi_domain_blitz_(ConvertPosArrayToBlitz(GetDefaultCoordinates(distype), distype, 3))
{
    return;
}
        
/*----------------------------------------------------------------------*
 |  copy-ctor                                                mwgee 11/06|
 *----------------------------------------------------------------------*/
DomainIntCell::DomainIntCell(
        const DomainIntCell& old) :
            IntCell(old),
            nodalpos_xi_domain_blitz_(old.nodalpos_xi_domain_blitz_)
{
    return;   
}
     
string DomainIntCell::toString() const
{
    stringstream s;
    s << "DomainIntCell" << endl;
    s << nodalpos_xi_domain_blitz_ << endl;
//    MCONST_FOREACH(vector< vector<double> >, coordinate, nodalpos_xi_domain_)
//    {
//        s << "[";
//        MPFOREACH(vector<double>, val, coordinate)
//        {
//            s << *val << " ";
//        };
//        s << "]" << endl;
//    };
    return s.str();
}

// set element nodal coordinates according to given distype
vector<vector<double> > DomainIntCell::GetDefaultCoordinates(
        const DRT::Element::DiscretizationType distype) const
{
    vector<vector<double> > coords;
    const int nsd = 3;
    const int numnode = DRT::UTILS::getNumberOfElementNodes(distype);
    
    for(int j = 0; j < numnode; ++j){
        vector<double> coord(nsd);
        switch (distype){
        case DRT::Element::hex8: case DRT::Element::hex20: case DRT::Element::hex27:
        {
            for(int k = 0; k < nsd; ++k){
                coord[k] = DRT::UTILS::eleNodeNumbering_hex27_nodes_reference[j][k];
                };
            break;
        }
        case DRT::Element::tet4: case DRT::Element::tet10:
        {
            for(int k = 0; k < nsd; ++k){
                coord[k] = DRT::UTILS::eleNodeNumbering_tet10_nodes_reference[j][k];
                }
            break;
        }
        default:
            dserror("not supported in integrationcells. can be coded easily... ;-)");
        }
        coords.push_back(coord);  
    }
    return coords;
}

//
// return the center of the cell in physical coordinates
//
BlitzVec3 DomainIntCell::GetPhysicalCenterPosition(const DRT::Element& ele) const
{
    // number of space dimensions
    //const int nsd = 3;
    
    // physical positions of cell nodes
    const BlitzMat physcoord = this->NodalPosXYZ(ele);
    
    // center in local coordinates
    static BlitzVec localcenterpos(3);
    localcenterpos = DRT::UTILS::getLocalCenterPosition(this->Shape());

    // shape functions
    const BlitzVec funct(DRT::UTILS::shape_function_3D(
            localcenterpos(0),
            localcenterpos(1),
            localcenterpos(2),
            this->Shape()));
    
    //interpolate position to x-space
    //blitz::firstIndex isd;
    //blitz::secondIndex inode;
    //static BlitzVec x_interpol(3);
    //x_interpol = blitz::sum(physcoord(isd,inode)*funct(inode),inode);
    
    static BlitzVec3 x_interpol;
    for (int isd = 0; isd < 3; ++isd)
    {
        x_interpol(isd) = 0.0;
        for (int inode = 0; inode < DRT::UTILS::getNumberOfElementNodes(this->Shape()); ++inode)
        {
            x_interpol(isd) += funct(inode)*physcoord(isd,inode);
        }
    }
    
    return x_interpol;
}


//
//  ctor
//
BoundaryIntCell::BoundaryIntCell(
        const DRT::Element::DiscretizationType    distype,
        const int                                 surface_ele_gid,
        const vector< vector<double> >&           domainCoordinates,
        const vector< vector<double> >&           boundaryCoordinates) :
            IntCell(distype),
            surface_ele_gid_(surface_ele_gid),
            nodalpos_xi_domain_blitz_(ConvertPosArrayToBlitz(domainCoordinates, distype, 3)),
            nodalpos_xi_boundary_blitz_(ConvertPosArrayToBlitz(boundaryCoordinates, distype, 2))
{
    return;
}

//
//  ctor
//
BoundaryIntCell::BoundaryIntCell(
        const DRT::Element::DiscretizationType    distype,
        const int                                 surface_ele_gid,
        const BlitzMat&                           domainCoordinates,
        const BlitzMat&                           boundaryCoordinates) :
            IntCell(distype),
            surface_ele_gid_(surface_ele_gid),
            nodalpos_xi_domain_blitz_(domainCoordinates),
            nodalpos_xi_boundary_blitz_(boundaryCoordinates)
{
    return;
}
        
        
/*----------------------------------------------------------------------*
 |  copy-ctor                                                mwgee 11/06|
 *----------------------------------------------------------------------*/
BoundaryIntCell::BoundaryIntCell(
        const BoundaryIntCell& old) :
            IntCell(old),
            surface_ele_gid_(old.surface_ele_gid_),
            nodalpos_xi_domain_blitz_(old.nodalpos_xi_domain_blitz_),
            nodalpos_xi_boundary_blitz_(old.nodalpos_xi_boundary_blitz_)
{
    return;   
}
     
string BoundaryIntCell::toString() const
{
    stringstream s;
    s << "BoundaryIntCell" << endl;
    s << nodalpos_xi_domain_blitz_ << endl;
//    MCONST_FOREACH(vector< vector<double> >, coordinate, nodalpos_xi_domain_)
//    {
//        s << "[";
//        MPFOREACH(vector<double>, val, coordinate)
//        {
//            s << *val << " ";
//        };
//        s << "]" << endl;
//    };
    return s.str();
}

#endif  // #ifdef CCADISCRET
