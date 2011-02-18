/*!
\file enrichment_utils.cpp

\brief describes the enrichment types and classes

<pre>
Maintainer: Axel Gerstenberger
            gerstenberger@lnm.mw.tum.de
            http://www.lnm.mw.tum.de
            089 - 289-15236
</pre>
*/
#ifdef CCADISCRET

#include <string>
#include <sstream>

#include "enrichment_utils.H"
#include "../drt_combust/combust_defines.H"
#include "../drt_lib/drt_dserror.H"
#include "../drt_fem_general/drt_utils_integration.H"
#include "../drt_geometry/integrationcell_coordtrafo.H"
#include "../drt_geometry/intersection_service.H"
#include "../drt_geometry/position_array.H"


/*----------------------------------------------------------------------*
 | constructor called for xfsi problems (void enrichment)            ag |
 *----------------------------------------------------------------------*/
XFEM::ElementEnrichmentValues::ElementEnrichmentValues(
        const DRT::Element&                    ele,
        XFEM::InterfaceHandle*      ih,                ///< interface information
        const XFEM::ElementDofManager&         dofman,
        const LINALG::Matrix<3,1>&             actpos,
        const bool                             boundary_integral,
        const int                              boundary_label
        ) :
          ele_(ele),
          dofman_(dofman)
{
    enrvals_.clear();
    enrvalnodes_.clear();  // not used for void enrichment
    enrvalderxy_.clear();  // not used for void enrichment
    enrvalderxy2_.clear(); // not used for void enrichment

    const std::set<XFEM::Enrichment>& enrset(dofman.getUniqueEnrichments());
    for (std::set<XFEM::Enrichment>::const_iterator enriter = enrset.begin(); enriter != enrset.end(); ++enriter)
    {
        XFEM::Enrichment::ApproachFrom   approachdirection = XFEM::Enrichment::approachUnknown;
        if (boundary_integral and enriter->XFEMConditionLabel() == boundary_label)
          approachdirection = XFEM::Enrichment::approachFromPlus;
        else
          approachdirection = XFEM::Enrichment::approachUnknown;

        const double enrval = enriter->EnrValue(actpos, *ih, approachdirection);
        enrvals_[*enriter] = enrval;
    }
    return;
}


/*----------------------------------------------------------------------*
 | interpolate field (for boundary discret. based XFEM problems)     ag |
 *----------------------------------------------------------------------*/
void XFEM::computeScalarCellNodeValuesFromNodalUnknowns(
  const DRT::Element&                   ele,
  XFEM::InterfaceHandle*                ih,
  const XFEM::ElementDofManager&        dofman,
  const GEO::DomainIntCell&             cell,
  const XFEM::PHYSICS::Field            field,
  const LINALG::SerialDenseVector&      elementvalues,
  LINALG::SerialDenseVector&            cellvalues)
{
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  const XFEM::ElementEnrichmentValues enrvals(
        ele,
        ih,
        dofman,
        cell.GetPhysicalCenterPosition(),
        false, -1);

  cellvalues.Zero();
  for (int inen = 0; inen < cell.NumNode(); ++inen)
  {
    LINALG::SerialDenseVector funct(ele.NumNode());
    // fill shape functions
    DRT::UTILS::shape_function_3D(funct,
      nodalPosXiDomain(0,inen),
      nodalPosXiDomain(1,inen),
      nodalPosXiDomain(2,inen),
      ele.Shape());

    const int numparam  = dofman.NumDofPerField(field);
    LINALG::SerialDenseVector enr_funct(numparam);
    enrvals.ComputeEnrichedNodalShapefunction(field, funct, enr_funct);
    // interpolate value
    for (int iparam = 0; iparam < numparam; ++iparam)
      cellvalues(inen) += elementvalues(iparam) * enr_funct(iparam);
  }
  return;
}


/*----------------------------------------------------------------------*
 | interpolate field from element node values to cell node values based |
 | on a level-set field                                     henke 10/09 |
 | remark: function used for modified jump enrichment strategy          |
 *----------------------------------------------------------------------*/
void XFEM::InterpolateCellValuesFromElementValuesLevelSet(
  const DRT::Element&                   ele,
  const XFEM::ElementDofManager&        dofman,
  const GEO::DomainIntCell&             cell,
  const std::vector<double>&            phivalues,
  const XFEM::PHYSICS::Field            field,
  const LINALG::SerialDenseMatrix&      elementvalues,
  LINALG::SerialDenseMatrix&            cellvalues)
{
  if (ele.Shape() != DRT::Element::hex8)
    dserror("OutputToGmsh() only available for hex8 elements! However, this is easy to extend.");
  const size_t numnode = DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement;
  const size_t numparam  = dofman.NumDofPerField(field);

  // copy element phi vector from std::vector (phivalues) to LINALG::Matrix (phi)
  LINALG::Matrix<numnode,1> phi;
  for (size_t inode=0; inode<numnode; ++inode)
    phi(inode) = phivalues[inode];

  // get coordinates of cell vertices
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  // compute enrichment values based on a level set field 'phi'
  const XFEM::ElementEnrichmentValues enrvals(ele, dofman, cell, phi);

  LINALG::SerialDenseVector enr_funct(numparam);
  LINALG::SerialDenseVector funct(numnode);

  cellvalues.Zero();
  for (int inode = 0; inode < cell.NumNode(); ++inode)
  {
    // evaluate shape functions
    DRT::UTILS::shape_function_3D(funct,nodalPosXiDomain(0,inode),nodalPosXiDomain(1,inode),nodalPosXiDomain(2,inode),DRT::Element::hex8);

    // evaluate enriched shape functions
    enrvals.ComputeModifiedEnrichedNodalShapefunction(field, funct, enr_funct);

    switch (field)
    {
    // scalar fields
    case XFEM::PHYSICS::Pres:
    {
    // interpolate value
    for (size_t iparam = 0; iparam < numparam; ++iparam)
      cellvalues(0,inode) += elementvalues(0,iparam) * enr_funct(iparam);
    break;
    }
    // vector fields
    case XFEM::PHYSICS::Velx:
    {
      for (std::size_t iparam = 0; iparam < numparam; ++iparam)
        for (std::size_t isd = 0; isd < 3; ++isd)
          cellvalues(isd,inode) += elementvalues(isd,iparam) * enr_funct(iparam);
      break;
    }
    default:
      dserror("interpolation to cells not available for this field");
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 | interpolate field from element node values to cell node values based |
 | on a level-set field                                     henke 01/11 |
 | remark: function used for modified jump normal enrichment strategy   |
 *----------------------------------------------------------------------*/
void XFEM::InterpolateCellValuesFromElementValuesLevelSetNormal(
  const DRT::Element&                   ele,
  const XFEM::ElementDofManager&        dofman,
  const GEO::DomainIntCell&             cell,
  const std::vector<double>&            phivalues,
  const LINALG::Matrix<3,8>&            gradphi,
  const XFEM::PHYSICS::Field            field,
  const LINALG::SerialDenseMatrix&      elementvalues,
  LINALG::SerialDenseMatrix&            cellvalues)
{
  if (ele.Shape() != DRT::Element::hex8)
    dserror("OutputToGmsh() only available for hex8 elements! However, this is easy to extend.");
  const size_t numnode = DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement;
  const size_t numparam  = dofman.NumDofPerField(field);

  // copy element phi vector from std::vector (phivalues) to LINALG::Matrix (phi)
  LINALG::Matrix<numnode,1> phi;
  for (size_t inode=0; inode<numnode; ++inode)
    phi(inode) = phivalues[inode];

  // get coordinates of cell vertices
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  // compute enrichment values based on a level set field 'phi'
  const XFEM::ElementEnrichmentValues enrvals(ele, dofman, cell, phi);

  LINALG::SerialDenseVector enr_funct(numparam,true);
  LINALG::SerialDenseVector funct(numnode,true);

  cellvalues.Zero();
  for (int ivertex = 0; ivertex < cell.NumNode(); ++ivertex)
  {
    // evaluate shape functions
    DRT::UTILS::shape_function_3D(funct,nodalPosXiDomain(0,ivertex),nodalPosXiDomain(1,ivertex),nodalPosXiDomain(2,ivertex),DRT::Element::hex8);

    XFEM::ApproxFuncNormalVector<0,8> shp(true);
    // fill approximation functions for XFEM
    for (size_t iparam = 0; iparam != numparam; ++iparam)
    {
      shp.velx.d0.s(iparam) = funct(iparam);
      shp.vely.d0.s(iparam) = funct(iparam);
      shp.velz.d0.s(iparam) = funct(iparam);
    }

#ifdef COLLAPSE_FLAME
    LINALG::Matrix<3,1> normal(true);
    // get coordinates of cell vertices
    const LINALG::SerialDenseMatrix& nodalPosXYZ(cell.CellNodalPosXYZ());
    normal(0) = nodalPosXYZ(0,ivertex);
    normal(1) = nodalPosXYZ(1,ivertex);
    normal(2) = 0.0;
    const double norm = normal.Norm2(); // sqrt(normal(0)*normal(0) + normal(1)*normal(1) + normal(2)*normal(2))
    if (norm == 0.0) dserror("norm of normal vector is zero!");
    normal.Scale(-1.0/norm);
#endif

    // shape functions and derivatives for nodal parameters (dofs)
    enrvals.ComputeNormalShapeFunction(funct,gradphi,
//#ifdef COLLAPSE_FLAME
        normal,
//#endif
        shp);

    switch (field)
    {
    // vector fields
    case XFEM::PHYSICS::Velx:
    {
      const int* nodeids = ele.NodeIds();

      std::size_t velncounter = 0;
      for (std::size_t inode=0; inode<numnode; ++inode)
      {
        // standard shape functions are identical for all vector components
        // shp.velx.d0.s == shp.vely.d0.s == shp.velz.d0.s
        cellvalues(0,ivertex) += elementvalues(0,inode)*shp.velx.d0.s(inode);
        cellvalues(1,ivertex) += elementvalues(1,inode)*shp.vely.d0.s(inode);
        cellvalues(2,ivertex) += elementvalues(2,inode)*shp.velz.d0.s(inode);

        const int gid = nodeids[inode];
        const std::set<XFEM::FieldEnr>& enrfieldset = dofman.FieldEnrSetPerNode(gid);

        for (std::set<XFEM::FieldEnr>::const_iterator enrfield =
            enrfieldset.begin(); enrfield != enrfieldset.end(); ++enrfield)
        {
          if (enrfield->getField() == XFEM::PHYSICS::Veln)
          {
            cellvalues(0,ivertex) += elementvalues(3,velncounter)*shp.velx.d0.n(velncounter);
            cellvalues(1,ivertex) += elementvalues(3,velncounter)*shp.vely.d0.n(velncounter);
            cellvalues(2,ivertex) += elementvalues(3,velncounter)*shp.velz.d0.n(velncounter);

            velncounter += 1;
          }
        }
      }
      // TODO @Florian remove this from release version
      if (velncounter != dofman.NumDofPerField(XFEM::PHYSICS::Veln)) dserror("Alles falsch, du Depp!");
      dsassert(velncounter == dofman.NumDofPerField(XFEM::PHYSICS::Veln), "mismatch in information from eledofmanager!");

      break;
    }
    default:
      dserror("interpolation to cells not available for this field");
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 | interpolate field from element node values to cell node values based |
 | on a level-set field                                     henke 10/09 |
 | remark: function used for modified kink enrichment strategy          |
 *----------------------------------------------------------------------*/
void XFEM::InterpolateCellValuesFromElementValuesLevelSetKink(
  const DRT::Element&                   ele,
  const XFEM::ElementDofManager&        dofman,
  const GEO::DomainIntCell&             cell,
  const std::vector<double>&            phivalues,
  const XFEM::PHYSICS::Field            field,
  const LINALG::SerialDenseMatrix&      elementvalues,
  LINALG::SerialDenseMatrix&            cellvalues)
{
  if (ele.Shape() != DRT::Element::hex8)
    dserror("OutputToGmsh() only available for hex8 elements! However, this is easy to extend.");
  const size_t numnode = DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement;
  const size_t numparam  = dofman.NumDofPerField(field);

  // copy element phi vector from std::vector (phivalues) to LINALG::Matrix (phi)
//  LINALG::Matrix<numnode,1> phi;
  LINALG::SerialDenseVector phi(numnode);
  for (size_t inode=0; inode<numnode; ++inode)
    phi(inode) = phivalues[inode];

  // get coordinates of cell vertices
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  LINALG::SerialDenseVector enr_funct(numparam);
  enr_funct.Zero();
  LINALG::SerialDenseVector funct(numnode);

  cellvalues.Zero();
  for (int inode = 0; inode < cell.NumNode(); ++inode)
  {
    // evaluate shape functions
    DRT::UTILS::shape_function_3D(funct,nodalPosXiDomain(0,inode),nodalPosXiDomain(1,inode),nodalPosXiDomain(2,inode),DRT::Element::hex8);
    // first and second derivatives are dummy matrices needed to call XFEM::ElementEnrichmentValues for kink enrichment
    const LINALG::Matrix<3,numnode> derxy(true);
    const LINALG::Matrix<DRT::UTILS::DisTypeToNumDeriv2<DRT::Element::hex8>::numderiv2, DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement> derxy2(true);

    // compute enrichment values based on a level set field 'phi'
    const XFEM::ElementEnrichmentValues enrvals(ele, dofman, phi, funct, derxy, derxy2);

    // evaluate enriched shape functions
    // remark: since we do not compute derivatives of enriched shape functions (chain rule for kink
    //         enrichment!), we can use the same function for all types of enrichments.
    enrvals.ComputeModifiedEnrichedNodalShapefunction(field, funct, enr_funct);

    switch (field)
    {
    // scalar fields
    case XFEM::PHYSICS::Pres:
    {
    // interpolate value
    for (size_t iparam = 0; iparam < numparam; ++iparam)
      cellvalues(0,inode) += elementvalues(0,iparam) * enr_funct(iparam);
    break;
    }
    // vector fields
    case XFEM::PHYSICS::Velx:
    {
      for (std::size_t iparam = 0; iparam < numparam; ++iparam)
        for (std::size_t isd = 0; isd < 3; ++isd)
          cellvalues(isd,inode) += elementvalues(isd,iparam) * enr_funct(iparam);
      break;
    }
    default:
      dserror("interpolation to cells not available for this field");
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 | interpolate field from element node values to cell node values based |
 | on a level-set field                                 schott 05/17/10 |
 | remark: function used for 2-phase-flow                               |
 | with jump enrichments in pressure and kink enrichment in velocity    |
 *----------------------------------------------------------------------*/
void XFEM::InterpolateCellValuesFromElementValuesLevelSetKinkJump(
  const DRT::Element&                   ele,
  const XFEM::ElementDofManager&        dofman,
  const GEO::DomainIntCell&             cell,
  const std::vector<double>&            phivalues,
  const XFEM::PHYSICS::Field            field,
  const LINALG::SerialDenseMatrix&      elementvalues,
  LINALG::SerialDenseMatrix&            cellvalues)
{
  if (ele.Shape() != DRT::Element::hex8)
    dserror("OutputToGmsh() only available for hex8 elements! However, this is easy to extend.");
  const size_t numnode = DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement;
  const size_t numparam  = dofman.NumDofPerField(field);

  // copy element phi vector from std::vector (phivalues) to LINALG::Matrix (phi)
//  LINALG::Matrix<numnode,1> phi;
  LINALG::SerialDenseVector phi(numnode);
  for (size_t inode=0; inode<numnode; ++inode)
    phi(inode) = phivalues[inode];

  // get coordinates of cell vertices
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  LINALG::SerialDenseVector funct(numnode);

  cellvalues.Zero();

  for (int inode = 0; inode < cell.NumNode(); ++inode)
  {
    // evaluate shape functions
    DRT::UTILS::shape_function_3D(funct,nodalPosXiDomain(0,inode),nodalPosXiDomain(1,inode),nodalPosXiDomain(2,inode),DRT::Element::hex8);
    // first and second derivatives are dummy matrices needed to call XFEM::ElementEnrichmentValues for kink enrichment
    const LINALG::Matrix<3,numnode> derxy(true);
    const LINALG::Matrix<DRT::UTILS::DisTypeToNumDeriv2<DRT::Element::hex8>::numderiv2, DRT::UTILS::DisTypeToNumNodePerEle<DRT::Element::hex8>::numNodePerElement> derxy2(true);

    const XFEM::ElementEnrichmentValues enrvals(ele, dofman, phi, cell, funct, derxy, derxy2);
    LINALG::SerialDenseVector enr_funct(numparam);
    enr_funct.Zero();

    enrvals.ComputeModifiedEnrichedNodalShapefunction(field, funct, enr_funct);

    switch (field)
    {
    // scalar fields
    case XFEM::PHYSICS::Pres:
    {
    // interpolate value
    for (size_t iparam = 0; iparam < numparam; ++iparam)
      cellvalues(0,inode) += elementvalues(0,iparam) * enr_funct(iparam);
    break;
    }
    // vector fields
    case XFEM::PHYSICS::Velx:
    {
      for (std::size_t iparam = 0; iparam < numparam; ++iparam)
        for (std::size_t isd = 0; isd < 3; ++isd)
            cellvalues(isd,inode) += elementvalues(isd,iparam) * enr_funct(iparam);
      break;
    }
    default:
      dserror("interpolation to cells not available for this field");
    }
  }
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void XFEM::computeScalarCellNodeValuesFromElementUnknowns(
  const DRT::Element&                 ele,
  XFEM::InterfaceHandle*   ih,
  const XFEM::ElementDofManager&      dofman,
  const GEO::DomainIntCell&           cell,
  const XFEM::PHYSICS::Field          field,
  const LINALG::SerialDenseVector&    elementvalues,
  LINALG::SerialDenseVector&          cellvalues)
{
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  const XFEM::ElementEnrichmentValues enrvals(
        ele,
        ih,
        dofman,
        cell.GetPhysicalCenterPosition(),
        false, -1);

  cellvalues.Zero();
  for (int incn = 0; incn < cell.NumNode(); ++incn)
  {
    const std::size_t numparam  = dofman.NumDofPerField(field);
    if (numparam == 0)
      continue;

    const DRT::Element::DiscretizationType eleval_distype = dofman.getDisTypePerField(field);
    const std::size_t numvirtnode = DRT::UTILS::getNumberOfElementNodes(eleval_distype);

    LINALG::SerialDenseVector funct(numvirtnode);
    // fill shape functions
    DRT::UTILS::shape_function_3D(funct,
      nodalPosXiDomain(0,incn),
      nodalPosXiDomain(1,incn),
      nodalPosXiDomain(2,incn),
      eleval_distype);

    LINALG::SerialDenseVector enr_funct(numparam);
    enrvals.ComputeEnrichedElementShapefunction(field, funct, enr_funct);
    // interpolate value
    for (std::size_t iparam = 0; iparam < numparam; ++iparam)
      cellvalues(incn) += elementvalues(iparam) * enr_funct(iparam);
  }
  return;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void XFEM::computeTensorCellNodeValuesFromElementUnknowns(
  const DRT::Element&                 ele,
  XFEM::InterfaceHandle*   ih,
  const XFEM::ElementDofManager&      dofman,
  const GEO::DomainIntCell&           cell,
  const XFEM::PHYSICS::Field          field,
  const LINALG::SerialDenseMatrix&    elementvalues,
  LINALG::SerialDenseMatrix&          cellvalues)
{
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  const XFEM::ElementEnrichmentValues enrvals(
        ele,
        ih,
        dofman,
        cell.GetPhysicalCenterPosition(),
        false, -1);

  cellvalues.Zero();
  for (int incn = 0; incn < cell.NumNode(); ++incn)
  {
    const int numparam  = dofman.NumDofPerField(field);
    if (numparam == 0)
      continue;

    const DRT::Element::DiscretizationType eleval_distype = dofman.getDisTypePerField(field);
    const int numvirtnode = DRT::UTILS::getNumberOfElementNodes(eleval_distype);
//    if (numvirtnode != numparam) dserror("bug");

    LINALG::SerialDenseVector funct(numvirtnode);
    // fill shape functions
    DRT::UTILS::shape_function_3D(funct,
      nodalPosXiDomain(0,incn),
      nodalPosXiDomain(1,incn),
      nodalPosXiDomain(2,incn),
      eleval_distype);

    LINALG::SerialDenseVector enr_funct(numparam);
    enrvals.ComputeEnrichedElementShapefunction(field, funct, enr_funct);
    // interpolate value
    for (int iparam = 0; iparam < numparam; ++iparam)
      for (int ientry = 0; ientry < 9; ++ientry)
        cellvalues(ientry,incn) += elementvalues(ientry,iparam) * enr_funct(iparam);
  }
  return;
}


/*----------------------------------------------------------------------*
 | domain integration cell
 *----------------------------------------------------------------------*/
void XFEM::computeVectorCellNodeValues(
  const DRT::Element&                 ele,
  XFEM::InterfaceHandle*   ih,
  const XFEM::ElementDofManager&      dofman,
  const GEO::DomainIntCell&           cell,
  const XFEM::PHYSICS::Field          field,
  const LINALG::SerialDenseMatrix&    elementvalues,
  LINALG::SerialDenseMatrix&          cellvalues)
{
  const std::size_t nen_cell = DRT::UTILS::getNumberOfElementNodes(cell.Shape());
  const std::size_t numparam  = dofman.NumDofPerField(field);
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  const XFEM::ElementEnrichmentValues enrvals(
        ele,
        ih,
        dofman,
        cell.GetPhysicalCenterPosition(),
        false, -1);

  // cell corner nodes
  LINALG::SerialDenseVector enr_funct(numparam);
  //LINALG::SerialDenseVector funct(DRT::UTILS::getNumberOfElementNodes(ele.Shape()));
  LINALG::SerialDenseVector funct(27);
  cellvalues.Zero();
  for (std::size_t inen = 0; inen < nen_cell; ++inen)
  {
    // fill shape functions
    DRT::UTILS::shape_function_3D(funct,
      nodalPosXiDomain(0,inen),
      nodalPosXiDomain(1,inen),
      nodalPosXiDomain(2,inen),
      ele.Shape());
    enrvals.ComputeEnrichedNodalShapefunction(field, funct, enr_funct);
    // interpolate value
    for (std::size_t iparam = 0; iparam < numparam; ++iparam)
      for (std::size_t isd = 0; isd < 3; ++isd)
        cellvalues(isd,inen) += elementvalues(isd,iparam) * enr_funct(iparam);
  }
  return;
}


/*----------------------------------------------------------------------*
 | boundary integration cell
 *----------------------------------------------------------------------*/
void XFEM::computeVectorCellNodeValues(
  const DRT::Element&                 ele,
  XFEM::InterfaceHandle*   ih,
  const XFEM::ElementDofManager&      dofman,
  const GEO::BoundaryIntCell&         cell,
  const XFEM::PHYSICS::Field          field,
  const int                           label,
  const LINALG::SerialDenseMatrix&    elementvalues,
  LINALG::SerialDenseMatrix&          cellvalues)
{
  const std::size_t nen_cell = DRT::UTILS::getNumberOfElementNodes(cell.Shape());
  const std::size_t numparam  = dofman.NumDofPerField(field);
  const LINALG::SerialDenseMatrix& nodalPosXiDomain(cell.CellNodalPosXiDomain());

  const XFEM::ElementEnrichmentValues enrvals(
        ele,
        ih,
        dofman,
        cell.GetPhysicalCenterPosition(),
        true,
        label);

  // cell corner nodes
  LINALG::SerialDenseVector enr_funct(numparam);
  //LINALG::SerialDenseVector funct(DRT::UTILS::getNumberOfElementNodes(ele.Shape()));
  LINALG::SerialDenseVector funct(27);
  cellvalues.Zero();
  for (std::size_t inen = 0; inen < nen_cell; ++inen)
  {
    // fill shape functions
    DRT::UTILS::shape_function_3D(funct,
      nodalPosXiDomain(0,inen),
      nodalPosXiDomain(1,inen),
      nodalPosXiDomain(2,inen),
      ele.Shape());

    enrvals.ComputeEnrichedNodalShapefunction(field, funct, enr_funct);
    // interpolate value
    for (std::size_t iparam = 0; iparam < numparam; ++iparam)
      for (std::size_t isd = 0; isd < 3; ++isd)
        cellvalues(isd,inen) += elementvalues(isd,iparam) * enr_funct(iparam);
  }
  return;
}



/*!
 * Calculate ratio between fictitious element size and normal size
 */
template <DRT::Element::DiscretizationType DISTYPE>
double DomainCoverageRatioT(
        const DRT::Element&           ele,           ///< the element whose area ratio we want to compute
        const XFEM::InterfaceHandle&  ih             ///< connection to the interface handler
        )
{
  // number of nodes for element
  const int numnode = DRT::UTILS::DisTypeToNumNodePerEle<DISTYPE>::numNodePerElement;
  // get node coordinates of the current element
  LINALG::Matrix<3,numnode> xyze;
  GEO::fillInitialPositionArray<DISTYPE>(&ele, xyze);

  //double
  double domain_ele  = 0.0;
  double domain_fict = 0.0;

  // information about domain integration cells
  const GEO::DomainIntCells&  domainIntCells(ih.GetDomainIntCells(&ele));
  // loop over integration cells
  for (GEO::DomainIntCells::const_iterator cell = domainIntCells.begin(); cell != domainIntCells.end(); ++cell)
  {
    const LINALG::Matrix<3,1> cellcenter(cell->GetPhysicalCenterPosition());

    const int label = ih.PositionWithinConditionNP(cellcenter);

    DRT::UTILS::GaussRule3D gaussrule = DRT::UTILS::intrule3D_undefined;
    switch (cell->Shape())
    {
      case DRT::Element::hex8: case DRT::Element::hex20: case DRT::Element::hex27:
      {
        gaussrule = DRT::UTILS::intrule_hex_8point;
        break;
      }
      case DRT::Element::tet4: case DRT::Element::tet10:
      {
        gaussrule = DRT::UTILS::intrule_tet_4point;
        break;
      }
      case DRT::Element::wedge6: case DRT::Element::wedge15:
      {
        gaussrule = DRT::UTILS::intrule_wedge_6point;
        break;
      }
      case DRT::Element::pyramid5:
      {
        gaussrule = DRT::UTILS::intrule_pyramid_8point;
        break;
      }
      default:
        dserror("add your element type here...");
    }

    // gaussian points
    const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule);

    // integration loop
    for (int iquad=0; iquad<intpoints.nquad; ++iquad)
    {
      // coordinates of the current integration point in cell coordinates \eta
      LINALG::Matrix<3,1> pos_eta_domain;
      pos_eta_domain(0) = intpoints.qxg[iquad][0];
      pos_eta_domain(1) = intpoints.qxg[iquad][1];
      pos_eta_domain(2) = intpoints.qxg[iquad][2];

      // coordinates of the current integration point in element coordinates \xi
      LINALG::Matrix<3,1> posXiDomain;
      GEO::mapEtaToXi3D<XFEM::xfem_assembly>(*cell, pos_eta_domain, posXiDomain);
      const double detcell = GEO::detEtaToXi3D<XFEM::xfem_assembly>(*cell, pos_eta_domain);

      // shape functions and their first derivatives
      LINALG::Matrix<numnode,1> funct;
      LINALG::Matrix<3,numnode> deriv;
      DRT::UTILS::shape_function_3D(funct,posXiDomain(0),posXiDomain(1),posXiDomain(2),DISTYPE);
      DRT::UTILS::shape_function_3D_deriv1(deriv,posXiDomain(0),posXiDomain(1),posXiDomain(2),DISTYPE);

      // get transposed of the jacobian matrix d x / d \xi
      //xjm = deriv(i,k)*xyze(j,k);
      LINALG::Matrix<3,3> xjm;
      xjm.MultiplyNT(deriv,xyze);

      const double det = xjm.Determinant();
      const double fac = intpoints.qwgt[iquad]*det*detcell;

      if(det < 0.0)
      {
        dserror("GLOBAL ELEMENT NO.%i\nNEGATIVE JACOBIAN DETERMINANT: %f", ele.Id(), det);
      }

      domain_ele += fac;

      if(label != 0)
        domain_fict += fac;

    } // end loop over gauss points
  } // end loop over integration cells
  return domain_fict / domain_ele;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double XFEM::DomainCoverageRatio(
        const DRT::Element&           ele,
        const XFEM::InterfaceHandle&  ih)
{
  switch (ele.Shape())
  {
    case DRT::Element::hex8:
      return DomainCoverageRatioT<DRT::Element::hex8>(ele,ih);
    case DRT::Element::hex20:
      return DomainCoverageRatioT<DRT::Element::hex20>(ele,ih);
    case DRT::Element::hex27:
      return DomainCoverageRatioT<DRT::Element::hex27>(ele,ih);
    case DRT::Element::tet4:
      return DomainCoverageRatioT<DRT::Element::tet4>(ele,ih);
    case DRT::Element::tet10:
      return DomainCoverageRatioT<DRT::Element::tet10>(ele,ih);
    case DRT::Element::wedge6:
      return DomainCoverageRatioT<DRT::Element::wedge6>(ele,ih);
    case DRT::Element::wedge15:
      return DomainCoverageRatioT<DRT::Element::wedge15>(ele,ih);
    case DRT::Element::pyramid5:
      return DomainCoverageRatioT<DRT::Element::pyramid5>(ele,ih);
    default:
      dserror("add you distype here...");
      exit(1);
  }
}



/*!
 * Calculate ratio between fictitious element size and normal size
 */
template <DRT::Element::DiscretizationType DISTYPE>
vector<double> DomainCoverageRatioPerNodeT(
        const DRT::Element&           ele,           ///< the element whose area ratio we want to compute
        const XFEM::InterfaceHandle&  ih             ///< connection to the interface handler
        )
{
  // number of nodes for element
  const int numnode = DRT::UTILS::DisTypeToNumNodePerEle<DISTYPE>::numNodePerElement;
  double area_ele  = 0.0;
  vector<double> portions(numnode,0.0);

  // information about domain integration cells
  const GEO::DomainIntCells&  domainIntCells(ih.GetDomainIntCells(&ele));
  // loop over integration cells
  for (GEO::DomainIntCells::const_iterator cell = domainIntCells.begin(); cell != domainIntCells.end(); ++cell)
  {
    const LINALG::Matrix<3,1> cellcenter(cell->GetPhysicalCenterPosition());
    const int label = ih.PositionWithinConditionNP(cellcenter);

    DRT::UTILS::GaussRule3D gaussrule = DRT::UTILS::intrule3D_undefined;
    switch (cell->Shape())
    {
      case DRT::Element::hex8:
      {
        gaussrule = DRT::UTILS::intrule_hex_8point;
        break;
      }
      case DRT::Element::hex20: case DRT::Element::hex27:
      {
        gaussrule = DRT::UTILS::intrule_hex_27point;
        break;
      }
      case DRT::Element::tet4: case DRT::Element::tet10:
      {
        gaussrule = DRT::UTILS::intrule_tet_4point;
        break;
      }
      case DRT::Element::wedge6: case DRT::Element::wedge15:
      {
        gaussrule = DRT::UTILS::intrule_wedge_6point;
        break;
      }
      case DRT::Element::pyramid5:
      {
        gaussrule = DRT::UTILS::intrule_pyramid_8point;
        break;
      }
      default:
        dserror("add your element type here...");
    }

    // gaussian points
    const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule);

    // integration loop
    for (int iquad=0; iquad<intpoints.nquad; ++iquad)
    {
      // coordinates of the current integration point in cell coordinates \eta
      LINALG::Matrix<3,1> pos_eta_domain;
      pos_eta_domain(0) = intpoints.qxg[iquad][0];
      pos_eta_domain(1) = intpoints.qxg[iquad][1];
      pos_eta_domain(2) = intpoints.qxg[iquad][2];

      // coordinates of the current integration point in element coordinates \xi
      LINALG::Matrix<3,1> posXiDomain;
      GEO::mapEtaToXi3D<XFEM::xfem_assembly>(*cell, pos_eta_domain, posXiDomain);
      const double detcell = GEO::detEtaToXi3D<XFEM::xfem_assembly>(*cell, pos_eta_domain);

      // shape functions and their first derivatives
      LINALG::Matrix<numnode,1> funct;
      DRT::UTILS::shape_function_3D(funct,posXiDomain(0),posXiDomain(1),posXiDomain(2),DISTYPE);

      const double fac = intpoints.qwgt[iquad]*detcell;

      area_ele += fac;

      if (label == 0)
        for (int inode = 0;inode < numnode;++inode)
          portions[inode] += funct(inode) * fac;

    } // end loop over gauss points
  } // end loop over integration cells

  for(int inode = 0; inode < numnode; ++inode)
    portions[inode] /= area_ele;

  return portions;
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
vector<double> XFEM::DomainCoverageRatioPerNode(
          const DRT::Element&           ele,
          const XFEM::InterfaceHandle&  ih)
{
  switch (ele.Shape())
  {
    case DRT::Element::hex8:
      return DomainCoverageRatioPerNodeT<DRT::Element::hex8>(ele,ih);
    case DRT::Element::hex20:
      return DomainCoverageRatioPerNodeT<DRT::Element::hex20>(ele,ih);
    case DRT::Element::hex27:
      return DomainCoverageRatioPerNodeT<DRT::Element::hex27>(ele,ih);
    case DRT::Element::tet4:
      return DomainCoverageRatioPerNodeT<DRT::Element::tet4>(ele,ih);
    case DRT::Element::tet10:
      return DomainCoverageRatioPerNodeT<DRT::Element::tet10>(ele,ih);
    case DRT::Element::wedge6:
      return DomainCoverageRatioPerNodeT<DRT::Element::wedge6>(ele,ih);
    case DRT::Element::wedge15:
      return DomainCoverageRatioPerNodeT<DRT::Element::wedge15>(ele,ih);
    case DRT::Element::pyramid5:
      return DomainCoverageRatioPerNodeT<DRT::Element::pyramid5>(ele,ih);
    default:
      dserror("add you distype here...");
      exit(1);
  }
}


/*!
  Calculate ratio between fictitious element size and normal size
 */
template <DRT::Element::DiscretizationType DISTYPE>
    double BoundaryCoverageRatioT(
        const DRT::Element&               xele,          ///< the element whose boundary ratio we want to compute
        const GEO::BoundaryIntCells&      boundaryIntCells,
        const XFEM::InterfaceHandle&      ih             ///< connection to the interface handler
        )
{
  static const Epetra_BLAS blas;

  double area_fict = 0.0;
  double base_area = 0.0;
  if (DISTYPE == DRT::Element::tet10 or DISTYPE == DRT::Element::tet4)
  {
    base_area = 0.5;
  }
  else if (DISTYPE == DRT::Element::hex8 or DISTYPE == DRT::Element::hex20 or DISTYPE == DRT::Element::hex27)
  {
    base_area = 4.0;
  }
  else if (DISTYPE == DRT::Element::wedge6 or DISTYPE == DRT::Element::wedge15)
  {
    base_area = 2.0;
  }
  else if (DISTYPE == DRT::Element::pyramid5)
  {
    base_area = 4.0;
  }
  else
  {
    dserror("think about it. factor at the end of this function needs another values");
  }

  // loop over boundary integration cells
  for (GEO::BoundaryIntCells::const_iterator cell = boundaryIntCells.begin(); cell != boundaryIntCells.end(); ++cell)
  {

    DRT::UTILS::GaussRule2D gaussrule = DRT::UTILS::intrule2D_undefined;
    switch (cell->Shape())
    {
    case DRT::Element::tri3: case DRT::Element::tri6:
    {
      gaussrule = DRT::UTILS::intrule_tri_1point;
      break;
    }
    case DRT::Element::quad4: case DRT::Element::quad9:
    {
      gaussrule = DRT::UTILS::intrule_quad_4point;
      break;
    }
    default:
      dserror("add your element type here...");
    }

    // gaussian points
    const DRT::UTILS::IntegrationPoints2D intpoints(gaussrule);

    const LINALG::SerialDenseMatrix& nodalpos_xi_domain(cell->CellNodalPosXiDomain());
    const int numnode_cell = cell->NumNode();

    // integration loop
    for (int iquad=0; iquad<intpoints.nquad; ++iquad)
    {
      // coordinates of the current integration point in cell coordinates \eta^\boundary
      LINALG::Matrix<2,1> pos_eta_boundary;
      pos_eta_boundary(0) = intpoints.qxg[iquad][0];
      pos_eta_boundary(1) = intpoints.qxg[iquad][1];

      LINALG::Matrix<3,1> posXiDomain;
      mapEtaBToXiD(*cell, pos_eta_boundary, posXiDomain);

      // shape functions and their first derivatives
      LINALG::SerialDenseVector funct_boundary(DRT::UTILS::getNumberOfElementNodes(cell->Shape()));
      DRT::UTILS::shape_function_2D(funct_boundary, pos_eta_boundary(0),pos_eta_boundary(1),cell->Shape());
      LINALG::SerialDenseMatrix deriv_boundary(3, DRT::UTILS::getNumberOfElementNodes(cell->Shape()));
      DRT::UTILS::shape_function_2D_deriv1(deriv_boundary, pos_eta_boundary(0),pos_eta_boundary(1),cell->Shape());

//       // shape functions and their first derivatives
//       LINALG::Matrix<DRT::UTILS::DisTypeToNumNodePerEle<DISTYPE>::numNodePerElement,1> funct;
//       DRT::UTILS::shape_function_3D(funct,posXiDomain(0),posXiDomain(1),posXiDomain(2),DISTYPE);

      // get jacobian matrix d x / d \xi  (3x2)
      // dxyzdrs = xyze_boundary(i,k)*deriv_boundary(j,k);
      LINALG::Matrix<3,2> dxyzdrs;
      blas.GEMM('N','T',3,2,numnode_cell,1.0,nodalpos_xi_domain.A(),nodalpos_xi_domain.LDA(),deriv_boundary.A(),deriv_boundary.LDA(),0.0,dxyzdrs.A(),dxyzdrs.M());

      // compute covariant metric tensor G for surface element (2x2)
      // metric = dxyzdrs(k,i)*dxyzdrs(k,j);
      LINALG::Matrix<2,2> metric_XiDToEtaB;
      metric_XiDToEtaB.MultiplyTN(dxyzdrs,dxyzdrs);

      const double detmetric_XiDToEtaB = sqrt(metric_XiDToEtaB.Determinant());

      const double fac = intpoints.qwgt[iquad]*detmetric_XiDToEtaB;
      if (fac < 0.0)
      {
        cout << endl;
        cout << "detmetric = " << detmetric_XiDToEtaB << endl;
        cout << "fac       = " << fac << endl;
        dserror("negative fac! should be a bug!");
      }

      area_fict += fac;

    } // end loop over gauss points
  } // end loop over integration cells

  // scale result by area of one surface of the volume element
  return area_fict / base_area;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double XFEM::BoundaryCoverageRatio(
        const DRT::Element&           xele,
        const GEO::BoundaryIntCells&  boundaryIntCells,
        const XFEM::InterfaceHandle&  ih)
{
  switch (xele.Shape())
  {
    case DRT::Element::hex8:
      return BoundaryCoverageRatioT<DRT::Element::hex8>(xele,boundaryIntCells,ih);
    case DRT::Element::hex20:
      return BoundaryCoverageRatioT<DRT::Element::hex20>(xele,boundaryIntCells,ih);
    case DRT::Element::hex27:
      return BoundaryCoverageRatioT<DRT::Element::hex27>(xele,boundaryIntCells,ih);
    case DRT::Element::tet4:
      return BoundaryCoverageRatioT<DRT::Element::tet4>(xele,boundaryIntCells,ih);
    case DRT::Element::tet10:
      return BoundaryCoverageRatioT<DRT::Element::tet10>(xele,boundaryIntCells,ih);
    case DRT::Element::wedge6:
      return BoundaryCoverageRatioT<DRT::Element::wedge6>(xele,boundaryIntCells,ih);
    case DRT::Element::wedge15:
      return BoundaryCoverageRatioT<DRT::Element::wedge15>(xele,boundaryIntCells,ih);
    case DRT::Element::pyramid5:
      return BoundaryCoverageRatioT<DRT::Element::pyramid5>(xele,boundaryIntCells,ih);
    default:
      dserror("add you distype here...");
      exit(1);
  }
}

/*!
 * Calculate ratio between fictitious element size and normal size
 */
template <DRT::Element::DiscretizationType DISTYPE>
vector<double> DomainIntCellCoverageRatioT(
        const DRT::Element&           ele,           ///< the element whose area ratio we want to compute
        const XFEM::InterfaceHandle&  ih             ///< connection to the interface handler
        )
{
  // number of nodes for element
  const int numnode = DRT::UTILS::DisTypeToNumNodePerEle<DISTYPE>::numNodePerElement;

  // information about domain integration cells
  const GEO::DomainIntCells&  domainIntCells(ih.GetDomainIntCells(&ele));

  double area_ele  = 0.0;

  std::vector<double> portions(domainIntCells.size(),0.0);

  // loop over integration cells
  int cellcount = 0;
  for (GEO::DomainIntCells::const_iterator cell = domainIntCells.begin(); cell != domainIntCells.end(); ++cell)
  {
    const LINALG::Matrix<3,1> cellcenter(cell->GetPhysicalCenterPosition());
    DRT::UTILS::GaussRule3D gaussrule = DRT::UTILS::intrule3D_undefined;
    switch (cell->Shape())
    {
      case DRT::Element::hex8:
      {
        gaussrule = DRT::UTILS::intrule_hex_8point;
        break;
      }
      case DRT::Element::hex20: case DRT::Element::hex27:
      {
        gaussrule = DRT::UTILS::intrule_hex_27point;
        break;
      }
      case DRT::Element::tet4: case DRT::Element::tet10:
      {
        gaussrule = DRT::UTILS::intrule_tet_4point;
        break;
      }
      case DRT::Element::wedge6: case DRT::Element::wedge15:
      {
        gaussrule = DRT::UTILS::intrule_wedge_6point;
        break;
      }
      case DRT::Element::pyramid5:
      {
        gaussrule = DRT::UTILS::intrule_pyramid_8point;
        break;
      }
      default:
        dserror("add your element type here...");
    }

    // gaussian points
    const DRT::UTILS::IntegrationPoints3D intpoints(gaussrule);

    // integration loop
    for (int iquad=0; iquad<intpoints.nquad; ++iquad)
    {
      // coordinates of the current integration point in cell coordinates \eta
      LINALG::Matrix<3,1> pos_eta_domain;
      pos_eta_domain(0) = intpoints.qxg[iquad][0];
      pos_eta_domain(1) = intpoints.qxg[iquad][1];
      pos_eta_domain(2) = intpoints.qxg[iquad][2];

      // coordinates of the current integration point in element coordinates \xi
      LINALG::Matrix<3,1> posXiDomain;
      GEO::mapEtaToXi3D<XFEM::xfem_assembly>(*cell, pos_eta_domain, posXiDomain);
      const double detcell = GEO::detEtaToXi3D<XFEM::xfem_assembly>(*cell, pos_eta_domain);

      // shape functions and their first derivatives
      LINALG::Matrix<numnode,1> funct;
      DRT::UTILS::shape_function_3D(funct,posXiDomain(0),posXiDomain(1),posXiDomain(2),DISTYPE);

      const double fac = intpoints.qwgt[iquad]*detcell;

      area_ele += fac;
      portions[cellcount] += fac;

    } // end loop over gauss points
    cellcount++;
  } // end loop over integration cells

  for (std::size_t icell = 0; icell < domainIntCells.size(); ++icell)
    portions[icell] /= area_ele;


  return portions;
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
std::vector<double> XFEM::DomainIntCellCoverageRatio(
        const DRT::Element&           ele,
        const XFEM::InterfaceHandle&  ih
        )
{
  switch (ele.Shape())
  {
    case DRT::Element::hex8:
      return DomainIntCellCoverageRatioT<DRT::Element::hex8>(ele,ih);
    case DRT::Element::hex20:
      return DomainIntCellCoverageRatioT<DRT::Element::hex20>(ele,ih);
    case DRT::Element::hex27:
      return DomainIntCellCoverageRatioT<DRT::Element::hex27>(ele,ih);
    case DRT::Element::tet4:
      return DomainIntCellCoverageRatioT<DRT::Element::tet4>(ele,ih);
    case DRT::Element::tet10:
      return DomainIntCellCoverageRatioT<DRT::Element::tet10>(ele,ih);
    case DRT::Element::wedge6:
      return DomainIntCellCoverageRatioT<DRT::Element::wedge6>(ele,ih);
    case DRT::Element::wedge15:
      return DomainIntCellCoverageRatioT<DRT::Element::wedge15>(ele,ih);
    case DRT::Element::pyramid5:
      return DomainIntCellCoverageRatioT<DRT::Element::pyramid5>(ele,ih);
    default:
      dserror("add you distype here...");
      exit(1);
  }
}



/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
XFEM::AssemblyType XFEM::ComputeAssemblyType(
    const ElementDofManager&   eleDofManager,
    const std::size_t          numnode,
    const int*                 nodeids)
{
  // find out whether we can use standard assembly or need xfem assembly
  XFEM::AssemblyType assembly_type = XFEM::standard_assembly;
  for (std::size_t inode = 0; inode < numnode; ++inode)
  {
    if (assembly_type == XFEM::xfem_assembly)
      break;

    const int gid = nodeids[inode];
    const std::set<XFEM::FieldEnr>& fields = eleDofManager.FieldEnrSetPerNode(gid);
    if (fields.size() != 4)
    {
      assembly_type = XFEM::xfem_assembly;
      break;
    }

    for (std::set<XFEM::FieldEnr>::const_iterator fieldenr = fields.begin(); fieldenr != fields.end(); ++fieldenr)
      if (fieldenr->getEnrichment().Type() != XFEM::Enrichment::typeStandard)
      {
        assembly_type = XFEM::xfem_assembly;
        break;
      };
  };

  if (eleDofManager.NumElemDof() != 0)
    assembly_type = XFEM::xfem_assembly;


  return assembly_type;
}


#endif  // #ifdef CCADISCRET
