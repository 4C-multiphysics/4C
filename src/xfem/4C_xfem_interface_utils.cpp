// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_xfem_interface_utils.hpp"

#include "4C_cut_boundarycell.hpp"
#include "4C_cut_volumecell.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_general_utils_integration.hpp"
#include "4C_fem_general_utils_local_connectivity_matrices.hpp"
#include "4C_fem_general_utils_nurbs_shapefunctions.hpp"
#include "4C_xfem_condition_manager.hpp"

FOUR_C_NAMESPACE_OPEN
//

/*----------------------------------------------------------------------*
 * Get the std - average weights kappa_m and kappa_s for the Nitsche calculations
 *----------------------------------------------------------------------*/
void XFEM::Utils::get_std_average_weights(
    const Inpar::XFEM::AveragingStrategy averaging_strategy, double& kappa_m)
{
  switch (averaging_strategy)
  {
    case Inpar::XFEM::Xfluid_Sided:
    {
      kappa_m = 1.0;
      break;
    }
    case Inpar::XFEM::Embedded_Sided:
    {
      kappa_m = 0.0;
      break;
    }
    case Inpar::XFEM::Mean:
    {
      kappa_m = 0.5;
      break;
    }
    default:
      FOUR_C_THROW("XFEM::Utils::GetAverageWeights: AveragingStrategy not implemented here!");
  }

  return;
}

/*--------------------------------------------------------------------------------
 * compute viscous part of Nitsche's penalty term scaling for Nitsche's method
 *--------------------------------------------------------------------------------*/
void XFEM::Utils::nit_compute_visc_penalty_stabfac(
    const Core::FE::CellType ele_distype,  ///< the discretization type of the element w.r.t which
                                           ///< the stabilization factor is computed
    const double& penscaling,  ///< material dependent penalty scaling (e.g. visceff) divided by h
    const double& NIT_stabscaling,  ///< basic nit penalty stab scaling
    const bool& is_pseudo_2D,       ///< is pseudo 2d
    const Inpar::XFEM::ViscStabTraceEstimate&
        visc_stab_trace_estimate,  ///< how to estimate the scaling from the trace inequality
    double& NIT_visc_stab_fac      ///< viscous part of Nitsche's penalty term
)
{
  //------------------------------
  // scaling factors for Nitsche's standard stabilization term
  /*
        viscous_Nitsche-part
      /                                        \        /                           i        \
     |  NIT_visc_stab_fac *  [ v ] , [ Du ]     | =  - |   NIT_visc_stab_fac * [ v ], [ u ]   |
      \                                        /        \                                    /


      with

            NIT_visc_stab_fac =   gamma * mu * C^2

      where
           - gamma is a dimensionless user-defined parameter
           - mu is the viscosity
           - C is an estimate of the following trace inequality with  C^2 ~ 1/h
             and C depends on element type, shape and polynomial degree


                    trace inequality to be satisfied for Nitsche's method

                    || grad v_h ||         <=   C  *  || grad v_h ||
                                  Gamma_K                           K

           - C^2 \approx lambda can be estimated as the maximal eigenvalue lambda of the generalized
     eigenvalue problem (A*x = lambda * B*x)

                    < grad w_h,  grad v_h >          =  lambda *  ( grad w_h, grad v_h)
                                           Gamma_K                                     K

           - alternatively we can estimate C^2 as C^2 \approx \rho_safety * CT / h_K (see
     generalized hp-framework by Erik Burman 2006)

                      with CT depending on the dimension d and the polynomial degree p

                               CT = p(p+1)/2 * (2+1/p)^d ( for tensor-product elements
     (hexahedral,quadrilateral...) CT = (p+1)*(p+d)/d        ( for simplices (lines, triangles,
     tetrahedral...)

                      and  1/h_K \approx meas(Gamma_K)/meas(K)
                      and rho_safety a safety factor depending on the element-type which is useful
     when the surface cuts the element in an inclined situation or when the surface contains small
     acute angles or corners
  */

  //------------------------------
  // 1 // to be filled viscous part of Nitsche's penalty term scaling for Nitsche's method
  // 2 // scale the viscous part of the penalty scaling with the effective viscosity of both sides
  // 3 // scale the viscous part of the penalty scaling with the dimensionless user defined
  // Nitsche-parameter gamma
  NIT_visc_stab_fac = penscaling * NIT_stabscaling;

  // compute the final viscous scaling part of Nitsche's penalty term
  if (visc_stab_trace_estimate == Inpar::XFEM::ViscStab_TraceEstimate_CT_div_by_hk)
  {
    // get an estimate of the hp-depending constant C_T satisfying the trace inequality w.r.t the
    // corresponding element (compare different weightings)
    double C_T = nit_get_trace_estimate_constant(ele_distype, is_pseudo_2D);

    // build the final viscous scaling
    NIT_visc_stab_fac *= C_T;
  }
  else if (visc_stab_trace_estimate != Inpar::XFEM::ViscStab_TraceEstimate_eigenvalue)
  {
    FOUR_C_THROW(
        "unknown trace-inequality-estimate type for viscous part of Nitsche's penalty term");
  }
  return;
}

/*--------------------------------------------------------------------------------
 * get the constant which satisfies the trace inequality depending on the spatial dimension and
 *polynomial order of the element
 *--------------------------------------------------------------------------------*/
double XFEM::Utils::nit_get_trace_estimate_constant(
    const Core::FE::CellType ele_distype, bool is_pseudo_2D)
{
  /*
  => return
       CT = p(p+1)/2 * (2+1/p)^d  OR an estimate obtained form solving an EVP ( for tensor-product
  elements (hexahedral,quadrilateral...) CT = (p+1)*(p+d)/d ( for simplices (lines, triangles,
  tetrahedral...) constant depending on element-type/polynomial order, see below

  - C is an estimate of the following trace inequality with  C^2 ~ 1/h
     and C depends on element type, shape and polynomial degree


            trace inequality to be satisfied for Nitsche's method

            || grad v_h ||         <=   C  *  || grad v_h ||
                          Gamma_K                           K

   - we can estimate C^2 as C^2 \approx \rho_safety * CT / h_K (see generalized hp-framework by Erik
  Burman 2006)

              with CT depending on the dimension d and the polynomial degree p of grad(v_h)

              - For tensor-product elements (hexahedral,quadrilateral,...)
                   CT = p(p+1)/2 * (2+1/p)^d
                      -> REMARK: the theoretical estimate of the constant CT for hexahedral and
  quadrilateral elements seems to overestimate the actual constant
                      -> therefore we take the approximate values from solving a local eigenvalue
  problem...)
              - for simplices (lines, triangles, tetrahedral...)
                   CT = (p+1)*(p+d)/d        ( this estimate is sharp and leads to the same
  constants as solving an EVP)

              and  1/h_K \approx meas(Gamma_K)/meas(K)
              and rho_safety a safety factor depending on the element-type which is useful when the
  surface cuts the element in an inclined situation or when the surface contains small acute angles
  or corners

   - Remark regarding the dimension of the problem (pseudo-2D simulations):
     -> important for CT is the degree of the element-wise polynomial which is clear for pure 2D and
  3D problems
     -> for pseudo-2D with one element-layer of 3D element in z-direction, but strong Dirichlet
  conditions which eliminate the z-contributions in the polynomial, we have to adapt the constant
  from d=3 to d=2 although a 3D element is used (use the "is_pseudo_2D"-flag)
        => this ensures same results of a 2D simulation and a pseudo-2D simulation using the 3D
  implementation in combination with strong Dirichlet conditions in z-direction
     -> an important property is that the elongation of the element in z-direction does not play a
  role when surface-element-measure based hk-definitions are used since then the relation
  meas(Gamma_K)/meas(K) remains constant
     -> using non-strong Dirichlet conditions for the enforcement of u_z = 0, CT for 3D is required
  which then leads to differences between a 2D implementation and pseudo-2D simulation using the 3D
  implementation
     -> IMPORTANT: to forget to set the "is_pseudo_2D"-flag to "true" for pseudo-2D examples leads
  to an overestimate of CT, and so does not result in stability problems, it might lead to a
  slightly worse convergence of the linear solver, however, usually it has not a to worse effect.
*/

  Core::FE::CellType trace_inequality_distype;

  if (!is_pseudo_2D)
  {
    // the standard case for real 2D or 3D simulations
    trace_inequality_distype = ele_distype;
  }
  else
  {
    // modification for pseudo 2D simulations
    switch (ele_distype)
    {
      case Core::FE::CellType::hex8:
        trace_inequality_distype = Core::FE::CellType::quad4;
        break;  // hex8 -> quad4 reduction
      case Core::FE::CellType::hex20:
        trace_inequality_distype = Core::FE::CellType::quad8;
        break;  // hex20-> quad8 reduction
      case Core::FE::CellType::hex27:
        trace_inequality_distype = Core::FE::CellType::quad9;
        break;  // hex27-> quad9 reduction
      case Core::FE::CellType::wedge15:
        trace_inequality_distype = Core::FE::CellType::tri6;
        break;  // wedge15 -> tri6 reduction (tri6 elements in 2D plane)
      case Core::FE::CellType::wedge6:
        trace_inequality_distype = Core::FE::CellType::tri3;
        break;  // wedge6 -> tri3 reduction (tri3 elements in 2D plane)
      default:
      {
        FOUR_C_THROW("not a valid pseudo 2D element-type - what to do?");
        trace_inequality_distype = Core::FE::CellType::dis_none;
        break;
      }
    };
  }

  //  return 2.0;

  // switch over the distype which determines the right polynomial degree and dimension
  switch (trace_inequality_distype)
  {
    // for triangluar/tetradhedral elements:
    //    -> the grad-operator reduces the polynomial degree p(grad(v_h)) = p(v_h)-1
    // for quadrilateral/hexahedral and wedge/pyramid elements:
    //    -> the grad-operator does not reduce the polynomial degree due to the mixed polynomials
    //    p(grad(v_h)) = p(v_h)

    /*
    // CT = p(p+1)/2 * (2+1/p)^d
    // 3D hexahedral elements
    case Core::FE::CellType::hex8:      return 27.0;                 break;  /// d=3,
    p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT=27.0 case Core::FE::CellType::hex20:
    return 46.875; break;
    /// d=3, p(v_h) = 2 -> p(grad(v_h)) = 2   =>  CT=46.875 (375/8) case
    Core::FE::CellType::hex27: return 46.875;               break;  /// d=3, p(v_h) =
    2 -> p(grad(v_h)) = 2   =>  CT=46.875 (375/8)
    // 2D quadrilateral elements
    case Core::FE::CellType::quad4:     return 9.0;                  break;  /// d=2,
    p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT=9.0 case Core::FE::CellType::quad8:
    return 9.0; break;
    /// d=2, p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT=9.0 case
    Core::FE::CellType::quad9:     return 18.75; break;  /// d=2, p(v_h) = 2 ->
    p(grad(v_h)) = 2   =>  CT=18.75  (75/4)
    */
    // REMARK: the theroretical estimate of the constant CT for hexahedral and quadrilateral
    // elements seems to overestimate the actual constant
    // -> therefore we take the approximate values from solving a local eigenvalue problem
    // estimates from solving the eigenvalue problem on regular elements
    case Core::FE::CellType::hex8:
      return 1.59307;
      break;  /// d=3, p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT= from eigenvalue problem
    case Core::FE::CellType::hex20:
      return 4.10462;
      break;  /// d=3, p(v_h) = 2 -> p(grad(v_h)) = 2   =>  CT= from eigenvalue problem
    case Core::FE::CellType::hex27:
      return 4.27784;
      break;  /// d=3, p(v_h) = 2 -> p(grad(v_h)) = 2   =>  CT= from eigenvalue problem
    // 2D quadrilateral elements
    case Core::FE::CellType::quad4:
      return 1.43426;
      break;  /// d=2, p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT= from eigenvalue problem
    case Core::FE::CellType::quad8:
      return 4.06462;
      break;  /// d=2, p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT= from eigenvalue problem
    case Core::FE::CellType::quad9:
      return 4.19708;
      break;  /// d=2, p(v_h) = 2 -> p(grad(v_h)) = 2   =>  CT= from eigenvalue problem
    //----------------------------------------------------------------
    // CT = (p+1)*(p+d)/d (this estimate leads to the same results as solving a local eigenvalue
    // problem) 3D tetrahedral elements
    case Core::FE::CellType::tet4:
      return 1.0;
      break;  /// d=3, p(v_h) = 1 -> p(grad(v_h)) = 0   =>  CT=1.0
    case Core::FE::CellType::tet10:
      return 2.6666666666666666;
      break;  /// d=3, p(v_h) = 2 -> p(grad(v_h)) = 1   =>  CT=2.6666666666666666 (8/3)
    // 2D triangular elements
    case Core::FE::CellType::tri3:
      return 1.0;
      break;  /// d=2, p(v_h) = 1 -> p(grad(v_h)) = 0   =>  CT=1.0
    case Core::FE::CellType::tri6:
      return 3.0;
      break;  /// d=2, p(v_h) = 2 -> p(grad(v_h)) = 1   =>  CT=3.0
    // 1D line elements
    case Core::FE::CellType::line2:
      return 1.0;
      break;  /// d=1, p(v_h) = 1 -> p(grad(v_h)) = 0   =>  CT=1.0
    case Core::FE::CellType::line3:
      return 4.0;
      break;  /// d=1, p(v_h) = 1 -> p(grad(v_h)) = 0   =>  CT=4.0
    //----------------------------------------------------------------
    // 3D wedge/pyramid elements, the current estimates are taken from the maximum of hex and tet
    // elements, the correct value has to switch between the faces!
    case Core::FE::CellType::pyramid5:
      std::cout << "WARNING: calibrate this value!" << std::endl;
      return 1.59307;
      break;  /// d=3, p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT taken from hex8
    case Core::FE::CellType::wedge6:
      std::cout << "WARNING: calibrate this value!" << std::endl;
      return 1.59307;
      break;  /// d=3, p(v_h) = 1 -> p(grad(v_h)) = 1   =>  CT taken from hex8
    case Core::FE::CellType::wedge15:
      std::cout << "WARNING: calibrate this value!" << std::endl;
      return 4.10462;
      break;  /// d=3, p(v_h) = 2 -> p(grad(v_h)) = 2   =>  CT taken from hex20
    default:
      FOUR_C_THROW("constant for trace inequality not specified for this element type yet: {}",
          trace_inequality_distype);
      break;
  };

  return 0.0;
}

/*----------------------------------------------------------------------*
 * Get Stabilization Parameters (for Penalty and Adjoint)
 * for Navier Slip Formulation
 *----------------------------------------------------------------------*/
void XFEM::Utils::get_navier_slip_stabilization_parameters(
    const double& NIT_visc_stab_fac,  ///< viscous Nitsche stab fac
    double& dynvisc,                  ///< average dynamic viscosity
    double& sliplength,               ///< sliplength
    double& stabnit,                  ///< stabilization factor NIT_Penalty
    double& stabadj                   ///< stabilization factor Adjoint
)
{
  // Create stabilization parameters needed for the Robin case
  //  NIT_visc_stab_fac     =   gamma' * mu * C^2
  //  NIT_visc_stab_fac_inv =   ( (1/gamma)*h_E )

  double NIT_visc_stab_fac_inv;

  if (NIT_visc_stab_fac <= 0.0)
    NIT_visc_stab_fac_inv = 1e15;  // If Nitsche parameter is 0
  else
    NIT_visc_stab_fac_inv = 1.0 / (NIT_visc_stab_fac / dynvisc);

  //  NIT_robin_denominator = [ mu/(epislon + gamma*h_E) ]
  double NIT_robin_denominator_no_mu = 1.0 / (sliplength + NIT_visc_stab_fac_inv);
  // NIT_robin_denominator_ = dyn_visc_*NIT_robin_denominator_no_mu;

  // Nitsche penalty term stabilization in tangential direction
  // ----------------------------------------------------------------------
  // stabnit =  [ { mu }/(epsilon + gamma*h_E) ) + extra_terms]
  stabnit = dynvisc * NIT_robin_denominator_no_mu;  // NIT_robin_denominator_;

  //  stabepsnit = [ epsilon / (epsilon + gamma*h_E) + extra_terms ];
  // stabepsnit = NIT_robin_denominator_no_mu*sliplength;

  // ----------------------------------------------------------------------

  // Adjoint-terms stabilization in tangential direction
  // ----------------------------------------------------------------------
  //  stabadj = [ gamma*h_E  /(epsilon + gamma*h_E)  ]
  stabadj = NIT_robin_denominator_no_mu * NIT_visc_stab_fac_inv;

#ifdef ENFORCE_URQUIZA_GNBC
  // In tangential direction stabilization:
  stabnit = (dynvisc / sliplength);  // Appears in NIT-penalty stabilization
  stabadj = 0.0;
  stabepsnit = 0.0;
  stabepsadj = 0.0;
#endif

  return;
}

/*--------------------------------------------------------------------------------
 * compute transformation factor for surface integration, normal, local and global gp coordinates
 *--------------------------------------------------------------------------------*/
void XFEM::Utils::compute_surface_transformation(double& drs,  ///< surface transformation factor
    Core::LinAlg::Matrix<3, 1>& x_gp_lin,  ///< global coordinates of gaussian point
    Core::LinAlg::Matrix<3, 1>& normal,    ///< normal vector on boundary cell
    Cut::BoundaryCell* bc,                 ///< boundary cell
    const Core::LinAlg::Matrix<2, 1>&
        eta,           ///< local coordinates of gaussian point w.r.t boundarycell
    bool referencepos  ///< use the bc reference position for transformation
)
{
  normal.clear();

  // get normal vector on linearized boundary cell, x-coordinates of gaussian point and surface
  // transformation factor
  switch (bc->shape())
  {
    case Core::FE::CellType::tri3:
    {
      bc->transform<Core::FE::CellType::tri3>(eta, x_gp_lin, normal, drs, referencepos);
      break;
    }
    case Core::FE::CellType::quad4:
    {
      bc->transform<Core::FE::CellType::quad4>(eta, x_gp_lin, normal, drs, referencepos);
      break;
    }
    default:
      FOUR_C_THROW("unsupported integration cell type");
  }

  return;
}

/*--------------------------------------------------------------------------------
 * pre-compute the measure of all side's surface cutting the element
 *--------------------------------------------------------------------------------*/
double XFEM::Utils::compute_meas_cut_surf(
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&
        bintpoints,  ///< boundary cell integration points
    const std::map<int, std::vector<Cut::BoundaryCell*>>& bcells  ///< boundary cells
)
{
  double surf = 0.0;

  //--------------------------------------------
  // loop intersecting sides
  // map of side-element id and Gauss points
  for (std::map<int, std::vector<Core::FE::GaussIntegration>>::const_iterator i =
           bintpoints.begin();
      i != bintpoints.end(); ++i)
  {
    int sid = i->first;
    const std::vector<Core::FE::GaussIntegration>& cutintpoints = i->second;

    // get side's boundary cells
    std::map<int, std::vector<Cut::BoundaryCell*>>::const_iterator j = bcells.find(sid);
    if (j == bcells.end()) FOUR_C_THROW("missing boundary cell");

    const std::vector<Cut::BoundaryCell*>& bcs = j->second;
    if (bcs.size() != cutintpoints.size()) FOUR_C_THROW("boundary cell integration rules mismatch");

    //--------------------------------------------
    // loop boundary cells w.r.t current cut side
    //--------------------------------------------
    for (std::vector<Core::FE::GaussIntegration>::const_iterator i = cutintpoints.begin();
        i != cutintpoints.end(); ++i)
    {
      const Core::FE::GaussIntegration& gi = *i;
      Cut::BoundaryCell* bc = bcs[i - cutintpoints.begin()];  // get the corresponding boundary cell

      //--------------------------------------------
      // loop gausspoints w.r.t current boundary cell
      //--------------------------------------------
      for (Core::FE::GaussIntegration::iterator iquad = gi.begin(); iquad != gi.end(); ++iquad)
      {
        double drs =
            0.0;  // transformation factor between reference cell and linearized boundary cell

        const Core::LinAlg::Matrix<2, 1> eta(iquad.point());  // xi-coordinates with respect to side

        Core::LinAlg::Matrix<3, 1> normal(Core::LinAlg::Initialization::zero);

        Core::LinAlg::Matrix<3, 1> x_gp_lin(
            Core::LinAlg::Initialization::zero);  // gp in xyz-system on linearized interface

        // compute transformation factor, normal vector and global Gauss point coordinates
        if (bc->shape() != Core::FE::CellType::dis_none)  // Tessellation approach
        {
          XFEM::Utils::compute_surface_transformation(drs, x_gp_lin, normal, bc, eta);
        }
        else  // MomentFitting approach
        {
          drs = 1.0;
          normal = bc->get_normal_vector();
          const double* gpcord = iquad.point();
          for (int idim = 0; idim < 3; ++idim)
          {
            x_gp_lin(idim, 0) = gpcord[idim];
          }
        }

        const double surf_fac = drs * iquad.weight();

        surf += surf_fac;

      }  // loop gausspoints w.r.t current boundary cell
    }  // loop boundary cells
  }  // loop intersecting sides

  return surf;
}

/*--------------------------------------------------------------------------------
 * compute the measure of the elements surface with given local id
 *--------------------------------------------------------------------------------*/
double XFEM::Utils::compute_meas_face(Core::Elements::Element* ele,  ///< fluid element
    Core::LinAlg::SerialDenseMatrix& ele_xyze,                       ///< element coordinates
    const int local_face_id,  ///< the local id of the face w.r.t the fluid element
    const int nsd             ///< number of space dimensions
)
{
  // get the shape of the face
  Core::FE::CellType face_shape = Core::FE::get_ele_face_shape_type(ele->shape(), local_face_id);

  // get the current node coordinates, extract them from the element's node coordinates
  const int numnode_face = Core::FE::get_number_of_element_nodes(face_shape);
  Core::LinAlg::SerialDenseMatrix xyze_face(nsd, numnode_face);

  // map for numbering of nodes of the surfaces
  std::vector<std::vector<int>> map = Core::FE::get_ele_node_numbering_faces(ele->shape());

  // extract the surface's node coordinates from the element's nodes coordinates
  for (int n = 0; n < numnode_face; ++n)
  {
    const int node_lid = map[local_face_id][n];
    for (int idim = 0; idim < nsd; ++idim) xyze_face(idim, n) = ele_xyze(idim, node_lid);
  }

  // the metric tensor and the area of an infinitesimal surface element
  Core::LinAlg::SerialDenseMatrix metrictensor(nsd - 1, nsd - 1);
  double drs = 0.0;

  if (nsd != 3)
    FOUR_C_THROW("don't call this function for non-3D examples, adapt the following for 2D!");

  Core::FE::GaussRule2D gaussrule = Core::FE::GaussRule2D::undefined;
  switch (face_shape)
  {
    case Core::FE::CellType::quad4:
    case Core::FE::CellType::quad8:
    case Core::FE::CellType::quad9:
      gaussrule = Core::FE::GaussRule2D::quad_1point;
      break;
    case Core::FE::CellType::tri3:
    case Core::FE::CellType::tri6:
      gaussrule = Core::FE::GaussRule2D::tri_1point;
      break;
    default:
      FOUR_C_THROW("shape type unknown!\n");
      break;
  }

  double meas_face = 0.0;

  /*----------------------------------------------------------------------*
    |               start loop over integration points                     |
   *----------------------------------------------------------------------*/
  const Core::FE::IntegrationPoints2D intpoints(gaussrule);
  for (int gpid = 0; gpid < intpoints.nquad; ++gpid)
  {
    const double e0 = intpoints.qxg[gpid][0];
    const double e1 = intpoints.qxg[gpid][1];

    Core::LinAlg::SerialDenseMatrix deriv(nsd - 1, numnode_face);

    // get shape functions and derivatives in the plane of the element
    Core::FE::shape_function_2d_deriv1(deriv, e0, e1, face_shape);

    // compute measure tensor for surface element and the infinitesimal
    // area element drs for the integration
    Core::FE::compute_metric_tensor_for_surface(xyze_face, deriv, metrictensor, &drs);

    meas_face += intpoints.qwgt[gpid] * drs;
  }

  return meas_face;
}

/*----------------------------------------------------------------------*
 | evaluate element volume                                              |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double XFEM::Utils::eval_element_volume(Core::LinAlg::Matrix<3, Core::FE::num_nodes<distype>> xyze,
    Core::LinAlg::Matrix<Core::FE::num_nodes<distype>, 1>* nurbs_weights,
    std::vector<Core::LinAlg::SerialDenseVector>* nurbs_knots)
{
  static const int nsd = 3;
  static const int nen = Core::FE::num_nodes<distype>;

  // use one-point Gauss rule
  Core::FE::IntPointsAndWeights<nsd> intpoints_stab(
      Discret::Elements::DisTypeToStabGaussRule<distype>::rule);

  const double* gpcoord = intpoints_stab.ip().qxg[0];   // actual integration point (coords)
  const double gpweight = intpoints_stab.ip().qwgt[0];  // actual integration point (weight)

  Core::LinAlg::Matrix<nsd, 1> xsi(gpcoord, true);
  static Core::LinAlg::Matrix<nen, 1> funct;
  static Core::LinAlg::Matrix<nsd, nen> deriv;
  static Core::LinAlg::Matrix<nsd, nsd> xjm;
  static Core::LinAlg::Matrix<nsd, nsd> xji;

  switch (distype)
  {
    case Core::FE::CellType::hex8:
    case Core::FE::CellType::hex20:
    case Core::FE::CellType::hex27:
    case Core::FE::CellType::tet4:
    case Core::FE::CellType::tet10:
    case Core::FE::CellType::wedge6:
    case Core::FE::CellType::wedge15:
    {
      // shape functions and their first derivatives
      Core::FE::shape_function<distype>(xsi, funct);
      Core::FE::shape_function_deriv1<distype>(xsi, deriv);
      break;
    }
    case Core::FE::CellType::nurbs8:
    case Core::FE::CellType::nurbs27:
    {
      if (nurbs_weights == nullptr || nurbs_knots == nullptr)
        FOUR_C_THROW("For Nurbs elements, weights and knots are required!");
      Core::FE::Nurbs::nurbs_get_funct_deriv(
          funct, deriv, xsi, *nurbs_knots, *nurbs_weights, distype);
      break;
    }
    default:
      FOUR_C_THROW("Distype not handled yet!");
  }

  //

  // get Jacobian matrix and determinant
  // actually compute its transpose....
  /*
    +-            -+ T      +-            -+
    | dx   dx   dx |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dr   dr   dr |
    |              |        |              |
    | dy   dy   dy |        | dx   dy   dz |
    | --   --   -- |   =    | --   --   -- |
    | dr   ds   dt |        | ds   ds   ds |
    |              |        |              |
    | dz   dz   dz |        | dx   dy   dz |
    | --   --   -- |        | --   --   -- |
    | dr   ds   dt |        | dt   dt   dt |
    +-            -+        +-            -+
  */
  xjm.multiply_nt(deriv, xyze);
  double det = xji.invert(xjm);

  if (det < 1E-16) FOUR_C_THROW("GLOBAL ELEMENT ZERO OR NEGATIVE JACOBIAN DETERMINANT: {}", det);

  // compute integration factor
  return gpweight * det;
}

/*--------------------------------------------------------------------------------
 * compute characteristic element length
 *--------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
double XFEM::Utils::compute_char_ele_length(Core::Elements::Element* ele,  ///< fluid element
    Core::LinAlg::SerialDenseMatrix& ele_xyze,                             ///< element coordinates
    XFEM::ConditionManager& cond_manager,    ///< XFEM condition manager
    const Cut::plain_volumecell_set& vcSet,  ///< volumecell sets for volume integration
    const std::map<int, std::vector<Cut::BoundaryCell*>>&
        bcells,  ///< bcells for boundary cell integration
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&
        bintpoints,  ///< integration points for boundary cell integration
    const Inpar::XFEM::ViscStabHk visc_stab_hk,  ///< h definition
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<distype>>
        emb,                       ///< pointer to the embedded coupling implementation
    Core::Elements::Element* face  ///< side element in 3D
)
{
  static const int nsd = 3;
  // TEUCHOS_FUNC_TIME_MONITOR("FluidEleCalcXFEM::compute_char_ele_length");

  Core::LinAlg::Matrix<3, Core::FE::num_nodes<distype>> xyze(ele_xyze, true);
  const int coup_sid = bintpoints.begin()->first;
  const Inpar::XFEM::AveragingStrategy averaging_strategy =
      cond_manager.get_averaging_strategy(coup_sid, ele->id());
  if (emb == nullptr and averaging_strategy == Inpar::XFEM::Embedded_Sided)
    FOUR_C_THROW("no coupling interface available, however Embedded_Sided coupling is activated!");

  // characteristic element length to be computed
  double h_k = 0.0;

  // measure of the face (surface in 3D, line in 2D) or measure of the cut-face
  double meas_surf = 0.0;

  // measure of the element volume or measure of the physical cut part of the element volume
  double meas_vol = 0.0;

  switch (visc_stab_hk)
  {
    //---------------------------------------------------
    // volume-equivalent diameter
    //---------------------------------------------------
    case Inpar::XFEM::ViscStab_hk_vol_equivalent:
    {
      // evaluate shape functions and derivatives at element center
      if (averaging_strategy == Inpar::XFEM::Embedded_Sided)
      {
        // evaluate shape functions and derivatives at element center w.r.t embedded element
        meas_vol = emb->eval_element_volume();
      }
      else
      {
        // meas_vol =
        // XFEM::Utils::eval_element_volume<distype>(my::xyze_,&(my::weights_),&(my::myknots_));
        meas_vol = XFEM::Utils::eval_element_volume<distype>(xyze);
      }

      // compute h_k as volume-equivalent diameter and directly return the value
      return h_k = XFEM::Utils::compute_vol_eq_diameter(meas_vol);

      break;
    }
    //---------------------------------------------------
    // compute h_k as physical/cut volume divided by physical partial/cut surface measure
    // ( used to estimate the cut-dependent inverse estimate on cut elements, not useful for sliver
    // and/or dotted cut situations)
    //---------------------------------------------------
    case Inpar::XFEM::ViscStab_hk_cut_vol_div_by_cut_surf:
    {
      if (averaging_strategy == Inpar::XFEM::Embedded_Sided)
        FOUR_C_THROW(
            "ViscStab_hk_cut_vol_div_by_cut_surf not reasonable for Embedded_Sided_Coupling!");

      // compute the cut surface measure
      meas_surf = XFEM::Utils::compute_meas_cut_surf(bintpoints, bcells);

      if (fabs(meas_surf) < 1.e-8) FOUR_C_THROW("Element contribution to interface has zero size.");

      // compute the cut volume measure
      for (Cut::plain_volumecell_set::const_iterator i = vcSet.begin(); i != vcSet.end(); ++i)
      {
        Cut::VolumeCell* vc = *i;
        meas_vol += vc->volume();
      }

      if (meas_vol < 0.0)
        FOUR_C_THROW(
            " measure of cut partial volume is smaller than 0.0: {} Attention with increasing "
            "Nitsche-Parameter!!!",
            meas_vol);

      break;
    }
    //---------------------------------------------------
    // full element volume divided by physical partial/cut surface measure ( used to estimate the
    // cut-dependent inverse estimate on cut elements, however, avoids problems with sliver cuts,
    // not useful for dotted cuts)
    //---------------------------------------------------
    case Inpar::XFEM::ViscStab_hk_ele_vol_div_by_cut_surf:
    {
      if (averaging_strategy == Inpar::XFEM::Embedded_Sided)
        FOUR_C_THROW(
            "ViscStab_hk_ele_vol_div_by_cut_surf not reasonable for Embedded_Sided_Coupling!");

      // compute the cut surface measure
      meas_surf = XFEM::Utils::compute_meas_cut_surf(bintpoints, bcells);

      // compute the full element volume measure
      meas_vol = XFEM::Utils::eval_element_volume<distype>(xyze);

      break;
    }
    //---------------------------------------------------
    // full element volume divided by surface measure ( used for uncut situations, standard weak
    // Dirichlet boundary/coupling conditions)
    //---------------------------------------------------
    case Inpar::XFEM::ViscStab_hk_ele_vol_div_by_ele_surf:
    {
      if (averaging_strategy != Inpar::XFEM::Embedded_Sided)
        FOUR_C_THROW(
            "ViscStab_hk_ele_vol_div_by_ele_surf just reasonable for Embedded_Sided_Coupling!");

      Core::Elements::FaceElement* fele = dynamic_cast<Core::Elements::FaceElement*>(face);
      if (!fele) FOUR_C_THROW("Cast to FaceElement failed!");

      //---------------------------------------------------
      // compute the uncut element's surface measure
      meas_surf = XFEM::Utils::compute_meas_face(ele, ele_xyze, fele->face_parent_number(), nsd);

      // compute the full element volume measure
      meas_vol = emb->eval_element_volume();

      break;
    }
    //---------------------------------------------------
    case Inpar::XFEM::ViscStab_hk_ele_vol_div_by_max_ele_surf:
      //---------------------------------------------------
      {
        if (averaging_strategy == Inpar::XFEM::Embedded_Sided)
          FOUR_C_THROW(
              "ViscStab_hk_ele_vol_div_by_max_ele_surf not reasonable for "
              "Embedded_Sided_Coupling!");

        // compute the uncut element's surface measure
        const int numfaces = Core::FE::get_number_of_element_faces(ele->shape());

        // loop all surfaces
        for (int lid = 0; lid < numfaces; ++lid)
        {
          meas_surf = std::max(meas_surf, XFEM::Utils::compute_meas_face(ele, ele_xyze, lid, nsd));
        }

        // compute the full element volume measure
        meas_vol = XFEM::Utils::eval_element_volume<distype>(xyze);

        break;
      }
    default:
      FOUR_C_THROW("unknown type of characteristic element length");
      break;
  }

  //--------------------------------------
  // compute the final element length if fraction-based computation and not returned yet
  h_k = meas_vol / meas_surf;
  //--------------------------------------

  // check plausibility
  if (h_k < 1e-14)
    FOUR_C_THROW(
        "the characteristic element length is zero or smaller, it has not been set properly!");

  return h_k;
}

/*--------------------------------------------------------------------------------
 *    compute stabilization factor for the Nitsche's penalty term
 *--------------------------------------------------------------------------------*/
void XFEM::Utils::nit_compute_full_penalty_stabfac(
    double& NIT_full_stab_fac,  ///< to be filled: full Nitsche's penalty term scaling
                                ///< (viscous+convective part)
    const Core::LinAlg::Matrix<3, 1>& normal,    ///< interface-normal vector
    const double h_k,                            ///< characteristic element length
    const double kappa_m,                        ///< Weight parameter (parameter +/master side)
    const double kappa_s,                        ///< Weight parameter (parameter -/slave  side)
    const Core::LinAlg::Matrix<3, 1>& velint_m,  ///< Master side velocity at gauss-point
    const Core::LinAlg::Matrix<3, 1>& velint_s,  ///< Slave side velocity at gauss-point
    const double NIT_visc_stab_fac,              ///< Nitsche's viscous scaling part of penalty term
    const double timefac,                        ///< timefac
    const bool isstationary,                     ///< isstationary
    const double densaf_master,                  ///< master density
    const double densaf_slave,                   ///< slave density
    Inpar::XFEM::MassConservationScaling
        mass_conservation_scaling,  ///< kind of mass conservation scaling
    Inpar::XFEM::MassConservationCombination
        mass_conservation_combination,  ///< kind of mass conservation combination
    const double NITStabScaling,        ///< scaling of nit stab fac
    Inpar::XFEM::ConvStabScaling
        ConvStabScaling,  ///< which convective stab. scaling of inflow stab
    Inpar::XFEM::XffConvStabScaling
        XFF_ConvStabScaling,    ///< which convective stab. scaling on XFF interface
    const bool IsConservative,  ///< conservative formulation of navier stokes
    bool error_calc             ///< when called in error calculation, don't add the inflow terms
)
{
  // TEUCHOS_FUNC_TIME_MONITOR("XFEM::Utils::::nit_compute_full_penalty_stabfac");

  //------------------------------------------------------------------------------
  // compute the full Nitsche parameter

  /*
   * Depending on the flow regime, the factor alpha of Nitsches penalty term
   * (\alpha * [v],[u]) can take various forms.
   * Based on Inpar::XFEM::mass_conservation_combination, we choose:
   *
   *                       (1)           (2)          (3)
   *                    /  \mu    \rho             h * \rho         \
   *  NIT :=  \gamma * |    --  +  -- * |u|_inf  + ----------------- |
   *                    \   h      6               12 * \theta * dt /
   *
   *       OR:
   *                    /  \mu    \rho             h * \rho         \
   *  NIT :=  max      |    --  ;  -- * |u|_inf  ; ----------------- | *\gamma
   *                    \   h      6               12 * \theta * dt /
   *
   *          (1) NIT_visc_stab_fac = \gamma * \mu/h
   *
   *          (2) convective contribution
   *
   *          (3) transient contribution
   *
   * see Schott and Rasthofer, 'A face-oriented stabilized Nitsche-type extended variational
   * multiscale method for incompressible two-phase flow', Int. J. Number. Meth. Engng, 2014
   *
   *
   * If Inpar::XFEM::MassConservationScaling_only_visc is set, we choose only (1),
   * no matter if the combination option max or sum is active!
   *
   */


  // (1)
  NIT_full_stab_fac = NIT_visc_stab_fac;

  if (mass_conservation_scaling == Inpar::XFEM::MassConservationScaling_full)
  {
    // TODO: Raffaela: which velocity has to be evaluated for these terms in ALE? the convective
    // velocity or the velint?
    double velnorminf_m = velint_m.norm_inf();  // relative convective velocity
    double velnorminf_s = velint_s.norm_inf();

    // take the maximum of viscous & convective contribution or the sum?
    if (mass_conservation_combination == Inpar::XFEM::MassConservationCombination_max)
    {
      NIT_full_stab_fac =
          std::max(NIT_full_stab_fac, NITStabScaling *
                                          (kappa_m * densaf_master * fabs(velnorminf_m) +
                                              kappa_s * densaf_slave * fabs(velnorminf_s)) /
                                          6.0);
      if (!isstationary)
        NIT_full_stab_fac =
            std::max(NITStabScaling * h_k * (kappa_m * densaf_master + kappa_s * densaf_slave) /
                         (12.0 * timefac),
                NIT_full_stab_fac);
    }
    else  // the sum
    {
      // (2)
      NIT_full_stab_fac += NITStabScaling *
                           (kappa_m * densaf_master * fabs(velnorminf_m) +
                               kappa_s * densaf_slave * fabs(velnorminf_s)) /
                           6.0;  // THIS ONE NEEDS CHANGING!

      // (3)
      if (!isstationary)
        NIT_full_stab_fac += NITStabScaling * h_k *
                             (kappa_m * densaf_master + kappa_s * densaf_slave) / (12.0 * timefac);
    }
  }
  else if (mass_conservation_scaling != Inpar::XFEM::MassConservationScaling_only_visc)
    FOUR_C_THROW("Unknown scaling choice in calculation of Nitsche's penalty parameter");

  if (IsConservative and (XFF_ConvStabScaling != Inpar::XFEM::XFF_ConvStabScaling_none or
                             ConvStabScaling != Inpar::XFEM::ConvStabScaling_none))
  {
    FOUR_C_THROW(
        "convective stabilization is not available for conservative form of Navier-Stokes, but "
        "possible to implement!");
  }

  //----------------------------------------------------------------------------------------------
  // add inflow terms to ensure coercivity at inflow boundaries in the convective limit

  if ((ConvStabScaling == Inpar::XFEM::ConvStabScaling_none &&
          XFF_ConvStabScaling == Inpar::XFEM::XFF_ConvStabScaling_none) ||
      XFF_ConvStabScaling == Inpar::XFEM::XFF_ConvStabScaling_only_averaged || error_calc)
    return;

  const double veln_normal = velint_m.dot(normal);

  double NIT_inflow_stab = 0.0;

  if (XFF_ConvStabScaling == Inpar::XFEM::XFF_ConvStabScaling_upwinding)
  {
    NIT_inflow_stab = fabs(veln_normal) * 0.5;
  }
  else
  {
    if (ConvStabScaling == Inpar::XFEM::ConvStabScaling_abs_inflow)
    {
      //      | u*n |
      NIT_inflow_stab = fabs(veln_normal);
    }
    else if (ConvStabScaling == Inpar::XFEM::ConvStabScaling_inflow)
    {
      //      ( -u*n ) if (u*n)<0 (inflow), conv_stabfac >= 0
      NIT_inflow_stab = std::max(0.0, -veln_normal);
    }
    else
      FOUR_C_THROW("No valid Inpar::XFEM::ConvStabScaling for xfluid/xfsi problems");
  }

  NIT_inflow_stab *= densaf_master;  // my::densaf_;

  // Todo (kruse): it is planned to add the inflow contributions independent from the max. option!
  // This version is only kept to shift the adaption of test results to a single commit.
  if (mass_conservation_combination == Inpar::XFEM::MassConservationCombination_max)
  {
    NIT_full_stab_fac = std::max(NIT_full_stab_fac, NIT_inflow_stab);
  }
  else if (mass_conservation_combination == Inpar::XFEM::MassConservationCombination_sum)
  {
    NIT_full_stab_fac += NIT_inflow_stab;
  }
  else
    FOUR_C_THROW("Unknown combination type in calculation of Nitsche's penalty parameter.");

  return;
}

double XFEM::Utils::evaluate_full_traction(const double& pres_m,
    const Core::LinAlg::Matrix<3, 3>& vderxy_m, const double& visc_m, const double& penalty_fac,
    const Core::LinAlg::Matrix<3, 1>& vel_m, const Core::LinAlg::Matrix<3, 1>& vel_s,
    const Core::LinAlg::Matrix<3, 1>& elenormal, const Core::LinAlg::Matrix<3, 1>& normal,
    const Core::LinAlg::Matrix<3, 1>& velpf_s, double porosity)
{
  Core::LinAlg::Matrix<3, 1> traction(Core::LinAlg::Initialization::zero);

  // pressure contribution
  if (porosity <= 0)
  {
    for (int i = 0; i < 3; ++i)
      traction(i, 0) = -pres_m * elenormal(i, 0) - penalty_fac * (vel_m(i, 0) - vel_s(i, 0));
  }
  else
  {
    for (int i = 0; i < 3; ++i)
      traction(i, 0) =
          -pres_m * elenormal(i, 0) -
          penalty_fac * (vel_m(i, 0) - (1 - porosity) * vel_s(i, 0) - porosity * velpf_s(i, 0));
  }

  // viscous contribution
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      traction(i, 0) += visc_m * (vderxy_m(i, j) + vderxy_m(j, i)) * elenormal(j, 0);

  return traction.dot(elenormal);
}

double XFEM::Utils::evaluate_full_traction(const Core::LinAlg::Matrix<3, 1>& intraction,
    const double& penalty_fac, const Core::LinAlg::Matrix<3, 1>& vel_m,
    const Core::LinAlg::Matrix<3, 1>& vel_s, const Core::LinAlg::Matrix<3, 1>& elenormal,
    const Core::LinAlg::Matrix<3, 1>& normal)
{
  Core::LinAlg::Matrix<3, 1> traction(Core::LinAlg::Initialization::zero);

  for (int i = 0; i < 3; ++i)
    traction(i, 0) = intraction(i, 0) - penalty_fac * (vel_m(i, 0) - vel_s(i, 0));
  return traction.dot(elenormal);
}

double XFEM::Utils::evaluate_full_traction(const double& intraction, const double& penalty_fac,
    const Core::LinAlg::Matrix<3, 1>& vel_m, const Core::LinAlg::Matrix<3, 1>& vel_s,
    const Core::LinAlg::Matrix<3, 1>& elenormal, const Core::LinAlg::Matrix<3, 1>& normal)
{
  Core::LinAlg::Matrix<3, 1> traction(Core::LinAlg::Initialization::zero);

  for (int i = 0; i < 3; ++i) traction(i, 0) = -penalty_fac * (vel_m(i, 0) - vel_s(i, 0));
  return intraction + traction.dot(elenormal);
}

void XFEM::Utils::evaluate_stateat_gp(const Core::Elements::Element* sele,
    const Core::LinAlg::Matrix<3, 1>& selexsi, const Core::FE::Discretization& discret,
    const std::string& state, Core::LinAlg::Matrix<3, 1>& vel_s)
{
  vel_s.clear();
  Core::Elements::LocationArray las(1);
  sele->location_vector(discret, las, false);
  std::shared_ptr<const Core::LinAlg::Vector<double>> matrix_state = discret.get_state(state);
  std::vector<double> ivel = Core::FE::extract_values(*matrix_state, las[0].lm_);

  // 4 // evaluate slave velocity at guasspoint
  if (sele->shape() == Core::FE::CellType::quad4)
  {
    Core::LinAlg::Matrix<3, 4> vels;
    for (int n = 0; n < sele->num_node(); ++n)
    {
      for (int dof = 0; dof < 3; ++dof)
      {
        vels(dof, n) = ivel[n * 3 + dof];
      }
    }

    const int numnodes = Core::FE::num_nodes<Core::FE::CellType::quad4>;
    static Core::LinAlg::Matrix<numnodes, 1> funct(Core::LinAlg::Initialization::uninitialized);
    Core::FE::shape_function_2d(funct, selexsi(0), selexsi(1), Core::FE::CellType::quad4);
    vel_s.multiply(vels, funct);
  }
  else
    FOUR_C_THROW("Your Slave Element is not a quad4?");
}


template double XFEM::Utils::eval_element_volume<Core::FE::CellType::hex8>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::hex8>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::hex8>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);
template double XFEM::Utils::eval_element_volume<Core::FE::CellType::hex20>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::hex20>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::hex20>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);
template double XFEM::Utils::eval_element_volume<Core::FE::CellType::hex27>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::hex27>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::hex27>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);
template double XFEM::Utils::eval_element_volume<Core::FE::CellType::tet4>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::tet4>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tet4>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);
template double XFEM::Utils::eval_element_volume<Core::FE::CellType::tet10>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::tet10>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::tet10>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);
template double XFEM::Utils::eval_element_volume<Core::FE::CellType::wedge6>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::wedge6>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::wedge6>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);
template double XFEM::Utils::eval_element_volume<Core::FE::CellType::wedge15>(
    Core::LinAlg::Matrix<3, Core::FE::num_nodes<Core::FE::CellType::wedge15>>,
    Core::LinAlg::Matrix<Core::FE::num_nodes<Core::FE::CellType::wedge15>, 1>*,
    std::vector<Core::LinAlg::SerialDenseVector>*);

template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::hex8>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::hex8>>,
    Core::Elements::Element*);
template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::hex20>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::hex20>>,
    Core::Elements::Element*);
template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::hex27>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::hex27>>,
    Core::Elements::Element*);
template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::tet4>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::tet4>>,
    Core::Elements::Element*);
template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::tet10>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::tet10>>,
    Core::Elements::Element*);
template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::wedge6>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::wedge6>>,
    Core::Elements::Element*);
template double XFEM::Utils::compute_char_ele_length<Core::FE::CellType::wedge15>(
    Core::Elements::Element*, Core::LinAlg::SerialDenseMatrix&, XFEM::ConditionManager&,
    const Cut::plain_volumecell_set&, const std::map<int, std::vector<Cut::BoundaryCell*>>&,
    const std::map<int, std::vector<Core::FE::GaussIntegration>>&, const Inpar::XFEM::ViscStabHk,
    std::shared_ptr<Discret::Elements::XFLUID::SlaveElementInterface<Core::FE::CellType::wedge15>>,
    Core::Elements::Element*);

FOUR_C_NAMESPACE_CLOSE
