// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_general_utils_shapevalues_hdg.hpp"

#include "4C_fem_general_utils_boundary_integration.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN


/*----------------------------------------------------------------------*
 |  Constructor (public)                              kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::FE::ShapeValues<distype>::ShapeValues(
    const unsigned int degree, const bool completepoly, const unsigned int quadratureDegree)
    : degree_(degree),
      quadrature_(Core::FE::create_gauss_points(distype, quadratureDegree)),
      usescompletepoly_(completepoly),
      nqpoints_(quadrature_->num_points())
{
  PolynomialSpaceParams params(distype, degree, completepoly);
  polySpace_ = Core::FE::PolynomialSpaceCache<nsd_>::instance().create(params);
  ndofs_ = polySpace_->size();

  Core::LinAlg::SerialDenseVector values(ndofs_);
  Core::LinAlg::SerialDenseMatrix derivs(nsd_, ndofs_);

  xyzreal.shape(nsd_, nqpoints_);

  funct.shape(nen_, nqpoints_);
  derxy.shape(nen_ * nsd_, nqpoints_);

  shfunct.shape(ndofs_, nqpoints_);
  shfunctAvg.resize(ndofs_);
  shderiv.shape(ndofs_ * nsd_, nqpoints_);
  shderxy.shape(ndofs_ * nsd_, nqpoints_);
  jfac.resize(nqpoints_);

  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    // gauss point in real coordinates
    const double* gpcoord = quadrature_->point(q);
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = gpcoord[idim];

    polySpace_->evaluate(xsi, values);
    polySpace_->evaluate_deriv1(xsi, derivs);

    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      shfunct(i, q) = values(i);
      for (unsigned int d = 0; d < nsd_; ++d) shderiv(i * nsd_ + d, q) = derivs(d, i);
    }

    Core::LinAlg::Matrix<nen_, 1> myfunct(funct.values() + q * nen_, true);
    Core::FE::shape_function<distype>(xsi, myfunct);
  }

  // Fill support points
  nodexyzreal.shape(nsd_, ndofs_);
  polySpace_->fill_unit_node_points(nodexyzunit);
  FOUR_C_ASSERT(nodexyzreal.numRows() == nodexyzunit.numRows() &&
                    nodexyzreal.numCols() == nodexyzunit.numCols(),
      "Dimension mismatch");
}



/*----------------------------------------------------------------------*
 |  Evaluate element-dependent shape data (public)    kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Core::FE::ShapeValues<distype>::evaluate(
    const Core::Elements::Element& ele, const std::vector<double>& aleDis)
{
  FOUR_C_ASSERT(ele.shape() == distype, "Internal error");
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      &ele, xyze);

  // update nodal coordinates
  if (!(aleDis.empty()))
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) xyze(d, n) += aleDis[n * nsd_ + d];

  for (unsigned int i = 0; i < ndofs_; ++i) shfunctAvg(i) = 0.;
  double faceVol = 0.;

  // evaluate geometry
  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    const double* gpcoord = quadrature_->point(q);
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = gpcoord[idim];

    Core::FE::shape_function_deriv1<distype>(xsi, deriv);
    xjm.multiply_nt(deriv, xyze);
    jfac(q) = xji.invert(xjm) * quadrature_->weight(q);

    Core::LinAlg::Matrix<nen_, 1> myfunct(funct.values() + q * nen_, true);
    Core::LinAlg::Matrix<nsd_, 1> mypoint(xyzreal.values() + q * nsd_, true);
    mypoint.multiply_nn(xyze, myfunct);

    // compute global first derivates
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        derxy(n * nsd_ + d, q) = xji(d, 0) * deriv(0, n);
        for (unsigned int e = 1; e < nsd_; ++e) derxy(n * nsd_ + d, q) += xji(d, e) * deriv(e, n);
      }

    // transform shape functions
    for (unsigned int i = 0; i < ndofs_; ++i)
      for (unsigned int d = 0; d < nsd_; ++d)
      {
        shderxy(i * nsd_ + d, q) = xji(d, 0) * shderiv(i * nsd_, q);
        for (unsigned int e = 1; e < nsd_; ++e)
          shderxy(i * nsd_ + d, q) += xji(d, e) * shderiv(i * nsd_ + e, q);
      }

    for (unsigned int i = 0; i < ndofs_; ++i) shfunctAvg(i) += shfunct(i, q) * jfac(q);
    faceVol += jfac(q);
  }
  faceVol = 1. / faceVol;
  for (unsigned int i = 0; i < ndofs_; ++i) shfunctAvg(i) *= faceVol;

  // evaluate unit cell support points
  for (unsigned int i = 0; i < ndofs_; ++i)
  {
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = nodexyzunit(idim, i);

    Core::LinAlg::Matrix<nen_, 1> myfunct;
    // Core::FE::shape_function<Core::FE::DisTypeToFaceShapeType<distype>::shape>(xsi,myfunct);
    Core::FE::shape_function<distype>(xsi, myfunct);
    Core::LinAlg::Matrix<nsd_, 1> mypoint(nodexyzreal.values() + i * nsd_, true);
    mypoint.multiply_nn(xyze, myfunct);
  }
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                 schoeder 06/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Core::FE::ShapeValuesFace<distype>::ShapeValuesFace(ShapeValuesFaceParams params)
    : params_(params), degree_(params.degree_)
{
  if (nsd_ == 2)
    faceNodeOrder = Core::FE::get_ele_node_numbering_lines(distype);
  else if (nsd_ == 3)
    faceNodeOrder = Core::FE::get_ele_node_numbering_surfaces(distype);
  else
    FOUR_C_THROW("Not implemented for dim != 2, 3");

  PolynomialSpaceParams polyparams(
      Core::FE::DisTypeToFaceShapeType<distype>::shape, degree_, params.completepoly_);
  polySpace_ = Core::FE::PolynomialSpaceCache<nsd_ - 1>::instance().create(polyparams);

  nfdofs_ = polySpace_->size();
  quadrature_ = Core::FE::create_gauss_points(
      Core::FE::DisTypeToFaceShapeType<distype>::shape, params.quadraturedegree_);
  nqpoints_ = quadrature_->num_points();

  face_values_.size(nfdofs_);
  xyzreal.shape(nsd_, nqpoints_);
  funct.shape(nfn_, nqpoints_);

  shfunctNoPermute.shape(nfdofs_, nqpoints_);
  shfunct.shape(nfdofs_, nqpoints_);
  normals.shape(nsd_, nqpoints_);
  jfac.resize(nqpoints_);

  shfunctI.shape(nfdofs_, nqpoints_);

  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    const double* gpcoord = quadrature_->point(q);

    const unsigned int codim = nsd_ - 1;
    for (unsigned int idim = 0; idim < codim; idim++) xsi(idim) = gpcoord[idim];

    polySpace_->evaluate(xsi, face_values_);
    for (unsigned int i = 0; i < nfdofs_; ++i) shfunctNoPermute(i, q) = face_values_(i);

    Core::LinAlg::Matrix<nfn_, 1> myfunct(funct.values() + q * nfn_, true);
    Core::FE::shape_function<Core::FE::DisTypeToFaceShapeType<distype>::shape>(xsi, myfunct);
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate face-dependent shape data (public)       kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Core::FE::ShapeValuesFace<distype>::evaluate_face(
    const Core::Elements::Element& ele, const unsigned int face, const std::vector<double>& aleDis)
{
  const Core::FE::CellType facedis = Core::FE::DisTypeToFaceShapeType<distype>::shape;

  // get face position array from element position array
  FOUR_C_ASSERT(faceNodeOrder[face].size() == nfn_, "Internal error");

  Core::LinAlg::Matrix<nsd_, nen_> xyzeElement;
  Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
      &ele, xyzeElement);

  // update nodal coordinates
  if (!(aleDis.empty()))
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) xyzeElement(d, n) += aleDis[n * nsd_ + d];

  for (unsigned int i = 0; i < nfn_; ++i)
    for (unsigned int d = 0; d < nsd_; ++d) xyze(d, i) = xyzeElement(d, faceNodeOrder[face][i]);

  // Fill face support points
  nodexyzreal.shape(nsd_, nfdofs_);
  polySpace_->fill_unit_node_points(nodexyzunit);
  FOUR_C_ASSERT(nodexyzreal.numRows() == nodexyzunit.numRows() + 1 &&
                    nodexyzreal.numCols() == nodexyzunit.numCols(),
      "Dimension mismatch");
  for (unsigned int i = 0; i < nfdofs_; ++i)
  {
    for (unsigned int idim = 0; idim < nsd_ - 1; idim++) xsi(idim) = nodexyzunit(idim, i);

    Core::LinAlg::Matrix<nfn_, 1> myfunct;
    Core::FE::shape_function<Core::FE::DisTypeToFaceShapeType<distype>::shape>(xsi, myfunct);
    Core::LinAlg::Matrix<nsd_, 1> mypoint(nodexyzreal.values() + i * nsd_, true);
    mypoint.multiply_nn(xyze, myfunct);
  }

  // evaluate geometry
  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    const double* gpcoord = quadrature_->point(q);
    for (unsigned int idim = 0; idim < nsd_ - 1; idim++) xsi(idim) = gpcoord[idim];

    Core::FE::shape_function_deriv1<facedis>(xsi, deriv);
    double jacdet = 0.0;
    Core::FE::compute_metric_tensor_for_boundary_ele<facedis>(
        xyze, deriv, metricTensor, jacdet, &normal);
    for (unsigned int d = 0; d < nsd_; ++d) normals(d, q) = normal(d);
    jfac(q) = jacdet * quadrature_->weight(q);

    Core::LinAlg::Matrix<nfn_, 1> myfunct(funct.values() + q * nfn_, true);
    Core::LinAlg::Matrix<nsd_, 1> mypoint(xyzreal.values() + q * nsd_, true);
    mypoint.multiply_nn(xyze, myfunct);
  }

  adjust_face_orientation(ele, face);
  compute_face_reference_system(ele, face);

  Core::FE::ShapeValuesFaceParams interiorparams = params_;
  interiorparams.degree_ = ele.degree();
  interiorparams.face_ = face;
  shfunctI = *(ShapeValuesInteriorOnFaceCache<distype>::instance().create(interiorparams));
}



/*----------------------------------------------------------------------*
 |  Reorder evaluated face shape functions (private)  kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Core::FE::ShapeValuesFace<distype>::adjust_face_orientation(
    const Core::Elements::Element& ele, const unsigned int face)
{
  // For the shape values on faces, we need to figure out how the master element of
  // a face walks over the face and how the current face element wants to walk over
  // it. The local trafo map of the face element holds that information for the slave
  // side, whereas the master side always uses the correct mapping. Thus, we need not
  // do anything in that case.
  if (ele.faces()[face]->parent_master_element() == &ele)
  {
    for (unsigned int q = 0; q < nqpoints_; ++q)
      for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, q);
    return;
  }

  // For the adjusted slave slide, we need to change the order of quadrature
  // points in the shape functions of the trace to match the orientation in the
  // transformation.
  const std::vector<int>& trafomap = ele.faces()[face]->get_local_trafo_map();
  FOUR_C_ASSERT(trafomap.size() == nfn_,
      "Transformation map from slave face coordinate system to master coordinates has not been "
      "filled.");

  const int nqpoints1d = std::pow(nqpoints_ + 0.001, 1. / (nsd_ - 1));
  // easy case: standard orientation
  bool standard = true;
  for (int i = 0; i < static_cast<int>(nfn_); ++i)
    if (trafomap[i] != i) standard = false;
  if (standard)
  {
    for (unsigned int q = 0; q < nqpoints_; ++q)
      for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, q);
  }
  // OK, the orientation is different from what I expect. see if we can find it
  else
    switch (nsd_)
    {
      case 2:
        // face flipped is the only case
        {
          FOUR_C_ASSERT(trafomap[1] == 0 && trafomap[0] == 1, "Unknown face orientation in 2D");
          for (unsigned int q = 0; q < nqpoints_; ++q)
          {
            for (unsigned int i = 0; i < nfdofs_; ++i)
              shfunct(i, q) = shfunctNoPermute(i, nqpoints_ - 1 - q);
          }
        }
        break;
      case 3:
        if (distype == Core::FE::CellType::hex8 || distype == Core::FE::CellType::hex20 ||
            distype == Core::FE::CellType::hex27 || distype == Core::FE::CellType::nurbs8 ||
            distype == Core::FE::CellType::nurbs27)
        {
          if (trafomap[0] == 1 && trafomap[1] == 0 && trafomap[2] == 3 &&
              trafomap[3] == 2)  // x-direction mirrored
          {
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              const int ax = q % nqpoints1d;
              const int ay = q / nqpoints1d;
              int permute = nqpoints1d - 1 - ax + ay * nqpoints1d;
              for (unsigned int i = 0; i < nfdofs_; ++i)
                shfunct(i, q) = shfunctNoPermute(i, permute);
            }
          }
          else if (trafomap[0] == 0 && trafomap[1] == 3 && trafomap[2] == 2 &&
                   trafomap[3] == 1)  // permute x and y
          {
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              const int ax = q % nqpoints1d;
              const int ay = q / nqpoints1d;
              int permute = ay + ax * nqpoints1d;
              for (unsigned int i = 0; i < nfdofs_; ++i)
                shfunct(i, q) = shfunctNoPermute(i, permute);
            }
          }
          else if (trafomap[0] == 3 && trafomap[1] == 2 && trafomap[2] == 1 &&
                   trafomap[3] == 0)  // y mirrored
          {
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              const int ax = q % nqpoints1d;
              const int ay = q / nqpoints1d;
              int permute = ax + (nqpoints1d - 1 - ay) * nqpoints1d;
              for (unsigned int i = 0; i < nfdofs_; ++i)
                shfunct(i, q) = shfunctNoPermute(i, permute);
            }
          }
          else if (trafomap[0] == 2 && trafomap[1] == 3 && trafomap[2] == 0 &&
                   trafomap[3] == 1)  // x and y mirrored
          {
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              const int ax = q % nqpoints1d;
              const int ay = q / nqpoints1d;
              int permute = (nqpoints1d - 1 - ax) + (nqpoints1d - 1 - ay) * nqpoints1d;
              for (unsigned int i = 0; i < nfdofs_; ++i)
                shfunct(i, q) = shfunctNoPermute(i, permute);
            }
          }
          else if (trafomap[0] == 2 && trafomap[1] == 1 && trafomap[2] == 0 &&
                   trafomap[3] == 3)  // x and y mirrored and permuted
          {
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              const int ax = q % nqpoints1d;
              const int ay = q / nqpoints1d;
              int permute = (nqpoints1d - 1 - ay) +
                            (nqpoints1d - 1 - ax) *
                                nqpoints1d;  // note that this is for lexicographic ordering
              for (unsigned int i = 0; i < nfdofs_; ++i)
                shfunct(i, q) = shfunctNoPermute(i, permute);
            }
          }
          else
          {
            FOUR_C_THROW("Unknown HEX face orientation in 3D");
          }
        }
        else if (distype == Core::FE::CellType::tet4 || distype == Core::FE::CellType::tet10)
        {
          if (trafomap[0] == 1 && trafomap[1] == 0 && trafomap[2] == 2)
          {
            // zeroth and first node permuted; the search is a bit stupid right now
            // (one could cache the permutation coefficients r of the quadrature formula
            // but it does not seem worth it)
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              std::array<double, 2> point = {quadrature_->point(q)[0], quadrature_->point(q)[1]};
              point[0] = 1. - point[1] - point[0];
              unsigned int r = 0;
              for (; r < nqpoints_; ++r)
                if (std::abs(quadrature_->point(r)[0] - point[0]) < 1e-14 &&
                    std::abs(quadrature_->point(r)[1] - point[1]) < 1e-14)
                  break;
              if (r < nqpoints_)
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, r);
              else
              {
                xsi(0) = point[0];
                xsi(1) = point[1];
                polySpace_->evaluate(xsi, face_values_);
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = face_values_(i);
              }
            }
          }
          else if (trafomap[0] == 2 && trafomap[1] == 1 && trafomap[2] == 0)
          {
            // zeroth and second node permuted
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              std::array<double, 2> point = {quadrature_->point(q)[0], quadrature_->point(q)[1]};
              point[1] = 1. - point[1] - point[0];
              unsigned int r = 0;
              for (; r < nqpoints_; ++r)
                if (std::abs(quadrature_->point(r)[0] - point[0]) < 1e-14 &&
                    std::abs(quadrature_->point(r)[1] - point[1]) < 1e-14)
                  break;
              if (r < nqpoints_)
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, r);
              else
              {
                xsi(0) = point[0];
                xsi(1) = point[1];
                polySpace_->evaluate(xsi, face_values_);
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = face_values_(i);
              }
            }
          }
          else if (trafomap[0] == 0 && trafomap[1] == 2 && trafomap[2] == 1)
          {
            // first and second node permuted
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              std::array<double, 2> point = {quadrature_->point(q)[1], quadrature_->point(q)[0]};
              unsigned int r = 0;
              for (; r < nqpoints_; ++r)
                if (std::abs(quadrature_->point(r)[0] - point[0]) < 1e-14 &&
                    std::abs(quadrature_->point(r)[1] - point[1]) < 1e-14)
                  break;
              if (r < nqpoints_)
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, r);
              else
              {
                xsi(0) = point[0];
                xsi(1) = point[1];
                polySpace_->evaluate(xsi, face_values_);
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = face_values_(i);
              }
            }
          }
          else
          {
            for (unsigned int i = 0; i < 3; ++i) std::cout << trafomap[i] << " ";
            std::cout << std::endl << std::flush;
            // need to transform quadrature point coordinate 0 into 1-p[0]-p[1]
            FOUR_C_THROW("Unknown TET face orientation in 3D");
          }
        }
        else
          FOUR_C_THROW(
              "Shape type {} not yet implemented", (Core::FE::cell_type_to_string(distype)));
        break;
      default:
        FOUR_C_THROW("Only implemented in 2D and 3D");
        break;
    }
}



/*----------------------------------------------------------------------*
 |  Compute the face reference system       (private)  berardocco 09/18 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Core::FE::ShapeValuesFace<distype>::compute_face_reference_system(
    const Core::Elements::Element& ele, const unsigned int face)
{
  // In the case in which the element is not the master element for the face there is the need to
  // find the master element and build the face reference system from the master side.

  Core::LinAlg::SerialDenseVector norm(nsd_ - 1);

  if (ele.faces()[face]->parent_master_element() != &ele)
  {
    // This is the same that is done before for the face element but we do it from the master side
    // get face position array from element position array
    FOUR_C_ASSERT(faceNodeOrder[face].size() == nfn_, "Internal error");
    const std::vector<int>& trafomap = ele.faces()[face]->get_local_trafo_map();

    // Core::LinAlg::SerialDenseMatrix nodexyzreal_master(nsd_, nfdofs_);
    Core::LinAlg::Matrix<nsd_, nen_> xyzeMasterElement;
    Core::Geo::fill_initial_position_array<distype, nsd_, Core::LinAlg::Matrix<nsd_, nen_>>(
        &ele, xyzeMasterElement);

    // Compute the reference system from the master side
    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int i = 0; i < nsd_ - 1; ++i)
        tangent(d, i) = xyzeMasterElement(d, faceNodeOrder[face][trafomap[i + 1]]) -
                        xyzeMasterElement(d, faceNodeOrder[face][trafomap[0]]);
  }
  else  // We already are on the master side
    for (unsigned int d = 0; d < nsd_; ++d)
      for (unsigned int i = 0; i < nsd_ - 1; ++i) tangent(d, i) = xyze(d, i + 1) - xyze(d, 0);

  // Normalizing the first vector
  for (unsigned int d = 0; d < nsd_; ++d) norm(0) += pow(tangent(d, 0), 2);
  for (unsigned int d = 0; d < nsd_; ++d) tangent(d, 0) /= sqrt(norm(0));

  // In 2D the face reference system is complete.
  // In 3D it is necessary to create an additional vector that is orthogonal to the first.

  // Orthonormalization procedure
  if (nsd_ == 3)
  {
    double tmp = 0;
    for (unsigned int d = 0; d < nsd_; ++d) tmp -= tangent(d, 0) * tangent(d, 1);
    for (unsigned int d = 0; d < nsd_; ++d)
    {
      tangent(d, 1) = tangent(d, 0) * tmp + tangent(d, 1);
      norm(1) += pow(tangent(d, 1), 2);
    }
    // Normalizing the rest of the reference system
    for (unsigned int d = 0; d < nsd_; ++d) tangent(d, 1) /= sqrt(norm(1));
  }

  return;
}


template <Core::FE::CellType distype>
Core::FE::ShapeValuesFaceCache<distype>& Core::FE::ShapeValuesFaceCache<distype>::instance()
{
  static Core::Utils::SingletonOwner<Core::FE::ShapeValuesFaceCache<distype>> owner(
      []()
      {
        return std::unique_ptr<ShapeValuesFaceCache<distype>>(new ShapeValuesFaceCache<distype>);
      });

  return *owner.instance(Core::Utils::SingletonAction::create);
}

template <Core::FE::CellType distype>
std::shared_ptr<Core::FE::ShapeValuesFace<distype>> Core::FE::ShapeValuesFaceCache<distype>::create(
    ShapeValuesFaceParams params)
{
  typename std::map<std::size_t, std::shared_ptr<Core::FE::ShapeValuesFace<distype>>>::iterator i =
      svf_cache_.find(params.to_int());
  if (i != svf_cache_.end())
  {
    return i->second;
  }

  // this is expensive and should not be done too often
  std::shared_ptr<ShapeValuesFace<distype>> svf;
  svf = std::make_shared<ShapeValuesFace<distype>>(params);

  svf_cache_[params.to_int()] = svf;

  return svf;
}



template <Core::FE::CellType distype>
Core::FE::ShapeValuesInteriorOnFaceCache<distype>&
Core::FE::ShapeValuesInteriorOnFaceCache<distype>::instance()
{
  static Core::Utils::SingletonOwner<Core::FE::ShapeValuesInteriorOnFaceCache<distype>> owner(
      []()
      {
        return std::unique_ptr<ShapeValuesInteriorOnFaceCache<distype>>(
            new ShapeValuesInteriorOnFaceCache<distype>);
      });

  return *owner.instance(Core::Utils::SingletonAction::create);
}


template <Core::FE::CellType distype>
std::shared_ptr<Core::FE::ShapeValuesInteriorOnFace>
Core::FE::ShapeValuesInteriorOnFaceCache<distype>::create(ShapeValuesFaceParams params)
{
  typename std::map<std::size_t, std::shared_ptr<ShapeValuesInteriorOnFace>>::iterator i =
      cache_.find(params.to_int());
  if (i != cache_.end())
  {
    return i->second;
  }

  // this is expensive and should not be done too often
  const int nsd = Core::FE::dim<distype>;

  PolynomialSpaceParams polyparams(distype, params.degree_, params.completepoly_);
  std::shared_ptr<PolynomialSpace<nsd>> polySpace =
      Core::FE::PolynomialSpaceCache<nsd>::instance().create(polyparams);
  Core::LinAlg::SerialDenseVector(polySpace->size());
  std::shared_ptr<Core::FE::GaussPoints> quadrature = Core::FE::create_gauss_points(
      Core::FE::get_ele_face_shape_type(distype, params.face_), params.quadraturedegree_);

  std::shared_ptr<ShapeValuesInteriorOnFace> container =
      std::make_shared<ShapeValuesInteriorOnFace>();
  container->shape(polySpace->size(), quadrature->num_points());

  Core::LinAlg::Matrix<nsd, nsd> trafo;
  Core::LinAlg::Matrix<nsd, 1> xsi;
  Core::LinAlg::SerialDenseMatrix faceQPoints;
  Core::FE::boundary_gp_to_parent_gp<nsd>(faceQPoints, trafo, *quadrature, distype,
      Core::FE::get_ele_face_shape_type(distype, params.face_), params.face_);
  Core::LinAlg::SerialDenseVector faceValues(polySpace->size());
  for (int q = 0; q < quadrature->num_points(); ++q)
  {
    for (int d = 0; d < nsd; ++d) xsi(d) = faceQPoints(q, d);
    polySpace->evaluate(xsi, faceValues);
    for (int i = 0; i < faceValues.numRows(); ++i)
    {
      container->matrix_(i, q) = faceValues(i);
      if (std::abs(faceValues(i)) > 1e-14) container->isNonzero_[i] = true;
    }
  }
  cache_[params.to_int()] = container;

  return container;
}


// explicit instantiation of template classes
template class Core::FE::ShapeValues<Core::FE::CellType::hex8>;
template class Core::FE::ShapeValues<Core::FE::CellType::hex20>;
template class Core::FE::ShapeValues<Core::FE::CellType::hex27>;
template class Core::FE::ShapeValues<Core::FE::CellType::tet4>;
template class Core::FE::ShapeValues<Core::FE::CellType::tet10>;
template class Core::FE::ShapeValues<Core::FE::CellType::wedge6>;
template class Core::FE::ShapeValues<Core::FE::CellType::wedge15>;
template class Core::FE::ShapeValues<Core::FE::CellType::pyramid5>;
template class Core::FE::ShapeValues<Core::FE::CellType::quad4>;
template class Core::FE::ShapeValues<Core::FE::CellType::quad8>;
template class Core::FE::ShapeValues<Core::FE::CellType::quad9>;
template class Core::FE::ShapeValues<Core::FE::CellType::tri3>;
template class Core::FE::ShapeValues<Core::FE::CellType::tri6>;
template class Core::FE::ShapeValues<Core::FE::CellType::nurbs9>;
template class Core::FE::ShapeValues<Core::FE::CellType::nurbs27>;

template class Core::FE::ShapeValuesFace<Core::FE::CellType::hex8>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::hex20>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::hex27>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::tet4>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::tet10>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::wedge6>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::wedge15>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::pyramid5>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::quad4>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::quad8>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::quad9>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::tri3>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::tri6>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::nurbs9>;
template class Core::FE::ShapeValuesFace<Core::FE::CellType::nurbs27>;

template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::hex8>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::hex20>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::hex27>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::tet4>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::tet10>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::wedge6>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::wedge15>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::pyramid5>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::quad4>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::quad8>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::quad9>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::tri3>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::tri6>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::nurbs9>;
template class Core::FE::ShapeValuesFaceCache<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE
