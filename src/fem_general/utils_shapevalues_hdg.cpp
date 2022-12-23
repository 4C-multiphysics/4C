/*----------------------------------------------------------------------*/
/*! \file
\brief Evaluation of scalar shape functions for HDG

\level 2

*----------------------------------------------------------------------*/

#include "utils_shapevalues_hdg.H"
#include "utils_boundary_integration.H"
#include "position_array.H"
#include "fluid_ele.H"
#include "singleton_owner.H"


/*----------------------------------------------------------------------*
 |  Constructor (public)                              kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::UTILS::ShapeValues<distype>::ShapeValues(
    const unsigned int degree, const bool completepoly, const unsigned int quadratureDegree)
    : degree_(degree),
      quadrature_(DRT::UTILS::GaussPointCache::Instance().Create(distype, quadratureDegree)),
      usescompletepoly_(completepoly),
      nqpoints_(quadrature_->NumPoints())
{
  PolynomialSpaceParams params(distype, degree, completepoly);
  polySpace_ = DRT::UTILS::PolynomialSpaceCache<nsd_>::Instance().Create(params);
  ndofs_ = polySpace_->Size();

  Epetra_SerialDenseVector values(ndofs_);
  Epetra_SerialDenseMatrix derivs(nsd_, ndofs_);

  xyzreal.Shape(nsd_, nqpoints_);

  funct.Shape(nen_, nqpoints_);
  derxy.Shape(nen_ * nsd_, nqpoints_);

  shfunct.Shape(ndofs_, nqpoints_);
  shfunctAvg.Resize(ndofs_);
  shderiv.Shape(ndofs_ * nsd_, nqpoints_);
  shderxy.Shape(ndofs_ * nsd_, nqpoints_);
  jfac.Resize(nqpoints_);

  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    // gauss point in real coordinates
    const double* gpcoord = quadrature_->Point(q);
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = gpcoord[idim];

    polySpace_->Evaluate(xsi, values);
    polySpace_->Evaluate_deriv1(xsi, derivs);

    for (unsigned int i = 0; i < ndofs_; ++i)
    {
      shfunct(i, q) = values(i);
      for (unsigned int d = 0; d < nsd_; ++d) shderiv(i * nsd_ + d, q) = derivs(d, i);
    }

    LINALG::Matrix<nen_, 1> myfunct(funct.A() + q * nen_, true);
    DRT::UTILS::shape_function<distype>(xsi, myfunct);
  }

  // Fill support points
  nodexyzreal.Shape(nsd_, ndofs_);
  polySpace_->FillUnitNodePoints(nodexyzunit);
  dsassert(nodexyzreal.M() == nodexyzunit.M() && nodexyzreal.N() == nodexyzunit.N(),
      "Dimension mismatch");
}



/*----------------------------------------------------------------------*
 |  Evaluate element-dependent shape data (public)    kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::UTILS::ShapeValues<distype>::Evaluate(
    const DRT::Element& ele, const std::vector<double>& aleDis)
{
  dsassert(ele.Shape() == distype, "Internal error");
  GEO::fillInitialPositionArray<distype, nsd_, LINALG::Matrix<nsd_, nen_>>(&ele, xyze);

  // update nodal coordinates
  if (!(aleDis.empty()))
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) xyze(d, n) += aleDis[n * nsd_ + d];

  for (unsigned int i = 0; i < ndofs_; ++i) shfunctAvg(i) = 0.;
  double faceVol = 0.;

  // evaluate geometry
  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    const double* gpcoord = quadrature_->Point(q);
    for (unsigned int idim = 0; idim < nsd_; idim++) xsi(idim) = gpcoord[idim];

    DRT::UTILS::shape_function_deriv1<distype>(xsi, deriv);
    xjm.MultiplyNT(deriv, xyze);
    jfac(q) = xji.Invert(xjm) * quadrature_->Weight(q);

    LINALG::Matrix<nen_, 1> myfunct(funct.A() + q * nen_, true);
    LINALG::Matrix<nsd_, 1> mypoint(xyzreal.A() + q * nsd_, true);
    mypoint.MultiplyNN(xyze, myfunct);

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

    LINALG::Matrix<nen_, 1> myfunct;
    // DRT::UTILS::shape_function<DRT::UTILS::DisTypeToFaceShapeType<distype>::shape>(xsi,myfunct);
    DRT::UTILS::shape_function<distype>(xsi, myfunct);
    LINALG::Matrix<nsd_, 1> mypoint(nodexyzreal.A() + i * nsd_, true);
    mypoint.MultiplyNN(xyze, myfunct);
  }
}

/*----------------------------------------------------------------------*
 |  Constructor (public)                                 schoeder 06/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
DRT::UTILS::ShapeValuesFace<distype>::ShapeValuesFace(ShapeValuesFaceParams params)
    : params_(params), degree_(params.degree_)
{
  if (nsd_ == 2)
    faceNodeOrder = DRT::UTILS::getEleNodeNumberingLines(distype);
  else if (nsd_ == 3)
    faceNodeOrder = DRT::UTILS::getEleNodeNumberingSurfaces(distype);
  else
    dserror("Not implemented for dim != 2, 3");

  PolynomialSpaceParams polyparams(
      DRT::UTILS::DisTypeToFaceShapeType<distype>::shape, degree_, params.completepoly_);
  polySpace_ = DRT::UTILS::PolynomialSpaceCache<nsd_ - 1>::Instance().Create(polyparams);

  nfdofs_ = polySpace_->Size();
  quadrature_ = DRT::UTILS::GaussPointCache::Instance().Create(
      DRT::UTILS::DisTypeToFaceShapeType<distype>::shape, params.quadraturedegree_);
  nqpoints_ = quadrature_->NumPoints();

  faceValues.LightSize(nfdofs_);
  xyzreal.LightShape(nsd_, nqpoints_);
  funct.LightShape(nfn_, nqpoints_);

  shfunctNoPermute.LightShape(nfdofs_, nqpoints_);
  shfunct.LightShape(nfdofs_, nqpoints_);
  normals.LightShape(nsd_, nqpoints_);
  jfac.LightResize(nqpoints_);

  shfunctI.Shape(nfdofs_, nqpoints_);

  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    const double* gpcoord = quadrature_->Point(q);

    const unsigned int codim = nsd_ - 1;
    for (unsigned int idim = 0; idim < codim; idim++) xsi(idim) = gpcoord[idim];

    polySpace_->Evaluate(xsi, faceValues);
    for (unsigned int i = 0; i < nfdofs_; ++i) shfunctNoPermute(i, q) = faceValues(i);

    LINALG::Matrix<nfn_, 1> myfunct(funct.A() + q * nfn_, true);
    DRT::UTILS::shape_function<DRT::UTILS::DisTypeToFaceShapeType<distype>::shape>(xsi, myfunct);
  }
}

/*----------------------------------------------------------------------*
 |  Evaluate face-dependent shape data (public)       kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::UTILS::ShapeValuesFace<distype>::EvaluateFace(
    const DRT::Element& ele, const unsigned int face, const std::vector<double>& aleDis)
{
  const DRT::Element::DiscretizationType facedis =
      DRT::UTILS::DisTypeToFaceShapeType<distype>::shape;

  // get face position array from element position array
  dsassert(faceNodeOrder[face].size() == nfn_, "Internal error");

  LINALG::Matrix<nsd_, nen_> xyzeElement;
  GEO::fillInitialPositionArray<distype, nsd_, LINALG::Matrix<nsd_, nen_>>(&ele, xyzeElement);

  // update nodal coordinates
  if (!(aleDis.empty()))
    for (unsigned int n = 0; n < nen_; ++n)
      for (unsigned int d = 0; d < nsd_; ++d) xyzeElement(d, n) += aleDis[n * nsd_ + d];

  for (unsigned int i = 0; i < nfn_; ++i)
    for (unsigned int d = 0; d < nsd_; ++d) xyze(d, i) = xyzeElement(d, faceNodeOrder[face][i]);

  // Fill face support points
  nodexyzreal.LightShape(nsd_, nfdofs_);
  polySpace_->FillUnitNodePoints(nodexyzunit);
  dsassert(nodexyzreal.M() == nodexyzunit.M() + 1 && nodexyzreal.N() == nodexyzunit.N(),
      "Dimension mismatch");
  for (unsigned int i = 0; i < nfdofs_; ++i)
  {
    for (unsigned int idim = 0; idim < nsd_ - 1; idim++) xsi(idim) = nodexyzunit(idim, i);

    LINALG::Matrix<nfn_, 1> myfunct;
    DRT::UTILS::shape_function<DRT::UTILS::DisTypeToFaceShapeType<distype>::shape>(xsi, myfunct);
    LINALG::Matrix<nsd_, 1> mypoint(nodexyzreal.A() + i * nsd_, true);
    mypoint.MultiplyNN(xyze, myfunct);
  }

  // evaluate geometry
  for (unsigned int q = 0; q < nqpoints_; ++q)
  {
    const double* gpcoord = quadrature_->Point(q);
    for (unsigned int idim = 0; idim < nsd_ - 1; idim++) xsi(idim) = gpcoord[idim];

    DRT::UTILS::shape_function_deriv1<facedis>(xsi, deriv);
    double jacdet = 0.0;
    DRT::UTILS::ComputeMetricTensorForBoundaryEle<facedis>(
        xyze, deriv, metricTensor, jacdet, &normal);
    for (unsigned int d = 0; d < nsd_; ++d) normals(d, q) = normal(d);
    jfac(q) = jacdet * quadrature_->Weight(q);

    LINALG::Matrix<nfn_, 1> myfunct(funct.A() + q * nfn_, true);
    LINALG::Matrix<nsd_, 1> mypoint(xyzreal.A() + q * nsd_, true);
    mypoint.MultiplyNN(xyze, myfunct);
  }

  AdjustFaceOrientation(ele, face);
  ComputeFaceReferenceSystem(ele, face);

  DRT::UTILS::ShapeValuesFaceParams interiorparams = params_;
  interiorparams.degree_ = ele.Degree();
  interiorparams.face_ = face;
  shfunctI = *(ShapeValuesInteriorOnFaceCache<distype>::Instance().Create(interiorparams));
}



/*----------------------------------------------------------------------*
 |  Reorder evaluated face shape functions (private)  kronbichler 05/14 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::UTILS::ShapeValuesFace<distype>::AdjustFaceOrientation(
    const DRT::Element& ele, const unsigned int face)
{
  // For the shape values on faces, we need to figure out how the master element of
  // a face walks over the face and how the current face element wants to walk over
  // it. The local trafo map of the face element holds that information for the slave
  // side, whereas the master side always uses the correct mapping. Thus, we need not
  // do anything in that case.
  if (ele.Faces()[face]->ParentMasterElement() == &ele)
  {
    for (unsigned int q = 0; q < nqpoints_; ++q)
      for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, q);
    return;
  }

  // For the adjusted slave slide, we need to change the order of quadrature
  // points in the shape functions of the trace to match the orientation in the
  // transformation.
  const std::vector<int>& trafomap = ele.Faces()[face]->GetLocalTrafoMap();
  dsassert(trafomap.size() == nfn_,
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
          dsassert(trafomap[1] == 0 && trafomap[0] == 1, "Unknown face orientation in 2D");
          for (unsigned int q = 0; q < nqpoints_; ++q)
          {
            for (unsigned int i = 0; i < nfdofs_; ++i)
              shfunct(i, q) = shfunctNoPermute(i, nqpoints_ - 1 - q);
          }
        }
        break;
      case 3:
        if (distype == DRT::Element::hex8 || distype == DRT::Element::hex20 ||
            distype == DRT::Element::hex27 || distype == DRT::Element::nurbs8 ||
            distype == DRT::Element::nurbs27)
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
            std::cout << "element id: " << ele.Id() << " " << ele.Neighbor(face)->Id() << std::endl;
            const DRT::ELEMENTS::FluidIntFace* faceel =
                dynamic_cast<const DRT::ELEMENTS::FluidIntFace*>(ele.Faces()[face].getRawPtr());
            if (faceel != NULL)
            {
              std::vector<int> trafo =
                  const_cast<DRT::ELEMENTS::FluidIntFace*>(faceel)->GetLocalTrafoMap();
              for (unsigned int i = 0; i < 4; ++i) std::cout << trafo[i] << " ";
            }
            for (unsigned int i = 0; i < 4; ++i) std::cout << trafomap[i] << " ";
            std::cout << std::endl << std::flush;
            dserror("Unknown HEX face orientation in 3D");
          }
        }
        else if (distype == DRT::Element::tet4 || distype == DRT::Element::tet10)
        {
          if (trafomap[0] == 1 && trafomap[1] == 0 && trafomap[2] == 2)
          {
            // zeroth and first node permuted; the search is a bit stupid right now
            // (one could cache the permutation coefficients r of the quadrature formula
            // but it does not seem worth it)
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              double point[2] = {quadrature_->Point(q)[0], quadrature_->Point(q)[1]};
              point[0] = 1. - point[1] - point[0];
              unsigned int r = 0;
              for (; r < nqpoints_; ++r)
                if (std::abs(quadrature_->Point(r)[0] - point[0]) < 1e-14 &&
                    std::abs(quadrature_->Point(r)[1] - point[1]) < 1e-14)
                  break;
              if (r < nqpoints_)
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, r);
              else
              {
                xsi(0) = point[0];
                xsi(1) = point[1];
                polySpace_->Evaluate(xsi, faceValues);
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = faceValues(i);
              }
            }
          }
          else if (trafomap[0] == 2 && trafomap[1] == 1 && trafomap[2] == 0)
          {
            // zeroth and second node permuted
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              double point[2] = {quadrature_->Point(q)[0], quadrature_->Point(q)[1]};
              point[1] = 1. - point[1] - point[0];
              unsigned int r = 0;
              for (; r < nqpoints_; ++r)
                if (std::abs(quadrature_->Point(r)[0] - point[0]) < 1e-14 &&
                    std::abs(quadrature_->Point(r)[1] - point[1]) < 1e-14)
                  break;
              if (r < nqpoints_)
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, r);
              else
              {
                xsi(0) = point[0];
                xsi(1) = point[1];
                polySpace_->Evaluate(xsi, faceValues);
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = faceValues(i);
              }
            }
          }
          else if (trafomap[0] == 0 && trafomap[1] == 2 && trafomap[2] == 1)
          {
            // first and second node permuted
            for (unsigned int q = 0; q < nqpoints_; ++q)
            {
              double point[2] = {quadrature_->Point(q)[1], quadrature_->Point(q)[0]};
              unsigned int r = 0;
              for (; r < nqpoints_; ++r)
                if (std::abs(quadrature_->Point(r)[0] - point[0]) < 1e-14 &&
                    std::abs(quadrature_->Point(r)[1] - point[1]) < 1e-14)
                  break;
              if (r < nqpoints_)
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = shfunctNoPermute(i, r);
              else
              {
                xsi(0) = point[0];
                xsi(1) = point[1];
                polySpace_->Evaluate(xsi, faceValues);
                for (unsigned int i = 0; i < nfdofs_; ++i) shfunct(i, q) = faceValues(i);
              }
            }
          }
          else
          {
            for (unsigned int i = 0; i < 3; ++i) std::cout << trafomap[i] << " ";
            std::cout << std::endl << std::flush;
            // need to transform quadrature point coordinate 0 into 1-p[0]-p[1]
            dserror("Unknown TET face orientation in 3D");
          }
        }
        else
          dserror("Shape type %s not yet implemented", (DRT::DistypeToString(distype)).c_str());
        break;
      default:
        dserror("Only implemented in 2D and 3D");
        break;
    }
}



/*----------------------------------------------------------------------*
 |  Compute the face reference system       (private)  berardocco 09/18 |
 *----------------------------------------------------------------------*/
template <DRT::Element::DiscretizationType distype>
void DRT::UTILS::ShapeValuesFace<distype>::ComputeFaceReferenceSystem(
    const DRT::Element& ele, const unsigned int face)
{
  // In the case in which the element is not the master element for the face there is the need to
  // find the master element and build the face reference system from the master side.

  Epetra_SerialDenseVector norm(nsd_ - 1);

  if (ele.Faces()[face]->ParentMasterElement() != &ele)
  {
    // This is the same that is done before for the face element but we do it from the master side
    // get face position array from element position array
    dsassert(faceNodeOrder[face].size() == nfn_, "Internal error");
    const std::vector<int>& trafomap = ele.Faces()[face]->GetLocalTrafoMap();

    // LINALG::SerialDenseMatrix nodexyzreal_master(nsd_, nfdofs_);
    LINALG::Matrix<nsd_, nen_> xyzeMasterElement;
    GEO::fillInitialPositionArray<distype, nsd_, LINALG::Matrix<nsd_, nen_>>(
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


template <DRT::Element::DiscretizationType distype>
DRT::UTILS::ShapeValuesFaceCache<distype>& DRT::UTILS::ShapeValuesFaceCache<distype>::Instance()
{
  static ::UTILS::SingletonOwner<DRT::UTILS::ShapeValuesFaceCache<distype>> owner(
      []() {
        return std::unique_ptr<ShapeValuesFaceCache<distype>>(new ShapeValuesFaceCache<distype>);
      });

  return *owner.Instance(::UTILS::SingletonAction::create);
}

template <DRT::Element::DiscretizationType distype>
Teuchos::RCP<DRT::UTILS::ShapeValuesFace<distype>>
DRT::UTILS::ShapeValuesFaceCache<distype>::Create(ShapeValuesFaceParams params)
{
  typename std::map<std::size_t, Teuchos::RCP<DRT::UTILS::ShapeValuesFace<distype>>>::iterator i =
      svf_cache_.find(params.ToInt());
  if (i != svf_cache_.end())
  {
    return i->second;
  }

  // this is expensive and should not be done too often
  Teuchos::RCP<ShapeValuesFace<distype>> svf;
  svf = Teuchos::rcp(new ShapeValuesFace<distype>(params));

  svf_cache_[params.ToInt()] = svf;

  return svf;
}



template <DRT::Element::DiscretizationType distype>
DRT::UTILS::ShapeValuesInteriorOnFaceCache<distype>&
DRT::UTILS::ShapeValuesInteriorOnFaceCache<distype>::Instance()
{
  static ::UTILS::SingletonOwner<DRT::UTILS::ShapeValuesInteriorOnFaceCache<distype>> owner(
      []()
      {
        return std::unique_ptr<ShapeValuesInteriorOnFaceCache<distype>>(
            new ShapeValuesInteriorOnFaceCache<distype>);
      });

  return *owner.Instance(::UTILS::SingletonAction::create);
}


template <DRT::Element::DiscretizationType distype>
Teuchos::RCP<DRT::UTILS::ShapeValuesInteriorOnFace>
DRT::UTILS::ShapeValuesInteriorOnFaceCache<distype>::Create(ShapeValuesFaceParams params)
{
  typename std::map<std::size_t, Teuchos::RCP<ShapeValuesInteriorOnFace>>::iterator i =
      cache_.find(params.ToInt());
  if (i != cache_.end())
  {
    return i->second;
  }

  // this is expensive and should not be done too often
  const int nsd = DisTypeToDim<distype>::dim;

  PolynomialSpaceParams polyparams(distype, params.degree_, params.completepoly_);
  Teuchos::RCP<PolynomialSpace<nsd>> polySpace =
      DRT::UTILS::PolynomialSpaceCache<nsd>::Instance().Create(polyparams);
  LINALG::SerialDenseVector(polySpace->Size());
  Teuchos::RCP<DRT::UTILS::GaussPoints> quadrature = DRT::UTILS::GaussPointCache::Instance().Create(
      getEleFaceShapeType(distype, params.face_), params.quadraturedegree_);

  Teuchos::RCP<ShapeValuesInteriorOnFace> container = Teuchos::rcp(new ShapeValuesInteriorOnFace());
  container->Shape(polySpace->Size(), quadrature->NumPoints());

  LINALG::Matrix<nsd, nsd> trafo;
  LINALG::Matrix<nsd, 1> xsi;
  LINALG::SerialDenseMatrix faceQPoints;
  DRT::UTILS::BoundaryGPToParentGP<nsd>(faceQPoints, trafo, *quadrature, distype,
      getEleFaceShapeType(distype, params.face_), params.face_);
  LINALG::SerialDenseVector faceValues(polySpace->Size());
  for (int q = 0; q < quadrature->NumPoints(); ++q)
  {
    for (int d = 0; d < nsd; ++d) xsi(d) = faceQPoints(q, d);
    polySpace->Evaluate(xsi, faceValues);
    for (int i = 0; i < faceValues.M(); ++i)
    {
      container->matrix_(i, q) = faceValues(i);
      if (std::abs(faceValues(i)) > 1e-14) container->isNonzero_[i] = true;
    }
  }
  cache_[params.ToInt()] = container;

  return container;
}


// explicit instantiation of template classes
template class DRT::UTILS::ShapeValues<DRT::Element::hex8>;
template class DRT::UTILS::ShapeValues<DRT::Element::hex20>;
template class DRT::UTILS::ShapeValues<DRT::Element::hex27>;
template class DRT::UTILS::ShapeValues<DRT::Element::tet4>;
template class DRT::UTILS::ShapeValues<DRT::Element::tet10>;
template class DRT::UTILS::ShapeValues<DRT::Element::wedge6>;
template class DRT::UTILS::ShapeValues<DRT::Element::wedge15>;
template class DRT::UTILS::ShapeValues<DRT::Element::pyramid5>;
template class DRT::UTILS::ShapeValues<DRT::Element::quad4>;
template class DRT::UTILS::ShapeValues<DRT::Element::quad8>;
template class DRT::UTILS::ShapeValues<DRT::Element::quad9>;
template class DRT::UTILS::ShapeValues<DRT::Element::tri3>;
template class DRT::UTILS::ShapeValues<DRT::Element::tri6>;
template class DRT::UTILS::ShapeValues<DRT::Element::nurbs9>;
template class DRT::UTILS::ShapeValues<DRT::Element::nurbs27>;

template class DRT::UTILS::ShapeValuesFace<DRT::Element::hex8>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::hex20>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::hex27>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::tet4>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::tet10>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::wedge6>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::wedge15>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::pyramid5>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::quad4>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::quad8>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::quad9>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::tri3>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::tri6>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::nurbs9>;
template class DRT::UTILS::ShapeValuesFace<DRT::Element::nurbs27>;

template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::hex8>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::hex20>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::hex27>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::tet4>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::tet10>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::wedge6>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::wedge15>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::pyramid5>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::quad4>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::quad8>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::quad9>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::tri3>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::tri6>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::nurbs9>;
template class DRT::UTILS::ShapeValuesFaceCache<DRT::Element::nurbs27>;
