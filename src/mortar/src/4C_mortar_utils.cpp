// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mortar_utils.hpp"

#include "4C_comm_exporter.hpp"
#include "4C_fem_nurbs_discretization.hpp"
#include "4C_fem_nurbs_discretization_control_point.hpp"
#include "4C_fem_nurbs_discretization_knotvector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"
#include "4C_linalg_utils_sparse_algebra_math.hpp"
#include "4C_structure_new_timint_base.hpp"
#include "4C_utils_exceptions.hpp"

FOUR_C_NAMESPACE_OPEN

/*!
\brief Sort vector in ascending order

This routine is taken from Trilinos MOERTEL package.

\param dlist (in): vector to be sorted (unsorted on input, sorted on output)
\param N (in):     length of vector to be sorted
\param list2 (in): another vector which is sorted accordingly

*/
void Mortar::sort(double* dlist, int N, int* list2)
{
  int l, r, j, i, flag;
  int RR2;
  double dRR, dK;

  if (N <= 1) return;

  l = N / 2 + 1;
  r = N - 1;
  l = l - 1;
  dRR = dlist[l - 1];
  dK = dlist[l - 1];

  if (list2 != nullptr)
  {
    RR2 = list2[l - 1];
    while (r != 0)
    {
      j = l;
      flag = 1;

      while (flag == 1)
      {
        i = j;
        j = j + j;

        if (j > r + 1)
          flag = 0;
        else
        {
          if (j < r + 1)
            if (dlist[j] > dlist[j - 1]) j = j + 1;

          if (dlist[j - 1] > dK)
          {
            dlist[i - 1] = dlist[j - 1];
            list2[i - 1] = list2[j - 1];
          }
          else
          {
            flag = 0;
          }
        }
      }
      dlist[i - 1] = dRR;
      list2[i - 1] = RR2;

      if (l == 1)
      {
        dRR = dlist[r];
        RR2 = list2[r];
        dK = dlist[r];
        dlist[r] = dlist[0];
        list2[r] = list2[0];
        r = r - 1;
      }
      else
      {
        l = l - 1;
        dRR = dlist[l - 1];
        RR2 = list2[l - 1];
        dK = dlist[l - 1];
      }
    }
    dlist[0] = dRR;
    list2[0] = RR2;
  }
  else
  {
    while (r != 0)
    {
      j = l;
      flag = 1;
      while (flag == 1)
      {
        i = j;
        j = j + j;
        if (j > r + 1)
          flag = 0;
        else
        {
          if (j < r + 1)
            if (dlist[j] > dlist[j - 1]) j = j + 1;
          if (dlist[j - 1] > dK)
          {
            dlist[i - 1] = dlist[j - 1];
          }
          else
          {
            flag = 0;
          }
        }
      }
      dlist[i - 1] = dRR;
      if (l == 1)
      {
        dRR = dlist[r];
        dK = dlist[r];
        dlist[r] = dlist[0];
        r = r - 1;
      }
      else
      {
        l = l - 1;
        dRR = dlist[l - 1];
        dK = dlist[l - 1];
      }
    }
    dlist[0] = dRR;
  }

  return;
}


/*----------------------------------------------------------------------*
 |  Sort points to obtain final clip polygon                  popp 11/08|
 *----------------------------------------------------------------------*/
int Mortar::sort_convex_hull_points(bool out, Core::LinAlg::SerialDenseMatrix& transformed,
    std::vector<Vertex>& collconvexhull, std::vector<Vertex>& respoly, double& tol)
{
  //**********************************************************************
  // - this yields the final clip polygon
  // - sanity of the generated output is checked
  //**********************************************************************

  // (1) Find point with smallest x-value
  // (if more than 1 point with identical x-value exists, choose the one with the smallest y-value)

  // initialize starting point
  int startindex = 0;
  std::array<double, 2> startpoint = {transformed(0, 0), transformed(1, 0)};

  int np = (int)collconvexhull.size();
  for (int i = 1; i < np; ++i)
  {
    if (transformed(0, i) < startpoint[0])
    {
      startpoint[0] = transformed(0, i);
      startpoint[1] = transformed(1, i);
      startindex = i;
    }
    else if (transformed(0, i) == startpoint[0])
    {
      if (transformed(1, i) < startpoint[1])
      {
        startpoint[1] = transformed(1, i);
        startindex = i;
      }
    }
    else
    {
      // do nothing: starting point did not change
    }
  }

  if (out)
    std::cout << "Start of convex hull: Index " << startindex << "\t" << startpoint[0] << "\t"
              << startpoint[1] << std::endl;

  // (2) Sort remaining points ascending w.r.t their angle with the y-axis
  // (if more than 1 point with identical angle exists, sort ascending w.r.t. their y-value)
  std::vector<double> cotangle;
  std::vector<double> yvalues;
  std::vector<int> sorted;
  std::vector<int> onxline;

  for (int i = 0; i < np; ++i)
  {
    // do nothing for starting point
    if (i == startindex) continue;

    // compute angle and store
    double xdiff = transformed(0, i) - startpoint[0];
    double ydiff = transformed(1, i) - startpoint[1];

    if (xdiff < 0) FOUR_C_THROW("Found point with x < x_start for convex hull!");
    if (xdiff >= tol)
    {
      cotangle.push_back(ydiff / xdiff);
      sorted.push_back(i);
    }
    else
    {
      // these points need further investigation
      onxline.push_back(i);
    }
  }

  // check points on x-line with starting point and only add
  // those with min and max value in y-direction
  {
    double y_max = std::numeric_limits<double>::min();
    double y_min = std::numeric_limits<double>::max();
    int i_max = -1;
    int i_min = -1;
    for (size_t i = 0; i < onxline.size(); ++i)
    {
      const double yval = transformed(1, onxline[i]) - startpoint[1];
      if (yval < y_min && yval < 0.0)
      {
        y_min = yval;
        i_min = onxline[i];
      }
      else if (yval > y_max && yval > 0.0)
      {
        y_max = yval;
        i_max = onxline[i];
      }
    }
    if (i_max > -1)
    {
      cotangle.push_back(std::numeric_limits<double>::max());
      sorted.push_back(i_max);
    }
    if (i_min > -1)
    {
      cotangle.push_back(-std::numeric_limits<double>::max());
      sorted.push_back(i_min);
    }
  }

  // start index not yet included
  np = (int)sorted.size() + 1;

  if (out)
  {
    std::cout << "Unsorted convex hull:\n";
    std::cout << "Index " << startindex << "\t" << startpoint[0] << "\t" << startpoint[1]
              << std::endl;
    for (int i = 0; i < np - 1; ++i)
      std::cout << "Index " << sorted[i] << "\t" << transformed(0, sorted[i]) << "\t"
                << transformed(1, sorted[i]) << "\t" << cotangle[i] << std::endl;
  }

  // check if sizes are correct
  if ((int)cotangle.size() != np - 1) FOUR_C_THROW("Size went wrong for cot angle!");

  // now sort descending w.r.t cotangle = ascending w.r.t angle
  Mortar::sort(cotangle.data(), np - 1, sorted.data());
  std::reverse(cotangle.begin(), cotangle.end());
  std::reverse(sorted.begin(), sorted.end());

  // get associated y-values
  for (int i = 0; i < np - 1; ++i) yvalues.push_back(transformed(1, sorted[i]));

  // now sort ascending w.r.t value wherever angles are identical
  // (bubblesort: we might need np-2 rounds if all np-1 angles identical)
  for (int round = 0; round < np - 2; ++round)
    for (int i = 0; i < np - 2; ++i)
      if (cotangle[i] == cotangle[i + 1])
        if (yvalues[i] > yvalues[i + 1])
        {
          std::swap(cotangle[i], cotangle[i + 1]);
          std::swap(yvalues[i], yvalues[i + 1]);
          std::swap(sorted[i], sorted[i + 1]);
        }

  if (out)
  {
    std::cout << "Sorted convex hull:\n";
    std::cout << "Index " << startindex << "\t" << startpoint[0] << "\t" << startpoint[1]
              << std::endl;
    for (int i = 0; i < np - 1; ++i)
      std::cout << "Index " << sorted[i] << "\t" << transformed(0, sorted[i]) << "\t"
                << transformed(1, sorted[i]) << "\t" << cotangle[i] << std::endl;
  }

  // (3) Go through sorted list of points
  // (keep adding points as long as the last 3 points rotate clockwise)
  // (if 3 points rotate counter-clockwise, do NOT add current point and continue)

  // always push pack starting point
  Vertex* current = &collconvexhull[startindex];
  respoly.push_back(Vertex(current->coord(), current->v_type(), current->nodeids(), nullptr,
      nullptr, false, false, nullptr, -1.0));

  // number of points removed from convex hull
  int removed = (int)collconvexhull.size() - np;

  // go through sorted list and check for clockwise rotation
  std::vector<bool> haveremovedthis(np - 1);
  for (int i = 0; i < np - 1; ++i) haveremovedthis[i] = false;

  for (int i = 0; i < np - 1; ++i)
  {
    std::array<double, 2> edge1 = {0.0, 0.0};
    std::array<double, 2> edge2 = {0.0, 0.0};

    // first triple
    if (i == 0)
    {
      edge1[0] = transformed(0, sorted[0]) - startpoint[0];
      edge1[1] = transformed(1, sorted[0]) - startpoint[1];
      edge2[0] = transformed(0, sorted[1]) - transformed(0, sorted[0]);
      edge2[1] = transformed(1, sorted[1]) - transformed(1, sorted[0]);
    }

    // standard case
    else if (i < np - 2)
    {
      // go back and find first non-removed partner
      bool foundpartner = false;
      int k = i - 1;

      while (foundpartner == false)
      {
        // found non-removed partner
        if (haveremovedthis[k] == false)
        {
          edge1[0] = transformed(0, sorted[i]) - transformed(0, sorted[k]);
          edge1[1] = transformed(1, sorted[i]) - transformed(1, sorted[k]);
          edge2[0] = transformed(0, sorted[i + 1]) - transformed(0, sorted[i]);
          edge2[1] = transformed(1, sorted[i + 1]) - transformed(1, sorted[i]);
          foundpartner = true;
        }
        else
        {
          // decrease counter
          k -= 1;

          // use starting point if all in between removed
          if (k < 0)
          {
            edge1[0] = transformed(0, sorted[i]) - startpoint[0];
            edge1[1] = transformed(1, sorted[i]) - startpoint[1];
            edge2[0] = transformed(0, sorted[i + 1]) - transformed(0, sorted[i]);
            edge2[1] = transformed(1, sorted[i + 1]) - transformed(1, sorted[i]);
            foundpartner = true;
          }
        }
      }
    }

    // last triple
    else /* if i = np-1 */
    {
      // go back and find first non-removed partner
      bool foundpartner = false;
      int k = i - 1;

      while (foundpartner == false)
      {
        // found non-removed partner
        if (haveremovedthis[k] == false)
        {
          edge1[0] = transformed(0, sorted[i]) - transformed(0, sorted[k]);
          edge1[1] = transformed(1, sorted[i]) - transformed(1, sorted[k]);
          edge2[0] = startpoint[0] - transformed(0, sorted[i]);
          edge2[1] = startpoint[1] - transformed(1, sorted[i]);
          foundpartner = true;
        }
        else
        {
          // decrease counter
          k -= 1;

          // use starting point if all in between removed
          if (k < 0)
          {
            edge1[0] = transformed(0, sorted[i]) - startpoint[0];
            edge1[1] = transformed(1, sorted[i]) - startpoint[1];
            edge2[0] = startpoint[0] - transformed(0, sorted[i]);
            edge2[1] = startpoint[1] - transformed(1, sorted[i]);
            foundpartner = true;
          }
        }
      }
    }

    // check for clockwise rotation
    double cw = edge1[0] * edge2[1] - edge1[1] * edge2[0];

    // add point to convex hull if clockwise triple
    // (use tolerance to remove almost straight lines of 3 points)
    if (cw <= -tol)
    {
      Vertex* current = &collconvexhull[sorted[i]];
      respoly.push_back(Vertex(current->coord(), current->v_type(), current->nodeids(), nullptr,
          nullptr, false, false, nullptr, -1.0));
    }
    // mark vertex as "removed" if counter-clockwise triple
    else
    {
      removed++;
      haveremovedthis[i] = true;
    }
  }

  return removed;
}

/*----------------------------------------------------------------------*/
/*----------------------------------------------------------------------*/
void Mortar::Utils::create_volume_ghosting(const Core::FE::Discretization& dis_src,
    const std::vector<std::shared_ptr<Core::FE::Discretization>>& voldis,
    std::vector<std::pair<int, int>> material_links, bool check_on_in, bool check_on_exit)
{
  if (voldis.size() == 0) return;

  if (check_on_in)
    for (int c = 1; c < (int)voldis.size(); ++c)
      if (voldis.at(c)->element_row_map()->same_as(*voldis.at(0)->element_row_map()) == false)
        FOUR_C_THROW("row maps on input do not coincide");

  const Core::LinAlg::Map* ielecolmap = dis_src.element_col_map();

  // 1 Ghost all Volume Element + Nodes,for all col elements in dis_src
  for (unsigned disidx = 0; disidx < voldis.size(); ++disidx)
  {
    std::vector<int> rdata;

    // Fill rdata with existing colmap
    const Core::LinAlg::Map* elecolmap = voldis[disidx]->element_col_map();
    const std::shared_ptr<Core::LinAlg::Map> allredelecolmap =
        Core::LinAlg::allreduce_e_map(*voldis[disidx]->element_row_map());

    for (int i = 0; i < elecolmap->num_my_elements(); ++i)
    {
      int gid = elecolmap->gid(i);
      rdata.push_back(gid);
    }

    // Find elements, which are ghosted on the interface but not in the volume discretization
    for (int i = 0; i < ielecolmap->num_my_elements(); ++i)
    {
      int gid = ielecolmap->gid(i);

      Core::Elements::Element* ele = dis_src.g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      Core::Elements::FaceElement* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele);
      if (!faceele) FOUR_C_THROW("source element is not a face element");
      int volgid = faceele->parent_element_id();
      // Ghost the parent element additionally
      if (elecolmap->lid(volgid) == -1 &&
          allredelecolmap->lid(volgid) !=
              -1)  // Volume discretization has not Element on this proc but on another
        rdata.push_back(volgid);
    }

    // re-build element column map
    Core::LinAlg::Map newelecolmap(
        -1, (int)rdata.size(), rdata.data(), 0, voldis[disidx]->get_comm());
    rdata.clear();

    // redistribute the volume discretization according to the
    // new (=old) element column layout & and ghost also nodes!
    voldis[disidx]->extended_ghosting(newelecolmap, true, true, true, false);  // no check!!!
  }

  // 2 Reconnect Face Element -- Parent Element Pointers to first dis in dis_tar
  {
    const Core::LinAlg::Map* elecolmap = voldis[0]->element_col_map();

    for (int i = 0; i < ielecolmap->num_my_elements(); ++i)
    {
      int gid = ielecolmap->gid(i);

      Core::Elements::Element* ele = dis_src.g_element(gid);
      if (!ele) FOUR_C_THROW("Cannot find element with gid %", gid);
      Core::Elements::FaceElement* faceele = dynamic_cast<Core::Elements::FaceElement*>(ele);
      if (!faceele) FOUR_C_THROW("source element is not a face element");
      int volgid = faceele->parent_element_id();

      if (elecolmap->lid(volgid) == -1)  // Volume discretization has not Element
        FOUR_C_THROW("create_volume_ghosting: Element {} does not exist on this Proc!", volgid);

      Core::Elements::Element* vele = voldis[0]->g_element(volgid);
      if (!vele) FOUR_C_THROW("Cannot find element with gid %", volgid);

      faceele->set_parent_master_element(vele, faceele->face_parent_number());

      if (voldis.size() == 2)
      {
        const auto* elecolmap2 = voldis[1]->element_col_map();
        if (elecolmap2->lid(volgid) == -1)
          faceele->set_parent_slave_element(nullptr, -1);
        else
        {
          auto* volele = voldis[1]->g_element(volgid);
          if (volele == nullptr) FOUR_C_THROW("Cannot find element with gid %", volgid);
          faceele->set_parent_slave_element(volele, faceele->face_parent_number());
        }
      }
    }
  }

  if (check_on_exit)
    for (int c = 1; c < (int)voldis.size(); ++c)
    {
      if (voldis.at(c)->element_row_map()->same_as(*voldis.at(0)->element_row_map()) == false)
        FOUR_C_THROW("row maps on exit do not coincide");
      if (voldis.at(c)->element_col_map()->same_as(*voldis.at(0)->element_col_map()) == false)
        FOUR_C_THROW("col maps on exit do not coincide");
    }

  // 3 setup material pointers between newly ghosted elements
  for (std::vector<std::pair<int, int>>::const_iterator m = material_links.begin();
      m != material_links.end(); ++m)
  {
    std::shared_ptr<Core::FE::Discretization> dis_src_mat = voldis.at(m->first);
    std::shared_ptr<Core::FE::Discretization> dis_tar_mat = voldis.at(m->second);

    for (int i = 0; i < dis_tar_mat->num_my_col_elements(); ++i)
    {
      Core::Elements::Element* targetele = dis_tar_mat->l_col_element(i);
      const int gid = targetele->id();

      Core::Elements::Element* sourceele = dis_src_mat->g_element(gid);

      targetele->add_material(sourceele->material());
    }
  }
}



/*----------------------------------------------------------------------*
 |  Prepare mortar element for nurbs-case                    farah 11/14|
 *----------------------------------------------------------------------*/
void Mortar::Utils::prepare_nurbs_element(Core::FE::Discretization& discret,
    std::shared_ptr<Core::Elements::Element> ele, Mortar::Element& cele, int dim)
{
  Core::FE::Nurbs::NurbsDiscretization* nurbsdis =
      dynamic_cast<Core::FE::Nurbs::NurbsDiscretization*>(&(discret));

  std::shared_ptr<Core::FE::Nurbs::Knotvector> knots = (*nurbsdis).get_knot_vector();
  std::vector<Core::LinAlg::SerialDenseVector> parentknots(dim);
  std::vector<Core::LinAlg::SerialDenseVector> mortarknots(dim - 1);

  double normalfac = 0.0;
  std::shared_ptr<Core::Elements::FaceElement> faceele =
      std::dynamic_pointer_cast<Core::Elements::FaceElement>(ele);
  bool zero_size = knots->get_boundary_ele_and_parent_knots(parentknots, mortarknots, normalfac,
      faceele->parent_master_element()->id(), faceele->face_master_number());

  // store nurbs specific data to node
  cele.zero_sized() = zero_size;
  cele.knots() = mortarknots;
  cele.normal_fac() = normalfac;

  return;
}


/*----------------------------------------------------------------------*
 |  Prepare mortar node for nurbs-case                       farah 11/14|
 *----------------------------------------------------------------------*/
void Mortar::Utils::prepare_nurbs_node(Core::Nodes::Node* node, Mortar::Node& mnode)
{
  Core::FE::Nurbs::ControlPoint* cp = dynamic_cast<Core::FE::Nurbs::ControlPoint*>(node);

  mnode.nurbs_w() = cp->w();

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Utils::mortar_matrix_condensation(std::shared_ptr<Core::LinAlg::SparseMatrix>& k,
    const std::shared_ptr<const Core::LinAlg::SparseMatrix>& p_row,
    const std::shared_ptr<const Core::LinAlg::SparseMatrix>& p_col)
{
  // prepare maps by making a deep copy of the map and wrap it in a shared_ptr
  auto gsrow = std::make_shared<Core::LinAlg::Map>(p_row->range_map());
  auto gmrow = std::make_shared<Core::LinAlg::Map>(p_row->domain_map());
  auto gscol = std::make_shared<Core::LinAlg::Map>(p_col->range_map());
  auto gmcol = std::make_shared<Core::LinAlg::Map>(p_col->domain_map());

  std::shared_ptr<Core::LinAlg::Map> gsmrow = Core::LinAlg::merge_map(gsrow, gmrow, false);
  std::shared_ptr<Core::LinAlg::Map> gnrow = Core::LinAlg::split_map(k->range_map(), *gsmrow);

  std::shared_ptr<Core::LinAlg::Map> gsmcol = Core::LinAlg::merge_map(gscol, gmcol, false);
  std::shared_ptr<Core::LinAlg::Map> gncol = Core::LinAlg::split_map(k->domain_map(), *gsmcol);

  /*--------------------------------------------------------------------*/
  /* Split kteff into 3x3 block matrix                                  */
  /*--------------------------------------------------------------------*/
  // we want to split k into 3 groups s,m,n = 9 blocks
  std::shared_ptr<Core::LinAlg::SparseMatrix> kss = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksm = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksn = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kms = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmm = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kmn = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> kns = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> knm = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> knn = nullptr;

  // temporarily we need the blocks ksmsm, ksmn, knsm
  // (FIXME: because a direct SplitMatrix3x3 is still missing!)
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksmsm = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> ksmn = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> knsm = nullptr;

  // some temporary std::shared_ptrs
  std::shared_ptr<Core::LinAlg::Map> tempmap;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx1 = nullptr;
  std::shared_ptr<Core::LinAlg::SparseMatrix> tempmtx2 = nullptr;

  // split
  Core::LinAlg::split_matrix2x2(k, gsmrow, gnrow, gsmcol, gncol, ksmsm, ksmn, knsm, knn);
  Core::LinAlg::split_matrix2x2(ksmsm, gsrow, gmrow, gscol, gmcol, kss, ksm, kms, kmm);
  Core::LinAlg::split_matrix2x2(ksmn, gsrow, gmrow, gncol, tempmap, ksn, tempmtx1, kmn, tempmtx2);
  Core::LinAlg::split_matrix2x2(knsm, gnrow, tempmap, gscol, gmcol, kns, knm, tempmtx1, tempmtx2);

  std::shared_ptr<Core::LinAlg::SparseMatrix> kteffnew =
      std::make_shared<Core::LinAlg::SparseMatrix>(
          k->row_map(), 81, true, false, k->get_matrixtype());

  // build new stiffness matrix
  Core::LinAlg::matrix_add(*knn, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*knm, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*kmn, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(*kmm, false, 1.0, *kteffnew, 1.0);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*kns, false, *p_col, false, true, false, true), false, 1.,
      *kteffnew, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*p_row, true, *ksn, false, true, false, true), false, 1.,
      *kteffnew, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*kms, false, *p_col, false, true, false, true), false, 1.,
      *kteffnew, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*p_row, true, *ksm, false, true, false, true), false, 1.,
      *kteffnew, 1.);
  Core::LinAlg::matrix_add(
      *Core::LinAlg::matrix_multiply(*p_row, true,
          *Core::LinAlg::matrix_multiply(*kss, false, *p_col, false, true, false, true), false,
          true, false, true),
      false, 1., *kteffnew, 1.);

  if (p_row == p_col)
    Core::LinAlg::matrix_add(
        *Core::LinAlg::create_identity_matrix(*gsrow), false, 1., *kteffnew, 1.);

  kteffnew->complete(k->domain_map(), k->range_map());

  // return new matrix
  k = kteffnew;

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Utils::mortar_rhs_condensation(
    Core::LinAlg::Vector<double>& rhs, Core::LinAlg::SparseMatrix& p)
{
  // prepare maps
  std::shared_ptr<Core::LinAlg::Map> gsdofrowmap = std::const_pointer_cast<Core::LinAlg::Map>(
      Core::Utils::shared_ptr_from_ref<const Core::LinAlg::Map>(p.range_map()));
  std::shared_ptr<Core::LinAlg::Map> gmdofrowmap = std::const_pointer_cast<Core::LinAlg::Map>(
      Core::Utils::shared_ptr_from_ref<const Core::LinAlg::Map>(p.domain_map()));

  Core::LinAlg::Vector<double> fs(*gsdofrowmap);
  Core::LinAlg::Vector<double> fm_cond(*gmdofrowmap);
  Core::LinAlg::export_to(rhs, fs);
  Core::LinAlg::Vector<double> fs_full(rhs.get_map());
  Core::LinAlg::export_to(fs, fs_full);
  rhs.update(-1., fs_full, 1.);

  p.multiply(true, fs, fm_cond);

  Core::LinAlg::Vector<double> fm_cond_full(rhs.get_map());
  Core::LinAlg::export_to(fm_cond, fm_cond_full);
  rhs.update(1., fm_cond_full, 1.);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Utils::mortar_recover(Core::LinAlg::Vector<double>& inc, Core::LinAlg::SparseMatrix& p)
{
  // prepare maps
  std::shared_ptr<Core::LinAlg::Map> gsdofrowmap = std::const_pointer_cast<Core::LinAlg::Map>(
      Core::Utils::shared_ptr_from_ref<const Core::LinAlg::Map>(p.range_map()));
  std::shared_ptr<Core::LinAlg::Map> gmdofrowmap = std::const_pointer_cast<Core::LinAlg::Map>(
      Core::Utils::shared_ptr_from_ref<const Core::LinAlg::Map>(p.domain_map()));

  Core::LinAlg::Vector<double> m_inc(*gmdofrowmap);
  Core::LinAlg::export_to(inc, m_inc);

  Core::LinAlg::Vector<double> s_inc(*gsdofrowmap);
  p.multiply(false, m_inc, s_inc);
  Core::LinAlg::Vector<double> s_inc_full(inc.get_map());
  Core::LinAlg::export_to(s_inc, s_inc_full);
  inc.update(1., s_inc_full, 1.);

  return;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Utils::mortar_matrix_condensation(
    std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase>& k,
    const std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>>& p)
{
  std::shared_ptr<Core::LinAlg::BlockSparseMatrixBase> cond_mat =
      std::make_shared<Core::LinAlg::BlockSparseMatrix<Core::LinAlg::DefaultBlockMatrixStrategy>>(
          k->domain_extractor(), k->range_extractor(), 81, false, true);

  for (int row = 0; row < k->rows(); ++row)
    for (int col = 0; col < k->cols(); ++col)
    {
      std::shared_ptr<Core::LinAlg::SparseMatrix> new_matrix =
          std::make_shared<Core::LinAlg::SparseMatrix>(k->matrix(row, col));
      mortar_matrix_condensation(new_matrix, p.at(row), p.at(col) /*,row!=col*/);
      cond_mat->assign(row, col, Core::LinAlg::DataAccess::Copy, *new_matrix);
    }

  cond_mat->complete();

  k = cond_mat;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Utils::mortar_rhs_condensation(Core::LinAlg::Vector<double>& rhs,
    const std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>>& p)
{
  for (unsigned i = 0; i < p.size(); mortar_rhs_condensation(rhs, *p[i++]));
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
void Mortar::Utils::mortar_recover(Core::LinAlg::Vector<double>& inc,
    const std::vector<std::shared_ptr<Core::LinAlg::SparseMatrix>>& p)
{
  for (unsigned i = 0; i < p.size(); mortar_recover(inc, *p[i++]));
}

FOUR_C_NAMESPACE_CLOSE
