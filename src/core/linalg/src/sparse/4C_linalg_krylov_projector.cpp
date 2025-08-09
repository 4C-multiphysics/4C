// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_linalg_krylov_projector.hpp"

#include "4C_linalg_map.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_linalg_transfer.hpp"
#include "4C_linalg_utils_densematrix_communication.hpp"
#include "4C_linalg_utils_densematrix_multiply.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_vector.hpp"
#include "4C_utils_exceptions.hpp"

#include <Teuchos_SerialDenseSolver.hpp>

FOUR_C_NAMESPACE_OPEN

/* ====================================================================
    public
   ==================================================================== */

/* --------------------------------------------------------------------
                          Constructor
   -------------------------------------------------------------------- */
Core::LinAlg::KrylovProjector::KrylovProjector(
    const std::vector<int> modeids, const std::string* weighttype, const Core::LinAlg::Map* map)
    : complete_(false), modeids_(modeids), weighttype_(weighttype), p_(nullptr), pt_(nullptr)
{
  nsdim_ = modeids_.size();
  c_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*map, nsdim_, false);
  if (*weighttype_ == "integration")
    w_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*map, nsdim_, false);
  else if (*weighttype_ == "pointvalues")
    w_ = c_;
  else
    FOUR_C_THROW("No permissible weight type.");

  invw_tc_ = std::make_shared<Core::LinAlg::SerialDenseMatrix>(nsdim_, nsdim_);
}  // Core::LinAlg::KrylovProjector::KrylovProjector


/* --------------------------------------------------------------------
                  Give out std::shared_ptr to c_ for change
   -------------------------------------------------------------------- */
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::LinAlg::KrylovProjector::get_non_const_kernel()
{
  // since c_ will be changed, need to call fill_complete() to recompute invwTc_
  complete_ = false;

  // projector matrices will change
  p_ = nullptr;
  pt_ = nullptr;

  return c_;
}

/* --------------------------------------------------------------------
                  Give out std::shared_ptr to w_ for change
   -------------------------------------------------------------------- */
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::LinAlg::KrylovProjector::get_non_const_weights()
{
  if ((*weighttype_) == "pointvalues")
    FOUR_C_THROW(
        "For weight type 'pointvalues' weight vector equals kernel vector and can thus only be "
        "changed implicitly by changing the kernel.");

  // since w_ will be changed, need to call fill_complete() to recompute invwTc_
  complete_ = false;

  // projector matrices will change
  p_ = nullptr;
  pt_ = nullptr;

  return w_;
}

void Core::LinAlg::KrylovProjector::set_cw(Core::LinAlg::MultiVector<double>& c0,
    Core::LinAlg::MultiVector<double>& w0, const Core::LinAlg::Map* newmap)
{
  c_ = nullptr;
  w_ = nullptr;

  c_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*newmap, nsdim_, false);
  w_ = std::make_shared<Core::LinAlg::MultiVector<double>>(*newmap, nsdim_, false);
  *c_ = c0;
  *w_ = w0;
  return;
}

void Core::LinAlg::KrylovProjector::set_cw(
    Core::LinAlg::MultiVector<double>& c0, Core::LinAlg::MultiVector<double>& w0)
{
  *c_ = c0;
  *w_ = w0;
  return;
}

/* --------------------------------------------------------------------
            Compute (w_^T c_)^(-1) and set complete flag
   -------------------------------------------------------------------- */
void Core::LinAlg::KrylovProjector::fill_complete()
{
  if (c_ == nullptr)
  {
    FOUR_C_THROW("No kernel vector supplied for projection");
  }

  if (w_ == nullptr)
  {
    FOUR_C_THROW("No weight vector supplied for projection");
  }

  if (c_->NumVectors() != nsdim_)
  {
    FOUR_C_THROW("Number of kernel vectors has been changed.");
  }

  if (w_->NumVectors() != nsdim_)
  {
    FOUR_C_THROW("Number of weight vectors has been changed.");
  }

  // projector matrices will change
  p_ = nullptr;
  pt_ = nullptr;

  // loop all kernel basis vectors
  for (int mm = 0; mm < nsdim_; ++mm)
  {
    // loop all weight vectors
    for (int rr = 0; rr < nsdim_; ++rr)
    {
      /*
        Compute dot product of all different combinations of c_ and w_ and
        put result in dense matrix. In case that all <w_i,c_j>=0 for all
        i!=j, wTc_ is diagonal.

               T
              w * c
       */
      double wTc;
      (*w_)(mm).dot((*c_)(rr), &wTc);

      // make sure c_i and w_i must not be krylov.
      if ((rr == mm) and (abs(wTc) < 1e-14))
      {
        // not sure whether c_i and w_i must not be krylov.
        // delete dserror in case you are sure what you are doing!
        FOUR_C_THROW("weight vector w_{} must not be orthogonal to c_{}", rr, mm);
      }
      // fill matrix (w_^T * c_) - not yet inverted!
      (*invw_tc_)(mm, rr) = wTc;
    }
  }

  // invert wTc-matrix (also done if it's only a scalar - check with Micheal
  // Gee before changing this)
  using ordinalType = Core::LinAlg::SerialDenseMatrix::ordinalType;
  using scalarType = Core::LinAlg::SerialDenseMatrix::scalarType;
  Teuchos::SerialDenseSolver<ordinalType, scalarType> densesolver;
  densesolver.setMatrix(Teuchos::rcpFromRef(*invw_tc_));
  int err = densesolver.invert();
  if (err)
    FOUR_C_THROW(
        "Error inverting dot-product matrix of kernels and weights for orthogonal (\"krylov\") "
        "projection.");

  complete_ = true;

  return;
}
// Core::LinAlg::KrylovProjector::fill_complete

/* --------------------------------------------------------------------
                    Create projector P(^T) (for direct solvers)
   -------------------------------------------------------------------- */
Core::LinAlg::SparseMatrix Core::LinAlg::KrylovProjector::get_p()
{
  /*
   *               / T   \ -1   T
   * P = I - c_ * | w_ c_ |  * w_
   *               \     /
   *             `----v-----'
   *                invwTc_
   */
  if (!complete_)
    FOUR_C_THROW(
        "Krylov space projector is not complete. Call fill_complete() after changing c_ or w_.");

  if (p_ == nullptr)
  {
    create_projector(p_, w_, c_, invw_tc_);
  }

  return *p_;
}

Core::LinAlg::SparseMatrix Core::LinAlg::KrylovProjector::get_pt()
{
  /*
   *  T             / T   \ -1   T
   * P  = x - w_ * | c_ w_ |  * c_
   *
   *                \     /
   *              `----v-----'
   *               (invwTc_)^T
   */
  if (!complete_)
    FOUR_C_THROW(
        "Krylov space projector is not complete. Call fill_complete() after changing c_ or w_.");

  if (pt_ == nullptr)
  {
    if ((*weighttype_) == "pointvalues")
    {
      if (p_ == nullptr)
        FOUR_C_THROW("When using type pointvalues, first get P_ than PT_. Don't ask - do it!");
      pt_ = p_;
    }
    else
    {
      std::shared_ptr<Core::LinAlg::SerialDenseMatrix> invwTcT =
          std::make_shared<Core::LinAlg::SerialDenseMatrix>(*invw_tc_, Teuchos::TRANS);
      create_projector(pt_, c_, w_, invwTcT);
    }
  }

  return *pt_;
}

/* --------------------------------------------------------------------
                  Apply projector P(^T) (for iterative solvers)
   -------------------------------------------------------------------- */
int Core::LinAlg::KrylovProjector::apply_p(Core::LinAlg::MultiVector<double>& Y) const
{
  /*
   *                  / T   \ -1   T
   * P(x) = x - c_ * | w_ c_ |  * w_ * x
   *                  \     /
   *                `----v-----'
   *                   invwTc_
   */

  if (!complete_)
    FOUR_C_THROW(
        "Krylov space projector is not complete. Call fill_complete() after changing c_ or w_.");

  return apply_projector(Y, *w_, *c_, *invw_tc_);
}

int Core::LinAlg::KrylovProjector::apply_pt(Core::LinAlg::MultiVector<double>& Y) const
{
  /*
   *  T                / T   \ -1   T
   * P (x) = x - w_ * | c_ w_ |  * c_ * x
   *                   \     /
   *                 `----v-----'
   *                  (invwTc_)^T
   */

  if (!complete_)
    FOUR_C_THROW(
        "Krylov space projector is not complete. Call fill_complete() after changing c_ or w_.");

  Core::LinAlg::SerialDenseMatrix invwTcT(*invw_tc_, Teuchos::TRANS);
  return apply_projector(Y, *c_, *w_, invwTcT);
}

/* --------------------------------------------------------------------
                  give out projection P^T A P
   -------------------------------------------------------------------- */
std::shared_ptr<Core::LinAlg::SparseMatrix> Core::LinAlg::KrylovProjector::project(
    const Core::LinAlg::SparseMatrix& A) const
{
  /*
   * P^T A P = A - { A c (w^T c)^-1 w^T + w (c^T w)^-1 c^T A } + w (c^T w)^-1 (c^T A c) (w^T c)^-1
   * w^T
   *                `--------v--------'   `--------v--------' `-----------------v------------------'
   *                        mat1                mat2                            mat3
   *
   *
   *
   */

  if (!complete_)
    FOUR_C_THROW(
        "Krylov space projector is not complete. Call fill_complete() after changing c_ or w_.");

  // auxiliary preliminary products

  std::shared_ptr<Core::LinAlg::MultiVector<double>> w_invwTc =
      multiply_multi_vector_dense_matrix(w_, invw_tc_);

  // here: matvec = A c_;
  std::shared_ptr<Core::LinAlg::MultiVector<double>> matvec =
      std::make_shared<Core::LinAlg::MultiVector<double>>(c_->get_map(), nsdim_, false);
  A.multiply(false, *c_, *matvec);

  // compute serial dense matrix c^T A c
  std::shared_ptr<Core::LinAlg::SerialDenseMatrix> cTAc =
      std::make_shared<Core::LinAlg::SerialDenseMatrix>(nsdim_, nsdim_, false);
  for (int i = 0; i < nsdim_; ++i)
    for (int j = 0; j < nsdim_; ++j) (*c_)(i).dot((*matvec)(j), &((*cTAc)(i, j)));
  // std::cout << *matvec << std::endl;
  // std::cout << A << std::endl;

  // std::cout << *w_invwTc << std::endl;
  // compute and add matrices
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat1 =
      multiply_multi_vector_multi_vector(matvec, w_invwTc, 1, false);
  {
    // put in brackets to delete mat2 immediately after being added to mat1
    // here: matvec = A^T c_;
    A.multiply(true, *c_, *matvec);
    std::shared_ptr<Core::LinAlg::SparseMatrix> mat2 =
        multiply_multi_vector_multi_vector(w_invwTc, matvec, 2, true);
    mat1->add(*mat2, false, 1.0, 1.0);
    mat1->complete();
  }

  // here: matvec = w (c^T w)^-1 (c^T A c);
  matvec = multiply_multi_vector_dense_matrix(w_invwTc, cTAc);
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat3 =
      multiply_multi_vector_multi_vector(matvec, w_invwTc, 1, false);
  mat3->add(*mat1, false, -1.0, 1.0);
  mat3->add(A, false, 1.0, 1.0);

  mat3->complete();
  return mat3;
}

/* ====================================================================
    private methods
   ==================================================================== */

/* --------------------------------------------------------------------
                    Create projector (for direct solvers)
   -------------------------------------------------------------------- */
void Core::LinAlg::KrylovProjector::create_projector(std::shared_ptr<Core::LinAlg::SparseMatrix>& P,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& v1,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& v2,
    const std::shared_ptr<Core::LinAlg::SerialDenseMatrix>& inv_v1Tv2)
{
  /*
   *               /  T  \ -1    T
   * P = I - v2 * | v1 v2 |  * v1
   *               \     /
   *      `--------v--------'
   *              temp1
   */

  // compute temp1
  std::shared_ptr<Core::LinAlg::MultiVector<double>> temp1 =
      multiply_multi_vector_dense_matrix(v2, inv_v1Tv2);
  temp1->Scale(-1.0);


  // compute P by multiplying upright temp1 with lying v1^T:
  P = Core::LinAlg::KrylovProjector::multiply_multi_vector_multi_vector(temp1, v1, 1, false);

  //--------------------------------------------------------
  // Add identity matrix
  //--------------------------------------------------------
  const int nummyrows = v1->MyLength();
  const double one = 1.0;
  // loop over all proc-rows
  for (int rr = 0; rr < nummyrows; ++rr)
  {
    // get global row id of current local row id
    const int grid = P->global_row_index(rr);

    // add identity matrix by adding 1 on diagonal entries
    int err = P->insert_global_values(grid, 1, &one, &grid);
    if (err < 0)
    {
      err = P->sum_into_global_values(grid, 1, &one, &grid);
      if (err < 0)
      {
        FOUR_C_THROW("insertion error when trying to computekrylov projection matrix.");
      }
    }
  }

  // call fill complete
  P->complete();

  return;
}


/* --------------------------------------------------------------------
                  Apply projector P(T) (for iterative solvers)
   -------------------------------------------------------------------- */
int Core::LinAlg::KrylovProjector::apply_projector(Core::LinAlg::MultiVector<double>& Y,
    Core::LinAlg::MultiVector<double>& v1, Core::LinAlg::MultiVector<double>& v2,
    Core::LinAlg::SerialDenseMatrix& inv_v1Tv2) const
{
  if (!complete_) FOUR_C_THROW("Krylov space projector is not complete. Call fill_complete().");

  int ierr = 0;

  // if necessary, project out matrix kernel to maintain well-posedness
  // of problem
  // there is only one solution vector --- so solution
  // vector index is zero
  if (Y.NumVectors() != 1)
  {
    FOUR_C_THROW("expecting only one solution vector during iterative solver Apply call\n");
  }

  /*
   *  (T)                /  T  \ -1    T
   * P   (x) = x - v2 * | v1 v2 |  * v1 * x
   *                     \     /
   *                                `---v---'
   *                                 =:temp1
   *                   `----------v----------'
   *                           =:temp2
   */

  // compute dot product of solution vector with all projection vectors
  // temp1(rr) = v1(rr)^T * Y
  Core::LinAlg::SerialDenseVector temp1(nsdim_);
  for (int rr = 0; rr < nsdim_; ++rr)
  {
    (Y(0)).dot((v1)(rr), &(temp1(rr)));
  }

  // compute temp2 from matrix-vector-product:
  // temp2 = (v1^T v2)^(-1) * temp1
  Core::LinAlg::SerialDenseVector temp2(nsdim_);
  Core::LinAlg::multiply(temp2, inv_v1Tv2, temp1);

  // loop
  for (int rr = 0; rr < nsdim_; ++rr)
  {
    Y(0).update(-temp2(rr), v2(rr), 1.0);
  }

  return (ierr);
}  // Core::LinAlg::KrylovProjector::apply_projector



/* --------------------------------------------------------------------
   -------------------------------------------------------------------- */
std::shared_ptr<Core::LinAlg::MultiVector<double>>
Core::LinAlg::KrylovProjector::multiply_multi_vector_dense_matrix(
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& mv,
    const std::shared_ptr<Core::LinAlg::SerialDenseMatrix>& dm) const
{
  if (mv == nullptr or dm == nullptr)
    FOUR_C_THROW("Either the multivector or the densematrix point to nullptr");

  // create empty multivector mvout
  std::shared_ptr<Core::LinAlg::MultiVector<double>> mvout =
      std::make_shared<Core::LinAlg::MultiVector<double>>(mv->get_map(), nsdim_);

  // loop over all vectors of mvout
  for (int rr = 0; rr < nsdim_; ++rr)
  {
    // extract i-th (rr-th) vector of mvout
    auto& mvouti = (*mvout)(rr);
    // loop over all vectors of mv
    for (int mm = 0; mm < nsdim_; ++mm)
    {
      // scale j-th (mm-th) vector of mv with corresponding entry of dm
      // and add to i-th (rr-th) vector of mvout
      mvouti.update((*dm)(mm, rr), (*mv)(mm), 1.0);
    }
  }

  return mvout;
}

/* --------------------------------------------------------------------
                  outer product of two Epetra_MultiVectors
   -------------------------------------------------------------------- */
std::shared_ptr<Core::LinAlg::SparseMatrix>
Core::LinAlg::KrylovProjector::multiply_multi_vector_multi_vector(
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& mv1,
    const std::shared_ptr<Core::LinAlg::MultiVector<double>>& mv2, const int id,
    const bool fill) const
{
  if (mv1 == nullptr or mv2 == nullptr)
    FOUR_C_THROW("At least one multi-vector points to nullptr.");

  // compute information about density of P^T A P
  std::shared_ptr<Core::LinAlg::MultiVector<double>> temp = nullptr;
  if (id == 1)
    temp = mv1;
  else if (id == 2)
    temp = mv2;
  else
    FOUR_C_THROW("id must be 1 or 2");

  Core::LinAlg::Vector<double> prod((*temp)(0));
  for (int i = 1; i < nsdim_; ++i) prod.multiply(1.0, (*temp)(i), prod, 1.0);
  int numnonzero = 0;
  for (int i = 0; i < prod.local_length(); ++i)
    if (prod[i] != 0.0) numnonzero++;

  int glob_numnonzero = 0;
  Core::Communication::sum_all(&numnonzero, &glob_numnonzero, 1, prod.get_comm());

  Core::LinAlg::Map mv1map(mv1->get_map().num_global_elements(), mv1->get_map().num_my_elements(),
      mv1->get_map().my_global_elements(), 0, mv1->get_map().get_comm());
  // initialization of mat with map of mv1
  std::shared_ptr<Core::LinAlg::SparseMatrix> mat =
      std::make_shared<Core::LinAlg::SparseMatrix>(mv1map, glob_numnonzero, false);

  //-------------------------------
  // make mv2 redundant on all procs:
  //-------------------------------
  // auxiliary variables
  const int nummyrows = mv1->MyLength();
  const int numvals = mv2->GlobalLength();

  Core::LinAlg::Map mv2map(mv2->get_map().num_global_elements(), mv2->get_map().num_my_elements(),
      mv2->get_map().my_global_elements(), 0, mv2->get_map().get_comm());

  // fully redundant/overlapping map
  std::shared_ptr<Core::LinAlg::Map> redundant_map = Core::LinAlg::allreduce_e_map(mv2map);
  // initialize global mv2 without setting to 0
  Core::LinAlg::MultiVector<double> mv2glob(*redundant_map, nsdim_);
  // create importer with redundant target map and distributed source map
  Core::LinAlg::Import importer(*redundant_map, mv2->get_map());
  // import values to global mv2
  mv2glob.Import(*mv2, importer, Insert);

  //--------------------------------------------------------
  // compute mat by multiplying upright mv1 with lying mv2^T:
  //--------------------------------------------------------
  // loop over all proc-rows
  for (int rr = 0; rr < nummyrows; ++rr)
  {
    // get global row id of current local row id
    const int grid = mat->global_row_index(rr);

    // vector of all row values - prevented from growing in following loops
    std::vector<double> rowvals;
    rowvals.reserve(numvals);

    // vector of indices corresponding to vector of row values
    std::vector<int> indices;
    indices.reserve(numvals);

    // loop over all entries of global w
    for (int mm = 0; mm < numvals; ++mm)
    {
      double sum = 0.0;
      // loop over all kernel/weight vector
      for (int vv = 0; vv < nsdim_; ++vv)
      {
        sum += (*mv1)(vv)[rr] * mv2glob(vv)[mm];
      }

      // add value to vector only if non-zero
      if (sum != 0.0)
      {
        rowvals.push_back(sum);
        indices.push_back(redundant_map->gid(mm));
      }
    }

    // insert values in mat
    int err = mat->insert_global_values(grid, indices.size(), rowvals.data(), indices.data());
    if (err < 0)
    {
      FOUR_C_THROW(
          "insertion error when trying to compute krylov projection matrix (error code: {}).", err);
    }
  }

  // call fill complete
  if (fill) mat->complete();

  return mat;
}

FOUR_C_NAMESPACE_CLOSE
