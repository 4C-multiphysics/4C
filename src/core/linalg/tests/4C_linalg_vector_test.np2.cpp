// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_linalg_vector.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_linalg_map.hpp"
#include "4C_linalg_multi_vector.hpp"
#include "4C_linalg_sparsematrix.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace
{
  class VectorTest : public testing::Test
  {
   public:
    MPI_Comm comm_;
    std::shared_ptr<Core::LinAlg::Map> map_;
    int NumGlobalElements = 10;

   protected:
    VectorTest()
    {
      // set up communicator
      comm_ = MPI_COMM_WORLD;

      // set up a map
      map_ = std::make_shared<Core::LinAlg::Map>(NumGlobalElements, 0, comm_);
    }
  };

  TEST_F(VectorTest, ConstructorsAndNorms)
  {
    // create an epetra vector
    Epetra_Vector my_epetra_vector = Epetra_Vector(map_->get_epetra_block_map(), true);

    // try to copy zero vector into wrapper
    Core::LinAlg::Vector<double> epetra_based_test_vector =
        Core::LinAlg::Vector<double>(my_epetra_vector);

    // create vector
    Core::LinAlg::Vector<double> test_vector = Core::LinAlg::Vector<double>(*map_, true);

    // initialize with wrong value
    double norm_of_test_vector = 1;

    test_vector.print(std::cout);
    test_vector.norm_2(&norm_of_test_vector);
    // test norm2 and success of both vectors
    ASSERT_FLOAT_EQ(0.0, norm_of_test_vector);

    // reset value
    norm_of_test_vector = 1;

    // check result of Norm2
    epetra_based_test_vector.norm_2(&norm_of_test_vector);
    ASSERT_FLOAT_EQ(0.0, norm_of_test_vector);

    // test element access function for proc 0
    if (Core::Communication::my_mpi_rank(comm_) == 0) test_vector.get_values()[1] = 1;

    // check result of Norm1
    test_vector.norm_1(&norm_of_test_vector);
    ASSERT_FLOAT_EQ(1.0, norm_of_test_vector);

    test_vector.get_values()[1] = 100.0;

    // check result of NormInf
    test_vector.norm_inf(&norm_of_test_vector);
    ASSERT_FLOAT_EQ(100.0, norm_of_test_vector);
  }

  TEST_F(VectorTest, DeepCopying)
  {
    Core::LinAlg::Vector<double> a(*map_, true);
    a.put_scalar(1.0);

    Core::LinAlg::Vector<double> b(*map_, true);
    // copy assign
    b = a;
    b.put_scalar(2.0);
    double norm_a = 0.0;
    double norm_b = 0.0;
    a.norm_2(&norm_a);
    b.norm_2(&norm_b);

    EXPECT_FLOAT_EQ(norm_a, 1.0 * std::sqrt(NumGlobalElements));
    EXPECT_FLOAT_EQ(norm_b, 2.0 * std::sqrt(NumGlobalElements));

    // copy constructor
    Core::LinAlg::Vector<double> c(a);
    c.put_scalar(3.0);
    double norm_c = 0.0;
    c.norm_2(&norm_c);
    EXPECT_FLOAT_EQ(norm_c, 3.0 * std::sqrt(NumGlobalElements));
  }

  TEST_F(VectorTest, PutScalar)
  {
    // initialize with false value
    double norm_of_test_vector = 0.0;

    // copy zero vector into new interface
    Core::LinAlg::Vector<double> test_vector = Core::LinAlg::Vector<double>(*map_, true);

    test_vector.put_scalar(2.0);

    // check result
    test_vector.norm_2(&norm_of_test_vector);
    ASSERT_FLOAT_EQ(NumGlobalElements * 2.0 * 2.0, norm_of_test_vector * norm_of_test_vector);
  }

  TEST_F(VectorTest, Update)
  {
    Core::LinAlg::Vector<double> a = Core::LinAlg::Vector<double>(*map_, true);
    a.put_scalar(1.0);

    Core::LinAlg::Vector<double> b = Core::LinAlg::Vector<double>(*map_, true);
    b.put_scalar(1.0);

    // update the vector
    b.update(2.0, a, 3.0);

    // initialize with false value
    double b_norm = 0.0;

    // check norm of vector
    b.norm_2(&b_norm);
    ASSERT_FLOAT_EQ(NumGlobalElements * (2.0 + 3.0) * (2.0 + 3.0), b_norm * b_norm);

    Core::LinAlg::Vector<double> c = Core::LinAlg::Vector<double>(*map_, true);
    c.update(1, a, -1, b, 0);

    // initialize with false value
    double c_norm = 0.0;

    // check norm of vector
    c.norm_1(&c_norm);
    ASSERT_FLOAT_EQ(4 * NumGlobalElements, c_norm);
  }


  TEST_F(VectorTest, View)
  {
    Epetra_Vector a(map_->get_epetra_block_map(), true);
    a.PutScalar(1.0);
    // Scope in which a is modified by the view
    {
      Core::LinAlg::View a_view(a);

      double norm = 0.0;
      ((Core::LinAlg::Vector<double>&)a_view).norm_2(&norm);
      EXPECT_EQ(norm, std::sqrt(NumGlobalElements));

      ((Core::LinAlg::Vector<double>&)a_view).put_scalar(2.0);
    }
    const Epetra_Vector& a_const = a;
    Core::LinAlg::View a_view_const(a_const);
    // Change must be reflected in a
    double norm = 0.0;
    static_cast<const Core::LinAlg::Vector<double>&>(a_view_const).norm_2(&norm);
    EXPECT_EQ(norm, 2.0 * std::sqrt(NumGlobalElements));
  }

  std::vector<double> means_multi_vector(const Core::LinAlg::MultiVector<double>& mv)
  {
    std::vector<double> means(mv.num_vectors());
    mv.mean_value(means.data());
    return means;
  }


  TEST_F(VectorTest, MultiVectorImplicitConversionView)
  {
    Core::LinAlg::Vector<double> a(*map_, true);
    a.put_scalar(1.0);

    // This views the data that is in a. It does not copy the data.
    // This results in the same behavior as inheritance would give.
    EXPECT_EQ(means_multi_vector(a)[0], 1.0);

    // This copies the data.
    Core::LinAlg::MultiVector<double> mv = a;
    a.put_scalar(2.0);

    // mv should still be 1.0 because we only modified a.
    EXPECT_EQ(means_multi_vector(mv)[0], 1.0);
  }

  TEST_F(VectorTest, MultiVectorImplicitConversionCopy)
  {
    auto a = std::make_shared<Core::LinAlg::Vector<double>>(*map_, true);
    a->put_scalar(1.0);

    // This copies the data.
    Core::LinAlg::MultiVector<double> mv = *a;
    a->put_scalar(2.0);
    // Explicitly deallocate a to make sure that mv is not a view.
    a = nullptr;

    // mv should still be 1.0 because we only modified a.
    EXPECT_EQ(means_multi_vector(mv)[0], 1.0);
  }

  TEST_F(VectorTest, MultiVectorImplicitConversionRef)
  {
    Core::LinAlg::Vector<double> a(*map_, true);
    a.put_scalar(1.0);

    Core::LinAlg::MultiVector<double>& mv = a;
    mv.put_scalar(2.0);
    EXPECT_EQ(means_multi_vector(a)[0], 2.0);

    // Reassigning to a must keep mv valid: move assign
    a = Core::LinAlg::Vector<double>(*map_, true);
    EXPECT_EQ(means_multi_vector(mv)[0], 0.0);
    a.put_scalar(3.0);
    EXPECT_EQ(means_multi_vector(mv)[0], 3.0);

    // Reassigning to a must keep mv valid: copy assign
    Core::LinAlg::Vector<double> b(*map_, true);
    a = b;
    EXPECT_EQ(means_multi_vector(mv)[0], 0.0);
    a.put_scalar(4.0);
    EXPECT_EQ(means_multi_vector(mv)[0], 4.0);
  }

  TEST_F(VectorTest, AssignToRef)
  {
    Core::LinAlg::Vector<double> a(*map_, true);
    a.put_scalar(1.0);
    EXPECT_EQ(means_multi_vector(a)[0], 1.0);
    Core::LinAlg::MultiVector<double>& mv = a;
    // Actually assign an MV to a via the ref. Note that this would throw in Trilinos if not using a
    // single column.
    mv = Core::LinAlg::MultiVector<double>(*map_, 1, true);
    EXPECT_EQ(means_multi_vector(mv)[0], 0.0);
  }

  TEST_F(VectorTest, VectorFromMultiVector)
  {
    Core::LinAlg::MultiVector<double> mv(*map_, 3, true);
    mv.put_scalar(1.0);

    const int index = 1;

    Core::LinAlg::Vector<double>& a = mv(index);
    EXPECT_EQ(means_multi_vector(a)[0], 1.0);

    a.put_scalar(2.0);

    // Check that the change is reflected in the MultiVector
    EXPECT_EQ(means_multi_vector(mv), (std::vector{1., 2., 1.}));

    // Another MultiVector conversion
    Core::LinAlg::MultiVector<double>& mv2 = a;
    mv2.put_scalar(3.0);
    EXPECT_EQ(means_multi_vector(mv), (std::vector{1., 3., 1.}));

    // Combine with taking a view
    {
      const auto put_scalar = [](Core::LinAlg::MultiVector<double>& v, double s)
      { v.put_scalar(s); };
      Core::LinAlg::View view_mv2(mv2.get_epetra_multi_vector());
      put_scalar(view_mv2, 4.0);
    }
    EXPECT_EQ(means_multi_vector(mv), (std::vector{1., 4., 1.}));
  }

  TEST_F(VectorTest, ReplaceMap)
  {
    Core::LinAlg::Vector<double> a(*map_, true);
    a.put_scalar(1.0);

    const Core::LinAlg::MultiVector<double>& b = a;
    ASSERT_EQ(b.num_vectors(), 1);
    const Core::LinAlg::Vector<double>& c = b(0);

    // New map where elements are distributed differently
    std::array<int, 5> my_elements;
    if (Core::Communication::my_mpi_rank(comm_) == 0)
      my_elements = {0, 2, 4, 6, 8};
    else
      my_elements = {1, 3, 5, 7, 9};
    Core::LinAlg::Map new_map(10, my_elements.size(), my_elements.data(), 0, comm_);

    // Before replacement, all maps are the same.
    EXPECT_TRUE(a.get_map().same_as(b.get_map()));
    EXPECT_TRUE(a.get_map().same_as(c.get_map()));

    EXPECT_EQ(&a.get_ref_of_epetra_vector(), &b.get_epetra_multi_vector());
    // A change of the map invalidates views, so we need to be careful.
    a.replace_map(new_map);

    {
      // This highlights a bug in Trilinos: the Epetra_Vector views into a MultiVector are only
      // set once and never updated, although a map replacement would require an update.
      const Core::LinAlg::MultiVector<double>& b_new = a;
      const Core::LinAlg::Vector<double>& c_new = b_new(0);
      // This is correct.
      EXPECT_TRUE(b_new.get_map().same_as(new_map));
      // This is the bug: c_new still has the old map although we just took a new view into b_new.
      EXPECT_TRUE(c_new.get_map().same_as(*map_));
    }
  }

}  // namespace

FOUR_C_NAMESPACE_CLOSE
