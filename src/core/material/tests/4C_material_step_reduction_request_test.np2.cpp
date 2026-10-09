// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_material_step_reduction_request.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"

#include <mpi.h>

#include <string>
#include <tuple>

namespace
{
  using namespace FourC;
  using namespace FourC::Core::Mat::StepReduction;

  void request_reduction_on_rank_1(const std::string& reason)
  {
    if (Core::Communication::my_mpi_rank(MPI_COMM_WORLD) == 1) request(reason);
  }

  void synchronize_without_request()
  {
    run_and_synchronize_request(MPI_COMM_WORLD, [] {});
  }

  TEST(MaterialStepReductionRequest, EvaluationWithoutRequestReturnsFalse)
  {
    const bool request_detected =
        run_and_detect_synchronized_request(true, synchronize_without_request);

    EXPECT_FALSE(request_detected);
  }

  TEST(MaterialStepReductionRequest, RequestOnOneRankIsDetectedOnEveryRank)
  {
    const bool request_detected = run_and_detect_synchronized_request(true,
        []
        {
          run_and_synchronize_request(
              MPI_COMM_WORLD, [] { request_reduction_on_rank_1("rank 1 requested."); });
        });

    EXPECT_TRUE(request_detected);
  }

  TEST(MaterialStepReductionRequest, RequestOnEveryRankIsDetected)
  {
    const bool request_detected = run_and_detect_synchronized_request(true, []
        { run_and_synchronize_request(MPI_COMM_WORLD, [] { request("all ranks requested."); }); });

    EXPECT_TRUE(request_detected);
  }

  TEST(MaterialStepReductionRequest, RequestOnOneRankAbortsOperationOnEveryRank)
  {
    bool operation_continued = false;
    const bool request_detected = run_and_detect_synchronized_request(true,
        [&]
        {
          run_and_synchronize_request(
              MPI_COMM_WORLD, [] { request_reduction_on_rank_1("rank 1 requested."); });
          operation_continued = true;
        });

    EXPECT_TRUE(request_detected);
    EXPECT_FALSE(operation_continued);  // still false on every rank
  }

  TEST(MaterialStepReductionRequest, SequentialSynchronizationRegionsAreAllowed)
  {
    const bool request_detected = run_and_detect_synchronized_request(true,
        []
        {
          run_and_synchronize_request(MPI_COMM_WORLD, [] {});
          run_and_synchronize_request(MPI_COMM_WORLD, [] {});
        });

    EXPECT_FALSE(request_detected);
  }

  TEST(MaterialStepReductionRequest, RequestInFirstSynchronizationRegionSkipsLaterRegions)
  {
    bool second_region_entered = false;
    const bool request_detected = run_and_detect_synchronized_request(true,
        [&]
        {
          run_and_synchronize_request(
              MPI_COMM_WORLD, [] { request_reduction_on_rank_1("rank 1 requested."); });
          run_and_synchronize_request(MPI_COMM_WORLD, [&] { second_region_entered = true; });
        });

    EXPECT_TRUE(request_detected);
    EXPECT_FALSE(second_region_entered);
  }

  TEST(MaterialStepReductionRequest, RequestInLaterSynchronizationRegionIsDetected)
  {
    const bool request_detected = run_and_detect_synchronized_request(true,
        []
        {
          run_and_synchronize_request(MPI_COMM_WORLD, [] {});
          run_and_synchronize_request(
              MPI_COMM_WORLD, [] { request_reduction_on_rank_1("rank 1 requested."); });
        });

    EXPECT_TRUE(request_detected);
  }

  TEST(MaterialStepReductionRequest, SynchronizationRegionOutsideDetectionRegionIsAllowed)
  {
    EXPECT_NO_THROW(run_and_synchronize_request(MPI_COMM_WORLD, [] {}));
  }

  TEST(MaterialStepReductionRequest, RequestOutsideSynchronizationRegionThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(std::ignore = run_and_detect_synchronized_request(
                                         true, [] { request("all ranks requested."); }),
        Core::Exception,
        "A material requested step reduction in an evaluation path that does not support "
        "it.\nRun the synchronized element/material evaluation through "
        "Core::Mat::StepReduction::run_and_synchronize_request().\nReason:\nall ranks "
        "requested.\n");
  }

  TEST(MaterialStepReductionRequest, RequestOutsideDetectionRegionThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        { run_and_synchronize_request(MPI_COMM_WORLD, [] { request("all ranks requested."); }); },
        Core::Exception,
        "A material requested step reduction but the calling algorithm does not support it.\n"
        "Run the operation through "
        "Core::Mat::StepReduction::run_and_detect_synchronized_request().\n"
        "Reason:\nall ranks requested.");

    EXPECT_NO_THROW(run_and_synchronize_request(MPI_COMM_WORLD, [] {}));
  }

  TEST(MaterialStepReductionRequest, NestedDetectionRegionsThrow)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = run_and_detect_synchronized_request(
            true, [] { std::ignore = run_and_detect_synchronized_request(true, [] {}); }),
        Core::Exception,
        "Nested Core::Mat::StepReduction::run_and_detect_synchronized_request() calls are not "
        "supported");
  }

  TEST(MaterialStepReductionRequest, NestedSynchronizationRegionsThrow)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = run_and_detect_synchronized_request(true,
            []
            {
              run_and_synchronize_request(
                  MPI_COMM_WORLD, [] { run_and_synchronize_request(MPI_COMM_WORLD, [] {}); });
            }),
        Core::Exception,
        "Nested Core::Mat::StepReduction::run_and_synchronize_request() calls are not "
        "supported");
  }

  TEST(MaterialStepReductionRequest, UnrelatedExceptionPropagatesAndResetsRegions)
  {
    // The exception must be thrown on all ranks: a rank throwing alone would skip the
    // synchronization collective, leaving the other ranks blocked in it. Since the test catches the
    // exception instead of aborting, this would deadlock.
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(std::ignore = run_and_detect_synchronized_request(true,
                                         []
                                         {
                                           run_and_synchronize_request(MPI_COMM_WORLD, []
                                               { FOUR_C_THROW("unrelated evaluation exception"); });
                                         }),
        Core::Exception, "unrelated evaluation exception");

    // A following operation can still enter a request-detection region because the state was reset
    // after the exception.
    EXPECT_FALSE(run_and_detect_synchronized_request(true, synchronize_without_request));
  }

  TEST(MaterialStepReductionRequest, RequestInDisabledDetectionRegionThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = run_and_detect_synchronized_request(false,
            []
            {
              run_and_synchronize_request(MPI_COMM_WORLD, [] { request("all ranks requested."); });
            }),
        Core::Exception,
        "A material requested step reduction, but the calling algorithm disabled handling of these "
        "requests.\nReason:\nall ranks requested.");
  }

  TEST(MaterialStepReductionRequest, DiagnosticUsesDetailedReasonIfGiven)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        std::ignore = run_and_detect_synchronized_request(false,
            []
            {
              run_and_synchronize_request(MPI_COMM_WORLD,
                  [] { request("short reason.", "detailed reason on all ranks."); });
            }),
        Core::Exception, "Reason:\ndetailed reason on all ranks.");
  }
}  // namespace
