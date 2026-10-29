// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "4C_io_runtime_csv_writer.hpp"

#include "4C_fem_general_shape_function_type.hpp"
#include "4C_io_control.hpp"
#include "4C_utils_exceptions.hpp"

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  using namespace FourC;
  using namespace FourC::Core::IO;

  //! A directory below the system temp dir that is removed again when the test ends
  class TempOutputDir
  {
   public:
    TempOutputDir()
    {
      path_ = std::filesystem::temp_directory_path() /
              ("4C_runtime_csv_writer_test_" + std::to_string(getpid()));
      std::filesystem::remove_all(path_);
      std::filesystem::create_directories(path_);
    }

    TempOutputDir(const TempOutputDir&) = delete;
    TempOutputDir& operator=(const TempOutputDir&) = delete;
    TempOutputDir(TempOutputDir&&) = delete;
    TempOutputDir& operator=(TempOutputDir&&) = delete;

    ~TempOutputDir() { std::filesystem::remove_all(path_); }

    //! The prefix handed to the output control. The writer appends "-<name>.csv" to it.
    std::string prefix() const { return (path_ / stem()).string(); }

    /*!
     * \brief The prefix of the run we restart from
     *
     * It differs from the output prefix because a restart always writes to a new file, which is
     * also why the writer may safely truncate its output file before reading the restart file.
     */
    std::string restart_prefix() const { return (path_ / (stem() + "-old")).string(); }

    //! Full path of the file the writer produces for @p output_name
    std::string file(const std::string& output_name) const
    {
      return (path_ / (stem() + "-" + output_name + ".csv")).string();
    }

    //! Full path of the file the writer takes its prior rows from
    std::string restart_file(const std::string& output_name) const
    {
      return (path_ / (stem() + "-old-" + output_name + ".csv")).string();
    }

   private:
    //! Name the output control gets, without any extension
    static std::string stem() { return "run"; }

    std::filesystem::path path_;
  };

  /*!
   * \brief An output control that writes no control file and keeps its name stable
   *
   * The name is adapted by the output control itself on a restart, which would move the file we
   * are looking at, hence adaptname is switched off.
   */
  std::unique_ptr<Core::IO::OutputControl> make_output_control(
      const TempOutputDir& dir, const int restart_step)
  {
    return std::make_unique<Core::IO::OutputControl>(MPI_COMM_WORLD, "test",
        Core::FE::ShapeFunctionType::polynomial, "input.xml", dir.restart_prefix(), dir.prefix(),
        /*ndim=*/3, restart_step, /*filesteps=*/0, /*write_binary_output=*/false,
        /*adaptname=*/false);
  }

  std::vector<std::string> read_lines(const TempOutputDir& dir, const std::string& output_name)
  {
    std::ifstream stream(dir.file(output_name));
    EXPECT_TRUE(stream.good()) << "output file '" << dir.file(output_name) << "' does not exist";

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line)) lines.push_back(line);
    return lines;
  }

  //! Write a file that stands in for the output of the run we restart from
  void write_restart_file(const TempOutputDir& dir, const std::string& output_name,
      const std::vector<std::string>& lines)
  {
    std::ofstream stream(dir.restart_file(output_name), std::ios_base::trunc);
    for (const std::string& line : lines) stream << line << "\n";
  }


  TEST(RuntimeCsvWriterTest, WritesOnlyRegisteredColumns)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "generic");

    writer.register_data_vector("beta", 1, 16);
    writer.register_data_vector("alpha", 1, 16);

    writer.write_data_to_file({{"alpha", {2.5}}, {"beta", {1.5}}});

    // neither a step nor a time, since nothing asked for them
    EXPECT_THAT(read_lines(dir, "generic"),
        testing::ElementsAre("alpha,beta", "2.5000000000000000e+00,1.5000000000000000e+00"));
  }

  TEST(RuntimeCsvWriterTest, FixedColumnsComeFirstInRegistrationOrder)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "order");

    // fixed columns keep their registration order, the remaining ones are sorted by name
    writer.register_column({"load", 1, CsvNumberFormat::DefaultFloat, 6, CsvColumnOrder::Fixed});
    writer.register_data_vector("zeta", 1, 16);
    writer.register_column({"iter", 1, CsvNumberFormat::Integer, 0, CsvColumnOrder::Fixed});
    writer.register_data_vector("beta", 1, 16);

    writer.append_data_vector("load", {1.5});
    writer.append_data_vector("iter", {7.0});
    writer.write_data_to_file({{"beta", {2.0}}, {"zeta", {3.0}}});

    EXPECT_THAT(
        read_lines(dir, "order"), testing::ElementsAre("load,iter,beta,zeta",
                                      "1.5,7,2.0000000000000000e+00,3.0000000000000000e+00"));
  }

  TEST(RuntimeCsvWriterTest, MultiComponentColumnsAreSplit)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "vector");

    writer.register_data_vector("f", 3, 10);
    writer.register_data_vector("s", 1, 10);

    writer.write_data_to_file({{"f", {1.0, 2.0, 3.0}}, {"s", {4.0}}});

    EXPECT_THAT(read_lines(dir, "vector"),
        testing::ElementsAre("f:0,f:1,f:2,s",
            "1.0000000000e+00,2.0000000000e+00,3.0000000000e+00,4.0000000000e+00"));
  }

  TEST(RuntimeCsvWriterTest, NumberFormatsAndPrecision)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "formats");

    writer.register_column({"step", 1, CsvNumberFormat::Integer, 0, CsvColumnOrder::Fixed});
    writer.register_column({"time", 1, CsvNumberFormat::DefaultFloat, 6, CsvColumnOrder::Fixed});
    writer.register_data_vector("value", 1, 4);

    writer.write_data_to_file({{"step", {100.0}}, {"time", {17.5}}, {"value", {1.0 / 3.0}}});

    EXPECT_THAT(
        read_lines(dir, "formats"), testing::ElementsAre("step,time,value", "100,17.5,3.3333e-01"));
  }

  TEST(RuntimeCsvWriterWithTimeAndStepTest, PrependsStepAndTime)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriterWithTimeAndStep writer(0, *output_control, "with_time");

    writer.register_data_vector("energy", 1, 16);

    writer.write_data_to_file(0.5, 3, {{"energy", {2.0}}});
    writer.write_data_to_file(0.01, 10, {{"energy", {4.0}}});

    EXPECT_THAT(read_lines(dir, "with_time"),
        testing::ElementsAre(
            "step,time,energy", "3,0.5,2.0000000000000000e+00", "10,0.01,4.0000000000000000e+00"));
  }

  TEST(RuntimeCsvWriterTest, BothWriteStylesAgree)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "styles");
    writer.register_data_vector("a", 2, 16);
    writer.register_data_vector("b", 1, 16);

    writer.append_data_vector("a", {1.0, 2.0});
    writer.append_data_vector("b", {3.0});
    writer.write_collected_data_to_file();

    writer.write_data_to_file({{"a", {1.0, 2.0}}, {"b", {3.0}}});

    EXPECT_THAT(read_lines(dir, "styles"),
        testing::ElementsAre("a:0,a:1,b",
            "1.0000000000000000e+00,2.0000000000000000e+00,3.0000000000000000e+00",
            "1.0000000000000000e+00,2.0000000000000000e+00,3.0000000000000000e+00"));
  }

  TEST(RuntimeCsvWriterTest, RegisteredColumnsKeepTheirValuesBetweenRows)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "values");

    writer.register_column({"counter", 1, CsvNumberFormat::Integer, 0, CsvColumnOrder::Fixed});
    writer.register_data_vector("result", 1, 4);

    writer.append_data_vector("counter", {1.0});
    writer.append_data_vector("result", {1.0});
    writer.write_collected_data_to_file();

    // the counter is not touched, so its previous value is written again
    writer.append_data_vector("result", {2.0});
    writer.write_collected_data_to_file();

    EXPECT_THAT(read_lines(dir, "values"),
        testing::ElementsAre("counter,result", "1,1.0000e+00", "1,2.0000e+00"));
  }

  TEST(RuntimeCsvWriterTest, RegisteringIdenticalColumnTwiceIsANoOp)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "idempotent");
    writer.register_data_vector("a", 1, 16);
    writer.register_data_vector("a", 1, 16);

    EXPECT_NO_THROW(writer.write_data_to_file({{"a", {1.0}}}));
    EXPECT_THAT(read_lines(dir, "idempotent"), testing::ElementsAre("a", "1.0000000000000000e+00"));
  }

  TEST(RuntimeCsvWriterTest, RejectsRegisteringAfterFirstRow)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "late");
    writer.register_data_vector("a", 1, 16);
    writer.write_data_to_file({{"a", {1.0}}});

    EXPECT_THROW(writer.register_data_vector("b", 1, 16), Core::Exception);
  }

  TEST(RuntimeCsvWriterTest, RejectsConflictingRegistration)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "conflict");
    writer.register_data_vector("a", 1, 16);

    EXPECT_THROW(writer.register_data_vector("a", 2, 16), Core::Exception);
    EXPECT_THROW(writer.register_data_vector("a", 1, 8), Core::Exception);
  }

  TEST(RuntimeCsvWriterTest, RejectsEmptyColumnName)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "empty");

    EXPECT_THROW(writer.register_data_vector("", 1, 16), Core::Exception);
  }

  TEST(RuntimeCsvWriterTest, RejectsUnregisteredOrWronglySizedData)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "invalid");
    writer.register_data_vector("a", 1, 16);
    writer.register_data_vector("b", 1, 16);

    // a sorted column without any data
    EXPECT_THROW(writer.write_data_to_file({{"a", {1.0}}}), Core::Exception);
    // a column that was never registered
    EXPECT_THROW(
        writer.write_data_to_file({{"a", {1.0}}, {"b", {2.0}}, {"c", {3.0}}}), Core::Exception);
    // the wrong number of components
    EXPECT_THROW(writer.write_data_to_file({{"a", {1.0}}, {"b", {2.0, 3.0}}}), Core::Exception);
    // appending to an unregistered column
    EXPECT_THROW(writer.append_data_vector("c", {1.0}), Core::Exception);
    // appending the wrong number of components
    EXPECT_THROW(writer.append_data_vector("a", {1.0, 2.0}), Core::Exception);
  }

  TEST(RuntimeCsvWriterWithTimeAndStepTest, RestartKeepsRowsUpToTheRestartStep)
  {
    TempOutputDir dir;
    write_restart_file(dir, "restarted",
        {"step,time,energy", "0,0,1.0", "1,0.1,2.0", "2,0.2,3.0", "3,0.3,4.0", "4,0.4,5.0"});

    // restart from step 2, so the header and the rows up to and including step 2 are taken over
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 2);

    Core::IO::RuntimeCsvWriterWithTimeAndStep writer(0, *output_control, "restarted");
    writer.register_data_vector("energy", 1, 1);

    writer.write_data_to_file(0.5, 5, {{"energy", {6.0}}});

    // the taken over rows are copied verbatim, only the new row is formatted by the writer
    EXPECT_THAT(read_lines(dir, "restarted"), testing::ElementsAre("step,time,energy", "0,0,1.0",
                                                  "1,0.1,2.0", "2,0.2,3.0", "5,0.5,6.0e+00"));
  }

  TEST(RuntimeCsvWriterTest, RestartWithHelperPolicyKeepsRowsUpToTheRestartStep)
  {
    TempOutputDir dir;
    write_restart_file(dir, "up_to_step", {"alpha", "1", "2", "3"});

    // restart from step 2, so the header and the rows of the steps 1 and 2 are taken over
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 2);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "up_to_step",
        Core::IO::RuntimeCsvWriter::keep_rows_up_to_restart_step());
    writer.register_data_vector("alpha", 1, 1);

    writer.write_data_to_file({{"alpha", {4.0}}});

    EXPECT_THAT(read_lines(dir, "up_to_step"), testing::ElementsAre("alpha", "1", "2", "4.0e+00"));
  }

  TEST(RuntimeCsvWriterTest, RestartWithArbitraryPolicyKeepsSelectedRows)
  {
    TempOutputDir dir;
    write_restart_file(dir, "custom", {"alpha", "drop", "keep", "drop"});

    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 1);

    // a policy may capture arbitrary state and need not know anything about time steps
    const std::string wanted = "keep";
    Core::IO::RuntimeCsvWriter writer(0, *output_control, "custom",
        [&wanted](std::string_view line, int) { return line == wanted; });
    writer.register_data_vector("alpha", 1, 1);

    writer.write_data_to_file({{"alpha", {4.0}}});

    EXPECT_THAT(read_lines(dir, "custom"), testing::ElementsAre("alpha", "keep", "4.0e+00"));
  }

  TEST(RuntimeCsvWriterTest, RestartWithoutPolicyOnlyKeepsTheHeader)
  {
    TempOutputDir dir;
    write_restart_file(dir, "no_policy", {"alpha", "1.0", "2.0", "3.0"});

    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 1);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "no_policy");
    writer.register_data_vector("alpha", 1, 1);

    writer.write_data_to_file({{"alpha", {4.0}}});

    EXPECT_THAT(read_lines(dir, "no_policy"), testing::ElementsAre("alpha", "4.0e+00"));
  }

  TEST(RuntimeCsvWriterTest, RestartWithoutPolicyWarnsOnlyOnce)
  {
    TempOutputDir dir;
    write_restart_file(
        dir, "no_policy_once", {"alpha", "beta", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"});

    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 1);

    std::stringstream captured;
    std::streambuf* const previous_cerr = std::cerr.rdbuf(captured.rdbuf());

    {
      Core::IO::RuntimeCsvWriter writer(0, *output_control, "no_policy_once");
      writer.register_data_vector("alpha", 1, 1);
      writer.write_data_to_file({{"alpha", {8.0}}});
    }

    std::cerr.rdbuf(previous_cerr);

    // the restart file holds seven data rows, but the user should not be spammed
    const std::string warnings = captured.str();
    EXPECT_THAT(warnings, testing::HasSubstr("no policy"));
    EXPECT_EQ(std::count(warnings.begin(), warnings.end(), '\n'), 1);
  }

  TEST(RuntimeCsvWriterTest, RestartWithoutFileStartsFromScratch)
  {
    TempOutputDir dir;

    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 1);

    Core::IO::RuntimeCsvWriter writer(0, *output_control, "missing");
    writer.register_data_vector("alpha", 1, 1);

    EXPECT_NO_THROW(writer.write_data_to_file({{"alpha", {4.0}}}));
    EXPECT_THAT(read_lines(dir, "missing"), testing::ElementsAre("alpha", "4.0e+00"));
  }

  TEST(RuntimeCsvWriterWithTimeAndStepTest, RejectsMalformedRestartRow)
  {
    TempOutputDir dir;
    write_restart_file(dir, "malformed", {"step,time,energy", "not_a_number,0,1.0"});

    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 1);

    Core::IO::RuntimeCsvWriterWithTimeAndStep writer(0, *output_control, "malformed");
    writer.register_data_vector("energy", 1, 1);

    EXPECT_THROW(writer.write_data_to_file(0.5, 5, {{"energy", {6.0}}}), Core::Exception);
  }

  TEST(RuntimeCsvWriterTest, DoesNotWriteOnRanksOtherThanZero)
  {
    TempOutputDir dir;
    const std::unique_ptr<Core::IO::OutputControl> output_control = make_output_control(dir, 0);

    // there is a single process, so rank 1 is a rank that does not write
    Core::IO::RuntimeCsvWriter writer(1, *output_control, "no_write");
    writer.register_data_vector("a", 1, 16);

    EXPECT_NO_THROW(writer.write_data_to_file({{"a", {1.0}}}));

    EXPECT_FALSE(std::filesystem::exists(dir.file("no_write")));
  }
}  // namespace
