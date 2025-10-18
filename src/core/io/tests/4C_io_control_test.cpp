// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include "4C_io_control.hpp"

namespace
{
  using namespace FourC;

  TEST(ControlFileWriterTest, Metadata)
  {
    std::stringstream stream;
    Core::IO::ControlFileWriter writer(true, stream);
    writer.write_metadata_header();
    writer.end_all_groups_and_flush();

    EXPECT_THAT(stream.str(), testing::HasSubstr("- metadata:"));
    EXPECT_THAT(stream.str(), testing::HasSubstr("    created_by:"));
    EXPECT_THAT(stream.str(), testing::HasSubstr("    host:"));
    EXPECT_THAT(stream.str(), testing::HasSubstr("    time:"));
    EXPECT_THAT(stream.str(), testing::HasSubstr("    sha:"));
    EXPECT_THAT(stream.str(), testing::HasSubstr("    version:"));
  }

  TEST(ControlFileWriterTest, SimpleGroup)
  {
    std::stringstream stream;
    Core::IO::ControlFileWriter writer(true, stream);
    writer.start_group("group")
        .write("string", "value")
        .write("int", 1)
        .write("double", 2.0)
        .end_group();

    EXPECT_EQ(stream.str(), R"(- group:
    string: "value"
    int: 1
    double: 2

)");
  }

  TEST(ControlFileWriterTest, UnclosedGroupFlushedAtEnd)
  {
    std::stringstream stream;

    {
      Core::IO::ControlFileWriter writer(true, stream);
      writer.start_group("group").write("string", "value").write("int", 1).write("double", 2.0);

      // Writer goes out of scope here and should flush the unclosed group.
    }

    EXPECT_EQ(stream.str(), R"(- group:
    string: "value"
    int: 1
    double: 2

)");
  }

  TEST(ControlFileWriterTest, NestedGroups)
  {
    std::stringstream stream;
    {
      Core::IO::ControlFileWriter writer(true, stream);
      writer.start_group("group1")
          .write("string", "value")
          .start_group("group2")
          .write("int", 1)
          .start_group("group3")
          .write("double", 2.0)
          .end_group()
          .end_group()
          .end_group();
    }

    EXPECT_EQ(stream.str(), R"(- group1:
    string: "value"
    group2:
      int: 1
      group3:
        double: 2

)");
  }


  TEST(ControlFileWriterTest, NestedGroupsFlushedAtEnd)
  {
    std::stringstream stream;
    {
      Core::IO::ControlFileWriter writer(true, stream);
      writer.start_group("group1")
          .write("string", "value")
          .start_group("group2")
          .write("int", 1)
          .start_group("group3")
          .write("double", 2.0);
    }

    EXPECT_EQ(stream.str(), R"(- group1:
    string: "value"
    group2:
      int: 1
      group3:
        double: 2

)");
  }

  TEST(ControlFileWriterTest, NoWrite)
  {
    std::stringstream stream;
    Core::IO::ControlFileWriter writer(false, stream);
    writer.write_metadata_header();
    writer.start_group("group")
        .write("string", "value")
        .write("int", 1)
        .write("double", 2.0)
        .end_group();

    // Expect an empty string since no writing is performed
    EXPECT_EQ("", stream.str());
  }

  TEST(ControlFileWriterTest, FlushWithoutGroups)
  {
    std::stringstream stream;
    {
      Core::IO::ControlFileWriter writer(true, stream);
      writer.end_all_groups_and_flush();
      writer.end_all_groups_and_flush();
    }

    EXPECT_EQ("", stream.str());
  }

}  // namespace