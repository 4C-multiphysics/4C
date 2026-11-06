// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_runtime_csv_writer.hpp"

#include "4C_io_control.hpp"
#include "4C_utils_exceptions.hpp"

#include <charconv>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <ranges>
#include <sstream>
#include <utility>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*/
namespace Core::IO
{
  //! Interface for the implementation of RuntimeCsvWriter based on an inheritance graph. The pure
  //! virtual interface class has two derived classes: One for proc 0 that does all the writing and
  //! one for all other procs that do nothing.
  class RuntimeCsvWriterImpl
  {
   public:
    virtual ~RuntimeCsvWriterImpl() = default;

    //! register a column
    virtual void register_column(const CsvColumn& column) = 0;

    //! set the values of a column
    virtual void append_data_vector(const std::string& name, const std::vector<double>& values) = 0;

    //! write one row with the currently collected values
    virtual void write_collected_data_to_file() = 0;

    //! write one row built from the given values
    virtual void write_data_to_file(const RuntimeCsvWriter::Data& data) = 0;
  };

  namespace
  {
    //! A registered column together with the values of the row that is currently being collected
    struct ColumnState
    {
      CsvColumn column;
      std::vector<double> values;
    };

    //! This class does the writing to the file. It is created on proc 0.
    class RuntimeCsvWriterProc0 : public RuntimeCsvWriterImpl
    {
     public:
      RuntimeCsvWriterProc0(const Core::IO::OutputControl& output_control,
          const std::string& output_name, RuntimeCsvWriter::RestartPolicy restart_policy)
          : output_file_(output_control.file_name() + "-" + output_name + ".csv"),
            restart_file_(output_control.restart_name() + "-" + output_name + ".csv"),
            restart_step_(output_control.restart_step()),
            restart_policy_(std::move(restart_policy))
      {
      }

      void register_column(const CsvColumn& column) override;

      void append_data_vector(const std::string& name, const std::vector<double>& values) override;

      void write_collected_data_to_file() override;

      void write_data_to_file(const RuntimeCsvWriter::Data& data) override;

     private:
      //! Create the output file and, in case of a restart, take over the rows prior to the restart
      void initialize_output_file();

      //! Write the header of the file, which precedes the very first row
      void write_file_header() const;

      //! All columns in the order in which they appear within a row
      [[nodiscard]] std::vector<const ColumnState*> columns_in_row_order() const;

      //! Write a single value with the formatting of its column
      static void write_value(std::ostream& out, const CsvColumn& column, double value);

      //! All registered columns, keyed by name. The order of the map is the position of a column
      //! that is not fixed.
      std::map<std::string, ColumnState> columns_;

      //! Names of the fixed columns in the order in which they were registered
      std::vector<std::string> fixed_columns_;

      //! Full path of the output file
      std::string output_file_;

      //! Full path of the file of the run we restart from
      std::string restart_file_;

      //! Step the run is restarted from
      const int restart_step_;

      //! Decides which of the rows of the restart file are taken over
      const RuntimeCsvWriter::RestartPolicy restart_policy_;

      //! Whether the output file was already set up
      bool initialized_{false};

      //! Whether the header was already written or taken over from a restart file
      bool header_written_{false};

      //! Whether we already complained about the absence of any column
      bool warned_no_columns_{false};
    };

    //! This class does intentionally nothing. It is created on all other procs except for proc 0.
    class RuntimeCsvWriterOtherProcs : public RuntimeCsvWriterImpl
    {
     public:
      RuntimeCsvWriterOtherProcs() = default;

      void register_column(const CsvColumn&) override {}

      void append_data_vector(const std::string&, const std::vector<double>&) override {}

      void write_collected_data_to_file() override {}

      void write_data_to_file(const RuntimeCsvWriter::Data&) override {}
    };
  }  // namespace

  void RuntimeCsvWriter::register_column(const CsvColumn& column) const
  {
    implementation_->register_column(column);
  }

  void RuntimeCsvWriter::register_data_vector(
      const std::string& name, const unsigned int num_components, const int precision) const
  {
    register_column(CsvColumn{.name = name,
        .num_components = num_components,
        .format = CsvNumberFormat::Scientific,
        .precision = precision,
        .order = CsvColumnOrder::Sorted});
  }

  void RuntimeCsvWriter::append_data_vector(
      const std::string& name, const std::vector<double>& values) const
  {
    implementation_->append_data_vector(name, values);
  }

  void RuntimeCsvWriter::write_collected_data_to_file() const
  {
    implementation_->write_collected_data_to_file();
  }

  void RuntimeCsvWriter::write_data_to_file(const Data& data) const
  {
    implementation_->write_data_to_file(data);
  }

  RuntimeCsvWriter::RestartPolicy RuntimeCsvWriter::keep_rows_up_to_restart_step()
  {
    return [](std::string_view line, const int restart_step)
    {
      // the step is the first field of a row
      const std::string_view first_field = line.substr(0, line.find(','));

      int step{};
      const auto [end, error] =
          std::from_chars(first_field.data(), first_field.data() + first_field.size(), step);

      if (error != std::errc{} or end != first_field.data() + first_field.size())
      {
        FOUR_C_THROW(
            "row '{}' of the restart file does not start with an integer step, so it cannot be "
            "matched against the restart step",
            line);
      }

      return step <= restart_step;
    };
  }

  RuntimeCsvWriter::RuntimeCsvWriter(const int myrank,
      const Core::IO::OutputControl& output_control, const std::string& output_name,
      RestartPolicy restart_policy)
  {
    if (myrank == 0)
    {
      implementation_ = std::make_unique<RuntimeCsvWriterProc0>(
          output_control, output_name, std::move(restart_policy));
    }
    else
    {
      implementation_ = std::make_unique<RuntimeCsvWriterOtherProcs>();
    }
  }

  RuntimeCsvWriter::~RuntimeCsvWriter() = default;

  void RuntimeCsvWriterProc0::register_column(const CsvColumn& column)
  {
    FOUR_C_ASSERT_ALWAYS(
        !column.name.empty(), "a column of output file '{}' has an empty name", output_file_);

    FOUR_C_ASSERT_ALWAYS(!header_written_,
        "column '{}' of output file '{}' was registered after the first row was written, but all "
        "columns have to be registered up front",
        column.name, output_file_);

    const auto [it, is_new] = columns_.try_emplace(column.name,
        ColumnState{.column = column, .values = std::vector<double>(column.num_components, 0.0)});

    if (is_new)
    {
      if (column.order == CsvColumnOrder::Fixed) fixed_columns_.push_back(column.name);
      return;
    }

    // registering the same column again is a no-op, but only if it is described identically
    const CsvColumn& registered = it->second.column;
    if (registered.num_components != column.num_components or registered.format != column.format or
        registered.precision != column.precision or registered.order != column.order)
    {
      FOUR_C_THROW(
          "column '{}' of output file '{}' is already registered as a column with {} components, "
          "format {} and precision {}, so it cannot be registered as a column with {} components, "
          "format {} and precision {}",
          column.name, output_file_, registered.num_components, static_cast<int>(registered.format),
          registered.precision, column.num_components, static_cast<int>(column.format),
          column.precision);
    }
  }

  void RuntimeCsvWriterProc0::append_data_vector(
      const std::string& name, const std::vector<double>& values)
  {
    const auto column = columns_.find(name);

    FOUR_C_ASSERT_ALWAYS(column != columns_.end(),
        "column '{}' of output file '{}' is not registered", name, output_file_);

    FOUR_C_ASSERT_ALWAYS(column->second.values.size() == values.size(),
        "column '{}' of output file '{}' was registered with {} components, but {} values were "
        "given",
        name, output_file_, column->second.values.size(), values.size());

    column->second.values = values;
  }

  void RuntimeCsvWriterProc0::write_data_to_file(const RuntimeCsvWriter::Data& data)
  {
    // every column that is not fixed must be given a value. The fixed ones carry over from their
    // most recent append_data_vector() call, since the writer fills them itself.
    std::string missing_columns;
    for (const auto& [column_name, column_state] : columns_)
    {
      if (column_state.column.order == CsvColumnOrder::Fixed or data.contains(column_name))
        continue;

      if (not missing_columns.empty()) missing_columns += ", ";
      missing_columns += column_name;
    }

    FOUR_C_ASSERT_ALWAYS(missing_columns.empty(),
        "no data given for the column(s) [{}] of output file '{}'", missing_columns, output_file_);


    for (const auto& data_name : data | std::views::keys)
    {
      if (not columns_.contains(data_name))
      {
        FOUR_C_THROW("column '{}' of output file '{}' is not registered", data_name, output_file_);
      }
    }

    for (const auto& [data_name, data_values] : data) append_data_vector(data_name, data_values);

    write_collected_data_to_file();
  }

  void RuntimeCsvWriterProc0::write_collected_data_to_file()
  {
    initialize_output_file();

    if (columns_.empty())
    {
      if (not warned_no_columns_)
      {
        std::cerr << "Warning: no columns are registered for output file '" << output_file_
                  << "', so no data is written.\n";
        warned_no_columns_ = true;
      }
      return;
    }

    if (not header_written_)
    {
      write_file_header();
      header_written_ = true;
    }

    std::stringstream row;
    bool is_first = true;
    for (const ColumnState* state : columns_in_row_order())
    {
      for (const double value : state->values)
      {
        if (not is_first) row << ",";
        is_first = false;
        write_value(row, state->column, value);
      }
    }

    std::ofstream output_file(output_file_, std::ios_base::app);
    output_file << row.str() << "\n";
    output_file.close();
  }

  void RuntimeCsvWriterProc0::initialize_output_file()
  {
    if (initialized_) return;
    initialized_ = true;

    // clear content
    {
      std::ofstream output_file(output_file_, std::ios_base::out | std::ios_base::trunc);
      output_file.close();
    }

    // without a restart the header is written together with the first row
    if (restart_step_ <= 0) return;

    std::ifstream restart_file(restart_file_, std::ios_base::in);
    if (not restart_file)
    {
      std::cerr << "Warning: restart file '" << restart_file_ << "' could not be found, so '"
                << output_file_ << "' starts from scratch without the rows of the run we restart "
                << "from.\n";
      return;
    }

    std::ofstream output_file(output_file_, std::ios_base::app);

    // the header of the restart file is the header of this run
    std::string line;
    if (std::getline(restart_file, line))
    {
      output_file << line << "\n";
      header_written_ = true;
    }

    // keep the rows that are still current. Without a policy nothing tells which of them are
    // still current, so only the header is taken over.
    if (restart_policy_)
    {
      while (std::getline(restart_file, line))
      {
        if (line.empty()) continue;
        if (restart_policy_(line, restart_step_)) output_file << line << "\n";
      }
    }
    else
    {
      std::cerr
          << "Warning: Core::IO::RuntimeCsvWriter has no policy to tell which rows of a restart "
             "file are still current, so only its header is taken over. Supply a "
             "RuntimeCsvWriter::RestartPolicy if the rows prior to the restart step shall be "
             "kept.\n";
    }

    output_file.close();
    restart_file.close();
  }

  std::vector<const ColumnState*> RuntimeCsvWriterProc0::columns_in_row_order() const
  {
    std::vector<const ColumnState*> row;
    row.reserve(columns_.size());

    // the fixed columns come first, in the order in which they were registered
    for (const std::string& name : fixed_columns_) row.push_back(&columns_.at(name));

    // the remaining columns follow in the order of the map, which is sorted by name
    for (const auto& column_state : columns_ | std::views::values)
    {
      if (column_state.column.order != CsvColumnOrder::Fixed) row.push_back(&column_state);
    }

    return row;
  }

  void RuntimeCsvWriterProc0::write_value(
      std::ostream& out, const CsvColumn& column, const double value)
  {
    switch (column.format)
    {
      case CsvNumberFormat::Integer:
        out << std::fixed << std::setprecision(0) << value;
        break;
      case CsvNumberFormat::DefaultFloat:
        out << std::defaultfloat << std::setprecision(column.precision) << value;
        break;
      case CsvNumberFormat::Scientific:
        out << std::scientific << std::setprecision(column.precision) << value;
        break;
    }
  }

  void RuntimeCsvWriterProc0::write_file_header() const
  {
    std::ofstream output_file(output_file_, std::ios_base::trunc);

    bool is_first = true;
    for (const ColumnState* state : columns_in_row_order())
    {
      const CsvColumn& column = state->column;

      for (unsigned int component = 0; component < column.num_components; ++component)
      {
        if (not is_first) output_file << ",";
        is_first = false;

        if (column.num_components == 1)
          output_file << column.name;
        else
          output_file << column.name << ":" << component;
      }
    }

    output_file << "\n";
    output_file.close();
  }

  RuntimeCsvWriterWithTimeAndStep::RuntimeCsvWriterWithTimeAndStep(const int myrank,
      const Core::IO::OutputControl& output_control, const std::string& output_name)
      : RuntimeCsvWriter(
            myrank, output_control, output_name, RuntimeCsvWriter::keep_rows_up_to_restart_step())
  {
    register_column(CsvColumn{.name = "step",
        .num_components = 1,
        .format = CsvNumberFormat::Integer,
        .precision = 0,
        .order = CsvColumnOrder::Fixed});
    register_column(CsvColumn{.name = "time",
        .num_components = 1,
        .format = CsvNumberFormat::DefaultFloat,
        .precision = 6,
        .order = CsvColumnOrder::Fixed});
  }

  void RuntimeCsvWriterWithTimeAndStep::reset_time_and_time_step(
      const double time, const unsigned int timestep) const
  {
    append_data_vector("step", {static_cast<double>(timestep)});
    append_data_vector("time", {time});
  }

  void RuntimeCsvWriterWithTimeAndStep::write_data_to_file(
      const double time, const unsigned int timestep, const Data& data) const
  {
    FOUR_C_ASSERT(!data.contains("step") && !data.contains("time"),
        "do not pass 'step' or 'time' in the data map to "
        "RuntimeCsvWriterWithTimeAndStep::write_data_to_file(); they are set via the time and "
        "timestep arguments");

    reset_time_and_time_step(time, timestep);
    RuntimeCsvWriter::write_data_to_file(data);
  }
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE
