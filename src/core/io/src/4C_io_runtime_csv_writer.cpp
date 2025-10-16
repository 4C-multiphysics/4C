// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_runtime_csv_writer.hpp"

#include "4C_io.hpp"
#include "4C_io_control.hpp"
#include "4C_utils_exceptions.hpp"

#include <fstream>
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

    //! register data vector
    virtual void register_data_vector(
        const std::string& dataname, unsigned int numcomponents, int precision) = 0;

    //! reset current time and time step number
    virtual void reset_time_and_time_step(double time, unsigned int timestep) = 0;

    //! append data vector
    virtual void append_data_vector(
        const std::string& dataname, const std::vector<double>& datavalues) = 0;

    //! write one line to csv file. Data must have been passed via AppendDataVector()
    virtual void write_collected_data_to_file() = 0;

    //! write @p data to file at @p time and @p timestep
    virtual void write_data_to_file(double time, unsigned int timestep,
        const std::map<std::string, std::vector<double>>& data) const = 0;
  };

  //! This class does the writing to the file. It is created on proc 0.
  class RuntimeCsvWriterProc0 : public RuntimeCsvWriterImpl
  {
   public:
    explicit RuntimeCsvWriterProc0(
        const Core::IO::OutputControl& output_control, std::string outputname);

    void register_data_vector(
        const std::string& dataname, unsigned int numcomponents, int precision) override;

    void reset_time_and_time_step(const double time, const unsigned int timestep) override
    {
      time_ = time;
      timestep_ = timestep;
    }

    void append_data_vector(
        const std::string& dataname, const std::vector<double>& datavalues) override;

    void write_collected_data_to_file() override;

    void write_data_to_file(double time, unsigned int timestep,
        const std::map<std::string, std::vector<double>>& data) const override;

   private:
    void write_file_header() const;

    //! key: result name, entry: (data vector, precision)
    std::map<std::string, std::pair<std::vector<double>, int>> data_vectors_;

    //! full path to output file
    std::string fullpathoutputfile_;

    //! output name
    std::string outputname_;

    //! number of restart step
    const int restart_step_;

    //! current time
    double time_;

    //! current time step
    unsigned int timestep_;
  };

  //! This class does intentionally nothing. It is created on all other procs except for proc 0.
  class RuntimeCsvWriterOtherProcs : public RuntimeCsvWriterImpl
  {
   public:
    void register_data_vector(
        const std::string& dataname, unsigned int numcomponents, int precision) override
    {
    }

    void reset_time_and_time_step(const double time, const unsigned int timestep) override {}

    void append_data_vector(
        const std::string& dataname, const std::vector<double>& datavalues) override
    {
    }

    void write_collected_data_to_file() override {}

    void write_data_to_file(double time, unsigned int timestep,
        const std::map<std::string, std::vector<double>>& data) const override
    {
    }
  };

  void RuntimeCsvWriter::register_data_vector(
      const std::string& dataname, unsigned int numcomponents, int precision)
  {
    implementation_->register_data_vector(dataname, numcomponents, precision);
  }

  void RuntimeCsvWriter::reset_time_and_time_step(double time, unsigned int timestep)
  {
    implementation_->reset_time_and_time_step(time, timestep);
  }

  void RuntimeCsvWriter::append_data_vector(
      const std::string& dataname, const std::vector<double>& datavalues)
  {
    implementation_->append_data_vector(dataname, datavalues);
  }

  void RuntimeCsvWriter::write_collected_data_to_file()
  {
    implementation_->write_collected_data_to_file();
  }

  void RuntimeCsvWriter::write_data_to_file(const double time, const unsigned int timestep,
      const std::map<std::string, std::vector<double>>& data) const
  {
    implementation_->write_data_to_file(time, timestep, data);
  }

  RuntimeCsvWriter::RuntimeCsvWriter(
      int myrank, const Core::IO::OutputControl& output_control, std::string outputname)
  {
    if (myrank == 0)
    {
      implementation_ =
          std::make_unique<RuntimeCsvWriterProc0>(output_control, std::move(outputname));
    }
    else
      implementation_ = std::make_unique<RuntimeCsvWriterOtherProcs>();
  }

  RuntimeCsvWriter::~RuntimeCsvWriter() = default;

  RuntimeCsvWriterProc0::RuntimeCsvWriterProc0(
      const Core::IO::OutputControl& output_control, std::string outputname)
      : outputname_(std::move(outputname)),
        restart_step_(output_control.restart_step()),
        time_(0.0),
        timestep_(-1)
  {
    // determine full path to output prefix
    const std::string fullpathoutputprefix = output_control.file_name();

    // set full path to output file
    fullpathoutputfile_ = fullpathoutputprefix + "-" + outputname_ + ".csv";

    // clear content
    {
      std::ofstream outputfile(fullpathoutputfile_, std::ios_base::out | std::ios_base::trunc);
      outputfile.close();
    }

    // in case of restart copy content of restart file to output file prior to restart time step
    if (restart_step_)
    {
      // determine full path to restart prefix
      const std::string fullpathrestartprefix = output_control.restart_name();

      // set full path to restart file
      const std::string fullpathrestartfile = fullpathrestartprefix + "-" + outputname_ + ".csv";

      std::ifstream restartfile(fullpathrestartfile, std::ios_base::out);

      // check if file was found
      if (not restartfile)
        FOUR_C_THROW("restart file '{}' could not be found", fullpathrestartfile);

      std::stringstream sectionpriorrestart;

      // loop over lines of restart file
      std::string line;
      while (std::getline(restartfile, line))
      {
        std::istringstream sline(line);

        // get first word of line
        std::string word;
        std::getline(sline, word, ',');

        if (word == "step")
        {
          sectionpriorrestart << line << "\n";
          continue;
        }

        // get time step of current line
        const int timestep = std::stoi(word);

        // append current line
        if (timestep <= restart_step_) sectionpriorrestart << line << "\n";
      }

      restartfile.close();

      // write to file
      std::ofstream outputfile(fullpathoutputfile_, std::ios_base::app);
      outputfile << sectionpriorrestart.str();
      outputfile.close();
    }
  }

  void RuntimeCsvWriterProc0::register_data_vector(
      const std::string& dataname, const unsigned int numcomponents, const int precision)
  {
    data_vectors_[dataname] = {std::vector<double>(numcomponents, 0.0), precision};
    if (!restart_step_) write_file_header();
  }

  void RuntimeCsvWriterProc0::append_data_vector(
      const std::string& dataname, const std::vector<double>& datavalues)
  {
    FOUR_C_ASSERT(data_vectors_.contains(dataname), "data vector '{}' not registered!", dataname);

    FOUR_C_ASSERT((data_vectors_[dataname].first).size() == datavalues.size(),
        "size of data vector '{}' changed!", dataname);

    data_vectors_[dataname].first = datavalues;
  }

  void RuntimeCsvWriterProc0::write_collected_data_to_file()
  {
    if (data_vectors_.empty()) FOUR_C_THROW("no data vectors registered!");

    std::ofstream outputfile(fullpathoutputfile_, std::ios_base::app);
    outputfile << timestep_ << "," << time_;

    for (const auto& data : data_vectors_)
    {
      const std::vector<double>& data_vector = (data.second).first;
      const int precision = (data.second).second;

      outputfile << std::setprecision(precision) << std::scientific;

      for (const auto& data_vector_component : data_vector)
        outputfile << "," << data_vector_component;
    }

    outputfile << "\n";
    outputfile.close();
  }

  void RuntimeCsvWriterProc0::write_data_to_file(const double time, const unsigned int timestep,
      const std::map<std::string, std::vector<double>>& data) const
  {
    if (data_vectors_.empty()) FOUR_C_THROW("no data vectors registered!");

    std::ofstream outputfile(fullpathoutputfile_, std::ios_base::app);
    outputfile << timestep << "," << time;

    for (const auto& [data_name, data_vector] : data)
    {
      FOUR_C_ASSERT(
          data_vectors_.contains(data_name), "data vector '{}' not registered!", data_name);

      FOUR_C_ASSERT((data_vectors_.at(data_name).first).size() == data_vector.size(),
          "size of data vector '{}' changed!", data_name);

      const int precision = (data_vectors_.at(data_name).second);

      outputfile << std::setprecision(precision) << std::scientific;

      for (const auto& data_vector_component : data_vector)
        outputfile << "," << data_vector_component;
    }

    outputfile << "\n";
    outputfile.close();
  }

  void RuntimeCsvWriterProc0::write_file_header() const
  {
    std::ofstream outputfile(fullpathoutputfile_, std::ios_base::trunc);
    outputfile << "step,time";

    for (const auto& data : data_vectors_)
    {
      const std::string& dataname = data.first;
      const unsigned int numcomponents = ((data.second).first).size();

      if (numcomponents == 1)
        outputfile << "," << dataname;
      else
        for (unsigned int i = 0; i < numcomponents; ++i) outputfile << "," << dataname << ":" << i;
    }

    outputfile << "\n";
    outputfile.close();
  }
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE
