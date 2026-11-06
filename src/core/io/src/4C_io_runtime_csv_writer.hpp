// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_IO_RUNTIME_CSV_WRITER_HPP
#define FOUR_C_IO_RUNTIME_CSV_WRITER_HPP

/*-----------------------------------------------------------------------------------------------*/
/* headers */

#include "4C_config.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{
  class OutputControl;
  class RuntimeCsvWriterImpl;

  /*!
   * \brief How the values of a CsvColumn are formatted when they are written
   */
  enum class CsvNumberFormat : std::uint8_t
  {
    //! Plain integer without any decimals. The precision is ignored.
    Integer,
    //! Stream default notation with the precision given by the CsvColumn.
    DefaultFloat,
    //! Scientific notation with the precision given by the CsvColumn.
    Scientific
  };

  /*!
   * \brief Where a column shows up within a row
   */
  enum class CsvColumnOrder : std::uint8_t
  {
    /*!
     * Sorted by name and written after all \ref CsvColumnOrder::Fixed columns. This is the natural
     * order for result columns, whose names are only known once all of them are registered.
     */
    Sorted,
    /*!
     * Written in the order in which the columns are registered. This is the natural choice for
     * metadata that qualifies a row, such as the time and the time step, which must not be sorted
     * in among the results.
     */
    Fixed
  };

  /*!
   * \brief Description of a single column of a runtime csv file
   *
   * A column with more than one component is split into the sub-columns `<name>:0` up to
   * `<name>:<num_components-1>`.
   */
  struct CsvColumn
  {
    //! Name of the column. Must be unique within a writer.
    std::string name;
    //! Number of components of the column.
    unsigned int num_components{1};
    //! Formatting of the column's values.
    CsvNumberFormat format{CsvNumberFormat::Scientific};
    //! Digits after the point for \ref CsvNumberFormat::DefaultFloat, significant digits for
    //! \ref CsvNumberFormat::Scientific. Ignored for \ref CsvNumberFormat::Integer.
    int precision{6};
    //! Position of the column within a row.
    CsvColumnOrder order{CsvColumnOrder::Sorted};
  };

  /*!
   * \brief Write output in csv format at runtime in serial
   *
   * The writer is generic. Whatever columns are registered appear in the file, and a row is written
   * from the values of exactly those columns. See \ref RuntimeCsvWriterWithTimeAndStep for a
   * specialization that prepends a time step and a time to every row, which is a common use case.
   *
   * Output is only written on proc 0
   *
   * \note In the case of a restart, the content prior to the restart step is copied from the output
   * file the restart is read from. Which of the copied rows are relevant is decided by the
   * RestartPolicy given to the constructor. Without one, only the header is taken over.
   */
  class RuntimeCsvWriter
  {
   public:
    //! Values of all columns of a single row, keyed by column name
    using Data = std::map<std::string, std::vector<double>>;

    /*!
     * \brief Decides which data rows of a restart file are taken over into the new file
     *
     * The first line of the restart file is always taken as the header and is copied verbatim.
     * This is asked about every further line.
     *
     * \param[in] line         a data row of the restart file
     * \param[in] restart_step the step the run is restarted from
     * \return whether the row shall be copied
     */
    using RestartPolicy = std::function<bool(std::string_view line, int restart_step)>;

    /*!
     * \brief Create a writer
     *
     * \param[in] myrank          rank of this process; only rank 0 writes the file
     * \param[in] output_control  the output control providing the output and restart file prefixes
     * \param[in] output_name     name of the file, appended to the prefixes as
     *                            `<prefix>-<output_name>.csv`
     * \param[in] restart_policy  decides which rows of a restart file are taken over. If it is
     *                            empty, only the header is copied and a warning is issued.
     */
    RuntimeCsvWriter(int myrank, const Core::IO::OutputControl& output_control,
        const std::string& output_name, RestartPolicy restart_policy = {});

    virtual ~RuntimeCsvWriter();

    /*!
     * \brief Restart policy that takes over every row whose first field is a time step that is not
     *        greater than the restart step
     *
     * This is the natural policy for a file whose rows are ordered by time steps.
     */
    static RestartPolicy keep_rows_up_to_restart_step();

    /*!
     * \brief Register a column
     *
     * All columns must be registered before the first row is written, since the header of the file
     * is written right before that first row. Registering a name a second time is a no-op as long
     * as the description is identical.
     */
    void register_column(const CsvColumn& column) const;

    //! Register a result column that is written in scientific notation
    void register_data_vector(
        const std::string& name, unsigned int num_components, int precision) const;

    /*!
     * \brief Set the values of a registered column
     *
     * The values are written with the next call to \ref write_collected_data_to_file. In case of
     * sub columns, i.e., columns with multiple components, \p values must have as many entries as
     * the number of sub columns.
     */
    void append_data_vector(const std::string& name, const std::vector<double>& values) const;

    //! Write a single row built from the values of all registered columns
    void write_collected_data_to_file() const;

    /*!
     * \brief Write a single row built from \p data
     *
     * \p data must hold an entry for every registered column that is not
     * \ref CsvColumnOrder::Fixed, since the values of the latter are supplied by the writer itself
     * via \ref append_data_vector.
     */
    void write_data_to_file(const Data& data) const;

   private:
    std::unique_ptr<RuntimeCsvWriterImpl> implementation_;
  };

  /*!
   * \brief \ref RuntimeCsvWriter that prepends a time step and a time to every row
   *
   * The two leading columns are named `step` and `time`, are written in this order, and hold the
   * values passed to \ref reset_time_and_time_step or to \ref write_data_to_file. In the case of a
   * restart, the rows of the restart file are kept based on their time step.
   */
  class RuntimeCsvWriterWithTimeAndStep : public RuntimeCsvWriter
  {
   public:
    RuntimeCsvWriterWithTimeAndStep(
        int myrank, const Core::IO::OutputControl& output_control, const std::string& output_name);

    //! Set the values of the leading `time` and `step` columns
    void reset_time_and_time_step(double time, unsigned int timestep) const;

    //! Write a single row built from \p data, prefixed with \p time and \p timestep
    void write_data_to_file(double time, unsigned int timestep, const Data& data) const;
  };
}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE

#endif
