/*----------------------------------------------------------------------*/
/*! \file

\brief Any data container based on vectors of std::any data.


\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_UTILS_ANY_DATA_CONTAINER_HPP
#define FOUR_C_UTILS_ANY_DATA_CONTAINER_HPP

#include "4C_config.hpp"

#include "4C_utils_demangle.hpp"
#include "4C_utils_exceptions.hpp"

#include <any>
#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// forward declaration
namespace CONTACT
{
  namespace Aug
  {
    template <typename EnumClass>
    class TimeMonitor;
  }  // namespace Aug
}  // namespace CONTACT

namespace Core::Gen
{
  /** \brief Data container of any content
   *
   *  The AnyDataContainer is meant as a container class of any content
   *  which can be used to exchange data of any type and any quantity over
   *  different function calls and/or class hierarchies without the need to
   *  adapt or extend your code. The container is NOT meant as a collective data
   *  storage but more as a vehicle to transport your data to the place you
   *  need it.
   *
   *  \author hiermeier \date 12/17 */
  class AnyDataContainer
  {
    //! alias templates
    //! @{

    //! alias for an unordered_map
    template <typename... Ts>
    using UMap = std::unordered_map<Ts...>;

    //! alias for a vector
    template <typename... Ts>
    using Vec = std::vector<Ts...>;

    //! alias for the time monitor
    template <typename EnumClass>
    using TimeMonitor = CONTACT::Aug::TimeMonitor<EnumClass>;

    //! @}

    /// internal data container
    struct AnyData
    {
      /// accessor
      [[nodiscard]] const std::any& try_get() const
      {
        if (!data_.has_value()) FOUR_C_THROW("The data is empty!");

        return data_;
      }

      /// set any data and perform a sanity check
      template <typename T>
      void try_set(const T& data)
      {
        if (data_.has_value())
        {
          FOUR_C_THROW(
              "There are already data:\n%s\nAre you sure, that you want "
              "to overwrite them? If yes: Clear the content first. Best practice "
              "is to clear the content right after you finished the "
              "respective task.",
              Core::Utils::try_demangle(data_.type().name()).c_str());
        }
        else
          data_ = data;
      }

     private:
      /// actual stored data (of any type)
      std::any data_;
    };

   public:
    /// supported (more specific) data types
    enum class DataType
    {
      vague,
      any,
      vector,
      unordered_map,
      time_monitor
    };

   public:
    /// @name any data
    /// @{

    template <typename T>
    void set(const T* data, const unsigned id = 0)
    {
      set_data<T, DataType::any>(data, id);
    }

    template <typename T>
    T* get(const unsigned id = 0)
    {
      return const_cast<T*>(get_data<T, DataType::any>(id));
    }

    template <typename T>
    const T* get(const unsigned id = 0) const
    {
      return get_data<T, DataType::any>(id);
    }

    /// @}

    /// @name std::vector
    /// @{

    template <typename... Ts>
    void set_vector(const Vec<Ts...>* unordered_map, const unsigned id = 0)
    {
      set_data<Vec<Ts...>, DataType::vector>(unordered_map, id);
    }

    template <typename... Ts>
    Vec<Ts...>* get_vector(const unsigned id = 0)
    {
      return const_cast<Vec<Ts...>*>(get_data<Vec<Ts...>, DataType::vector>(id));
    }

    template <typename... Ts>
    const Vec<Ts...>* get_vector(const unsigned id = 0) const
    {
      return get_data<Vec<Ts...>, DataType::vector>(id);
    }

    /// @}

    /// @name std::unordered_map
    /// @{

    template <typename... Ts>
    void set_unordered_map(const UMap<Ts...>* unordered_map, const unsigned id = 0)
    {
      set_data<UMap<Ts...>, DataType::unordered_map>(unordered_map, id);
    }

    template <typename... Ts>
    UMap<Ts...>* get_unordered_map(const unsigned id = 0)
    {
      return const_cast<UMap<Ts...>*>(get_data<UMap<Ts...>, DataType::unordered_map>(id));
    }

    template <typename... Ts>
    const UMap<Ts...>* get_unordered_map(const unsigned id = 0) const
    {
      return get_data<UMap<Ts...>, DataType::unordered_map>(id);
    }

    /// @}

    /// @name Time monitoring
    /// @{

    template <typename EnumClass>
    void set_timer(const CONTACT::Aug::TimeMonitor<EnumClass>* timer, const unsigned id = 0)
    {
      set_data<TimeMonitor<EnumClass>, DataType::time_monitor>(timer, id);
    }

    template <typename EnumClass>
    TimeMonitor<EnumClass>* get_timer(const unsigned id = 0)
    {
      return const_cast<TimeMonitor<EnumClass>*>(
          get_data<TimeMonitor<EnumClass>, DataType::time_monitor>(id));
    }

    template <typename EnumClass>
    const TimeMonitor<EnumClass>* get_timer(const unsigned id = 0) const
    {
      return get_data<TimeMonitor<EnumClass>, DataType::time_monitor>(id);
    }

    /// @}

    /// @name general methods
    /// @{

    /// clear an entry in the respective container
    void clear_entry(const DataType type, const int id)
    {
      switch (type)
      {
        case DataType::vector:
        {
          clear(vector_data_, id);

          break;
        }
        case DataType::unordered_map:
        {
          clear(unordered_map_data_, id);

          break;
        }
        case DataType::time_monitor:
        {
          clear(time_monitor_data_, id);

          break;
        }
        case DataType::any:
        {
          clear(any_data_, id);

          break;
        }
        default:
        {
          FOUR_C_THROW("Unsupported DataType!");
          exit(EXIT_FAILURE);
        }
      }
    }

    // clear all entries in the respective container
    void clear_all(const DataType type) { clear_entry(type, -1); }

    /// @}

   private:
    /// helper function to clear content
    void clear(std::vector<AnyData>& any_data_vec, const int id)
    {
      // clear all entries
      if (id < 0)
      {
        for (auto& any_data : any_data_vec) any_data = AnyData{};

        return;
      }

      // direct return if the id exceeds the vector size
      if (id >= static_cast<int>(any_data_vec.size())) return;

      // clear only one entry
      any_data_vec[id] = AnyData{};
    }

    /// pack and set the data pointer
    template <typename T, DataType type>
    void set_data(const T* data, const unsigned id)
    {
      std::any any_data(data);
      set_any_data<type>(any_data, id);
    }

    /// set the data in the respective container
    template <DataType type>
    void set_any_data(const std::any& any_data, const unsigned id)
    {
      switch (type)
      {
        case DataType::vector:
        {
          add_to_any_data_vec(any_data, id, vector_data_);

          break;
        }
        case DataType::unordered_map:
        {
          add_to_any_data_vec(any_data, id, unordered_map_data_);

          break;
        }
        case DataType::time_monitor:
        {
          add_to_any_data_vec(any_data, id, time_monitor_data_);

          break;
        }
        case DataType::any:
        {
          add_to_any_data_vec(any_data, id, any_data_);

          break;
        }
        default:
          FOUR_C_THROW("Unsupported DataType!");
      }
    }

    /// access the data and cast the any pointer
    template <typename T, DataType type>
    const T* get_data(const unsigned id) const
    {
      const std::any& any_data = get_any_data<type>(id);
      return std::any_cast<const T*>(any_data);
    }

    /// access the data
    template <DataType type>
    const std::any& get_any_data(const unsigned id) const
    {
      switch (type)
      {
        case DataType::vector:
        {
          return get_from_any_data_vec(id, vector_data_);
        }
        case DataType::unordered_map:
        {
          return get_from_any_data_vec(id, unordered_map_data_);
        }
        case DataType::time_monitor:
        {
          return get_from_any_data_vec(id, time_monitor_data_);
        }
        case DataType::any:
        {
          return get_from_any_data_vec(id, any_data_);
        }
        default:
        {
          FOUR_C_THROW("Unsupported DataType!");
          exit(EXIT_FAILURE);
        }
      }
    }

    /// add to any data vector
    void add_to_any_data_vec(
        const std::any& any_data, const unsigned id, std::vector<AnyData>& any_data_vec) const
    {
      if (any_data_vec.size() <= id) any_data_vec.resize(id + 1);

      AnyData& data_id = any_data_vec[id];
      data_id.try_set(any_data);
    }

    /// access content of any data vector
    inline const std::any& get_from_any_data_vec(
        const unsigned id, const std::vector<AnyData>& any_data_vec) const
    {
      if (id >= any_data_vec.size())
        FOUR_C_THROW(
            "Requested ID #%d exceeds the AnyData vector size (=%d).", id, any_data_vec.size());


      return any_data_vec[id].try_get();
    }

   private:
    /// specific container for vector data
    std::vector<AnyData> vector_data_;

    /// specific container for unordered map data
    std::vector<AnyData> unordered_map_data_;

    /// specific container for time monitor data
    std::vector<AnyData> time_monitor_data_;

    /// container for any data
    std::vector<AnyData> any_data_;
  };
}  // namespace Core::Gen


FOUR_C_NAMESPACE_CLOSE

#endif
