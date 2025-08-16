// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config_revision.hpp"

#include "4C_io_input_file.hpp"

#include "4C_comm_mpi_utils.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_io_value_parser.hpp"
#include "4C_utils_string.hpp"

#include <filesystem>
#include <fstream>
#include <list>
#include <sstream>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace Core::IO
{

  namespace
  {
    std::string to_string(const ryml::csubstr str) { return std::string(str.data(), str.size()); };

    // Copy the content of the source node to the destination node. Useful to copy a node from one
    // tree to another.
    void deep_copy(ryml::NodeRef src, ryml::NodeRef dst)
    {
      dst.set_type(src.type());

      if (src.has_key())
      {
        dst.set_key(src.key());
      }
      if (src.has_val())
      {
        dst.set_val(src.val());
      }
      // Recursively copy children
      for (size_t i = 0; i < src.num_children(); ++i)
      {
        c4::yml::NodeRef src_child = src.child(i);
        c4::yml::NodeRef dst_child = dst.append_child();
        deep_copy(src_child, dst_child);
      }
    };

  }  // namespace

  namespace Internal
  {
    struct SectionContent;

    //! Helper for PIMPL idiom.
    class InputFileFragmentImpl
    {
     public:
      ryml::ConstNodeRef fragment;

      /**
       * Store a pointer to the whole section this fragment belongs to.
       */
      SectionContent* section;
    };

    /**
     * Internal storage for the content of a section.
     */
    struct SectionContent
    {
      struct YamlContent
      {
        //! Node in the tree of the associated InputFileImpl.
        ryml::NodeRef node;
      } content;

      /**
       * The actual content of the section sorted into fragments.
       */
      std::vector<InputFileFragmentImpl> fragments_impl;

      /**
       * The InputFile::Fragment is exposed to the user and contains a pointer to the actual
       * data stored in #fragments_impl.
       */
      std::vector<InputFile::Fragment> fragments;

      //! The file the section was read from.
      std::string file;

      /**
       * Depending on what is stored in this section, set up the Fragment objects to point to the
       * correct backend data.
       */
      void set_up_fragments()
      {
        // extract the fragments from the yaml node
        auto& node = content.node;
        fragments_impl.clear();
        fragments.clear();
        fragments_impl.reserve(node.num_children());
        fragments.reserve(node.num_children());

        FOUR_C_ASSERT(node.is_seq() || node.is_map(),
            "Section '{}' is neither a sequence nor a map.", to_string(node.key()));

        for (auto child : node.children())
        {
          auto& ref = fragments_impl.emplace_back(InputFileFragmentImpl{
              .fragment = child,
              .section = this,
          });
          fragments.emplace_back(&ref);
        }
      }

      SectionContent() = default;

      // Prevent copies: making a copy would invalidate the string views.
      SectionContent(const SectionContent&) = delete;
      SectionContent& operator=(const SectionContent&) = delete;

      SectionContent(SectionContent&&) = default;
      SectionContent& operator=(SectionContent&&) = default;
    };


    //! Helper for PIMPL idiom.
    class InputFileImpl
    {
     public:
      InputFileImpl(MPI_Comm comm) : comm_(comm), yaml_tree_(init_yaml_tree_with_exceptions())
      {
        // Root node is a map that stores the file paths as keys and the file content as values.
        yaml_tree_.rootref() |= ryml::MAP;
      }

      MPI_Comm comm_;

      std::unordered_map<std::string, SectionContent> content_by_section_;

      /**
       * Store the order in which the sections were read.
       */
      std::vector<std::string> section_order_;

      /**
       * This is the merged tree of all yaml input files combined, excluding any "INCLUDES"
       * sections. The data from different files will be stored under top-level keys corresponding
       * to the file paths.
       */
      ryml::Tree yaml_tree_;

      std::map<std::string, InputSpec> valid_sections_;
      std::vector<std::string> legacy_section_names_;

      bool is_section_known(const std::string& section_name) const
      {
        return is_hacky_function_section(section_name) ||
               (valid_sections_.find(section_name) != valid_sections_.end()) ||
               std::ranges::any_of(
                   legacy_section_names_, [&](const auto& name) { return name == section_name; }) ||
               (section_name == InputFile::description_section_name);
      }

      // The input for functions might introduce an arbitrary number of sections called
      // FUNCT<n>, where n is a number. As long as this input is not restructured, we need this
      // manual hack.
      bool is_hacky_function_section(const std::string& section_name) const
      {
        return section_name.starts_with("FUNCT") &&
               std::all_of(section_name.begin() + 5, section_name.end(),
                   [](const char c) { return std::isdigit(c); });
      }

      bool is_legacy_section(const std::string& section_name) const
      {
        return std::ranges::any_of(
            legacy_section_names_, [&](const auto& name) { return name == section_name; });
      }

      InputFile::FragmentIteratorRange in_section(const std::string& section_name) const
      {
        static const std::vector<InputFile::Fragment> empty;

        if (!content_by_section_.contains(section_name))
        {
          return std::views::all(empty);
        }

        // Take a const reference to the section content to match the return type.
        const auto& lines = content_by_section_.at(section_name).fragments;
        return std::views::all(lines);
      }
    };

  }  // namespace Internal

  namespace
  {
    //! The different ways we want to handle sections in the input file.
    enum class SectionType
    {
      //! A section that is read directly.
      normal,
      //! A section that mentions other files that are included and need to be read.
      include,
    };


    std::filesystem::path get_include_path(
        const std::string& include_line, const std::filesystem::path& current_file)
    {
      // Interpret the path as relative to the currently read file, if is not absolute
      std::filesystem::path included_file(include_line);
      if (!included_file.is_absolute())
      {
        included_file = current_file.parent_path() / included_file;
      }
      FOUR_C_ASSERT_ALWAYS(
          std::filesystem::status(included_file).type() == std::filesystem::file_type::regular,
          "Included file '{}' is not a regular file. Does the file exist?", included_file.string());
      return included_file;
    }

    std::vector<std::filesystem::path> read_yaml_content(
        const std::filesystem::path& file_path, Internal::InputFileImpl& input_file_impl)
    {
      std::vector<std::filesystem::path> included_files;

      // Read the whole file into a string ...
      std::ifstream file(file_path);
      std::ostringstream ss;
      ss << file.rdbuf();
      std::string file_content = ss.str();

      // ... and parse it into a new yaml tree in-place.
      ryml::Tree tmp_tree = init_yaml_tree_with_exceptions();
      ryml::parse_in_place(ryml::to_substr(file_content), &tmp_tree);

      auto tmp_tree_root_without_docs = tmp_tree.rootref();
      if (tmp_tree.rootref().is_stream())
      {
        FOUR_C_ASSERT_ALWAYS(tmp_tree.rootref().num_children() == 1,
            "The input file '{}' may only contain one YAML document.", file_path.string());
        tmp_tree_root_without_docs = tmp_tree.rootref().first_child();
      }
      FOUR_C_ASSERT_ALWAYS(tmp_tree_root_without_docs.is_map(),
          "The input file '{}' must contain a map as root node.", file_path.string());

      std::stringstream file_content_without_docs;
      file_content_without_docs << tmp_tree_root_without_docs;
      const std::string file_content_without_docs_str = file_content_without_docs.str();

      // Store the content of the file in the merged tree.
      ryml::NodeRef file_node = input_file_impl.yaml_tree_.rootref().append_child();
      {
        std::string file_path_str = file_path.string();
        file_node << ryml::key(file_path_str);
        ryml::parse_in_arena(ryml::to_csubstr(file_content_without_docs_str), file_node);
      }

      // Treat the special section "INCLUDES" to find more included files. Remove the section once
      // processed.
      if (file_node.has_child(InputFile::includes_section_name))
      {
        ryml::ConstNodeRef node = file_node[InputFile::includes_section_name];
        if (node.is_seq())
        {
          for (const auto& include_node : node)
          {
            included_files.emplace_back(get_include_path(to_string(include_node.val()), file_path));
          }
        }
        else if (!(node.has_val() && node.val_is_null()))
        {
          FOUR_C_THROW("INCLUDES section must contain a sequence of files.");
        }

        // Now we can drop the node containing the INCLUDES from the tree.
        input_file_impl.yaml_tree_.remove(node.id());
      }

      return included_files;
    }
  }  // namespace


  std::string_view InputFile::Fragment::get_as_dat_style_string() const
  {
    FOUR_C_ASSERT(pimpl_, "Implementation error: fragment is not initialized.");
    if (const auto node = pimpl_->fragment; node.is_val())
    {
      return std::string_view(node.val().data(), node.val().size());
    }
    else
    {
      FOUR_C_THROW(
          "Yaml node does not contain a string. This legacy function is only meant for strings.");
    }
  }


  InputFile::InputFile(std::map<std::string, InputSpec> valid_sections,
      std::vector<std::string> legacy_section_names, MPI_Comm comm)
      : pimpl_(std::make_unique<Internal::InputFileImpl>(comm))
  {
    pimpl_->valid_sections_ = std::move(valid_sections);
    pimpl_->legacy_section_names_ = std::move(legacy_section_names);

    FOUR_C_ASSERT_ALWAYS(!pimpl_->is_section_known("INCLUDES"),
        "Section 'INCLUDES' is a reserved section name with special meaning. Please choose a "
        "different name.");
    pimpl_->valid_sections_["INCLUDES"] =
        InputSpecBuilders::parameter<std::optional<std::vector<std::filesystem::path>>>("INCLUDES",
            {
                .description = "Path to files that should be included into this file. "
                               "The paths can be either absolute or relative to the file.",
            });
  }


  // Note: defaulted in implementation file to allow for use of incomplete type in PIMPL
  // unique_ptr.
  InputFile::~InputFile() = default;


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  std::filesystem::path InputFile::file_for_section(const std::string& section_name) const
  {
    auto it = pimpl_->content_by_section_.find(section_name);
    if (it == pimpl_->content_by_section_.end()) return std::filesystem::path{};
    return std::filesystem::path{it->second.file};
  }


  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  bool InputFile::has_section(const std::string& section_name) const
  {
    return pimpl_->content_by_section_.contains(section_name);
  }



  /*----------------------------------------------------------------------*/
  /*----------------------------------------------------------------------*/
  void InputFile::read(const std::filesystem::path& top_level_file)
  {
    if (Core::Communication::my_mpi_rank(pimpl_->comm_) == 0)
    {
      // the top_level_file must exist
      if (!std::filesystem::is_regular_file(top_level_file))
        FOUR_C_THROW("Input file '{}' does not exist.", top_level_file.string());

      // Start by "including" the top-level file.
      std::list<std::filesystem::path> included_files{top_level_file};

      // We use a hand-rolled loop here because the list keeps growing; thus we need to
      // continuously re-evaluate where the end of the list is.
      for (auto file_it = included_files.begin(); file_it != included_files.end(); ++file_it)
      {
        std::vector<std::filesystem::path> new_include_files = std::invoke(
            [&]
            {
              const auto file_extension = file_it->extension().string();
              // Note that json is valid yaml and we can read it with the yaml parser.
              if (file_extension == ".yaml" || file_extension == ".yml" ||
                  file_extension == ".json")
              {
                return read_yaml_content(*file_it, *pimpl_);
              }
              else
              {
                FOUR_C_THROW(
                    "Cannot infer format of input file '{}'. "
                    "Only .yaml, .yml, and .json are supported.",
                    file_it->string());
              }
            });

        // Check that the file is not included twice
        for (const auto& file : new_include_files)
        {
          if (std::ranges::find(included_files, file) != included_files.end())
          {
            FOUR_C_THROW(
                "File '{}' was already included before.\n Cycles are not allowed.", file.string());
          }
          else
          {
            included_files.emplace_back(file);
          }
        }
      }
    }

    // Communicate the yaml content
    {
      if (Core::Communication::my_mpi_rank(pimpl_->comm_) == 0)
      {
        ryml::Tree tree_with_small_sections = init_yaml_tree_with_exceptions();
        tree_with_small_sections.rootref() |= ryml::MAP;
        // Go through the tree and drop the legacy sections from the tree.
        for (auto file_node : pimpl_->yaml_tree_.rootref())
        {
          auto new_file_node = tree_with_small_sections.rootref().append_child();
          new_file_node.set_type(file_node.type());
          new_file_node.set_key(file_node.key());

          for (auto section_node : file_node.children())
          {
            if (!pimpl_->is_legacy_section(to_string(section_node.key())))
            {
              // Copy the node to the new tree.
              auto new_section_node = new_file_node.append_child();
              deep_copy(section_node, new_section_node);
            }
          }
        }

        auto serialized_tree = ryml::emitrs_yaml<std::string>(tree_with_small_sections);
        Core::Communication::broadcast(serialized_tree, /*root*/ 0, pimpl_->comm_);
      }
      else
      {
        std::string yaml_str;
        Core::Communication::broadcast(yaml_str, /*root*/ 0, pimpl_->comm_);
        pimpl_->yaml_tree_.reserve_arena(yaml_str.size());
        ryml::parse_in_arena(ryml::to_csubstr(yaml_str), pimpl_->yaml_tree_.rootref());
      }

      // Now all ranks agree on the yaml tree. Parse out the sections and fill the SectionContent.
      for (auto file_node : pimpl_->yaml_tree_.rootref())
      {
        for (auto node : file_node.children())
        {
          const std::string section_name = to_string(node.key());
          FOUR_C_ASSERT_ALWAYS(!pimpl_->content_by_section_.contains(section_name),
              "Section '{}' is defined more than once.", section_name);

          pimpl_->section_order_.emplace_back(section_name);
          Internal::SectionContent& content = pimpl_->content_by_section_[section_name];
          content.content = Internal::SectionContent::YamlContent{.node = node};
          content.file = to_string(file_node.key());
        }
      }
    }

    for (auto& [name, content] : pimpl_->content_by_section_)
    {
      if (pimpl_->is_hacky_function_section(name))
      {
        // Take the special spec of FUNCT<n> because it is the same for all function sections.
        // Make a copy and replace the name. This is a pretty insane hack and should be removed when
        // the input of the functions is restructured.
        auto spec = InputSpec(pimpl_->valid_sections_.at("FUNCT<n>").impl().clone());
        spec.impl().data.name = name;
        dynamic_cast<Internal::InputSpecTypeErasedImplementation<Internal::ListSpec>&>(spec.impl())
            .wrapped.name = name;
        pimpl_->valid_sections_.emplace(name, std::move(spec));
      }

      content.set_up_fragments();
    }

    // All content has been read. Now validate. In the first iteration of this new feature,
    // we only validate the section names, not the content.
    if (Core::Communication::my_mpi_rank(pimpl_->comm_) == 0)
    {
      for (const auto& [section_name, content] : pimpl_->content_by_section_)
      {
        if (!pimpl_->is_section_known(section_name))
        {
          FOUR_C_THROW("Section '{}' is not a valid section name.", section_name);
        }
      }
    }
  }


  InputFile::FragmentIteratorRange InputFile::in_section_rank_0_only(
      const std::string& section_name) const
  {
    FOUR_C_ASSERT_ALWAYS(pimpl_->is_legacy_section(section_name),
        "You tried to process section '{}' on rank 0 only, but this feature is meant for special "
        "legacy sections. Please use match_section() instead.",
        section_name);

    if (Core::Communication::my_mpi_rank(pimpl_->comm_) == 0 &&
        pimpl_->content_by_section_.contains(section_name))
    {
      const auto& lines = pimpl_->content_by_section_.at(section_name).fragments;
      return std::views::all(lines);
    }
    else
    {
      static const std::vector<Fragment> empty;
      return std::views::all(empty);
    }
  }


  void InputFile::match_section(
      const std::string& section_name, FourC::Core::IO::InputParameterContainer& container) const
  {
    if (!pimpl_->valid_sections_.contains(section_name))
    {
      if (section_name == description_section_name)
      {
        FOUR_C_THROW(
            "Tried to match section '{}' which is a special section that cannot be matched against "
            "any InputSpec.",
            section_name.c_str());
      }
      else if (std::ranges::find(pimpl_->legacy_section_names_, section_name) ==
               pimpl_->legacy_section_names_.end())
      {
        FOUR_C_THROW(
            "Tried to match section '{}' which is not a valid section name.", section_name);
      }
      else
      {
        FOUR_C_THROW(
            "Tried to match section '{}' but it is a legacy string section "
            "and cannot be matched against any InputSpec. The string lines "
            "in this section need manual parsing.",
            section_name.c_str());
      }
    }

    const auto& spec = pimpl_->valid_sections_.at(section_name);
    auto section_it = pimpl_->content_by_section_.find(section_name);
    if (section_it == pimpl_->content_by_section_.end())
    {
      if (spec.impl().has_default_value())
      {
        // Matching against an empty node will set the default values.
        auto tree = init_yaml_tree_with_exceptions();
        spec.match(ConstYamlNodeRef(tree.rootref(), ""), container);
        return;
      }
      else if (spec.impl().required())
      {
        FOUR_C_THROW("Required section '{}' not found in input file.", section_name);
      }
      else
      {
        return;
      }
    }

    // Section must be present, so we can match the content.
    spec.match(ConstYamlNodeRef(pimpl_->content_by_section_.at(section_name).content.node,
                   pimpl_->content_by_section_.at(section_name).file),
        container);
  }


  void InputFile::emit_metadata(YamlNodeRef node) const
  {
    auto& root = node.node;

    {
      auto sections = root["sections"];
      sections |= ryml::SEQ;
      for (const auto& [name, spec] : pimpl_->valid_sections_)
      {
        auto section = sections.append_child();
        YamlNodeRef spec_emitter{section, ""};
        spec.emit_metadata(spec_emitter);
      }
    }

    {
      auto legacy_string_sections = root["legacy_string_sections"];
      legacy_string_sections |= ryml::SEQ;
      for (const auto& name : pimpl_->legacy_section_names_)
      {
        legacy_string_sections.append_child() << name;
      }
    }
  }

  void InputFile::write_as_yaml(std::ostream& out, const std::filesystem::path& file_name) const
  {
    auto tree = init_yaml_tree_with_exceptions();
    tree.rootref() |= ryml::MAP;
    // Iterate the sections, parse them into a container, and emit the container data.
    InputParameterContainer container;

    // Write the yaml file in the same order as the input was read.
    for (const auto& section_name : pimpl_->section_order_)
    {
      container.clear();

      if (pimpl_->valid_sections_.contains(section_name))
      {
        auto& spec = pimpl_->valid_sections_.at(section_name);
        match_section(section_name, container);
        YamlNodeRef section_node{
            tree.rootref(), file_name.empty() ? std::filesystem::path{} : file_name};
        spec.emit(section_node, container);
      }
      else
      {
        // Emit the section as a sequence of strings. This also works for the special
        // description section.
        auto section = tree.rootref().append_child();
        section << ryml::key(section_name);
        section |= ryml::SEQ;
        for (const auto& line : pimpl_->in_section(section_name))
        {
          YamlNodeRef line_node{section.append_child(), file_name};
          emit_value_as_yaml(line_node, line.get_as_dat_style_string());
        }
      }
    }

    out << tree;
  }


  MPI_Comm InputFile::get_comm() const { return pimpl_->comm_; }


}  // namespace Core::IO

FOUR_C_NAMESPACE_CLOSE
