// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_symbolic_expression.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils::SymbolicExpressionDetails
{
  char Lexer::get_next()
  {
    if (pos_ < funct_.length())
    {
      return funct_[pos_++];
    }
    else
    {
      return EOF;
    }
  }

  char Lexer::peek() const { return (pos_ < funct_.length()) ? funct_[pos_] : EOF; }


  Lexer::Lexer(std::string_view funct) : funct_(funct), pos_(0) {}


  std::vector<Token> Lexer::tokenize()
  {
    std::vector<Token> tokens;
    while (true)
    {
      tokens.emplace_back(advance());
      if (tokens.back().type == tok_done)
      {
        break;
      }
    }
    return tokens;
  }

  Token Lexer::advance()
  {
    for (;;)
    {
      char c = get_next();
      auto start = pos_ - 1;
      const auto make_token = [&](TokenType type)
      { return Token{type, funct_.substr(start, pos_ - start), start}; };

      //! skip whitespace
      if ((c == ' ') || (c == '\t'))
      {
        continue;
      }

      if (c == '\n')
      {
        throw_at_pos(pos_ - 1, "Unexpected newline.");
      }

      if (c == EOF)
      {
        return make_token(tok_done);
      }

      if (isdigit(c))
      {
        while (isdigit(c))
        {
          c = get_next();
        }
        if ((c != '.') && (c != 'E') && (c != 'e'))
        {
          if (c != EOF) pos_--;
          return make_token(tok_real);
        }
        if (c == '.')
        {
          c = get_next();
          if (isdigit(c))
          {
            while (isdigit(c))
            {
              c = get_next();
            }
          }
          else
          {
            throw_at_pos(pos_ - 1, "No digits after decimal.");
          }
        }
        if ((c == 'E') || (c == 'e'))
        {
          c = get_next();
          if ((c == '-') || (c == '+'))
          {
            c = get_next();
          }
          if (isdigit(c))
          {
            while (isdigit(c))
            {
              c = get_next();
            }
          }
          else
          {
            throw_at_pos(pos_ - 1, "No digits in exponent.");
          }
        }
        if (c != EOF) pos_--;
        return make_token(tok_real);
      }

      if (isalpha(c) || (c == '_'))
      {
        while (isalnum(c) || (c == '_'))
        {
          c = get_next();
        }
        if (c != EOF) pos_--;
        return make_token(tok_name);
      }
      switch (c)
      {
        case '+':
          return make_token(tok_add);
        case '-':
          return make_token(tok_sub);
        case '*':
          return make_token(tok_mul);
        case '/':
          return make_token(tok_div);
        case '^':
          return make_token(tok_pow);
        case '(':
          return make_token(tok_lpar);
        case ')':
          return make_token(tok_rpar);
        case ',':
          return make_token(tok_comma);
        case '>':
        {
          if (peek() == '=')
          {
            get_next();
            return make_token(tok_ge);
          }
          else
          {
            return make_token(tok_gt);
          }
        }
        case '<':
        {
          if (peek() == '=')
          {
            get_next();
            return make_token(tok_le);
          }
          else
          {
            return make_token(tok_lt);
          }
        }
        case '=':
        {
          char next = get_next();
          if (next == '=')
          {
            return make_token(tok_eq);
          }
          throw_at_pos(pos_ - 1, "Unexpected character.");
        }
        case '&':
        {
          char next = get_next();
          if (next == '&')
          {
            return make_token(tok_and);
          }
          throw_at_pos(pos_ - 1, "Unexpected character.");
        }
        case '|':
        {
          char next = get_next();
          if (next == '|')
          {
            return make_token(tok_or);
          }
          throw_at_pos(pos_ - 1, "Unexpected character.");
        }
        case '!':
        {
          if (peek() == '=')
          {
            get_next();
            return make_token(tok_ne);
          }
          else
          {
            return make_token(tok_bang);
          }
        }
        default:
          throw_at_pos(pos_ - 1, "Unknown character.");
      }
    }
  }


  void Lexer::throw_at_pos(std::size_t pos, const std::string& msg) const
  {
    // Construct a string with the position of the error marked by a caret.
    FOUR_C_THROW(
        "Error while reading:\n"
        "{}\n"
        "{:>{}}\n"
        "{:>{}}{}",
        funct_, "^", pos + 1, " ", pos, msg);
  }


  Parser::Parser(
      std::string expression, const std::vector<std::string_view>& allowed_variable_names)
      : allowed_variable_names_(allowed_variable_names),
        expression_(std::move(expression)),
        allow_any_variable_(allowed_variable_names_.empty())
  {
    // Run the lexer to tokenize the input expression. This will throw on any lexical errors.
    tokens_ = Lexer(expression_).tokenize();

    // Now parse the tokens into a syntax tree. This will throw on any syntax errors.
    parse();
  }


  void Parser::parse()
  {
    //! create syntax tree equivalent to funct
    root_ = parse_expr();

    // check if parsing ended before processing the entire string
    if (peek().type != tok_done)
    {
      FOUR_C_THROW("Invalid syntax: The remaining string '{}' is not parsed.",
          std::string_view(expression_.data() + peek().start_pos));
    }
  }


  IndexType Parser::parse_expr() { return parse_logical_or(); }


  IndexType Parser::parse_logical_or()
  {
    IndexType lhs = parse_logical_and();
    for (;;)
    {
      if (match(tok_or))
      {
        IndexType rhs = parse_logical_and();
        lhs = create_node(NodeType::binary_function,
            {.binary_function = BinaryFunctionType::logical_or}, lhs, rhs);
      }
      else
      {
        break;
      }
    }
    return lhs;
  }


  IndexType Parser::parse_logical_and()
  {
    IndexType lhs = parse_equality();
    for (;;)
    {
      if (match(tok_and))
      {
        IndexType rhs = parse_equality();
        lhs = create_node(NodeType::binary_function,
            {.binary_function = BinaryFunctionType::logical_and}, lhs, rhs);
      }
      else
      {
        break;
      }
    }
    return lhs;
  }


  IndexType Parser::parse_equality()
  {
    IndexType lhs = parse_comparison();
    for (;;)
    {
      if (match(tok_eq))
      {
        IndexType rhs = parse_comparison();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::eq}, lhs, rhs);
      }
      else if (match(tok_ne))
      {
        IndexType rhs = parse_comparison();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::ne}, lhs, rhs);
      }
      else
      {
        break;
      }
    }
    return lhs;
  }


  IndexType Parser::parse_comparison()
  {
    IndexType lhs = parse_term();
    for (;;)
    {
      if (match(tok_gt))
      {
        IndexType rhs = parse_term();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::gt}, lhs, rhs);
      }
      else if (match(tok_lt))
      {
        IndexType rhs = parse_term();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::lt}, lhs, rhs);
      }
      else if (match(tok_ge))
      {
        IndexType rhs = parse_term();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::ge}, lhs, rhs);
      }
      else if (match(tok_le))
      {
        IndexType rhs = parse_term();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::le}, lhs, rhs);
      }
      else
      {
        break;
      }
    }

    return lhs;
  }



  IndexType Parser::parse_term()
  {
    IndexType lhs = parse_factor();
    for (;;)
    {
      if (match(tok_add))
      {
        IndexType rhs = parse_factor();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::add}, lhs, rhs);
      }
      else if (match(tok_sub))
      {
        IndexType rhs = parse_factor();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::sub}, lhs, rhs);
      }
      else
      {
        break;
      }
    }

    return lhs;
  }


  IndexType Parser::parse_factor()
  {
    IndexType lhs = parse_pow();
    for (;;)
    {
      if (match(tok_mul))
      {
        IndexType rhs = parse_pow();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::mul}, lhs, rhs);
      }
      else if (match(tok_div))
      {
        IndexType rhs = parse_pow();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::div}, lhs, rhs);
      }
      else
      {
        break;
      }
    }

    return lhs;
  }


  IndexType Parser::parse_pow()
  {
    IndexType lhs = parse_primary();
    for (;;)
    {
      if (match(tok_pow))
      {
        IndexType rhs = parse_primary();
        lhs = create_node(
            NodeType::binary_function, {.binary_function = BinaryFunctionType::pow}, lhs, rhs);
      }
      else
      {
        break;
      }
    }

    return lhs;
  }


  IndexType Parser::parse_primary()
  {
    if (match(tok_lpar))
    {
      auto id = parse_expr();
      consume(tok_rpar, "Expected closing parenthesis.");
      return id;
    }
    else if (match(tok_real))
    {
      auto token = previous();
      return create_node(NodeType::number, {.number = std::strtod(token.text.data(), nullptr)});
    }
    else if (match(tok_bang))
    {
      return create_node(NodeType::unary_function,
          {.unary_function = UnaryFunctionType::logical_not}, parse_primary());
    }
    else if (match(tok_sub))
    {
      // This is a unary minus operator.
      IndexType rhs = parse_pow();
      auto& rhs_node = node_arena_[rhs];
      if (rhs_node.type == NodeType::number)
      {
        rhs_node.as.number *= -1;
        return rhs;
      }
      else
      {
        auto prefactor = create_node(NodeType::number, {.number = -1});
        return create_node(NodeType::binary_function, {.binary_function = BinaryFunctionType::mul},
            prefactor, rhs);
      }
    }
    else if (match(tok_name))
    {
      auto token = previous();
      if (token.text == "pi")
      {
        return create_node(NodeType::number, {.number = M_PI});
      }
      else
      {
        auto maybe_unary_function = EnumTools::enum_cast<UnaryFunctionType>(token.text);
        if (maybe_unary_function)
        {
          consume(tok_lpar, "Expected '(' after function name.");

          auto id = create_node(
              NodeType::unary_function, {.unary_function = *maybe_unary_function}, parse_expr());

          consume(tok_rpar, "Expected closing parenthesis.");
          return id;
        }

        auto maybe_binary_function = EnumTools::enum_cast<BinaryFunctionType>(token.text);
        if (maybe_binary_function)
        {
          consume(tok_lpar, "Expected '(' after function name.");
          auto first_arg = parse_expr();
          consume(tok_comma, "Expected comma between function arguments.");
          auto second_arg = parse_expr();
          consume(tok_rpar, "Expected closing parenthesis.");

          return create_node(NodeType::binary_function, {.binary_function = *maybe_binary_function},
              first_arg, second_arg);
        }

        // Some other identifier that is treated as a variable.
        {
          return create_node(NodeType::variable, {.var_index = create_variable(token.text)});
        }
      }
    }
    else
    {
      throw_syntax_error("Expected a primary expression.", peek().start_pos);
    }
  }


  Token Parser::advance()
  {
    return current_token_index_ < tokens_.size() ? tokens_[current_token_index_++] : tokens_.back();
  }


  Token Parser::peek() const
  {
    return current_token_index_ < tokens_.size() ? tokens_[current_token_index_] : tokens_.back();
  }


  Token Parser::previous() const
  {
    FOUR_C_ASSERT(current_token_index_ > 0, "Internal error: no previous token available.");
    return tokens_[current_token_index_ - 1];
  }


  bool Parser::match(TokenType type)
  {
    if (peek().type == type)
    {
      advance();
      return true;
    }
    return false;
  }


  void Parser::consume(TokenType expected, const std::string& message)
  {
    if (!match(expected))
    {
      throw_syntax_error(message, peek().start_pos);
    }
  }


  IndexType Parser::create_node(NodeType type, Value value, IndexType lhs, IndexType rhs)
  {
    FOUR_C_ASSERT_ALWAYS(node_arena_.size() < invalid_index,
        "Could not parse expression '{}'\nThis expression is too complicated and the number of AST "
        "nodes exceeds the maximum allowed size of {}.",
        expression_, invalid_index);

    SyntaxTreeNode& node = node_arena_.emplace_back();
    node.as = value;
    node.type = type;
    node.lhs_index = lhs;
    node.rhs_index = rhs;
    // Index of the node that was just created.
    return node_arena_.size() - 1;
  }

  IndexType Parser::create_variable(std::string_view var)
  {
    auto it = std::ranges::find(allowed_variable_names_, var);
    // Variable is already known and allowed.
    if (it != allowed_variable_names_.end())
    {
      return std::distance(allowed_variable_names_.begin(), it);
    }
    else
    {
      if (allow_any_variable_)
      {
        // If we allow any variable, we just add it to the list of parsed variables.
        allowed_variable_names_.emplace_back(var);
        return allowed_variable_names_.size() - 1;
      }
      else
      {
        FOUR_C_THROW("While parsing '{}': variable '{}' is not allowed.", expression_, var);
      }
    }
  }

  void Parser::throw_syntax_error(const std::string& msg, std::size_t pos) const
  {
    // Construct a string with the position of the error marked by a caret.
    FOUR_C_THROW(
        "Error while parsing:\n"
        "{}\n"
        "{:>{}}\n"
        "{:>{}}{}",
        expression_, "^", pos + 1, " ", pos, msg);
  }


}  // namespace Core::Utils::SymbolicExpressionDetails


FOUR_C_NAMESPACE_CLOSE
