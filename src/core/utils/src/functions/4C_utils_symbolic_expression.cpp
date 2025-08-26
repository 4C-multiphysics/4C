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
}  // namespace Core::Utils::SymbolicExpressionDetails


FOUR_C_NAMESPACE_CLOSE
