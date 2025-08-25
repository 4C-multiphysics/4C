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
  /*======================================================================*/
  /* Lexer methods */

  /*----------------------------------------------------------------------*/
  /*!
  \brief method used to step through std::string funct_
         delivers its character at position pos_++
  */
  int Lexer::get_next()
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


  void Lexer::advance()
  {
    for (;;)
    {
      int t = get_next();
      if ((t == ' ') || (t == '\t'))
      {
        /* ignore whitespaces */
      }
      else if (t == '\n')
      {
        FOUR_C_THROW("newline in function definition");
      }
      else if (t == EOF)
      {
        tok_ = Lexer::Lexer::tok_done;
        return;
      }
      else
      {
        if (isdigit(t))
        {
          str_ = &(funct_[pos_ - 1]);
          while (isdigit(t))
          {
            t = get_next();
          }
          if ((t != '.') && (t != 'E') && (t != 'e'))
          {
            if (t != EOF)
            {
              pos_--;
            }
            integer_ = atoi(str_);
            tok_ = Lexer::tok_int;
            return;
          }
          if (t == '.')
          {
            t = get_next();
            if (isdigit(t))
            {
              while (isdigit(t))
              {
                t = get_next();
              }
            }
            else
            {
              FOUR_C_THROW("no digits after point at pos {}", pos_);
            }
          }
          if ((t == 'E') || (t == 'e'))
          {
            t = get_next();
            if ((t == '-') || (t == '+'))
            {
              t = get_next();
            }
            if (isdigit(t))
            {
              while (isdigit(t))
              {
                t = get_next();
              }
            }
            else
            {
              FOUR_C_THROW("no digits after exponent at pos {}", pos_);
            }
          }
          if (t != EOF)
          {
            pos_--;
          }
          real_ = strtod(str_, nullptr);
          tok_ = Lexer::tok_real;
          return;
        }
        else if (isalpha(t) || (t == '_'))
        {
          str_ = &(funct_[pos_ - 1]);
          while (isalnum(t) || (t == '_'))
          {
            t = get_next();
          }
          if (t != EOF)
          {
            pos_--;
          }
          tok_ = Lexer::tok_name;
          integer_ = funct_.data() + pos_ - str_;  // length of operator name, e.g. 'sin' has '3'
          return;
        }
        else if (t == '+')
        {
          tok_ = Lexer::tok_add;
          return;
        }
        else if (t == '-')
        {
          tok_ = Lexer::tok_sub;
          return;
        }
        else if (t == '*')
        {
          tok_ = Lexer::tok_mul;
          return;
        }
        else if (t == '/')
        {
          tok_ = Lexer::tok_div;
          return;
        }
        else if (t == '^')
        {
          tok_ = Lexer::tok_pow;
          return;
        }
        else if (t == '(')
        {
          tok_ = Lexer::tok_lpar;
          return;
        }
        else if (t == ')')
        {
          tok_ = Lexer::tok_rpar;
          return;
        }
        else if (t == ',')
        {
          tok_ = Lexer::tok_comma;
          return;
        }
        else if (t == '>')
        {
          t = get_next();
          if (t == '=')
          {
            tok_ = Lexer::tok_ge;
            return;
          }
          if (t != EOF)
          {
            pos_--;
          }
          tok_ = Lexer::tok_gt;
          return;
        }
        else if (t == '<')
        {
          t = get_next();
          if (t == '=')
          {
            tok_ = Lexer::tok_le;
            return;
          }
          if (t != EOF)
          {
            pos_--;
          }
          tok_ = Lexer::tok_lt;
          return;
        }
        else if (t == '&')
        {
          t = get_next();
          if (t == '&')
          {
            tok_ = Lexer::tok_and;
            return;
          }
          else
          {
            FOUR_C_THROW("unexpected char '{}' at pos {}", t, pos_);
          }
        }
        else if (t == '|')
        {
          t = get_next();
          if (t == '|')
          {
            tok_ = Lexer::tok_or;
            return;
          }
          else
          {
            FOUR_C_THROW("unexpected char '{}' at pos {}", t, pos_);
          }
        }
        else if (t == '!')
        {
          tok_ = Lexer::tok_bang;
          return;
        }
        else
        {
          if (t >= 32)
            FOUR_C_THROW("unexpected char '{}' at pos {}", t, pos_);
          else
            FOUR_C_THROW("unexpected char '{}' at pos {}", t, pos_);
          tok_ = Lexer::tok_none;
          return;
        }
      }
    }
  }
}  // namespace Core::Utils::SymbolicExpressionDetails


FOUR_C_NAMESPACE_CLOSE
