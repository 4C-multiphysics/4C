// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_utils_symbolic_expression.hpp"

#include "4C_utils_exceptions.hpp"

#include <Sacado.hpp>

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace
{
  template <std::size_t n>
  [[nodiscard]] bool contains(const std::array<std::string, n>& array, const std::string& element)
  {
    return std::find(array.begin(), array.end(), element) != array.end();
  }
}  // namespace

namespace Core::Utils::SymbolicExpressionDetails
{
  /*----------------------------------------------------------------------*/
  /*!
  \brief Syntax tree node holding binary, unary operator or literals
  */
  template <class T>
  class SyntaxTreeNode
  {
   public:
    using NodePtr = std::unique_ptr<SyntaxTreeNode<T>>;

    //! destructor
    ~SyntaxTreeNode() = default;

    //! copy constructor
    SyntaxTreeNode(const SyntaxTreeNode& other)
        : type_{other.type_},
          variable_{other.variable_},
          function_{other.function_},
          lhs_{std::make_unique<SyntaxTreeNode>(*other.lhs_)},
          rhs_{std::make_unique<SyntaxTreeNode>(*other.rhs_)} {};

    //! copy assignment operator
    SyntaxTreeNode& operator=(const SyntaxTreeNode& other)
    {
      if (&other == this) return *this;
      type_ = other.type_;
      variable_ = other.variable_;
      function_ = other.function_;
      lhs_ = std::make_unique<SyntaxTreeNode>(*other.lhs_);
      rhs_ = std::make_unique<SyntaxTreeNode>(*other.rhs_);
      return *this;
    };

    //! move constructor
    SyntaxTreeNode(SyntaxTreeNode&& other) noexcept = default;

    //! move assignment operator
    SyntaxTreeNode& operator=(SyntaxTreeNode&& other) noexcept = default;

    enum NodeType
    {
      lt_variable,  // independent variable: 't', 'x', ...
      lt_number,    // number (literal)
      lt_function,
      lt_operator
    };

    //! Construct a SyntaxTreeNode with content @p type and a @p lhs and  @p rhs operand.
    SyntaxTreeNode(NodeType type, NodePtr lhs, NodePtr rhs)
        : type_(type), lhs_(std::move(lhs)), rhs_(std::move(rhs))
    {
    }

    //! type of the node, ie operator, literal, ...
    NodeType type_;

    //! particularise the node content: a literal number is double, an operatir has a character
    union
    {
      double number;  // holds 1.0, 7.e-9, etc
      char op;        // hold '+', '*', etc
    } v_;

    //! input string expressing the variable, holds 't', 'x', etc
    std::string variable_;
    //! input string expressing the function
    std::string function_;

    NodePtr lhs_;  // left hand side node
    NodePtr rhs_;  // right hand side node
  };


  /*----------------------------------------------------------------------*/
  /*!
  \brief Class holds auxiliar variables for Lexan method which steps through
         the string destilling the function tokens
  */
  class Lexer
  {
   public:
    //! constructor
    Lexer(std::string funct) : funct_(std::move(funct)), pos_(0) {}

    //! delivers funct_ character at position pos_++
    int get_next();

    //! identifies a token (value and kind) in the funct_ string
    void lexan();

    //! type of identifiable tokens in string funct_
    enum TokenType
    {
      tok_none,
      tok_done,
      tok_name,  // operator name, e.g. 'sin'
      tok_int,   // integer number
      tok_real,  // reals number
      tok_add,   // addition '+'
      tok_sub,   // subtraction and negation '-'
      tok_mul,   // multiplication '*'
      tok_div,   // division '/'
      tok_pow,   // power '^'
      tok_lpar,  // left parenthesis '('
      tok_rpar,  // right parenthesis ')'
      tok_comma  // comma ',' (used to separate function arguments)
    };

    std::string funct_;  // function description as string, i.e. "t^2", "sin(t)", etc
    unsigned pos_;       // current position in string funct_
    TokenType tok_;      // current token of string funct_
    char* str_;          // pointer to current character in funct_
    int integer_;        // translated integer number or length of operator word
    double real_;        // translated real number
  };

  //! helper structs
  template <typename T>
  struct IsFAD : public std::false_type
  {
  };

  template <typename T>
  struct IsFAD<Sacado::Fad::DFad<T>> : public std::true_type
  {
  };


  /*----------------------------------------------------------------------*/
  /*!
  \brief Parser
  */
  template <class T>
  class Parser
  {
   public:
    using NodePtr = typename SyntaxTreeNode<T>::NodePtr;

    //! constructor
    Parser(std::string funct);

    //! destructor
    ~Parser() = default;

    //! copy constuctor
    Parser(const Parser& other)
        : symbolicexpression_{other.symbolicexpression_},
          expr_{std::make_unique<SyntaxTreeNode<T>>(*other.expr_)},
          parsed_variable_constant_names_{other.parsed_variable_constant_names_} {};

    //! copy assignment operator
    Parser& operator=(const Parser& other)
    {
      if (&other == this) return *this;
      symbolicexpression_ = other.symbolicexpression_;
      expr_ = std::make_unique<NodePtr>(*other.expr_);
      parsed_variable_constant_names_ = other.parsed_variable_constant_names_;
      return *this;
    };

    //! move constructor
    Parser(Parser&& other) noexcept = default;

    //! move assignment operator
    Parser& operator=(Parser&& other) noexcept = default;

    /*!
     * @brief evaluates the parsed expression for a given set of variables
     *
     * @param[in] variable_values A map containing all variables (variablename, value) necessary to
     * evaluate the parsed expression
     * @return Value of the parsed expression
     */
    T evaluate_expression(const std::map<std::string, T>& variable_values) const;

    /*!
     * @brief evaluates the derivative of the parsed expression with respect to a given set of
     * variables
     *
     * @param[in] variable_values A map containing all variables (variablename, value) necessary to
     * evaluate the parsed expression. Since the derivative of the parsed expression is evaluated,
     * only  Sacado::Fad::DFad<T> types are allowed
     * @param[in] constants A map containing all constants (constantname, value) necessary
     * to evaluate the parsed expression
     * @return  Derivative of the parsed expression with respect to the variables
     */
    template <typename T2, std::enable_if_t<IsFAD<T2>::value, void*> = nullptr>
    T evaluate_derivative(const std::map<std::string, T2>& variable_values,
        const std::map<std::string, double>& constants = {}) const;

    //! Check if a variable with name 'varname' exists
    [[nodiscard]] bool is_variable(const std::string& varname) const;

   private:
    NodePtr parse_primary(Lexer& lexer);
    NodePtr parse_pow(Lexer& lexer);
    NodePtr parse_term(Lexer& lexer);
    NodePtr parse_expr(Lexer& lexer);
    NodePtr parse(Lexer& lexer);

    //! given symbolic expression
    std::string symbolicexpression_;

    //! evaluates the parsed expression
    T evaluate(const std::map<std::string, T>& variable_values,
        const std::map<std::string, double>& constants = {}) const;

    //! recursively extract corresponding number out of a syntax tree node
    T interpret(const SyntaxTreeNode<T>& node) const;

    //! syntax tree root
    NodePtr expr_;

    //! set of all parsed variables
    std::set<std::string> parsed_variable_constant_names_;

    //! necessary variable values for evaluating the parsed expression
    mutable const std::map<std::string, T>* variable_values_{nullptr};

    //! necessary constant values for evaluating the parsed expression
    mutable const std::map<std::string, double>* constant_values_{nullptr};
  };


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

  /*----------------------------------------------------------------------*/
  /*!
  \brief Identify current token
         type: tok_,
         value: integer_, real_,
         operator name: str_
  */
  void Lexer::lexan()
  {
    for (;;)
    {
      int t = get_next();
      if ((t == ' ') || (t == '\t'))
      {
        /* ignore whitespaces */
        /* this should never happen because we cannot read strings with
         * whitespaces from .dat files. :( */
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
              FOUR_C_THROW("no digits after point at pos %d", pos_);
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
              FOUR_C_THROW("no digits after exponent at pos %d", pos_);
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
          integer_ = &(funct_[pos_]) - str_;  // length of operator name, e.g. 'sin' has '3'
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
        else
        {
          if (t >= 32)
            FOUR_C_THROW("unexpected char '%c' at pos %d", t, pos_);
          else
            FOUR_C_THROW("unexpected char '%d' at pos %d", t, pos_);
          tok_ = Lexer::tok_none;
          return;
        }
      }
    }
  }

  /*===================================== da=================================*/
  /* Parser methods */

  /*----------------------------------------------------------------------*/
  /*!
  \brief Constructor of parser object
  */
  template <class T>
  Parser<T>::Parser(std::string funct)
  {
    //! set symbolic expression
    symbolicexpression_ = funct;

    //! create Lexer which stores all token of funct
    Lexer lexer{std::move(funct)};

    //! retrieve first token of funct
    lexer.lexan();

    //! create syntax tree equivalent to funct
    expr_ = parse(lexer);
  }

  /*----------------------------------------------------------------------*/
  /*!
  \brief Check if input is a variable name
  */
  template <class T>
  bool Parser<T>::is_variable(const std::string& varname) const
  {
    //! check if variable exists
    return (parsed_variable_constant_names_.count(varname) != 0);
  }


  template <class T>
  T Parser<T>::evaluate_expression(const std::map<std::string, T>& variable_values) const
  {
    return evaluate(variable_values);
  }


  template <class T>
  template <typename T2, std::enable_if_t<IsFAD<T2>::value, void*>>
  T Parser<T>::evaluate_derivative(const std::map<std::string, T2>& variable_values,
      const std::map<std::string, double>& constants) const
  {
    return evaluate(variable_values, constants);
  }


  template <class T>
  T Parser<T>::evaluate(const std::map<std::string, T>& variable_values,
      const std::map<std::string, double>& constants) const
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    const bool all_required_variables_passed =
        std::all_of(parsed_variable_constant_names_.begin(), parsed_variable_constant_names_.end(),
            [&](const auto& var_name)
            { return (variable_values.count(var_name) + constants.count(var_name)) == 1; });

    if (!all_required_variables_passed)
    {
      std::string evaluate_variable_names =
          std::accumulate(variable_values.begin(), variable_values.end(), std::string(),
              [](const std::string& acc, const auto& v)
              { return acc.empty() ? v.first : acc + ", " + v.first; });

      std::string evaluate_constant_names =
          std::accumulate(constants.begin(), constants.end(), std::string(),
              [](const std::string& acc, const auto& v)
              { return acc.empty() ? v.first : acc + ", " + v.first; });

      FOUR_C_THROW(
          "Some variables that this parser encountered in the expression are not passed to "
          "the Evaluate function.\n\n"
          "Expression:  %s \n"
          "Variables passed to Evaluate: %s \n"
          "Constants passed to Evaluate: %s",
          symbolicexpression_.c_str(), evaluate_variable_names.c_str(),
          evaluate_constant_names.c_str());
    }
#endif

    //! safety check if variable_values_ and constant_values_ are nullptr
    FOUR_C_ASSERT(variable_values_ == nullptr, "Internal error");
    FOUR_C_ASSERT(constant_values_ == nullptr, "Internal error");

    //! set variable values
    variable_values_ = &variable_values;
    //! set constant values if map of constants is not empty
    if (constants.size() != 0) constant_values_ = &constants;

    //! check if function has been parsed
    FOUR_C_ASSERT(expr_ != nullptr, "Internal error");

    //! evaluate syntax tree of function depending on set variables
    auto result = this->interpret(*expr_);

    //! set variable_values_ and constant_values_ as nullptr again for safety reasons
    variable_values_ = nullptr;
    constant_values_ = nullptr;

    return result;
  }

  /*----------------------------------------------------------------------*/
  /*!
  \brief Parse primary entities, i.e. literals and unary operators,
         such as numbers, parentheses, independent variables, operator names
  */
  template <class T>
  auto Parser<T>::parse_primary(Lexer& lexer) -> NodePtr
  {
    NodePtr lhs = nullptr;

    switch (lexer.tok_)
    {
      case Lexer::tok_lpar:
        lexer.lexan();
        lhs = parse_expr(lexer);
        if (lexer.tok_ != Lexer::tok_rpar) FOUR_C_THROW("')' expected");
        lexer.lexan();
        break;
      case Lexer::tok_int:
        lhs = std::make_unique<SyntaxTreeNode<T>>(SyntaxTreeNode<T>::lt_number, nullptr, nullptr);
        lhs->v_.number = lexer.integer_;
        lexer.lexan();
        break;
      case Lexer::tok_real:
        lhs = std::make_unique<SyntaxTreeNode<T>>(SyntaxTreeNode<T>::lt_number, nullptr, nullptr);
        lhs->v_.number = lexer.real_;
        lexer.lexan();
        break;
      case Lexer::tok_sub:
      {
        NodePtr rhs;
        lexer.lexan();
        /*rhs = parse_primary();*/
        rhs = parse_pow(lexer);
        if (rhs->type_ == SyntaxTreeNode<T>::lt_number)
        {
          rhs->v_.number *= -1;
          lhs = std::move(rhs);
        }
        else
        {
          lhs = std::make_unique<SyntaxTreeNode<T>>(SyntaxTreeNode<T>::lt_number, nullptr, nullptr);
          lhs->v_.number = -1;
          lhs = std::make_unique<SyntaxTreeNode<T>>(
              SyntaxTreeNode<T>::lt_operator, std::move(lhs), std::move(rhs));
          lhs->v_.op = '*';
        }
        break;
      }
      case Lexer::tok_name:
      {
        // get substring starting from str_ with length of lexer.integer_
        std::string name(lexer.str_, lexer.integer_);
        if ((lexer.integer_ == 2) && (std::strncmp("pi", lexer.str_, lexer.integer_) == 0))
        {
          lhs = std::make_unique<SyntaxTreeNode<T>>(SyntaxTreeNode<T>::lt_number, nullptr, nullptr);
          lhs->v_.number = M_PI;
          lexer.lexan();
          break;
        }
        else
        {
          if (name == "acos" or name == "asin" or name == "atan" or name == "cos" or
              name == "sin" or name == "tan" or name == "cosh" or name == "sinh" or
              name == "tanh" or name == "exp" or name == "log" or name == "log10" or
              name == "sqrt" or name == "ceil" or name == "heaviside" or name == "fabs" or
              name == "floor")
          {
            lhs = std::make_unique<SyntaxTreeNode<T>>(
                SyntaxTreeNode<T>::lt_function, nullptr, nullptr);
            lhs->function_ = name;
            lexer.lexan();
            if (lexer.tok_ != Lexer::tok_lpar)
              FOUR_C_THROW("'(' expected after function name '%s'", name.c_str());
            lexer.lexan();
            lhs->lhs_ = parse_expr(lexer);
            if (lexer.tok_ != Lexer::tok_rpar) FOUR_C_THROW("')' expected");
            lexer.lexan();
            break;
          }
          else if (name == "atan2")
          {
            lhs = std::make_unique<SyntaxTreeNode<T>>(
                SyntaxTreeNode<T>::lt_function, nullptr, nullptr);
            lhs->function_ = name;
            lexer.lexan();
            if (lexer.tok_ != Lexer::tok_lpar)
              FOUR_C_THROW("'(' expected after function name '%s'", name.c_str());
            lexer.lexan();
            lhs->lhs_ = parse_expr(lexer);
            if (lexer.tok_ != Lexer::tok_comma) FOUR_C_THROW("',' expected");
            lexer.lexan();
            lhs->rhs_ = parse_expr(lexer);
            if (lexer.tok_ != Lexer::tok_rpar)
              FOUR_C_THROW("')' expected for function name '%s'", name.c_str());
            lexer.lexan();
            break;
          }
          else
          {
            lhs = std::make_unique<SyntaxTreeNode<T>>(
                SyntaxTreeNode<T>::lt_variable, nullptr, nullptr);
            lhs->variable_ = name;
            parsed_variable_constant_names_.insert(name);
            lexer.lexan();
          }
        }
        break;
      }
      default:
        FOUR_C_THROW("unexpected token %d", lexer.tok_);
        break;
    }

    return lhs;
  }


  /*----------------------------------------------------------------------*/
  /*!
  \brief Parse entities connected by power: a^b
  */
  template <class T>
  auto Parser<T>::parse_pow(Lexer& lexer) -> NodePtr
  {
    NodePtr lhs;
    NodePtr rhs;

    lhs = parse_primary(lexer);
    for (;;)
    {
      if (lexer.tok_ == Lexer::tok_pow)
      {
        lexer.lexan();
        rhs = parse_primary(lexer);
        lhs = std::make_unique<SyntaxTreeNode<T>>(
            SyntaxTreeNode<T>::lt_operator, std::move(lhs), std::move(rhs));
        lhs->v_.op = '^';
      }
      else
      {
        break;
      }
    }

    return lhs;
  }


  /*----------------------------------------------------------------------*/
  /*!
  \brief Parse entities connected by multiplication or division: a*b, a/b
  */
  template <class T>
  auto Parser<T>::parse_term(Lexer& lexer) -> NodePtr
  {
    NodePtr lhs;
    NodePtr rhs;

    lhs = parse_pow(lexer);
    for (;;)
    {
      if (lexer.tok_ == Lexer::tok_mul)
      {
        lexer.lexan();
        rhs = parse_pow(lexer);
        lhs = std::make_unique<SyntaxTreeNode<T>>(
            SyntaxTreeNode<T>::lt_operator, std::move(lhs), std::move(rhs));
        lhs->v_.op = '*';
      }
      else if (lexer.tok_ == Lexer::tok_div)
      {
        lexer.lexan();
        rhs = parse_pow(lexer);
        lhs = std::make_unique<SyntaxTreeNode<T>>(
            SyntaxTreeNode<T>::lt_operator, std::move(lhs), std::move(rhs));
        lhs->v_.op = '/';
      }
      else
      {
        break;
      }
    }

    return lhs;
  }

  /*----------------------------------------------------------------------*/
  /*!
  \brief Parse entity
  */
  template <class T>
  auto Parser<T>::parse(Lexer& lexer) -> NodePtr
  {
    NodePtr lhs;

    lhs = parse_expr(lexer);

    // check if parsing ended before processing the entire string
    if (lexer.tok_ != Lexer::tok_done)
    {
      FOUR_C_THROW("Invalid syntax: The remaining string '%s' is not parsed.",
          lexer.funct_.c_str() + lexer.pos_);
    }

    return lhs;
  }

  /*----------------------------------------------------------------------*/
  /*!
  \brief Parse entities connected by addition or subtraction: a+b, a-b
  */
  template <class T>
  auto Parser<T>::parse_expr(Lexer& lexer) -> NodePtr
  {
    typename SyntaxTreeNode<T>::NodePtr lhs;
    typename SyntaxTreeNode<T>::NodePtr rhs;

    lhs = parse_term(lexer);
    for (;;)
    {
      if (lexer.tok_ == Lexer::tok_add)
      {
        lexer.lexan();
        rhs = parse_term(lexer);
        lhs = std::make_unique<SyntaxTreeNode<T>>(
            SyntaxTreeNode<T>::lt_operator, std::move(lhs), std::move(rhs));
        lhs->v_.op = '+';
      }
      else if (lexer.tok_ == Lexer::tok_sub)
      {
        lexer.lexan();
        rhs = parse_term(lexer);
        lhs = std::make_unique<SyntaxTreeNode<T>>(
            SyntaxTreeNode<T>::lt_operator, std::move(lhs), std::move(rhs));
        lhs->v_.op = '-';
      }
      else
      {
        break;
      }
    }

    return lhs;
  }


  /*----------------------------------------------------------------------*/
  /*!
  \brief Recursively extract corresponding number out of a syntax tree node
  */
  template <class T>
  T Parser<T>::interpret(const SyntaxTreeNode<T>& node) const
  {
    T res = 0;  // the result

    switch (node.type_)
    {
      // literal numbers: leaf of syntax tree node
      case SyntaxTreeNode<T>::lt_number:
        res = node.v_.number;
        break;
      // binary operators: bifurcating branch of syntax tree node
      case SyntaxTreeNode<T>::lt_operator:
      {
        T lhs;
        T rhs;

        // recursively visit branches and obtain sub-results
        lhs = interpret(*node.lhs_);
        rhs = interpret(*node.rhs_);

        // evaluate the node operator
        switch (node.v_.op)
        {
          case '+':
            res = lhs + rhs;
            break;
          case '-':
            res = lhs - rhs;
            break;
          case '*':
            res = lhs * rhs;
            break;
          case '/':
            /* check for rhs==0.0? */
            res = lhs / rhs;
            break;
          case '^':
            res = std::pow(lhs, rhs);
            break;
          default:
            FOUR_C_THROW("unsupported operator '%c'", node.v_.op);
        }
        break;
      }
      // independent variables: as set by user
      case SyntaxTreeNode<T>::lt_variable:
      {
        if (parsed_variable_constant_names_.count(node.variable_) != 0)
        {
          if (constant_values_ == nullptr)
          {
            if (variable_values_->find(node.variable_) == variable_values_->end())
            {
              FOUR_C_THROW("variable or constant '%s' not given as input in evaluate()",
                  node.variable_.c_str());
            }
            else
            {
              res = variable_values_->at(std::string(node.variable_));
            }
          }
          else
          {
            if ((variable_values_->find(node.variable_) == variable_values_->end()) &&
                (constant_values_->find(node.variable_) == constant_values_->end()))
            {
              FOUR_C_THROW("variable or constant '%s' not given as input in EvaluateDeriv()",
                  node.variable_.c_str());
            }
            else
            {
              if (variable_values_->find(node.variable_) != variable_values_->end())
              {
                res = variable_values_->at(node.variable_);
              }
              else if (constant_values_->find(node.variable_) != constant_values_->end())
              {
                res = constant_values_->at(node.variable_);
              }
              else
                FOUR_C_THROW("Something went really wrong!");
            }
          }
        }
        else
          FOUR_C_THROW("unknown variable '%s'", node.variable_.c_str());
        break;
      }
      // unary operators
      case SyntaxTreeNode<T>::lt_function:
      {
        T arg;
        arg = interpret(*node.lhs_);
        if (node.function_ == "acos")
          res = acos(arg);
        else if (node.function_ == "asin")
          res = asin(arg);
        else if (node.function_ == "atan")
          res = atan(arg);
        else if (node.function_ == "cos")
          res = cos(arg);
        else if (node.function_ == "sin")
          res = sin(arg);
        else if (node.function_ == "tan")
          res = tan(arg);
        else if (node.function_ == "cosh")
          res = cosh(arg);
        else if (node.function_ == "sinh")
          res = sinh(arg);
        else if (node.function_ == "tanh")
          res = tanh(arg);
        else if (node.function_ == "exp")
          res = exp(arg);
        else if (node.function_ == "log")
          res = log(arg);
        else if (node.function_ == "log10")
          res = log10(arg);
        else if (node.function_ == "sqrt")
          res = sqrt(arg);
        else if (node.function_ == "atan2")
        {
          T arg2;
          // recursively visit branches and obtain sub-results
          arg2 = interpret(*node.rhs_);
          res = atan2(arg, arg2);
        }
        else if (node.function_ == "fabs")
          res = fabs(arg);
        else if (node.function_ == "heaviside")
        {
          if (arg > 0)
          {
            res = 1.0;
          }
          else
          {
            res = 0.0;
          }
        }
        else
          FOUR_C_THROW("unknown function_ '%s'", node.function_.c_str());
        break;
      }
      default:
        FOUR_C_THROW("unknown syntax tree node type");
        break;
    }

    return res;
  }

}  // namespace Core::Utils::SymbolicExpressionDetails

template <typename T>
Core::Utils::SymbolicExpression<T>::SymbolicExpression(const std::string& expression)
    : parser_for_value_(
          std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<ValueType>>(expression)),
      parser_for_firstderivative_(
          std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<FirstDerivativeType>>(
              expression)),
      parser_for_secondderivative_(
          std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<SecondDerivativeType>>(
              expression))
{
}



template <typename T>
auto Core::Utils::SymbolicExpression<T>::value(
    const std::map<std::string, ValueType>& variable_values) const -> ValueType
{
  return parser_for_value_->evaluate_expression(variable_values);
}


template <typename T>
auto Core::Utils::SymbolicExpression<T>::first_derivative(
    std::map<std::string, FirstDerivativeType> variable_values,
    const std::map<std::string, ValueType>& constant_values) const -> FirstDerivativeType
{
  return parser_for_firstderivative_->evaluate_derivative(variable_values, constant_values);
}


template <typename T>
auto Core::Utils::SymbolicExpression<T>::second_derivative(
    const std::map<std::string, SecondDerivativeType>& variable_values,
    const std::map<std::string, ValueType>& constant_values) const -> SecondDerivativeType
{
  return parser_for_secondderivative_->evaluate_derivative(variable_values, constant_values);
}


template <typename Number>
Core::Utils::SymbolicExpression<Number>::SymbolicExpression(
    const Core::Utils::SymbolicExpression<Number>& other)
    : parser_for_value_{std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<ValueType>>(
          *other.parser_for_value_)},
      parser_for_firstderivative_{
          std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<FirstDerivativeType>>(
              *other.parser_for_firstderivative_)},
      parser_for_secondderivative_{
          std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<SecondDerivativeType>>(
              *other.parser_for_secondderivative_)}
{
}


template <typename Number>
Core::Utils::SymbolicExpression<Number>& Core::Utils::SymbolicExpression<Number>::operator=(
    const Core::Utils::SymbolicExpression<Number>& other)
{
  parser_for_value_ = std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<ValueType>>(
      *other.parser_for_value_);
  parser_for_firstderivative_ =
      std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<FirstDerivativeType>>(
          *other.parser_for_firstderivative_);
  parser_for_secondderivative_ =
      std::make_unique<Core::Utils::SymbolicExpressionDetails::Parser<SecondDerivativeType>>(
          *other.parser_for_secondderivative_);
  return *this;
}



template <typename Number>
Core::Utils::SymbolicExpression<Number>::~SymbolicExpression() = default;

// explicit instantiations
template class Core::Utils::SymbolicExpression<double>;

FOUR_C_NAMESPACE_CLOSE
