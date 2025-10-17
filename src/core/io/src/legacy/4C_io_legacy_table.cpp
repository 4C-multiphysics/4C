// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_io_legacy_table.hpp"

#include "4C_io_legacy_types.hpp"
#include "4C_io_yaml.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"

#include <cctype>
#include <cstring>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*/
/*!
  \brief Clean up.

*/
/*----------------------------------------------------------------------*/
static void destroy_symbol(SYMBOL* symbol)
{
  while (symbol != nullptr)
  {
    SYMBOL* next;

    if (symbol->type == sym_string)
    {
      free(symbol->s.string);
    }
    if (symbol->type == sym_map)
    {
      destroy_map(symbol->s.dir);
      delete symbol->s.dir;
    }
    next = symbol->next;
    delete symbol;
    symbol = next;
  }
}


/*----------------------------------------------------------------------*/
/*!
  \brief Clean up.

*/
/*----------------------------------------------------------------------*/
static void destroy_node(MapNode* node)
{
  if (node != nullptr)
  {
    destroy_node(node->lhs);
    destroy_node(node->rhs);
    if (node->symbol != nullptr)
    {
      destroy_symbol(node->symbol);
    }

    if (node->key) free(node->key);

    node->key = nullptr;
    node->lhs = nullptr;
    node->rhs = nullptr;
    node->symbol = nullptr;

    delete node;
  }
}


/*----------------------------------------------------------------------*/
/*!
  \brief Bring a map variable up to a clean state.

  That's needed before anything can be done with a map.

*/
/*----------------------------------------------------------------------*/
void init_map(MAP* map)
{
  /* We have a dummy node at the root to make life easier. The empty
   * key is not legal. */
  map->root.key = nullptr;
  map->root.symbol = nullptr;
  map->root.lhs = nullptr;
  map->root.rhs = nullptr;
  map->count = 0;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Clean up.

*/
/*----------------------------------------------------------------------*/
void destroy_map(MAP* map)
{
  destroy_node(map->root.lhs);
  destroy_node(map->root.rhs);
}


/*----------------------------------------------------------------------*/
/*!
  \brief See whether a node matches a certain key.

*/
/*----------------------------------------------------------------------*/
static int map_cmp_nodes(const MapNode* lhs, const char* rhs_key)
{
  if (lhs->key == nullptr) return -1;
  return strcmp(lhs->key, rhs_key);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the node in the map that matches the \a key.

  \return nullptr if there's no such node.

*/
/*----------------------------------------------------------------------*/
static MapNode* map_find_node(MAP* map, const char* key)
{
  MapNode* node;

  node = &(map->root);

  for (;;)
  {
    int cmp;
    cmp = map_cmp_nodes(node, key);
    if (cmp < 0)
    {
      if (node->rhs == nullptr)
      {
        node = nullptr;
        goto end;
      }
      else
      {
        node = node->rhs;
      }
    }
    else if (cmp > 0)
    {
      if (node->lhs == nullptr)
      {
        node = nullptr;
        goto end;
      }
      else
      {
        node = node->lhs;
      }
    }
    else
    {
      goto end;
    }
  }

end:
  return node;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol in the map that matches the \a key.

  \return nullptr if there's no such symbol.

*/
/*----------------------------------------------------------------------*/
SYMBOL* map_find_symbol(MAP* map, const char* key)
{
  MapNode* node;
  SYMBOL* symbol = nullptr;

  node = map_find_node(map, key);
  if (node != nullptr)
  {
    symbol = node->symbol;
  }

  return symbol;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a string.

*/
/*----------------------------------------------------------------------*/
int map_find_string(MAP* map, const char* key, const char** string)
{
  SYMBOL* symbol;
  int ret;

  symbol = map_find_symbol(map, key);
  ret = symbol_get_string(symbol, string);

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a integer.

*/
/*----------------------------------------------------------------------*/
int map_find_int(MAP* map, const char* key, int* integer)
{
  SYMBOL* symbol;
  int ret;

  symbol = map_find_symbol(map, key);
  ret = symbol_get_int(symbol, integer);

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a real.

*/
/*----------------------------------------------------------------------*/
int map_find_real(MAP* map, const char* key, double* real)
{
  SYMBOL* symbol;
  int ret;

  symbol = map_find_symbol(map, key);
  ret = symbol_get_real(symbol, real);

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a map.

*/
/*----------------------------------------------------------------------*/
int map_find_map(MAP* map, const char* key, MAP** dir)
{
  SYMBOL* symbol;
  int ret;

  symbol = map_find_symbol(map, key);
  ret = symbol_get_map(symbol, dir);

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a string.

  Stops if no string is found.

*/
/*----------------------------------------------------------------------*/
const char* map_read_string(MAP* map, const char* key)
{
  const char* string;

  if (!map_find_string(map, key, &string))
  {
    FOUR_C_THROW("no string attribute '{}' in map", key);
  }

  return string;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a integer.

  Stops if no integer is found.

*/
/*----------------------------------------------------------------------*/
int map_read_int(MAP* map, const char* key)
{
  int integer;

  if (!map_find_int(map, key, &integer))
  {
    FOUR_C_THROW("no int attribute '{}' in map", key);
  }

  return integer;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a real.

  Stops if no real is found.

*/
/*----------------------------------------------------------------------*/
double map_read_real(MAP* map, const char* key)
{
  double real;

  if (!map_find_real(map, key, &real))
  {
    int value;
    if (!map_find_int(map, key, &value)) FOUR_C_THROW("no real attribute '{}' in map", key);
    real = value;
  }

  return real;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Find the first symbol, return its value if it's a map.

  Stops if no map is found.

*/
/*----------------------------------------------------------------------*/
MAP* map_read_map(MAP* map, const char* key)
{
  MAP* dir;

  if (!map_find_map(map, key, &dir))
  {
    FOUR_C_THROW("no dir attribute '{}' in map", key);
  }

  return dir;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell if the first symbol with that \a key has this \a value.

*/
/*----------------------------------------------------------------------*/
int map_has_string(MAP* map, const char* key, const char* value)
{
  SYMBOL* symbol;
  const char* string;
  int ret;

  symbol = map_find_symbol(map, key);
  if (symbol != nullptr)
  {
    ret = symbol_get_string(symbol, &string);
    if (ret)
    {
      ret = strcmp(string, value) == 0;
    }
  }
  else
  {
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell if the first symbol with that \a key has this \a value.

*/
/*----------------------------------------------------------------------*/
int map_has_int(MAP* map, const char* key, const int value)
{
  SYMBOL* symbol;
  int integer;
  int ret;

  symbol = map_find_symbol(map, key);
  if (symbol != nullptr)
  {
    ret = symbol_get_int(symbol, &integer);
    if (ret)
    {
      ret = integer == value;
    }
  }
  else
  {
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell if the first symbol with that \a key has this \a value.

*/
/*----------------------------------------------------------------------*/
int map_has_real(MAP* map, const char* key, const double value)
{
  SYMBOL* symbol;
  double real;
  int ret;

  symbol = map_find_symbol(map, key);
  if (symbol != nullptr)
  {
    ret = symbol_get_real(symbol, &real);
    if (ret)
    {
      ret = real == value;
    }
  }
  else
  {
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell if the first symbol with that \a key is a map.

  No value comparison here.

*/
/*----------------------------------------------------------------------*/
int map_has_map(MAP* map, const char* key)
{
  SYMBOL* symbol;
  int ret;

  symbol = map_find_symbol(map, key);
  if (symbol != nullptr)
  {
    ret = symbol_is_map(symbol);
  }
  else
  {
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Insert a symbol.

  Any new symbol becomes the first one with that key.

  Ownership of the symbol and the key is taken. Both have to be
  allocated using malloc or the like.

*/
/*----------------------------------------------------------------------*/
static void map_insert_symbol(MAP* map, SYMBOL* symbol, char* key)
{
  MapNode* node;

  node = &(map->root);
  for (;;)
  {
    int cmp;
    cmp = map_cmp_nodes(node, key);
    if (cmp < 0)
    {
      if (node->rhs == nullptr)
      {
        node->rhs = new MapNode;
        node->rhs->key = key;
        node->rhs->symbol = symbol;
        node->rhs->count = 1;
        node->rhs->lhs = nullptr;
        node->rhs->rhs = nullptr;
        map->count++;
        goto end;
      }
      else
      {
        node = node->rhs;
      }
    }
    else if (cmp > 0)
    {
      if (node->lhs == nullptr)
      {
        node->lhs = new MapNode;
        node->lhs->key = key;
        node->lhs->symbol = symbol;
        node->lhs->count = 1;
        node->lhs->lhs = nullptr;
        node->lhs->rhs = nullptr;
        map->count++;
        goto end;
      }
      else
      {
        node = node->lhs;
      }
    }
    else
    {
      /* This key is already there. Free the duplicated memory. */
      free(key);

      /* append symbol */
      symbol->next = node->symbol;
      node->symbol = symbol;
      node->count++;
      map->count++;
      goto end;
    }
  }

end:
  return;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Insert that string with this key.

*/
/*----------------------------------------------------------------------*/
void map_insert_string(MAP* map, char* string, char* key)
{
  SYMBOL* symbol;

  symbol = new SYMBOL;
  symbol->type = sym_string;
  symbol->s.string = string;
  symbol->next = nullptr;

  map_insert_symbol(map, symbol, key);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Insert that integer with this key.

*/
/*----------------------------------------------------------------------*/
void map_insert_int(MAP* map, int integer, char* key)
{
  SYMBOL* symbol;

  symbol = new SYMBOL;
  symbol->type = sym_int;
  symbol->s.integer = integer;
  symbol->next = nullptr;

  map_insert_symbol(map, symbol, key);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Insert that real with this key.

*/
/*----------------------------------------------------------------------*/
void map_insert_real(MAP* map, double real, char* key)
{
  SYMBOL* symbol;

  symbol = new SYMBOL;
  symbol->type = sym_real;
  symbol->s.real = real;
  symbol->next = nullptr;

  map_insert_symbol(map, symbol, key);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Insert that map with this key.

*/
/*----------------------------------------------------------------------*/
void map_insert_map(MAP* map, MAP* dir, char* key)
{
  SYMBOL* symbol;

  symbol = new SYMBOL;
  symbol->type = sym_map;
  symbol->s.dir = dir;
  symbol->next = nullptr;

  map_insert_symbol(map, symbol, key);
}



/*----------------------------------------------------------------------*/
/*!
  \brief Tell how many symbols of the given name there are.

*/
/*----------------------------------------------------------------------*/
int map_symbol_count(MAP* map, const char* key)
{
  int count = 0;

  const MapNode* node = map_find_node(map, key);
  if (node != nullptr)
  {
    count = node->count;
  }

  return count;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Take a symbol chain out of the map. Leave the symbol alive.

  This is for the experienced user only. A symbol chain is removed
  from the map, but the key (the node behind it) stays alive. Also the
  symbols are not deallocated. The caller must already have a pointer
  to the symbol chain and takes responsibility for it.

*/
/*----------------------------------------------------------------------*/
void map_disconnect_symbols(MAP* map, const char* key)
{
  MapNode* node = map_find_node(map, key);
  if (node != nullptr)
  {
    node->symbol = nullptr;
    node->count = 0;
  }
}


/*----------------------------------------------------------------------*/
/*!
  \brief Prepend the symbol chain to one under the given key.

  \param map    (i/o) map we work with
  \param key      (i) key to those chain we want to prepend
  \param symbol   (i) start of the new symbol chain
  \param count    (i) number of symbol in the chain

*/
/*----------------------------------------------------------------------*/
void map_prepend_symbols(MAP* map, const char* key, SYMBOL* symbol, int count)
{
  MapNode* node;

  node = map_find_node(map, key);
  if (node != nullptr)
  {
    if (node->symbol != nullptr)
    {
      SYMBOL* s;
      s = node->symbol;

      while (s->next != nullptr)
      {
        s = s->next;
      }
      s->next = symbol;
      node->count += count;
    }
    else
    {
      node->symbol = symbol;
      node->count = count;
    }

    map->count += count;
  }
  else
  {
    FOUR_C_THROW("no node for key '{}'", key);
  }
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell whether the symbol is a string.

*/
/*----------------------------------------------------------------------*/
int symbol_is_string(const SYMBOL* symbol)
{
  return (symbol != nullptr) && (symbol->type == sym_string);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell whether the symbol is an integer.

*/
/*----------------------------------------------------------------------*/
int symbol_is_int(const SYMBOL* symbol) { return (symbol != nullptr) && (symbol->type == sym_int); }


/*----------------------------------------------------------------------*/
/*!
  \brief Tell whether the symbol is a real.

*/
/*----------------------------------------------------------------------*/
int symbol_is_real(const SYMBOL* symbol)
{
  return (symbol != nullptr) && (symbol->type == sym_real);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Tell whether the symbol is a map.

*/
/*----------------------------------------------------------------------*/
int symbol_is_map(const SYMBOL* symbol) { return (symbol != nullptr) && (symbol->type == sym_map); }



/*----------------------------------------------------------------------*/
/*!
  \brief Extract the value if its a string.

*/
/*----------------------------------------------------------------------*/
int symbol_get_string(const SYMBOL* symbol, const char** string)
{
  int ret;

  if (symbol && (symbol->type == sym_string))
  {
    *string = symbol->s.string;
    ret = 1;
  }
  else
  {
    *string = "";
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Extract the value if its an integer.

*/
/*----------------------------------------------------------------------*/
int symbol_get_int(const SYMBOL* symbol, int* integer)
{
  int ret;

  if (symbol && (symbol->type == sym_int))
  {
    *integer = symbol->s.integer;
    ret = 1;
  }
  else
  {
    *integer = 0;
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Extract the value if its a real.

*/
/*----------------------------------------------------------------------*/
int symbol_get_real(const SYMBOL* symbol, double* real)
{
  int ret;

  if (symbol && (symbol->type == sym_real))
  {
    *real = symbol->s.real;
    ret = 1;
  }
  else
  {
    *real = 0.0;
    ret = 0;
  }

  return ret;
}

/*----------------------------------------------------------------------*/
/*!
  \brief Extract the value if its a real.

 */
/*----------------------------------------------------------------------*/
int symbol_get_real_as_float(const SYMBOL* symbol, float* real)
{
  int ret;

  if (symbol && (symbol->type == sym_real))
  {
    *real = (float)symbol->s.real;
    ret = 1;
  }
  else
  {
    *real = 0.0;
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Extract the value if its a map.

*/
/*----------------------------------------------------------------------*/
int symbol_get_map(const SYMBOL* symbol, MAP** map)
{
  int ret;

  if (symbol && (symbol->type == sym_map))
  {
    *map = symbol->s.dir;
    ret = 1;
  }
  else
  {
    *map = nullptr;
    ret = 0;
  }

  return ret;
}


/*----------------------------------------------------------------------*/
/*!
  \brief Extract the value if its a map.

*/
/*----------------------------------------------------------------------*/
MAP* symbol_map(const SYMBOL* symbol)
{
  MAP* ret = nullptr;

  if (symbol->type == sym_map)
  {
    ret = symbol->s.dir;
  }
  else
  {
    FOUR_C_THROW("Wrong symbol type {}", symbol->type);
  }

  return ret;
}

/*----------------------------------------------------------------------*/
/*!
  \brief The types of tokens recognized by the lexer.

*/
/*----------------------------------------------------------------------*/
enum TokenType
{
  tok_none,
  tok_done,
  tok_name,
  tok_string,
  tok_int,
  tok_real,
  tok_colon,
  tok_equal,
  tok_indent,
  tok_dedent
};


/*----------------------------------------------------------------------*/
/*!
 \brief The parsers internal state

 Here we have the parsers internal state. It consists of the current
 token (depending on the token type these variables have different
 meanings), the file buffer with the current read position, the
 current line number and indention level. These variables are very
 internal and only used while a control file is read.

 */
/*----------------------------------------------------------------------*/
struct ParserData
{
  TokenType tok;
  char* token_string;
  int token_int;
  double token_real;

  char* file_buffer;
  int file_size;

  int pos;
  int lineno;
  int indent_level;
  int indent_step;
};


/*----------------------------------------------------------------------*/
/*!
  \brief Init the data structure needed to read a file.

  The file is read on processor 0 and broadcasted to the others.

*/
/*----------------------------------------------------------------------*/
static void init_parser_data(ParserData* data, const char* filename, MPI_Comm comm)
{
  data->tok = tok_none;
  data->lineno = 1;
  data->pos = 0;
  data->indent_level = 0;
  data->indent_step = -1;

  int myrank = 0;
  int nprocs = 1;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &nprocs);

  /* No copy here. Valid only as long as the calling functions
   * filename is valid. */
  /*data->filename = filename;*/

  /* We need to have the information on all processes. That's why we
   * read the file on process 0 and broadcast it. The other way would
   * be to use MPI IO, but then we'd have to implement a separate
   * sequential version. */
  if (myrank == 0)
  {
    int bytes_read;
    FILE* file;
    file = fopen(filename, "rb");

    if (file == nullptr)
    {
      FOUR_C_THROW("cannot read file '{}'", filename);
    }

    /* find out the control file size */
    fseek(file, 0, SEEK_END);
    data->file_size = ftell(file);

    /* read file to local buffer */
    data->file_buffer = (char*)malloc((data->file_size + 1) * sizeof(char));
    fseek(file, 0, SEEK_SET);
    /*bytes_read = fread(data->file_buffer, sizeof(char), data->file_size, file);*/
    bytes_read = fread(data->file_buffer, sizeof(char), (size_t)data->file_size, file);
    if (bytes_read != data->file_size)
    {
      FOUR_C_THROW("failed to read file {}", filename);
    }
    /* a trailing zero helps a lot */
    data->file_buffer[data->file_size] = '\0';

    fclose(file);
  }

  if (nprocs > 1)
  {
    int err;
    err = MPI_Bcast(&data->file_size, 1, MPI_INT, 0, comm);
    if (err != 0)
    {
      FOUR_C_THROW("MPI_Bcast failed: {}", err);
    }
    if (myrank > 0)
    {
      data->file_buffer = (char*)malloc((data->file_size + 1) * sizeof(char));
    }
    err = MPI_Bcast(data->file_buffer, data->file_size + 1, MPI_CHAR, 0, comm);
    if (err != 0)
    {
      FOUR_C_THROW("MPI_Bcast failed: {}", err);
    }
  }
}


/*----------------------------------------------------------------------*/
/*!
  \brief Clean up.

*/
/*----------------------------------------------------------------------*/
static void destroy_parser_data(ParserData* data) { free(data->file_buffer); }

/*----------------------------------------------------------------------*/
/*!
  \brief Get the next char.

*/
/*----------------------------------------------------------------------*/
static int getnext(ParserData* data)
{
  if (data->pos < data->file_size)
  {
    /* ignore dos line endings */
    if (data->file_buffer[data->pos] == '\r')
    {
      data->pos++;
    }

    /* Increment the counter and return the char at the old position. */
    return data->file_buffer[data->pos++];
  }
  return EOF;
}


enum
{
  TABWIDTH = 8
};


/*----------------------------------------------------------------------*/
/*!
  \brief Get the next token.

*/
/*----------------------------------------------------------------------*/
static void lexan(ParserData* data)
{
  int line_begin = 0;
  int t;
  int indention = data->indent_level;

  for (;;)
  {
    t = getnext(data);
    if (t == ' ')
    {
      /* ignore whitespaces */
      if (line_begin)
      {
        indention++;
      }
    }
    else if (t == '\t')
    {
      /* ignore whitespaces */
      if (line_begin)
      {
        indention = ((indention + TABWIDTH - 1) / TABWIDTH) * TABWIDTH;
      }
    }
    else if (t == '\n')
    {
      data->lineno++;
      line_begin = 1;
      indention = 0;
    }
    else if (t == '#')
    {
      for (;;)
      {
        t = getnext(data);
        if (t == '\n')
        {
          break;
        }
      }
      data->lineno++;
      line_begin = 1;
      indention = 0;
    }
    else if (t == EOF)
    {
      data->tok = tok_done;
      goto end;
    }
    else
    {
      if (line_begin && (indention != data->indent_level))
      {
        if (data->indent_step == -1)
        {
          if (indention > data->indent_level)
          {
            FOUR_C_ASSERT(data->indent_level == 0, "non-zero intention at first line?!");
            data->indent_step = indention;
            data->indent_level = indention;
            data->token_int = 1;
            data->tok = tok_indent;
          }
          else
          {
            FOUR_C_THROW("dedent at toplevel!");
          }
        }
        else
        {
          if (indention > data->indent_level)
          {
            data->tok = tok_indent;
            FOUR_C_ASSERT(
                (indention - data->indent_level) % data->indent_step == 0, "malformed indention");
            data->token_int = (indention - data->indent_level) / data->indent_step;
            data->indent_level = indention;
          }
          else
          {
            data->tok = tok_dedent;
            FOUR_C_ASSERT(
                (data->indent_level - indention) % data->indent_step == 0, "malformed dedention");
            data->token_int = (data->indent_level - indention) / data->indent_step;
            data->indent_level = indention;
          }
        }
        data->pos--;
        goto end;
      }
      else
      {
        line_begin = 0;

        if ((t == '-') || isdigit(t))
        {
          data->token_string = &(data->file_buffer[data->pos - 1]);
          if (t == '-')
          {
            t = getnext(data);
          }
          while (isdigit(t))
          {
            t = getnext(data);
          }
          if ((t != '.') && (t != 'E') && (t != 'e'))
          {
            if (t != EOF)
            {
              data->pos--;
            }
            data->token_int = atoi(data->token_string);
            data->tok = tok_int;
            goto end;
          }
          if (t == '.')
          {
            t = getnext(data);
            if (isdigit(t))
            {
              while (isdigit(t))
              {
                t = getnext(data);
              }
            }
            else
            {
              FOUR_C_THROW("no digits after point at line {}", data->lineno);
            }
          }
          if ((t == 'E') || (t == 'e'))
          {
            t = getnext(data);
            if ((t == '-') || (t == '+'))
            {
              t = getnext(data);
            }
            if (isdigit(t))
            {
              while (isdigit(t))
              {
                t = getnext(data);
              }
            }
            else
            {
              FOUR_C_THROW("no digits after exponent at line {}", data->lineno);
            }
          }
          if (t != EOF)
          {
            data->pos--;
          }
          data->token_real = strtod(data->token_string, nullptr);
          data->tok = tok_real;
          goto end;
        }
        else if (isalpha(t) || (t == '_'))
        {
          data->token_string = &(data->file_buffer[data->pos - 1]);
          while (isalnum(t) || (t == '_'))
          {
            t = getnext(data);
          }
          if (t != EOF)
          {
            data->pos--;
          }
          data->tok = tok_name;
          data->token_int = &(data->file_buffer[data->pos]) - data->token_string;
          goto end;
        }
        else if (t == '"')
        {
          data->token_string = &(data->file_buffer[data->pos]);
          t = getnext(data);
          while (t != '"')
          {
            t = getnext(data);
            if (t == EOF)
            {
              FOUR_C_THROW("expected closing \" on line {}", data->lineno);
            }
          }
          data->tok = tok_string;
          data->token_int = &(data->file_buffer[data->pos - 1]) - data->token_string;
          goto end;
        }
        else if (t == ':')
        {
          data->tok = tok_colon;
          goto end;
        }
        else if (t == '=')
        {
          data->tok = tok_equal;
          goto end;
        }
        else
        {
          if (t >= 32)
            FOUR_C_THROW("unexpected char '{}' at line {}", t, data->lineno);
          else
            FOUR_C_THROW("unexpected char '{}' at line {}", t, data->lineno);
          data->tok = tok_none;
          goto end;
        }
      }
    }
  }

end:


  return;
}


/*----------------------------------------------------------------------*/
/*!
  \brief The top down parser.

*/
/*----------------------------------------------------------------------*/
static void parse_definitions(ParserData* data, MAP* dir)
{
  lexan(data);

  while (data->tok != tok_done)
  {
    switch (data->tok)
    {
      case tok_name:
      {
        char* name;

        /*
         * The string is not null terminated as it's a simple pointer
         * into the file buffer. However, we know its length so we can
         * handle that. */
        name = (char*)malloc((data->token_int + 1) * sizeof(char));
        /*strncpy(name, data->token_string, data->token_int);*/
        strncpy(name, data->token_string, (size_t)data->token_int);
        name[data->token_int] = '\0';

        lexan(data);
        switch (data->tok)
        {
          case tok_colon:
          {
            MAP* map;

            lexan(data);
            if ((data->tok != tok_indent) || (data->token_int != 1))
            {
              FOUR_C_THROW("Syntaxerror at line {}: single indention expected", data->lineno);
            }

            map = new MAP;
            init_map(map);
            parse_definitions(data, map);

            map_insert_map(dir, map, name);

            if ((data->tok == tok_dedent) && (data->token_int > 0))
            {
              data->token_int--;
              goto end;
            }

            break;
          }
          case tok_equal:
            lexan(data);
            switch (data->tok)
            {
              case tok_string:
              {
                char* string;

                /* Again, be carefully with those pointers... */
                string = (char*)malloc((data->token_int + 1) * sizeof(char));
                /*strncpy(string, data->token_string, data->token_int);*/
                strncpy(string, data->token_string, (size_t)data->token_int);
                string[data->token_int] = '\0';

                map_insert_string(dir, string, name);
                break;
              }
              case tok_int:
                map_insert_int(dir, data->token_int, name);
                break;
              case tok_real:
                map_insert_real(dir, data->token_real, name);
                break;
              default:
                FOUR_C_THROW("Syntaxerror at line {}: string, int or real expected", data->lineno);
            }
            break;
          default:
            FOUR_C_THROW("Syntaxerror at line {}: ':' or '=' expected", data->lineno);
        }
        break;
      }
      case tok_dedent:
        data->token_int--;
        goto end;
      default:
        FOUR_C_THROW("Syntaxerror at line {}: name expected", data->lineno);
    }

    lexan(data);
  }

end:
  return;
}

/// Legacy table manages strings itself, so we have to give it a full copy that can be freed later
static char* string_copy(ryml::csubstr in) { return strndup(in.data(), in.size()); }

static void parse_yaml_node(ryml::ConstNodeRef node, MAP* parent)
{
  if (node.is_map())
  {
    char* key = string_copy(node.key());

    MAP* map = new MAP;
    init_map(map);
    for (auto item : node.children())
    {
      parse_yaml_node(item, map);
    }

    map_insert_map(parent, map, key);
  }
  else
  {
    char* key = string_copy(node.key());

    if (node.is_val_dquo())
    {
      const auto value = node.val();
      map_insert_string(parent, string_copy(value), key);
      return;
    }

    if (node.has_val())
    {
      int int_value;
      auto status = Core::IO::read_value_from_yaml(Core::IO::ConstYamlNodeRef{node, ""}, int_value);
      if (status == Core::IO::YamlReadStatus::success)
      {
        map_insert_int(parent, int_value, key);
        return;
      }

      double real_value;
      status = Core::IO::read_value_from_yaml(Core::IO::ConstYamlNodeRef{node, ""}, real_value);
      if (status == Core::IO::YamlReadStatus::success)
      {
        map_insert_real(parent, real_value, key);
        return;
      }
    }

    FOUR_C_THROW("Unsupported YAML node type for key '{}'", key);
  }
}

static void parse_from_yaml(ParserData* data, MAP* map)
{
  auto tree = Core::IO::init_yaml_tree_with_exceptions();
  ryml::parse_in_place(ryml::substr(data->file_buffer, data->file_size), &tree);

  FOUR_C_ASSERT_ALWAYS(tree.rootref().is_seq(), "Expected a top-level sequence");
  for (auto item : tree.rootref().children())
  {
    FOUR_C_ASSERT_ALWAYS(item.is_map(), "Expected sequence items to be maps");
    FOUR_C_ASSERT_ALWAYS(
        item.num_children() == 1, "Expected single key-value pairs in top-level sequence");
    parse_yaml_node(item.child(0), map);
  }
}


/*----------------------------------------------------------------------*/
/*!
  \brief Parse the file given by name and fill the map with this
  file's content (serial only!)

*/
/*----------------------------------------------------------------------*/
void parse_control_file_serial(MAP* map, const char* filename)
{
  ParserData data;

  /*
   * So here we are. Before the symbol table can be filled with values
   * it has to be initialized. That is we expect to get an
   * uninitialized (virgin) map. */
  init_map(map);

  data.tok = tok_none;
  data.lineno = 1;
  data.pos = 0;
  data.indent_level = 0;
  data.indent_step = -1;

  int bytes_read;
  FILE* file;
  file = fopen(filename, "rb");

  if (file == nullptr)
  {
    FOUR_C_THROW("cannot read file '{}'", filename);
  }

  /* find out the control file size */
  fseek(file, 0, SEEK_END);
  data.file_size = ftell(file);

  /* read file to local buffer */
  data.file_buffer = (char*)malloc((data.file_size + 1) * sizeof(char));
  fseek(file, 0, SEEK_SET);
  /*bytes_read = fread(data.file_buffer, sizeof(char), data.file_size, file);*/
  bytes_read = fread(data.file_buffer, sizeof(char), (size_t)data.file_size, file);
  if (bytes_read != data.file_size)
  {
    FOUR_C_THROW("failed to read file {}", filename);
  }
  /* a trailing zero helps a lot */
  data.file_buffer[data.file_size] = '\0';

  fclose(file);

  parse_from_yaml(&data, map);
  destroy_parser_data(&data);
}


/*----------------------------------------------------------------------*/
/*!
  \brief Parse the file given by name and fill the map with this
  file's content.

*/
/*----------------------------------------------------------------------*/
void parse_control_file(MAP* map, const char* filename, MPI_Comm comm)
{
  ParserData data;

  /*
   * So here we are. Before the symbol table can be filled with values
   * it has to be initialized. That is we expect to get an
   * uninitialized (virgin) map. */
  init_map(map);

  init_parser_data(&data, filename, comm);
  parse_from_yaml(&data, map);
  destroy_parser_data(&data);
}

FOUR_C_NAMESPACE_CLOSE
