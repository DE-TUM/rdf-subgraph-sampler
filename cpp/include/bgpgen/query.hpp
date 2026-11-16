#pragma once

#include "bgpgen/term_table.hpp"
#include "bgpgen/types.hpp"

#include <string>
#include <vector>

namespace bgpgen {

/// Query term: a TermId constant or a variable id, tagged in the top bit.
/// Variable ids index Query::var_names.
using QTerm = uint32_t;
inline constexpr QTerm kQVarTag = 0x80000000u;
constexpr QTerm qvar(uint32_t var_id) { return var_id | kQVarTag; }
constexpr bool q_is_var(QTerm t) { return (t & kQVarTag) != 0; }
constexpr uint32_t q_var_id(QTerm t) { return t & ~kQVarTag; }

struct TriplePattern {
  QTerm s, p, o;
};

/// A sampled BGP, fully integer-coded; strings are materialized only at
/// output time.
struct Query {
  std::vector<TriplePattern> patterns;
  std::vector<std::string> var_names; // var id -> name including '?'
  std::vector<TermId> x;              // instantiated entities (Python's "x")
  int64_t y = -1;                     // cardinality, filled by the counter
};

/// Token as it appears in the Python tool's output: IRIs in <>, literals as
/// their raw token, blank nodes as _:label.
inline std::string term_token(const TermTable& terms, TermId id) {
  std::string_view lex = terms.lexical(id);
  if (terms.is_iri(id)) return "<" + std::string(lex) + ">";
  if (terms.is_bnode(id)) return "_:" + std::string(lex);
  return std::string(lex);
}

inline std::string q_token(const Query& q, const TermTable& terms, QTerm t) {
  return q_is_var(t) ? q.var_names[q_var_id(t)] : term_token(terms, t);
}

/// "SELECT * WHERE { tp1 . tp2 ... }" — exactly the Python generators' shape.
inline std::string render_sparql(const Query& q, const TermTable& terms) {
  std::string out = "SELECT * WHERE { ";
  for (size_t i = 0; i < q.patterns.size(); ++i) {
    if (i) out += " . ";
    const TriplePattern& tp = q.patterns[i];
    out += q_token(q, terms, tp.s);
    out += ' ';
    out += q_token(q, terms, tp.p);
    out += ' ';
    out += q_token(q, terms, tp.o);
  }
  out += " }";
  return out;
}

} // namespace bgpgen
