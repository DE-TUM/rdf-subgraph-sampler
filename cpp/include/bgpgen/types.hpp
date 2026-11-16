#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace bgpgen {

/// Dictionary-encoded RDF term. The id space is segmented by kind:
/// [0, n_iris) are IRIs, [n_iris, n_iris + n_bnodes) are blank nodes, and the
/// rest are literals. See TermTable.
using TermId = uint32_t;

/// Dense per-data-graph predicate index (see TripleStore).
using PredIndex = uint32_t;

inline constexpr TermId kInvalidTerm = 0xFFFFFFFFu;
inline constexpr PredIndex kNoPred = 0xFFFFFFFFu; ///< Predicate absent from the data graph.

enum class TermKind : uint8_t { Iri, BNode, Literal };

struct Triple {
  TermId s, p, o;
};

/// Thrown for unrecoverable problems: unreadable files, RDF syntax errors,
/// and ill-formed templates or arguments.
class BgpError : public std::runtime_error {
public:
  explicit BgpError(const std::string& what) : std::runtime_error(what) {}
};

} // namespace bgpgen
