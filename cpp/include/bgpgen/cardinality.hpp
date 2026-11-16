#pragma once

#include "bgpgen/graph.hpp"
#include "bgpgen/query.hpp"
#include "bgpgen/subject_index.hpp"

#include <span>
#include <vector>

namespace bgpgen {

struct CardinalityOptions {
  /// Enumeration steps allowed per query; exhausted -> y = -1 (the analog of
  /// the old endpoint timeout).
  uint64_t budget = 50'000'000;
  unsigned threads = 0; // 0 = all hardware threads
};

/// Exact SPARQL COUNT(*) of the query over the data — homomorphism semantics
/// (no injectivity), matching what an endpoint returns. Variables occurring
/// once are never enumerated: they contribute closed-form factors off the
/// CSR spans (a star is one pass over the hub candidates, not a join).
/// Returns -1 if the budget runs out; saturates at INT64_MAX.
///
/// `fwd`/`rev` are the subject-/object-major indexes over the full data
/// (the same ones the samplers use) — needed for variable-predicate
/// patterns, where a node's row length is the extension count.
int64_t count_query(const TripleStore& store, const SubjectIndex& fwd, const SubjectIndex& rev,
                    const Query& q, uint64_t budget = 50'000'000);

/// Fills q->y for every query, in parallel over an atomic work index.
void count_cardinalities(const TripleStore& store, const SubjectIndex& fwd,
                         const SubjectIndex& rev, std::span<Query* const> queries,
                         const CardinalityOptions& opt = {});

inline void count_cardinalities(const TripleStore& store, const SubjectIndex& fwd,
                                const SubjectIndex& rev, std::vector<Query>& queries,
                                const CardinalityOptions& opt = {}) {
  std::vector<Query*> ptrs;
  ptrs.reserve(queries.size());
  for (Query& q : queries) ptrs.push_back(&q);
  count_cardinalities(store, fwd, rev, ptrs, opt);
}

} // namespace bgpgen
