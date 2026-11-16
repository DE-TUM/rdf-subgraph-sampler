#pragma once

#include "bgpgen/match_weights.hpp"
#include "bgpgen/subject_index.hpp"
#include "bgpgen/template.hpp"

#include <random>
#include <vector>

namespace bgpgen {

/// DFS subgraph matcher over the IRI->IRI subgraph (the Python generators'
/// adjacency only ever contained <s> <p> <o> triples).
///
/// Recursive DFS with undo: a fixed node_assign array per slot, injectivity
/// via linear scan, candidates drawn as up to max_candidates random distinct
/// offsets into the adjacency row (incremental Fisher-Yates over reusable
/// per-depth index buffers — full rows are never materialized or shuffled).
class Matcher {
public:
  Matcher(const TripleStore& store, const SubjectIndex& fwd, const SubjectIndex& rev,
          const CompiledTemplate& tmpl, TermId bnode_base);

  /// Optional match counts (must outlive the matcher, and be computed for
  /// the same template). With weights set, match() skips neighbours that
  /// cannot complete the spanning tree; sample_weighted() requires them.
  void set_weights(const MatchWeights* weights) { weights_ = weights; }

  /// match() gives up on a start node after looking at this many neighbour
  /// candidates (0 = search until every capped branch is exhausted). Without
  /// a budget one start node that has no match can cost up to
  /// max_candidates^steps candidates.
  void set_attempt_budget(uint64_t n) { budget_ = n; }
  /// True if the last match() stopped because of the budget.
  bool gave_up() const { return gave_up_; }

  /// With the check on, match() skips neighbours that have fewer different
  /// neighbours than their slot has in the template (out, in, and both
  /// together); no match can use them, since different slots get different
  /// nodes.
  void set_degree_check(bool on) { degree_check_ = on; }
  /// False if the degree check is on and rules `start` out as the root.
  bool can_root(TermId start) const {
    return !degree_check_ || degree_ok(tmpl_.root_slot, start);
  }

  /// Tries to match the template with the root slot pinned to `start`.
  /// On success node_assign()/pred_assign() hold the match.
  bool match(TermId start, std::mt19937_64& rng);

  /// One draw without backtracking: every neighbour is chosen with
  /// probability proportional to its weight, so below a given root each
  /// spanning-tree match has probability 1 / weight(root). Returns false
  /// (reject the whole draw) if the draw reuses a graph node or a
  /// cycle-closing edge is missing; rejecting keeps the accepted matches
  /// equally likely.
  bool sample_weighted(TermId start, std::mt19937_64& rng);

  const std::vector<TermId>& node_assign() const { return node_assign_; }
  const std::vector<TermId>& pred_assign() const { return pred_assign_; } // per edge, TermIds

  /// Neighbour candidates looked at so far (work counter for benchmarks).
  uint64_t expansions() const { return expansions_; }

private:
  bool dfs(size_t step_idx, std::mt19937_64& rng);
  bool used(TermId node) const;
  bool degree_ok(uint8_t slot, TermId node) const {
    return enough_neighbours(fwd_, rev_, node, deg_.out[slot], deg_.in[slot], deg_.all[slot],
                             bnode_base_);
  }

  const TripleStore& store_;
  const SubjectIndex& fwd_;
  const SubjectIndex& rev_;
  const CompiledTemplate& tmpl_;
  TermId bnode_base_;
  const MatchWeights* weights_ = nullptr;
  SlotDegrees deg_;
  bool degree_check_ = false;

  std::vector<TermId> node_assign_; // per slot; kInvalidTerm = unassigned
  std::vector<TermId> pred_assign_; // per edge
  std::vector<std::vector<uint32_t>> idx_buf_;     // per step depth; identity between calls
  std::vector<std::vector<uint32_t>> touched_buf_; // per step depth: swapped positions
  std::vector<TermId> check_preds_;            // scratch for Check steps
  uint64_t expansions_ = 0;
  uint64_t budget_ = 0;        // per match(); 0 = none
  uint64_t attempt_start_ = 0; // expansions_ at the start of match()
  bool gave_up_ = false;
};

} // namespace bgpgen
