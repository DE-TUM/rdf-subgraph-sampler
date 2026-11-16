#pragma once

#include "bgpgen/match_weights.hpp"
#include "bgpgen/subject_index.hpp"
#include "bgpgen/template.hpp"

#include <functional>
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

  /// Predicate first: at every step choose a predicate of the known node,
  /// then one of its neighbours under that predicate. Without it a neighbour
  /// entry is chosen, so a predicate is chosen in proportion to its edges at
  /// the node. A predicate p at template edge e is chosen with weight
  /// 1 / (1 + uses[e * n_preds + p])^novelty; novelty 0 = every predicate of
  /// the node equally likely. `uses` (may be null if novelty is 0) is read on
  /// every step and owned by the caller. A leaf of the template that is a
  /// variable gets one neighbour per predicate: another neighbour under the
  /// same predicate would give the same query.
  void set_predicate_first(bool on, double novelty = 0.0, const uint32_t* uses = nullptr,
                           size_t n_preds = 0) {
    pred_first_ = on;
    novelty_ = novelty;
    uses_ = uses;
    n_preds_ = n_preds;
  }

  /// Predicate first only: multiplies the weight of predicate p by bias[p]
  /// (one entry per predicate of the data; null = no bias).
  void set_pred_bias(const double* bias) { bias_ = bias; }

  /// Called on every complete match. If it returns false the search goes on
  /// as if the match had failed (the caller knows the query already, or wants
  /// more matches from this start node). Not used by sample_weighted().
  void set_on_match(std::function<bool()> fn) { on_match_ = std::move(fn); }
  /// Gives the search a fresh attempt budget; for use inside on_match.
  void reset_budget() { attempt_start_ = expansions_; }

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
  const std::vector<PredIndex>& pred_index_assign() const { return pred_index_assign_; }

  /// Neighbour candidates looked at so far (work counter for benchmarks).
  uint64_t expansions() const { return expansions_; }

private:
  bool dfs(size_t step_idx, std::mt19937_64& rng);
  bool extend_pred_first(size_t step_idx, std::span<const SubjectIndex::Entry> row,
                         std::mt19937_64& rng);
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
  std::vector<PredIndex> pred_index_assign_; // per edge
  std::vector<std::vector<uint32_t>> idx_buf_;     // per step depth; identity between calls
  std::vector<std::vector<uint32_t>> touched_buf_; // per step depth: swapped positions
  std::vector<TermId> check_preds_;            // scratch for Check steps
  std::vector<PredIndex> check_pidx_;          // scratch for Check steps

  // Predicate first.
  struct Group {
    uint32_t begin, end; // row offsets of the entries with one predicate
    uint32_t taken;      // entries already tried
    double weight;       // 0 = no longer chosen
  };
  bool pred_first_ = false;
  double novelty_ = 0.0;
  const uint32_t* uses_ = nullptr;
  const double* bias_ = nullptr;
  size_t n_preds_ = 0;
  std::vector<std::vector<Group>> group_buf_; // per step depth
  const SubjectIndex::Entry* bounds_row_ = nullptr; // row whose group starts are in bounds_
  std::vector<uint32_t> bounds_;                    // group starts, then the row size
  std::vector<uint8_t> leaf_var_;             // per slot: one edge, always a variable
  std::function<bool()> on_match_;
  uint64_t expansions_ = 0;
  uint64_t budget_ = 0;        // per match(); 0 = none
  uint64_t attempt_start_ = 0; // expansions_ at the start of match()
  bool gave_up_ = false;
};

} // namespace bgpgen
