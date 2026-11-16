#pragma once

#include "bgpgen/dedup.hpp"
#include "bgpgen/query.hpp"
#include "bgpgen/sampler_star.hpp" // SampleReport
#include "bgpgen/subject_index.hpp"
#include "bgpgen/template.hpp"

#include <random>
#include <vector>

namespace bgpgen {

struct ComplexOptions {
  size_t num = 1000;
  double p_edge = 1.0;
  double p_node = 0.0; // for path templates this carries p_start_end
  size_t max_reshuffles = 30;
  size_t min_new_per_pass = 5; // 0 disables the diminishing-returns stop (path)
  size_t max_consecutive = 0;  // 0 = max(500, n_subjects/10); path passes 200

  // --- Match-weight options (all off = the Python-compatible sampler) ---

  /// DFS as before, but start nodes and neighbours are restricted to those
  /// with a positive match weight (no dead ends on tree-shaped templates).
  bool prune = false;

  /// DFS only: start without pruning and switch it on once the search
  /// overhead (candidates looked at beyond the one per step that a found
  /// match needs) has cost as much as the match-weight pass would. A sampler
  /// with little overhead never pays for the pass. Nodes already tried in the
  /// current pass are not tried again. Also ends the run at once when no node
  /// can root the template.
  bool auto_prune = false;

  /// DFS only: give up on a start node after this many neighbour candidates
  /// (0 = no limit). See Matcher::set_attempt_budget.
  uint64_t attempt_budget = 0;

  /// Skip start nodes and neighbours that have fewer different neighbours
  /// than their template node (out, in, and both together). No match is lost: different
  /// template nodes always get different graph nodes. With pruning the check
  /// is part of the match weights, so it also rules out nodes whose
  /// neighbours all fail it. Changes the order of random draws.
  bool degree_check = false;

  enum class Sampling {
    Dfs,     // shuffled start nodes, randomized DFS, first match wins
    Weighted // root ~ weight^root_alpha, neighbours ~ weight, no backtracking
  };
  Sampling sampling = Sampling::Dfs;

  /// Weighted only. A match below root r is drawn with probability
  /// weight(r)^(root_alpha - 1) / Z: 1 = every match equally likely,
  /// 0 = every feasible root equally likely.
  double root_alpha = 1.0;

  /// Weighted only: stop after this many draws. 0 = max(1e6, 1000 * num).
  size_t max_attempts = 0;
};

/// Generates up to opt.num unique queries matching the compiled template.
/// Start candidates are IRI subjects with at least one IRI out-edge (the
/// Python generators' adjacency keys). Loop semantics mirror the Python
/// complex generator; path differs only in the constants (see ComplexOptions).
/// The match-weight options change the start set and the search as
/// described on ComplexOptions.
std::vector<Query> sample_complex(const TripleStore& store, const SubjectIndex& fwd,
                                  const SubjectIndex& rev, const TermTable& terms,
                                  const CompiledTemplate& tmpl, const ComplexOptions& opt,
                                  Dedup& dedup, std::mt19937_64& rng,
                                  SampleReport* report = nullptr);

} // namespace bgpgen
