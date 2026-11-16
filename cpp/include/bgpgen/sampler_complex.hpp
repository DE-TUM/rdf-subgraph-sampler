#pragma once

#include "bgpgen/cardinality.hpp"
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

  /// DFS only: choose a predicate of the node first, then a neighbour under
  /// it (see Matcher::set_predicate_first). `novelty` > 0 prefers predicates
  /// that the queries emitted so far use less often at the same template edge.
  bool pred_first = false;
  double novelty = 0.0;

  /// DFS only: a match that gives a query already in the workload does not
  /// end the search on that start node; it goes on until it finds a new
  /// query, runs out of candidates or reaches the attempt budget. Which
  /// positions become constants is then drawn before the search.
  bool skip_known = false;

  /// With skip_known: end the search on a start node after it has met this
  /// many known queries (0 = no limit other than the attempt budget). Near a
  /// start node whose queries are mostly known, most further matches are
  /// known too, and a fresh start node is the better try.
  size_t max_known = 0;

  /// With skip_known: how many new queries one visit of a start node may
  /// give (the search goes on from the last match, so the later ones are
  /// cheap, and they share most of their predicates with the first).
  /// 0 = one in the first pass over the start nodes, twice as many in every
  /// further pass (at most 64).
  size_t per_seed = 1;

  /// With skip_known: a start node that gave no new query in a pass is left
  /// out of all later passes.
  bool retire_seeds = false;

  /// DFS only: order the start nodes so that nodes with different sets of
  /// out-predicates come first (one node of every set, then a second one of
  /// every set that has one, and so on; random inside). Without it the order
  /// is uniformly random, so a set shared by many nodes is visited often.
  bool seed_groups = false;

  /// Cardinality targets: the workload gets the same number of queries in
  /// each of `buckets` ranges of the result size, equally wide on a log
  /// scale between 1 and `max_card`. Every new query is counted when it is
  /// found and dropped if its range is full or it lies outside.
  /// Implies skip_known. The search on a start node ends at the first query
  /// that is known or dropped, unless max_known is set (then known and
  /// dropped queries count together).
  struct CardTargets {
    unsigned buckets = 0;                // 0 = off
    double max_card = 1e6;               // upper end of the last range (exclusive)
    uint64_t count_budget = 2'000'000;   // enumeration steps per count; over = too large
    /// Steer the predicate choice (implies pred_first): every range learns
    /// an exponent g and chooses predicates with weight (triples with the
    /// predicate)^g. A count above the aimed range lowers g, one below
    /// raises it.
    bool steer = true;
    /// Stop after this many counted queries in a row that were dropped.
    /// 0 = max(5000, 20 * num).
    size_t max_dropped_in_a_row = 0;
  };
  CardTargets card;

  /// DFS only: threads that search at the same time (implies skip_known).
  /// The start nodes are handed out in rounds; inside a round a thread does
  /// not see what the others find, and what two threads both find is kept
  /// once. The workload is the same for the same seed and number of threads,
  /// and differs between numbers of threads.
  unsigned threads = 1;

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
