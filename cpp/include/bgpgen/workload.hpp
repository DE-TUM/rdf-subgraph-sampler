#pragma once

#include "bgpgen/dedup.hpp"
#include "bgpgen/graph.hpp"
#include "bgpgen/query.hpp"
#include "bgpgen/sampler_complex.hpp"
#include "bgpgen/sampler_star.hpp"
#include "bgpgen/subject_index.hpp"
#include "bgpgen/term_table.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bgpgen {

/// One sampling run as a user describes it. The command line and the Python
/// module both fill this struct, so they give the same workload for the same
/// values. The defaults are the defaults of the bgpgen binary.
struct WorkloadOptions {
  size_t num = 1000;  ///< queries to sample (per size)
  uint64_t seed = 42;
  /// Stars and paths: seed each size with a value derived from (seed, size).
  /// Off: every size uses `seed` itself.
  bool independent_sizes = true;
  DedupMethod dedup = DedupMethod::Symmetry;

  // --- stars ---
  double p_predicate = 1.0; ///< probability that a predicate is a constant
  int min_objects = 0;      ///< fewest objects that are constants
  int max_objects = -1;     ///< most objects that are constants; -1 = ceil(size / 2)

  // --- paths and templates ---
  double p_edge = 1.0;      ///< probability that a `?/B` predicate is a constant
  double p_node = 0.0;      ///< templates: probability that a `?x/B` node is a constant
  double p_start_end = 0.3; ///< paths: probability that an end node is a constant
  bool early_checks = true; ///< templates: check a cycle-closing edge as early as possible

  // Search options, see ComplexOptions for what each of them does.
  bool prune = false;
  bool auto_prune = true;
  uint64_t attempt_budget = 0;
  bool degree_check = true;
  bool pred_first = false;
  double novelty = 0.0;
  bool skip_known = false;
  size_t per_seed = 1; ///< 0 = auto
  size_t max_known = 0;
  bool retire_seeds = false;
  bool seed_groups = false;
  unsigned card_buckets = 0;
  double card_max = 1e6;
  bool card_steer = true;
  unsigned sample_threads = 1;
  ComplexOptions::Sampling sampling = ComplexOptions::Sampling::Dfs;
  double root_alpha = 1.0;
};

/// splitmix64 over (seed, size): unrelated streams for different sizes.
uint64_t derive_seed(uint64_t seed, unsigned size);

/// Star queries with `size` triple patterns. `fwd` is the out-index.
std::vector<StarRecord> sample_star_workload(const TripleStore& store, const SubjectIndex& fwd,
                                             const TermTable& terms, unsigned size,
                                             const WorkloadOptions& opt,
                                             SampleReport* report = nullptr);

/// Path queries with `size` triple patterns. With card_buckets the queries
/// come back counted; otherwise their y is -1 until count_cardinalities runs.
std::vector<Query> sample_path_workload(const TripleStore& store, const SubjectIndex& fwd,
                                        const SubjectIndex& rev, const TermTable& terms,
                                        unsigned size, const WorkloadOptions& opt,
                                        SampleReport* report = nullptr);

/// Queries of the shape given by a template string such as
/// "?x B ?y, ?y B ?z, ?z B ?x". Counting as for sample_path_workload.
std::vector<Query> sample_template_workload(const TripleStore& store, const SubjectIndex& fwd,
                                            const SubjectIndex& rev, const TermTable& terms,
                                            const std::string& shape, const WorkloadOptions& opt,
                                            SampleReport* report = nullptr);

} // namespace bgpgen
