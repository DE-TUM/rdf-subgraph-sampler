#pragma once

#include "bgpgen/dedup.hpp"
#include "bgpgen/query.hpp"
#include "bgpgen/subject_index.hpp"

#include <random>
#include <string>
#include <vector>

namespace bgpgen {

struct StarOptions {
  unsigned size = 4;        // triple patterns per query
  size_t num = 1000;        // target query count
  double p_predicate = 1.0; // probability a predicate is bound
  unsigned min_objects = 0; // bound-object count range, already clamped to size
  unsigned max_objects = 0;
};

/// Per-record metadata mirroring the Python star generator's extra fields.
struct StarMeta {
  TermId source_subject = kInvalidTerm;
  uint32_t star_size = 0;  // subject's full out-degree
  uint64_t query_hash = 0; // pattern hash (Python: md5; here: hash_bytes), hex in JSON
  std::vector<TermId> predicates_used;
  std::vector<TermId> instantiated_objects;
  std::vector<uint32_t> object_instantiation_pattern;
  uint32_t n_objects_instantiated = 0;
};

struct StarRecord {
  Query query;
  StarMeta meta;
};

struct SampleReport {
  size_t duplicates_skipped = 0;
  size_t reshuffles = 0;
  std::string stop_reason; // empty if the target count was reached

  // Filled by sample_complex only (path / complex shapes).
  size_t attempts = 0;          // start nodes tried
  size_t failed_matches = 0;    // attempts that found no match
  size_t gave_up = 0;           // of those: stopped by the attempt budget
  uint64_t expansions = 0;      // neighbour candidates looked at
  size_t start_candidates = 0;  // size of the start set
  double tree_matches = 0.0;    // spanning-tree matches in the graph (with weights)
  double weights_seconds = 0.0; // time spent computing match weights
  size_t skipped_starts = 0;    // start nodes ruled out by the degree check
  bool auto_pruned = false;     // auto_prune switched pruning on during the run
  std::vector<TermId> roots;    // root node of every emitted query, in order
};

/// Generates up to opt.num unique star queries. Loop semantics mirror the
/// Python generator: shuffle candidate subjects, pass through without
/// replacement, reshuffle between passes (max 300), stop on an unproductive
/// pass (0 new), a diminishing pass (<5 new), or max(500, n_stars/10)
/// consecutive duplicates.
std::vector<StarRecord> sample_stars(const TripleStore& store, const SubjectIndex& index,
                                     const TermTable& terms, const StarOptions& opt, Dedup& dedup,
                                     std::mt19937_64& rng, SampleReport* report = nullptr);

} // namespace bgpgen
