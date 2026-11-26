#include "bgpgen/workload.hpp"

#include "bgpgen/template.hpp"

#include <algorithm>
#include <random>
#include <utility>

namespace bgpgen {

namespace {

/// Copies the search options that paths and templates share.
ComplexOptions match_options(const WorkloadOptions& opt) {
  if (opt.card_buckets && opt.card_max <= 1.0) throw BgpError("--card-max must be above 1");
  ComplexOptions co;
  co.num = opt.num;
  co.p_edge = opt.p_edge;
  co.prune = opt.prune;
  co.auto_prune = opt.auto_prune;
  co.sampling = opt.sampling;
  co.root_alpha = opt.root_alpha;
  co.attempt_budget = opt.attempt_budget;
  co.degree_check = opt.degree_check;
  co.pred_first = opt.pred_first;
  co.novelty = opt.novelty;
  co.skip_known = opt.skip_known;
  co.per_seed = opt.per_seed;
  co.max_known = opt.max_known;
  co.retire_seeds = opt.retire_seeds;
  co.seed_groups = opt.seed_groups;
  co.card.buckets = opt.card_buckets;
  co.card.max_card = opt.card_max;
  co.card.steer = opt.card_steer;
  co.threads = opt.sample_threads;
  return co;
}

} // namespace

uint64_t derive_seed(uint64_t seed, unsigned size) {
  uint64_t z = seed + 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(size) + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

std::vector<StarRecord> sample_star_workload(const TripleStore& store, const SubjectIndex& fwd,
                                             const TermTable& terms, unsigned size,
                                             const WorkloadOptions& opt, SampleReport* report) {
  StarOptions so;
  so.size = size;
  so.num = opt.num;
  so.p_predicate = opt.p_predicate;
  // Python clamps: 0 <= min,max <= size; swap if min > max; default
  // max = ceil(size/2).
  unsigned min_o = std::min<unsigned>(static_cast<unsigned>(std::max(opt.min_objects, 0)), size);
  unsigned max_o = opt.max_objects < 0 ? (size + 1) / 2
                                       : std::min(static_cast<unsigned>(opt.max_objects), size);
  if (min_o > max_o) std::swap(min_o, max_o);
  so.min_objects = min_o;
  so.max_objects = max_o;

  // A star is a template too: with it the symmetry key applies and the
  // slow WL form is not needed (same queries as --dedup wl).
  std::string star_shape;
  for (unsigned i = 0; i < size; ++i)
    star_shape += (i ? ", ?s B ?o" : "?s B ?o") + std::to_string(i);
  CompiledTemplate star_tmpl = parse_template(star_shape);
  Dedup dedup(opt.dedup);
  dedup.set_template(star_tmpl);
  std::mt19937_64 rng(opt.independent_sizes ? derive_seed(opt.seed, size) : opt.seed);
  return sample_stars(store, fwd, terms, so, dedup, rng, report);
}

std::vector<Query> sample_path_workload(const TripleStore& store, const SubjectIndex& fwd,
                                        const SubjectIndex& rev, const TermTable& terms,
                                        unsigned size, const WorkloadOptions& opt,
                                        SampleReport* report) {
  // A path is a chain template; p_node carries p_start_end, and the two
  // stopping constants are those of the Python path sampler.
  CompiledTemplate tmpl = make_chain_template(size);
  ComplexOptions co = match_options(opt);
  co.p_node = opt.p_start_end;
  co.min_new_per_pass = 0;
  co.max_consecutive = 200;

  Dedup dedup(opt.dedup);
  std::mt19937_64 rng(opt.independent_sizes ? derive_seed(opt.seed, size) : opt.seed);
  return sample_complex(store, fwd, rev, terms, tmpl, co, dedup, rng, report);
}

std::vector<Query> sample_template_workload(const TripleStore& store, const SubjectIndex& fwd,
                                            const SubjectIndex& rev, const TermTable& terms,
                                            const std::string& shape, const WorkloadOptions& opt,
                                            SampleReport* report) {
  CompiledTemplate tmpl = parse_template(shape);
  if (opt.early_checks) reorder_checks_early(tmpl);
  ComplexOptions co = match_options(opt);
  co.p_node = opt.p_node;

  Dedup dedup(opt.dedup);
  std::mt19937_64 rng(opt.seed);
  return sample_complex(store, fwd, rev, terms, tmpl, co, dedup, rng, report);
}

} // namespace bgpgen
