#include "bgpgen/sampler_complex.hpp"

#include "bgpgen/matcher.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>

namespace bgpgen {

namespace {

/// Which positions of the query become constants: one decision per triple
/// and position, drawn for the Prob positions.
using Bindings = std::vector<std::array<bool, 3>>;

/// Draws per-occurrence binding decisions for Prob positions exactly like the
/// Python generators (note: strict '<', unlike the star sampler's '<=').
void draw_bindings(const CompiledTemplate& tmpl, double p_edge, double p_node,
                   std::mt19937_64& rng, Bindings& b) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  b.resize(tmpl.triples.size());
  for (size_t i = 0; i < tmpl.triples.size(); ++i) {
    const TemplateTriple& tt = tmpl.triples[i];
    b[i][0] = tt.s_bind == Bind::Bound || (tt.s_bind == Bind::Prob && unit(rng) < p_node);
    b[i][1] = tt.p_bind == Bind::Bound || (tt.p_bind == Bind::Prob && unit(rng) < p_edge);
    b[i][2] = tt.o_bind == Bind::Bound || (tt.o_bind == Bind::Prob && unit(rng) < p_node);
  }
}

/// Builds the output query from a complete match.
Query build_query(const CompiledTemplate& tmpl, const Matcher& m, const Bindings& b) {
  Query q;
  std::vector<int> slot_var(tmpl.n_slots, -1);
  auto var_for_slot = [&](uint8_t slot) -> QTerm {
    if (slot_var[slot] < 0) {
      slot_var[slot] = static_cast<int>(q.var_names.size());
      q.var_names.push_back(tmpl.slot_names[slot]);
    }
    return qvar(static_cast<uint32_t>(slot_var[slot]));
  };

  for (size_t i = 0; i < tmpl.triples.size(); ++i) {
    const TemplateTriple& tt = tmpl.triples[i];
    TermId s_uri = m.node_assign()[tt.s_slot];
    TermId p_uri = m.pred_assign()[i];
    TermId o_uri = m.node_assign()[tt.o_slot];
    const bool s_bound = b[i][0], p_bound = b[i][1], o_bound = b[i][2];

    QTerm s = s_bound ? s_uri : var_for_slot(tt.s_slot);
    QTerm p;
    if (p_bound) {
      p = p_uri;
    } else {
      p = qvar(static_cast<uint32_t>(q.var_names.size()));
      q.var_names.push_back(tmpl.pred_var_names[i]);
    }
    QTerm o = o_bound ? o_uri : var_for_slot(tt.o_slot);

    // x order per triple: complex appends (s, p, o), path (s, o, p).
    if (s_bound) q.x.push_back(s_uri);
    if (tmpl.path_entity_order) {
      if (o_bound) q.x.push_back(o_uri);
      if (p_bound) q.x.push_back(p_uri);
    } else {
      if (p_bound) q.x.push_back(p_uri);
      if (o_bound) q.x.push_back(o_uri);
    }
    q.patterns.push_back({s, p, o});
  }
  return q;
}

Query build_query(const CompiledTemplate& tmpl, const Matcher& m, double p_edge, double p_node,
                  std::mt19937_64& rng) {
  Bindings b;
  draw_bindings(tmpl, p_edge, p_node, rng, b);
  return build_query(tmpl, m, b);
}

/// Reorders the start nodes by their set of out-predicates: first one node
/// of every set, then a second node of every set that has one, and so on.
/// The nodes of one set are in random order, and so are the sets in a round.
void order_by_group(std::vector<TermId>& subjects, const SubjectIndex& fwd,
                    std::mt19937_64& rng) {
  struct Key {
    uint64_t sig;
    uint32_t r;
    TermId node;
  };
  std::vector<Key> keys;
  keys.reserve(subjects.size());
  for (TermId v : subjects) {
    uint64_t h = 1469598103934665603ull;
    PredIndex last = kNoPred;
    for (const SubjectIndex::Entry& e : fwd.row_of(v))
      if (e.pred != last) {
        last = e.pred;
        h = (h ^ e.pred) * 1099511628211ull;
      }
    keys.push_back({h, static_cast<uint32_t>(rng()), v});
  }
  std::sort(keys.begin(), keys.end(),
            [](const Key& a, const Key& b) { return a.sig != b.sig ? a.sig < b.sig : a.r < b.r; });

  // Round k holds the k-th node of every set: count the rounds' sizes, place.
  std::vector<uint32_t> round(keys.size());
  std::vector<size_t> start;
  for (size_t i = 0; i < keys.size(); ++i) {
    round[i] = i > 0 && keys[i].sig == keys[i - 1].sig ? round[i - 1] + 1 : 0;
    if (round[i] >= start.size()) start.resize(round[i] + 1, 0);
    ++start[round[i]];
  }
  size_t acc = 0;
  for (size_t& c : start) {
    size_t n = c;
    c = acc;
    acc += n;
  }
  std::vector<size_t> cursor = start;
  for (size_t i = 0; i < keys.size(); ++i) subjects[cursor[round[i]]++] = keys[i].node;
  for (size_t k = 0; k < start.size(); ++k)
    std::shuffle(subjects.begin() + static_cast<std::ptrdiff_t>(start[k]),
                 subjects.begin() + static_cast<std::ptrdiff_t>(cursor[k]), rng);
}

/// Cardinality targets (ComplexOptions::CardTargets).
class Buckets {
public:
  Buckets(const ComplexOptions& opt, const TripleStore& store, SampleReport& rep)
      : n_(opt.card.buckets), log_max_(std::log10(opt.card.max_card)), rep_(rep) {
    rep.bucket_fill.assign(n_, 0);
    for (unsigned b = 0; b < n_; ++b) cap_.push_back(opt.num / n_ + (b < opt.num % n_));
    if (!opt.card.steer) return;
    // Exponents -2, -1.75, ..., 2; every range starts at 0.
    grid_.resize(17);
    for (size_t g = 0; g < grid_.size(); ++g) {
      double exponent = -2.0 + 0.25 * static_cast<double>(g);
      for (PredIndex p = 0; p < store.predicate_count(); ++p)
        grid_[g].push_back(std::pow(static_cast<double>(store.triples_with(p)), exponent));
    }
    exponent_.assign(n_, 8);
  }

  bool steer() const { return !grid_.empty(); }
  unsigned size() const { return n_; }

  /// The range of a result size (-1 = over the count budget); size() = outside.
  unsigned range_of(int64_t y) const {
    if (y < 1) return n_;
    double pos = std::log10(static_cast<double>(y)) / log_max_ * n_;
    return pos < n_ ? static_cast<unsigned>(pos) : n_;
  }

  /// True if range b still lacks queries (`extra`: queries not yet added).
  bool open(unsigned b, const std::vector<size_t>* extra = nullptr) const {
    return rep_.bucket_fill[b] + (extra ? (*extra)[b] : 0) < cap_[b];
  }
  void add(unsigned b) { ++rep_.bucket_fill[b]; }

  /// Picks the range to aim at (an open one, in proportion to what it still
  /// lacks) and the predicate weights for it.
  struct Aim {
    unsigned target;
    const double* bias;
  };
  Aim aim(std::mt19937_64& rng) const {
    size_t lacking = 0;
    for (unsigned b = 0; b < n_; ++b) lacking += cap_[b] - rep_.bucket_fill[b];
    if (lacking == 0) return {0, grid_[exponent_[0]].data()};
    size_t r = rng() % lacking;
    unsigned target = 0;
    for (; target + 1 < n_; ++target) {
      size_t lack = cap_[target] - rep_.bucket_fill[target];
      if (r < lack) break;
      r -= lack;
    }
    return {target, grid_[exponent_[target]].data()};
  }

  /// A query found while aiming at `target` fell into range b: moves the
  /// exponent of `target` towards it.
  void feedback(unsigned target, unsigned b) {
    if (!steer()) return;
    size_t& e = exponent_[target];
    if (b > target && e > 0) --e;
    if (b < target && e + 1 < grid_.size()) ++e;
  }

private:
  unsigned n_;
  double log_max_;
  SampleReport& rep_;
  std::vector<size_t> cap_;
  std::vector<std::vector<double>> grid_; // per exponent: weight of every predicate
  std::vector<size_t> exponent_;          // per range: index into grid_
};

/// One thread of a round (ComplexOptions::threads > 1). It searches from its
/// share of the round's start nodes and keeps what it finds; the known
/// queries, the predicate uses and the ranges are only read during a round
/// and brought up to date between rounds.
struct Worker {
  Worker(const TripleStore& store, const SubjectIndex& fwd, const SubjectIndex& rev,
         const CompiledTemplate& tmpl, TermId bnode_base, DedupMethod method)
      : matcher(store, fwd, rev, tmpl, bnode_base), keymaker(method) {
    keymaker.set_template(tmpl);
  }
  struct Found {
    Query q;
    std::string key;
    std::vector<PredIndex> preds;
    unsigned range;
  };
  struct Tried {
    TermId node;
    uint32_t found = 0, known = 0, dropped = 0;
    bool can_root = true, gave_up = false;
  };
  Matcher matcher;
  Dedup keymaker; // for keys only; the set is the caller's
  std::mt19937_64 rng;
  Bindings bindings;
  std::string key;
  unsigned target = 0;
  // Results of the round, in the order of the thread's start nodes.
  std::vector<Tried> tried;
  std::vector<Found> found;
  std::unordered_set<std::string> keys;              // found or dropped in this round
  std::vector<std::string> dropped_keys;             // counted and dropped
  std::vector<std::pair<unsigned, unsigned>> counts; // (range aimed at, range of the count)
  std::vector<size_t> fill;                          // found per range
  size_t counted = 0, dropped_outside = 0, dropped_full = 0;
  double count_seconds = 0.0;
};

} // namespace

std::vector<Query> sample_complex(const TripleStore& store, const SubjectIndex& fwd,
                                  const SubjectIndex& rev, const TermTable& terms,
                                  const CompiledTemplate& tmpl, const ComplexOptions& options,
                                  Dedup& dedup, std::mt19937_64& rng, SampleReport* report) {
  ComplexOptions opt = options;
  if (opt.card.buckets) {
    opt.skip_known = true;
    if (opt.card.steer) opt.pred_first = true;
  }
  if (opt.threads > 1) opt.skip_known = true;
  std::vector<Query> out;
  SampleReport local;
  SampleReport& rep = report ? *report : local;
  TermId bnode_base = terms.bnode_base();

  const bool weighted = opt.sampling == ComplexOptions::Sampling::Weighted;
  MatchWeights weights;
  if (opt.prune || weighted) {
    auto t0 = std::chrono::steady_clock::now();
    weights.compute(fwd, rev, tmpl, bnode_base, opt.degree_check);
    rep.weights_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    rep.tree_matches = weights.total();
  }

  Matcher matcher(store, fwd, rev, tmpl, bnode_base);
  matcher.set_attempt_budget(opt.attempt_budget);
  matcher.set_degree_check(opt.degree_check);
  dedup.set_template(tmpl);

  if (weighted) {
    // Roots with a positive weight, drawn with replacement ~ weight^alpha.
    std::vector<TermId> roots;
    std::vector<double> cum;
    double acc = 0.0;
    for (TermId v = 0; v < bnode_base; ++v) {
      double w = weights.weight(tmpl.root_slot, v);
      if (!(w > 0.0)) continue;
      acc += opt.root_alpha == 1.0 ? w : opt.root_alpha == 0.0 ? 1.0 : std::pow(w, opt.root_alpha);
      roots.push_back(v);
      cum.push_back(acc);
    }
    rep.start_candidates = roots.size();
    if (roots.empty()) {
      rep.stop_reason = "no node can root the template";
      return out;
    }
    const size_t max_consecutive =
        opt.max_consecutive ? opt.max_consecutive : std::max<size_t>(500, roots.size() / 10);
    const size_t max_attempts =
        opt.max_attempts ? opt.max_attempts : std::max<size_t>(1'000'000, 1000 * opt.num);
    matcher.set_weights(&weights);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    size_t consecutive_dups = 0;

    while (out.size() < opt.num) {
      if (rep.attempts >= max_attempts) {
        rep.stop_reason = "attempt budget reached (" + std::to_string(max_attempts) + ")";
        break;
      }
      size_t pos = static_cast<size_t>(
          std::upper_bound(cum.begin(), cum.end(), unit(rng) * acc) - cum.begin());
      if (pos >= roots.size()) pos = roots.size() - 1;
      ++rep.attempts;
      if (!matcher.sample_weighted(roots[pos], rng)) {
        ++rep.failed_matches;
        continue;
      }
      Query q = build_query(tmpl, matcher, opt.p_edge, opt.p_node, rng);
      if (!dedup.insert(q, terms)) {
        ++rep.duplicates_skipped;
        if (++consecutive_dups >= max_consecutive) {
          rep.stop_reason =
              "too many consecutive duplicates (" + std::to_string(consecutive_dups) + ")";
          break;
        }
        continue;
      }
      consecutive_dups = 0;
      rep.roots.push_back(roots[pos]);
      out.push_back(std::move(q));
    }
    rep.expansions = matcher.expansions();
    return out;
  }

  // Start candidates: IRI subjects with at least one IRI out-edge, or with
  // pruning the nodes that can root the template's spanning tree.
  std::vector<TermId> subjects;
  if (opt.prune) {
    for (TermId v = 0; v < bnode_base; ++v)
      if (weights.weight(tmpl.root_slot, v) > 0.0) subjects.push_back(v);
    matcher.set_weights(&weights);
  } else {
    for (size_t i = 0; i < fwd.iri_subject_count(); ++i) {
      for (const SubjectIndex::Entry& e : fwd.row(i)) {
        if (e.obj < bnode_base) {
          subjects.push_back(fwd.subject_at(i));
          break;
        }
      }
    }
  }
  rep.start_candidates = subjects.size();
  if (subjects.empty()) {
    rep.stop_reason = opt.prune ? "no node can root the template" : "no IRI->IRI edges in the data";
    return out;
  }
  const size_t max_consecutive =
      opt.max_consecutive ? opt.max_consecutive : std::max<size_t>(500, subjects.size() / 10);

  // Predicate first, with the number of uses of every predicate at every
  // template edge in the queries emitted so far.
  const size_t n_preds = store.predicate_count();
  std::vector<uint32_t> uses;
  if (opt.pred_first) {
    if (opt.novelty != 0.0) uses.assign(tmpl.triples.size() * n_preds, 0);
    matcher.set_predicate_first(true, opt.novelty, uses.empty() ? nullptr : uses.data(), n_preds);
  }
  auto note_uses = [&] {
    if (uses.empty()) return;
    for (size_t e = 0; e < tmpl.triples.size(); ++e)
      ++uses[e * n_preds + matcher.pred_index_assign()[e]];
  };

  // skip_known: a match whose query is already known does not end the
  // search. The constants of the query are decided before the search, so
  // that a match fixes the query. The search also goes on after a new query
  // until the start node has given `quota` of them.
  Bindings bindings;
  size_t known_hits = 0, taken = 0, quota = 1;
  TermId current_start = kInvalidTerm;
  auto quota_in_pass = [&](size_t pass) -> size_t {
    return opt.per_seed ? opt.per_seed : size_t{1} << std::min<size_t>(pass, 6);
  };
  std::unique_ptr<Buckets> buckets;
  if (opt.card.buckets) buckets = std::make_unique<Buckets>(opt, store, rep);
  const size_t max_dropped = opt.card.max_dropped_in_a_row
                                 ? opt.card.max_dropped_in_a_row
                                 : std::max<size_t>(5000, 20 * opt.num);
  size_t dropped_in_a_row = 0;
  unsigned target = 0; // the range aimed at
  // With cardinality ranges the search on a start node ends at the first
  // query that is known or dropped, unless max_known says otherwise: the
  // other queries near it mostly fall into the same range.
  const size_t max_known = opt.max_known ? opt.max_known : buckets ? 1 : SIZE_MAX;
  if (opt.skip_known) {
    quota = quota_in_pass(0);
    matcher.set_on_match([&] {
      Query q = build_query(tmpl, matcher, bindings);
      if (!dedup.insert(q, terms)) {
        ++rep.duplicates_skipped;
        return ++known_hits >= max_known; // true ends the search on this start node
      }
      if (buckets) {
        auto t0 = std::chrono::steady_clock::now();
        q.y = count_query(store, fwd, rev, q, opt.card.count_budget);
        rep.count_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++rep.counted;
        const unsigned range = buckets->range_of(q.y);
        buckets->feedback(target, range);
        const bool outside = range == buckets->size();
        if (outside || !buckets->open(range)) {
          ++(outside ? rep.dropped_outside : rep.dropped_full);
          ++dropped_in_a_row;
          return ++known_hits >= max_known;
        }
        buckets->add(range);
        dropped_in_a_row = 0;
      }
      note_uses();
      rep.roots.push_back(current_start);
      out.push_back(std::move(q));
      if (++taken >= quota || out.size() >= opt.num) return true;
      matcher.reset_budget(); // every new query gets the full budget
      return false;
    });
  }
  // Several threads: see Worker. Thread t always gets the same share of a
  // round and has its own random numbers, so a run is repeatable for a
  // given seed and number of threads.
  std::vector<std::unique_ptr<Worker>> workers;
  if (opt.threads > 1 && !weighted)
    for (unsigned t = 0; t < opt.threads; ++t) {
      workers.push_back(
          std::make_unique<Worker>(store, fwd, rev, tmpl, bnode_base, dedup.method()));
      Worker* w = workers.back().get();
      w->rng.seed(rng());
      w->matcher.set_attempt_budget(opt.attempt_budget);
      w->matcher.set_degree_check(opt.degree_check);
      if (opt.prune) w->matcher.set_weights(&weights);
      if (opt.pred_first)
        w->matcher.set_predicate_first(true, opt.novelty, uses.empty() ? nullptr : uses.data(),
                                       n_preds);
      if (buckets) w->fill.assign(buckets->size(), 0);
      w->matcher.set_on_match([&, w] {
        Query q = build_query(tmpl, w->matcher, w->bindings);
        w->keymaker.key(q, terms, w->key);
        Worker::Tried& t = w->tried.back();
        if (dedup.contains(w->key) || !w->keys.insert(w->key).second)
          return ++t.known + t.dropped >= max_known;
        unsigned range = 0;
        if (buckets) {
          auto t0 = std::chrono::steady_clock::now();
          q.y = count_query(store, fwd, rev, q, opt.card.count_budget);
          w->count_seconds +=
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          ++w->counted;
          range = buckets->range_of(q.y);
          w->counts.push_back({w->target, range});
          const bool outside = range == buckets->size();
          if (outside || !buckets->open(range, &w->fill)) {
            ++(outside ? w->dropped_outside : w->dropped_full);
            w->dropped_keys.push_back(w->key);
            return ++t.dropped + t.known >= max_known;
          }
          ++w->fill[range];
        }
        w->found.push_back({std::move(q), w->key, w->matcher.pred_index_assign(), range});
        if (++t.found >= quota) return true;
        w->matcher.reset_budget();
        return false;
      });
    }

  auto order_starts = [&] {
    if (opt.seed_groups)
      order_by_group(subjects, fwd, rng);
    else
      std::shuffle(subjects.begin(), subjects.end(), rng);
  };
  // retire_seeds: the start nodes of the next pass.
  std::vector<TermId> productive;

  order_starts();
  size_t idx = 0;
  size_t pass_start_count = 0;
  size_t consecutive_dups = 0;

  // auto_prune. The weight pass reads every adjacency entry once per
  // spanning-tree step. The search overhead is every candidate looked at
  // beyond the one per step that a found match needs. One DFS candidate costs
  // at least kCandidateCost sequential entry reads (measured: >= 19 ns against
  // 1-2 ns), so the pass is run once overhead * kCandidateCost >= pass cost.
  constexpr uint64_t kCandidateCost = 8;
  bool auto_armed = opt.auto_prune && !opt.prune;
  uint64_t tree_steps = 0, pass_cost = 0, successes = 0;
  for (const Step& st : tmpl.steps) tree_steps += st.dir != Step::Dir::Check;
  pass_cost = tree_steps * static_cast<uint64_t>(store.triple_count());
  std::vector<TermId> next_subjects; // start set from the next pass on

  // Degree check on start nodes: done during the first pass only, on the
  // nodes it reaches. The ones that pass are the start set of later passes.
  // (With pruning the check is already in the weights.)
  bool checking_roots = opt.degree_check && !opt.prune;
  std::vector<TermId> rootable;

  auto expansions = [&] {
    uint64_t n = matcher.expansions();
    for (const auto& w : workers) n += w->matcher.expansions();
    return n;
  };

  // One round with several threads: the next start nodes are split evenly,
  // each thread searches its share, then the results are taken over in the
  // order of the start nodes. Returns false if the run has to stop.
  auto run_round = [&]() -> bool {
    const size_t n_threads = workers.size();
    const size_t round = std::min({subjects.size() - idx, 8192 * n_threads,
                                   std::max(64 * n_threads, 2 * (opt.num - out.size()))});
    {
      std::vector<std::jthread> pool;
      for (size_t t = 0; t < n_threads; ++t)
        pool.emplace_back([&, t] {
          Worker& w = *workers[t];
          w.tried.clear();
          w.found.clear();
          w.keys.clear();
          w.dropped_keys.clear();
          w.counts.clear();
          std::fill(w.fill.begin(), w.fill.end(), 0);
          for (size_t i = idx + t * round / n_threads; i < idx + (t + 1) * round / n_threads;
               ++i) {
            const TermId start = subjects[i];
            w.tried.push_back({start});
            if (checking_roots && !w.matcher.can_root(start)) {
              w.tried.back().can_root = false;
              continue;
            }
            draw_bindings(tmpl, opt.p_edge, opt.p_node, w.rng, w.bindings);
            if (buckets && buckets->steer()) {
              Buckets::Aim aim = buckets->aim(w.rng);
              w.target = aim.target;
              w.matcher.set_pred_bias(aim.bias);
            }
            w.matcher.match(start, w.rng);
            w.tried.back().gave_up = w.matcher.gave_up();
          }
        });
    }
    idx += round;

    const size_t before = out.size(), counted_before = rep.counted;
    for (auto& wp : workers) {
      Worker& w = *wp;
      size_t f = 0;
      for (const Worker::Tried& t : w.tried) {
        if (!t.can_root) {
          ++rep.skipped_starts;
          continue;
        }
        if (checking_roots) rootable.push_back(t.node);
        ++rep.attempts;
        rep.gave_up += t.gave_up;
        rep.duplicates_skipped += t.known;
        size_t emitted = 0;
        for (uint32_t k = 0; k < t.found; ++k, ++f) {
          Worker::Found& x = w.found[f];
          if (out.size() >= opt.num) continue;
          if (!dedup.insert_key(std::move(x.key))) { // another thread found it in this round
            ++rep.duplicates_skipped;
            continue;
          }
          if (buckets) {
            if (!buckets->open(x.range)) {
              ++rep.dropped_full;
              continue;
            }
            buckets->add(x.range);
          }
          if (!uses.empty())
            for (size_t e = 0; e < tmpl.triples.size(); ++e) ++uses[e * n_preds + x.preds[e]];
          rep.roots.push_back(t.node);
          out.push_back(std::move(x.q));
          ++emitted;
        }
        if (emitted) {
          ++successes;
          consecutive_dups = 0;
          if (opt.retire_seeds) productive.push_back(t.node);
        } else if (t.found + t.known + t.dropped == 0) {
          ++rep.failed_matches;
        } else {
          ++successes;
          ++rep.known_only;
          ++consecutive_dups;
        }
      }
      for (std::string& k : w.dropped_keys) dedup.insert_key(std::move(k));
      if (buckets)
        for (auto [aimed, range] : w.counts) buckets->feedback(aimed, range);
      rep.counted += w.counted;
      rep.count_seconds += w.count_seconds;
      rep.dropped_outside += w.dropped_outside;
      rep.dropped_full += w.dropped_full;
      w.counted = w.dropped_outside = w.dropped_full = 0;
      w.count_seconds = 0.0;
    }
    if (consecutive_dups >= max_consecutive) {
      rep.stop_reason = "too many start nodes in a row with known queries only (" +
                        std::to_string(consecutive_dups) + ")";
      return false;
    }
    dropped_in_a_row = out.size() > before ? 0 : dropped_in_a_row + rep.counted - counted_before;
    if (buckets && dropped_in_a_row >= max_dropped) {
      rep.stop_reason = std::to_string(dropped_in_a_row) +
                        " counted queries in a row fell outside the ranges that still lack queries";
      return false;
    }
    return true;
  };

  while (out.size() < opt.num) {
    if (auto_armed &&
        (expansions() - successes * tree_steps) * kCandidateCost >= pass_cost) {
      auto_armed = false;
      rep.auto_pruned = true;
      auto t0 = std::chrono::steady_clock::now();
      weights.compute(fwd, rev, tmpl, bnode_base, opt.degree_check);
      rep.weights_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      rep.tree_matches = weights.total();
      matcher.set_weights(&weights);
      for (auto& w : workers) w->matcher.set_weights(&weights);
      for (TermId v = 0; v < bnode_base; ++v)
        if (weights.weight(tmpl.root_slot, v) > 0.0) next_subjects.push_back(v);
      rep.start_candidates = next_subjects.size();
      if (next_subjects.empty()) {
        rep.stop_reason = "no node can root the template";
        break;
      }
      // Finish the current pass on its untried nodes that can still root a
      // match; nothing already tried is tried again.
      size_t kept = 0;
      for (size_t i = idx; i < subjects.size(); ++i)
        if (weights.weight(tmpl.root_slot, subjects[i]) > 0.0) subjects[kept++] = subjects[i];
      subjects.resize(kept);
      idx = 0;
    }

    if (idx >= subjects.size()) {
      size_t new_in_pass = out.size() - pass_start_count;
      if (new_in_pass == 0) {
        rep.stop_reason = "dataset exhausted (no new queries in a full pass)";
        break;
      }
      if (opt.min_new_per_pass && new_in_pass < opt.min_new_per_pass) {
        rep.stop_reason = "diminishing returns (<" + std::to_string(opt.min_new_per_pass) +
                          " new queries in a pass)";
        break;
      }
      if (rep.reshuffles >= opt.max_reshuffles) {
        rep.stop_reason = "max reshuffles reached";
        break;
      }
      if (opt.skip_known && opt.retire_seeds) {
        // The known queries only grow: a start node without a new query in
        // this pass is unlikely to give one later.
        subjects.swap(productive);
        productive.clear();
        next_subjects.clear();
      } else if (!next_subjects.empty()) {
        subjects.swap(next_subjects);
        next_subjects.clear();
      } else if (checking_roots) {
        subjects.swap(rootable);
      }
      checking_roots = false;
      quota = quota_in_pass(rep.reshuffles + 1);
      order_starts();
      idx = 0;
      pass_start_count = out.size();
      ++rep.reshuffles;
      consecutive_dups = 0;
    }

    if (!workers.empty()) {
      if (!run_round()) break;
      continue;
    }

    TermId start = subjects[idx++];
    if (checking_roots) {
      if (!matcher.can_root(start)) {
        ++rep.skipped_starts;
        continue;
      }
      rootable.push_back(start);
    }
    ++rep.attempts;
    if (opt.skip_known) {
      draw_bindings(tmpl, opt.p_edge, opt.p_node, rng, bindings);
      known_hits = taken = 0;
      current_start = start;
      if (buckets) {
        if (dropped_in_a_row >= max_dropped) {
          rep.stop_reason = std::to_string(dropped_in_a_row) +
                            " counted queries in a row fell outside the ranges that still lack queries";
          break;
        }
        if (buckets->steer()) {
          Buckets::Aim aim = buckets->aim(rng);
          target = aim.target;
          matcher.set_pred_bias(aim.bias);
        }
      }
      matcher.match(start, rng);
      rep.gave_up += matcher.gave_up();
      if (taken > 0) {
        ++successes;
        consecutive_dups = 0;
        if (opt.retire_seeds) productive.push_back(start);
        continue;
      }
      if (known_hits == 0) {
        ++rep.failed_matches;
        continue;
      }
      ++successes;
      ++rep.known_only;
      if (++consecutive_dups >= max_consecutive) {
        rep.stop_reason = "too many start nodes in a row with known queries only (" +
                          std::to_string(consecutive_dups) + ")";
        break;
      }
      continue;
    }
    if (!matcher.match(start, rng)) {
      ++rep.failed_matches;
      rep.gave_up += matcher.gave_up();
      continue;
    }
    ++successes;

    Query q = build_query(tmpl, matcher, opt.p_edge, opt.p_node, rng);
    if (!dedup.insert(q, terms)) {
      ++rep.duplicates_skipped;
      if (++consecutive_dups >= max_consecutive) {
        rep.stop_reason =
            "too many consecutive duplicates (" + std::to_string(consecutive_dups) + ")";
        break;
      }
      continue;
    }
    consecutive_dups = 0;
    note_uses();
    rep.roots.push_back(start);
    out.push_back(std::move(q));
  }
  rep.expansions = expansions();
  return out;
}

} // namespace bgpgen
