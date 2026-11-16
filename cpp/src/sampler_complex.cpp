#include "bgpgen/sampler_complex.hpp"

#include "bgpgen/matcher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace bgpgen {

namespace {

/// Builds the output query from a complete match, drawing per-occurrence
/// binding decisions for Prob positions exactly like the Python generators
/// (note: strict '<', unlike the star sampler's '<=').
Query build_query(const CompiledTemplate& tmpl, const Matcher& m, double p_edge, double p_node,
                  std::mt19937_64& rng) {
  Query q;
  std::uniform_real_distribution<double> unit(0.0, 1.0);
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

    bool s_bound = tt.s_bind == Bind::Bound || (tt.s_bind == Bind::Prob && unit(rng) < p_node);
    bool p_bound = tt.p_bind == Bind::Bound || (tt.p_bind == Bind::Prob && unit(rng) < p_edge);
    bool o_bound = tt.o_bind == Bind::Bound || (tt.o_bind == Bind::Prob && unit(rng) < p_node);

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

} // namespace

std::vector<Query> sample_complex(const TripleStore& store, const SubjectIndex& fwd,
                                  const SubjectIndex& rev, const TermTable& terms,
                                  const CompiledTemplate& tmpl, const ComplexOptions& opt,
                                  Dedup& dedup, std::mt19937_64& rng, SampleReport* report) {
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

  std::shuffle(subjects.begin(), subjects.end(), rng);
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

  while (out.size() < opt.num) {
    if (auto_armed &&
        (matcher.expansions() - successes * tree_steps) * kCandidateCost >= pass_cost) {
      auto_armed = false;
      rep.auto_pruned = true;
      auto t0 = std::chrono::steady_clock::now();
      weights.compute(fwd, rev, tmpl, bnode_base, opt.degree_check);
      rep.weights_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      rep.tree_matches = weights.total();
      matcher.set_weights(&weights);
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
      if (!next_subjects.empty()) {
        subjects.swap(next_subjects);
        next_subjects.clear();
      } else if (checking_roots) {
        subjects.swap(rootable);
      }
      checking_roots = false;
      std::shuffle(subjects.begin(), subjects.end(), rng);
      idx = 0;
      pass_start_count = out.size();
      ++rep.reshuffles;
      consecutive_dups = 0;
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
    rep.roots.push_back(start);
    out.push_back(std::move(q));
  }
  rep.expansions = matcher.expansions();
  return out;
}

} // namespace bgpgen
