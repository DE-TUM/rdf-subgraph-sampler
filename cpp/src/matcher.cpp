#include "bgpgen/matcher.hpp"

#include <algorithm>
#include <cmath>

namespace bgpgen {

Matcher::Matcher(const TripleStore& store, const SubjectIndex& fwd, const SubjectIndex& rev,
                 const CompiledTemplate& tmpl, TermId bnode_base)
    : store_(store), fwd_(fwd), rev_(rev), tmpl_(tmpl), bnode_base_(bnode_base),
      deg_(slot_degrees(tmpl)), node_assign_(tmpl.n_slots, kInvalidTerm), pred_assign_(tmpl.triples.size(), kInvalidTerm),
      pred_index_assign_(tmpl.triples.size(), kNoPred),
      idx_buf_(tmpl.steps.size()), touched_buf_(tmpl.steps.size()), group_buf_(tmpl.steps.size()),
      leaf_var_(tmpl.n_slots, 0) {
  std::vector<int> edges(tmpl.n_slots, 0);
  std::vector<bool> var(tmpl.n_slots, true);
  for (const TemplateTriple& tt : tmpl.triples) {
    ++edges[tt.s_slot];
    ++edges[tt.o_slot];
    if (tt.s_bind != Bind::Var) var[tt.s_slot] = false;
    if (tt.o_bind != Bind::Var) var[tt.o_slot] = false;
  }
  for (size_t t = 0; t < tmpl.n_slots; ++t) leaf_var_[t] = edges[t] == 1 && var[t];
}

bool Matcher::used(TermId node) const {
  for (TermId t : node_assign_)
    if (t == node) return true;
  return false;
}

bool Matcher::match(TermId start, std::mt19937_64& rng) {
  std::fill(node_assign_.begin(), node_assign_.end(), kInvalidTerm);
  std::fill(pred_assign_.begin(), pred_assign_.end(), kInvalidTerm);
  node_assign_[tmpl_.root_slot] = start;
  attempt_start_ = expansions_;
  gave_up_ = false;
  return dfs(0, rng);
}

bool Matcher::dfs(size_t step_idx, std::mt19937_64& rng) {
  if (step_idx == tmpl_.steps.size()) return on_match_ ? on_match_() : true;
  const Step& st = tmpl_.steps[step_idx];
  const TemplateTriple& tt = tmpl_.triples[st.edge];

  if (st.dir == Step::Dir::Check) {
    // Both endpoints assigned: collect the predicates connecting s -> o and
    // pick one at random (Python: single continuation, no backtracking over
    // the predicate choice).
    TermId s = node_assign_[tt.s_slot];
    TermId o = node_assign_[tt.o_slot];
    check_preds_.clear();
    check_pidx_.clear();
    for (const SubjectIndex::Entry& e : fwd_.row_of(s))
      if (e.obj == o) {
        check_preds_.push_back(store_.pred_term(e.pred));
        check_pidx_.push_back(e.pred);
      }
    if (check_preds_.empty()) return false;
    size_t pick = rng() % check_preds_.size();
    pred_assign_[st.edge] = check_preds_[pick];
    pred_index_assign_[st.edge] = check_pidx_[pick];
    if (dfs(step_idx + 1, rng)) return true;
    pred_assign_[st.edge] = kInvalidTerm;
    return false;
  }

  bool fwd = st.dir == Step::Dir::Fwd;
  TermId known = node_assign_[fwd ? tt.s_slot : tt.o_slot];
  std::span<const SubjectIndex::Entry> row = (fwd ? fwd_ : rev_).row_of(known);
  if (row.empty()) return false;
  if (pred_first_) return extend_pred_first(step_idx, row, rng);

  // Up to max_candidates random distinct row offsets, via incremental
  // Fisher-Yates over a per-depth index buffer. The buffer holds the identity
  // between calls and only the swapped positions are reset afterwards, so a
  // step costs O(candidates looked at), not O(row size).
  std::vector<uint32_t>& idx = idx_buf_[step_idx];
  std::vector<uint32_t>& touched = touched_buf_[step_idx];
  uint32_t n = static_cast<uint32_t>(row.size());
  for (uint32_t k = static_cast<uint32_t>(idx.size()); k < n; ++k) idx.push_back(k);
  touched.clear();
  uint32_t limit = std::min<uint32_t>(n, tmpl_.max_candidates);

  bool found = false;
  for (uint32_t i = 0; i < limit; ++i) {
    if (budget_ && expansions_ - attempt_start_ >= budget_) {
      gave_up_ = true;
      break;
    }
    uint32_t j = i + static_cast<uint32_t>(rng() % (n - i));
    std::swap(idx[i], idx[j]);
    touched.push_back(j);
    const SubjectIndex::Entry& e = row[idx[i]];
    TermId cand = e.obj;
    ++expansions_;
    if (cand >= bnode_base_) continue; // IRI->IRI subgraph only
    if (weights_ && !(weights_->weight(st.new_slot, cand) > 0.0)) continue; // dead end
    if (used(cand)) continue;          // injective match (simple paths)
    // A slot with one template neighbour is reached through it; nothing to check.
    if (degree_check_ && deg_.all[st.new_slot] > 1 &&
        !degree_ok(st.new_slot, cand))
      continue;

    node_assign_[st.new_slot] = cand;
    pred_assign_[st.edge] = store_.pred_term(e.pred);
    pred_index_assign_[st.edge] = e.pred;
    if (dfs(step_idx + 1, rng)) {
      found = true;
      break;
    }
    node_assign_[st.new_slot] = kInvalidTerm;
    pred_assign_[st.edge] = kInvalidTerm;
    if (gave_up_) break;
  }
  for (uint32_t t = 0; t < touched.size(); ++t) {
    idx[t] = t;
    idx[touched[t]] = touched[t];
  }
  return found;
}

bool Matcher::extend_pred_first(size_t step_idx, std::span<const SubjectIndex::Entry> row,
                                std::mt19937_64& rng) {
  const Step& st = tmpl_.steps[step_idx];
  const uint32_t n = static_cast<uint32_t>(row.size());

  // The row is sorted by predicate: one group of entries per predicate. The
  // group boundaries of the last row are kept, because the steps that leave
  // the same template node (all of a star's) come one after the other.
  if (row.data() != bounds_row_) {
    bounds_row_ = row.data();
    bounds_.clear();
    for (uint32_t a = 0; a < n;) {
      const PredIndex p = row[a].pred;
      uint32_t b = a + 1;
      if (row[n - 1].pred == p) {
        b = n;
      } else {
        uint32_t step = 1; // gallop, then binary search
        while (a + step < n && row[a + step].pred == p) step *= 2;
        b = static_cast<uint32_t>(
            std::upper_bound(row.begin() + a + step / 2, row.begin() + std::min(n, a + step), p,
                             [](PredIndex v, const SubjectIndex::Entry& e) { return v < e.pred; }) -
            row.begin());
      }
      bounds_.push_back(a);
      a = b;
    }
    bounds_.push_back(n);
  }
  std::vector<Group>& groups = group_buf_[step_idx];
  groups.clear();
  const uint32_t* uses = uses_ ? uses_ + st.edge * n_preds_ : nullptr;
  double total = 0.0;
  for (size_t k = 0; k + 1 < bounds_.size(); ++k) {
    const PredIndex p = row[bounds_[k]].pred;
    double w = 1.0;
    if (uses && novelty_ != 0.0)
      w = novelty_ == 1.0 ? 1.0 / (1.0 + uses[p]) : std::pow(1.0 + uses[p], -novelty_);
    if (bias_) w *= bias_[p];
    groups.push_back({bounds_[k], bounds_[k + 1], 0, w});
    total += w;
  }
  size_t live = groups.size();

  // As in dfs(): random distinct entries through a partial Fisher-Yates
  // shuffle of the index buffer, here inside the chosen predicate's group.
  std::vector<uint32_t>& idx = idx_buf_[step_idx];
  std::vector<uint32_t>& touched = touched_buf_[step_idx];
  for (uint32_t k = static_cast<uint32_t>(idx.size()); k < n; ++k) idx.push_back(k);
  touched.clear();
  const uint32_t limit = std::min<uint32_t>(n, tmpl_.max_candidates);
  const bool one_per_pred = leaf_var_[st.new_slot];
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  bool found = false;
  for (uint32_t i = 0; i < limit && live > 0; ++i) {
    if (budget_ && expansions_ - attempt_start_ >= budget_) {
      gave_up_ = true;
      break;
    }
    Group* g = nullptr;
    if (live == 1 || groups.size() == 1) {
      for (Group& c : groups)
        if (c.weight > 0.0) g = &c;
    } else {
      const double r = unit(rng) * total;
      double acc = 0.0;
      for (Group& c : groups) {
        if (!(c.weight > 0.0)) continue;
        g = &c; // the last live group also absorbs rounding
        acc += c.weight;
        if (acc > r) break;
      }
    }
    auto retire = [&](Group& c) {
      total -= c.weight;
      c.weight = 0.0;
      --live;
    };
    const uint32_t pos = g->begin + g->taken;
    const uint32_t j = pos + static_cast<uint32_t>(rng() % (g->end - pos));
    std::swap(idx[pos], idx[j]);
    touched.push_back(pos);
    touched.push_back(j);
    if (++g->taken == g->end - g->begin) retire(*g);
    const SubjectIndex::Entry& e = row[idx[pos]];
    TermId cand = e.obj;
    ++expansions_;
    if (cand >= bnode_base_) continue;
    if (weights_ && !(weights_->weight(st.new_slot, cand) > 0.0)) continue;
    if (used(cand)) continue;
    if (degree_check_ && deg_.all[st.new_slot] > 1 && !degree_ok(st.new_slot, cand)) continue;

    node_assign_[st.new_slot] = cand;
    pred_assign_[st.edge] = store_.pred_term(e.pred);
    pred_index_assign_[st.edge] = e.pred;
    if (dfs(step_idx + 1, rng)) {
      found = true;
      break;
    }
    node_assign_[st.new_slot] = kInvalidTerm;
    pred_assign_[st.edge] = kInvalidTerm;
    if (gave_up_) break;
    if (one_per_pred && g->weight > 0.0) retire(*g);
  }
  for (uint32_t t : touched) idx[t] = t;
  return found;
}

bool Matcher::sample_weighted(TermId start, std::mt19937_64& rng) {
  std::fill(node_assign_.begin(), node_assign_.end(), kInvalidTerm);
  std::fill(pred_assign_.begin(), pred_assign_.end(), kInvalidTerm);
  node_assign_[tmpl_.root_slot] = start;
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  for (size_t i = 0; i < tmpl_.steps.size(); ++i) {
    const Step& st = tmpl_.steps[i];
    const TemplateTriple& tt = tmpl_.triples[st.edge];

    if (st.dir == Step::Dir::Check) {
      TermId s = node_assign_[tt.s_slot];
      TermId o = node_assign_[tt.o_slot];
      check_preds_.clear();
      for (const SubjectIndex::Entry& e : fwd_.row_of(s))
        if (e.obj == o) check_preds_.push_back(store_.pred_term(e.pred));
      if (check_preds_.empty()) return false;
      pred_assign_[st.edge] = check_preds_[rng() % check_preds_.size()];
      continue;
    }

    const bool fwd = st.dir == Step::Dir::Fwd;
    const TermId known = node_assign_[fwd ? tt.s_slot : tt.o_slot];
    std::span<const SubjectIndex::Entry> row = (fwd ? fwd_ : rev_).row_of(known);
    if (row.empty()) return false;
    const SubjectIndex::Entry* pick = nullptr;

    if (weights_->leaf(st.new_slot)) {
      // All IRI neighbours weigh 1: uniform draw. Try a few random offsets,
      // then fall back to an exact scan (rows dominated by literals).
      for (int tries = 0; tries < 16 && !pick; ++tries) {
        const SubjectIndex::Entry& e = row[rng() % row.size()];
        ++expansions_;
        if (e.obj < bnode_base_) pick = &e;
      }
      if (!pick) {
        size_t n_iri = 0;
        for (const SubjectIndex::Entry& e : row) n_iri += e.obj < bnode_base_;
        expansions_ += row.size();
        if (n_iri == 0) return false;
        size_t k = rng() % n_iri;
        for (const SubjectIndex::Entry& e : row) {
          if (e.obj >= bnode_base_) continue;
          if (k-- == 0) {
            pick = &e;
            break;
          }
        }
      }
    } else {
      const double total = weights_->step_sum(i, known);
      if (!(total > 0.0)) return false;
      const double r = unit(rng) * total;
      double acc = 0.0;
      for (const SubjectIndex::Entry& e : row) {
        if (e.obj >= bnode_base_) continue;
        const double w = weights_->weight(st.new_slot, e.obj);
        if (!(w > 0.0)) continue;
        ++expansions_;
        pick = &e; // the last positive candidate also absorbs rounding
        acc += w;
        if (acc > r) break;
      }
      if (!pick) return false;
    }

    if (used(pick->obj)) return false; // reject: keeps accepted matches equally likely
    node_assign_[st.new_slot] = pick->obj;
    pred_assign_[st.edge] = store_.pred_term(pick->pred);
  }
  return true;
}

} // namespace bgpgen
