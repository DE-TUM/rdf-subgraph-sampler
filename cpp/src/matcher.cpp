#include "bgpgen/matcher.hpp"

namespace bgpgen {

Matcher::Matcher(const TripleStore& store, const SubjectIndex& fwd, const SubjectIndex& rev,
                 const CompiledTemplate& tmpl, TermId bnode_base)
    : store_(store), fwd_(fwd), rev_(rev), tmpl_(tmpl), bnode_base_(bnode_base),
      deg_(slot_degrees(tmpl)), node_assign_(tmpl.n_slots, kInvalidTerm), pred_assign_(tmpl.triples.size(), kInvalidTerm),
      idx_buf_(tmpl.steps.size()), touched_buf_(tmpl.steps.size()) {}

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
  if (step_idx == tmpl_.steps.size()) return true;
  const Step& st = tmpl_.steps[step_idx];
  const TemplateTriple& tt = tmpl_.triples[st.edge];

  if (st.dir == Step::Dir::Check) {
    // Both endpoints assigned: collect the predicates connecting s -> o and
    // pick one at random (Python: single continuation, no backtracking over
    // the predicate choice).
    TermId s = node_assign_[tt.s_slot];
    TermId o = node_assign_[tt.o_slot];
    check_preds_.clear();
    for (const SubjectIndex::Entry& e : fwd_.row_of(s))
      if (e.obj == o) check_preds_.push_back(store_.pred_term(e.pred));
    if (check_preds_.empty()) return false;
    pred_assign_[st.edge] = check_preds_[rng() % check_preds_.size()];
    if (dfs(step_idx + 1, rng)) return true;
    pred_assign_[st.edge] = kInvalidTerm;
    return false;
  }

  bool fwd = st.dir == Step::Dir::Fwd;
  TermId known = node_assign_[fwd ? tt.s_slot : tt.o_slot];
  std::span<const SubjectIndex::Entry> row = (fwd ? fwd_ : rev_).row_of(known);
  if (row.empty()) return false;

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
