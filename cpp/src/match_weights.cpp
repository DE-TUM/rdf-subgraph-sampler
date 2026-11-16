#include "bgpgen/match_weights.hpp"

namespace bgpgen {

void MatchWeights::compute(const SubjectIndex& fwd, const SubjectIndex& rev,
                           const CompiledTemplate& tmpl, TermId bnode_base, bool degree_check) {
  const size_t n = bnode_base; // IRIs are the id prefix [0, bnode_base)
  SlotDegrees deg;
  if (degree_check) deg = slot_degrees(tmpl);
  w_.assign(tmpl.n_slots, {});
  step_sum_.assign(tmpl.steps.size(), {});

  // Steps are in traversal order, so a slot's own child steps all come after
  // the step that assigns it: walking backwards sees children first.
  for (size_t i = tmpl.steps.size(); i-- > 0;) {
    const Step& st = tmpl.steps[i];
    if (st.dir == Step::Dir::Check) continue;
    const TemplateTriple& tt = tmpl.triples[st.edge];
    const bool is_fwd = st.dir == Step::Dir::Fwd;
    const uint8_t known = is_fwd ? tt.s_slot : tt.o_slot;
    const SubjectIndex& idx = is_fwd ? fwd : rev;
    const std::vector<double>& wc = w_[st.new_slot];

    std::vector<double> sum(n, 0.0);
    for (size_t pos = 0; pos < idx.iri_subject_count(); ++pos) {
      double s = 0.0;
      for (const SubjectIndex::Entry& e : idx.row(pos)) {
        if (e.obj >= bnode_base) continue;
        s += wc.empty() ? 1.0 : wc[e.obj];
      }
      sum[idx.subject_at(pos)] = s;
    }

    std::vector<double>& wk = w_[known];
    if (wk.empty()) {
      wk = sum;
      if (degree_check)
        for (size_t v = 0; v < n; ++v)
          if (wk[v] > 0.0 && !enough_neighbours(fwd, rev, static_cast<TermId>(v), deg.out[known],
                                                deg.in[known], deg.all[known], bnode_base))
            wk[v] = 0.0;
    } else {
      for (size_t v = 0; v < n; ++v) wk[v] *= sum[v];
    }
    if (!wc.empty()) step_sum_[i] = std::move(sum);
  }

  total_ = 0.0;
  for (double w : w_[tmpl.root_slot]) total_ += w;
}

} // namespace bgpgen
