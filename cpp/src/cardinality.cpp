#include "bgpgen/cardinality.hpp"

#include "bgpgen/thread_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace bgpgen {

namespace {

constexpr uint64_t sat_add(uint64_t a, uint64_t b) {
  return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}
constexpr uint64_t sat_mul(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) return 0;
  return a > UINT64_MAX / b ? UINT64_MAX : a * b;
}

// --- Counting plan -------------------------------------------------------
//
// Variables occurring once ("Free") are never enumerated — the number of
// extensions they admit is a span length, multiplied in as a factor. Only
// multi-occurrence (join) variables get binding slots. Per pattern and
// position the runtime state is static, so it is resolved at compile time:
//   Const  — constant term (predicates pre-resolved to PredIndex)
//   Bound  — join slot bound by an earlier pattern
//   New    — first occurrence of a join slot; binds during enumeration
//   Check  — slot bound earlier within this same pattern (self-join);
//            candidates are compared, not bound
//   Free   — single-occurrence variable; contributes a factor

enum class PS : uint8_t { Const, Bound, New, Check, Free };

struct Pos {
  PS st;
  uint32_t v; // Const: TermId (s/o) or PredIndex (p); slots otherwise
};

struct PlanPat {
  Pos s, p, o;
};

struct Plan {
  std::vector<PlanPat> pats;
  uint32_t n_slots = 0;
  bool never_matches = false; // a constant predicate absent from the data
};

Plan compile(const Query& q, const TripleStore& store) {
  uint32_t n_vars = 0;
  for (const TriplePattern& tp : q.patterns)
    for (QTerm t : {tp.s, tp.p, tp.o})
      if (q_is_var(t)) n_vars = std::max(n_vars, q_var_id(t) + 1);

  std::vector<uint32_t> occ(n_vars, 0);
  for (const TriplePattern& tp : q.patterns)
    for (QTerm t : {tp.s, tp.p, tp.o})
      if (q_is_var(t)) ++occ[q_var_id(t)];

  Plan plan;
  std::vector<int32_t> slot_of(n_vars, -1);
  for (uint32_t v = 0; v < n_vars; ++v)
    if (occ[v] >= 2) slot_of[v] = static_cast<int32_t>(plan.n_slots++);

  // Greedy execution order: most bound positions first (nodes weigh double),
  // counting join slots bound by already-placed patterns. Connected patterns
  // score through their shared slots, so the order stays connected whenever
  // the query is.
  size_t n = q.patterns.size();
  std::vector<bool> bound(plan.n_slots, false);
  std::vector<bool> used(n, false);
  std::vector<size_t> order;
  auto pos_score = [&](QTerm t, int w) {
    if (!q_is_var(t)) return w;
    int32_t sl = slot_of[q_var_id(t)];
    return sl >= 0 && bound[static_cast<size_t>(sl)] ? w : 0;
  };
  for (size_t k = 0; k < n; ++k) {
    size_t best = 0;
    int best_score = -1;
    for (size_t j = 0; j < n; ++j) {
      if (used[j]) continue;
      const TriplePattern& tp = q.patterns[j];
      int sc = pos_score(tp.s, 2) + pos_score(tp.p, 1) + pos_score(tp.o, 2);
      if (sc > best_score) {
        best_score = sc;
        best = j;
      }
    }
    used[best] = true;
    order.push_back(best);
    for (QTerm t : {q.patterns[best].s, q.patterns[best].p, q.patterns[best].o})
      if (q_is_var(t) && slot_of[q_var_id(t)] >= 0)
        bound[static_cast<size_t>(slot_of[q_var_id(t)])] = true;
  }

  // Classify positions in execution binding order (p, then s, then o) so
  // within-pattern repeats become Check on the later position.
  std::fill(bound.begin(), bound.end(), false);
  for (size_t j : order) {
    const TriplePattern& tp = q.patterns[j];
    std::vector<bool> before = bound;
    auto classify = [&](QTerm t, bool is_pred) -> Pos {
      if (!q_is_var(t)) {
        if (is_pred) {
          PredIndex pi = store.pred_index(t);
          if (pi == kNoPred) plan.never_matches = true;
          return {PS::Const, pi};
        }
        return {PS::Const, t};
      }
      int32_t sl = slot_of[q_var_id(t)];
      if (sl < 0) return {PS::Free, 0};
      auto s = static_cast<size_t>(sl);
      if (before[s]) return {PS::Bound, static_cast<uint32_t>(s)};
      if (bound[s]) return {PS::Check, static_cast<uint32_t>(s)};
      bound[s] = true;
      return {PS::New, static_cast<uint32_t>(s)};
    };
    PlanPat pp;
    pp.p = classify(tp.p, true);
    pp.s = classify(tp.s, false);
    pp.o = classify(tp.o, false);
    plan.pats.push_back(pp);
  }
  return plan;
}

// --- Executor -------------------------------------------------------------

struct Ctx {
  const TripleStore& store;
  const SubjectIndex& fwd;
  const SubjectIndex& rev;
  const Plan& plan;
  std::vector<TermId> slots;
  uint64_t budget;
  bool exhausted = false;

  bool charge(uint64_t steps) {
    if (budget < steps) {
      budget = 0;
      exhausted = true;
      return false;
    }
    budget -= steps;
    return true;
  }

  bool fixed(const Pos& p, TermId& out) const {
    if (p.st == PS::Const) {
      out = p.v;
      return true;
    }
    if (p.st == PS::Bound) {
      out = slots[p.v];
      return true;
    }
    return false;
  }

  /// Accepts an enumerated candidate: New binds, Check compares, Free
  /// matches anything.
  bool place(const Pos& p, TermId val) {
    if (p.st == PS::New) {
      slots[p.v] = val;
      return true;
    }
    if (p.st == PS::Check) return slots[p.v] == val;
    return true; // Free
  }

  uint64_t eval(size_t i) {
    if (i == plan.pats.size()) return 1;
    if (!charge(1)) return 0;
    const Pos& p = plan.pats[i].p;
    switch (p.st) {
      case PS::Const:
        return eval_pred(i, p.v);
      case PS::Bound:
      case PS::Check: { // a slot holding a term that must be a predicate
        PredIndex pi = store.pred_index(slots[p.v]);
        return pi == kNoPred ? 0 : eval_pred(i, pi);
      }
      case PS::New: { // joining predicate var: try every predicate
        uint64_t sum = 0;
        for (size_t pi = 0; pi < store.predicate_count(); ++pi) {
          if (!charge(1)) return 0;
          slots[p.v] = store.pred_term(static_cast<PredIndex>(pi));
          sum = sat_add(sum, eval_pred(i, static_cast<PredIndex>(pi)));
          if (exhausted) return 0;
        }
        return sum;
      }
      case PS::Free:
        return eval_any(i);
    }
    return 0;
  }

  /// Pattern i with the predicate fixed to pi.
  uint64_t eval_pred(size_t i, PredIndex pi) {
    const PlanPat& pp = plan.pats[i];
    TermId sv, ov;
    bool sf = fixed(pp.s, sv), of = fixed(pp.o, ov);

    if (sf && of) return store.has(sv, pi, ov) ? eval(i + 1) : 0;

    if (sf) {
      IdSpan objs = store.objects_of(sv, pi);
      if (objs.empty()) return 0;
      if (pp.o.st == PS::Free) return sat_mul(objs.size(), eval(i + 1));
      if (!charge(objs.size())) return 0;
      uint64_t sum = 0;
      for (TermId o : objs) {
        if (!place(pp.o, o)) continue;
        sum = sat_add(sum, eval(i + 1));
        if (exhausted) return 0;
      }
      return sum;
    }

    if (of) {
      IdSpan subs = store.subjects_of(ov, pi);
      if (subs.empty()) return 0;
      if (pp.s.st == PS::Free) return sat_mul(subs.size(), eval(i + 1));
      if (!charge(subs.size())) return 0;
      uint64_t sum = 0;
      for (TermId s : subs) {
        if (!place(pp.s, s)) continue;
        sum = sat_add(sum, eval(i + 1));
        if (exhausted) return 0;
      }
      return sum;
    }

    bool s_free = pp.s.st == PS::Free, o_free = pp.o.st == PS::Free;
    if (s_free && o_free) return sat_mul(store.triples_with(pi), eval(i + 1));

    uint64_t sum = 0;
    if (o_free) { // s joins: one recursion per subject, span length as factor
      store.for_each_subject(pi, [&](TermId s, IdSpan objs) {
        if (exhausted || !charge(1)) return;
        if (!place(pp.s, s)) return;
        sum = sat_add(sum, sat_mul(objs.size(), eval(i + 1)));
      });
      return exhausted ? 0 : sum;
    }
    if (s_free) {
      store.for_each_object(pi, [&](TermId o, IdSpan subs) {
        if (exhausted || !charge(1)) return;
        if (!place(pp.o, o)) return;
        sum = sat_add(sum, sat_mul(subs.size(), eval(i + 1)));
      });
      return exhausted ? 0 : sum;
    }
    // Both nodes join: enumerate the predicate's triples.
    store.for_each_subject(pi, [&](TermId s, IdSpan objs) {
      if (exhausted) return;
      if (!place(pp.s, s)) return;
      if (!charge(objs.size())) return;
      for (TermId o : objs) {
        if (!place(pp.o, o)) continue;
        sum = sat_add(sum, eval(i + 1));
        if (exhausted) return;
      }
    });
    return exhausted ? 0 : sum;
  }

  /// Pattern i with a Free (non-joining) predicate variable: every
  /// subject-major row entry is one extension.
  uint64_t eval_any(size_t i) {
    const PlanPat& pp = plan.pats[i];
    TermId sv, ov;
    bool sf = fixed(pp.s, sv), of = fixed(pp.o, ov);

    if (sf && of) { // factor = number of predicates linking sv -> ov
      std::span<const SubjectIndex::Entry> row = fwd.row_of(sv);
      if (!charge(row.size())) return 0;
      uint64_t links = 0;
      for (const SubjectIndex::Entry& e : row) links += e.obj == ov;
      return links == 0 ? 0 : sat_mul(links, eval(i + 1));
    }

    if (sf) {
      std::span<const SubjectIndex::Entry> row = fwd.row_of(sv);
      if (row.empty()) return 0;
      if (pp.o.st == PS::Free) return sat_mul(row.size(), eval(i + 1));
      if (!charge(row.size())) return 0;
      uint64_t sum = 0;
      for (const SubjectIndex::Entry& e : row) {
        if (!place(pp.o, e.obj)) continue;
        sum = sat_add(sum, eval(i + 1));
        if (exhausted) return 0;
      }
      return sum;
    }

    if (of) {
      std::span<const SubjectIndex::Entry> row = rev.row_of(ov);
      if (row.empty()) return 0;
      if (pp.s.st == PS::Free) return sat_mul(row.size(), eval(i + 1));
      if (!charge(row.size())) return 0;
      uint64_t sum = 0;
      for (const SubjectIndex::Entry& e : row) { // e.obj = subject (reverse)
        if (!place(pp.s, e.obj)) continue;
        sum = sat_add(sum, eval(i + 1));
        if (exhausted) return 0;
      }
      return sum;
    }

    bool s_free = pp.s.st == PS::Free, o_free = pp.o.st == PS::Free;
    if (s_free && o_free) return sat_mul(store.triple_count(), eval(i + 1));

    uint64_t sum = 0;
    if (o_free) { // s joins: per subject, out-degree as factor
      for (size_t pos = 0; pos < fwd.subject_count(); ++pos) {
        if (!charge(1)) return 0;
        if (!place(pp.s, fwd.subject_at(pos))) continue;
        sum = sat_add(sum, sat_mul(fwd.degree(pos), eval(i + 1)));
        if (exhausted) return 0;
      }
      return sum;
    }
    if (s_free) { // o joins: per object, in-degree as factor
      for (size_t pos = 0; pos < rev.subject_count(); ++pos) {
        if (!charge(1)) return 0;
        if (!place(pp.o, rev.subject_at(pos))) continue;
        sum = sat_add(sum, sat_mul(rev.degree(pos), eval(i + 1)));
        if (exhausted) return 0;
      }
      return sum;
    }
    // Both nodes join with the predicate free: walk every row entry.
    for (size_t pos = 0; pos < fwd.subject_count(); ++pos) {
      if (!place(pp.s, fwd.subject_at(pos))) continue;
      std::span<const SubjectIndex::Entry> row = fwd.row(pos);
      if (!charge(row.size())) return 0;
      for (const SubjectIndex::Entry& e : row) {
        if (!place(pp.o, e.obj)) continue;
        sum = sat_add(sum, eval(i + 1));
        if (exhausted) return 0;
      }
    }
    return sum;
  }
};

} // namespace

int64_t count_query(const TripleStore& store, const SubjectIndex& fwd, const SubjectIndex& rev,
                    const Query& q, uint64_t budget) {
  Plan plan = compile(q, store);
  if (plan.never_matches) return 0;
  Ctx ctx{store, fwd, rev, plan, std::vector<TermId>(plan.n_slots, kInvalidTerm), budget};
  uint64_t count = ctx.eval(0);
  if (ctx.exhausted) return -1;
  constexpr auto kMax = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  return static_cast<int64_t>(std::min(count, kMax));
}

void count_cardinalities(const TripleStore& store, const SubjectIndex& fwd,
                         const SubjectIndex& rev, std::span<Query* const> queries,
                         const CardinalityOptions& opt) {
  parallel_for(queries.size(), 1, opt.threads, [&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i)
      queries[i]->y = count_query(store, fwd, rev, *queries[i], opt.budget);
  });
}

} // namespace bgpgen
