#include "bgpgen/template.hpp"

#include <algorithm>
#include <deque>

namespace bgpgen {

namespace {

struct NodeToken {
  std::string name; // without '?' / '/B'
  Bind bind;
};

std::string strip(const std::string& s) {
  size_t b = s.find_first_not_of(" \t");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t");
  return s.substr(b, e - b + 1);
}

NodeToken parse_node_token(std::string tok) {
  bool prob = tok.size() > 2 && tok.compare(tok.size() - 2, 2, "/B") == 0;
  if (prob) tok.resize(tok.size() - 2);
  bool var = !tok.empty() && tok[0] == '?';
  if (var) tok.erase(0, 1);
  if (tok.empty()) throw BgpError("empty node name in template");
  return {tok, prob ? Bind::Prob : (var ? Bind::Var : Bind::Bound)};
}

Bind parse_pred_token(const std::string& tok) {
  if (tok == "B") return Bind::Bound;
  if (tok == "?") return Bind::Var;
  if (tok == "?/B") return Bind::Prob;
  throw BgpError("unknown predicate binding '" + tok + "' (use B, ?, or ?/B)");
}

/// Computes the BFS traversal order and resolves each edge's direction.
/// `root_slot` < 0 picks the highest-degree subject node (first wins on
/// ties, like Python's max()).
void compile_traversal(CompiledTemplate& t, int forced_root) {
  size_t n_edges = t.triples.size();
  std::vector<uint32_t> degree(t.n_slots, 0);
  std::vector<bool> is_subject(t.n_slots, false);
  for (const TemplateTriple& tt : t.triples) {
    ++degree[tt.s_slot];
    ++degree[tt.o_slot];
    is_subject[tt.s_slot] = true;
  }

  uint8_t root;
  if (forced_root >= 0) {
    root = static_cast<uint8_t>(forced_root);
  } else {
    // Slots are numbered in first-appearance order, matching the Python
    // Counter's iteration order; pick the first max-degree subject node.
    int best = -1;
    for (uint8_t s = 0; s < t.n_slots; ++s)
      if (is_subject[s] && (best < 0 || degree[s] > degree[best])) best = s;
    if (best < 0) best = 0; // no subject nodes can't happen, but stay safe
    root = static_cast<uint8_t>(best);
  }
  t.root_slot = root;

  std::vector<bool> visited(t.n_slots, false);
  std::vector<bool> assigned(t.n_slots, false);
  visited[root] = assigned[root] = true;
  std::deque<uint8_t> queue{root};
  std::vector<bool> done(n_edges, false);
  size_t remaining = n_edges;

  while (!queue.empty() && remaining > 0) {
    uint8_t node = queue.front();
    queue.pop_front();
    for (size_t e = 0; e < n_edges; ++e) {
      if (done[e]) continue;
      const TemplateTriple& tt = t.triples[e];
      uint8_t other;
      if (tt.s_slot == node)
        other = tt.o_slot;
      else if (tt.o_slot == node)
        other = tt.s_slot;
      else
        continue;
      done[e] = true;
      --remaining;

      Step st;
      st.edge = static_cast<uint8_t>(e);
      if (assigned[tt.s_slot] && assigned[tt.o_slot]) {
        st.dir = Step::Dir::Check;
        st.new_slot = 0;
      } else if (assigned[tt.s_slot]) {
        st.dir = Step::Dir::Fwd;
        st.new_slot = tt.o_slot;
        assigned[tt.o_slot] = true;
      } else {
        st.dir = Step::Dir::Rev;
        st.new_slot = tt.s_slot;
        assigned[tt.s_slot] = true;
      }
      t.steps.push_back(st);

      if (!visited[other]) {
        visited[other] = true;
        queue.push_back(other);
      }
    }
  }
  if (remaining > 0)
    throw BgpError("query shape is disconnected: " + std::to_string(remaining) +
                   " edge(s) unreachable from the root");
}

} // namespace

CompiledTemplate parse_template(const std::string& shape) {
  CompiledTemplate t;
  std::vector<std::string> names;        // slot -> bare name
  std::vector<Bind> bindings;            // slot -> binding

  auto slot_of = [&](const NodeToken& tok) -> uint8_t {
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == tok.name) {
        if (bindings[i] != tok.bind)
          throw BgpError("node '" + tok.name + "' has inconsistent binding across occurrences");
        return static_cast<uint8_t>(i);
      }
    }
    if (names.size() >= 255) throw BgpError("too many template nodes");
    names.push_back(tok.name);
    bindings.push_back(tok.bind);
    return static_cast<uint8_t>(names.size() - 1);
  };

  size_t pos = 0;
  while (pos <= shape.size()) {
    size_t comma = shape.find(',', pos);
    std::string part = strip(shape.substr(pos, comma == std::string::npos ? std::string::npos
                                                                          : comma - pos));
    pos = comma == std::string::npos ? shape.size() + 1 : comma + 1;
    if (part.empty()) {
      if (comma == std::string::npos) break;
      throw BgpError("empty triple group in template");
    }

    std::vector<std::string> toks;
    size_t i = 0;
    while (i < part.size()) {
      size_t b = part.find_first_not_of(" \t", i);
      if (b == std::string::npos) break;
      size_t e = part.find_first_of(" \t", b);
      if (e == std::string::npos) e = part.size();
      toks.push_back(part.substr(b, e - b));
      i = e;
    }
    if (toks.size() != 3)
      throw BgpError("expected 3 tokens in '" + part + "', got " + std::to_string(toks.size()));

    NodeToken s = parse_node_token(toks[0]);
    Bind p = parse_pred_token(toks[1]);
    NodeToken o = parse_node_token(toks[2]);

    TemplateTriple tt;
    tt.s_slot = slot_of(s);
    tt.s_bind = s.bind;
    tt.p_bind = p;
    tt.o_slot = slot_of(o);
    tt.o_bind = o.bind;
    t.triples.push_back(tt);
  }
  if (t.triples.empty()) throw BgpError("template has no triples");
  if (t.triples.size() > 255) throw BgpError("too many template triples");

  t.n_slots = static_cast<uint8_t>(names.size());
  for (const std::string& n : names) t.slot_names.push_back("?" + n);
  for (size_t e = 0; e < t.triples.size(); ++e)
    t.pred_var_names.push_back("?p" + std::to_string(e));

  compile_traversal(t, -1);
  return t;
}

CompiledTemplate make_chain_template(unsigned length) {
  if (length < 1) throw BgpError("path length must be >= 1");
  if (length > 254) throw BgpError("path length too large");
  CompiledTemplate t;
  t.n_slots = static_cast<uint8_t>(length + 1);
  for (unsigned i = 0; i <= length; ++i) t.slot_names.push_back("?o" + std::to_string(i));
  for (unsigned i = 0; i < length; ++i) {
    TemplateTriple tt;
    tt.s_slot = static_cast<uint8_t>(i);
    tt.o_slot = static_cast<uint8_t>(i + 1);
    tt.s_bind = i == 0 ? Bind::Prob : Bind::Var;
    tt.o_bind = i == length - 1 ? Bind::Prob : Bind::Var;
    tt.p_bind = Bind::Prob;
    t.triples.push_back(tt);
    t.pred_var_names.push_back("?p" + std::to_string(i + 1)); // path names are 1-based
  }
  t.max_candidates = 0xFFFFFFFFu; // the Python path DFS has no branching cap
  t.path_entity_order = true;
  compile_traversal(t, /*forced_root=*/0); // walk the chain from ?o0
  return t;
}

void reorder_checks_early(CompiledTemplate& t) {
  std::vector<Step> checks;
  for (const Step& st : t.steps)
    if (st.dir == Step::Dir::Check) checks.push_back(st);
  if (checks.empty()) return;

  std::vector<Step> out;
  out.reserve(t.steps.size());
  std::vector<bool> assigned(t.n_slots, false);
  std::vector<bool> emitted(checks.size(), false);
  auto flush = [&] {
    for (size_t i = 0; i < checks.size(); ++i) {
      const TemplateTriple& tt = t.triples[checks[i].edge];
      if (!emitted[i] && assigned[tt.s_slot] && assigned[tt.o_slot]) {
        out.push_back(checks[i]);
        emitted[i] = true;
      }
    }
  };
  assigned[t.root_slot] = true;
  flush(); // self-loops on the root
  for (const Step& st : t.steps) {
    if (st.dir == Step::Dir::Check) continue;
    out.push_back(st);
    assigned[st.new_slot] = true;
    flush();
  }
  t.steps = std::move(out);
}

SlotDegrees slot_degrees(const CompiledTemplate& t) {
  SlotDegrees d;
  d.out.assign(t.n_slots, 0);
  d.in.assign(t.n_slots, 0);
  d.all.assign(t.n_slots, 0);
  for (size_t i = 0; i < t.triples.size(); ++i) {
    const TemplateTriple& a = t.triples[i];
    if (a.s_slot == a.o_slot) continue; // a loop needs no second node
    // Earlier edges between the same two slots: the neighbour is counted once.
    bool parallel = false, opposite = false;
    for (size_t j = 0; j < i; ++j) {
      parallel |= t.triples[j].s_slot == a.s_slot && t.triples[j].o_slot == a.o_slot;
      opposite |= t.triples[j].s_slot == a.o_slot && t.triples[j].o_slot == a.s_slot;
    }
    if (parallel) continue;
    ++d.out[a.s_slot];
    ++d.in[a.o_slot];
    if (opposite) continue;
    ++d.all[a.s_slot];
    ++d.all[a.o_slot];
  }
  return d;
}

TemplateSymmetries template_symmetries(const CompiledTemplate& t, size_t max_count) {
  TemplateSymmetries result;
  const size_t n = t.n_slots, m = t.triples.size();
  std::vector<int> edge_at(n * n, -1); // (s, o) -> edge
  for (size_t i = 0; i < m; ++i) {
    int& e = edge_at[t.triples[i].s_slot * n + t.triples[i].o_slot];
    if (e >= 0) return result;
    e = static_cast<int>(i);
  }
  const SlotDegrees deg = slot_degrees(t);

  // Twin leaves: slots with a single edge, grouped by the slot they hang at
  // and the direction of that edge. twin_size / twin_index place every slot
  // in its group (a slot outside any group is a group of its own).
  std::vector<size_t> incident(n, 0), twin_size(n, 1), twin_index(n, 0);
  for (size_t i = 0; i < m; ++i) {
    ++incident[t.triples[i].s_slot];
    ++incident[t.triples[i].o_slot];
  }
  for (size_t hub = 0; hub < n; ++hub)
    for (bool leaf_is_object : {true, false}) {
      std::vector<uint8_t> group;
      std::vector<size_t> leaves;
      for (size_t i = 0; i < m; ++i) {
        size_t s = t.triples[i].s_slot, o = t.triples[i].o_slot;
        size_t leaf = leaf_is_object ? o : s;
        if ((leaf_is_object ? s : o) != hub || leaf == hub || incident[leaf] != 1) continue;
        group.push_back(static_cast<uint8_t>(i));
        leaves.push_back(leaf);
      }
      if (group.size() < 2) continue;
      for (size_t k = 0; k < leaves.size(); ++k) {
        twin_size[leaves[k]] = leaves.size();
        twin_index[leaves[k]] = k;
      }
      result.twin_edges.push_back(std::move(group));
    }

  // Backtracking over the slots in order. A one-to-one slot map that sends
  // every edge to an edge is a symmetry, because the number of edges is the
  // same on both sides. A twin leaf may only go to the leaf at the same place
  // in a group of the same size, which keeps the order inside the groups.
  std::vector<std::vector<uint8_t>>& out = result.perms;
  std::vector<size_t> image(n, 0);
  std::vector<bool> taken(n, false);
  bool too_many = false;
  auto place = [&](auto&& self, size_t slot) -> void {
    if (too_many) return;
    if (slot == n) {
      if (out.size() == max_count) {
        too_many = true;
        return;
      }
      std::vector<uint8_t> perm(m);
      for (size_t i = 0; i < m; ++i)
        perm[edge_at[image[t.triples[i].s_slot] * n + image[t.triples[i].o_slot]]] =
            static_cast<uint8_t>(i);
      out.push_back(std::move(perm));
      return;
    }
    for (size_t c = 0; c < n; ++c) {
      if (taken[c] || deg.out[c] != deg.out[slot] || deg.in[c] != deg.in[slot]) continue;
      if (twin_size[c] != twin_size[slot] || twin_index[c] != twin_index[slot]) continue;
      image[slot] = c;
      bool ok = true;
      for (size_t i = 0; i < m && ok; ++i) {
        size_t s = t.triples[i].s_slot, o = t.triples[i].o_slot;
        if (std::max(s, o) != slot) continue; // edges that just became complete
        ok = edge_at[image[s] * n + image[o]] >= 0;
      }
      if (!ok) continue;
      taken[c] = true;
      self(self, slot + 1);
      taken[c] = false;
    }
  };
  place(place, 0);
  if (too_many) out.clear();
  return result;
}

} // namespace bgpgen
