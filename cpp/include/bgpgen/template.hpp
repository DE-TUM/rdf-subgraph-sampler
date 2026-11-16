#pragma once

#include "bgpgen/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bgpgen {

/// How a template position is rendered in the final query.
enum class Bind : uint8_t {
  Var,   // always a variable
  Bound, // always the matched constant
  Prob   // constant with probability p_node / p_edge, drawn per occurrence
};

struct TemplateTriple {
  uint8_t s_slot, o_slot; // node slots
  Bind s_bind, o_bind;
  Bind p_bind;
};

/// One DFS extension step, fully resolved at compile time (the analog of a
/// FastShacl shape IR instruction): which edge, which direction, which slot
/// is known and which gets assigned.
struct Step {
  enum class Dir : uint8_t {
    Fwd,  // subject known -> assign object from forward adjacency
    Rev,  // object known -> assign subject from reverse adjacency
    Check // both known -> verify the edge exists, pick a predicate
  };
  uint8_t edge;
  Dir dir;
  uint8_t new_slot; // unused for Check
};

/// A parsed + compiled shape template. Traversal order is BFS from the
/// highest-degree subject node (Python semantics); for path templates the
/// root is forced to ?o0 so the DFS walks the chain from one end.
struct CompiledTemplate {
  std::vector<TemplateTriple> triples; // template order (= output order)
  std::vector<Step> steps;             // traversal order
  uint8_t root_slot = 0;
  uint8_t n_slots = 0;
  std::vector<std::string> slot_names;     // including '?', e.g. "?o0", "?r"
  std::vector<std::string> pred_var_names; // per edge, e.g. "?p0" (path: "?p1")
  uint32_t max_candidates = 50;            // DFS branching cap per step
  bool path_entity_order = false; // x order per triple: s,o,p (path) vs s,p,o
};

/// Parses a shape string like "?r B ?a, ?r ?/B ?b/B" (see the Python
/// complex generator's format). Throws BgpError on malformed input or a
/// disconnected shape.
CompiledTemplate parse_template(const std::string& shape);

/// Chain template for `--shape path --size N`: ?o0 -[?p1]-> ?o1 ... ?oN.
/// Ends are Prob (bound with p_start_end), intermediates Var, predicates
/// Prob (bound with p_edge). No candidate cap (the Python path DFS had
/// none).
CompiledTemplate make_chain_template(unsigned length);

/// Per slot: to how many different slots it has an out-edge, from how many an
/// in-edge, and how many different neighbour slots it has in all. In a match
/// that gives different slots different graph nodes, the node of a slot needs
/// at least that many different out-neighbours, in-neighbours and neighbours.
struct SlotDegrees {
  std::vector<uint8_t> out, in, all;
};
SlotDegrees slot_degrees(const CompiledTemplate& t);

/// The symmetries of the template: the renamings of its slots that map the
/// set of edges onto itself. They come in two parts.
///
/// `twin_edges`: groups of two or more edges whose far ends are leaves hanging
/// at the same slot in the same direction (edge indices ascending). Any
/// reordering inside a group is a symmetry.
///
/// `perms`: every symmetry that keeps the order inside each group, as the edge
/// permutation it causes (`perm[j]` = the edge whose image is edge j). The
/// identity is always among them.
///
/// Every symmetry of the template is one of `perms` followed by reorderings
/// inside the groups. `perms` is empty (and the result unusable) if the
/// template has parallel edges or more than `max_count` such symmetries.
struct TemplateSymmetries {
  std::vector<std::vector<uint8_t>> twin_edges;
  std::vector<std::vector<uint8_t>> perms;
};
TemplateSymmetries template_symmetries(const CompiledTemplate& t, size_t max_count);

/// Moves every Check step directly behind the step that assigns its second
/// endpoint, so a missing cycle-closing edge is noticed before unrelated
/// branches are extended. The set of matches is unchanged; the order of
/// random draws (and so the sampled workload for a given seed) is not.
void reorder_checks_early(CompiledTemplate& t);

} // namespace bgpgen
