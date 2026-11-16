#pragma once

#include "bgpgen/subject_index.hpp"
#include "bgpgen/template.hpp"

#include <vector>

namespace bgpgen {

/// Match counts over the template's spanning tree (its non-Check steps).
///
/// weight(t, v) is the number of ways to match the part of the template
/// below slot t when t is mapped to graph node v. It is counted over the
/// IRI->IRI subgraph the matcher walks, one per (node assignment, triple per
/// edge), and it does NOT require distinct slots to get distinct nodes.
/// Leaf slots have weight 1.
///
/// Two uses: weight > 0 is a necessary condition for a match (pruning), and
/// choosing the root and every neighbour proportionally to its weight draws
/// each spanning-tree match with the same probability.
class MatchWeights {
public:
  /// One pass over the adjacency rows per non-Check step. Call after any
  /// reordering of tmpl.steps: step sums are indexed by step position.
  ///
  /// With `degree_check` a node gets weight 0 for a slot if it has fewer
  /// different neighbours than the slot has in the template (see
  /// SlotDegrees). Such a node is in no match with different nodes for
  /// different slots. The weights then count fewer than all spanning-tree
  /// matches, and total() with them.
  void compute(const SubjectIndex& fwd, const SubjectIndex& rev, const CompiledTemplate& tmpl,
               TermId bnode_base, bool degree_check = false);

  bool leaf(uint8_t slot) const { return w_[slot].empty(); }
  double weight(uint8_t slot, TermId node) const {
    const std::vector<double>& w = w_[slot];
    return w.empty() ? 1.0 : w[node];
  }
  /// Sum of the child's weights over the neighbours of `node` along step
  /// `step`. Only stored for steps whose child slot is not a leaf.
  double step_sum(size_t step, TermId node) const { return step_sum_[step][node]; }

  /// Number of spanning-tree matches in the graph (sum of the root weights).
  double total() const { return total_; }

private:
  std::vector<std::vector<double>> w_;        // per slot; empty = leaf (all ones)
  std::vector<std::vector<double>> step_sum_; // per step; empty unless child is inner
  double total_ = 0.0;
};

} // namespace bgpgen
