// Evaluation driver for the sampling options of sample_complex: pruning by
// match counts, weighted sampling, early cycle checks, per-size seeds.
//
//   bgpgen_eval --data graph.nt --name NAME --out results.jsonl
//               [--num 1000] [--seeds 3] [--card-budget 5000000] [-j N]
//
// Writes one JSON object per line: one per (template, mode, seed), plus
// "seed_overlap" records. Analysed by bench/analyze_improvements.py.

#include "bgpgen/cardinality.hpp"
#include "bgpgen/graph.hpp"
#include "bgpgen/nt_parser.hpp"
#include "bgpgen/sampler_complex.hpp"
#include "bgpgen/subject_index.hpp"
#include "bgpgen/template.hpp"
#include "bgpgen/term_table.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace bgpgen;

namespace {

struct TemplateSpec {
  std::string name;
  unsigned chain_len; // > 0: chain template; 0: parse `shape`
  std::string shape;
  bool cyclic;
};

struct Mode {
  std::string name;
  bool prune;
  bool weighted;
  double alpha;
  bool early;
  bool auto_prune = false;
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

CompiledTemplate compile(const TemplateSpec& spec, bool early) {
  CompiledTemplate t =
      spec.chain_len ? make_chain_template(spec.chain_len) : parse_template(spec.shape);
  if (early) reorder_checks_early(t);
  return t;
}

ComplexOptions options_for(const TemplateSpec& spec, const Mode& mode, size_t num) {
  ComplexOptions co;
  co.num = num;
  co.p_edge = 1.0;
  co.p_node = 0.3; // chain: probability that an end node is bound
  if (spec.chain_len) { // the CLI's path constants
    co.min_new_per_pass = 0;
    co.max_consecutive = 200;
  }
  co.prune = mode.prune;
  co.auto_prune = mode.auto_prune;
  co.sampling = mode.weighted ? ComplexOptions::Sampling::Weighted : ComplexOptions::Sampling::Dfs;
  co.root_alpha = mode.alpha;
  return co;
}

template <class T>
void write_array(FILE* f, const char* key, const std::vector<T>& v) {
  std::fprintf(f, ",\"%s\":[", key);
  for (size_t i = 0; i < v.size(); ++i)
    std::fprintf(f, i ? ",%lld" : "%lld", static_cast<long long>(v[i]));
  std::fputc(']', f);
}

} // namespace

int main(int argc, char** argv) {
  std::string data, name, out_path;
  size_t num = 1000;
  unsigned seeds = 3, threads = 0;
  uint64_t card_budget = 5'000'000;
  std::string only; // comma-separated mode names; empty = all
  bool count = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--data") data = val();
    else if (a == "--name") name = val();
    else if (a == "--out") out_path = val();
    else if (a == "--num") num = std::stoul(val());
    else if (a == "--seeds") seeds = static_cast<unsigned>(std::stoul(val()));
    else if (a == "--card-budget") card_budget = std::stoull(val());
    else if (a == "--only") only = "," + val() + ",";
    else if (a == "--no-count") count = false;
    else if (a == "-j") threads = static_cast<unsigned>(std::stoul(val()));
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (data.empty() || name.empty() || out_path.empty()) {
    std::cerr << "usage: bgpgen_eval --data graph.nt --name NAME --out results.jsonl "
                 "[--num N] [--seeds N] [--card-budget N] [-j N]\n";
    return 2;
  }

  try {
    auto t_load = std::chrono::steady_clock::now();
    TermTable terms;
    std::vector<Triple> triples = load_nt_file(data, terms, 0, threads);
    std::vector<Triple>* sets[] = {&triples};
    terms.freeze(sets);
    TripleStore store;
    store.build(std::move(triples), threads);
    SubjectIndex fwd, rev;
    fwd.build(store, terms.bnode_base());
    rev.build(store, terms.bnode_base(), /*reverse=*/true);
    double load_s = seconds_since(t_load);
    std::fprintf(stderr, "%s: %zu triples, %zu predicates, %zu IRIs, loaded in %.1f s\n",
                 name.c_str(), store.triple_count(), store.predicate_count(), terms.iri_count(),
                 load_s);

    FILE* f = std::fopen(out_path.c_str(), "w");
    if (!f) throw BgpError("cannot write " + out_path);
    std::fprintf(f,
                 "{\"kind\":\"graph\",\"kg\":\"%s\",\"triples\":%zu,\"predicates\":%zu,"
                 "\"iris\":%zu,\"load_s\":%.3f}\n",
                 name.c_str(), store.triple_count(), store.predicate_count(), terms.iri_count(),
                 load_s);

    const std::vector<TemplateSpec> templates = {
        {"path-2", 2, "", false},
        {"path-3", 3, "", false},
        {"path-5", 5, "", false},
        {"path-8", 8, "", false},
        {"in-star-3", 0, "?x/B B ?m, ?y/B B ?m, ?m B ?z/B", false},
        {"tree-6", 0, "?r B ?a, ?r B ?b, ?a B ?c/B, ?a B ?d/B, ?b B ?e/B, ?b B ?f/B", false},
        {"flower-7", 0,
         "?a B ?l1/B, ?a B ?l2/B, ?a B ?b, ?b B ?m1/B, ?b B ?c, ?c B ?r1/B, ?c B ?r2/B", false},
        {"snowflake-8", 0,
         "?center B ?a, ?center B ?b, ?center B ?c, ?center B ?d, ?a B ?l1/B, ?a B ?l2/B, "
         "?b B ?r1/B, ?b B ?r2/B",
         false},
        {"triangle-3", 0, "?a B ?b, ?b B ?c, ?c B ?a", true},
        {"diamond-4", 0, "?a B ?b, ?a B ?c, ?b B ?d, ?c B ?d", true},
        {"triangle-tails-5", 0, "?a B ?b, ?b B ?c, ?c B ?a, ?a B ?x/B, ?a B ?y/B", true},
    };
    const std::vector<Mode> modes = {
        {"baseline", false, false, 1.0, false},
        {"prune", true, false, 1.0, false},
        {"early", false, false, 1.0, true},
        {"prune+early", true, false, 1.0, true},
        {"weighted-a0", false, true, 0.0, false},
        {"weighted-a0.5", false, true, 0.5, false},
        {"weighted-a1", false, true, 1.0, false},
        {"v2-default", false, false, 1.0, true, true}, // early checks + auto prune
    };

    for (const TemplateSpec& spec : templates) {
      for (const Mode& mode : modes) {
        if (!only.empty() && only.find("," + mode.name + ",") == std::string::npos) continue;
        if (mode.early && !spec.cyclic && !mode.auto_prune) continue; // no Check steps to move
        for (unsigned seed = 0; seed < seeds; ++seed) {
          CompiledTemplate tmpl = compile(spec, mode.early);
          ComplexOptions co = options_for(spec, mode, num);
          Dedup dedup(DedupMethod::Wl);
          std::mt19937_64 rng(1000 + seed);
          SampleReport rep;
          auto t0 = std::chrono::steady_clock::now();
          std::vector<Query> qs = sample_complex(store, fwd, rev, terms, tmpl, co, dedup, rng, &rep);
          double total_s = seconds_since(t0);

          std::fprintf(f,
                       "{\"kind\":\"run\",\"kg\":\"%s\",\"template\":\"%s\",\"n_triples\":%zu,"
                       "\"cyclic\":%s,\"mode\":\"%s\",\"seed\":%u,\"requested\":%zu,"
                       "\"unique\":%zu,\"attempts\":%zu,\"failed\":%zu,\"expansions\":%llu,"
                       "\"duplicates\":%zu,\"reshuffles\":%zu,\"start_candidates\":%zu,"
                       "\"tree_matches\":%.6g,\"weights_s\":%.6f,\"sample_s\":%.6f,"
                       "\"stop\":\"%s\"",
                       name.c_str(), spec.name.c_str(), tmpl.triples.size(),
                       spec.cyclic ? "true" : "false", mode.name.c_str(), seed, num, qs.size(),
                       rep.attempts, rep.failed_matches,
                       static_cast<unsigned long long>(rep.expansions), rep.duplicates_skipped,
                       rep.reshuffles, rep.start_candidates, rep.tree_matches,
                       rep.weights_seconds, total_s - rep.weights_seconds,
                       rep.stop_reason.c_str());

          std::fprintf(f, ",\"auto_pruned\":%s", rep.auto_pruned ? "true" : "false");
          if (count && seed == 0 && !qs.empty()) {
            // Distribution data: exact cardinalities and the degree of each root.
            auto tc = std::chrono::steady_clock::now();
            count_cardinalities(store, fwd, rev, qs, {card_budget, threads});
            double count_s = seconds_since(tc);
            std::vector<int64_t> y, out_deg, in_deg;
            for (size_t i = 0; i < qs.size(); ++i) {
              y.push_back(qs[i].y);
              out_deg.push_back(fwd.out_degree(rep.roots[i]));
              in_deg.push_back(rev.out_degree(rep.roots[i]));
            }
            std::fprintf(f, ",\"count_s\":%.6f", count_s);
            write_array(f, "y", y);
            write_array(f, "root_out_degree", out_deg);
            write_array(f, "root_in_degree", in_deg);
          }
          std::fputs("}\n", f);
          std::fflush(f);
          std::fprintf(stderr, "  %-17s %-14s seed %u: %zu unique, %zu attempts, %.3f s\n",
                       spec.name.c_str(), mode.name.c_str(), seed, qs.size(), rep.attempts,
                       total_s);
        }
      }
    }

    // Seed overlap between path sizes: the CLI used to seed every size with
    // the same value. Compare the root sequences of two sizes.
    auto roots_of = [&](unsigned len, uint64_t seed) {
      TemplateSpec spec{"path", len, "", false};
      Mode mode{"baseline", false, false, 1.0, false};
      CompiledTemplate tmpl = compile(spec, false);
      ComplexOptions co = options_for(spec, mode, num);
      Dedup dedup(DedupMethod::Wl);
      std::mt19937_64 rng(seed);
      SampleReport rep;
      sample_complex(store, fwd, rev, terms, tmpl, co, dedup, rng, &rep);
      return rep.roots;
    };
    const unsigned pairs[][2] = {{2, 3}, {3, 5}, {5, 8}};
    for (const auto& pr : pairs) {
      for (int independent = 0; independent < 2; ++independent) {
        // "independent" mimics --independent-sizes with two unrelated seeds.
        std::vector<TermId> a = roots_of(pr[0], 42);
        std::vector<TermId> b = roots_of(pr[1], independent ? 4242 : 42);
        size_t n = std::min(a.size(), b.size()), same_pos = 0;
        for (size_t i = 0; i < n; ++i) same_pos += a[i] == b[i];
        std::set<TermId> sa(a.begin(), a.end()), sb(b.begin(), b.end());
        size_t inter = 0;
        for (TermId v : sa) inter += sb.count(v);
        size_t uni = sa.size() + sb.size() - inter;
        std::fprintf(f,
                     "{\"kind\":\"seed_overlap\",\"kg\":\"%s\",\"size_a\":%u,\"size_b\":%u,"
                     "\"independent\":%s,\"n\":%zu,\"same_position\":%zu,\"jaccard\":%.4f}\n",
                     name.c_str(), pr[0], pr[1], independent ? "true" : "false", n, same_pos,
                     uni ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0);
      }
    }
    std::fclose(f);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "bgpgen_eval: " << e.what() << "\n";
    return 2;
  }
}
