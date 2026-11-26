#include "bgpgen/cardinality.hpp"
#include "bgpgen/graph.hpp"
#include "bgpgen/gz_reader.hpp"
#include "bgpgen/json_writer.hpp"
#include "bgpgen/nt_parser.hpp"
#include "bgpgen/sampler_complex.hpp"
#include "bgpgen/sampler_star.hpp"
#include "bgpgen/subject_index.hpp"
#include "bgpgen/template.hpp"
#include "bgpgen/term_table.hpp"
#include "bgpgen/workload.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* kUsage = R"(bgpgen - fast BGP query sampler with in-memory cardinalities

Usage: bgpgen --data graph.nt[.gz] --shape SHAPE --size N [options]

  --data FILE             N-Triples data graph (.nt or .nt.gz)
  --shape SHAPE           star | path | complex (omit to just load & print stats)
  --size N                query size; repeatable for batch runs (one output per size)
  --template "..."        complex shape template, e.g. "?r B ?a, ?r B ?b" (complex only)
  --num N                 queries per size (default 1000)
  --seed N                RNG seed (default 42)
  --p-edge P              prob. a path/complex predicate is bound (default 1.0)
  --p-node P              prob. a Prob-marked template node is bound (default 0.0)
  --p-start-end P         prob. a path endpoint is bound (default 0.3)
  --p-predicate P         prob. a star predicate is bound (default 1.0)
  --min-objects-instantiated N   star: min bound objects (default 0)
  --max-objects-instantiated N   star: max bound objects (default ceil(size/2))
  --dedup METHOD          symmetry (default) | wl | hash. symmetry gives the same
                          result as wl and is much faster for path/complex
  --no-cardinality        skip the in-memory COUNT(*) pass (y = -1)
  --card-budget N         max enumeration steps per count (default 50000000)
  --prune                 path/complex: skip start nodes and neighbours that cannot
                          complete the template (match counts over its spanning tree)
                          from the start. Default: switched on during the run once
                          failed attempts have cost as much as computing the counts
  --no-auto-prune         never switch pruning on during the run
  --attempt-budget N      path/complex, dfs: give up on a start node after looking at
                          N neighbour candidates (default 0 = no limit)
  --no-degree-check       path/complex: also try graph nodes with fewer different
                          neighbours than their template node needs (no match
                          can use them; default: skip them)
  --pred-first            path/complex, dfs: choose a predicate of the node first, then
                          a neighbour under it (default: choose among all edges of
                          the node, so frequent predicates are chosen more often)
  --novelty B             with --pred-first (implied): prefer predicates used less
                          often so far at the same template edge, weight
                          1/(1+uses)^B (default 0 = every predicate equally likely)
  --skip-known            path/complex, dfs: when a match gives a query that is already
                          in the workload, keep searching from the same start node
  --max-known K           with --skip-known: give up on a start node after K matches
                          with known queries (default 0 = no limit)
  --per-seed K            with --skip-known: up to K new queries per visit of a start
                          node (default 1). auto = 1 in the first pass over the start
                          nodes, twice as many in every further pass
  --retire-seeds          with --skip-known: a start node that gave no new query in a
                          pass is left out of later passes
  --seed-groups           path/complex, dfs: visit start nodes with different sets of
                          out-predicates first (default: uniformly random order)
  --card-buckets N        path/complex, dfs: the same number of queries in each of N
                          ranges of the result size, equally wide on a log scale
                          between 1 and --card-max. Counts every new query when it
                          is found; implies --skip-known and --pred-first
  --card-max X            upper end of the last range (default 1e6)
  --no-card-steer         with --card-buckets: do not steer the predicate choice
                          towards the ranges that still lack queries
  --sample-threads N      path/complex, dfs: search with N threads (default 1; implies
                          --skip-known). The workload is repeatable for the same
                          seed and N, and differs between values of N
  --sampling MODE         path/complex: dfs (default) | weighted. weighted draws the
                          root ~ count^alpha and neighbours ~ count, no backtracking
  --root-alpha A          weighted: 1 = every match equally likely (default),
                          0 = every feasible root equally likely
  --no-early-checks       complex: check cycle-closing edges in BFS order (default:
                          as early as possible)
  --shared-seed           seed every size with --seed itself (default: a seed
                          derived from --seed and the size)
  --v1-compat             the three switches above and --no-degree-check: same
                          output as the first C++ version for the same seed
  --dataset NAME          dataset name for output files (default: data file stem)
  --output-dir DIR        output directory (default .)
  -j, --threads N         worker threads (default: all cores)
      --no-output         do not write the output file (for timing runs)
      --timing            print per-phase timings to stderr
  -h, --help              show this help

Output: {DATASET}_{SHAPE}_{timestamp}_{SIZE}.json with records
{"query": "SELECT * WHERE {...}", "triples": [[s,p,o],...], "x": [...], "y": N}
)";

struct Options {
  std::string data;
  std::string shape;
  std::vector<unsigned> sizes;
  std::string template_str;
  bgpgen::WorkloadOptions w; // what to sample; shared with the Python module
  bool cardinality = true;
  uint64_t card_budget = 50'000'000;
  std::string dataset;
  std::string output_dir = ".";
  unsigned threads = 0;
  bool timing = false;
  bool write_output = true;
};

Options parse_args(int argc, char** argv) {
  Options opt;
  auto value = [&](int& i, const char* flag) -> std::string {
    if (i + 1 >= argc) throw bgpgen::BgpError(std::string(flag) + " needs a value");
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--data") opt.data = value(i, "--data");
    else if (arg == "--shape") opt.shape = value(i, "--shape");
    else if (arg == "--size")
      opt.sizes.push_back(static_cast<unsigned>(std::stoul(value(i, "--size"))));
    else if (arg == "--template") opt.template_str = value(i, "--template");
    else if (arg == "--num") opt.w.num = std::stoul(value(i, "--num"));
    else if (arg == "--seed") opt.w.seed = std::stoull(value(i, "--seed"));
    else if (arg == "--p-edge") opt.w.p_edge = std::stod(value(i, "--p-edge"));
    else if (arg == "--p-node") opt.w.p_node = std::stod(value(i, "--p-node"));
    else if (arg == "--p-start-end") opt.w.p_start_end = std::stod(value(i, "--p-start-end"));
    else if (arg == "--p-predicate") opt.w.p_predicate = std::stod(value(i, "--p-predicate"));
    else if (arg == "--min-objects-instantiated")
      opt.w.min_objects = std::stoi(value(i, "--min-objects-instantiated"));
    else if (arg == "--max-objects-instantiated")
      opt.w.max_objects = std::stoi(value(i, "--max-objects-instantiated"));
    else if (arg == "--dedup") {
      std::string v = value(i, "--dedup");
      if (v == "symmetry") opt.w.dedup = bgpgen::DedupMethod::Symmetry;
      else if (v == "wl") opt.w.dedup = bgpgen::DedupMethod::Wl;
      else if (v == "hash") opt.w.dedup = bgpgen::DedupMethod::Hash;
      else throw bgpgen::BgpError("unknown dedup method '" + v + "'");
    }
    else if (arg == "--no-cardinality") opt.cardinality = false;
    else if (arg == "--card-budget") opt.card_budget = std::stoull(value(i, "--card-budget"));
    else if (arg == "--dataset") opt.dataset = value(i, "--dataset");
    else if (arg == "--output-dir") opt.output_dir = value(i, "--output-dir");
    else if (arg == "-j" || arg == "--threads")
      opt.threads = static_cast<unsigned>(std::stoul(value(i, "--threads")));
    else if (arg == "--timing") opt.timing = true;
    else if (arg == "--no-output") opt.write_output = false;
    else if (arg == "--prune") opt.w.prune = true;
    else if (arg == "--sampling") {
      std::string v = value(i, "--sampling");
      if (v == "dfs") opt.w.sampling = bgpgen::ComplexOptions::Sampling::Dfs;
      else if (v == "weighted") opt.w.sampling = bgpgen::ComplexOptions::Sampling::Weighted;
      else throw bgpgen::BgpError("unknown sampling mode '" + v + "'");
    }
    else if (arg == "--root-alpha") opt.w.root_alpha = std::stod(value(i, "--root-alpha"));
    else if (arg == "--no-auto-prune") opt.w.auto_prune = false;
    else if (arg == "--attempt-budget")
      opt.w.attempt_budget = std::stoull(value(i, "--attempt-budget"));
    else if (arg == "--no-degree-check") opt.w.degree_check = false;
    else if (arg == "--pred-first") opt.w.pred_first = true;
    else if (arg == "--novelty") {
      opt.w.novelty = std::stod(value(i, "--novelty"));
      opt.w.pred_first = true;
    }
    else if (arg == "--skip-known") opt.w.skip_known = true;
    else if (arg == "--per-seed") {
      std::string v = value(i, "--per-seed");
      opt.w.per_seed = v == "auto" ? 0 : std::stoul(v);
    }
    else if (arg == "--retire-seeds") opt.w.retire_seeds = true;
    else if (arg == "--max-known") opt.w.max_known = std::stoul(value(i, "--max-known"));
    else if (arg == "--seed-groups") opt.w.seed_groups = true;
    else if (arg == "--card-buckets")
      opt.w.card_buckets = static_cast<unsigned>(std::stoul(value(i, "--card-buckets")));
    else if (arg == "--card-max") opt.w.card_max = std::stod(value(i, "--card-max"));
    else if (arg == "--no-card-steer") opt.w.card_steer = false;
    else if (arg == "--sample-threads")
      opt.w.sample_threads = static_cast<unsigned>(std::stoul(value(i, "--sample-threads")));
    else if (arg == "--no-early-checks") opt.w.early_checks = false;
    else if (arg == "--shared-seed") opt.w.independent_sizes = false;
    else if (arg == "--v1-compat") {
      opt.w.auto_prune = false;
      opt.w.early_checks = false;
      opt.w.independent_sizes = false;
      opt.w.degree_check = false;
    }
    else if (arg == "-h" || arg == "--help") {
      std::cout << kUsage;
      std::exit(0);
    } else {
      throw bgpgen::BgpError("unknown argument '" + arg + "' (see --help)");
    }
  }
  if (opt.data.empty()) throw bgpgen::BgpError("--data is required (see --help)");
  if (!opt.shape.empty() && opt.shape != "star" && opt.shape != "path" && opt.shape != "complex")
    throw bgpgen::BgpError("unknown shape '" + opt.shape + "'");
  if (!opt.shape.empty() && opt.sizes.empty() && opt.shape != "complex")
    throw bgpgen::BgpError("--shape needs at least one --size");
  return opt;
}

struct PhaseTimer {
  bool enabled;
  std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();

  void lap(const char* phase) {
    auto now = std::chrono::steady_clock::now();
    if (enabled) {
      auto ms = std::chrono::duration<double, std::milli>(now - last).count();
      std::fprintf(stderr, "%-10s %8.1f ms\n", phase, ms);
    }
    last = now;
  }
};

/// "path/to/lubm-1.nt.gz" -> "lubm-1"
std::string data_stem(const std::string& path) {
  size_t slash = path.find_last_of('/');
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  for (const char* ext : {".gz", ".nt"}) {
    size_t n = std::strlen(ext);
    if (name.size() > n && name.compare(name.size() - n, n, ext) == 0)
      name.resize(name.size() - n);
  }
  return name;
}

std::string timestamp_now() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%S", &tm);
  return buf;
}

std::string hex16(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

void write_star_json(const std::string& path, const std::vector<bgpgen::StarRecord>& records,
                     const bgpgen::TermTable& terms) {
  using namespace bgpgen;
  std::ofstream f(path);
  if (!f) throw BgpError("cannot write '" + path + "'");
  JsonWriter jw(f);
  jw.begin_array();
  for (const StarRecord& r : records) {
    const Query& q = r.query;
    jw.begin_object();
    jw.key("query");
    jw.string(render_sparql(q, terms));
    jw.key("triples");
    jw.begin_array();
    for (const TriplePattern& tp : q.patterns) {
      jw.begin_array();
      jw.string(q_token(q, terms, tp.s));
      jw.string(q_token(q, terms, tp.p));
      jw.string(q_token(q, terms, tp.o));
      jw.end_array();
    }
    jw.end_array();
    jw.key("x");
    jw.begin_array();
    for (TermId t : q.x) jw.string(terms.lexical(t));
    jw.end_array();
    jw.key("y");
    jw.number(q.y);
    jw.key("source_subject");
    jw.string(terms.lexical(r.meta.source_subject));
    jw.key("star_size");
    jw.number(r.meta.star_size);
    jw.key("query_hash");
    jw.string(hex16(r.meta.query_hash));
    jw.key("predicates_used");
    jw.begin_array();
    for (TermId t : r.meta.predicates_used) jw.string(terms.lexical(t));
    jw.end_array();
    jw.key("instantiated_objects");
    jw.begin_array();
    for (TermId t : r.meta.instantiated_objects) jw.string(term_token(terms, t));
    jw.end_array();
    jw.key("object_instantiation_pattern");
    jw.begin_array();
    for (uint32_t i : r.meta.object_instantiation_pattern) jw.number(i);
    jw.end_array();
    jw.key("n_objects_instantiated");
    jw.number(r.meta.n_objects_instantiated);
    jw.end_object();
  }
  jw.end_array();
}

/// Plain {query,triples,x,y} records; complex uses that key order, path
/// uses (x, y, query, triples) — both matching their Python generators.
void write_query_json(const std::string& path, const std::vector<bgpgen::Query>& records,
                      const bgpgen::TermTable& terms, bool path_order) {
  using namespace bgpgen;
  std::ofstream f(path);
  if (!f) throw BgpError("cannot write '" + path + "'");
  JsonWriter jw(f);
  jw.begin_array();
  for (const Query& q : records) {
    jw.begin_object();
    auto write_x_y = [&] {
      jw.key("x");
      jw.begin_array();
      for (TermId t : q.x) jw.string(terms.lexical(t));
      jw.end_array();
      jw.key("y");
      jw.number(q.y);
    };
    auto write_query_triples = [&] {
      jw.key("query");
      jw.string(render_sparql(q, terms));
      jw.key("triples");
      jw.begin_array();
      for (const TriplePattern& tp : q.patterns) {
        jw.begin_array();
        jw.string(q_token(q, terms, tp.s));
        jw.string(q_token(q, terms, tp.p));
        jw.string(q_token(q, terms, tp.o));
        jw.end_array();
      }
      jw.end_array();
    };
    if (path_order) {
      write_x_y();
      write_query_triples();
    } else {
      write_query_triples();
      write_x_y();
    }
    jw.end_object();
  }
  jw.end_array();
}

} // namespace

int main(int argc, char** argv) {
  using namespace bgpgen;
  try {
    Options opt = parse_args(argc, argv);
    PhaseTimer timer{opt.timing};

    TermTable terms;
    std::vector<Triple> triples;
    if (is_gz_path(opt.data)) {
      std::vector<char> buf = read_gz_file(opt.data);
      timer.lap("gunzip");
      triples = parse_nt_buffer(buf.data(), buf.size(), terms, 0, opt.threads);
    } else {
      triples = load_nt_file(opt.data, terms, 0, opt.threads);
    }
    timer.lap("parse");

    std::vector<Triple>* sets[] = {&triples};
    terms.freeze(sets);
    size_t parsed = triples.size();
    TripleStore store;
    store.build(std::move(triples), opt.threads);
    timer.lap("index");

    std::fprintf(stderr, "%zu triples (%zu distinct), %zu predicates, %zu terms "
                         "(%zu iris, %u bnodes, %zu literals)\n",
                 parsed, store.triple_count(), store.predicate_count(), terms.size(),
                 terms.iri_count(), terms.literal_base() - terms.bnode_base(),
                 terms.size() - terms.literal_base());

    if (opt.shape.empty()) return 0;

    std::string dataset = opt.dataset.empty() ? data_stem(opt.data) : opt.dataset;
    std::string ts = timestamp_now();
    if (opt.shape == "star") {
      SubjectIndex sindex, rindex;
      sindex.build(store, terms.bnode_base());
      if (opt.cardinality) rindex.build(store, terms.bnode_base(), /*reverse=*/true);
      timer.lap("subj-index");

      for (unsigned size : opt.sizes) {
        SampleReport rep;
        std::vector<StarRecord> records =
            sample_star_workload(store, sindex, terms, size, opt.w, &rep);
        timer.lap("sample");

        if (opt.cardinality) {
          std::vector<Query*> qs;
          qs.reserve(records.size());
          for (StarRecord& r : records) qs.push_back(&r.query);
          count_cardinalities(store, sindex, rindex, qs, {opt.card_budget, opt.threads});
          timer.lap("count");
          size_t timeouts = 0;
          for (const StarRecord& r : records) timeouts += r.query.y < 0;
          if (timeouts)
            std::fprintf(stderr, "size %u: %zu cardinalities hit the budget (y = -1)\n", size,
                         timeouts);
        }

        std::string fn = opt.output_dir + "/" + dataset + "_stars_" + ts + "_" +
                         std::to_string(size) + ".json";
        write_star_json(fn, records, terms);
        timer.lap("write");

        std::fprintf(stderr, "size %u: %zu/%zu unique queries (%zu duplicates skipped, "
                             "%zu reshuffles)%s%s -> %s\n",
                     size, records.size(), opt.w.num, rep.duplicates_skipped, rep.reshuffles,
                     rep.stop_reason.empty() ? "" : " - stopped: ",
                     rep.stop_reason.c_str(), fn.c_str());
      }
      return 0;
    }

    // path / complex: DFS template matching on the IRI->IRI subgraph.
    auto print_match_stats = [&](const SampleReport& rep, const std::vector<Query>& records) {
      if (opt.timing) {
        std::fprintf(stderr, "  %zu attempts (%zu without a match, %zu of them given up), "
                             "%llu candidates looked at, %zu start nodes skipped, "
                             "%zu attempts with known queries only\n",
                     rep.attempts, rep.failed_matches, rep.gave_up,
                     static_cast<unsigned long long>(rep.expansions), rep.skipped_starts,
                     rep.known_only);
        std::vector<TermId> preds;
        for (const Query& q : records)
          for (const TriplePattern& tp : q.patterns)
            if (!q_is_var(tp.p)) preds.push_back(tp.p);
        std::sort(preds.begin(), preds.end());
        std::fprintf(stderr, "  %zu different predicates in the queries\n",
                     static_cast<size_t>(std::unique(preds.begin(), preds.end()) - preds.begin()));
      }
      if (!rep.bucket_fill.empty()) {
        std::fprintf(stderr, "  queries per range:");
        for (size_t n : rep.bucket_fill) std::fprintf(stderr, " %zu", n);
        std::fprintf(stderr, "; %zu counted in %.3f s, dropped: %zu range full, %zu outside\n",
                     rep.counted, rep.count_seconds, rep.dropped_full, rep.dropped_outside);
      }
      if (!opt.w.prune && opt.w.sampling != ComplexOptions::Sampling::Weighted && !rep.auto_pruned)
        return;
      if (rep.auto_pruned) std::fprintf(stderr, "  pruning switched on during the run\n");
      std::fprintf(stderr, "  %zu start nodes, %zu attempts (%zu without a match), "
                           "%.3g spanning-tree matches, weights %.3f s\n",
                   rep.start_candidates, rep.attempts, rep.failed_matches, rep.tree_matches,
                   rep.weights_seconds);
    };
    SubjectIndex fwd, rev;
    fwd.build(store, terms.bnode_base());
    rev.build(store, terms.bnode_base(), /*reverse=*/true);
    timer.lap("adj-index");

    if (opt.shape == "complex") {
      if (opt.template_str.empty())
        throw BgpError("--shape complex needs --template (see --help)");
      size_t n_patterns = parse_template(opt.template_str).triples.size();
      SampleReport rep;
      std::vector<Query> records =
          sample_template_workload(store, fwd, rev, terms, opt.template_str, opt.w, &rep);
      timer.lap("sample");

      if (opt.cardinality && !opt.w.card_buckets) {
        count_cardinalities(store, fwd, rev, records, {opt.card_budget, opt.threads});
        timer.lap("count");
        size_t timeouts = 0;
        for (const Query& q : records) timeouts += q.y < 0;
        if (timeouts)
          std::fprintf(stderr, "complex: %zu cardinalities hit the budget (y = -1)\n", timeouts);
      }

      std::string fn = opt.output_dir + "/" + dataset + "_complex_" + ts + "_" +
                       std::to_string(n_patterns) + ".json";
      if (opt.write_output) write_query_json(fn, records, terms, /*path_order=*/false);
      timer.lap("write");
      std::fprintf(stderr, "complex: %zu/%zu unique queries (%zu duplicates skipped, "
                           "%zu reshuffles)%s%s -> %s\n",
                   records.size(), opt.w.num, rep.duplicates_skipped, rep.reshuffles,
                   rep.stop_reason.empty() ? "" : " - stopped: ", rep.stop_reason.c_str(),
                   fn.c_str());
      print_match_stats(rep, records);
      return 0;
    }

    // path: a chain template (see sample_path_workload).
    for (unsigned size : opt.sizes) {
      SampleReport rep;
      std::vector<Query> records =
          sample_path_workload(store, fwd, rev, terms, size, opt.w, &rep);
      timer.lap("sample");

      if (opt.cardinality && !opt.w.card_buckets) {
        count_cardinalities(store, fwd, rev, records, {opt.card_budget, opt.threads});
        timer.lap("count");
        size_t timeouts = 0;
        for (const Query& q : records) timeouts += q.y < 0;
        if (timeouts)
          std::fprintf(stderr, "size %u: %zu cardinalities hit the budget (y = -1)\n", size,
                       timeouts);
      }

      std::string fn = opt.output_dir + "/" + dataset + "_path_" + ts + "_" +
                       std::to_string(size) + "_dfs.json";
      if (opt.write_output) write_query_json(fn, records, terms, /*path_order=*/true);
      timer.lap("write");
      std::fprintf(stderr, "size %u: %zu/%zu unique queries (%zu duplicates skipped, "
                           "%zu reshuffles)%s%s -> %s\n",
                   size, records.size(), opt.w.num, rep.duplicates_skipped, rep.reshuffles,
                   rep.stop_reason.empty() ? "" : " - stopped: ", rep.stop_reason.c_str(),
                   fn.c_str());
      print_match_stats(rep, records);
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "bgpgen: " << e.what() << "\n";
    return 2;
  }
}
