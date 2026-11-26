// Python module _bgpgen: load a graph once, sample workloads from it.
// The records are the ones the bgpgen binary writes as JSON, as dicts.

#include "bgpgen/cardinality.hpp"
#include "bgpgen/gz_reader.hpp"
#include "bgpgen/nt_parser.hpp"
#include "bgpgen/workload.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace bgpgen;

namespace {

// --- Options ------------------------------------------------------------------

/// Everything a sampling call can set.
struct Run {
  WorkloadOptions w;
  bool cardinality = true;
  uint64_t card_budget = 50'000'000;
};

enum Kind : unsigned { Star = 1, Path = 2, Template = 4, Search = Path | Template, All = 7 };

struct OptSpec {
  const char* name;
  unsigned kinds; // the sampling calls that accept the option
  void (*set)(Run&, nb::handle);
  nb::object (*get)(const Run&);
};

#define BGPGEN_OPT(name, kinds, field, type)                                             \
  OptSpec {                                                                              \
    name, kinds, [](Run& r, nb::handle v) { r.field = nb::cast<type>(v); },              \
        [](const Run& r) -> nb::object { return nb::cast(static_cast<type>(r.field)); }  \
  }

const OptSpec kOptions[] = {
    BGPGEN_OPT("num", All, w.num, size_t),
    BGPGEN_OPT("seed", All, w.seed, uint64_t),
    {"shared_seed", Star | Path,
     [](Run& r, nb::handle v) { r.w.independent_sizes = !nb::cast<bool>(v); },
     [](const Run& r) -> nb::object { return nb::cast(!r.w.independent_sizes); }},
    {"dedup", All,
     [](Run& r, nb::handle v) {
       std::string s = nb::cast<std::string>(v);
       if (s == "symmetry") r.w.dedup = DedupMethod::Symmetry;
       else if (s == "wl") r.w.dedup = DedupMethod::Wl;
       else if (s == "hash") r.w.dedup = DedupMethod::Hash;
       else throw nb::value_error("dedup must be 'symmetry', 'wl' or 'hash'");
     },
     [](const Run& r) -> nb::object {
       return nb::str(r.w.dedup == DedupMethod::Wl     ? "wl"
                      : r.w.dedup == DedupMethod::Hash ? "hash"
                                                       : "symmetry");
     }},
    BGPGEN_OPT("cardinality", All, cardinality, bool),
    BGPGEN_OPT("card_budget", All, card_budget, uint64_t),

    BGPGEN_OPT("p_predicate", Star, w.p_predicate, double),
    BGPGEN_OPT("min_objects", Star, w.min_objects, int),
    {"max_objects", Star,
     [](Run& r, nb::handle v) { r.w.max_objects = v.is_none() ? -1 : nb::cast<int>(v); },
     [](const Run& r) -> nb::object {
       return r.w.max_objects < 0 ? nb::object(nb::none()) : nb::cast(r.w.max_objects);
     }},

    BGPGEN_OPT("p_edge", Search, w.p_edge, double),
    BGPGEN_OPT("p_node", Template, w.p_node, double),
    BGPGEN_OPT("p_start_end", Path, w.p_start_end, double),
    BGPGEN_OPT("early_checks", Template, w.early_checks, bool),
    BGPGEN_OPT("prune", Search, w.prune, bool),
    BGPGEN_OPT("auto_prune", Search, w.auto_prune, bool),
    BGPGEN_OPT("attempt_budget", Search, w.attempt_budget, uint64_t),
    BGPGEN_OPT("degree_check", Search, w.degree_check, bool),
    BGPGEN_OPT("pred_first", Search, w.pred_first, bool),
    {"novelty", Search,
     [](Run& r, nb::handle v) {
       r.w.novelty = nb::cast<double>(v);
       r.w.pred_first = true; // as --novelty does
     },
     [](const Run& r) -> nb::object { return nb::cast(r.w.novelty); }},
    BGPGEN_OPT("skip_known", Search, w.skip_known, bool),
    {"per_seed", Search,
     [](Run& r, nb::handle v) {
       if (nb::isinstance<nb::str>(v)) {
         if (nb::cast<std::string>(v) != "auto") throw nb::value_error("per_seed must be a number or 'auto'");
         r.w.per_seed = 0;
       } else {
         r.w.per_seed = nb::cast<size_t>(v);
       }
     },
     [](const Run& r) -> nb::object {
       return r.w.per_seed == 0 ? nb::object(nb::str("auto")) : nb::cast(r.w.per_seed);
     }},
    BGPGEN_OPT("max_known", Search, w.max_known, size_t),
    BGPGEN_OPT("retire_seeds", Search, w.retire_seeds, bool),
    BGPGEN_OPT("seed_groups", Search, w.seed_groups, bool),
    BGPGEN_OPT("card_buckets", Search, w.card_buckets, unsigned),
    BGPGEN_OPT("card_max", Search, w.card_max, double),
    BGPGEN_OPT("card_steer", Search, w.card_steer, bool),
    BGPGEN_OPT("sample_threads", Search, w.sample_threads, unsigned),
    {"sampling", Search,
     [](Run& r, nb::handle v) {
       std::string s = nb::cast<std::string>(v);
       if (s == "dfs") r.w.sampling = ComplexOptions::Sampling::Dfs;
       else if (s == "weighted") r.w.sampling = ComplexOptions::Sampling::Weighted;
       else throw nb::value_error("sampling must be 'dfs' or 'weighted'");
     },
     [](const Run& r) -> nb::object {
       return nb::str(r.w.sampling == ComplexOptions::Sampling::Weighted ? "weighted" : "dfs");
     }},
    BGPGEN_OPT("root_alpha", Search, w.root_alpha, double),
};

#undef BGPGEN_OPT

/// The options of one sampling call. Unknown names and names that the call
/// does not use are errors, so a typo cannot pass unnoticed.
Run parse_options(const nb::kwargs& kwargs, Kind kind, const char* function) {
  Run run;
  for (auto [key, value] : kwargs) {
    std::string name = nb::cast<std::string>(key);
    const OptSpec* spec = nullptr;
    for (const OptSpec& o : kOptions)
      if (name == o.name && (o.kinds & kind)) spec = &o;
    if (!spec) {
      std::string msg = std::string(function) + "() has no option '" + name + "'. Options:";
      for (const OptSpec& o : kOptions)
        if (o.kinds & kind) msg += std::string(" ") + o.name;
      throw nb::type_error(msg.c_str());
    }
    try {
      spec->set(run, value);
    } catch (const nb::cast_error&) {
      throw nb::type_error(("option '" + name + "' has the wrong type").c_str());
    }
  }
  return run;
}

nb::dict default_options() {
  Run run;
  nb::dict d;
  for (const OptSpec& o : kOptions) d[o.name] = o.get(run);
  return d;
}

// --- Records ------------------------------------------------------------------

nb::str py_str(std::string_view s) { return nb::str(s.data(), s.size()); }

void add_query_and_triples(nb::dict& d, const Query& q, const TermTable& terms) {
  d["query"] = py_str(render_sparql(q, terms));
  nb::list triples;
  for (const TriplePattern& tp : q.patterns) {
    nb::list t;
    t.append(py_str(q_token(q, terms, tp.s)));
    t.append(py_str(q_token(q, terms, tp.p)));
    t.append(py_str(q_token(q, terms, tp.o)));
    triples.append(t);
  }
  d["triples"] = triples;
}

void add_x_and_y(nb::dict& d, const Query& q, const TermTable& terms) {
  nb::list x;
  for (TermId t : q.x) x.append(py_str(terms.lexical(t)));
  d["x"] = x;
  d["y"] = q.y;
}

/// Same keys, in the same order, as the JSON files of the binary.
nb::dict query_record(const Query& q, const TermTable& terms, bool path_order) {
  nb::dict d;
  if (path_order) {
    add_x_and_y(d, q, terms);
    add_query_and_triples(d, q, terms);
  } else {
    add_query_and_triples(d, q, terms);
    add_x_and_y(d, q, terms);
  }
  return d;
}

nb::dict star_record(const StarRecord& r, const TermTable& terms) {
  nb::dict d = query_record(r.query, terms, /*path_order=*/false);
  d["source_subject"] = py_str(terms.lexical(r.meta.source_subject));
  d["star_size"] = r.meta.star_size;
  char hash[17];
  std::snprintf(hash, sizeof hash, "%016llx", static_cast<unsigned long long>(r.meta.query_hash));
  d["query_hash"] = nb::str(hash);
  nb::list preds, objects, pattern;
  for (TermId t : r.meta.predicates_used) preds.append(py_str(terms.lexical(t)));
  for (TermId t : r.meta.instantiated_objects) objects.append(py_str(term_token(terms, t)));
  for (uint32_t i : r.meta.object_instantiation_pattern) pattern.append(i);
  d["predicates_used"] = preds;
  d["instantiated_objects"] = objects;
  d["object_instantiation_pattern"] = pattern;
  d["n_objects_instantiated"] = r.meta.n_objects_instantiated;
  return d;
}

// --- Graph --------------------------------------------------------------------

struct PyGraph {
  TermTable terms;
  TripleStore store;
  SubjectIndex fwd, rev;
  unsigned threads;
  SampleReport report; // of the last sampling call
  size_t uncounted = 0;

  PyGraph(const std::filesystem::path& path, unsigned threads_) : threads(threads_) {
    nb::gil_scoped_release release;
    std::string file = path.string();
    std::vector<Triple> triples;
    if (is_gz_path(file)) {
      std::vector<char> buf = read_gz_file(file);
      triples = parse_nt_buffer(buf.data(), buf.size(), terms, 0, threads);
    } else {
      triples = load_nt_file(file, terms, 0, threads);
    }
    std::vector<Triple>* sets[] = {&triples};
    terms.freeze(sets);
    store.build(std::move(triples), threads);
    fwd.build(store, terms.bnode_base());
    rev.build(store, terms.bnode_base(), /*reverse=*/true);
  }

  nb::list sample_template(const std::string& shape, const nb::kwargs& kwargs) {
    Run run = parse_options(kwargs, Template, "sample");
    return finish(run, false, [&] {
      return sample_template_workload(store, fwd, rev, terms, shape, run.w, &report);
    });
  }

  nb::list sample_paths(unsigned size, const nb::kwargs& kwargs) {
    Run run = parse_options(kwargs, Path, "sample_paths");
    return finish(run, true, [&] {
      return sample_path_workload(store, fwd, rev, terms, size, run.w, &report);
    });
  }

  nb::list sample_stars(unsigned size, const nb::kwargs& kwargs) {
    Run run = parse_options(kwargs, Star, "sample_stars");
    std::vector<StarRecord> records;
    {
      nb::gil_scoped_release release;
      report = SampleReport{};
      records = sample_star_workload(store, fwd, terms, size, run.w, &report);
      if (run.cardinality) {
        std::vector<Query*> qs;
        qs.reserve(records.size());
        for (StarRecord& r : records) qs.push_back(&r.query);
        count_cardinalities(store, fwd, rev, qs, {run.card_budget, threads});
      }
      uncounted = 0;
      for (const StarRecord& r : records) uncounted += r.query.y < 0;
    }
    nb::list out;
    for (const StarRecord& r : records) out.append(star_record(r, terms));
    return out;
  }

  /// Samples (without the GIL), counts unless the sampling already did, and
  /// turns the queries into records.
  template <class Sample>
  nb::list finish(const Run& run, bool path_order, Sample&& sample) {
    std::vector<Query> queries;
    {
      nb::gil_scoped_release release;
      report = SampleReport{};
      queries = sample();
      if (run.cardinality && !run.w.card_buckets)
        count_cardinalities(store, fwd, rev, queries, {run.card_budget, threads});
      uncounted = 0;
      for (const Query& q : queries) uncounted += q.y < 0;
    }
    nb::list out;
    for (const Query& q : queries) out.append(query_record(q, terms, path_order));
    return out;
  }

  nb::dict last_report() const {
    nb::dict d;
    d["stop_reason"] = report.stop_reason;
    d["duplicates_skipped"] = report.duplicates_skipped;
    d["reshuffles"] = report.reshuffles;
    d["attempts"] = report.attempts;
    d["failed_matches"] = report.failed_matches;
    d["uncounted"] = uncounted;
    nb::list fill;
    for (size_t n : report.bucket_fill) fill.append(n);
    d["queries_per_range"] = fill;
    return d;
  }
};

} // namespace

NB_MODULE(_bgpgen, m) {
  m.doc() = "Samples SPARQL basic graph pattern queries with exact cardinalities.";
  nb::exception<BgpError>(m, "BgpgenError", PyExc_RuntimeError);
  m.attr("DEFAULTS") = default_options();

  nb::class_<PyGraph>(m, "Graph")
      .def(nb::init<const std::filesystem::path&, unsigned>(), "path"_a, "threads"_a = 0,
           "Loads an N-Triples file (.nt or .nt.gz) into memory. threads = 0 uses all cores.")
      .def_prop_ro("triples", [](const PyGraph& g) { return g.store.triple_count(); },
                   "Number of distinct triples.")
      .def_prop_ro("predicates", [](const PyGraph& g) { return g.store.predicate_count(); },
                   "Number of distinct predicates.")
      .def_prop_ro("terms", [](const PyGraph& g) { return g.terms.size(); },
                   "Number of distinct terms.")
      .def("sample", &PyGraph::sample_template, "template"_a, "options"_a,
           "Queries of the shape of a template such as '?x B ?y, ?y B ?z, ?z B ?x'.")
      .def("sample_paths", &PyGraph::sample_paths, "size"_a, "options"_a,
           "Path queries with `size` triple patterns.")
      .def("sample_stars", &PyGraph::sample_stars, "size"_a, "options"_a,
           "Star queries with `size` triple patterns.")
      .def_prop_ro("last_report", &PyGraph::last_report,
                   "What the last sampling call did: why it stopped, duplicates, attempts.");
}
