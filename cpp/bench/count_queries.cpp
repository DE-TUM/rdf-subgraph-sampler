// Exact COUNT(*) for queries given in a text file, with bgpgen's counter.
// Used to label and cross-check the workloads of other generators.
//
//   bgpgen_count --data graph.nt --queries q.txt --out counts.txt
//                [--card-budget N] [-j N]
//
// One query per line: triple patterns separated by " . ", each "s p o" with
// ?name for variables and <iri> for constants. Writes one count per line
// (-1 = budget exhausted, 0 also when a constant does not occur in the data).

#include "bgpgen/cardinality.hpp"
#include "bgpgen/graph.hpp"
#include "bgpgen/nt_parser.hpp"
#include "bgpgen/subject_index.hpp"
#include "bgpgen/term_table.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace bgpgen;

int main(int argc, char** argv) {
  std::string data, queries_path, out_path;
  uint64_t budget = 50'000'000;
  unsigned threads = 0;
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
    else if (a == "--queries") queries_path = val();
    else if (a == "--out") out_path = val();
    else if (a == "--card-budget") budget = std::stoull(val());
    else if (a == "-j") threads = static_cast<unsigned>(std::stoul(val()));
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (data.empty() || queries_path.empty() || out_path.empty()) {
    std::cerr << "usage: bgpgen_count --data graph.nt --queries q.txt --out counts.txt\n";
    return 2;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    auto since = [&] {
      double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      t0 = std::chrono::steady_clock::now();
      return s;
    };
    TermTable terms;
    std::vector<Triple> triples = load_nt_file(data, terms, 0, threads);
    std::vector<Triple>* sets[] = {&triples};
    terms.freeze(sets);
    TripleStore store;
    store.build(std::move(triples), threads);
    SubjectIndex fwd, rev;
    fwd.build(store, terms.bnode_base());
    rev.build(store, terms.bnode_base(), /*reverse=*/true);
    double load_s = since();

    std::vector<Query> queries;
    std::vector<char> known; // 0 = a constant is missing from the data
    std::ifstream in(queries_path);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      Query q;
      bool ok = true;
      std::unordered_map<std::string, uint32_t> vars;
      std::istringstream ss(line);
      std::string tok;
      std::vector<QTerm> buf;
      auto flush = [&] {
        if (buf.size() == 3) q.patterns.push_back({buf[0], buf[1], buf[2]});
        else if (!buf.empty()) throw BgpError("malformed query line: " + line);
        buf.clear();
      };
      while (ss >> tok) {
        if (tok == ".") {
          flush();
        } else if (tok[0] == '?') {
          auto it = vars.find(tok);
          if (it == vars.end()) {
            it = vars.emplace(tok, static_cast<uint32_t>(q.var_names.size())).first;
            q.var_names.push_back(tok);
          }
          buf.push_back(qvar(it->second));
        } else if (tok.size() >= 2 && tok.front() == '<' && tok.back() == '>') {
          TermId id = terms.find_iri(std::string_view(tok).substr(1, tok.size() - 2));
          if (id == kInvalidTerm) ok = false;
          buf.push_back(id == kInvalidTerm ? 0 : id);
        } else {
          throw BgpError("only ?variables and <iris> are supported: " + tok);
        }
      }
      flush();
      queries.push_back(std::move(q));
      known.push_back(ok);
    }
    double parse_s = since();

    count_cardinalities(store, fwd, rev, queries, {budget, threads});
    double count_s = since();

    std::ofstream out(out_path);
    size_t over = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
      int64_t y = known[i] ? queries[i].y : 0;
      over += y < 0;
      out << y << "\n";
    }
    std::fprintf(stderr, "%zu queries: load %.3f s, parse %.3f s, count %.3f s, %zu over budget\n",
                 queries.size(), load_s, parse_s, count_s, over);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "bgpgen_count: " << e.what() << "\n";
    return 2;
  }
}
