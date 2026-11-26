"""Tests of the Python module. Run with: pytest python/tests"""

import gzip
import json
import os
import pathlib
import subprocess
import sysconfig

import pytest

import bgpgen

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOY = ROOT / "examples" / "toy.nt"
TRIANGLE = "?x B ?y, ?y B ?z, ?z B ?x"


@pytest.fixture(scope="module")
def graph():
    return bgpgen.Graph(TOY)


def test_graph_sizes(graph):
    assert (graph.triples, graph.predicates, graph.terms) == (49, 8, 30)


def test_gzipped_input(tmp_path):
    packed = tmp_path / "toy.nt.gz"
    packed.write_bytes(gzip.compress(TOY.read_bytes()))
    try:
        assert bgpgen.Graph(packed).triples == 49
    except bgpgen.BgpgenError as error:
        # A build without zlib cannot read .gz; the prebuilt wheels must.
        if "zlib" not in str(error) or os.environ.get("BGPGEN_REQUIRE_GZIP"):
            raise
        pytest.skip("built without zlib")


def test_triangle_queries_and_counts(graph):
    queries = graph.sample(TRIANGLE)
    assert sorted(q["y"] for q in queries) == [3, 6]
    q = next(q for q in queries if q["y"] == 6)
    assert set(q["x"]) == {"http://example.org/author", "http://example.org/worksFor",
                           "http://example.org/funded"}
    assert q["query"].startswith("SELECT * WHERE { ") and len(q["triples"]) == 3
    assert "diminishing returns" in graph.last_report["stop_reason"]


def test_same_seed_same_workload(graph):
    a = graph.sample_paths(2, num=8, seed=1)
    assert a == graph.sample_paths(2, num=8, seed=1)
    assert a != graph.sample_paths(2, num=8, seed=2)


def test_star_records(graph):
    stars = graph.sample_stars(2, num=5)
    assert len(stars) == 5
    for s in stars:
        assert s["y"] >= 1 and len(s["triples"]) == 2
        assert {"source_subject", "star_size", "predicates_used"} <= set(s)


def test_no_cardinality(graph):
    assert all(q["y"] == -1 for q in graph.sample_paths(2, num=5, cardinality=False))


def test_options(graph):
    queries = graph.sample_paths(2, num=6, card_buckets=2, card_max=100, per_seed="auto")
    assert all(1 <= q["y"] < 100 for q in queries)
    assert sum(graph.last_report["queries_per_range"]) == len(queries)
    assert bgpgen.DEFAULTS["num"] == 1000 and bgpgen.DEFAULTS["dedup"] == "symmetry"


def test_errors(graph, tmp_path):
    with pytest.raises(TypeError, match="no option 'nmu'"):
        graph.sample(TRIANGLE, nmu=5)
    with pytest.raises(TypeError, match="no option 'p_node'"):  # a template option
        graph.sample_paths(2, p_node=0.5)
    with pytest.raises(TypeError, match="wrong type"):
        graph.sample(TRIANGLE, num="many")
    with pytest.raises(ValueError):
        graph.sample(TRIANGLE, dedup="none")
    with pytest.raises(bgpgen.BgpgenError):
        graph.sample("?x B")
    with pytest.raises(bgpgen.BgpgenError):
        bgpgen.Graph(tmp_path / "missing.nt")


# The module and the binary have to give the same records for the same options.
CLI = pathlib.Path(sysconfig.get_path("scripts")) / "bgpgen"
CASES = [
    (["--shape", "star", "--size", "2", "--num", "6", "--seed", "5"],
     lambda g: g.sample_stars(2, num=6, seed=5)),
    (["--shape", "path", "--size", "3", "--num", "6", "--p-edge", "0.5", "--p-start-end", "0.9"],
     lambda g: g.sample_paths(3, num=6, p_edge=0.5, p_start_end=0.9)),
    (["--shape", "path", "--size", "2", "--num", "6", "--novelty", "1", "--skip-known",
      "--max-known", "1", "--per-seed", "auto", "--retire-seeds", "--seed-groups"],
     lambda g: g.sample_paths(2, num=6, novelty=1, skip_known=True, max_known=1, per_seed="auto",
                              retire_seeds=True, seed_groups=True)),
    (["--shape", "complex", "--template", "?a/B B ?b, ?b ?/B ?c", "--num", "6", "--p-node", "0.5",
      "--p-edge", "0.5", "--seed", "3"],
     lambda g: g.sample("?a/B B ?b, ?b ?/B ?c", num=6, p_node=0.5, p_edge=0.5, seed=3)),
]


@pytest.mark.skipif(not CLI.exists(), reason="the bgpgen binary is not installed next to Python")
@pytest.mark.parametrize("args, call", CASES)
def test_same_records_as_the_binary(graph, tmp_path, args, call):
    subprocess.run([CLI, "--data", TOY, "--output-dir", tmp_path, *args], check=True,
                   capture_output=True)
    (out,) = tmp_path.glob("*.json")
    assert call(graph) == json.loads(out.read_text())
