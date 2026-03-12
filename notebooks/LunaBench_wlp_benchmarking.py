# =============================================================================
# Benchmarking FlexQAOA on the Warehouse Location Problem (WLP) using LunaBench
# =============================================================================
#
# This script showcases how to benchmark different configurations of the FlexQAOA
# quantum optimization algorithm on a Warehouse Location Problem (WLP) instance.
# See the WLP_with_FlexQAOA notebook for more details on the problem setup.
#
# We use our benchmarking framework LunaBench to systematically compare two
# FlexQAOA variants (2 vs. 8 QAOA repetitions) and evaluate them on feasibility.
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Imports and setup
# -----------------------------------------------------------------------------


import multiprocessing
import os
from dotenv import load_dotenv

load_dotenv()

from luna_quantum import algorithms
from luna_quantum import Model

from luna_bench import Benchmark, ModelSet
from luna_bench.configs import config
from luna_bench.features import VarNumberFeature, OptSolFeature
from luna_bench.metrics import FeasibilityRatio, Runtime
from luna_bench.plots import (
    AverageFeasibilityRatioPlot,
)

# Known macOS issue with multiprocessing — must be set before spawning processes
multiprocessing.set_start_method("fork")


# LunaBench persists results in a local SQLite database. To start fresh,
# we delete the existing database file (if any) before running.
def delete_tables() -> None:
    if config.DB_CONNECTION_STRING != ":memory:":
        if os.path.isfile(config.DB_CONNECTION_STRING):
            os.remove(config.DB_CONNECTION_STRING)


delete_tables()

# -----------------------------------------------------------------------------
# 2. Load the WLP model from a serialized file
# -----------------------------------------------------------------------------
# The file 'wlp.bytes' contains a pre-built optimization model for the
# Warehouse Location Problem.
with open("wlp.bytes", "rb") as f:
    m = Model.deserialize(f.read())

# -----------------------------------------------------------------------------
# 3. Create a ModelSet and register the model
# -----------------------------------------------------------------------------
# A ModelSet groups one or more optimization models together for benchmarking.
# Here we have a single WLP instance, but LunaBench supports benchmarking
# with multiple problems. Just add more models to the set using the 'add' method.
model_set = ModelSet.create("wlp_example")
model_set.add(m)

# -----------------------------------------------------------------------------
# 4. Set up the Benchmark
# -----------------------------------------------------------------------------
# Opening a Benchmark creates a named benchmarking session that tracks all
# configurations, runs, and results.
bench = Benchmark.open("wlp_example_benchmark")
bench.set_modelset(model_set)

# -----------------------------------------------------------------------------
# 5. Define features to extract from each model
# -----------------------------------------------------------------------------
# Features describe properties of the problem instance itself (not the solver).
# We have selcted two features, however, LunaBench supports more predefined metrics
# and any custom feature. See the LunaBench docs for more details.
#
# - OptSolFeature:   Extracts the known optimal solution value,
#                    useful for computing optimality gaps.
# - VarNumberFeature: Counts the number of decision variables in the model,
#                     which indicates problem size / complexity.

bench.add_feature(name="opt_sol", feature=OptSolFeature())
bench.add_feature(name="var_num", feature=VarNumberFeature())

# -----------------------------------------------------------------------------
# 6. Define metrics to measure for each algorithm run
# -----------------------------------------------------------------------------
# Metrics capture solver performance on each problem instance. As for features,
# LunaBench has more predefined metrics available and users can easily implement
# their own.
#
# - Runtime:          Wall-clock time the algorithm takes to find a solution.
# - FeasibilityRatio: Fraction of returned solutions that satisfy all
#                     constraints.

bench.add_metric(name="runtime", metric=Runtime())
bench.add_metric(name="feasibility", metric=FeasibilityRatio())

# -----------------------------------------------------------------------------
# 7. Register algorithms to benchmark
# -----------------------------------------------------------------------------
# We compare two FlexQAOA configurations that differ in the number of QAOA
# repetitions (layers).
#
# - FlexQAOA_rep2: 2 QAOA layers — faster but may produce lower-quality solutions
# - FlexQAOA_rep8: 8 QAOA layers — slower but expected to find better solutions

bench.add_algorithm(algorithm=algorithms.FlexQAOA(reps=2), name="FlexQAOA_rep2")
bench.add_algorithm(algorithm=algorithms.FlexQAOA(reps=8), name="FlexQAOA_rep8")

# -----------------------------------------------------------------------------
# 8. Configure plots for result visualization
# -----------------------------------------------------------------------------
# AverageFeasibilityRatioPlot shows a bar chart comparing the average
# feasibility ratio across algorithms — this tells us which FlexQAOA
# configuration more reliably produces valid (constraint-satisfying) solutions.

bench.add_plot(name="avg_feasibility", plot=AverageFeasibilityRatioPlot())

# -----------------------------------------------------------------------------
# 9. Run the benchmark
# -----------------------------------------------------------------------------
# This executes all registered algorithms on all models in the ModelSet,
# computes the specified metrics and features, and generates the plots.
# Results are persisted to the benchmark database for later analysis.
bench.run()
