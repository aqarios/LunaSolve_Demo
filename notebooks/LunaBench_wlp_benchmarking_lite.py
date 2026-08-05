# =============================================================================
# Benchmarking FlexQAOA on the Warehouse Location Problem (WLP) using LunaBench
# =============================================================================

from luna_bench import Benchmark, ModelSet
from luna_bench.features import OptSolFeature, VarNumberFeature
from luna_bench.metrics import FeasibilityRatio, Runtime
from luna_bench.plots import (
    AverageFeasibilityRatioPlot,
)
from luna_quantum import Model, algorithms
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
with open("wlp.bytes", "rb") as f:
    m = Model.deserialize(f.read())

model_set = ModelSet.create("wlp_example")
model_set.add(m)

bench = Benchmark.open("wlp_example_benchmark")
bench.set_modelset(model_set)

bench.add_feature(name="opt_sol", feature=OptSolFeature())
bench.add_feature(name="var_num", feature=VarNumberFeature())

bench.add_metric(name="runtime", metric=Runtime())
bench.add_metric(name="feasibility", metric=FeasibilityRatio())


bench.add_algorithm(algorithm=algorithms.FlexQAOA(reps=2), name="FlexQAOA_rep2")
bench.add_algorithm(algorithm=algorithms.FlexQAOA(reps=8), name="FlexQAOA_rep8")

bench.add_plot(name="avg_feasibility", plot=AverageFeasibilityRatioPlot())

bench.run()
