from .neural_nbv.neural_nbv_planner import NeuralNBVPlanner
from .baselines.random_planner import RandomPlanner
from .baselines.max_distance_planner import MaxDistancePlanner


def get_planner(cfg):
    planner_type = cfg["planner_type"]

    if planner_type == "neural_nbv":
        return NeuralNBVPlanner(cfg)
    elif planner_type == "random":
        return RandomPlanner(cfg)
    elif planner_type == "max_distance":
        return MaxDistancePlanner(cfg)
