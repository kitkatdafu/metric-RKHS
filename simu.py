import hydra
import warnings
import numpy as cp
import cvxpy as cvp
from kernel import RKHSKernel
from solver import solve
from utils import calc_accuracy, triplets_distance, get_triplets

warnings.simplefilter(action="ignore", category=FutureWarning)

@hydra.main(version_base=None, config_path=".", config_name="config_tatli")
def main():
    pass


if __name__ == "__main__":
    main()