import cupy as cupy
import cvxpy as cp


def solve(
    p: int,
    triplets: cupy.ndarray,
    y_true: cupy.ndarray,
    gamma: float,
    lambda_: float,
    loss_type: str,
    constraint_type: str,
    solver: str,
    verbose: bool,
):
    diff_i_j = triplets[:, 0, :] - triplets[:, 1, :]
    diff_i_k = triplets[:, 0, :] - triplets[:, 2, :]

    K_hat = cp.Variable((p, p), PSD=True)
    M_t_K_s = cp.sum(cp.multiply(K_hat @ diff_i_j.T, diff_i_j.T), axis=0) - cp.sum(
        cp.multiply(K_hat @ diff_i_k.T, diff_i_k.T), axis=0
    )

    if loss_type == "logistic":
        risk = cp.sum(cp.logistic(-1 * cp.multiply(y_true, M_t_K_s))) / len(triplets)
    elif loss_type == "hinge":
        risk = cp.sum(cp.pos(1 - cp.multiply(y_true, M_t_K_s))) / len(triplets)
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}")

    if constraint_type == "nuc":
        constraints = [cp.norm(K_hat, p="nuc") <= lambda_]
    elif constraint_type == "fro":
        constraints = [cp.norm(K_hat, p="fro") <= lambda_]
    elif constraint_type == "l_12":
        constraints = [cp.mixed_norm(K_hat, p=2, q=1) <= lambda_]
    else:
        raise ValueError(f"Invalid constraint_type: {constraint_type}")
    constraints.append(cp.abs(M_t_K_s) <= gamma)

    problem = cp.Problem(cp.Minimize(risk), constraints=constraints)

    problem.solve(solver=solver, verbose=verbose)

    return problem, K_hat.value
