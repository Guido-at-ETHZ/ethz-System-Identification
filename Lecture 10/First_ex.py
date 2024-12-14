import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def exercise1_ridge_regression():
    # Generate synthetic data
    np.random.seed(0)
    N = 1000  # number of data points
    tau_max = 50  # true system order

    # Generate input and true impulse response
    u = np.random.randn(N)
    g_true = 0.8 ** np.arange(tau_max)

    # Generate output with noise
    y = np.zeros(N)
    for i in range(tau_max):
        y[i:] += g_true[i] * u[:N - i]
    y += 0.1 * np.random.randn(N)

    # Build regression matrix
    Phi = build_regression_matrix(u, tau_max, N)

    # Standard LS with different orders
    orders = [10, 20, 30, 40, 50]
    ls_estimates = np.zeros((tau_max, len(orders)))
    for i, order in enumerate(orders):
        Phi_n = Phi[:, :order]
        ls_estimates[:order, i] = np.linalg.solve(Phi_n.T @ Phi_n, Phi_n.T @ y)

    # Ridge regression with different gamma values
    gamma_vals = np.logspace(-3, 3, 20)
    ridge_estimates = np.zeros((tau_max, len(gamma_vals)))
    for i, gamma in enumerate(gamma_vals):
        ridge_estimates[:, i] = np.linalg.solve(
            Phi.T @ Phi + gamma * np.eye(tau_max),
            Phi.T @ y
        )

    # Cross validation for ridge regression
    K = 5  # number of folds
    cv_errors = np.zeros(len(gamma_vals))
    fold_size = N // K

    for i, gamma in enumerate(gamma_vals):
        cv_error = 0
        for k in range(K):
            # Split data
            val_idx = np.arange(k * fold_size, (k + 1) * fold_size)
            train_idx = np.setdiff1d(np.arange(N), val_idx)

            # Train model
            Phi_train = Phi[train_idx]
            y_train = y[train_idx]
            theta = np.linalg.solve(
                Phi_train.T @ Phi_train + gamma * np.eye(tau_max),
                Phi_train.T @ y_train
            )

            # Validate
            Phi_val = Phi[val_idx]
            y_val = y[val_idx]
            cv_error += np.mean((y_val - Phi_val @ theta) ** 2)

        cv_errors[i] = cv_error / K

    # Find best gamma
    best_idx = np.argmin(cv_errors)
    best_gamma = gamma_vals[best_idx]
    best_ridge_estimate = np.linalg.solve(
        Phi.T @ Phi + best_gamma * np.eye(tau_max),
        Phi.T @ y
    )

    # Plotting results
    plot_results(g_true, ls_estimates, ridge_estimates, cv_errors,
                 gamma_vals, best_gamma, best_ridge_estimate, orders)


def exercise2_tc_kernel():
    # Generate synthetic data (same as Exercise 1)
    np.random.seed(0)
    N = 1000
    tau_max = 50

    u = np.random.randn(N)
    g_true = 0.8 ** np.arange(tau_max)

    y = np.zeros(N)
    for i in range(tau_max):
        y[i:] += g_true[i] * u[:N - i]
    y += 0.1 * np.random.randn(N)

    # Build regression matrix
    Phi = build_regression_matrix(u, tau_max, N)

    # TC-kernel parameters
    gamma_vals = np.logspace(-3, 3, 10)
    alpha_vals = np.linspace(0.1, 0.9, 9)

    # Cross validation for TC-kernel
    K = 5
    cv_errors = np.zeros((len(gamma_vals), len(alpha_vals)))
    fold_size = N // K

    for i, gamma in enumerate(gamma_vals):
        for j, alpha in enumerate(alpha_vals):
            # Build TC-kernel
            P = build_tc_kernel(tau_max, alpha)
            P_inv = np.linalg.inv(P)

            cv_error = 0
            for k in range(K):
                # Split data
                val_idx = np.arange(k * fold_size, (k + 1) * fold_size)
                train_idx = np.setdiff1d(np.arange(N), val_idx)

                # Train model
                Phi_train = Phi[train_idx]
                y_train = y[train_idx]
                theta = np.linalg.solve(
                    Phi_train.T @ Phi_train + gamma * P_inv,
                    Phi_train.T @ y_train
                )

                # Validate
                Phi_val = Phi[val_idx]
                y_val = y[val_idx]
                cv_error += np.mean((y_val - Phi_val @ theta) ** 2)

            cv_errors[i, j] = cv_error / K

    # Find best parameters
    min_idx = np.unravel_index(np.argmin(cv_errors), cv_errors.shape)
    best_gamma = gamma_vals[min_idx[0]]
    best_alpha = alpha_vals[min_idx[1]]

    # Compute best estimates
    P_best = build_tc_kernel(tau_max, best_alpha)
    tc_estimate = np.linalg.solve(
        Phi.T @ Phi + best_gamma * np.linalg.inv(P_best),
        Phi.T @ y
    )

    ridge_estimate = np.linalg.solve(
        Phi.T @ Phi + best_gamma * np.eye(tau_max),
        Phi.T @ y
    )

    # Plot results
    plot_tc_results(g_true, tc_estimate, ridge_estimate, cv_errors,
                    gamma_vals, alpha_vals, P_best)


def build_regression_matrix(u, tau_max, N):
    """Build regression matrix for FIR model"""
    Phi = np.zeros((N, tau_max))
    for i in range(tau_max):
        Phi[i:, i] = u[:N - i]
    return Phi


def build_tc_kernel(tau_max, alpha):
    """Build TC-kernel matrix"""
    i, j = np.meshgrid(np.arange(tau_max), np.arange(tau_max))
    return alpha ** np.maximum(i, j)


def plot_results(g_true, ls_estimates, ridge_estimates, cv_errors,
                 gamma_vals, best_gamma, best_ridge_estimate, orders):
    """Plot results for Exercise 1"""
    plt.figure(figsize=(15, 10))

    # Plot LS estimates
    plt.subplot(221)
    plt.plot(g_true, 'k-', linewidth=2, label='True')
    for i, order in enumerate(orders):
        plt.plot(ls_estimates[:, i], '--', label=f'Order {order}')
    plt.title('LS Estimates with Different Orders')
    plt.legend()

    # Plot ridge estimates
    plt.subplot(222)
    plt.plot(g_true, 'k-', linewidth=2, label='True')
    for i in [0, 4, 9, 14, 19]:
        plt.plot(ridge_estimates[:, i], '--',
                 label=f'γ = {gamma_vals[i]:.1e}')
    plt.title('Ridge Estimates with Different γ')
    plt.legend()

    # Plot CV error
    plt.subplot(223)
    plt.semilogx(gamma_vals, cv_errors)
    plt.plot(best_gamma, cv_errors[np.argmin(cv_errors)], 'ro')
    plt.title('Cross-validation Error vs γ')
    plt.xlabel('γ')
    plt.ylabel('CV Error')

    # Plot best estimate
    plt.subplot(224)
    plt.plot(g_true, 'k-', linewidth=2, label='True')
    plt.plot(best_ridge_estimate, 'r--', linewidth=2, label='Ridge')
    plt.title(f'Best Ridge Estimate (γ = {best_gamma:.1e})')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_tc_results(g_true, tc_estimate, ridge_estimate, cv_errors,
                    gamma_vals, alpha_vals, P_best):
    """Plot results for Exercise 2"""
    plt.figure(figsize=(15, 10))

    # Plot CV error heatmap
    plt.subplot(221)
    plt.imshow(cv_errors, aspect='auto',
               extent=[alpha_vals[0], alpha_vals[-1],
                       np.log10(gamma_vals[0]), np.log10(gamma_vals[-1])])
    plt.colorbar()
    plt.xlabel('α')
    plt.ylabel('log₁₀(γ)')
    plt.title('Cross-validation Error')

    # Plot estimates comparison
    plt.subplot(222)
    plt.plot(g_true, 'k-', linewidth=2, label='True')
    plt.plot(tc_estimate, 'r--', linewidth=2, label='TC-kernel')
    plt.plot(ridge_estimate, 'b:', linewidth=2, label='Ridge')
    plt.title('Comparison of Estimates')
    plt.legend()

    # Plot kernel matrix
    plt.subplot(223)
    plt.imshow(P_best)
    plt.colorbar()
    plt.title('TC-kernel Matrix Structure')
    plt.xlabel('i')
    plt.ylabel('j')

    # Plot absolute errors
    plt.subplot(224)
    plt.semilogy(np.abs(g_true - tc_estimate), 'r-', label='TC-kernel')
    plt.semilogy(np.abs(g_true - ridge_estimate), 'b-', label='Ridge')
    plt.title('Absolute Errors')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('|Error|')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running Exercise 1: Ridge Regression")
    exercise1_ridge_regression()

    print("\nRunning Exercise 2: TC-Kernel")
    exercise2_tc_kernel()
