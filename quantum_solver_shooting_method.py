import numpy as np
import matplotlib.pyplot as plt

def quantum_system(X, xi, energy, potential):

    psi = X[0]
    dpsi = X[1]

    V = potential(xi)

    return np.array([dpsi, (V-energy)*psi])


def residual(potential, xi_initial, xi_final, energy, h):

    xi_values = np.arange(xi_initial, xi_final+h, h)
    N = len(xi_values)

    X = np.zeros((N,2))
    X[0] = [0,1]

    for i in range(N-1):

        k1 = h*quantum_system(X[i], xi_values[i], energy, potential)
        k2 = h*quantum_system(X[i]+k1/2, xi_values[i]+h/2, energy, potential)
        k3 = h*quantum_system(X[i]+k2/2, xi_values[i]+h/2, energy, potential)
        k4 = h*quantum_system(X[i]+k3, xi_values[i]+h, energy, potential)

        X[i+1] = X[i]+(k1+2*k2+2*k3+k4)/6

    R = X[-1,0]

    return X, xi_values, R


def quantum_shooting(potential, xi_initial, xi_final,
                     energy_low, energy_high, h,
                     energy_tol, residual_tol, max_iter):

    X, xi_values, R_low = residual(
        potential, xi_initial, xi_final, energy_low, h)

    X, xi_values, R_high = residual(
        potential, xi_initial, xi_final, energy_high, h)

    if R_low*R_high > 0:
        raise ValueError("No sign change in energy interval")

    for i in range(max_iter):

        energy_mid = (energy_low+energy_high)/2

        X, xi_values, R_mid = residual(
            potential, xi_initial, xi_final, energy_mid, h)

        if abs(R_mid) < residual_tol:
            return energy_mid

        if abs(energy_high-energy_low) < energy_tol:
            return energy_mid

        if R_low*R_mid < 0:
            energy_high = energy_mid
            R_high = R_mid

        else:
            energy_low = energy_mid
            R_low = R_mid

    return (energy_low+energy_high)/2


def wavefunction(potential, xi_initial, xi_final, energy, h):

    X, xi_values, R = residual(
        potential, xi_initial, xi_final, energy, h)

    psi = X[:,0]

    I = np.trapezoid(psi**2, xi_values)
    psi_normalized = psi/np.sqrt(I)

    return xi_values, psi, psi_normalized


def plot_wavefunction(psi, psi_normalized, energy,
                      xi_values, potential_name):

    I1 = np.trapezoid(psi**2, xi_values)
    I2 = np.trapezoid(psi_normalized**2, xi_values)

    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    ax[0].plot(xi_values, psi, label=f"Integral = {I1:.6f}")
    ax[0].set_title("Non-normalized wavefunction")
    ax[0].set_xlabel(r"$\xi$")
    ax[0].set_ylabel(r"$\psi(\xi)$")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(xi_values, psi_normalized,
               label=f"Integral = {I2:.6f}")
    ax[1].set_title("Normalized wavefunction")
    ax[1].set_xlabel(r"$\xi$")
    ax[1].set_ylabel(r"$\psi(\xi)$")
    ax[1].legend()
    ax[1].grid()

    plt.suptitle(f"{potential_name} (Energy = {energy:.6f})")
    plt.tight_layout()
    plt.show()