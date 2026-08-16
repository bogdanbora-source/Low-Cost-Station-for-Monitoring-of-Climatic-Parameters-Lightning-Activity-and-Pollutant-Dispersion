import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def concentration_OML(Q, u, He, Hr, x, y, sigma_y, sigma_z):
    """Compute concentration based on the OML multi-Gaussian model."""
    term1 = Q / (u * np.maximum(sigma_z, 1e-6) * np.sqrt(2 * np.pi))  # Avoid division by zero
    term2 = np.exp(-y**2 / (2 * np.maximum(sigma_y, 1e-6)**2))
    term3 = np.exp(-(Hr - He)**2 / (2 * np.maximum(sigma_z, 1e-6)**2)) + np.exp(-(Hr + He)**2 / (2 * np.maximum(sigma_z, 1e-6)**2))
    
    return term1 * term2 * term3

def generate_dispersion_plot(Q, u, He):
    # Define the zoomed-in range for highest concentration area
    x_zoom_vals = np.linspace(0, 2000, 150)
    y_zoom_vals = np.linspace(-200, 200, 100)

    # Create mesh grid
    X_zoom, Y_zoom = np.meshgrid(x_zoom_vals, y_zoom_vals)

    # Adjust dispersion coefficients for better spread
    sigma_y_zoom = np.maximum(0.6 * (X_zoom**0.6), 1e-6)
    sigma_z_zoom = np.maximum(0.5 * (X_zoom**0.7), 1e-6)  # Avoid zero values

    # Compute concentration field with improved scaling
    C_zoom = concentration_OML(Q, u, He, 0, X_zoom, Y_zoom, sigma_y_zoom, sigma_z_zoom)
    C_zoom = C_zoom / np.max(C_zoom) * Q * (5 / u) * 10  # Adjust based on Q and u, scale x10
    C_zoom[C_zoom < 1] = 1  # Ensure minimum value is positive
    
    # Improve contour levels to reflect variation properly
    contour_levels = np.linspace(np.min(C_zoom), np.max(C_zoom), 10)
    
    # Define a more visually friendly color map for better contrast
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("presentation", ["#b0d2ff", "#6fa3d1", "#f8c471", "#f39c73", "#e57373", "#c62828"], N=256)

    # Create a visually structured contour plot
    plt.figure(figsize=(10, 5))
    contour = plt.contourf(X_zoom, Y_zoom, C_zoom, levels=contour_levels, cmap=custom_cmap)
    cbar = plt.colorbar(contour)
    cbar.set_label("Concentration (µg/m³)")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.text(50, 180, "Stability Class A", fontsize=12, fontweight='bold', color='black')
    plt.grid(True, linestyle='-', linewidth=0.5, color='black', alpha=0.7)
    plt.show()

# Request user input for parameters
Q_input = float(input("Enter Emission Rate (Q) [1-50]: "))
u_input = float(input("Enter Wind Speed (m/s) [0.5-10]: "))
He_input = float(input("Enter Source Height (m) [10-100]: "))

# Generate the dispersion plot based on user inputs
generate_dispersion_plot(Q_input, u_input, He_input)
