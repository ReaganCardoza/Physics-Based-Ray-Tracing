import numpy as np

def sample_ggx_scattering_angle(alpha, num_samples=1):
    """
    Sample scattering angles from a GGX distribution with roughness parameter alpha.
    
    Parameters:
    - alpha (float): Roughness parameter (alpha = roughness^2, typically > 0).
    - num_samples (int): Number of samples to generate.
    
    Returns:
    - theta (np.ndarray): Sampled scattering angles in degrees.
    """
    # Generate uniform random variables
    xi = np.random.uniform(0, 1, num_samples)
    
    # Sample cos(theta) using GGX sampling formula
    cos_theta = np.sqrt((1 - xi) / (1 + (alpha**2 - 1) * xi))
    
    # Convert to theta (in radians, then degrees)
    theta = np.arccos(cos_theta) * 180 / np.pi
    
    return theta

def ggx_pdf(theta, alpha):
    """
    Compute the GGX PDF for a given angle and roughness parameter.
    
    Parameters:
    - theta (float or np.ndarray): Angle(s) in degrees.
    - alpha (float): Roughness parameter.
    
    Returns:
    - pdf (float or np.ndarray): PDF value(s).
    """
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    denom = (alpha**2 - 1) * cos_theta**2 + 1
    D = alpha**2 / (np.pi * denom**2)  # GGX normal distribution
    pdf = D * sin_theta  # Include solid angle term
    return pdf / np.max(pdf)  # Normalize for plotting (approximate)

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Parameters
    alpha = 0.5
    num_samples = 10000
    
    # Sample scattering angles
    angles = sample_ggx_scattering_angle(alpha, num_samples)
    
    # Plot histogram of sampled angles
    plt.hist(angles, bins=50, density=True, alpha=0.7, label='Sampled')
    
    # Plot theoretical PDF for comparison
    theta_range = np.linspace(0, 90, 100)
    pdf = ggx_pdf(theta_range, alpha)
    plt.plot(theta_range, pdf, 'r-', label='GGX PDF')
    
    plt.xlabel('Scattering Angle (degrees)')
    plt.ylabel('Density')
    plt.title(f'GGX Scattering Angle Sampling (α={alpha})')
    plt.legend()
    plt.show()
    
    # Example: Print first few sampled angles
    print(f"First 5 sampled angles: {angles[:5]} degrees")

    # Example: Use sampled angle in ultrasound simulation
    theta = sample_ggx_scattering_angle(alpha=0.5, num_samples=1)[0]
    phi = np.random.uniform(0, 2 * np.pi)  # Uniform azimuthal angle
    direction = [np.sin(np.radians(theta)) * np.cos(phi),
                np.sin(np.radians(theta)) * np.sin(phi),
                np.cos(np.radians(theta))]  # 3D direction vector
    print(f"Scattered wave direction: {direction}")
    # Pass direction to ultrasound simulator (e.g., k-Wave)