"""
Young-Laplace equation solver for pendant drops

Based on equations (1), (2), (9) from:
Kratz & Kierfeld (2020) - Pendant drop tensiometry: A machine learning approach
"""
import numpy as np
from scipy.integrate import solve_ivp


class PendantDropSolver:
    """Solve forward problem: generate droplet shape from parameters"""
    
    def __init__(self, Bo=0.3, pL_tilde=2.0):
        """
        Initialize solver with dimensionless parameters
        
        Args:
            Bo: Bond number (dimensionless density difference Δρ̃)
            pL_tilde: Dimensionless apex Laplace pressure p̃_L
        """
        self.Bo = Bo
        self.pL_tilde = pL_tilde
    
    def rhs(self, s, y):
        """
        Right-hand side of Young-Laplace ODE system
        
        Implements equations (1), (2), (9) from paper:
        - dr/ds = cos(ψ)
        - dz/ds = sin(ψ)  
        - dψ/ds = p̃_L - Δρ̃·z - sin(ψ)/r
        
        Args:
            s: Arc length parameter
            y: State vector [φ, r, z]
        
        Returns:
            [dφ/ds, dr/ds, dz/ds]
        """
        phi, r, z = y
        eps = 1e-12  # Avoid division by zero at apex
        
        # Equation (9): angle derivative
        dphi_ds = self.pL_tilde - self.Bo * z - np.sin(phi)/(r + eps)
        
        # Equation (1): radial position derivative  
        dr_ds = np.cos(phi)
        
        # Equation (2): vertical position derivative
        dz_ds = np.sin(phi)
        
        return [dphi_ds, dr_ds, dz_ds]
    
    def solve(self, s_max=10, n_points=500):
        """
        Solve ODE system to generate droplet shape
        
        Args:
            s_max: Maximum arc length to integrate
            n_points: Number of points along droplet profile
        
        Returns:
            dict with keys: 's', 'phi', 'r', 'z', 'Bo', 'pL_tilde', 'volume', 'Wo'
        """
        # Initial conditions at apex: φ=0, r≈0, z=0
        y0 = [0, 1e-6, 0]
        s_eval = np.linspace(0, s_max, n_points)
        
        # Solve ODE system
        sol = solve_ivp(
            self.rhs, 
            (0, s_max), 
            y0,
            t_eval=s_eval,
            method='RK45',
            dense_output=True
        )
        
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
        
        # Extract solution
        r_vals = sol.y[1]
        z_vals = sol.y[2]
        
        # Compute volume and Worthington number
        volume = self.compute_volume(r_vals, z_vals)
        Wo = self.Bo * volume / np.pi  # Equation (18) from paper
        
        return {
            's': sol.t,
            'phi': sol.y[0],
            'r': r_vals,
            'z': z_vals,
            'Bo': self.Bo,
            'pL_tilde': self.pL_tilde,
            'volume': volume,
            'Wo': Wo
        }
    
    def compute_volume(self, r_vals, z_vals):
        """
        Compute dimensionless volume
        
        Ṽ = π ∫ r²(dz/ds) ds
        
        Args:
            r_vals: Radial coordinates
            z_vals: Vertical coordinates
        
        Returns:
            Dimensionless volume
        """
        return np.pi * np.trapezoid(r_vals**2, z_vals)
    
    def compute_curvatures(self, phi_vals, r_vals, z_vals):
        """
        Compute principal curvatures along droplet
        
        Returns:
            dict with 'kappa_s' (meridional) and 'kappa_phi' (azimuthal)
        """
        # Meridional curvature: κ_s = dψ/ds
        ds = np.gradient(z_vals)
        kappa_s = np.gradient(phi_vals, ds)
        
        # Azimuthal curvature: κ_φ = sin(ψ)/r
        eps = 1e-12
        kappa_phi = np.sin(phi_vals) / (r_vals + eps)
        
        return {
            'kappa_s': kappa_s,
            'kappa_phi': kappa_phi,
            'kappa_total': kappa_s + kappa_phi
        }
