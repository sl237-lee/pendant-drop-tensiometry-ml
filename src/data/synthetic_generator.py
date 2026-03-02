"""
Generate synthetic training dataset
"""
import numpy as np
import pickle
from tqdm import tqdm
from ..physics.young_laplace import PendantDropSolver

class SyntheticDataGenerator:
    """Generate large datasets of pendant drop shapes"""
    
    def __init__(self, output_dir='data/synthetic'):
        self.output_dir = output_dir
    
    def generate_dataset(self, n_samples=10000, shape_class=2):
        """Generate training dataset
        
        Args:
            n_samples: Number of shapes to generate
            shape_class: 2 or 3 (Ω classification from paper)
        
        Returns:
            List of shape dictionaries
        """
        dataset = []
        
        for i in tqdm(range(n_samples), desc="Generating shapes"):
            # Sample parameters from appropriate region
            if shape_class == 2:
                Bo = np.random.uniform(0.1, 3.0)
                pL = np.random.uniform(1.5, 4.5)
            
            try:
                solver = PendantDropSolver(Bo=Bo, pL_tilde=pL)
                shape_data = solver.solve()
                
                # Add volume and Worthington number
                V = solver.compute_volume(shape_data['r'], shape_data['z'])
                shape_data['volume'] = V
                shape_data['Wo'] = Bo * V / np.pi
                
                dataset.append(shape_data)
                
            except Exception as e:
                print(f"Failed for Bo={Bo}, pL={pL}: {e}")
                continue
        
        return dataset
    
    def save_dataset(self, dataset, filename='training_data.pkl'):
        """Save to disk"""
        filepath = f"{self.output_dir}/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved {len(dataset)} shapes to {filepath}")