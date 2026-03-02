"""
Standalone script to generate training data

Usage:
    python scripts/generate_dataset.py --n_samples 50000 --shape_class 2
"""
import argparse
import sys
sys.path.append('.')  # Add project root to path

from src.data.synthetic_generator import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=50000)
    parser.add_argument('--shape_class', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='data/synthetic')
    args = parser.parse_args()
    
    print(f"Generating {args.n_samples} shapes of class {args.shape_class}...")
    
    generator = SyntheticDataGenerator(output_dir=args.output_dir)
    dataset = generator.generate_dataset(
        n_samples=args.n_samples,
        shape_class=args.shape_class
    )
    
    generator.save_dataset(dataset, f'class{args.shape_class}_{args.n_samples}.pkl')
    print("Done!")

if __name__ == '__main__':
    main()