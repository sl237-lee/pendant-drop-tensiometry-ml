cat > README.md << 'EOF'
# Pendant Drop Tensiometry - Machine Learning Approach

Machine learning model for determining surface tension from pendant drop images.

Based on: Kratz & Kierfeld (2020) - "Pendant drop tensiometry: A machine learning approach"

## Project Structure
```
pendant-drop-ml/
├── src/                  # Source code modules
│   ├── physics/         # Young-Laplace solver, shape equations
│   ├── data/            # Data generation & augmentation
│   ├── preprocessing/   # Image processing & edge detection
│   ├── models/          # Neural network architecture
│   └── utils/           # Helper functions
├── notebooks/           # Jupyter notebooks for exploration
├── scripts/            # Standalone executable scripts
├── data/               # Training and test data
├── models/             # Saved model weights
└── results/            # Output plots and analysis
```

## Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Mac/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Generate training data
python scripts/generate_dataset.py --n_samples 50000

# Train model
python scripts/train_model.py --epochs 100

# Evaluate model
python scripts/evaluate_model.py
```

## Team

- Lab Team: Image acquisition & calibration
- ML Team: Model development & training

## Status

- [x] Project structure setup
- [ ] Synthetic data generation
- [ ] Shape diagram reproduction (Figure 4)
- [ ] Model training pipeline
- [ ] Image preprocessing pipeline
- [ ] Integration with lab images
EOF