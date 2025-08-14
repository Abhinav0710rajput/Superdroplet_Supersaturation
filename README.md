# Superdroplet Supersaturation

A collection of scripts and tools for analyzing and visualizing supersaturation and superdroplet data, including data preparation, machine learning model, and plotting utilities.

## Features

- Data preparation and preprocessing scripts
- Multiple plotting utilities for error analysis and supersaturation visualization
- Machine learning training scripts for various models
- Modular codebase for easy extension

## Repository Structure

```
.
├── check_data.py
├── compare_generate_plots.py
├── compare_generate_plots_ss.py
├── error_plots.py
├── generate_plots.py
├── prepare_data.py
├── prepare_data_cnn.py
├── prepare_data_hist.py
├── prepare_data_min.py
├── supersaturation_generate_plots.py
├── train_ml_1.py
├── train_ml_2.py
├── train_ml_3.py
```

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Abhinav0710rajput/Superdroplet_Supersaturation.git
   cd Superdroplet_Supersaturation
   ```

2. **Install dependencies:**  
   *(List your dependencies here, e.g. numpy, matplotlib, scikit-learn, etc.)*
   ```sh
   pip install -r requirements.txt
   ```

3. **Run scripts:**  
   Each script is modular. For example:
   ```sh
   python prepare_data.py
   python train_ml_1.py
   python generate_plots.py
   ```

## Scripts Overview

- **prepare_data.py / prepare_data_cnn.py / prepare_data_hist.py / prepare_data_min.py**  
  Data preparation scripts for various formats and models.

- **train_ml_1.py / train_ml_2.py / train_ml_3.py**  
  Train different machine learning models on the prepared data.

- **generate_plots.py / error_plots.py / compare_generate_plots.py / compare_generate_plots_ss.py / supersaturation_generate_plots.py**  
  Generate and compare plots for analysis and visualization.

- **check_data.py**  
  Utility for checking and validating data integrity.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
