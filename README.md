# Earthquake Aftershock Prediction using Deep Learning

A comparative analysis of four deep learning architectures (DNN, LSTM, GNN, CNN) for predicting earthquake aftershocks using the Southern California Earthquake Catalog (1990-2024).

## Project Overview

This project evaluates multiple deep learning approaches to predict whether an earthquake will trigger aftershocks within 7 days and a 50km radius. We compare 14 model variants across 4 architectures to identify the most effective approach for aftershock prediction.

### Key Results

| Architecture | Best Model | F1 Score | Precision | Recall | Key Strength |
|--------------|------------|----------|-----------|--------|--------------|
| **LSTM** | Shallow LSTM | **0.6958** | 78.59% | 62.42% | Temporal patterns |
| **DNN** | Standard DNN | 0.6667 | 91.76% | 52.35% | Simple & fast |
| **GNN** | GCN | 0.6555 | 77.44% | 56.82% | Spatial relationships |
| **CNN** | Shallow CNN | 0.5448 | 57.30% | 55.70% | Spatial grids |

**Winner:** Shallow LSTM achieved the best F1 score of 0.6958, demonstrating that temporal sequence modeling provides a significant advantage (+4.4% over DNN baseline) for aftershock prediction.

## Dataset

- **Source:** USGS Earthquake Catalog (Southern California)
- **Time Period:** 1990-2024
- **Size:** 5,727 earthquakes
- **Magnitude Range:** 3.5 to 7.2
- **Task:** Binary classification (triggers aftershock: yes/no)
- **Class Balance:** 48% negative, 52% positive (well-balanced)
- **Aftershock Definition:** Subsequent earthquake within 7 days and 50km radius

## Architecture Comparisons

### 1. Deep Neural Networks (DNN)
**Approach:** Tabular feature-based classification
- **Input:** 28 engineered features per earthquake
- **Features:** Spatial (fault distance, clustering), Temporal (inter-event time, rolling windows), Seismic (magnitude trends)
- **Variants:** Standard (3 layers), Shallow (2 layers), Deep (4 layers)
- **Best:** Standard DNN (F1 = 0.6667)
- **Strength:** High precision (91.76%), fast training
- **Weakness:** Lower recall (52.35%)

### 2. Long Short-Term Memory (LSTM)
**Approach:** Temporal sequence modeling
- **Input:** Sequences of 10 consecutive earthquakes
- **Features:** 28 features per earthquake in sequence
- **Variants:** Standard (2 LSTM layers), Shallow (1 layer), Deep (3 layers), Bidirectional
- **Best:** Shallow LSTM (F1 = 0.6958) **WINNER**
- **Strength:** Captures temporal patterns, best overall performance
- **Improvement:** +4.4% F1 over DNN, +19.2% recall improvement

### 3. Graph Neural Networks (GNN)
**Approach:** Graph-based spatial-temporal relationships
- **Input:** Graph with earthquakes as nodes, proximity as edges
- **Graph Construction:** k-NN spatial edges (within 50km) + temporal edges (within 6 hours)
- **Variants:** GCN, GAT, GraphSAGE, Temporal-GNN
- **Best:** GCN (F1 = 0.6555)
- **Finding:** Graph structure didn't improve over tabular features (-1.7% vs DNN)
- **Hypothesis:** k-NN graphs may not reflect true stress transfer physics

### 4. Convolutional Neural Networks (CNN)
**Approach:** Spatial grid-based modeling
- **Input:** 32×32 spatial grids centered on each earthquake
- **Channels:** 3 (earthquake density, average magnitude, temporal recency)
- **Context:** 50-event lookback window, 0.5° spatial radius
- **Variants:** Standard (4 conv blocks), Shallow (2 blocks), Deep (residual connections)
- **Best:** Shallow CNN (F1 = 0.5448)
- **Finding:** Poor performance (-18% vs DNN)
- **Issue:** Spatial grids alone insufficient; averaging loses individual earthquake information

## Project Structure

```
Seismic-GNN-Prediction/
├── data/
│   ├── raw/                          # USGS earthquake catalog
│   └── processed/
│       ├── features/                 # Engineered features
│       └── model_ready/              # Prepared inputs for each architecture
├── models/
│   ├── dnn/                          # DNN architectures
│   ├── lstm/                         # LSTM architectures
│   ├── gnn/                          # GNN architectures (GCN, GAT, GraphSAGE)
│   └── cnn/                          # CNN architectures
├── notebooks/
│   ├── 01_data_acquisition.ipynb    # Data loading and initial processing
│   ├── 02_exploratory_analysis.ipynb # EDA and visualization
│   ├── 03_feature_engineering.ipynb  # Feature creation for all architectures
│   ├── 04_dnn_training.ipynb        # DNN model training
│   ├── 05_lstm_training.ipynb       # LSTM model training (deprecated)
│   ├── 06_cnn_aftershock_prediction.ipynb # CNN model training
│   ├── 07_lstm_sequence_prediction.ipynb  # LSTM sequence training (current)
│   ├── 08_model_comparison_analysis.ipynb # Cross-architecture comparison (deprecated)
│   └── 09_model_comparison_analysis.ipynb # Cross-architecture comparison (current)
├── results/
│   ├── dnn/                          # DNN results and visualizations
│   ├── lstm/                         # LSTM results and visualizations
│   ├── gnn/                          # GNN results and visualizations
│   └── cnn/                          # CNN results and visualizations
├── requirements.txt                  # Python dependencies
├── report_generation_prompt.md       # Comprehensive report structure
├── VIDEO_WALKTHROUGH_PROMPT.md       # 10-minute video presentation guide
└── README.md                         # This file
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for GNN/CNN training)

### Setup

```bash
# Clone the repository
git clone https://github.com/IT22000644/Seismic-GNN-Prediction.git
cd Seismic-GNN-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch & PyTorch Geometric (GNN support)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualization)
- Obspy (seismology data processing)

## Usage

### 1. Data Acquisition and Exploration
```bash
jupyter notebook notebooks/01_data_acquisition.ipynb
jupyter notebook notebooks/02_exploratory_analysis.ipynb
```

### 2. Feature Engineering
```bash
# Generate features for all architectures
jupyter notebook notebooks/03_feature_engineering.ipynb
```

This notebook creates:
- **DNN features:** 28 tabular features per earthquake
- **LSTM features:** 10-event sequences with 28 features each
- **GNN features:** Graph structure with node features and edge indices
- **CNN features:** 32×32×3 spatial context grids

### 3. Model Training

**Train DNN models:**
```bash
jupyter notebook notebooks/04_dnn_training.ipynb
```

**Train LSTM models:**
```bash
jupyter notebook notebooks/07_lstm_sequence_prediction.ipynb
```

**Train CNN models:**
```bash
jupyter notebook notebooks/06_cnn_aftershock_prediction.ipynb
```

**Train GNN models:** (Use individual GNN training notebooks in `notebooks/`)

### 4. Results Comparison
```bash
jupyter notebook notebooks/09_model_comparison_analysis.ipynb
```

This notebook:
- Loads results from all architectures
- Selects best model from each
- Generates comprehensive comparison visualizations
- Provides detailed performance analysis

## Key Findings

### 1. Temporal Context is Crucial (+4.4%)
- LSTM's sequential modeling captured temporal patterns that static DNNs missed
- Recall improvement: +19.2% (62.42% vs 52.35%)
- Aligns with seismological theory (Omori's Law: aftershock rate decays with time)

### 2. Graph Structure Didn't Help (-1.7%)
- k-NN spatial graphs didn't improve over tabular features
- GCN performance slightly worse than DNN baseline
- Hypothesis: Simple proximity doesn't capture stress transfer physics
- Future work: Physics-informed graph edges using Coulomb stress models

### 3. Spatial Grids Alone Insufficient (-18%)
- CNNs performed poorly compared to other architectures
- 32×32 averaging lost individual earthquake information
- Spatial patterns alone can't predict temporal aftershock sequences
- Better suited for regional risk assessment than individual prediction

### 4. Complexity Paradox: Simpler Models Often Better
- Shallow LSTM (1 layer) > Deep LSTM (3 layers)
- Standard DNN > Deep DNN
- Dataset size (5,727 samples) limits deep model effectiveness
- More parameters ≠ better performance

### 5. Precision-Recall Trade-offs
- **DNN:** High precision (91.76%), low recall (52.35%) - very conservative
- **LSTM:** Balanced trade-off (78.59% precision, 62.42% recall)
- **CNN:** Near-random performance (57% precision, 56% recall)

## Practical Implications

### Real-Time Early Warning
- Shallow LSTM suitable for deployment (fast inference, high accuracy)
- 78% precision means 22% false alarm rate (acceptable for emergency response)
- 62% recall means 38% missed aftershocks (room for improvement)

### Model Efficiency
- LSTM: 26K parameters, F1 = 0.6958
- DNN: 14.5K parameters, F1 = 0.6667
- GNN: 12.6K parameters, F1 = 0.6555
- CNN: 1.07M parameters, F1 = 0.5448
- **Lesson:** More parameters don't guarantee better performance

### Integration with USGS Systems
- Models can process new earthquakes in real-time
- Compatible with existing ShakeAlert early warning infrastructure
- Potential to improve post-earthquake response planning

## Future Work

### 1. Ensemble Methods
- Combine LSTM + DNN predictions for improved robustness
- Weighted averaging based on confidence scores
- Stack models with meta-learner

### 2. Physics-Informed Graph Neural Networks
- Use Coulomb stress transfer models for graph edges
- Incorporate fault geometry and slip models
- Weight edges by stress change magnitude

### 3. Attention Mechanisms
- Temporal attention: which past earthquakes matter most?
- Spatial attention: which locations are most relevant?
- Interpret attention weights for scientific insights

### 4. Transfer Learning
- Train on California data, test on other seismic regions (Japan, Chile, Turkey)
- Fine-tune on regional datasets
- Evaluate cross-region generalization

### 5. Multi-Task Learning
- Jointly predict: aftershock occurrence, magnitude distribution, spatial extent
- Shared representations across tasks
- Improve overall prediction quality

### 6. Real-Time Deployment
- Integrate with USGS real-time feeds
- Containerized deployment (Docker/Kubernetes)
- API for emergency response systems
- Dashboard for seismologists

## Limitations

1. **Geographic Scope:** Trained only on Southern California; may not generalize globally
2. **Temporal Definition:** 7-day window is arbitrary; aftershocks can occur months later
3. **Class Balance:** Near-balanced dataset may not reflect all seismic regions
4. **Feature Engineering:** Hand-crafted features may miss important patterns
5. **Dataset Size:** 5,727 samples limits deep learning effectiveness
6. **Physics Integration:** Models are data-driven; don't explicitly incorporate seismic physics

## Academic Context

### Course Information
- **Course:** SE4050 - Deep Learning
- **Institution:** Sri Lanka Institute of Information Technology (SLIIT)
- **Semester:** Year 4, Semester 1 (2024/2025)
- **Project Type:** Comparative Deep Learning Analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Data Source:** USGS Earthquake Catalog
- **Inspiration:** ShakeAlert Early Warning System
- **Libraries:** PyTorch, PyTorch Geometric, Scikit-learn, ObsPy

## Contact

For questions or collaboration opportunities:
- **GitHub:** [@IT22000644](https://github.com/IT22000644)
- **Email:** manilkadikkumbura17@gmail.com
- **Repository:** https://github.com/IT22000644/Seismic-GNN-Prediction

## Additional Resources

**Best Model:** Shallow LSTM (F1 = 0.6958) - Ready for deployment testing
