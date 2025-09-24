# Taxi Driver Anomaly Detection System

A Graph Neural Network-based system for detecting anomalous driving patterns in taxi drivers using GPS trajectory data and POI information.

## 🚀 Features

- **Graph Attention Network (GAT)** for anomaly detection
- **Multi-cluster driver modeling** based on comfort zones
- **POI feature integration** for enhanced spatial understanding
- **Sparse graph processing** for efficient computation
- **Grid-based spatial analysis** with configurable granularity
- **Multiple loss functions** (Focal Loss, F1 Loss, Precision Loss)
- **Data augmentation** and caching mechanisms

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy, Pandas, Scikit-learn
- Matplotlib, Folium (for visualization)
- SciPy

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/taxi-anomaly-detection.git
cd taxi-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up data directories:
```bash
mkdir -p data/model data/similarity_matrix data/augmented_data_v2
```

## 📊 Data Format

### Input Data Requirements

1. **POI Data** (`poi_path`):
   - Format: Text file with POI information
   - Columns: Site coordinates, POI types and counts
   - Example: `examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt`

2. **Driver Order Data** (`driver_order_path`):
   - Format: CSV with driver trajectories
   - Columns: Driver ID, Order ID, Pickup/Dropoff coordinates and timestamps
   - Example: `order_driver_01.txt`

3. **Ground Truth** (`ground_truth_path`):
   - Format: Excel file with labeled anomaly locations
   - Columns: Coordinates and labels
   - Example: `ground_truth.xlsx`

### Data Structure
```
data/
├── model/                    # Trained models
├── similarity_matrix/        # Driver similarity matrices
├── augmented_data_v2/        # Augmented driver data
└── cluster_data_cache_v2.pkl # Cached cluster data
```

## 🚀 Quick Start

### Basic Usage

```python
from data_processing import data_prepare
from train import train

# Prepare data
poi_path = 'path/to/poi_data.txt'
driver_order_path = 'path/to/driver_orders.txt'
ground_truth_path = 'path/to/ground_truth.xlsx'
grid_granularity = 500

# Load and process data
features_by_cluster, adjs_by_cluster, labels_by_cluster, driver_clusters, cluster_boundaries = data_prepare(
    poi_path, driver_order_path, ground_truth_path, grid_granularity
)

# Train model
model, cluster_data = train(features_by_cluster, adjs_by_cluster, labels_by_cluster)
```

### Advanced Usage

```python
# Grid compression for efficiency
from main import compress_grid_data

# Compress grid data to smaller region
compressed_features = compress_grid_data(features_by_cluster)
compressed_labels = compress_grid_data(labels_by_cluster)

# Train with compressed data
model, cluster_data = train(compressed_features, adjs_by_cluster, compressed_labels)
```

## 📈 Model Architecture

### Sparse Graph Attention Network (SparseGAT)

- **Input**: Node features (POI + driver trajectory features)
- **Graph Structure**: Spatial adjacency + driver trajectory connections
- **Attention Mechanism**: Multi-head attention for neighbor aggregation
- **Output**: Binary classification (anomaly/normal)

### Key Components

1. **SparseGraphAttentionLayer**: Core attention mechanism
2. **Multi-head attention**: 8 attention heads for robust feature learning
3. **Dropout regularization**: 0.2 dropout rate
4. **LeakyReLU activation**: α=0.2 for negative slope

## 🔧 Configuration

### Model Parameters

```python
# Model hyperparameters
hidden_dim = 16          # Hidden layer dimension
dropout = 0.2           # Dropout rate
alpha = 0.2             # LeakyReLU negative slope
nheads = 8              # Number of attention heads
lr = 5e-4              # Learning rate
weight_decay = 5e-4    # Weight decay
epochs = 10000         # Training epochs
```

### Grid Configuration

```python
# Grid boundaries (adjust based on your data)
poi_boundary = [-28, -75, 73, 29]  # [min_x, min_y, max_x, max_y]
grid_granularity = 500             # Grid size in meters
```

## 📊 Performance Metrics

The system evaluates performance using:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### Example Results
```
Threshold: 0.5000, Avg Test Acc: 0.8542, 
Avg F1: 0.7234, Avg Precision: 0.6891, Avg Recall: 0.7612
```

## 🗂️ Project Structure

```
├── data_processing.py      # Data loading and preprocessing
├── model.py               # SparseGAT model definition
├── train.py               # Training and evaluation
├── main.py                # Main execution script
├── gps_analysis.py        # GPS trajectory analysis
├── utils.py               # Utility functions
├── preprocess.py          # Data preprocessing utilities
├── test_advanced.py       # Advanced testing
├── example_multi_cluster.py # Multi-cluster example
├── data/                  # Data directory
│   ├── model/            # Model files
│   ├── similarity_matrix/ # Similarity matrices
│   └── augmented_data_v2/ # Augmented data
└── README.md             # This file
```

## 🔬 Research Applications

This system can be used for:

- **Traffic Management**: Monitor taxi driver behavior patterns
- **Anomaly Detection**: Identify unusual driving patterns
- **Urban Planning**: Understand city traffic flows
- **Policy Making**: Data-driven transportation policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{taxi_anomaly_detection,
  title={Taxi Driver Anomaly Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/taxi-anomaly-detection}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Data format errors**: Check input data format and encoding
3. **Missing dependencies**: Install all requirements from requirements.txt

### Performance Tips

- Use GPU acceleration when available
- Enable data caching for repeated runs
- Compress grid data for large-scale analysis

## 📞 Support

For questions and support, please open an issue on GitHub or contact [your-email@example.com].

## 🔄 Changelog

### Version 1.0.0
- Initial release
- SparseGAT implementation
- Multi-cluster driver modeling
- POI feature integration
- Grid-based spatial analysis

---

**Note**: This system is designed for research purposes. Please ensure compliance with local data privacy regulations when using real-world data.
