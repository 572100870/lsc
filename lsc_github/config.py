"""
Configuration file for Taxi Driver Anomaly Detection System
"""

import os

class Config:
    """Main configuration class"""
    
    # Data paths
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    MODEL_PATH = os.path.join(DATA_PATH, 'model')
    
    # Default data paths (update these for your environment)
    POI_PATH = 'E:/Project/traffic/order_data/examine_neighbor_poi_anomaly_detect_chengdu_in_500_1101.txt'
    DRIVER_ORDER_PATH = 'E:/Project/traffic/order_data/order_driver_01.txt'
    GROUND_TRUTH_PATH = 'E:/Project/traffic/order_data/ground_truth.xlsx'
    
    # Grid configuration
    GRID_GRANULARITY = 500  # Grid size in meters
    POI_BOUNDARY = [-28, -75, 73, 29]  # [min_x, min_y, max_x, max_y]
    
    # Model hyperparameters
    HIDDEN_DIM = 16
    DROPOUT = 0.2
    ALPHA = 0.2  # LeakyReLU negative slope
    NHEADS = 8
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 5e-4
    EPOCHS = 10000
    
    # Training configuration
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # Loss function weights
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    FOCAL_WEIGHT = 0.5
    F1_WEIGHT = 0.5
    
    # Device configuration
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    # Cache configuration
    CACHE_ENABLED = True
    CACHE_PATH = os.path.join(DATA_PATH, 'cluster_data_cache_v2.pkl')
    
    # Similarity matrix configuration
    SIMILARITY_MATRIX_PATH = os.path.join(DATA_PATH, 'similarity_matrix')
    
    # Augmented data configuration
    AUGMENTED_DATA_PATH = os.path.join(DATA_PATH, 'augmented_data_v2')
    
    # Model save paths
    BEST_MODEL_PATH = os.path.join(MODEL_PATH, 'best_model_multi_cluster.pth')
    SINGLE_MODEL_PATH = os.path.join(MODEL_PATH, 'best_model_new.pth')
    
    # Visualization configuration
    PLOT_DPI = 300
    PLOT_FIGSIZE = (10, 6)
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Data processing configuration
    MIN_DRIVER_ORDERS = 2  # Minimum orders per driver
    MIN_POSITIVE_SAMPLES = 300  # Minimum positive samples per cluster
    MAX_DRIVERS_FOR_PROCESSING = 10000  # Limit for initial processing
    
    # Clustering configuration
    CLUSTERING_THRESHOLD = 0.8  # IoU threshold for cluster assignment
    MIN_CLUSTER_SIZE = 10  # Minimum drivers per cluster
    
    @classmethod
    def update_paths(cls, poi_path=None, driver_order_path=None, ground_truth_path=None):
        """Update data paths"""
        if poi_path:
            cls.POI_PATH = poi_path
        if driver_order_path:
            cls.DRIVER_ORDER_PATH = driver_order_path
        if ground_truth_path:
            cls.GROUND_TRUTH_PATH = ground_truth_path
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_PATH,
            cls.MODEL_PATH,
            cls.SIMILARITY_MATRIX_PATH,
            cls.AUGMENTED_DATA_PATH
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")

class ModelConfig:
    """Model-specific configuration"""
    
    # SparseGAT parameters
    INPUT_FEATURES = 782  # Total feature dimension
    OUTPUT_CLASSES = 1    # Binary classification
    
    # Attention mechanism
    ATTENTION_DROPOUT = 0.2
    ATTENTION_ALPHA = 0.2
    
    # Training parameters
    PATIENCE = 50  # Early stopping patience
    MIN_DELTA = 1e-4  # Minimum change for early stopping
    
    # Evaluation thresholds
    EVALUATION_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]

class DataConfig:
    """Data processing configuration"""
    
    # POI categories
    POI_TABLE = [
        '生活服务', '教育培训', '交通设施', '汽车服务', '道路', '休闲娱乐', 
        '文化传媒', '丽人', '房地产', '美食', '酒店', '公司企业', '购物', 
        '政府机构', '金融', '医疗', '旅游景点', '运动健身', '自然地物', 
        '铁路', '公交线路'
    ]
    
    # Feature dimensions
    POI_FEATURE_DIM = 22  # 21 POI types + 1 kind type
    LAYER_DIMENSIONS = [10, 10, 10, 10, 20, 20, 20, 30, 40, 40, 40, 40, 
                       50, 50, 60, 40, 40, 40, 40, 40, 30, 30, 30, 20]
    
    # Time window configuration
    TIME_WINDOWS = [3600, 7200, 3600, 3600 * 3, 1800, 7200, 1800, 3600 * 3]
    
    # Grid compression
    ORIGINAL_X_RANGE = (-89, 146)
    ORIGINAL_Y_RANGE = (-107, 68)
    NEW_X_RANGE = (-28, 73)
    NEW_Y_RANGE = (-75, 29)

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    EPOCHS = 100  # Reduced for development

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    CACHE_ENABLED = True

class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    EPOCHS = 10
    MAX_DRIVERS_FOR_PROCESSING = 100

# Configuration factory
def get_config(env='development'):
    """Get configuration based on environment"""
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    return configs.get(env, Config)
