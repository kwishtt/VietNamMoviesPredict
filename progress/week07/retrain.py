"""
Pre-Release Movie Success Prediction Model
==========================================
Train Random Forest model chỉ sử dụng features biết trước khi phim ra mắt.
Loại bỏ data leakage từ revenue, vote_average, roi.

Author: Team Do An
Date: 2024
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Cấu hình logging - log ra cả console và file
def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging to both console and file."""
    logger = logging.getLogger('PreReleaseModel')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Placeholder - will be set in main()
logger = logging.getLogger('PreReleaseModel')


# ============================================
# ĐỊNH NGHĨA FEATURES
# ============================================

# Features POST-RELEASE (CẦN LOẠI BỎ - data leakage)
POST_RELEASE_FEATURES = [
    'revenue', 'Revenue_log', 
    'vote_average', 'Vote Average', 'vote_count',
    'roi', 'roi_clipped', 'roi_vs_vote', 'budget_per_year',
    'success'  # Label, không phải feature
]

# Features không dùng (metadata)
METADATA_COLUMNS = [
    'Id', 'Title', 'Original Title', 'Original Language', 'Overview',
    'Release Date', 'Genres', 'Production Companies', 'Production Countries',
    'Spoken Languages', 'Director', 'Stars', 'genres_list',
    'country_simple', 'country_grouped', 'main_genre', 'runtime_group'
]

# Features PRE-RELEASE (sử dụng)
PRE_RELEASE_FEATURES = [
    # Basic
    'budget', 'Budget_log', 'runtime', 'runtime_minutes', 'runtime_hours',
    
    # Time features
    'release_year', 'release_month', 'release_weekday', 
    'release_quarter', 'is_holiday_season',
    
    # Genre features (one-hot encoded)
    'num_genres',
    'genre_Action', 'genre_Adventure', 'genre_Comedy', 'genre_Drama',
    'genre_Thriller', 'genre_Science Fiction', 'genre_Family', 'genre_Fantasy',
    'genre_Crime', 'genre_Animation', 'genre_Horror', 'genre_Romance',
    'genre_Mystery', 'genre_History', 'genre_Music',
    
    # Country features
    'is_united_states_of_america', 'is_united_kingdom', 'is_canada',
    'is_vietnam', 'is_china', 'is_france', 'is_south_korea',
    'is_australia', 'is_japan', 'is_india', 'is_usa',
    
    # Cast features
    'num_main_cast', 'cast_genre_interaction'
]


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset từ CSV file."""
    logger.info(f"Đang load data từ: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} phim với {len(df.columns)} cột")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Chọn Pre-Release features và loại bỏ Post-Release features.
    
    Returns:
        X: DataFrame chứa features
        y: Series chứa labels
        feature_names: List tên các features được sử dụng
    """
    logger.info("Đang chọn Pre-Release features...")
    
    # Lấy label
    if 'success' not in df.columns:
        raise ValueError("Không tìm thấy cột 'success' trong dataset")
    y = df['success'].copy()
    
    # Chọn features có trong dataset
    available_features = []
    missing_features = []
    
    for feat in PRE_RELEASE_FEATURES:
        if feat in df.columns:
            available_features.append(feat)
        else:
            missing_features.append(feat)
    
    if missing_features:
        logger.warning(f"Features không có trong dataset: {missing_features}")
    
    logger.info(f"Sử dụng {len(available_features)} features")
    
    X = df[available_features].copy()
    
    # Xử lý missing values
    X = X.fillna(0)
    
    # Xử lý infinity values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y, available_features


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Train Random Forest model với cross-validation.
    
    Returns:
        model: Trained model
        scaler: Fitted scaler
        metrics: Dictionary chứa evaluation metrics
    """
    logger.info("Bắt đầu train model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Xử lý class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    logger.info("Model đã train xong!")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    logger.info("=" * 50)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred, target_names=['Thất bại', 'Thành công']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Cross-validation
    logger.info("\n" + "=" * 50)
    logger.info("5-FOLD CROSS VALIDATION")
    logger.info("=" * 50)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=cv, scoring='accuracy')
    
    metrics['cv_mean'] = cv_scores.mean()
    metrics['cv_std'] = cv_scores.std()
    
    logger.info(f"CV Scores: {cv_scores}")
    logger.info(f"CV Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return model, scaler, metrics, X.columns.tolist()


def analyze_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Phân tích Feature Importance."""
    logger.info("\n" + "=" * 50)
    logger.info("FEATURE IMPORTANCE")
    logger.info("=" * 50)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 15 features
    logger.info("\nTop 15 Features quan trọng nhất:")
    for i, row in importance_df.head(15).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']*100:.2f}%")
    
    return importance_df


def save_model(model, scaler, feature_names: list, metrics: dict, output_dir: str) -> None:
    """Lưu model và metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_path / 'pre_release_rf_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'metrics': metrics,
            'model_type': 'pre_release',
            'description': 'Random Forest model cho Pre-Release Prediction (không có data leakage)'
        }, f)
    
    logger.info(f"\nModel đã lưu tại: {model_path}")


def main():
    """Main function."""
    global logger
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_path = project_root / 'data' / 'clean_movies_features.csv'
    output_dir = script_dir / 'output'  # Lưu vào week07/output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_file = output_dir / 'training_log.txt'
    logger = setup_logging(str(log_file))
    
    logger.info("=" * 60)
    logger.info("PRE-RELEASE MOVIE SUCCESS PREDICTION MODEL")
    logger.info("=" * 60)
    
    # Load data
    df = load_data(str(data_path))
    
    # Thống kê label
    success_rate = df['success'].mean()
    logger.info(f"\nPhân bố label:")
    logger.info(f"  Thành công: {df['success'].sum()} ({success_rate*100:.1f}%)")
    logger.info(f"  Thất bại: {len(df) - df['success'].sum()} ({(1-success_rate)*100:.1f}%)")
    
    # Select features
    X, y, feature_names = select_features(df)
    
    # Train model
    model, scaler, metrics, used_features = train_model(X, y)
    
    # Feature importance
    importance_df = analyze_feature_importance(model, used_features)
    
    # Save importance to CSV
    importance_csv = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_csv, index=False)
    logger.info(f"Feature importance saved to: {importance_csv}")
    
    # Save model to output dir
    save_model(model, scaler, used_features, metrics, str(output_dir))
    
    # Also save to data/pkl for web app
    pkl_dir = project_root / 'data' / 'pkl'
    pkl_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, scaler, used_features, metrics, str(pkl_dir))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Model type: Pre-Release Prediction")
    logger.info(f"✅ Features used: {len(used_features)}")
    logger.info(f"✅ Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"✅ F1-Score: {metrics['f1_score']*100:.2f}%")
    logger.info(f"✅ CV Mean: {metrics['cv_mean']*100:.2f}%")
    logger.info(f"✅ Output directory: {output_dir}")
    logger.info(f"✅ Log file: {log_file}")
    logger.info("=" * 60)
    
    return model, scaler, metrics


if __name__ == '__main__':
    main()

