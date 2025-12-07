"""
Pre-Release Movie Success Prediction Service
=============================================
Sử dụng Random Forest model đã được training với PRE-RELEASE features
Không có data leakage (không dùng revenue, vote_average)
"""

import pickle
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreReleaseMoviePredictionService:
    """
    Service dự đoán thành công phim TRƯỚC KHI PHÁT HÀNH.
    Chỉ sử dụng features biết trước: budget, runtime, genres, release timing, etc.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.model_accuracy = 0.6765  # Accuracy từ training
        self.model_info = {
            'model_type': 'Pre-Release Random Forest',
            'accuracy': 0.6765,
            'f1_score': 0.6796,
            'cv_mean': 0.6931,
            'training_data_size': 1020,
            'features_count': 37,
            'description': 'Dự đoán trước phát hành - không có data leakage'
        }
        self._load_model()
    
    def _load_model(self) -> None:
        """Load Pre-Release model từ file pkl."""
        try:
            # Đường dẫn tới project root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            
            # Load Pre-Release model
            model_path = os.path.join(project_root, 'data', 'pkl', 'pre_release_rf_model.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                
                # Cập nhật metrics từ model
                if 'metrics' in model_data:
                    self.model_accuracy = model_data['metrics'].get('accuracy', 0.6765)
                    self.model_info['accuracy'] = self.model_accuracy
                    self.model_info['f1_score'] = model_data['metrics'].get('f1_score', 0.6796)
                    self.model_info['cv_mean'] = model_data['metrics'].get('cv_mean', 0.6931)
                
                logger.info(f"Pre-Release Model loaded: acc={self.model_accuracy*100:.2f}%, features={len(self.feature_names)}")
            else:
                raise FileNotFoundError(f"Không tìm thấy Pre-Release model tại: {model_path}")
                
        except Exception as e:
            logger.error(f"Lỗi khi load Pre-Release model: {e}")
            raise e
    
    def prepare_features(self, input_data: dict) -> np.ndarray:
        """
        Chuẩn bị features từ input data.
        Chỉ sử dụng PRE-RELEASE features (không có revenue, vote_average).
        """
        try:
            # Khởi tạo tất cả features = 0
            features = {name: 0.0 for name in self.feature_names}
            
            # === BASIC FEATURES ===
            budget = float(input_data.get('budget', 0))
            runtime = float(input_data.get('runtime', 120))
            
            # Budget log
            if 'Budget_log' in features:
                features['Budget_log'] = np.log10(budget + 1) if budget > 0 else 0
            
            # Runtime features
            if 'runtime_minutes' in features:
                features['runtime_minutes'] = runtime
            if 'runtime_hours' in features:
                features['runtime_hours'] = runtime / 60.0
            
            # === TIME FEATURES ===
            release_month = int(input_data.get('releaseMonth', datetime.now().month))
            release_year = int(input_data.get('releaseYear', datetime.now().year))
            release_weekday = int(input_data.get('releaseWeekday', 4))  # Default Friday
            
            if 'release_year' in features:
                features['release_year'] = release_year
            if 'release_month' in features:
                features['release_month'] = release_month
            if 'release_weekday' in features:
                features['release_weekday'] = release_weekday
            if 'release_quarter' in features:
                features['release_quarter'] = (release_month - 1) // 3 + 1
            if 'is_holiday_season' in features:
                features['is_holiday_season'] = 1 if release_month in [6, 7, 11, 12] else 0
            
            # === GENRE FEATURES ===
            genres = input_data.get('genres', [])
            if isinstance(genres, str):
                genres = [g.strip() for g in genres.split(',') if g.strip()]
            
            # Đếm số genres
            if 'num_genres' in features:
                features['num_genres'] = len(genres)
            
            # One-hot encode genres
            genre_mapping = {
                'Action': 'genre_Action',
                'Adventure': 'genre_Adventure',
                'Comedy': 'genre_Comedy',
                'Drama': 'genre_Drama',
                'Thriller': 'genre_Thriller',
                'Science Fiction': 'genre_Science Fiction',
                'Family': 'genre_Family',
                'Fantasy': 'genre_Fantasy',
                'Crime': 'genre_Crime',
                'Animation': 'genre_Animation',
                'Horror': 'genre_Horror',
                'Romance': 'genre_Romance',
                'Mystery': 'genre_Mystery',
                'History': 'genre_History',
                'Music': 'genre_Music'
            }
            
            for genre in genres:
                feature_name = genre_mapping.get(genre)
                if feature_name and feature_name in features:
                    features[feature_name] = 1
            
            # === COUNTRY FEATURES ===
            countries = input_data.get('countries', [])
            if isinstance(countries, str):
                countries = [c.strip() for c in countries.split(',') if c.strip()]
            
            country_mapping = {
                'United States of America': ['is_united_states_of_america', 'is_usa'],
                'USA': ['is_united_states_of_america', 'is_usa'],
                'United Kingdom': ['is_united_kingdom'],
                'UK': ['is_united_kingdom'],
                'Vietnam': ['is_vietnam'],
                'China': ['is_china'],
                'France': ['is_france'],
                'South Korea': ['is_south_korea'],
                'Korea': ['is_south_korea'],
                'Australia': ['is_australia'],
                'Japan': ['is_japan'],
                'India': ['is_india'],
                'Canada': ['is_canada']
            }
            
            for country in countries:
                feature_names_list = country_mapping.get(country, [])
                for feat in feature_names_list:
                    if feat in features:
                        features[feat] = 1
            
            # Default to USA if no country specified
            if not countries:
                if 'is_usa' in features:
                    features['is_usa'] = 1
                if 'is_united_states_of_america' in features:
                    features['is_united_states_of_america'] = 1
            
            # === CAST FEATURES ===
            num_cast = int(input_data.get('numCast', 3))
            if 'num_main_cast' in features:
                features['num_main_cast'] = num_cast
            if 'cast_genre_interaction' in features:
                features['cast_genre_interaction'] = num_cast * len(genres)
            
            # Tạo feature vector theo đúng thứ tự
            feature_vector = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                feature_vector = self.scaler.transform(feature_vector)
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị features: {e}")
            raise e
    
    def predict(self, input_data: dict) -> dict:
        """
        Dự đoán thành công của phim.
        
        Returns:
            dict chứa kết quả dự đoán
        """
        try:
            # Prepare features
            features = self.prepare_features(input_data)
            
            # Predict
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            success_prob = probability[1]  # Xác suất thành công
            
            # Determine risk level
            if success_prob >= 0.7:
                risk_level = 'LOW'
                risk_description = 'Phim có tiềm năng thành công cao'
            elif success_prob >= 0.5:
                risk_level = 'MEDIUM'
                risk_description = 'Phim có tiềm năng trung bình'
            else:
                risk_level = 'HIGH'
                risk_description = 'Phim có rủi ro thất bại cao'
            
            # Calculate estimated metrics
            budget = float(input_data.get('budget', 0))
            estimated_roi = self._estimate_roi(success_prob, budget)
            
            result = {
                'success': bool(prediction == 1),
                'success_probability': float(success_prob),
                'confidence': float(max(probability)),
                'risk_level': risk_level,
                'risk_description': risk_description,
                'metrics': {
                    'estimated_roi': estimated_roi,
                    'risk_score': round((1 - success_prob) * 100, 1),
                    'success_score': round(success_prob * 100, 1)
                },
                'feature_importance': self._get_top_features(),
                'model_info': self.model_info,
                'prediction_type': 'pre_release'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán: {e}")
            raise e
    
    def _estimate_roi(self, success_prob: float, budget: float) -> float:
        """Ước tính ROI dựa trên xác suất thành công."""
        if budget <= 0:
            return 0.0
        
        # ROI ước tính dựa trên probability
        # Phim thành công thường có ROI 2-5x
        # Phim thất bại thường có ROI 0.3-0.8x
        if success_prob >= 0.7:
            estimated_roi = 2.0 + (success_prob - 0.7) * 10  # 2.0 - 5.0
        elif success_prob >= 0.5:
            estimated_roi = 1.0 + (success_prob - 0.5) * 5  # 1.0 - 2.0
        else:
            estimated_roi = 0.3 + success_prob * 1.4  # 0.3 - 1.0
        
        return round(estimated_roi, 2)
    
    def _get_top_features(self) -> list:
        """Trả về top features quan trọng nhất."""
        if self.model is None:
            return []
        
        importances = self.model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {'feature': name, 'importance': round(imp * 100, 2)}
            for name, imp in feature_importance[:10]
        ]


# Singleton instance
_service_instance = None

def get_prediction_service() -> PreReleaseMoviePredictionService:
    """Get singleton instance of prediction service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PreReleaseMoviePredictionService()
    return _service_instance
