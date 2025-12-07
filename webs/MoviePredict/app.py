from flask import Flask, render_template, request, jsonify
import sys
import os
from datetime import datetime
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import Pre-Release prediction service
from models.pre_release_service import get_prediction_service

app = Flask(__name__)

# Setup logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
app.config['SECRET_KEY'] = 'not-so-secret-key-lol'
app.config['DEBUG'] = True

# Get Pre-Release prediction service instance
prediction_service = get_prediction_service()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', model_accuracy=prediction_service.model_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle Pre-Release prediction requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'success': False
            }), 400
        
        # Validate required fields for Pre-Release prediction
        required_fields = ['title', 'budget']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'success': False
                }), 400
        
        # Set defaults for optional fields
        if 'runtime' not in data:
            data['runtime'] = 120
        if 'genres' not in data:
            data['genres'] = []
        if 'releaseMonth' not in data:
            data['releaseMonth'] = datetime.now().month
        if 'releaseYear' not in data:
            data['releaseYear'] = datetime.now().year
        
        # Use Pre-Release prediction service
        prediction_result = prediction_service.predict(data)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'will_succeed': prediction_result['success'],
                'confidence': round(prediction_result['success_probability'] * 100, 1),
                'success_probability': prediction_result['success_probability']
            },
            'metrics': prediction_result['metrics'],
            'feature_importance': prediction_result['feature_importance'],
            'input_data': {
                'title': data.get('title', 'Unknown'),
                'budget': float(data.get('budget', 0)),
                'runtime': int(data.get('runtime', 120)),
                'genres': data.get('genres', []),
                'release_month': int(data.get('releaseMonth', 6)),
                'prediction_type': 'pre_release'
            },
            'model_info': {
                **prediction_result['model_info'],
                'is_real_model': prediction_service.model is not None,
                'prediction_type': 'pre_release',
                'note': 'Pre-Release prediction - chỉ dùng thông tin biết trước'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.exception(f"Lỗi khi thực hiện dự đoán: {e}")
        return jsonify({
            'error': f'Lỗi khi thực hiện dự đoán: {str(e)}',
            'success': False
        }), 500

@app.route('/api/model-info')
def model_info():
    """Get information about the loaded Pre-Release model"""
    return jsonify({
        'model_loaded': prediction_service.model is not None,
        'model_type': 'Pre-Release Random Forest',
        'accuracy': prediction_service.model_accuracy,
        'features_count': len(prediction_service.feature_names) if prediction_service.feature_names else 0,
        'features': prediction_service.feature_names[:10] if prediction_service.feature_names else [],
        'status': 'ready',
        'prediction_type': 'pre_release',
        'description': 'Dự đoán trước phát hành - không có data leakage',
        'is_real_model': prediction_service.model is not None
    })

@app.route('/api/sample-data')
def sample_data():
    """Get sample data for Pre-Release testing"""
    samples = [
        {
            'title': 'Blockbuster Action',
            'budget': 200000000,
            'runtime': 150,
            'releaseMonth': 6,
            'genres': ['Action', 'Adventure', 'Sci-Fi']
        },
        {
            'title': 'Indie Drama',
            'budget': 5000000,
            'runtime': 105,
            'releaseMonth': 10,
            'genres': ['Drama']
        },
        {
            'title': 'Summer Comedy',
            'budget': 40000000,
            'runtime': 98,
            'releaseMonth': 7,
            'genres': ['Comedy']
        },
        {
            'title': 'Holiday Horror',
            'budget': 15000000,
            'runtime': 95,
            'releaseMonth': 10,
            'genres': ['Horror', 'Thriller']
        }
    ]
    return jsonify(samples)

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    logger.info("%s", "="*50)
    logger.info("> Pre-Release Movie Success Prediction")
    logger.info("> Model: %s", 'Pre-Release Random Forest' if prediction_service.model else 'Not loaded')
    logger.info("> Accuracy: %.2f%%", prediction_service.model_accuracy*100)
    logger.info("> Features: %s", len(prediction_service.feature_names) if prediction_service.feature_names else 0)
    logger.info("> Note: Không có data leakage")
    logger.info("%s", "="*50)
    logger.info("> Starting Flask server...")
    logger.info("> Open: http://localhost:8000")
    logger.info("%s", "="*50)

    app.run(host='0.0.0.0', port=8000, debug=True)