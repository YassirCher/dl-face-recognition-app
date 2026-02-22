from flask import Flask, render_template, Response, jsonify, request
import os
import cv2
from werkzeug.utils import secure_filename
from camera import VideoCamera
from model_loader import ModelManager
from analytics import Analytics

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR_PART1 = os.path.join(BASE_DIR, 'model', 'part 1')
MODEL_DIR_PART2 = os.path.join(BASE_DIR, 'model', 'part 2')
MODEL_DIR_PART3 = os.path.join(BASE_DIR, 'model', 'part 3')
RESULTS_FILE_PART1 = os.path.join(MODEL_DIR_PART1, 'part1_model_results.json')
RESULTS_FILE_PART2 = os.path.join(MODEL_DIR_PART2, 'part2_model_results.json')
RESULTS_FILE_PART3 = os.path.join(MODEL_DIR_PART3, 'part3_model_results.json')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize components
model_manager = ModelManager([MODEL_DIR_PART1, MODEL_DIR_PART2, MODEL_DIR_PART3])
analytics = Analytics([RESULTS_FILE_PART1, RESULTS_FILE_PART2, RESULTS_FILE_PART3])
video_camera = VideoCamera(model_manager)

# Global model state
current_model_config = {
    'backbone': None,
    'classifier': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify(model_manager.get_available_models())

@app.route('/api/current_model', methods=['GET'])
def get_current_model():
    return jsonify(current_model_config)

@app.route('/api/set_model', methods=['POST'])
def set_model():
    data = request.json
    backbone = data.get('backbone')
    classifier = data.get('classifier')
    
    if not backbone or not classifier:
        return jsonify({'error': 'Missing backbone or classifier'}), 400
    
    try:
        video_camera.set_model(backbone, classifier)
        current_model_config['backbone'] = backbone
        current_model_config['classifier'] = classifier
        return jsonify({'success': True, 'message': f'Model switched to {backbone} + {classifier}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            video_camera.set_video_source(filepath)
            return jsonify({'success': True, 'message': f'Video uploaded and playing: {filename}'})
        except Exception as e:
            return jsonify({'error': f'Failed to process video: {str(e)}'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    return jsonify(analytics.get_analytics_data())

@app.route('/api/system_stats', methods=['GET'])
def system_stats():
    """Return CPU and RAM usage."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
    except ImportError:
        # Fallback if psutil not installed
        cpu = 0
        ram = 0
    return jsonify({'cpu': cpu, 'ram': ram})

@app.route('/api/inference_stats', methods=['GET'])
def inference_stats():
    """Return real-time inference statistics from the camera."""
    return jsonify(video_camera.stats.get_stats())

@app.route('/api/detection_summary', methods=['GET'])
def detection_summary():
    """Return detection summary for the video."""
    return jsonify(video_camera.get_detection_summary())

@app.route('/api/replay_video', methods=['POST'])
def replay_video():
    """Replay the current video from the beginning."""
    success = video_camera.replay_video()
    if success:
        return jsonify({'success': True, 'message': 'Video replaying from start'})
    return jsonify({'success': False, 'error': 'No video to replay'}), 400

if __name__ == '__main__':
    # Threaded=True is important for video streaming to not block other requests
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
