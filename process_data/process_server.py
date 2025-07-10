# process_server.py (Updated with Anomaly Clearing)
import json
import re
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from gensim.models import Word2Vec
import threading
import os
from waitress import serve
import collections
import datetime
import io
import tempfile
import zipfile

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector

# --- Global State & Configuration ---
app = Flask(__name__, template_folder='.')
STATE_DIR = "/state" 

# In-memory data
collected_logs_corpus = [] 
word2vec_model, lstm_model = None, None
MODEL_LOCK = threading.Lock()
RECENT_LOGS_BUFFER = collections.deque(maxlen=30)
RECENTLY_COLLECTED_LOGS = collections.deque(maxlen=100)
FLAGGED_ANOMALIES, ANOMALY_THRESHOLD, COLLECTION_TIMER, COLLECTION_END_TIME = [], 0.0, None, None

# State flags
COLLECTING_STATE_FILE = os.path.join(STATE_DIR, "collecting.state")
MONITORING_STATE_FILE = os.path.join(STATE_DIR, "monitoring.state")


# --- Helper Functions ---
def preprocess_log_message(log_text):
    """Cleans and tokenizes a single log message."""
    if not isinstance(log_text, str): return []
    text = log_text.lower()
    text = re.sub(r'[^\w\s\.]', ' ', text) # Keep periods for IPs/versions
    tokens = text.split()
    stop_words = {'a', 'an', 'the', 'in', 'is', 'to', 'of', 'for', 'go', 'i', 'e', 'my', 's', 'at', 'on', 'and', 'are', 'was', 'were','t', 'z', 'f'}
    processed_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return processed_tokens

def get_vector_for_log(log_tokens, model, vector_size=100):
    """Calculates the average vector for a tokenized log message."""
    vectors = [model.wv[word] for word in log_tokens if word in model.wv]
    if not vectors: return np.zeros(vector_size)
    return np.mean(vectors, axis=0)

def create_sequences(vectors, sequence_length):
    """Creates overlapping sequences from a list of vectors."""
    sequences = []
    for i in range(len(vectors) - sequence_length + 1):
        sequences.append(vectors[i:i + sequence_length])
    return np.array(sequences)

def summarize_locally(log_sequence_text, w2v_model):
    """
    Summarizes a sequence of logs by finding the log line most central to the cluster.
    """
    if not log_sequence_text:
        return "No log sequence provided for summary."
    if not w2v_model:
        return "Word2Vec model not available for summarization."

    log_vectors = []
    original_logs_for_summary = []

    for log_line in log_sequence_text:
        tokens = preprocess_log_message(log_line)
        if not tokens:
            continue
        
        original_logs_for_summary.append(log_line)
        log_vectors.append(get_vector_for_log(tokens, w2v_model))

    if not log_vectors:
        return log_sequence_text[0]

    log_vectors_np = np.array(log_vectors)
    centroid = np.mean(log_vectors_np, axis=0)

    similarities = []
    for vector in log_vectors_np:
        dot_product = np.dot(vector, centroid)
        norm_vector = np.linalg.norm(vector)
        norm_centroid = np.linalg.norm(centroid)
        
        if norm_vector == 0 or norm_centroid == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_vector * norm_centroid)
        similarities.append(similarity)

    if not similarities:
        return log_sequence_text[0]

    most_similar_index = np.argmax(similarities)
    
    return original_logs_for_summary[most_similar_index]


# --- Main Log Reception Endpoint ---
@app.route('/', methods=['POST'])
def receive_logs():
    """Endpoint to receive logs for collection or monitoring."""
    is_collecting = os.path.exists(COLLECTING_STATE_FILE)
    is_monitoring = os.path.exists(MONITORING_STATE_FILE)
    if not is_collecting and not is_monitoring: return jsonify({"status": "ignored"}), 200
    
    try:
        logs = request.get_json(force=True)
        if not logs: return jsonify({"status": "error", "message": "Empty JSON payload"}), 400
        
        if is_collecting:
            logs_processed_count = 0
            for log_entry in logs:
                message = ""
                if "log_processed" in log_entry and isinstance(log_entry.get("log_processed"), dict): message = log_entry.get("log_processed", {}).get("log", "")
                elif "MESSAGE" in log_entry: message = log_entry.get("MESSAGE", "")
                elif isinstance(log_entry, dict):
                    longest_value = max(log_entry.values(), key=lambda v: len(v) if isinstance(v, str) else -1)
                    if isinstance(longest_value, str): message = longest_value
                
                if message:
                    RECENTLY_COLLECTED_LOGS.appendleft(message)
                    processed_tokens = preprocess_log_message(message)
                    if processed_tokens:
                        collected_logs_corpus.append(processed_tokens)
                        logs_processed_count += 1
            if logs_processed_count > 0: print(f"INFO: Added {logs_processed_count} logs to corpus. Total: {len(collected_logs_corpus)}", flush=True)

        elif is_monitoring:
            if not (word2vec_model and lstm_model): return jsonify({"status": "ignored", "reason": "models_not_trained"})
            SEQUENCE_LENGTH = 10
            for log_entry in logs:
                message = ""
                if "log_processed" in log_entry and isinstance(log_entry.get("log_processed"), dict): message = log_entry.get("log_processed", {}).get("log", "")
                elif "MESSAGE" in log_entry: message = log_entry.get("MESSAGE", "")
                elif isinstance(log_entry, dict):
                    longest_value = max(log_entry.values(), key=lambda v: len(v) if isinstance(v, str) else -1)
                    if isinstance(longest_value, str): message = longest_value

                if message:
                    tokens = preprocess_log_message(message)
                    if tokens:
                        vector = get_vector_for_log(tokens, word2vec_model)
                        RECENT_LOGS_BUFFER.append({"tokens": tokens, "vector": vector, "original_log": message})
            
            if len(RECENT_LOGS_BUFFER) >= SEQUENCE_LENGTH:
                current_sequence_vectors = np.array([item['vector'] for item in list(RECENT_LOGS_BUFFER)[-SEQUENCE_LENGTH:]]).reshape(1, SEQUENCE_LENGTH, 100)
                reconstructed_sequence = lstm_model.predict(current_sequence_vectors, verbose=0)
                mae = np.mean(np.abs(current_sequence_vectors - reconstructed_sequence), axis=(1, 2))
                
                if mae[0] > ANOMALY_THRESHOLD:
                    original_logs = [item['original_log'] for item in list(RECENT_LOGS_BUFFER)[-SEQUENCE_LENGTH:]]
                    
                    summary = summarize_locally(original_logs, word2vec_model)
                    
                    new_anomaly = {
                        "score": float(round(mae[0], 4)), 
                        "summary": summary,
                        "sequence": original_logs,
                        "detected_at": datetime.datetime.now().isoformat()
                    }
                    FLAGGED_ANOMALIES.append(new_anomaly)
                    FLAGGED_ANOMALIES.sort(key=lambda x: x['score'], reverse=True)
                    FLAGGED_ANOMALIES[:] = FLAGGED_ANOMALIES[:10]

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"ERROR: Exception in receive_logs: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# --- UI and Control Endpoints ---
@app.route('/ui')
def ui():
    return render_template('index.html')

def do_stop_collecting():
    global COLLECTION_TIMER, COLLECTION_END_TIME
    if os.path.exists(COLLECTING_STATE_FILE): 
        os.remove(COLLECTING_STATE_FILE)
        print("INFO: Collection stopped.", flush=True)
    if COLLECTION_TIMER:
        COLLECTION_TIMER.cancel()
        COLLECTION_TIMER = None
    COLLECTION_END_TIME = None

@app.route('/collect/start', methods=['POST'])
def start_collecting():
    global COLLECTION_TIMER, COLLECTION_END_TIME
    try:
        if COLLECTION_TIMER: COLLECTION_TIMER.cancel()
        data = request.get_json(force=True) or {}
        duration_seconds = data.get('duration_seconds', 0)
        
        RECENTLY_COLLECTED_LOGS.clear()
        
        if os.path.exists(MONITORING_STATE_FILE): os.remove(MONITORING_STATE_FILE)
        with open(COLLECTING_STATE_FILE, 'w') as f: f.write(datetime.datetime.now().isoformat())

        if duration_seconds > 0:
            COLLECTION_END_TIME = datetime.datetime.now() + datetime.timedelta(seconds=duration_seconds)
            COLLECTION_TIMER = threading.Timer(duration_seconds, do_stop_collecting)
            COLLECTION_TIMER.start()
            return jsonify({"status": "collection_started", "duration": f"{duration_seconds} seconds", "end_time_iso": COLLECTION_END_TIME.isoformat()})
        
        COLLECTION_END_TIME = None
        return jsonify({"status": "collection_started", "duration": "manual"})
    except Exception as e:
        print(f"ERROR in /collect/start: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/collect/stop', methods=['POST'])
def stop_collecting():
    do_stop_collecting()
    return jsonify({"status": "collection_stopped"})

@app.route('/collect/clear', methods=['POST'])
def clear_corpus():
    global collected_logs_corpus, FLAGGED_ANOMALIES, RECENTLY_COLLECTED_LOGS
    collected_logs_corpus.clear()
    FLAGGED_ANOMALIES.clear()
    RECENTLY_COLLECTED_LOGS.clear()
    print("INFO: In-memory corpus, anomalies, and recent logs cleared.", flush=True)
    return jsonify({"status": "corpus_and_anomalies_cleared"})

@app.route('/collect/recent', methods=['GET'])
def get_recent_logs():
    return jsonify({"recent_logs": list(RECENTLY_COLLECTED_LOGS)})

@app.route('/monitoring/start', methods=['POST'])
def start_monitoring():
    if not lstm_model: return jsonify({"status": "error", "message": "LSTM Model not trained yet."}), 400
    if os.path.exists(COLLECTING_STATE_FILE): os.remove(COLLECTING_STATE_FILE)
    RECENT_LOGS_BUFFER.clear()
    with open(MONITORING_STATE_FILE, 'w') as f: f.write(datetime.datetime.now().isoformat())
    return jsonify({"status": "monitoring_started"})

@app.route('/monitoring/stop', methods=['POST'])
def stop_monitoring():
    if os.path.exists(MONITORING_STATE_FILE): os.remove(MONITORING_STATE_FILE)
    return jsonify({"status": "monitoring_stopped"})

@app.route('/monitoring/results', methods=['GET'])
def get_monitoring_results():
    return jsonify({"flagged_anomalies": FLAGGED_ANOMALIES})

# --- NEW Endpoint to clear specific anomalies ---
@app.route('/monitoring/clear_anomalies', methods=['POST'])
def clear_anomalies():
    global FLAGGED_ANOMALIES
    data = request.get_json()
    if not data or 'anomalies_to_clear' not in data:
        return jsonify({"status": "error", "message": "Missing list of anomalies to clear."}), 400
    
    anomalies_to_clear = set(data['anomalies_to_clear'])
    
    with MODEL_LOCK:
        # Rebuild the list, excluding the ones to be cleared
        FLAGGED_ANOMALIES[:] = [
            anomaly for anomaly in FLAGGED_ANOMALIES 
            if anomaly['detected_at'] not in anomalies_to_clear
        ]
    
    print(f"INFO: Cleared {len(anomalies_to_clear)} anomalies.", flush=True)
    return jsonify({"status": "success", "cleared_count": len(anomalies_to_clear)})


# --- Training Endpoints ---
@app.route('/train', methods=['POST'])
def train_w2v_model():
    global word2vec_model
    if not collected_logs_corpus: return jsonify({"status": "error", "message": "No logs in corpus to train on."}), 400
    with MODEL_LOCK:
        print("INFO: Starting Word2Vec model training...", flush=True)
        model = Word2Vec(sentences=collected_logs_corpus, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
        word2vec_model = model
        print("INFO: Word2Vec model training complete.", flush=True)
    
    report, vector_sample_word, vector_sample_data, similar_sample_word, similar_sample_data = {}, "N/A", "N/A", "N/A", "N/A"
    report['corpus_sample'] = [" ".join(log) for log in collected_logs_corpus[:5]]
    report['model_stats'] = {"vocabulary_size": len(word2vec_model.wv.key_to_index), "logs_trained_on": len(collected_logs_corpus)}
    report['model_samples'] = {}
    for word in ['unhealthy', 'failedscheduling', 'backoff', 'failed', 'error']:
        if word in word2vec_model.wv: vector_sample_word, vector_sample_data = word, word2vec_model.wv[word].tolist()[:10]; break
    report['model_samples'][f'vector_for_word_{vector_sample_word}'] = vector_sample_data
    for word in ['pod', 'container', 'deployment', 'job', 'cronjob']:
        if word in word2vec_model.wv: similar_sample_word, similar_sample_data = word, word2vec_model.wv.most_similar(word, topn=5); break
    report['model_samples'][f'similar_to_word_{similar_sample_word}'] = similar_sample_data
    return jsonify({"status": "success", "message": "Word2Vec model training complete.", "training_report": report})

@app.route('/train/lstm', methods=['POST'])
def train_lstm_model():
    global lstm_model, ANOMALY_THRESHOLD
    if not word2vec_model: return jsonify({"status": "error", "message": "You must train Word2Vec model first."}), 400

    data = request.get_json() or {}
    epochs = data.get('epochs', 20)
    batch_size = data.get('batch_size', 32)
    threshold_percentile = data.get('threshold_percentile', 95)

    try:
        epochs = int(epochs)
        batch_size = int(batch_size)
        threshold_percentile = int(threshold_percentile)
        if not (0 < threshold_percentile <= 100):
            raise ValueError("Threshold percentile must be between 1 and 100.")
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid training parameters provided."}), 400

    SEQUENCE_LENGTH, VECTOR_SIZE = 10, 100
    log_vectors = [get_vector_for_log(log, word2vec_model, VECTOR_SIZE) for log in collected_logs_corpus]
    sequences = create_sequences(log_vectors, SEQUENCE_LENGTH)
    if len(sequences) < 20: return jsonify({"status": "error", "message": f"Not enough logs to create sequences for LSTM training. Need at least {SEQUENCE_LENGTH + 1} distinct log entries."}), 400
    
    with MODEL_LOCK:
        print(f"INFO: Starting LSTM model training with epochs={epochs}, batch_size={batch_size}...", flush=True)
        inputs = Input(shape=(SEQUENCE_LENGTH, VECTOR_SIZE)); encoded = LSTM(64, activation='relu')(inputs)
        decoded = RepeatVector(SEQUENCE_LENGTH)(encoded); decoded = LSTM(VECTOR_SIZE, activation='sigmoid', return_sequences=True)(decoded)
        autoencoder = Model(inputs, decoded); autoencoder.compile(optimizer='adam', loss='mae')
        
        autoencoder.fit(sequences, sequences, epochs=epochs, batch_size=batch_size, verbose=0)
        
        reconstructed = autoencoder.predict(sequences, verbose=0)
        mae_losses = np.mean(np.abs(sequences - reconstructed), axis=(1, 2))
        
        ANOMALY_THRESHOLD = np.percentile(mae_losses, threshold_percentile) 
        lstm_model = autoencoder
        print(f"INFO: LSTM model trained. Anomaly threshold set to {ANOMALY_THRESHOLD:.4f} at {threshold_percentile}% percentile.", flush=True)

    report = {
        "message": f"LSTM model trained. Anomaly threshold set to {ANOMALY_THRESHOLD:.4f}", 
        "sequences_trained_on": len(sequences),
        "training_parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold_percentile": threshold_percentile
        }
    }
    return jsonify({"status": "success", "training_report": report})

# --- Download / Upload Endpoints ---
@app.route('/corpus/download', methods=['GET'])
def download_corpus():
    if not collected_logs_corpus:
        print("WARN: Download requested for empty corpus. Sending empty list.", flush=True)

    try:
        mem_file = io.BytesIO()
        mem_file.write(json.dumps(collected_logs_corpus, indent=2).encode('utf-8'))
        mem_file.seek(0)
        return send_file(mem_file, mimetype='application/json', as_attachment=True, download_name='corpus.json')
    except Exception as e:
        print(f"ERROR in /corpus/download: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/models/download', methods=['GET'])
def download_models():
    if not all([word2vec_model, lstm_model]):
        return jsonify({"status": "error", "message": "Both models must be trained before downloading."}), 400

    mem_zip = io.BytesIO()
    try:
        with MODEL_LOCK, tempfile.TemporaryDirectory() as tmpdir:
            w2v_path = os.path.join(tmpdir, "w2v.model")
            lstm_path = os.path.join(tmpdir, "lstm.keras") 
            threshold_path = os.path.join(tmpdir, "threshold.json")
            
            word2vec_model.save(w2v_path)
            lstm_model.save(lstm_path)
            with open(threshold_path, 'w') as f: json.dump({'threshold': ANOMALY_THRESHOLD}, f)
            
            with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(w2v_path, os.path.basename(w2v_path))
                zf.write(lstm_path, os.path.basename(lstm_path))
                zf.write(threshold_path, os.path.basename(threshold_path))

    except Exception as e:
        print(f"ERROR during model zipping: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500
        
    mem_zip.seek(0)
    return send_file(mem_zip, mimetype='application/zip', as_attachment=True, download_name='models.zip')

@app.route('/corpus/load', methods=['POST'])
def upload_corpus():
    global collected_logs_corpus
    if 'corpus_upload_file' not in request.files: return jsonify({"status": "error", "message": "No file part in the request."}), 400
    file = request.files['corpus_upload_file']
    if file.filename == '': return jsonify({"status": "error", "message": "No selected file."}), 400
    
    try:
        corpus_data = json.load(file)
        if not isinstance(corpus_data, list): raise ValueError("Uploaded file is not a valid JSON list.")
        
        with MODEL_LOCK: collected_logs_corpus = corpus_data
        print(f"INFO: Corpus loaded from '{file.filename}'. Total logs: {len(collected_logs_corpus)}", flush=True)
        return jsonify({"status": "success", "message": f"Corpus loaded from {file.filename}", "logs_loaded": len(collected_logs_corpus)})
    except Exception as e:
        print(f"ERROR in /corpus/load: {e}", flush=True)
        return jsonify({"status": "error", "message": f"Failed to load corpus: {e}"}), 500

@app.route('/models/load', methods=['POST'])
def upload_models():
    global word2vec_model, lstm_model, ANOMALY_THRESHOLD
    if 'model_upload_file' not in request.files: return jsonify({"status": "error", "message": "No file part in the request."}), 400
    file = request.files['model_upload_file']
    if file.filename == '': return jsonify({"status": "error", "message": "No selected file."}), 400

    try:
        with tempfile.TemporaryDirectory() as tmpdir, zipfile.ZipFile(file, 'r') as zf:
            zip_contents = zf.namelist()
            print(f"DEBUG: Contents of uploaded zip: {zip_contents}", flush=True)
            
            if not ("w2v.model" in zip_contents and "threshold.json" in zip_contents and "lstm.keras" in zip_contents):
                 return jsonify({"status": "error", "message": "ZIP file is missing required model components (w2v.model, lstm.keras, threshold.json)."}), 400
            
            zf.extractall(tmpdir)

            with MODEL_LOCK:
                word2vec_model = Word2Vec.load(os.path.join(tmpdir, "w2v.model"))
                lstm_model = tf.keras.models.load_model(os.path.join(tmpdir, "lstm.keras"))
                with open(os.path.join(tmpdir, "threshold.json"), 'r') as f: ANOMALY_THRESHOLD = json.load(f)['threshold']
        
        print(f"INFO: Models loaded from '{file.filename}'.", flush=True)
        return jsonify({"status": "success", "message": f"Model set loaded from '{file.filename}'."})
    except Exception as e:
        print(f"ERROR in /models/load: {e}", flush=True)
        return jsonify({"status": "error", "message": f"Failed to load models: {e}"}), 500

# --- Status Endpoint ---
@app.route('/status', methods=['GET'])
def get_status():
    """Provides the current status of the application to the frontend."""
    is_collecting = os.path.exists(COLLECTING_STATE_FILE)
    is_monitoring = os.path.exists(MONITORING_STATE_FILE)
    
    collection_end_time_iso = COLLECTION_END_TIME.isoformat() if is_collecting and COLLECTION_END_TIME else None

    status = {
        "status": "ok",
        "is_collecting": is_collecting,
        "collection_end_time": collection_end_time_iso,
        "is_monitoring": is_monitoring,
        "logs_in_corpus": len(collected_logs_corpus),
        "word2vec_model_status": "Trained" if word2vec_model else "Untrained",
        "lstm_model_status": "Trained" if lstm_model else "Untrained",
        "anomalies_flagged": len(FLAGGED_ANOMALIES)
    }
    return jsonify(status)

if __name__ == '__main__':
    if not os.path.exists(STATE_DIR): os.makedirs(STATE_DIR)
    print("INFO: Starting Waitress server on http://0.0.0.0:5000", flush=True)
    serve(app, host='0.0.0.0', port=5000)
