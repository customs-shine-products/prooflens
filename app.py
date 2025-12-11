from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import time
import hashlib
import os
from collections import deque
import threading

app = Flask(__name__)
CORS(app)  # Allow your GitHub page to talk to this server

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- MEMORY SYSTEMS ---
response_cache = {}
request_queue = deque()
last_request_time = 0
RATE_LIMIT_DELAY = 4.0 

def process_queue():
    """Background worker that processes requests one by one with delays"""
    global last_request_time
    while True:
        if request_queue:
            task = request_queue[0]
            time_since_last = time.time() - last_request_time
            if time_since_last < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - time_since_last)

            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(task['prompt'])
                response_cache[task['hash']] = response.text
                task['result'] = response.text
                task['event'].set()
            except Exception as e:
                print(f"API Error: {e}")
                task['error'] = str(e)
                task['event'].set()
            finally:
                last_request_time = time.time()
                request_queue.popleft()
        else:
            time.sleep(0.1)

worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    """Health check route to prevent 404s on the root URL"""
    return "ProofLens Backend is Active and Running!", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_hash in response_cache:
        return jsonify({"result": response_cache[prompt_hash], "cached": True})

    event = threading.Event()
    task = { 'prompt': prompt, 'hash': prompt_hash, 'event': event, 'result': None, 'error': None }
    
    request_queue.append(task)
    
    # Wait max 60s for the queue to process this task
    if event.wait(timeout=60):
        if task['error']:
            return jsonify({"error": task['error']}), 500
        return jsonify({"result": task['result'], "cached": False})
    
    return jsonify({"error": "Timeout: Server is busy"}), 503

if __name__ == '__main__':
    # Use PORT provided by Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
