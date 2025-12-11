from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import time
import hashlib
import os
from collections import deque
import threading

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- MEMORY SYSTEMS ---
response_cache = {}
request_queue = deque()
last_request_time = 0
RATE_LIMIT_DELAY = 4.5  # Increased slightly to be safe
MAX_QUEUE_SIZE = 15     # Fail fast if line is too long

def process_queue():
    """Background worker that processes requests one by one with delays"""
    global last_request_time
    while True:
        if request_queue:
            task = request_queue[0]
            
            # Rate Limit Governor
            time_since_last = time.time() - last_request_time
            if time_since_last < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - time_since_last)

            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                # Safety check for empty prompts
                if not task['prompt']:
                    task['error'] = "Empty prompt"
                else:
                    response = model.generate_content(task['prompt'])
                    response_cache[task['hash']] = response.text
                    task['result'] = response.text
                
            except Exception as e:
                print(f"API Error: {e}")
                # Handle quota limits specifically
                if "429" in str(e):
                    time.sleep(10) # Extra cool down
                task['error'] = str(e)
            
            finally:
                last_request_time = time.time()
                # Signal completion
                if not task['event'].is_set():
                    task['event'].set()
                
                # Remove from queue safely
                if request_queue:
                    request_queue.popleft()
        else:
            time.sleep(0.1)

# Start background thread
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return "ProofLens Backend is Active!", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Input Validation
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    prompt = data.get('prompt', '')
    
    # 2. Check Cache
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_hash in response_cache:
        return jsonify({"result": response_cache[prompt_hash], "cached": True})

    # 3. Queue Management
    # If queue is full, return 503 immediately (Better UX than timeout)
    if len(request_queue) >= MAX_QUEUE_SIZE:
        return jsonify({"error": "Server is busy. Please try again in a moment."}), 503

    # 4. Add to Queue
    event = threading.Event()
    task = { 
        'prompt': prompt, 
        'hash': prompt_hash, 
        'event': event, 
        'result': None, 
        'error': None 
    }
    
    request_queue.append(task)
    
    # 5. Wait for Processing
    # We wait up to 110s (just under the 120s Gunicorn limit we set)
    is_done = event.wait(timeout=110)
    
    if not is_done:
        # If we timed out, we might still be in the queue. 
        # Ideally we'd remove ourselves, but for simplicity we just error out.
        return jsonify({"error": "Request timed out waiting for AI."}), 504
        
    if task['error']:
        return jsonify({"error": task['error']}), 500
        
    return jsonify({"result": task['result'], "cached": False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
