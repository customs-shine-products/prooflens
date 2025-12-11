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
# Get API KEY from environment variables (secure)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- MEMORY SYSTEMS ---
# 1. The Cache: Stores results so we don't ask Google twice
# Format: { "hashed_prompt": "cached_response_text" }
response_cache = {}

# 2. The Queue: Ensures we never hit rate limits
request_queue = deque()
processing_lock = threading.Lock()
last_request_time = 0
RATE_LIMIT_DELAY = 4.0 # Seconds between calls to Google

def process_queue():
    """Background worker that processes requests one by one with delays"""
    global last_request_time
    while True:
        if request_queue:
            # Get next task
            task = request_queue[0]
            
            # Rate Limit Governor
            time_since_last = time.time() - last_request_time
            if time_since_last < RATE_LIMIT_DELAY:
                time.sleep(RATE_LIMIT_DELAY - time_since_last)

            # Execute
            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(task['prompt'])
                
                # Save to Cache
                response_cache[task['hash']] = response.text
                
                # Mark as done (in a real app, you'd use webhooks or polling)
                # For this simple version, we block the HTTP request until done
                task['result'] = response.text
                task['event'].set() # Wake up the waiting HTTP thread
                
            except Exception as e:
                print(f"API Error: {e}")
                task['error'] = str(e)
                task['event'].set()
            
            finally:
                last_request_time = time.time()
                request_queue.popleft() # Remove from queue
                
        else:
            time.sleep(0.1) # Sleep if idle

# Start the background worker
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # 1. Check Cache
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_hash in response_cache:
        print("Cache Hit! Returning instant result.")
        return jsonify({"result": response_cache[prompt_hash], "cached": True})

    # 2. Add to Queue
    event = threading.Event()
    task = {
        'prompt': prompt,
        'hash': prompt_hash,
        'event': event,
        'result': None,
        'error': None
    }
    
    request_queue.append(task)
    
    # 3. Wait for Worker to finish this specific task
    # This keeps the connection open until the "Conveyor Belt" reaches this item
    event.wait(timeout=60) # Timeout after 60s
    
    if task['error']:
        return jsonify({"error": task['error']}), 500
    
    if task['result']:
        return jsonify({"result": task['result'], "cached": False})
    
    return jsonify({"error": "Timeout or Queue overload"}), 503

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)