from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import time
import hashlib
import os
import threading
from queue import PriorityQueue, Empty

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- MEMORY SYSTEMS ---
response_cache = {}
# Priority Queue: (priority_number, timestamp, task_dict)
# Priority 1 = High (Grammar, etc.), Priority 5 = Low (Plagiarism chunks)
request_queue = PriorityQueue()
last_request_time = 0
current_delay = 0  # Start with 0 delay (Optimistic)

def process_queue():
    """Smart background worker with adaptive rate limiting"""
    global last_request_time, current_delay
    
    while True:
        try:
            # Get next task (blocks for 0.1s to avoid CPU spin)
            # PriorityQueue returns lowest number first
            priority, _, task = request_queue.get(timeout=0.1)
            
            # Rate Limit Governor (Adaptive)
            now = time.time()
            elapsed = now - last_request_time
            if elapsed < current_delay:
                time.sleep(current_delay - elapsed)

            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                if not task['prompt']:
                    task['error'] = "Empty prompt"
                else:
                    response = model.generate_content(task['prompt'])
                    response_cache[task['hash']] = response.text
                    task['result'] = response.text
                
                # Success! Slowly reduce delay to speed up
                current_delay = max(0, current_delay * 0.8)
                
            except Exception as e:
                err_str = str(e)
                print(f"API Error: {err_str}")
                
                # Handle 429 Rate Limit specifically
                if "429" in err_str:
                    print("Rate limit hit. Engaging backoff.")
                    # drastic backoff
                    current_delay = 10.0 
                    # Re-queue the failed task with same priority
                    # We create a new event so the original waiter doesn't timeout if we can avoid it,
                    # but simplest is just to put it back.
                    # Ideally we sleep here to let API cool down
                    time.sleep(10) 
                    request_queue.put((priority, time.time(), task))
                    continue # Skip marking as done
                else:
                    task['error'] = err_str
            
            finally:
                last_request_time = time.time()
                # Only signal if we didn't re-queue
                if task.get('result') or task.get('error'):
                    task['event'].set()
                
        except Empty:
            continue
        except Exception as e:
            print(f"Worker Error: {e}")

# Start background thread
worker_thread = threading.Thread(target=process_queue, daemon=True)
worker_thread.start()

@app.route('/', methods=['GET'])
def home():
    return "ProofLens Backend is Active (Priority Queue Enabled)!", 200

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "No prompt provided"}), 400
    
    prompt = data.get('prompt', '')
    priority = data.get('priority', 5) # Default to Low priority (5)
    
    # 1. Check Cache
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    if prompt_hash in response_cache:
        return jsonify({"result": response_cache[prompt_hash], "cached": True})

    # 2. Queue Management (Fail fast if overloaded)
    if request_queue.qsize() >= 20:
        return jsonify({"error": "Server busy. Try again in a few seconds."}), 503

    # 3. Add to Priority Queue
    event = threading.Event()
    task = { 
        'prompt': prompt, 
        'hash': prompt_hash, 
        'event': event, 
        'result': None, 
        'error': None 
    }
    
    # Use time.time() as tie-breaker for FIFO within same priority
    request_queue.put((priority, time.time(), task))
    
    # 4. Wait
    # Increased timeout to handle backoff delays
    is_done = event.wait(timeout=115)
    
    if not is_done:
        return jsonify({"error": "Request timed out (Queue too slow)."}), 504
        
    if task['error']:
        status_code = 500
        if "429" in task['error']: status_code = 429
        return jsonify({"error": task['error']}), status_code
        
    return jsonify({"result": task['result'], "cached": False})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
