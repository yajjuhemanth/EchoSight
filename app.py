# Added 'request' to the import from flask
# Added 'io' import
import threading
import time
import logging
from queue import Queue, Empty
import concurrent.futures
import io # <-- ADDED Import

import speech_recognition as sr
import cv2
import numpy as np
import pyttsx3
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
# Make sure request is imported from Flask
from flask import Flask, render_template, request, jsonify

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Global Variables & Initialization ---

# Hearing Mode (variables unchanged)
hearing_stop_event = threading.Event()
hearing_thread = None
hearing_transcript = ""
hearing_lock = threading.Lock()
recognizer = None
hearing_initial_message_active = False

# Visual Mode (variables mostly unchanged)
visual_stop_event = threading.Event()
visual_thread = None # For webcam
tts_thread = None
visual_caption = "Ready for webcam or upload" # Changed initial message
visual_lock = threading.Lock()
tts_queue = Queue()
# Keep executor for potential parallel processing if needed, or single upload handling
visual_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='CaptionExec')
last_caption_time = 0
caption_interval = 4 # Only for continuous webcam mode
visual_mode_is_continuous = True # Default for webcam mode

# Model/Processor/Camera
processor = None
model = None
cap = None # Camera might not be needed if only uploading
# TTS Engine handled by tts_worker

# --- Helper Functions ---
def initialize_hearing():
    # (Hearing initialization code remains the same)
    global recognizer
    if recognizer is None:
        logging.info("Initializing speech recognizer...")
        try:
            recognizer = sr.Recognizer()
            # Balanced Parameters
            recognizer.energy_threshold = 400
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.7
            recognizer.non_speaking_duration = 0.5
            # Mic check might fail if no mic, handle gracefully or make optional
            try:
                with sr.Microphone() as source:
                    logging.debug("Adjusting for ambient noise...")
                    recognizer.adjust_for_ambient_noise(source, duration=0.7)
                    logging.debug("Ambient noise adjustment complete.")
            except Exception as mic_e:
                 logging.warning(f"Microphone check/adjustment failed: {mic_e}. Proceeding without.")
            logging.info("Speech recognizer initialized with balanced settings.")
            return True
        except Exception as e:
            logging.error(f"Recognizer init failed: {e}")
            recognizer = None
            return False
    return True

def initialize_visual(initialize_camera=True): # <-- Added flag
    """Initializes visual components. Camera init is optional."""
    global processor, model, cap, visual_executor
    model_initialized = True
    camera_initialized = True

    if model is None:
        logging.info("Loading visual model (BLIP)...")
        try:
            caption_model_name = "Salesforce/blip-image-captioning-base"
            processor = BlipProcessor.from_pretrained(caption_model_name)
            model = BlipForConditionalGeneration.from_pretrained(caption_model_name).to("cpu").eval()
            logging.info("Visual model loaded.")
        except Exception as e:
             logging.error(f"Model load failed: {e}"); processor = None; model = None; model_initialized = False

    if initialize_camera and cap is None: # <-- Check flag
        logging.info("Initializing webcam...")
        try:
            cap = cv2.VideoCapture(0); assert cap.isOpened(), "Cannot open webcam"
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360); cap.set(cv2.CAP_PROP_FPS, 15)
            time.sleep(1); logging.info("Webcam initialized.")
        except Exception as e:
             logging.error(f"Webcam init failed: {e}"); cap = None; camera_initialized = False
    elif not initialize_camera:
         camera_initialized = False # Explicitly false if not initializing
         logging.info("Skipping camera initialization as requested.")


    if visual_executor._shutdown:
        logging.info("Re-initializing visual executor.")
        visual_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='CaptionExec')

    # Return success based on what was requested/needed
    if initialize_camera:
        return model_initialized and camera_initialized and model is not None and cap is not None
    else:
        # Success if only model was needed and it loaded
        return model_initialized and model is not None


def release_visual_resources():
    global cap; logging.info("Releasing visual resources...")
    if cap:
        cap.release(); cap = None; logging.info("Webcam released.")
    # Keep model loaded unless explicitly told otherwise
    logging.info("Visual resources released (Camera only).")

# Modified to accept PIL Image directly
def generate_caption(image_input):
    """Generates caption for a PIL Image or NumPy array."""
    if model is None or processor is None: return "[Model not loaded]"

    img = None
    try:
        if isinstance(image_input, np.ndarray):
             # Convert NumPy array (from webcam frame) to PIL Image
            img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
             # Use PIL Image directly (from upload)
             # Ensure it's RGB
             if image_input.mode != 'RGB':
                 img = image_input.convert('RGB')
             else:
                 img = image_input
        else:
            logging.error("Invalid input type for generate_caption.")
            return "[Invalid Input Type]"

        inputs = processor(images=img, return_tensors="pt").to("cpu")
    except Exception as e_img:
         logging.error(f"Image processing error: {e_img}"); return "[Image Error]"
    try:
        with torch.no_grad():
             output_ids = model.generate(**inputs, max_length=32, num_beams=3, early_stopping=True)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        if not caption or len(caption.strip()) < 3: return None # Return None for empty/short captions
        return caption
    except Exception as e_gen:
         logging.error(f"Caption generation failed: {e_gen}"); return None # Return None on generation error


# --- Background Worker Threads ---

# Hearing worker remains the same
def hearing_worker():
    # (Hearing worker code remains the same)
    global hearing_transcript, hearing_initial_message_active
    assert recognizer is not None, "Recognizer not initialized"
    logging.info("Hearing worker started.")
    with sr.Microphone() as source:
        while not hearing_stop_event.is_set():
            try:
                logging.debug("Listening for audio chunk...")
                audio = recognizer.listen(source, timeout=1.5, phrase_time_limit=5)
                logging.debug("Audio chunk received, recognizing...")
                text = recognizer.recognize_google(audio, language="en-US")
                logging.info(f"Recognized: {text}")
                with hearing_lock:
                    if hearing_initial_message_active:
                        hearing_transcript = text
                        hearing_initial_message_active = False
                    else:
                        separator = " " if hearing_transcript and not hearing_transcript.endswith((" ", "\n")) else ""
                        hearing_transcript += separator + text
            except sr.WaitTimeoutError: logging.debug("No speech detected."); continue
            except sr.UnknownValueError: logging.debug("Could not understand audio"); continue
            except sr.RequestError as e: logging.error(f"API Error: {e}"); time.sleep(1)
            except OSError as e: logging.error(f"Audio input error: {e}"); time.sleep(1)
            except Exception as e: logging.error(f"Unexpected hearing error: {e}"); time.sleep(1)
    logging.info("Hearing worker stopped.")


# Visual worker (for webcam) remains largely the same
def visual_worker():
    global visual_caption, last_caption_time;
    # Ensure camera is ready IF this worker is started
    if not cap or not cap.isOpened():
        logging.error("Visual worker started but camera is not ready.")
        with visual_lock: visual_caption = "[Error: Camera not ready]"
        return # Stop the worker thread

    logging.info("Visual worker started.")
    while not visual_stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame from webcam.")
            time.sleep(0.5); continue
        if visual_mode_is_continuous:
            current_time = time.time()
            if visual_executor and not visual_executor._shutdown and (current_time - last_caption_time >= caption_interval):
                last_caption_time = current_time
                # Submit NumPy frame
                submit_caption_task(frame.copy())
        time.sleep(0.05) # Small sleep to prevent busy-waiting
    logging.info("Visual worker stopped.")

# Modified to accept PIL Image or NumPy Array
def submit_caption_task(image_data):
    """Submits image data (PIL Image or NumPy array) for captioning."""
    global visual_caption
    if visual_executor and not visual_executor._shutdown:
        future = visual_executor.submit(generate_caption, image_data) # Pass data directly
        try:
            caption_timeout = 25.0 # Slightly longer timeout for potentially larger uploads
            caption = future.result(timeout=caption_timeout)
        except concurrent.futures.TimeoutError:
             logging.warning(f"Caption generation timed out.")
             with visual_lock: visual_caption = "[Caption timed out]"
             return
        except Exception as e:
             logging.error(f"Caption result error: {e}")
             with visual_lock: visual_caption = "[Caption generation error]"
             return

        # Process valid caption
        if caption:
            logging.info(f"Generated Caption: {caption}")
            with visual_lock:
                visual_caption = caption
            if tts_thread and tts_thread.is_alive():
                tts_queue.put(caption)
            else:
                logging.warning("TTS thread not running or not initialized.")
        else:
            # Handle cases where generate_caption returns None (e.g., empty caption, error)
            logging.warning("Caption generation returned None or empty.")
            # Optionally update visual_caption to indicate no meaningful caption
            with visual_lock:
                 if visual_caption not in ["[Caption generation error]", "[Caption timed out]"]: # Avoid overwriting specific errors
                    visual_caption = "[No caption generated]"

    else:
        logging.warning("Visual executor not available or shut down.")
        with visual_lock:
             visual_caption = "[Executor not ready]"


# TTS worker remains the same
def tts_worker():
    # (TTS worker code remains the same)
    logging.info("TTS worker started.")
    while not visual_stop_event.is_set():
        caption = None; tts_engine = None
        try:
            caption = tts_queue.get(timeout=0.5)
            logging.info(f"TTS Worker got: '{caption}'")
            try:
                logging.debug("Initializing TTS engine...")
                tts_engine = pyttsx3.init()
                if tts_engine:
                    logging.debug(f"Speaking: '{caption}'")
                    tts_engine.say(caption)
                    tts_engine.runAndWait()
                    logging.debug("Finished speaking.")
                else:
                    logging.warning("TTS init failed.")
            except RuntimeError as e: logging.error(f"TTS runtime error: {e}")
            except Exception as e: logging.error(f"TTS execution error: {e}")
            finally:
                 if tts_engine is not None:
                     try:
                        # Attempt to stop if runAndWait didn't finish cleanly
                        # Note: pyttsx3 cleanup can be tricky
                        # del tts_engine might be enough on some systems
                        pass
                     except Exception as te:
                        logging.error(f"Error during TTS engine cleanup: {te}")
                     finally:
                        del tts_engine # Ensure it's deleted
                        logging.debug("Cleaned up TTS instance.")
            tts_queue.task_done()
        except Empty: continue
        except Exception as e:
             logging.error(f"Error in tts_worker loop: {e}")
             try:
                 if caption is not None: tts_queue.task_done()
             except (ValueError, UnboundLocalError): pass
             time.sleep(0.5)
    logging.info("TTS Worker stopping. Clearing queue.")
    while not tts_queue.empty():
         try:
             tts_queue.get_nowait()
             tts_queue.task_done()
         except (Empty, ValueError): break
    logging.info("TTS queue cleared.")


# --- Flask Routes & API Endpoints ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/hearing')
def hearing_mode(): return render_template('hearing.html')
@app.route('/visual')
def visual_mode():
     # Ensure model is loaded when accessing the visual page, even if camera isn't used initially
     # We don't need the camera just to load the page or for uploads.
     if not initialize_visual(initialize_camera=False): # Only ensure model loads
        logging.error("Visual mode page: Failed to initialize model.")
        # Maybe render an error template or return an error message
        # For now, let it proceed, but captioning won't work.
        pass
     return render_template('visual.html')

# --- Hearing Routes (Unchanged) ---
@app.route('/hearing/start', methods=['POST'])
def start_hearing():
    # (start_hearing code remains the same)
    global hearing_thread, hearing_transcript, hearing_initial_message_active
    if hearing_thread and hearing_thread.is_alive(): return jsonify({"status": "already_running"})
    if not initialize_hearing(): return jsonify({"status": "error", "message": "Mic init failed."}), 500
    logging.info("Starting hearing mode...")
    hearing_stop_event.clear()
    with hearing_lock:
        hearing_transcript = "(Waiting for speech...)"
        hearing_initial_message_active = True
    hearing_thread = threading.Thread(target=hearing_worker, daemon=True, name="HearingWorker")
    hearing_thread.start()
    return jsonify({"status": "started"})

@app.route('/hearing/stop', methods=['POST'])
def stop_hearing():
    # (stop_hearing code remains the same)
    global hearing_thread, hearing_initial_message_active
    logging.info("Stopping hearing mode...")
    hearing_stop_event.set()
    if hearing_thread and hearing_thread.is_alive():
        hearing_thread.join(timeout=2.0)
    hearing_thread = None
    with hearing_lock:
        hearing_transcript = "Stopped."
        hearing_initial_message_active = False
    logging.info("Hearing mode stopped.")
    return jsonify({"status": "stopped"})

@app.route('/hearing/transcript', methods=['GET'])
def get_transcript():
    # (get_transcript code remains the same)
    with hearing_lock:
        transcript = hearing_transcript
    return jsonify({"transcript": transcript})


# --- Visual Routes (Modified/Added) ---

@app.route('/visual/start', methods=['POST'])
def start_visual():
    """Starts webcam capture and processing"""
    global visual_thread, tts_thread, visual_caption
    if (visual_thread and visual_thread.is_alive()):
        return jsonify({"status": "already_running", "message": "Webcam already running."})

    # Initialize visual components INCLUDING camera this time
    if not initialize_visual(initialize_camera=True):
        with visual_lock: visual_caption = "[Error: Visual Init failed.]"
        return jsonify({"status": "error", "message": "Visual (camera) init failed."}), 500

    # Start TTS thread if not already running (might be running from a previous upload)
    if not tts_thread or not tts_thread.is_alive():
        visual_stop_event.clear() # Ensure stop event is clear for TTS too
        tts_thread = threading.Thread(target=tts_worker, daemon=True, name="TTSWorker")
        tts_thread.start()
        logging.info("TTS Thread started for visual mode.")
    else:
        logging.info("TTS Thread already running.")


    logging.info("Starting visual mode (Webcam)...")
    visual_stop_event.clear() # Clear stop event specifically for visual worker
    with visual_lock: visual_caption = "Starting Webcam..."
    global visual_mode_is_continuous
    logging.info(f"Visual mode starting in {'Continuous' if visual_mode_is_continuous else 'Single Frame'} mode.")

    # Start the webcam processing thread
    visual_thread = threading.Thread(target=visual_worker, daemon=True, name="VisualWorker")
    visual_thread.start()

    return jsonify({"status": "started", "mode": "webcam"})

@app.route('/visual/stop', methods=['POST'])
def stop_visual():
    """Stops webcam capture and TTS"""
    global visual_thread, tts_thread; logging.info("Stopping visual mode (Webcam and TTS)...")
    visual_stop_event.set() # Signal all related threads to stop

    threads_to_join = []
    if tts_thread and tts_thread.is_alive(): threads_to_join.append(tts_thread)
    if visual_thread and visual_thread.is_alive(): threads_to_join.append(visual_thread)

    for t in threads_to_join:
        logging.debug(f"Joining {t.name}...")
        t.join(timeout=3.0)
        if t.is_alive(): logging.warning(f"{t.name} didn't stop gracefully.")

    visual_thread = None
    tts_thread = None
    release_visual_resources() # Release camera
    with visual_lock: visual_caption = "Stopped."
    logging.info("Visual mode stopped.")
    return jsonify({"status": "stopped"})

@app.route('/visual/caption', methods=['GET'])
def get_caption():
    # (get_caption code remains the same)
    caption = None
    with visual_lock:
        caption = visual_caption
    return jsonify({"caption": caption})

@app.route('/visual/set_mode/<mode>', methods=['POST'])
def set_visual_mode(mode):
     # (set_visual_mode code remains the same, affects webcam mode)
    global visual_mode_is_continuous
    msg = ""
    if mode == 'continuous':
        visual_mode_is_continuous = True
        msg = "Continuous"
    elif mode == 'single':
        visual_mode_is_continuous = False
        msg = "Single Frame"
    else:
        return jsonify({"status": "error", "message": "Invalid mode"}), 400
    logging.info(f"Visual (Webcam) mode set to {msg}.")
    return jsonify({"status": f"set_{mode}"})

@app.route('/visual/capture_single', methods=['POST'])
def capture_single_frame():
    """Captures a single frame from the webcam"""
    global visual_caption
    if visual_mode_is_continuous: return jsonify({"status": "error", "message": "Not in single frame webcam mode"}), 400
    if not cap or not cap.isOpened(): return jsonify({"status": "error", "message": "Camera not ready"}), 500

    logging.info("Attempting single frame capture from webcam...")
    ret, frame = cap.read()

    if not ret:
        logging.error("Failed webcam capture.")
        with visual_lock: visual_caption = "[Webcam Capture failed]"
        return jsonify({"status": "error", "message": "Webcam capture failed"}), 500
    else:
        with visual_lock: visual_caption = "Processing single webcam frame..."
        submit_caption_task(frame.copy()) # Submit NumPy frame
        return jsonify({"status": "capture_submitted"})


# --- NEW ROUTE FOR IMAGE UPLOAD ---
@app.route('/visual/upload', methods=['POST'])
def upload_image():
    """Handles image upload and generates caption"""
    global visual_caption

    # Ensure model is loaded (camera not needed)
    if not model or not processor:
        if not initialize_visual(initialize_camera=False):
             logging.error("Upload handler: Visual model not initialized.")
             return jsonify({"status": "error", "message": "Visual model not loaded."}), 500

    # Start TTS thread if not running (needed to speak the result)
    # Use a separate stop event maybe? Or rely on visual_stop_event clearing?
    # For simplicity, let's start it if it's not alive. User can stop it via main stop button.
    global tts_thread
    if not tts_thread or not tts_thread.is_alive():
        visual_stop_event.clear() # Ensure stop event is clear for TTS too
        tts_thread = threading.Thread(target=tts_worker, daemon=True, name="TTSWorker")
        tts_thread.start()
        logging.info("TTS Thread started for upload caption.")
    else:
         logging.info("TTS Thread already running for upload caption.")


    if 'imageFile' not in request.files:
        return jsonify({"status": "error", "message": "No image file part"}), 400
    file = request.files['imageFile']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected image file"}), 400

    if file:
        try:
            # Read image file stream into PIL Image
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))

            logging.info(f"Received image upload: {file.filename}, size: {len(img_bytes)} bytes")
            with visual_lock:
                visual_caption = "Processing uploaded image..."

            # Submit PIL image directly
            submit_caption_task(img)

            # Note: submit_caption_task runs generate_caption in background thread.
            # The result will be updated in visual_caption and fetched by the frontend poll.
            # We return success here, frontend will show "Processing..." then update.
            return jsonify({"status": "upload_submitted", "filename": file.filename})

        except Exception as e:
            logging.error(f"Error processing uploaded image: {e}")
            with visual_lock:
                visual_caption = "[Error processing upload]"
            return jsonify({"status": "error", "message": "Error processing image"}), 500

    return jsonify({"status": "error", "message": "Unknown upload error"}), 500
# --- END NEW ROUTE ---


if __name__ == '__main__':
    # Disable debug mode for production/threading stability
    # Use host='0.0.0.0' to make accessible on local network if needed, else 127.0.0.1
    app.run(debug=False, host='0.0.0.0', port=5000)
