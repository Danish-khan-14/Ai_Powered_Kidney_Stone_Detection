"""
KidneyStone AI - AI-powered kidney stone detection (Flask).

Runs locally with `python app.py` and on Vercel (serverless).

Secrets are read from environment variables (see .env.example):
    SECRET_KEY, FIREBASE_WEB_API_KEY, EMAIL_ADDRESS, EMAIL_PASSWORD,
    FIREBASE_SERVICE_ACCOUNT_JSON (Vercel) or serviceAccountKey.json (local).
"""
import os

# ==================== ENVIRONMENT ====================
# Load a local .env file first (skipped if python-dotenv is not installed).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

IS_VERCEL = bool(os.environ.get('VERCEL'))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# These must be set BEFORE TensorFlow is imported.
if IS_VERCEL:
    # Vercel's disk is read-only except /tmp (Keras downloads VGG16 weights here).
    os.environ.setdefault('KERAS_HOME', '/tmp/.keras')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import hashlib
import hmac
import io
import json
import re
import secrets
import smtplib
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps

import firebase_admin
import numpy as np
import requests
from firebase_admin import auth, credentials, db
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   session, url_for)
from flask_cors import CORS
from PIL import Image
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename


def log(message):
    """print() that never crashes on consoles that cannot show emoji."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(str(message).encode('ascii', 'replace').decode('ascii'), flush=True)


# ==================== FLASK APP ====================
# Vercel rejects request bodies over 4.5 MB before Flask sees them, so stay below that there.
MAX_UPLOAD_MB = 4 if IS_VERCEL else 16

app = Flask(__name__)

_secret_key = os.environ.get('SECRET_KEY')
if not _secret_key:
    log("⚠️ SECRET_KEY is not set - using a random key. Users will be logged out "
        "on every restart. Set SECRET_KEY in .env (local) or Vercel settings.")
    _secret_key = secrets.token_hex(32)

app.config['SECRET_KEY'] = _secret_key
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'jfif'}
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = IS_VERCEL  # HTTPS only when deployed
CORS(app)

# ==================== SETTINGS ====================
IMG_SIZE = 224
FIREBASE_WEB_API_KEY = os.environ.get('FIREBASE_WEB_API_KEY', '')
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS', '')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', '')
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', '587'))

DATABASE_URL = os.environ.get(
    'FIREBASE_DATABASE_URL',
    'https://ai-powered-kidney-stone-default-rtdb.asia-southeast1.firebasedatabase.app')
STORAGE_BUCKET = os.environ.get(
    'FIREBASE_STORAGE_BUCKET', 'ai-powered-kidney-stone.firebasestorage.app')

MODEL_URL = os.environ.get(
    'MODEL_URL',
    'https://github.com/Danish-khan-14/Ai_Powered_Kidney_Stone_Detection/'
    'releases/download/v1.0/best_model.keras')
MODEL_H5_URL = os.environ.get(
    'MODEL_H5_URL',
    'https://github.com/Danish-khan-14/Ai_Powered_Kidney_Stone_Detection/'
    'releases/download/v1.0/best_model.h5')
# Cloud instances use temporary storage for downloaded model assets.
MODEL_DIR = os.environ.get(
    'MODEL_DIR',
    '/tmp/model' if (IS_VERCEL or os.environ.get('RENDER')) else os.path.join(BASE_DIR, 'model'))
MODEL_SEARCH_DIRS = list(dict.fromkeys([os.path.join(BASE_DIR, 'model'), MODEL_DIR]))
MIN_MODEL_BYTES = 1024 * 1024  # anything smaller is not a real model file

CODE_TTL_SECONDS = 10 * 60   # verification code lifetime
MAX_CODE_ATTEMPTS = 5
PENDING_KEY = 'pending_registration'
DETECTION_ID_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


# ==================== FIREBASE ====================
DB_ENABLED = False  # True only when Firebase initialised and the database answers


def _load_firebase_credentials():
    """Service account from the FIREBASE_SERVICE_ACCOUNT_JSON variable (Vercel),
    otherwise from serviceAccountKey.json next to this file (local)."""
    raw = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON', '').strip()
    if raw:
        return credentials.Certificate(json.loads(raw))
    return credentials.Certificate(os.path.join(BASE_DIR, 'serviceAccountKey.json'))


def _check_db(attempts=2):
    """Cheap connectivity test; disables DB features if Firebase is unreachable."""
    global DB_ENABLED
    last_error = None
    for _ in range(attempts):
        try:
            db.reference('_connectivity_check').get()  # reads a tiny, empty node
            log("✅ Realtime Database reachable")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    log(f"⚠️ Realtime DB unreachable ({last_error}); disabling DB features")
    DB_ENABLED = False


def _init_firebase():
    global DB_ENABLED
    try:
        firebase_admin.get_app()
    except ValueError:
        try:
            firebase_admin.initialize_app(_load_firebase_credentials(), {
                'databaseURL': DATABASE_URL,
                'storageBucket': STORAGE_BUCKET,
            })
            log("✅ Firebase Admin initialized successfully!")
        except Exception as exc:
            log(f"❌ Firebase initialization failed: {type(exc).__name__}: {exc}")
            log("⚠️ Continuing without Firebase...")
            return
    DB_ENABLED = True
    _check_db()


_init_firebase()


# ==================== MODEL (loaded on first use) ====================
# TensorFlow is imported lazily: pages open instantly and the heavy model
# loads on the first prediction (or in the background when /upload is opened).
_model = None
_model_state = 'pending'  # pending | loading | ready | failed
_model_error = ''
_model_failed_at = 0.0
_model_lock = threading.RLock()
_predict_lock = threading.Lock()
MODEL_RETRY_SECONDS = 60  # wait this long before retrying after a failure


def _find_local_model(filename):
    for directory in MODEL_SEARCH_DIRS:
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and os.path.getsize(path) >= MIN_MODEL_BYTES:
            return path
    return None


def _download_model(filename, url):
    """Download one model asset from the configured GitHub Release URL."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    destination = os.path.join(MODEL_DIR, filename)
    partial = destination + '.part'
    log(f"⬇️ Downloading {filename} from GitHub Releases ...")
    try:
        with requests.get(url, stream=True, timeout=(15, 300)) as response:
            response.raise_for_status()
            with open(partial, 'wb') as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if os.path.getsize(partial) < MIN_MODEL_BYTES:
            raise ValueError(f'downloaded {filename} is too small to be a valid model')
        os.replace(partial, destination)
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
    log(f"✅ {filename} downloaded")
    return destination


def _model_candidates():
    """Yield available/downloadable models, preferring the HDF5 release asset."""
    candidates = (
        ('best_model.h5', MODEL_H5_URL),
        ('best_model.keras', MODEL_URL),
    )
    for filename, url in candidates:
        path = _find_local_model(filename)
        if path:
            yield path
            continue
        try:
            yield _download_model(filename, url)
        except Exception as exc:
            log(f"⚠️ Could not obtain {filename}: {type(exc).__name__}: {exc}")

def _archive_config_is_corrupt(path):
    """Detects the known 'dense_4 has several inbound nodes' config bug."""
    try:
        with zipfile.ZipFile(path) as archive:
            config = json.loads(archive.read('config.json').decode('utf-8'))
    except Exception:
        return False
    for layer in config.get('config', {}).get('layers', []):
        if layer.get('config', {}).get('name') == 'dense_4':
            nodes = layer.get('inbound_nodes')
            if not nodes or len(nodes) != 1:
                return True
    return False


def _build_architecture(keras, base_weights='imagenet'):
    """VGG16 base + the trained classifier head (layer names match the archive)."""
    layers = keras.layers
    base = keras.applications.VGG16(
        weights=base_weights, include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu', name='dense_4')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', name='dense_5')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', name='dense_6')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid', name='dense_7')(x)
    return keras.Model(inputs=base.input, outputs=x)


def _rebuild_from_weights(path, keras, base_weights='imagenet'):
    """Rebuild the network and copy the trained classifier weights out of the
    .keras archive. Raises if any classifier layer could not be restored, so the
    app never makes predictions with random weights."""
    import h5py

    log("🔧 Attempting manual reconstruction from weights (imagenet base)")
    architecture = _build_architecture(keras, base_weights)
    name_map = {
        'dense': 'dense_4',
        'dense_1': 'dense_5',
        'dense_2': 'dense_6',
        'dense_3': 'dense_7',
    }
    restored = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path) as archive:
            archive.extract('model.weights.h5', tmpdir)
        with h5py.File(os.path.join(tmpdir, 'model.weights.h5'), 'r') as weights_file:
            layers_group = weights_file['layers']
            for source, target in name_map.items():
                if source not in layers_group:
                    continue
                variables = layers_group[source]['vars']
                kernel = variables['0'][()]  # dataset '0' is the kernel
                bias = variables['1'][()]    # dataset '1' is the bias
                architecture.get_layer(name=target).set_weights([kernel, bias])
                restored += 1
                log(f"   loaded weights for {target}")
    if restored != len(name_map):
        raise RuntimeError(
            f"only {restored} of {len(name_map)} classifier layers could be restored")
    log("✅ Manual model built; top-layer weights assigned")
    return architecture


def _load_model_blocking():
    log("\\n" + "=" * 50)
    log("🚀 KIDNEY STONE DETECTION API - loading model")
    log("=" * 50)
    import keras  # heavy import (loads TensorFlow), done only when a prediction is needed

    last_error = None
    for path in _model_candidates():
        log(f"🔄 Attempting to load: {path}")
        loaded = None

        # HDF5 is tried first because it stores the complete model weights.
        if path.endswith('.keras') and _archive_config_is_corrupt(path):
            log("⚠️ Keras archive configuration is corrupt; trying weight reconstruction")
        else:
            try:
                loaded = keras.models.load_model(path, compile=False)
            except Exception as exc:
                last_error = exc
                log(f"⚠️ Standard model load failed for {os.path.basename(path)}: {type(exc).__name__}: {exc}")

        # The .keras release has a known config issue; retain the project's
        # manual reconstruction path as a fallback for that archive only.
        if loaded is None and path.endswith('.keras'):
            try:
                loaded = _rebuild_from_weights(path, keras)
            except Exception as exc:
                last_error = exc
                log(f"⚠️ Keras weight reconstruction failed: {type(exc).__name__}: {exc}")

        if loaded is not None:
            log(f"✅ Model ready: {os.path.basename(path)}")
            log("=" * 50 + "\\n")
            return loaded

    if last_error:
        raise RuntimeError(f'Could not load either released model: {type(last_error).__name__}: {last_error}')
    raise FileNotFoundError('Neither best_model.h5 nor best_model.keras could be downloaded from GitHub Releases')

def get_model():
    """Returns the model, loading it on first call. Returns None if loading failed
    (a failed load is retried after MODEL_RETRY_SECONDS)."""
    global _model, _model_state, _model_error, _model_failed_at
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if (_model_state == 'failed'
                and time.time() - _model_failed_at < MODEL_RETRY_SECONDS):
            return None
        _model_state = 'loading'
        try:
            loaded = _load_model_blocking()
            error = ''
        except Exception as exc:
            loaded = None
            error = f"{type(exc).__name__}: {exc}"
        if loaded is None:
            _model_state = 'failed'
            _model_error = error
            _model_failed_at = time.time()
            log(f"❌ Model failed to load: {error}")
            return None
        _model = loaded
        _model_state = 'ready'
        _model_error = ''
        return _model


def model_available():
    """True when the model is loaded or can still be loaded; passed to the
    templates as `model_loaded`."""
    if _model is not None:
        return True
    if (_model_state == 'failed'
            and time.time() - _model_failed_at < MODEL_RETRY_SECONDS):
        return False
    return True


def start_model_warmup():
    """Start loading the model in the background (no-op when loaded/loading)."""
    if _model is None and _model_state != 'loading':
        threading.Thread(target=get_model, daemon=True).start()


# ==================== EMAIL ====================
def send_verification_email(recipient_email, code):
    """Send the verification code to the user's email. Returns True on success."""
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        log("❌ Email error: EMAIL_ADDRESS / EMAIL_PASSWORD are not configured")
        return False
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = recipient_email
        msg['Subject'] = "KidneyStone AI - Email Verification Code"

        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          color: white; padding: 20px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .code {{ font-size: 36px; font-weight: bold; color: #667eea;
                        text-align: center; padding: 20px; background: white;
                        border-radius: 10px; margin: 20px 0; letter-spacing: 5px; }}
                .footer {{ text-align: center; margin-top: 20px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 KidneyStone AI</h1>
                </div>
                <div class="content">
                    <h2>Welcome to KidneyStone AI!</h2>
                    <p>Your verification code is:</p>
                    <div class="code">{code}</div>
                    <p>This code will expire in <strong>10 minutes</strong>.</p>
                    <p>If you didn't request this, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>KidneyStone AI - AI-Powered Kidney Stone Detection</p>
                    <p>© {datetime.now().year} All rights reserved</p>
                </div>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=20) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        log(f"✅ Verification email sent to {recipient_email}")
        return True
    except Exception as exc:
        log(f"❌ Email error: {type(exc).__name__}: {exc}")
        return False


# ==================== HELPERS ====================
def _wants_json():
    return request.path == '/predict' or request.path.startswith('/api/')


def login_required(view):
    @wraps(view)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if _wants_json():
                return jsonify({'error': 'Please log in first'}), 401
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return ('.' in filename
            and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])


def _normalize_email(value):
    return (value or '').strip().lower()


def _code_digest(email, code):
    """Keyed hash of the code, so the code itself is never stored in the cookie."""
    message = f"{email}|{code}".encode('utf-8')
    return hmac.new(app.config['SECRET_KEY'].encode('utf-8'), message,
                    hashlib.sha256).hexdigest()


def preprocess_image(data):
    """Turn uploaded image bytes into a (1, 224, 224, 3) float array, or None."""
    try:
        # Pillow handles the supported upload formats without the large OpenCV
        # native binaries, keeping the deployment smaller.
        with Image.open(io.BytesIO(data)) as source:
            img = source.convert('RGB').resize(
                (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
            array = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)
    except Exception as exc:
        log(f"Error preprocessing: {type(exc).__name__}: {exc}")
        return None


def _new_user_record(name, email):
    return {
        'name': name,
        'email': email,
        'created_at': datetime.now().isoformat(),
        'total_detections': 0,
        'stones_detected': 0,
        'normal_results': 0,
    }


def save_detection_to_firebase(user_id, filename, result, confidence):
    """Save a detection result to Firebase. Returns the detection id or None."""
    if not DB_ENABLED:
        log("⚠️ Realtime DB disabled, skipping save_detection_to_firebase")
        return None
    try:
        user_ref = db.reference(f'users/{user_id}')
        if user_ref.get() is None:
            user_ref.set(_new_user_record(
                session.get('user_name', 'User'), session.get('user_email', '')))
            log(f"✅ Created user node for {user_id}")

        detection_id = str(uuid.uuid4())[:8]
        user_ref.child('detections').child(detection_id).set({
            'id': detection_id,
            'filename': filename,
            'result': result,
            'confidence': float(confidence),
            'result_text': 'Stone Detected' if result == 1 else 'Normal',
            'timestamp': datetime.now().isoformat(),
        })
        log(f"✅ Detection saved with ID: {detection_id}")

        user_ref.child('total_detections').transaction(lambda current: (current or 0) + 1)
        counter = 'stones_detected' if result == 1 else 'normal_results'
        user_ref.child(counter).transaction(lambda current: (current or 0) + 1)
        return detection_id
    except Exception as exc:
        log(f"⚠️ Firebase save failed: {exc}")
        return None


def get_user_detections(user_id):
    """The user's detection history, newest first (empty list if unavailable)."""
    if not DB_ENABLED:
        return []
    try:
        user_ref = db.reference(f'users/{user_id}')
        if user_ref.get() is None:
            user_ref.set(_new_user_record(
                session.get('user_name', 'User'), session.get('user_email', '')))
            return []

        detections = user_ref.child('detections').get()
        if not detections:
            return []
        items = []
        for key, value in detections.items():
            value['firebase_key'] = key
            items.append(value)
        items.sort(key=lambda item: item.get('timestamp', ''), reverse=True)
        return items
    except Exception as exc:
        log(f"⚠️ Error retrieving detections: {exc}")
        return []


# ==================== AUTH ROUTES ====================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = _normalize_email(request.form.get('email'))
        password = request.form.get('password') or ''

        if not email or not password:
            flash('Please enter your email and password', 'danger')
            return render_template('login.html')
        if not FIREBASE_WEB_API_KEY:
            log("❌ FIREBASE_WEB_API_KEY is not configured")
            flash('Login is not configured on the server', 'danger')
            return render_template('login.html')

        try:
            response = requests.post(
                'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword',
                params={'key': FIREBASE_WEB_API_KEY},
                json={'email': email, 'password': password, 'returnSecureToken': True},
                timeout=15)
        except requests.RequestException as exc:
            # only the class name is logged: the message would contain the API key
            log(f"Login request failed: {type(exc).__name__}")
            flash('Could not reach the login service. Please try again.', 'danger')
            return render_template('login.html')

        if response.status_code == 200:
            try:
                user = auth.get_user(response.json()['localId'])
            except Exception as exc:
                log(f"Login lookup failed: {type(exc).__name__}: {exc}")
                flash('Login failed. Please try again.', 'danger')
                return render_template('login.html')

            session.clear()
            session['user_id'] = user.uid
            session['user_email'] = user.email or email
            session['user_name'] = user.display_name or email.split('@')[0]
            flash('Login successful!', 'success')
            return redirect(url_for('index'))

        try:
            error_code = response.json().get('error', {}).get('message', '')
        except ValueError:
            error_code = ''
        if 'USER_DISABLED' in error_code:
            flash('This account has been disabled', 'danger')
        elif 'TOO_MANY_ATTEMPTS' in error_code:
            flash('Too many attempts. Please try again later.', 'danger')
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = (request.form.get('name') or '').strip()[:100]
        email = _normalize_email(request.form.get('email'))

        if not name or '@' not in email:
            flash('Please enter your name and a valid email address', 'danger')
            return render_template('register.html')

        # Is this email already registered?
        try:
            auth.get_user_by_email(email)
            flash('Email already registered. Please login.', 'warning')
            return redirect(url_for('login'))
        except auth.UserNotFoundError:
            pass
        except ValueError:
            flash('Please enter a valid email address', 'danger')
            return render_template('register.html')
        except Exception as exc:
            log(f"Email lookup failed: {type(exc).__name__}: {exc}")
            flash('Could not check your email right now. Please try again.', 'danger')
            return render_template('register.html')

        # 6-digit code. It is kept (as a keyed hash) in the user's own session
        # cookie, so it works across serverless instances.
        code = str(secrets.randbelow(900000) + 100000)
        session[PENDING_KEY] = {
            'email': email,
            'name': name,
            'digest': _code_digest(email, code),
            'expires': time.time() + CODE_TTL_SECONDS,
            'attempts': 0,
        }

        if send_verification_email(email, code):
            flash(f'Verification code sent to {email}', 'success')
            return redirect(url_for('verify', email=email))
        session.pop(PENDING_KEY, None)
        flash('Failed to send verification email. Please try again.', 'danger')

    return render_template('register.html')


@app.route('/verify/<email>', methods=['GET', 'POST'])
def verify(email):
    email = _normalize_email(email)

    if request.method == 'POST':
        code = (request.form.get('code') or '').strip()
        password = request.form.get('password') or ''
        pending = session.get(PENDING_KEY)

        if not pending or pending.get('email') != email:
            flash('No verification code found. Please register again.', 'danger')
        elif time.time() > pending['expires']:
            session.pop(PENDING_KEY, None)
            flash('Invalid or expired verification code', 'danger')
        elif pending['attempts'] >= MAX_CODE_ATTEMPTS:
            session.pop(PENDING_KEY, None)
            flash('Too many incorrect attempts. Please register again.', 'danger')
        elif not hmac.compare_digest(pending['digest'], _code_digest(email, code)):
            pending['attempts'] += 1
            session[PENDING_KEY] = pending
            flash('Invalid or expired verification code', 'danger')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long', 'danger')
        else:
            try:
                user = auth.create_user(
                    email=email, password=password, display_name=pending['name'])
            except auth.EmailAlreadyExistsError:
                session.pop(PENDING_KEY, None)
                flash('This email was registered by another account while you were '
                      'verifying.\nPlease try logging in or restart registration.', 'danger')
            except Exception as exc:
                flash(f'Registration failed: {exc}', 'danger')
            else:
                if DB_ENABLED:
                    try:
                        db.reference(f'users/{user.uid}').set(
                            _new_user_record(pending['name'], email))
                    except Exception as db_err:
                        # a database error must not fail the whole registration
                        log(f"⚠️ failed to write user to RTDB: {db_err}")
                        flash('Registration succeeded but realtime database is '
                              'unavailable.\nSome features may be missing.', 'warning')
                session.pop(PENDING_KEY, None)
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))

    return render_template('verify.html', email=email)


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))


# ==================== MAIN ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_available())


@app.route('/dashboard')
@login_required
def dashboard():
    detections = get_user_detections(session['user_id'])
    total = len(detections)
    stones = sum(1 for d in detections if d.get('result_text') == 'Stone Detected')
    return render_template('dashboard.html',
                           user_name=session.get('user_name', 'User'),
                           total_detections=total,
                           stones_detected=stones,
                           normal_results=total - stones,
                           detections=detections[:10],
                           model_loaded=model_available())


@app.route('/upload')
@login_required
def upload():
    """Upload page for image analysis."""
    if not model_available():
        flash('AI model is not loaded. Please contact administrator.', 'danger')
        return redirect(url_for('dashboard'))
    start_model_warmup()  # load the model while the user picks an image
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG files.'}), 400

        data = file.read()  # the image is processed in memory, nothing is saved to disk
        if not data:
            return jsonify({'error': 'The uploaded file is empty'}), 400

        filename = secure_filename(file.filename) or 'scan.' + file.filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"

        started = time.time()
        processed_image = preprocess_image(data)
        if processed_image is None:
            return jsonify({'error': 'Invalid image format. Please upload a valid medical image.'}), 400
        preprocess_seconds = time.time() - started

        current_model = get_model()  # may take a while on the first request
        if current_model is None:
            return jsonify({'error': 'Model not loaded. Please try again in a minute or contact the administrator.'}), 503

        started = time.time()
        with _predict_lock:
            prediction = current_model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        processing_time = round(preprocess_seconds + (time.time() - started), 2)

        result = 1 if confidence > 0.5 else 0
        confidence_percent = confidence * 100 if result == 1 else (1 - confidence) * 100

        if result == 1:
            result_text = "Stone Detected"
            result_class = "danger"
            icon = "⚠️"
            description = f"""The AI model has detected indications of kidney stones in the ultrasound image with {confidence_percent:.1f}% confidence.

Medical Recommendations:
• Consult with a nephrologist or urologist immediately
• Additional imaging tests may be recommended
• Drink plenty of water (2-3 liters daily unless contraindicated)
• Monitor for symptoms like flank pain, blood in urine
• Share this report with your healthcare provider

This result is an AI screening aid, not a medical diagnosis."""
        else:
            result_text = "No Stone Detected"
            result_class = "success"
            icon = "✅"
            description = f"""The ultrasound image appears normal with no signs of kidney stones ({confidence_percent:.1f}% confidence).

Recommendations:
• Regular checkups are still recommended
• Maintain healthy lifestyle with proper hydration
• Limit salt and animal protein intake
• If symptoms persist, consult a doctor
• Continue regular medical checkups

This result is an AI screening aid, not a medical diagnosis."""

        # Save to Firebase if available
        detection_id = None
        try:
            detection_id = save_detection_to_firebase(
                session['user_id'], unique_filename, result, confidence)
        except Exception as exc:
            log(f"Firebase save failed: {exc}")

        session['last_result'] = {
            'result': result,
            'result_text': result_text,
            'confidence': confidence,
            'confidence_percent': confidence_percent,
            'description': description,
            'result_class': result_class,
            'icon': icon,
            'processing_time': processing_time,
            'detection_id': detection_id,
            'filename': unique_filename,
        }

        return jsonify({
            'success': True,
            'result': result,
            'result_text': result_text,
            'confidence': confidence,
            'confidence_percent': confidence_percent,
            'description': description,
            'result_class': result_class,
            'icon': icon,
            'processing_time': processing_time,
            'detection_id': detection_id,
            'redirect': url_for('results'),
        })

    except HTTPException:
        raise  # e.g. 413 file too large: let the error handlers answer it
    except Exception as exc:
        log(f"Error in prediction: {type(exc).__name__}: {exc}")
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


@app.route('/results')
@login_required
def results():
    """Display prediction results."""
    last_result = session.get('last_result')
    if not last_result:
        flash('No recent prediction found', 'warning')
        return redirect(url_for('upload'))
    return render_template('results.html', result=last_result)


@app.route('/history')
@login_required
def history():
    detections = get_user_detections(session['user_id'])
    return render_template('history.html', detections=detections,
                           model_loaded=model_available())


@app.route('/api/history')
@login_required
def api_history():
    return jsonify(get_user_detections(session['user_id']))


@app.route('/detection/<detection_id>')
@login_required
def view_detection(detection_id):
    detection = None
    if DB_ENABLED and DETECTION_ID_PATTERN.match(detection_id):
        try:
            detection = db.reference(
                f"users/{session['user_id']}/detections/{detection_id}").get()
        except Exception as exc:
            log(f"⚠️ Error loading detection: {exc}")

    if detection:
        return render_template('detection_detail.html', detection=detection,
                               model_loaded=model_available())
    flash('Detection not found', 'danger')
    return redirect(url_for('history'))


@app.route('/profile')
@login_required
def profile():
    user_id = session['user_id']
    try:
        user = auth.get_user(user_id)
        user_data = {}
        if DB_ENABLED:
            user_data = db.reference(f'users/{user_id}').get() or {}
        return render_template('profile.html', user=user, user_data=user_data,
                               model_loaded=model_available())
    except Exception as exc:
        flash(f'Error loading profile: {exc}', 'danger')
        return redirect(url_for('dashboard'))


@app.route('/health')
def health():
    authenticated = 'user_id' in session
    user_name = ''
    user_email = ''
    stats = {}

    if authenticated:
        user_name = session.get('user_name', '')
        user_email = session.get('user_email', '')
        try:
            if DB_ENABLED:
                user_data = db.reference(f"users/{session['user_id']}").get() or {}
            else:
                user_data = {}
            stats = {
                'total_detections': user_data.get('total_detections', 0),
                'stones_detected': user_data.get('stones_detected', 0),
                'normal_results': user_data.get('normal_results', 0),
            }
        except Exception:
            stats = {'total_detections': 0, 'stones_detected': 0, 'normal_results': 0}

    available = model_available()
    return jsonify({
        'status': 'healthy' if available else 'degraded',
        'model_loaded': available,
        'model_state': _model_state,
        'authenticated': authenticated,
        'user_name': user_name,
        'user_email': user_email,
        'stats': stats,
        'timestamp': datetime.now().isoformat(),
    })


# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found_error(error):
    if _wants_json():
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html', model_loaded=model_available()), 404


@app.errorhandler(500)
def internal_error(error):
    if _wants_json():
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html', model_loaded=model_available()), 500


@app.errorhandler(413)
def too_large_error(error):
    message = f'File too large. Maximum size is {MAX_UPLOAD_MB} MB.'
    if _wants_json():
        return jsonify({'error': message}), 413
    flash(message, 'danger')
    return redirect(url_for('upload'))


# ==================== RUN LOCALLY ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG') == '1'
    host = os.environ.get('HOST', '127.0.0.1')  # use HOST=0.0.0.0 to open it on your Wi-Fi
    # Load the model at start-up (once - not in the debug reloader's parent process).
    if not debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        get_model()
    app.run(host=host, port=port, debug=debug)
