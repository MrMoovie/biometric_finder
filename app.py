import os
import shutil
import base64
import json
import threading
import socket
import sys
import streamlit as st
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- WINDOWS SYMLINK FIX ---
if hasattr(os, 'symlink'):
    _orig_symlink = os.symlink


    def _safe_symlink(src, dst, target_is_directory=False, **kwargs):
        try:
            _orig_symlink(src, dst, target_is_directory=target_is_directory, **kwargs)
        except OSError as e:
            if getattr(e, 'winerror', None) == 1314:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                raise


    os.symlink = _safe_symlink

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Helper function to crop the favicon image as a centered circle
def get_circular_icon(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        from PIL import Image, ImageDraw
        img = Image.open(image_path).convert("RGBA")
        
        # Crop to a centered square first
        width, height = img.size
        min_dim = min(width, height)
        
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img_square = img.crop((left, top, right, bottom))
        
        # Create circular mask
        mask = Image.new("L", (min_dim, min_dim), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, min_dim, min_dim), fill=255)
        
        # Apply the circular mask to create transparent corners
        img_circle = Image.new("RGBA", (min_dim, min_dim), (0, 0, 0, 0))
        img_circle.paste(img_square, (0, 0), mask=mask)
        return img_circle
    except Exception as e:
        print(f"Error cropping circular icon: {e}")
        return None

icon_image = get_circular_icon("b.png")
if icon_image:
    st.set_page_config(page_title="Biomatrix Forensic Search", page_icon=icon_image, layout="wide")
else:
    st.set_page_config(page_title="Biomatrix Forensic Search", layout="wide")

_engine = None
_engine_lock = threading.Lock()

def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                # Heavy imports deferred to the background thread!
                from fusion_engine import BiometricFusionEngine
                _engine = BiometricFusionEngine()
    return _engine

# --- BACKGROUND FLASK API SERVER ---
app = Flask(__name__)
CORS(app)


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    try:
        engine = get_engine()
        data = request.json
        national_id = data.get('national_id')
        full_name = data.get('full_name')
        file_data_b64 = data.get('file_data')
        filename = data.get('filename')
        modality_key = data.get('modality_key')

        if not all([national_id, full_name, file_data_b64, filename, modality_key]):
            return jsonify({"error": "Missing required fields"}), 400

        # Decode base64
        raw_bytes = base64.b64decode(file_data_b64)

        # Call fusion engine
        result = engine.enroll_identity(
            national_id=national_id,
            full_name=full_name,
            raw_bytes=raw_bytes,
            filename=filename,
            modality_key=modality_key
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/identify', methods=['POST'])
def api_identify():
    try:
        engine = get_engine()
        data = request.json
        file_data_b64 = data.get('file_data')
        filename = data.get('filename')
        modality_key = data.get('modality_key')

        if not all([file_data_b64, filename, modality_key]):
            return jsonify({"error": "Missing required fields"}), 400

        # Decode base64
        raw_bytes = base64.b64decode(file_data_b64)

        # Call fusion engine
        result = engine.identify_probe(
            raw_bytes=raw_bytes,
            filename=filename,
            modality_key=modality_key
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/identify_combined', methods=['POST'])
def api_identify_combined():
    import tempfile
    import os
    from db_manager import DatabaseManager
    db = None
    try:
        engine = get_engine()
        data = request.json
        files = data.get('files', {})  # expected: {"FACE": {"file_data": "...", "filename": "..."}, ...}

        if not files:
            return jsonify({"error": "No files provided for combined search"}), 400

        db = DatabaseManager(database="biometric_system")

        # Step 1: For each uploaded modality, extract the probe vector and load database vectors
        probes = {}
        db_vectors = {}

        for modality_key, file_info in files.items():
            if not file_info:
                continue
            raw_bytes = base64.b64decode(file_info["file_data"])
            filename = file_info["filename"]

            active_module = engine.modules.get(modality_key)
            if not active_module:
                continue

            # Process and extract probe vector
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as t:
                t.write(raw_bytes)
                temp_path = t.name
            try:
                cleaned_bytes = active_module.process_raw(temp_path)
                probe_vector = active_module.extract_vector(cleaned_bytes)
                probes[modality_key] = probe_vector

                # Load all DB vectors for this modality
                db_vectors[modality_key] = db.get_all_vectors(modality_key)
            finally:
                os.unlink(temp_path)

        if not probes:
            return jsonify({"error": "No valid modalities processed"}), 400

        # Step 2: Fetch all unique persons from the database
        db.cursor.execute("SELECT random_id, full_name FROM Identity_map")
        persons = db.cursor.fetchall()

        if not persons:
            return jsonify({"status": "empty_db"})

        candidate_fused_scores = {}

        for person in persons:
            uuid = person['random_id']
            person_scores = {}  # normalized scores for this person

            # Check all three possible modalities: FACE, VOICE, GAIT
            for modality_key in ["FACE", "VOICE", "GAIT"]:
                if modality_key not in probes:
                    person_scores[modality_key] = None
                    continue

                active_module = engine.modules[modality_key]
                rule = engine.thresholds[modality_key]

                # Find all vectors in DB for this person and this modality
                person_db_vectors = [v for v in db_vectors[modality_key] if v['random_id'] == uuid]

                if not person_db_vectors:
                    person_scores[modality_key] = None
                    continue

                # Compare probe with all enrolled vectors of this person, selecting the best score
                best_score = 999.0 if rule["lower_is_better"] else -1.0

                for record in person_db_vectors:
                    metrics = active_module.compare_vectors(record['vector'], probes[modality_key])
                    score = metrics['euclidean_distance'] if rule["lower_is_better"] else metrics['cosine_similarity']

                    if rule["lower_is_better"] and score < best_score:
                        best_score = score
                    elif not rule["lower_is_better"] and score > best_score:
                        best_score = score

                # Normalize the best score
                norm_score = engine._normalize_score(best_score, modality_key)
                person_scores[modality_key] = norm_score

            # Run the weighted fusion matrix to combine the scores!
            fused_score = engine.evaluate_fusion_matrix(person_scores)
            candidate_fused_scores[uuid] = {
                "person_name": person['full_name'],
                "fused_score": fused_score,
                "modality_scores": person_scores
            }

        if not candidate_fused_scores:
            return jsonify({"status": "empty_db"})

        # Step 3: Select the candidate with the highest fused score
        best_uuid = max(candidate_fused_scores, key=lambda k: candidate_fused_scores[k]["fused_score"])
        best_match = candidate_fused_scores[best_uuid]

        # The candidate is a match if their fused score is above 0.0 (meaning they are within the thresholds)
        is_match = best_match["fused_score"] > 0.0

        # Build individual confidence percentages for reporting
        confidence_details = {}
        for mod, score in best_match["modality_scores"].items():
            if score is not None:
                confidence_details[mod] = f"{score * 100:.2f}%"
            else:
                confidence_details[mod] = "N/A"

        return jsonify({
            "status": "complete",
            "scanned_count": len(persons),
            "is_match": is_match,
            "person_name": best_match["person_name"],
            "uuid": best_uuid,
            "top_score": best_match["fused_score"],
            "confidence_pct": best_match["fused_score"] * 100,
            "confidence_details": confidence_details
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if db:
            db.close()


@app.route('/api/logs', methods=['GET'])
def api_logs():
    from db_manager import DatabaseManager
    db = None
    try:
        db = DatabaseManager(database="biometric_system")
        db.cursor.execute("""
            SELECT i.random_id, i.national_id, i.full_name, 
                   COUNT(m.id) as template_count
            FROM Identity_map i
            LEFT JOIN Method_retrieval_vectors m ON i.random_id = m.random_id
            GROUP BY i.random_id, i.national_id, i.full_name
            ORDER BY i.full_name ASC
        """)
        records = db.cursor.fetchall()
        return jsonify({"status": "success", "records": records})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    finally:
        if db:
            db.close()


@app.route('/api/network', methods=['GET'])
def api_network():
    from db_manager import DatabaseManager
    db = None
    try:
        db = DatabaseManager(database="biometric_system")
        
        # Count vectors per modality
        db.cursor.execute("""
            SELECT method, COUNT(id) as count 
            FROM Method_retrieval_vectors 
            GROUP BY method
        """)
        counts = db.cursor.fetchall()
        
        modality_counts = {"FACE": 0, "VOICE": 0, "GAIT": 0}
        for row in counts:
            m = row['method']
            if m in modality_counts:
                modality_counts[m] = row['count']
                
        # Total identities
        db.cursor.execute("SELECT COUNT(random_id) as count FROM Person_directory")
        total_identities = db.cursor.fetchone()['count']
        
        return jsonify({
            "status": "online",
            "database": "connected",
            "host": "localhost",
            "port": 3306,
            "total_identities": total_identities,
            "modality_counts": modality_counts,
            "latency_ms": 14
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "database": "disconnected",
            "error": str(e)
        }), 500
    finally:
        if db:
            db.close()


# Safe port binder that starts the Flask server exactly once
if not hasattr(sys, "_biometric_flask_server"):
    def get_port_or_free(preferred_port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('localhost', preferred_port))
            s.close()
            return preferred_port
        except OSError:
            s.close()
            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s2.bind(('localhost', 0))
            free_port = s2.getsockname()[1]
            s2.close()
            return free_port

    port = get_port_or_free(5050)
    
    def run_flask():
        # Start pre-warming the engine asynchronously inside the background thread
        threading.Thread(target=get_engine, daemon=True).start()
        try:
            app.run(host='localhost', port=port, debug=False, use_reloader=False)
        except Exception as e:
            print(f"Flask background server error: {e}")
            
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    
    sys._biometric_flask_server = {
        "port": port,
        "thread": t
    }

flask_port = sys._biometric_flask_server["port"]

# --- STREAMLIT FULL-SCREEN WRAPPER ---
# Inject custom CSS and JS to hide all default Streamlit branding and prevent theme flashes
st.markdown("""
<style>
    /* Hide Streamlit header, footer, sidebar, decoration, status widgets, and connection alerts */
    header, 
    footer, 
    [data-testid="stHeader"], 
    [data-testid="stSidebar"], 
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    .stSpinner,
    #connection-status {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
    }
    
    /* Make the app container take up 100% of the viewport and remove margins */
    .appview-container, 
    .main, 
    .stApp,
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"], 
    [data-testid="stMainBlockContainer"], 
    [data-testid="stVerticalBlock"] {
        padding: 0px !important;
        margin: 0px !important;
        max-width: 100% !important;
        width: 100% !important;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    /* Default Streamlit host background to solid white to match light theme loading screen */
    html,
    body, 
    .stApp,
    [data-testid="stApp"],
    .main, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
    }

    /* Dark theme host background override */
    html.dark-theme,
    html.dark-theme body,
    html.dark-theme .stApp,
    html.dark-theme [data-testid="stApp"],
    html.dark-theme .main, 
    html.dark-theme [data-testid="stAppViewContainer"], 
    html.dark-theme [data-testid="stMainBlockContainer"],
    body.dark-theme, 
    body.dark-theme .stApp,
    body.dark-theme [data-testid="stApp"],
    body.dark-theme .main, 
    body.dark-theme [data-testid="stAppViewContainer"], 
    body.dark-theme [data-testid="stMainBlockContainer"] {
        background-color: #000000 !important;
    }
    
    /* Clean up default iframe borders and force full viewport positioning */
    iframe {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        z-index: 999999 !important;
    }
</style>
<script>
    // Read theme from localStorage (shared origin) and apply it to the host page instantly
    try {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            document.documentElement.classList.add('dark-theme');
            document.body.classList.add('dark-theme');
        }
    } catch (e) {
        console.error("Error setting host theme:", e);
    }
</script>
""", unsafe_allow_html=True)

# Load the custom HTML UI
try:
    with open("index_ui.html", "r", encoding="utf-8") as f:
        html_template = f.read()
    
    # Inject the actual Flask port dynamically
    html_content = html_template.replace("{{FLASK_PORT}}", str(flask_port))
    
    # Render the interface in a full-screen iframe
    st.components.v1.html(html_content, height=1000, scrolling=False)
except Exception as e:
    st.error(f"Error loading interface: {e}")