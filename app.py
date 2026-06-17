import os
import shutil

# --- WINDOWS SYMLINK FIX (No Admin Required) ---
if hasattr(os, 'symlink'):
    _orig_symlink = os.symlink


    def _safe_symlink(src, dst, target_is_directory=False, **kwargs):
        try:
            _orig_symlink(src, dst, target_is_directory=target_is_directory, **kwargs)
        except OSError as e:
            if getattr(e, 'winerror', None) == 1314:  # Privilege not held
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                raise


    os.symlink = _safe_symlink
# -----------------------------------------------

import streamlit as st
import tempfile
import numpy as np
import json
from db_manager import DatabaseManager
from face_module import FaceBiometricModule
from voice_module import VoiceBiometricModule

# Suppress TensorFlow logs for a clean terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Initialization ---
st.set_page_config(page_title="Biometric PoC", layout="wide")


@st.cache_resource
def load_modules():
    # This caches the AI models so they don't reload on every click
    return FaceBiometricModule(), VoiceBiometricModule()


face_module, voice_module = load_modules()

# --- Session State ---
if 'target_uuid' not in st.session_state:
    st.session_state['target_uuid'] = None

# --- UI Layout ---
st.title("🧬 Biometric Authentication Pipeline")
st.markdown("Proof of Concept: Independent Modality Processing & Database Routing")

st.divider()

# The "Architecture Flex" Dropdown
modality = st.radio("Select Active Biometric Modality:",
                    ["Face Recognition (ArcFace + MTCNN)", "Acoustic Voice (SpeechBrain X-Vector)"])
active_module = face_module if "Face" in modality else voice_module
file_type = ["jpg", "jpeg", "png"] if "Face" in modality else ["wav", "mp3", "m4a"]

st.divider()

col1, col2 = st.columns(2)

# --- PHASE 1: ENROLLMENT ---
with col1:
    st.header("1. Baseline Enrollment")
    st.write("Register a new user identity.")

    enroll_file = st.file_uploader("Upload Enrollment Sample", type=file_type, key="enroll")

    if st.button("Register Identity") and enroll_file:
        with st.spinner("Processing biometric data..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{enroll_file.name.split('.')[-1]}") as t:
                t.write(enroll_file.read())
                temp_path = t.name

            db = DatabaseManager(database="biometric_system")
            try:
                random_id = db.create_person(national_id=str(np.random.randint(100000, 999999)), full_name="Demo User")

                cleaned_bytes = active_module.process_raw(temp_path)
                vector = active_module.extract_vector(cleaned_bytes)

                db.save_raw_data(random_id, active_module.method_name, enroll_file.name.split('.')[-1], cleaned_bytes)
                db.save_vector(random_id, active_module.method_name, "enrollment", vector.tolist())

                st.session_state['target_uuid'] = random_id
                st.success(f"Identity Enrolled Successfully!\n\n**UUID:** `{random_id}`")

            except Exception as e:
                st.error(f"System Error: {e}")
            finally:
                db.close()
                os.unlink(temp_path)

# --- PHASE 2: VERIFICATION ---
with col2:
    st.header("2. Live Authentication")
    st.write("Compare a live sample against the database.")

    if st.session_state['target_uuid'] is None:
        st.info("👈 Please register a baseline identity first.")
    else:
        st.info(f"**Target Identity:** `{st.session_state['target_uuid']}`")
        verify_file = st.file_uploader("Upload Live Sample", type=file_type, key="verify")

        if st.button("Authenticate") and verify_file:
            with st.spinner("Analyzing biometric signature..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{verify_file.name.split('.')[-1]}") as t:
                    t.write(verify_file.read())
                    temp_path = t.name

                db = DatabaseManager(database="biometric_system")
                try:
                    cleaned_bytes = active_module.process_raw(temp_path)
                    live_vector = active_module.extract_vector(cleaned_bytes)

                    db.cursor.execute(
                        "SELECT vector_512 FROM Method_retrieval_vectors WHERE random_id = %s AND method = %s",
                        (st.session_state['target_uuid'], active_module.method_name)
                    )
                    result = db.cursor.fetchone()

                    if result and result['vector_512']:
                        saved_vector = np.array(json.loads(result['vector_512']))

                        scores = active_module.compare_vectors(saved_vector, live_vector)

                        st.write(f"**Cosine Similarity:** `{scores['cosine_similarity']:.4f}`")
                        st.write(f"**Euclidean Distance:** `{scores['euclidean_distance']:.4f}`")

                        # Dynamic threshold based on active module
                        threshold = 0.35 if "Face" in modality else 0.80

                        if scores['cosine_similarity'] > threshold:
                            st.success("✅ AUTHENTICATION SUCCESSFUL: Identity Confirmed.")
                            st.balloons()
                        else:
                            st.error("❌ AUTHENTICATION FAILED: Vectors do not match.")
                    else:
                        st.error("Database Error: No baseline found for this specific modality.")

                except Exception as e:
                    st.error(f"System Error: {e}")
                finally:
                    db.close()
                    os.unlink(temp_path)