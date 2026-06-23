import os
import shutil
import streamlit as st

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

from fusion_engine import BiometricFusionEngine

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

st.set_page_config(page_title="Forensic Biometric Search", layout="wide")


@st.cache_resource
def load_engine():
    return BiometricFusionEngine()


engine = load_engine()

st.title("🌐 Global Biometric Identification System (1-to-N)")
st.markdown("Forensic Search Engine: Upload a probe to identify unknown subjects across the database.")
st.divider()

# UI Routing Dictionary
MODALITY_MAP = {
    "Face Recognition (ArcFace + MTCNN)": {"key": "FACE", "ext": ["jpg", "jpeg", "png"]},
    "Acoustic Voice (WavLM-SV)": {"key": "VOICE", "ext": ["wav"]},
    "Gait Dynamics (MediaPipe)": {"key": "GAIT", "ext": ["mp4", "mov", "avi"]}
}

selection = st.radio("Select Target Sensor Data:", list(MODALITY_MAP.keys()), horizontal=True)
active_key = MODALITY_MAP[selection]["key"]
allowed_ext = MODALITY_MAP[selection]["ext"]

st.divider()
col1, col2 = st.columns(2)

# --- PHASE 1: DATABASE POPULATION ---
with col1:
    st.header("1. Populate Database")
    st.write("Enroll a known identity into the watchlist.")

    enroll_name = st.text_input("Subject Full Name (e.g., John Doe)")
    national_id = st.text_input("Subject National ID (123456789)")
    enroll_file = st.file_uploader("Upload Baseline Sample", type=allowed_ext, key="enroll")

    if st.button("Add to Watchlist") and enroll_file and enroll_name and national_id:
        with st.spinner("Processing biometric signature..."):
            try:
                # ALL LOGIC DELEGATED TO ENGINE
                result = engine.enroll_identity(
                    national_id=national_id,
                    full_name=enroll_name,
                    raw_bytes=enroll_file.read(),
                    filename=enroll_file.name,
                    modality_key=active_key
                )

                if result["exists"]:
                    st.success(f"Identity Exists: {result['name']}\nID: {national_id}\n\nUUID: `{result['uuid']}`")
                else:
                    st.success(f"Identity Enrolled: {result['name']}\nID: {national_id}\n\nUUID: `{result['uuid']}`")
            except Exception as e:
                st.error(f"System Error: {e}")

# --- PHASE 2: FORENSIC SEARCH (1-to-N) ---
with col2:
    st.header("2. Forensic Search")
    st.write("Upload an unknown probe to scan the database for matches.")

    search_file = st.file_uploader("Upload Unknown Probe", type=allowed_ext, key="search")

    if st.button("Initiate Database Scan") and search_file:
        with st.spinner("Scanning database..."):
            try:
                # ALL LOGIC DELEGATED TO ENGINE
                result = engine.identify_probe(
                    raw_bytes=search_file.read(),
                    filename=search_file.name,
                    modality_key=active_key
                )

                if result["status"] == "empty_db":
                    st.error("The database is currently empty for this modality.")
                else:
                    st.markdown("### 📊 Search Results")
                    st.write(f"Scanned **{result['scanned_count']}** distinct biometric records.")

                    if result["is_match"]:
                        st.success(f"🚨 **MATCH FOUND: {result['person_name']}**")
                        st.write(f"**UUID Profile:** `{result['uuid']}`")
                        st.write(f"**Score:** `{result['top_score']:.4f}`")
                        st.balloons()
                    else:
                        st.warning("⚠️ **NO MATCH FOUND**")
                        st.write("The closest candidate fell outside the security threshold.")
                        st.write(f"**Closest Candidate:** {result['person_name']} (`{result['uuid']}`)")
                        st.write(f"**Score (FAILED):** `{result['top_score']:.4f}`")

            except Exception as e:
                st.error(f"Search Failed: {e}")