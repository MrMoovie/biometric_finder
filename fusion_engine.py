import os
import tempfile
import numpy as np

from db_manager import DatabaseManager
from face_module import FaceBiometricModule
from voice_module import VoiceBiometricModule
from gait_module import GaitBiometricModule

# --- SYSTEM CONFIGURATION STRATEGY ---
# Changing these values instantly swaps the underlying algorithms
# across the entire system without altering the Streamlit UI.
CONFIG = {
    "search_mode": "LINEAR",  # Future Options: "FAISS"
    "aggregation": "SINGLE",  # Future Options: "KNN"
}


class BiometricFusionEngine:
    def __init__(self):
        # 1. Total Isolation: Drop new modules here, and the system auto-adapts.
        self.modules = {
            "FACE": FaceBiometricModule(),
            "VOICE": VoiceBiometricModule(),
            "GAIT": GaitBiometricModule()
        }

        # 2. Thresholds are locked in the backend, not exposed to the UI
        self.thresholds = {
            "FACE": {"value": 1.10, "type": "euclidean", "lower_is_better": True},
            "VOICE": {"value": 0.72, "type": "cosine", "lower_is_better": False},
            "GAIT": {"value": 0.960, "type": "cosine", "lower_is_better": False}
        }

    def enroll_identity(self, national_id: str, full_name: str, raw_bytes: bytes, filename: str, modality_key: str):
        """Processes raw bytes, extracts the vector, and routes to the database."""
        active_module = self.modules.get(modality_key)
        if not active_module:
            raise ValueError(f"Critical Error: Module {modality_key} not loaded in engine.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as t:
            t.write(raw_bytes)
            temp_path = t.name

        db = DatabaseManager(database="biometric_system")
        try:
            random_id, exists = db.create_person(national_id=national_id, full_name=full_name)

            cleaned_bytes = active_module.process_raw(temp_path)
            vector = active_module.extract_vector(cleaned_bytes)

            db.save_raw_data(random_id, modality_key, filename.split('.')[-1], cleaned_bytes)
            db.save_vector(random_id, modality_key, "enrollment", vector.tolist())

            return {"status": "success", "uuid": random_id, "exists": exists, "name": full_name}
        finally:
            db.close()
            os.unlink(temp_path)

    def identify_probe(self, raw_bytes: bytes, filename: str, modality_key: str):
        """1-to-N Search utilizing Min-Pooling to defeat class imbalance."""
        active_module = self.modules.get(modality_key)
        rule = self.thresholds.get(modality_key)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as t:
            t.write(raw_bytes)
            temp_path = t.name

        db = DatabaseManager(database="biometric_system")
        try:
            cleaned_bytes = active_module.process_raw(temp_path)
            probe_vector = active_module.extract_vector(cleaned_bytes)
            database_vectors = db.get_all_vectors(modality_key)

            if not database_vectors:
                return {"status": "empty_db"}

            # --- PHASE 2.1: MIN-POOLING ARCHITECTURE ---
            identity_scores = {}  # Dictionary mapping { UUID : Absolute_Best_Score }

            for record in database_vectors:
                uuid = record['random_id']
                metrics = active_module.compare_vectors(record['vector'], probe_vector)
                score = metrics['euclidean_distance'] if rule["lower_is_better"] else metrics['cosine_similarity']

                # If this is the first photo we've seen of this UUID, save it
                if uuid not in identity_scores:
                    identity_scores[uuid] = score
                else:
                    # If we already have a score, only overwrite if THIS photo is a closer match
                    current_best = identity_scores[uuid]
                    if rule["lower_is_better"] and score < current_best:
                        identity_scores[uuid] = score
                    elif not rule["lower_is_better"] and score > current_best:
                        identity_scores[uuid] = score

            # --- PHASE 2.2: RESOLVE THE WINNING IDENTITY ---
            best_match_id = None
            best_score = 999.0 if rule["lower_is_better"] else -1.0

            # Only compare the absolute best photo from each user
            for uuid, best_user_score in identity_scores.items():
                is_new_best = (best_user_score < best_score) if rule["lower_is_better"] else (
                            best_user_score > best_score)
                if is_new_best:
                    best_score = best_user_score
                    best_match_id = uuid

            # Evaluate against backend threshold
            is_match = (best_score <= rule["value"]) if rule["lower_is_better"] else (best_score >= rule["value"])

            db.cursor.execute("SELECT full_name FROM identity_map WHERE random_id = %s", (best_match_id,))
            person_data = db.cursor.fetchone()
            person_name = person_data['full_name'] if person_data else "Unknown Profile"

            return {
                "status": "complete",
                "scanned_count": len(database_vectors),
                "is_match": is_match,
                "person_name": person_name,
                "uuid": best_match_id,
                "top_score": best_score,
                "confidence_pct": self._normalize_score(best_score, modality_key) * 100
            }

        finally:
            db.close()
            os.unlink(temp_path)

    # --- PHASE 2.3: THE FUSION CORE ---

    def _normalize_score(self, score, modality_key):
        """Converts raw Euclidean/Cosine math into a universal 0.0 to 1.0 confidence gauge."""
        rule = self.thresholds[modality_key]
        if rule["type"] == "euclidean":
            # Euclidean: Distance 0.0 is 100%. Distance at threshold (1.10) is 0%.
            return max(0.0, min(1.0, (rule["value"] - score) / rule["value"]))
        else:
            # Cosine: Distance 1.0 is 100%. Distance at threshold is 0%.
            return max(0.0, min(1.0, (score - rule["value"]) / (1.0 - rule["value"])))

    def evaluate_fusion_matrix(self, identity_results: dict):
        """
        Dynamically redistributes weight if data is missing, and calculates the final fused score.
        Accepts a dictionary of normalized scores: {"FACE": 0.95, "VOICE": 0.80, "GAIT": None}
        """
        base_weights = {"FACE": 0.60, "VOICE": 0.25, "GAIT": 0.15}
        active_weights = {}
        total_active_weight = 0.0

        # 1. Detect which sensors actually provided data
        for mod, score in identity_results.items():
            if score is not None:
                total_active_weight += base_weights[mod]

        if total_active_weight == 0:
            return 0.0

        # 2. Dynamic Weight Redistribution (The Forensic Wildcard)
        for mod, score in identity_results.items():
            if score is not None:
                # If Gait is missing, Face automatically scales up from 60% to ~70%
                active_weights[mod] = base_weights[mod] / total_active_weight

        # 3. Calculate Final Matrix
        final_fused_score = 0.0
        for mod, score in identity_results.items():
            if score is not None:
                final_fused_score += (score * active_weights[mod])

        return final_fused_score