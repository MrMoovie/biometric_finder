import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial import distance
from biometric_interface import BiometricModule


class FaceBiometricModule(BiometricModule):
    def __init__(self):
        # We explicitly use Facenet512 to match our database column (vector_512)
        self.model_name = "ArcFace"

    @property
    def method_name(self) -> str:
        return "FACE"

    def process_raw(self, raw_file_path: str) -> bytes:
        """Reads the image, validates it, and returns bytes for DB storage."""
        print(f"[{self.method_name}] Loading raw file: {raw_file_path}")
        img = cv2.imread(raw_file_path)
        if img is None:
            raise ValueError(f"Could not read image at {raw_file_path}. Check the path.")

        # Encode to bytes for LONGBLOB database storage
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()

    def extract_vector(self, cleaned_data: bytes) -> np.ndarray:
        """Uses DeepFace to extract a 512D vector from the image bytes."""
        print(f"[{self.method_name}] Extracting 512-D vector using {self.model_name}...")

        # Convert DB bytes back to a numpy image matrix for DeepFace
        nparr = np.frombuffer(cleaned_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        try:
            # align=True aligns the eyes, enforce_detection=True fails if no face is found
            result = DeepFace.represent(
                img_path=img,
                model_name=self.model_name,
                detector_backend="mtcnn",
                enforce_detection=True,
                align=True
            )
            embedding = result[0]['embedding']

            # Normalize the vector (standard practice for cosine similarity accuracy)
            vector = np.array(embedding)
            normalized_vector = vector / np.linalg.norm(vector)
            return normalized_vector

        except Exception as e:
            print(f"[{self.method_name}] Extraction failed (No face detected?): {e}")
            # Return a zero vector so the system doesn't crash, but auth will naturally fail
            return np.zeros(512)

    def compare_vectors(self, vector_a: np.ndarray, vector_b: np.ndarray) -> dict:
        """Compares vectors. Handles zero-vectors if extraction failed."""
        if np.all(vector_a == 0) or np.all(vector_b == 0):
            return {"euclidean_distance": 999.0, "cosine_similarity": 0.0}

        euc_dist = distance.euclidean(vector_a, vector_b)
        cos_sim = 1 - distance.cosine(vector_a, vector_b)

        return {
            "euclidean_distance": float(euc_dist),
            "cosine_similarity": float(cos_sim)
        }