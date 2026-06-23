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

    def process_raw(self, file_path):
        """
        Reads the image, downscales it to prevent OOM,
        and re-encodes it back into raw bytes for the extraction engine.
        """
        # 1. Read the raw image using OpenCV
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"System Error: OpenCV could not read the image at {file_path}. File may be corrupted.")

        # 2. The Defense Mechanism: Clamp maximum resolution to 1080 pixels
        max_dimension = 1080
        height, width = img.shape[:2]

        if max(height, width) > max_dimension:
            scaling_factor = max_dimension / float(max(height, width))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # 3. RE-ENCODE: Compress the shrunken image back into standard JPEG bytes
        success, buffer = cv2.imencode('.jpg', img)
        if not success:
            raise ValueError("System Error: OpenCV failed to re-encode the image buffer.")

        # Return raw bytes exactly as the original architecture expected
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