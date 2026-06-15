import numpy as np
from scipy.spatial import distance
from biometric_interface import BiometricModule


class VoiceBiometricModule(BiometricModule):
    def __init__(self):
        # A placeholder name for the presentation
        self.model_name = "AcousticStub_v1"

    @property
    def method_name(self) -> str:
        return "VOICE"

    def process_raw(self, raw_file_path: str) -> bytes:
        """STUB: Pretends to read an audio file (e.g., .wav) and return bytes."""
        print(f"[{self.method_name}] Loading raw audio file: {raw_file_path}")
        # In the future, this would use librosa or scipy.io.wavfile to clean background noise
        return b"fake_audio_bytes_98765"

    def extract_vector(self, cleaned_data: bytes) -> np.ndarray:
        """STUB: Pretends to extract a 512-D acoustic feature vector."""
        print(f"[{self.method_name}] Extracting 512-D acoustic feature vector using {self.model_name}...")

        # Generate a normalized random 512-D vector for the presentation stub
        vector = np.random.rand(512)
        normalized_vector = vector / np.linalg.norm(vector)
        return normalized_vector

    def compare_vectors(self, vector_a: np.ndarray, vector_b: np.ndarray) -> dict:
        """Compares acoustic vectors using standard math."""
        # Even though the extraction is fake, the math is very real
        euc_dist = distance.euclidean(vector_a, vector_b)
        cos_sim = 1 - distance.cosine(vector_a, vector_b)

        return {
            "euclidean_distance": float(euc_dist),
            "cosine_similarity": float(cos_sim)
        }