import numpy as np
import io
import soundfile as sf
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
from scipy.spatial import distance
from biometric_interface import BiometricModule


class VoiceBiometricModule(BiometricModule):
    def __init__(self):
        self.model_name = "Microsoft WavLM-SV (X-Vector)"
        # Load the true Speaker Verification feature extractor and model
        self.processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = AutoModelForAudioXVector.from_pretrained("microsoft/wavlm-base-plus-sv")

    @property
    def method_name(self) -> str:
        return "VOICE"

    def process_raw(self, raw_file_path: str) -> bytes:
        """Reads the raw file from Streamlit into bytes."""
        print(f"\n[{self.method_name}] Loading raw audio file: {raw_file_path}")
        with open(raw_file_path, "rb") as f:
            return f.read()

    def extract_vector(self, cleaned_data: bytes) -> np.ndarray:
        """Extracts a true biometric X-Vector from the audio waveform."""
        print(f"[{self.method_name}] Extracting 512-D acoustic vector using {self.model_name}...")

        if len(cleaned_data) == 0:
            raise ValueError("The audio file is completely empty (0 bytes).")

        try:
            # 1. Read directly from memory to avoid Windows file locks
            waveform, sr = sf.read(io.BytesIO(cleaned_data))

            # 2. Convert to Mono if the audio is Stereo
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)

            # 3. Resample to 16kHz for the AI
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)

            # 4. Extract true acoustic geometry (X-Vector)
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                embeddings = self.model(**inputs).embeddings

            # 5. Process the 512-dimension output
            # Normalize it to ensure Cosine Similarity math is perfectly scaled between 0 and 1
            vector_512 = torch.nn.functional.normalize(embeddings, dim=-1).cpu().squeeze().numpy()

            print(f"[{self.method_name}] Feature extraction complete. Vector shape: {vector_512.shape}")
            return vector_512

        except Exception as e:
            print(f"[{self.method_name}] Extraction critically failed: {e}")
            raise Exception(f"Acoustic Extraction Failed: {e}")

    def compare_vectors(self, vector_a: np.ndarray, vector_b: np.ndarray) -> dict:
        """Calculates distance between acoustic signatures and prints to terminal."""
        if np.all(vector_a == 0) or np.all(vector_b == 0):
            return {"euclidean_distance": 999.0, "cosine_similarity": 0.0}

        euc_dist = distance.euclidean(vector_a, vector_b)
        cos_sim = 1 - distance.cosine(vector_a, vector_b)

        # Print the mathematical verification directly to the console log
        print(f"\n=== {self.method_name} Comparison Results ===")
        print(f"Euclidean Distance: {euc_dist:.4f} (Closer to 0 is better)")
        print(f"Cosine Similarity:  {cos_sim:.4f} (Closer to 1 is better)")

        return {
            "euclidean_distance": float(euc_dist),
            "cosine_similarity": float(cos_sim)
        }