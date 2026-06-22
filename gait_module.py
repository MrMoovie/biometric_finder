import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tempfile
import os
import urllib.request
from scipy.spatial import distance
from biometric_interface import BiometricModule


class GaitBiometricModule(BiometricModule):
    def __init__(self):
        self.model_name = "MediaPipe (Biomechanical Ratios)"
        self.model_path = "pose_landmarker.task"

        if not os.path.exists(self.model_path):
            print(f"[{self.method_name}] Downloading MediaPipe Pose Model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    @property
    def method_name(self) -> str:
        return "GAIT"

    def process_raw(self, raw_file_path: str) -> bytes:
        print(f"\n[{self.method_name}] Loading raw video file: {raw_file_path}")
        with open(raw_file_path, "rb") as f:
            return f.read()

    def extract_vector(self, cleaned_data: bytes) -> np.ndarray:
        print(f"[{self.method_name}] Extracting biomechanical bone ratios using {self.model_name}...")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(cleaned_data)
            temp_video_path = temp_video.name

        cap = None
        try:
            cap = cv2.VideoCapture(temp_video_path)
            gait_sequence = []

            # Helper function for 3D Euclidean distance between two landmarks
            def calc_3d_distance(lm1, lm2):
                return np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2 + (lm1.z - lm2.z) ** 2)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = self.detector.detect(mp_image)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks[0]

                    # 1. Grab Shoulders and Hips to find the Torso Length
                    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
                    left_hip, right_hip = landmarks[23], landmarks[24]

                    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
                    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    mid_hip_x = (left_hip.x + right_hip.x) / 2
                    mid_hip_y = (left_hip.y + right_hip.y) / 2

                    # 2D Torso Length (Reference Scale)
                    torso_length = np.sqrt((mid_shoulder_x - mid_hip_x) ** 2 + (mid_shoulder_y - mid_hip_y) ** 2)

                    if torso_length == 0:
                        continue

                    # 2. Grab Legs
                    left_knee, right_knee = landmarks[25], landmarks[26]
                    left_ankle, right_ankle = landmarks[27], landmarks[28]

                    # 3. Calculate Biomechanical Ratios (Divided by Torso Length)
                    pelvis_width = calc_3d_distance(left_hip, right_hip) / torso_length
                    left_thigh = calc_3d_distance(left_hip, left_knee) / torso_length
                    right_thigh = calc_3d_distance(right_hip, right_knee) / torso_length
                    left_calf = calc_3d_distance(left_knee, left_ankle) / torso_length
                    right_calf = calc_3d_distance(right_knee, right_ankle) / torso_length
                    stride_width = calc_3d_distance(left_ankle, right_ankle) / torso_length  # Dynamic step size

                    frame_features = [
                        pelvis_width, left_thigh, right_thigh,
                        left_calf, right_calf, stride_width
                    ]
                    gait_sequence.append(frame_features)

            if len(gait_sequence) == 0:
                raise ValueError("No human skeleton detected in the video.")

            gait_matrix = np.array(gait_sequence)
            mean_pose = np.mean(gait_matrix, axis=0)  # Static bone geometry
            std_pose = np.std(gait_matrix, axis=0)  # Dynamic swing behavior

            raw_signature = np.concatenate([mean_pose, std_pose])

            vector_512 = np.zeros(512)
            vector_512[:len(raw_signature)] = raw_signature

            normalized_vector = vector_512 / np.linalg.norm(vector_512)

            print(f"[{self.method_name}] Feature extraction complete. Vector shape: {normalized_vector.shape}")
            return normalized_vector

        except Exception as e:
            print(f"[{self.method_name}] Gait Extraction critically failed: {e}")
            return np.zeros(512)
        finally:
            if cap is not None:
                cap.release()
            try:
                os.unlink(temp_video_path)
            except Exception as e:
                pass

    def compare_vectors(self, vector_a: np.ndarray, vector_b: np.ndarray) -> dict:
        if np.all(vector_a == 0) or np.all(vector_b == 0):
            return {"euclidean_distance": 999.0, "cosine_similarity": 0.0}

        euc_dist = distance.euclidean(vector_a, vector_b)
        cos_sim = 1 - distance.cosine(vector_a, vector_b)

        print(f"\n=== {self.method_name} Comparison Results ===")
        print(f"Euclidean Distance: {euc_dist:.4f}")
        print(f"Cosine Similarity:  {cos_sim:.4f}")

        return {
            "euclidean_distance": float(euc_dist),
            "cosine_similarity": float(cos_sim)
        }