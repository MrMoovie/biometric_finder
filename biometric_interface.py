from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import numpy as np


class BiometricModule(ABC):
    """
    Abstract base class for all biometric modules.
    Enforces the pipeline: Collect -> Clean -> Extract -> Compare.
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Returns the name of the biometric method (e.g., 'FACE', 'VOICE')."""
        pass

    @abstractmethod
    def process_raw(self, raw_file_path: str) -> bytes:
        """
        Step 1 & 2: Collect, clean, and optimize the raw sample[cite: 7, 8].
        Returns the cleaned bytes ready for DB storage and extraction.
        """
        pass

    @abstractmethod
    def extract_vector(self, cleaned_data: bytes) -> np.ndarray:
        """
        Step 3: Process the sample and create the matching feature vector[cite: 9].
        Returns a numpy array representing the vector.
        """
        pass

    @abstractmethod
    def compare_vectors(self, vector_a: np.ndarray, vector_b: np.ndarray) -> Dict[str, float]:
        """
        Step 5: Compare vectors using at least two known comparison methods[cite: 11].
        Returns a dictionary with the distance/similarity scores (e.g., {'euclidean': 0.5, 'cosine': 0.9}).
        """
        pass