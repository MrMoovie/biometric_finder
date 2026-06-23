from datetime import date, datetime, timedelta, time
from decimal import Decimal
from typing import Any

import mysql.connector
import uuid
import json
import numpy as np


class DatabaseManager:
    def __init__(self, host="localhost", user="root", password="102030", database="biometric_system"):
        # Connect to MySQL
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor(dictionary=True)

    def create_person(self, national_id: str, full_name: str):
        """Creates a new person in the directory and identity map. Returns random_id."""
        self.cursor.execute("SELECT random_id FROM Identity_map WHERE national_id = %s", (national_id,))
        result = self.cursor.fetchone()

        if result:
            # Because dictionary=True, we access it via the string key, not [0]
            random_id = result['random_id']
            exists = True
        else:
            # The user doesn't exist, generate a new UUID
            exists = False
            random_id = str(uuid.uuid4())
            self.cursor.execute("INSERT INTO Person_directory (random_id) VALUES (%s)", (random_id,))
            self.cursor.execute(
                "INSERT INTO Identity_map (random_id, national_id, full_name) VALUES (%s, %s, %s)",
                (random_id, national_id, full_name)
            )
            self.conn.commit()

        return random_id, exists

    def save_raw_data(self, random_id: str, capture_type: str, ext: str, raw_bytes: bytes):
        """Saves the raw biometric sample."""
        self.cursor.execute(
            "INSERT INTO Raw_data (random_id, capture, ext, byte_size, image_bytes) VALUES (%s, %s, %s, %s, %s)",
            (random_id, capture_type, ext, len(raw_bytes), raw_bytes)
        )
        self.conn.commit()

    def save_vector(self, random_id: str, method: str, vector_type: str, vector_data: list):
        """
        Saves the extracted vector.
        NOTE: Currently using JSON for rapid prototyping. Will switch to BLOB later.
        """
        # Ensure legacy compatibility table entry exists
        self.cursor.execute(
            "INSERT IGNORE INTO Feature_vectors (random_id, method) VALUES (%s, %s)",
            (random_id, method)
        )

        # Save the actual vector as JSON into vector_512
        vector_json = json.dumps(vector_data)
        self.cursor.execute(
            "INSERT INTO Method_retrieval_vectors (random_id, method, vector_type, vector_512) VALUES (%s, %s, %s, %s)",
            (random_id, method, vector_type, vector_json)
        )
        self.conn.commit()

    def get_all_vectors(self, method_name: str) -> list:
        """
        Fetches ALL vectors, allowing multiple samples per UUID.
        Returns a list of dictionaries: [{'random_id': uuid, 'vector': np_array}, ...]
        """
        try:
            self.cursor.execute(
                "SELECT random_id, vector_512 FROM Method_retrieval_vectors WHERE method = %s",
                (method_name,)
            )
            results = self.cursor.fetchall()

            database_vectors = []
            for row in results:
                if row['vector_512']:
                    import numpy as np
                    import json
                    database_vectors.append({
                        'random_id': row['random_id'],
                        'vector': np.array(json.loads(row['vector_512']))
                    })

            return database_vectors
        except Exception as e:
            print(f"Database Fetch Error: {e}")
            return []

    def close(self):
        self.cursor.close()
        self.conn.close()

