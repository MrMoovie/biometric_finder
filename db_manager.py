import mysql.connector
import uuid
import json


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

    def create_person(self, national_id: str, full_name: str) -> str:
        """Creates a new person in the directory and identity map. Returns random_id."""
        random_id = str(uuid.uuid4())

        # Insert into Directory
        self.cursor.execute("INSERT INTO Person_directory (random_id) VALUES (%s)", (random_id,))

        # Insert into Identity Map
        self.cursor.execute(
            "INSERT INTO Identity_map (random_id, national_id, full_name) VALUES (%s, %s, %s)",
            (random_id, national_id, full_name)
        )
        self.conn.commit()
        return random_id

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

    def close(self):
        self.cursor.close()
        self.conn.close()