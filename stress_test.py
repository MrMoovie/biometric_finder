import time
import uuid
import json
import numpy as np
from db_manager import DatabaseManager
from face_module import FaceBiometricModule

# --- TEST CONFIGURATION ---
TARGET_POPULATION = 25000  # Number of fake users to generate
CHUNK_SIZE = 1000  # How many to push to MySQL at once to prevent timeouts


def generate_dummy_data(db):
    print(f"INITIATING STRESS TEST: Generating {TARGET_POPULATION} synthetic profiles...")

    # 1. Check current database size
    db.cursor.execute("SELECT COUNT(*) as count FROM method_retrieval_vectors WHERE method = 'FACE'")
    current_count = db.cursor.fetchone()['count']

    if current_count >= TARGET_POPULATION:
        print(f"Database already contains {current_count} vectors. Skipping generation.")
        return

    needed = TARGET_POPULATION - current_count
    print(f"Injecting {needed} new profiles. This will take a moment...")

    # 2. Fast Bulk Insert using executemany
    db.cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")

    for i in range(0, needed, CHUNK_SIZE):
        chunk = min(CHUNK_SIZE, needed - i)

        # Generate random UUIDs and random 512-D vectors (simulating ArcFace)
        uuids = [str(uuid.uuid4()) for _ in range(chunk)]

        # Insert into person_directory
        person_query = "INSERT INTO person_directory (random_id) VALUES (%s)"
        db.cursor.executemany(person_query, [(u,) for u in uuids])

        # Insert into method_retrieval_vectors
        vector_query = """
                       INSERT INTO method_retrieval_vectors (random_id, method, vector_type, vector_512)
                       VALUES (%s, %s, %s, %s) \
                       """
        # Normalize the random vectors so they act like real faces
        random_vectors = [np.random.rand(512) for _ in range(chunk)]
        normalized = [v / np.linalg.norm(v) for v in random_vectors]
        vector_data = [(uuids[j], "FACE", "synthetic", json.dumps(normalized[j].tolist())) for j in range(chunk)]

        db.cursor.executemany(vector_query, vector_data)
        db.conn.commit()  # <--- Changed to match your class
        print(f"   -> Inserted {i + chunk} / {needed}...")

    db.cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
    print("Synthetic population fully injected.\n")


def run_benchmark(db, face_module):
    print("STARTING BENCHMARK: Linear Search Loop")

    # Generate one fake "Unknown Probe" to search for
    probe_vector = np.random.rand(512)
    probe_vector = probe_vector / np.linalg.norm(probe_vector)

    # --- STOPWATCH START ---
    t0 = time.time()

    # Step 1: Fetch the massive dictionary from MySQL
    t_fetch_start = time.time()
    database_vectors = db.get_all_vectors("FACE")
    t_fetch_end = time.time()

    # Step 2: The Python For-Loop (The Bottleneck)
    best_score = 999.0
    for record in database_vectors:
        # We simulate the exact logic happening inside your fusion_engine.py
        metrics = face_module.compare_vectors(record['vector'], probe_vector)
        if metrics['euclidean_distance'] < best_score:
            best_score = metrics['euclidean_distance']

    # --- STOPWATCH END ---
    t_total_end = time.time()

    # Print the Executive Report
    print("\n" + "=" * 50)
    print("STRESS TEST RESULTS")
    print("=" * 50)
    print(f"Total Profiles Scanned: {len(database_vectors):,}")
    print(f"Database Fetch Time:    {(t_fetch_end - t_fetch_start):.4f} seconds")
    print(f"Python CPU Math Time:   {(t_total_end - t_fetch_end):.4f} seconds")
    print("-" * 50)
    print(f"TOTAL SEARCH LATENCY: {(t_total_end - t0):.4f} seconds")
    print("=" * 50)


if __name__ == "__main__":
    db = DatabaseManager()
    face_module = FaceBiometricModule()
    try:
        generate_dummy_data(db)
        run_benchmark(db, face_module)
    finally:
        db.close()