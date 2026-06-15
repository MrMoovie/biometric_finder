import json
import numpy as np
from db_manager import DatabaseManager
from face_module import FaceBiometricModule
import os


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    print("=== Biometric System Prototype Initialization ===")
    #
    # # 1. Initialize our decoupled modules
    # # Make sure your local MySQL server is running and credentials match
    db = DatabaseManager(database="biometric_system")
    face_module = FaceBiometricModule()
    #
    print("\n=== Phase 1: User Enrollment ===")
    # 2. Create a new person in the directory
    random_id = db.create_person(national_id="666666666", full_name="Test Subject Alpha")
    print(f"Created new user with Random ID: {random_id}")

    # 3. Simulate processing a raw capture during enrollment
    enrollment_image_path = "test_images/tpm.jpg"
    cleaned_enrollment_bytes = face_module.process_raw(enrollment_image_path)

    # Save the raw data to the database
    db.save_raw_data(
        random_id=random_id,
        capture_type="face_image",
        ext="jpg",
        raw_bytes=cleaned_enrollment_bytes
    )

    # 4. Extract the feature vector and save it to the database
    enrollment_vector = face_module.extract_vector(cleaned_enrollment_bytes)
    db.save_vector(
        random_id=random_id,
        method=face_module.method_name,
        vector_type="enrollment_base",
        vector_data=enrollment_vector.tolist()  # Convert numpy array to list for JSON serialization
    )
    print("Enrollment complete. Vector saved to database.")
    # random_id = "3a728c57-9e9b-48ff-aa7e-a4770792a03f"

    images = ["test_images/tpm2.jpg", "test_images/tpm3.jpg", "test_images/tpm4.jpg"]
    for img in images:
        print(f"\n=== Phase 2: Authentication Attempt {img} ===")
        # 5. Simulate a new login attempt (someone steps in front of the camera)
        login_image_path = img
        cleaned_login_bytes = face_module.process_raw(login_image_path)
        login_vector = face_module.extract_vector(cleaned_login_bytes)

        # 6. Retrieve the original vector from the database to compare
        # In a real scenario, you'd fetch this using a query, but we'll simulate the retrieval
        db.cursor.execute(
            "SELECT vector_512 FROM Method_retrieval_vectors WHERE random_id = %s AND method = %s",
            (random_id, face_module.method_name)
        )
        result = db.cursor.fetchone()

        if result and result['vector_512']:
            # Deserialize the JSON back into a numpy array
            retrieved_vector_data = json.loads(result['vector_512'])
            retrieved_vector = np.array(retrieved_vector_data)

            # 7. Run the comparison logic (Euclidean and Cosine)
            print("\n=== Phase 3: Comparison Results ===")
            scores = face_module.compare_vectors(retrieved_vector, login_vector)

            print(f"Euclidean Distance: {scores['euclidean_distance']:.4f} (Closer to 0 is better)")
            print(f"Cosine Similarity:  {scores['cosine_similarity']:.4f} (Closer to 1 is better)")

            # Simple threshold logic for the prototype
            if scores['cosine_similarity'] > 0.6:
                print(">>> AUTHENTICATION SUCCESSFUL <<<")
            else:
                print(">>> AUTHENTICATION FAILED: Vectors do not match <<<")
        else:
            print("Error: Could not retrieve user vector from the database.")

    # 8. Cleanup
    db.close()
    print("\nSystem shut down cleanly.")


if __name__ == "__main__":
    main()