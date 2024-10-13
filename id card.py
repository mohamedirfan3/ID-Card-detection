import cv2
import numpy as np
import face_recognition
import os

# Initialize camera
cap = cv2.VideoCapture(0)

# Define a more precise strap color range in HSV for orange color
lower_orange = (10, 100, 100)  # Lower bound of orange in HSV
upper_orange = (25, 255, 255)  # Upper bound of orange in HSV

# Load known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

def load_known_faces(directory="known_faces"):
    """Load and encode known faces from a directory."""
    known_face_encodings = []
    known_face_names = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            try:
                # Load the image using OpenCV to handle format issues
                image = cv2.imread(img_path)
                
                # Convert the image to RGB format
                if image is None:
                    print(f"Error loading image {filename}. It might be corrupted.")
                    continue
                
                # Convert the image from BGR (OpenCV default) to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Now use face_recognition to find face encodings
                encoding = face_recognition.face_encodings(rgb_image)
                if encoding:  # Ensure there is at least one encoding found
                    known_face_encodings.append(encoding[0])
                    known_face_names.append(os.path.splitext(filename)[0])
                    print(f"Encoded face for {filename}.")
                else:
                    print(f"No face found in {filename}. Skipping.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return known_face_encodings, known_face_names

# Load known faces
known_face_encodings, known_face_names = load_known_faces("known_faces")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for strap detection
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Find contours of the strap (orange color detection)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    strap_detected = False  # Flag to track orange strap detection

    if contours:  # If strap is detected
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter based on area (adjust the threshold for small objects)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Orange Strap Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                strap_detected = True
    
    if not strap_detected:  # If no orange strap is detected, recognize face
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Speed up by resizing
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations and face encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Check each detected face
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Display the name on the frame if no orange strap is detected
            top, right, bottom, left = face_location
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('ID Strap & Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
