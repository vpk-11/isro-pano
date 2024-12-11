import face_recognition
from collections import Counter
import pickle
from pathlib import Path
import numpy as np
import requests
from flask import Flask, request, jsonify
import cv2

DEFAULT_ENCODINGS_PATH = Path("./facesV2/output/encodings.pkl")

def recognize_face_aux(unknown_encoding, loaded_encodings):
	boolean_matches = face_recognition.compare_faces(
		loaded_encodings["encodings"], unknown_encoding
	)
	votes = Counter(
		name
		for match, name in zip(boolean_matches, loaded_encodings["names"])
		if match
	)

	# Check if any votes exceed the confidence threshold
	if votes:
		return votes.most_common(1)[0]
	
def recognizeFace(image: np.ndarray, encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> str:
	# Load the pre-trained face encodings
	with encodings_location.open(mode="rb") as f:
		loaded_encodings = pickle.load(f)

	# Detect faces in the input image
	input_face_locations = face_recognition.face_locations(image)
	input_face_encodings = face_recognition.face_encodings(image, input_face_locations)

	if len(input_face_encodings) == 0:
		return "No face detected"  # Return a message if no face is found

	# Compare the detected face encoding with known encodings
	unknown_encoding = input_face_encodings[0]  # Assuming only one face in the image
	x = recognize_face_aux(unknown_encoding, loaded_encodings)

	if not x:
		return "Unknown"
	print(x)
	if not x or int(x[1]) < 190:
		return "Unknown"
	return x[0]


# Flask server Code
app = Flask(__name__)
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Retrieve the binary data of the image
        file = request.files['image']
        if not file:
            return "No image received", 400
        
        # Decode the image from binary data
        np_arr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Validate the image decoding
        if frame is None:
            return "Failed to decode image", 400
        
        # Display the received frame for debugging (optional)
        cv2.imwrite("debug_frame.jpg", frame)  # Save the frame for verification
        
        # Process the frame
        x = recognizeFace(frame)  # Replace this with your actual processing logic
        
        return jsonify({'face':x}), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
	app.run(port=7000,debug=True)