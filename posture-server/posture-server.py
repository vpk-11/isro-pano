import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import io
import matplotlib as plt

# Action names
action_names = ["drink water", "eat meal", "brush teeth", "brush hair", "drop", "pick up", "throw", "sit down", "stand up", 
	"clapping", "reading", "writing", "tear up paper", "put on jacket", "take off jacket", "put on a shoe", 
	"take off a shoe", "put on glasses", "take off glasses", "put on a hat or cap", "take off a hat or cap", 
	"cheer up", "hand waving", "kicking something", "reach into pocket", "hopping", "jump up", "phone call", 
	"play with phone or tablet", "type on a keyboard", "point to something", "taking a selfie", 
	"check time (from watch)", "rub two hands", "nod head or bow", "shake head", "wipe face", "salute", 
	"put palms together", "cross hands in front", "sneeze or cough", "staggering", "falling down", "headache", 
	"chest pain", "back pain", "neck pain", "nausea or vomiting", "fan self", "punch or slap", "kicking", "pushing"]

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Function to perform pose detection and return annotated frame and landmarks
def detect_pose(frame, pose_video):
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = pose_video.process(frame_rgb)
	if results.pose_landmarks:
		mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
	return frame, results

# Function to analyze the results and get coordinates of each landmark
def get_landmark_coordinates(results):
	if not results.pose_landmarks:
		return []
	landmarks = []
	for lm in results.pose_landmarks.landmark:
		landmarks.append((lm.x, lm.y, lm.z))
	return landmarks
		
frame_sequence = []
# Function to switch on camera and check if the model is working
def processFrame(model, frame, sequence_length=30, num_joints=33):
	# Perform pose detection
		frame, results = detect_pose(frame, pose_video)
		coordinates = get_landmark_coordinates(results)
		
		if coordinates:
			frame_sequence.append(coordinates)
			if len(frame_sequence) > sequence_length:
				frame_sequence.pop(0)
			
			if len(frame_sequence) == sequence_length:
				sequence_array = np.array(frame_sequence).reshape((1, sequence_length, num_joints, 3))
				prediction = model.predict(sequence_array)
				predicted_class = np.argmax(prediction, axis=1)[0]
				action_name = action_names[predicted_class]
				return action_name
			
# Flask server Code
app = Flask(__name__)
model = load_model("models/activity_recognition_model.keras")

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
        
        # Process the frame (example: return a dummy action name)
        action_name = processFrame(model,frame)  # Replace this with your actual processing logic
        
        return action_name, 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
	app.run(port=5000,debug=True)