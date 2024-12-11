import requests
import cv2
import matplotlib as plt
from ultralytics import YOLO
import torch


emotion_detection_server_url = f"http://localhost:8000/process_frame"
hand_gesture_recognition_server_url = f"http://localhost:3000/process_frame"
finger_counting_server_url = f"http://localhost:6000/process_frame"
body_posture_recognition_server_url = f"http://localhost:5000/process_frame"
face_recognition_server_url = f"http://localhost:8001/process_frame"


# Funtion that calls a flask server to process frame and return the detected emotion
def detectEmotionFlask(frame):
    # Encode the frame to JPEG format for transmission
    _, encoded_image = cv2.imencode('.jpg', frame)
    
    # Define the Flask server URL
    server_url = emotion_detection_server_url
    
    try:
        # Send the frame to the Flask server
        response = requests.post(server_url, files={"image": encoded_image.tobytes()})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json
            emotion = data['emotion']
            return emotion  # Return the emotion from the server
        else:
            print(f"Server error: {response.status_code} - {response.text}")
            return "Error: Unable to process frame"
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")
        return "Error: Communication failure"


# Funtion that calls a flask server to process frame and return the detected hand gesture
def detectHandGestureFlask(frame):
    # Encode the frame to JPEG format for transmission
    _, encoded_image = cv2.imencode('.jpg', frame)
    
    # Define the Flask server URL
    server_url = hand_gesture_recognition_server_url
    
    try:
        # Send the frame to the Flask server
        response = requests.post(server_url, files={"image": encoded_image.tobytes()})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json
            gesture = data['gesture']
            return gesture  # Return the gesture from the server
        else:
            print(f"Server error: {response.status_code} - {response.text}")
            return "Error: Unable to process frame"
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")
        return "Error: Communication failure"


# Funtion that calls a flask server to process frame and return the count of fingers
def countFingersFlask(frame):
    # Encode the frame to JPEG format for transmission
    _, encoded_image = cv2.imencode('.jpg', frame)
    
    # Define the Flask server URL
    server_url = finger_counting_server_url
    
    try:
        # Send the frame to the Flask server
        response = requests.post(server_url, files={"image": encoded_image.tobytes()})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json
            fingersStatus = data['fingersStatus']
            count = data['count']
            return fingersStatus, count  # Return the count from the server
        else:
            print(f"Server error: {response.status_code} - {response.text}")
            return "Error: Unable to process frame"
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")
        return "Error: Communication failure"


# Funtion that calls a flask server to process frame and return the detected body posture
def detectBodyPostureFlask(frame):
    # Encode the frame to JPEG format for transmission
    _, encoded_image = cv2.imencode('.jpg', frame)
    
    # Define the Flask server URL
    server_url = body_posture_recognition_server_url
    
    try:
        # Send the frame to the Flask server
        response = requests.post(server_url, files={"image": encoded_image.tobytes()})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json
            posture = data['posture']
            return posture  # Return the posture from the server
        else:
            print(f"Server error: {response.status_code} - {response.text}")
            return "Error: Unable to process frame"
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")
        return "Error: Communication failure"


# Funtion that calls a flask server to process frame and return the detected person name
def recognizeFaceFlask(frame):
    # Encode the frame to JPEG format for transmission
    _, encoded_image = cv2.imencode('.jpg', frame)
    
    # Define the Flask server URL
    server_url = face_recognition_server_url
    
    try:
        # Send the frame to the Flask server
        response = requests.post(server_url, files={"image": encoded_image.tobytes()})
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json
            emotion = data['emotion']
            return emotion  # Return the emotion from the server
        else:
            print(f"Server error: {response.status_code} - {response.text}")
            return "Error: Unable to process frame"
    except Exception as e:
        print(f"Error communicating with server: {str(e)}")
        return "Error: Communication failure"


# Load YOLOv8 model (replace with your model path)
yoloModel = YOLO("YoloV8/yolov8m.pt")  # Assuming you have a lightweight YOLOv8 model

# Function to detect persons using YOLOv8
def detectPersons(yoloModel, image, inputSize=(640, 640)):
	"""
	Detects persons using a YOLOv8 model.

	Args:
		model (YOLO): YOLOv8 model loaded from ultralytics.
		image (np.ndarray): Input image as a NumPy array.
		input_size (tuple, optional): Size to resize the input image. Defaults to (640, 640).

	Returns:
		list: List of bounding boxes for detected persons in format [x1, y1, x2, y2].
	"""
	results = yoloModel.predict(image, imgsz = 640,conf=0.5)

	boundingBoxes = []  # List to store all bounding boxes (including non-person)

	for detection in results:
		boxes = detection.boxes
		# Process each bounding box
		for box in boxes:
			if detection.names and detection.names[box.cls[0].item()] != 'person':
				continue  # Skip if not "person" class

			if isinstance(box, torch.Tensor):
				box = box.cpu().numpy()
			xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

			boundingBoxes.append((xmin, ymin, xmax, ymax))  # Add bounding box to general list

	# Now 'all_detections' contains detailed info for all detections
	# 'boundingBoxes' contains all bounding boxes (including non-person)
	return boundingBoxes


# Function to extract ROI from the image
def extractROI(image, bbox):
	"""
	Extracts a region of interest (ROI) from the image based on the bounding box.

	Args:
		image (np.ndarray): Input image as a NumPy array.
		bbox (tuple): Bounding box coordinates. Can be in either format:
			- (x, y, width, height)
			- (xmin, ymin, xmax, ymax)

	Returns:
		np.ndarray: Cropped image containing the ROI.
	"""

	if len(bbox) == 4:  # Check if format is (xmin, ymin, xmax, ymax)
		x, y, xmax, ymax = bbox
		w = xmax - x
		h = ymax - y
	else:  # Assume format is (x, y, width, height)
		x, y, w, h = bbox

	return image[y:y + h, x:x + w]


# Import threading module
import threading
from concurrent.futures import ThreadPoolExecutor

# Define the number of worker threads (adjust as needed)
num_threads = 5

# Create a thread pool using ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=num_threads)

# Main function to process the image
def processROI(roi):

	# Submit tasks to the thread pool
	future1 = executor.submit(detectEmotionFlask, roi)
	future2 = executor.submit(detectHandGestureFlask, roi)
	future3 = executor.submit(countFingersFlask, roi)
	future4 = executor.submit(detectBodyPostureFlask, roi)
	future5 = executor.submit(recognizeFaceFlask, roi)

	# Wait for all tasks to finish and collect outputs
	emotionLabel = future1.result()
	gestureLabel = future2.result()
	fingerStatus, fingerCount = future3.result()
	postureLabel = future4.result()
	faceLabel = future5.result()

	# emotionLabel = detectEmotion(roi)
	# print(emotionLabel)

	# gestureLabel = recognizeGesture(roi)
	# print(gestureLabel)

	# fingerStatus, fingerCount = countFingers(roi)
	# print(fingerCount)

	# postureLabel = recognizeBodyPosture(roi)
	# print(postureLabel)

	return emotionLabel, gestureLabel, fingerCount, postureLabel, faceLabel


"""
Funtion gets image as an input
uses YoloV8 to get the bboxes for the ROIs
Called Process ROI to get the label's from different models
returns the image (with the bounding boxes)
"""
from ultralytics.utils.plotting import Annotator

def processImage(image, c):
	# Detect persons and get bounding boxes
	person_bboxes = detectPersons(yoloModel, image)
	print("Detected Persons:", len(person_bboxes))
	if len(person_bboxes) == 0:
		return image, None, None, None, None, None
		
	# Draw the bounding box	
	# Loop through detected persons and extract ROIs
	counter = 0
	copy = image.copy()
	annotator = Annotator(image)
	for bbox in person_bboxes:
		counter+=1
		roi = extractROI(copy, bbox)
		# cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2) # snippet to draw bounding box
		roi = cv2.resize(roi, dsize=(512, 512))
		emotionLabel, gestureLabel, fingerCount, postureLabel, faceLabel = processROI(roi)
		if faceLabel == "unknown":
			faceLabel = f"Person {counter}"
		annotator.box_label(bbox, faceLabel)  # Annotate the image with class names
		plt.imshow(roi)
		plt.show()

		text = f"Recognized Person: {faceLabel}\nEmotion: {emotionLabel}\nGesture: {gestureLabel}\nFinger Counts: {fingerCount}\nPosture: {postureLabel}"
		print(text)
	
	return annotator.result(), emotionLabel, gestureLabel, fingerCount, postureLabel, faceLabel

# Open video capture from webcam
cap = cv2.VideoCapture(0)
c = 0
cap.set(3, 640)
cap.set(4, 640)

# Font color, format, and position of emotion detail displayed in frame  
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 0.5
fontColor              = (0,255,0)
thickness              = 2
lineType               = 1

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	frame, emotionLabel, gestureLabel, fingerCount, postureLabel, faceLabel = processImage(frame, c)
	# text = f"Recognized Person: {faceLabel} | Emotion: {emotionLabel} | Gesture: {gestureLabel} | Finger Counts: {fingerCount} | Posture: {postureLabel}"

	# Display the frame with bounding boxes
	cv2.putText(frame,f"Recognized Person: {faceLabel}", (50,100), 
		 font, fontScale, fontColor, thickness, lineType)
	# Display the frame with bounding boxes
	cv2.putText(frame,f"Emotion: {emotionLabel}", (50,120), 
		 font, fontScale, fontColor, thickness, lineType)
	 # Display the frame with bounding boxes
	cv2.putText(frame,f"Gesture: {gestureLabel}", (50,140), 
		 font, fontScale, fontColor, thickness, lineType)
	 # Display the frame with bounding boxes
	cv2.putText(frame,f"Finger Counts: {fingerCount}", (50,160), 
		 font, fontScale, fontColor, thickness, lineType)
	 # Display the frame with bounding boxes
	cv2.putText(frame,f"Posture: {postureLabel}", (50,180), 
		 font, fontScale, fontColor, thickness, lineType)
	
	# Draw banner on the frame
	# frame_with_banner = draw_banner_on_frame(frame.copy(), [emotionLabel, gestureLabel, fingerCount, postureLabel, faceLabel])
	cv2.imshow("YOLOv8 Person Detection", frame)
	# cv2.imwrite(f'./test/{c}_img.png', frame)
	 
   
	c+=1

	# Exit on 'q' press
	if cv2.waitKey(1) == ord("q"):
		break

# Shut down the thread pool (optional)
executor.shutdown()
cap.release()
cv2.destroyAllWindows()


