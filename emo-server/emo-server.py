import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import pandas as pd
import dlib
from imutils import face_utils
from tensorflow.keras.layers import *
from flask import Flask, request, jsonify

# Uncomment line no 33,34 on this block to see the points plotted on the image 
# Function to calculate distance between two facial points
def distWithMagnitude(a,b):
	return math.dist(a,b)

"""
Function to get all the specific landmarks from face and 
	construct a distance vector by strategically taking specific points

"""
def getLandMarkFromFace(image):
	# Load the detector
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

	# Convert image into grayscale
	gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

	# Use detector to find landmarks
	faces = detector(gray)
	allFaces=[]
	row=[]
	for (i, rect) in enumerate(faces):		
		
		"""
		# Determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy array
		"""
		shape = predictor(gray, rect)
		points = face_utils.shape_to_np(shape)
		# for (sX, sY) in points:                            # Uncomment to see the 68 landmark points drawn over the face
		#     cv2.circle(image, (sX, sY), 2, (0, 0, 255), -1) 
		
		# Construction of distance vector by taking speciific points
		# Check previous image for reference to map points to number
		
		
		# Eyes - Left 
		row.append(distWithMagnitude(points[36],points[37]))
		row.append(distWithMagnitude(points[37],points[38]))
		row.append(distWithMagnitude(points[38],points[39]))
		row.append(distWithMagnitude(points[39],points[40]))        
		row.append(distWithMagnitude(points[40],points[41]))             
		row.append(distWithMagnitude(points[41],points[36]))
		
		# Eyes - Right 
		row.append(distWithMagnitude(points[42],points[43]))
		row.append(distWithMagnitude(points[43],points[44]))
		row.append(distWithMagnitude(points[44],points[45]))
		row.append(distWithMagnitude(points[45],points[46]))        
		row.append(distWithMagnitude(points[46],points[47]))             
		row.append(distWithMagnitude(points[47],points[42]))
					
		# Nose
		row.append(distWithMagnitude(points[27],points[28]))
		row.append(distWithMagnitude(points[28],points[29]))
		row.append(distWithMagnitude(points[29],points[30]))
		row.append(distWithMagnitude(points[30],points[31]))        
		row.append(distWithMagnitude(points[30],points[32]))             
		row.append(distWithMagnitude(points[30],points[33]))          
		row.append(distWithMagnitude(points[30],points[34]))         
		row.append(distWithMagnitude(points[30],points[35]))
		
		#Upper Lips
		row.append(distWithMagnitude(points[48],points[49]))
		row.append(distWithMagnitude(points[49],points[50]))
		row.append(distWithMagnitude(points[50],points[51]))
		row.append(distWithMagnitude(points[51],points[52]))        
		row.append(distWithMagnitude(points[52],points[53]))             
		row.append(distWithMagnitude(points[53],points[54]))          
		row.append(distWithMagnitude(points[54],points[55]))         
		row.append(distWithMagnitude(points[55],points[56]))      
		row.append(distWithMagnitude(points[56],points[57]))         
		row.append(distWithMagnitude(points[57],points[58]))      
		row.append(distWithMagnitude(points[58],points[59]))         
		row.append(distWithMagnitude(points[59],points[48]))
		
		#Lower Lips        
		row.append(distWithMagnitude(points[60],points[61]))
		row.append(distWithMagnitude(points[61],points[62]))
		row.append(distWithMagnitude(points[62],points[63]))
		row.append(distWithMagnitude(points[63],points[64]))        
		row.append(distWithMagnitude(points[64],points[65]))             
		row.append(distWithMagnitude(points[65],points[66]))          
		row.append(distWithMagnitude(points[66],points[67]))         
		row.append(distWithMagnitude(points[67],points[60]))
		
		#Between Lips        
		row.append(distWithMagnitude(points[48],points[60]))
		row.append(distWithMagnitude(points[64],points[54]))
		row.append(distWithMagnitude(points[60],points[64]))
		row.append(distWithMagnitude(points[62],points[66]))
		
		
		#Eyebrow                
		row.append(distWithMagnitude(points[36],points[17]))
		row.append(distWithMagnitude(points[37],points[18]))
		row.append(distWithMagnitude(points[38],points[20]))
		row.append(distWithMagnitude(points[39],points[21]))
		
		row.append(distWithMagnitude(points[42],points[22]))
		row.append(distWithMagnitude(points[43],points[23]))
		row.append(distWithMagnitude(points[44],points[25]))
		row.append(distWithMagnitude(points[45],points[26]))
		
	# # row variable consists of the distance vector of length 52 and returned along with the image
	# plt.imshow(image)    # If image should be displayed uncomment
	# plt.show()
	return image,row

# Function to predict single emotion and its intensity by modelname and points extracted from image
def modelPrediction(modelName,row):
	emoModel = tf.keras.models.load_model("./Model/"+modelName) 
	if len(row)==52:
		row=pd.DataFrame(np.array([row]))
		p = emoModel.predict(row,verbose = 0)
		print(p)
		if modelName=="happySad":
			if p[0][0]>0.5:
				return "happy", p[0][0]
			else: 
				return "sad", p[0][1]
		elif p[0][0]>0.5:
			return modelName,p[0][0]
		else: return "", 0
	else:
		return "", 0
	
"""
# Requires previous block to be run as function is used
# Given an image and row 
# 	If singleEmotionBool equals to false Function to predict all the set of emotions
# 	If singleEmotionBool equal to true Function to predict single emotion with maximum intensity
# 	Image passed in function is redundant and is required only if image has to be showed in notebook output
"""
def allModelPrediction(singleEmotionBool, row, image):
	happySadEmotion, happySadVal = modelPrediction("happySad",row)
	singleEmotion = happySadEmotion
	singleEmotionVal = happySadVal
	if happySadEmotion != "":
		angryEmotion, angryVal = modelPrediction("angry",row)
		if angryVal> singleEmotionVal:
			singleEmotion = angryEmotion
			singleEmotionVal = angryVal
		disgustEmotion, disgustVal = modelPrediction("disgust",row)
		if disgustVal> singleEmotionVal:
			singleEmotion = disgustEmotion
			singleEmotionVal = disgustVal            
		surpriseEmotion, surpriseVal = modelPrediction("surprise",row)
		if surpriseVal> singleEmotionVal:
			singleEmotion = surpriseEmotion
			singleEmotionVal = surpriseVal
		if singleEmotionBool == False:
			# print("Happy: " + happySadEmotion+", "+angryEmotion+", "+disgustEmotion+", "+surpriseEmotion)
			print(f"Happy/Sad: {happySadVal}\nAngry: {angryVal}\nDisgust: {disgustVal}\nSurprise: {surpriseVal}")
		# print("Result: ", singleEmotion, singleEmotionVal)
		return singleEmotion
#         plt.imshow(image)     # Uncomment to print image with points 
#         plt.show()

def detectEmotion(image, modelName=None, singleEmotionBool=True):
	if image.shape[0] < 512 and image.shape[1] < 512:
		image = cv2.resize(image, dsize=(512, 512))
		# image = cv2.resize(image, width = 512)
	frame, row = getLandMarkFromFace(image)
	# plt.imshow(frame)
	# plt.show()

	# if modelName != None:
	#     emotion, value = modelPrediction(modelName, row)
	#     print(emotion)
	
	emotion = allModelPrediction(singleEmotionBool, row, frame)
	return emotion

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
        emotion = detectEmotion(frame)  # Replace this with your actual processing logic
        
        # return jsonify({'emotion':emotion}), 200
        return emotion, 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
	app.run(port=8000,debug=True)



