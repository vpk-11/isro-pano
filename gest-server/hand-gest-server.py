import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

def detectHandsLandmarks(image, hands, draw=True, display=True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the output 
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                       connections=mp_hands.HAND_CONNECTIONS,
                                       landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                    thickness=2, circle_radius=2),
                                       connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0),
                                                                                      thickness=2,
                                                                                      circle_radius=2))

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        cv2.imshow("Output", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Return the output image and results of hands landmarks detection.
    return output_image, results


def detectHandGesture(results):
    '''
    This function detects thumbs up, thumbs down, okay, and unknown gestures.
    Args:
        results: The output of the hands landmarks detection performed on the image of the hands.
    Returns:
        gesture: The detected hand gesture ("Thumbs Up", "Thumbs Down", "Okay", "Unknown").
    '''

    # Initialize the gesture as "Unknown" initially.
    gesture = "Unknown"

    # Check if hand landmarks are detected.
    if results.multi_hand_landmarks:

        # Consider only the first detected hand.
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get the coordinates of the landmarks for thumb, index, middle, ring, and pinky fingers.
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Check if thumb finger tip is in I or II quadrant (thumbs up).
        if thumb_tip.x < 0.5 and thumb_tip.y < 0.5:
            # Condition for thumbs up gesture: only thumb finger is shown and thumb finger tip is in I or II quadrant.
            if index_tip.x > thumb_tip.x and middle_tip.x > thumb_tip.x and ring_tip.x > thumb_tip.x and pinky_tip.x > thumb_tip.x:
                gesture = "Thumbs Up"

        # Check if thumb finger tip is in III or IV quadrant (thumbs down).
        elif thumb_tip.x > 0.5 and thumb_tip.y > 0.5:
            # Condition for thumbs down gesture: only thumb finger is shown and thumb finger tip is in III or IV quadrant.
            if index_tip.x > thumb_tip.x and middle_tip.x > thumb_tip.x and ring_tip.x > thumb_tip.x and pinky_tip.x > thumb_tip.x:
                gesture = "Thumbs Down"

        # Check if only pinky, ring, and middle finger tip are open (okay).
        elif index_tip.x < thumb_tip.x and middle_tip.x < thumb_tip.x and ring_tip.x < thumb_tip.x and pinky_tip.x < thumb_tip.x:
            gesture = "Okay"

    return gesture


def processFrame(frame):
        # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)

    # Perform Hands landmarks detection on the frame.
    _, results = detectHandsLandmarks(frame, hands, display=False)

    # Detect hand gesture.
    gesture = detectHandGesture(results)
    return gesture


# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands function for videos.
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


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
        gesture = processFrame(frame)  # Replace this with your actual processing logic
        
        return jsonify({'gesture':gesture}), 200
    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
	app.run(port=3000,debug=True)
