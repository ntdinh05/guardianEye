import cv2
import mediapipe as mp

# Initialize drawing and pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Set up the video capture
cap = cv2.VideoCapture(0)

# Define landmark drawing function
def draw_landmarks(image, results):
  # Iterate over the detected pose landmarks
  for landmark in results.pose_landmarks.landmark:
    # Get landmark coordinates (normalized to image dimensions)
    x = int(landmark.x * image.shape[1])
    y = int(landmark.y * image.shape[0])
    # Draw circle at landmark position
    cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)

while cap.isOpened():
  success, image = cap.read()

  if not success:
    print("Ignoring empty camera frame.")
    # If loading image fails, continue to next frame
    continue

  # Convert image to RGB format (MediaPipe expects RGB)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Perform pose detection on the frame
  results = mp_pose.process(image)

  # Re-enable writing to the image
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Draw landmarks on the processed frame
  if results.pose_landmarks:
    draw_landmarks(image, results)

  # Display the resulting frame
  cv2.imshow('MediaPipe Pose', image)

  # Exit on 'q' key press
  if cv2.waitKey(5) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
