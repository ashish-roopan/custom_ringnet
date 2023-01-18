from imutils import face_utils
import dlib
import cv2
import os
import json





data_folder = 'dataset/flicker/test_set/'
image_folder = data_folder + 'images/'
landmark_file = data_folder + 'landmark_data.json'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/ashish/Downloads/shape_predictor_68_face_landmarks_GTX.dat")

landmark_data = {}
image_files = os.listdir(image_folder)
for image_file in image_files:
    image_path = image_folder + image_file
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for j, (x, y) in enumerate(shape):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(image, str(j), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    landmark_data[image_file] = shape.tolist()
    print('shape: ', shape.shape)
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


 

# save landmark data
with open(landmark_file, 'w') as f:
    json.dump(landmark_data, f)