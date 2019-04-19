import cv2
import logging
import numpy as np

from facenet.src import facenet
from facenet.src.align import detect_face as mtcnn
from camera.VideoStream import VideoStream
from scipy import misc

import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

cam = VideoStream(use_pi_camera=False,
                  resolution=(1920, 1080),
                  src=0
                  ).start()

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = mtcnn.create_mtcnn(sess, None)

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor


def look_for_faces():
    while True:
        img = cam.read()
        total_boxes, points = mtcnn.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        faces = []

        logger.info(f'Found {len(total_boxes)} faces')
        for idx, (bounding_box, keypoints) in enumerate(zip(total_boxes, points.T)):
            det1 = np.squeeze(total_boxes[0, 0:4])
            x, y, w, h = (
                int(bounding_box[0]),
                int(bounding_box[1]),
                int(bounding_box[2] - bounding_box[0]),
                int(bounding_box[3] - bounding_box[1])
            )
            face = {
                'box': [x, y, w, h],
                'confidence': bounding_box[-1],
                'keypoints': {
                    'left_eye': (int(keypoints[0]), int(keypoints[5])),
                    'right_eye': (int(keypoints[1]), int(keypoints[6])),
                    'nose': (int(keypoints[2]), int(keypoints[7])),
                    'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                    'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                }
            }
            faces.append(face)
            logger.info(f"Found face with a confidence of: {face['confidence']}")

            # Draw rectangle around face on the original image
            x1, y1, x2, y2 = (x, y, x + w, y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Show cropped face image
            face_img = img[y1:y2, x1:x2]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:
                logger.debug(face_img.shape)
                cv2.imshow(f'cropped face {idx}', face_img)

            face_img_2 = crop_and_align_image(img, bounding_box, face['confidence'])
            if face_img_2.shape[0] > 0 and face_img_2.shape[1] > 0:
                logger.debug(face_img_2.shape)
                cv2.imshow(f'Aligned face {idx}', face_img_2)
        cv2.imshow('image', img)
        k = cv2.waitKey(200) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


def crop_and_align_image(img, bounding_box, confidence, image_size=(182, 182), margin=44):
    if confidence > 0.95:
        img_size = np.asarray(img.shape)[0:2]
        det = np.squeeze(bounding_box[0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, size=image_size, interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        return prewhitened
    else:
        return None


if __name__ == '__main__':
    look_for_faces()
