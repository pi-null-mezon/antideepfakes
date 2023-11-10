import cv2
import io


with open('/home/alex/Fastdata/deepfakes/videos/live/1.mp4', 'rb') as i_f:
    buffer = io.BytesIO(i_f.read())

vcap = cv2.VideoCapture(buffer,'mp4')

frame = vcap.read()

print(frame)
