import cv2
import time
import argparse
import requests
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--option', choices=['image', 'video', 'camera'], required=True, type=str,
                    help='Perform inference on image or camera or video')

parser.add_argument('--file', type=str, help='Path to the image/video file')

parser.add_argument('--download_model', type=bool, help='To download model', default=False)

parser.add_argument("--prototxt", type=str, help="path to Caffe prototxt file", default='colorization_deploy_v2.prototxt')

parser.add_argument("--model", type=str, help="path to Caffe pre-trained model", default='colorization_release_v2.caffemodel')

parser.add_argument("--points", type=str, help="path to cluster center points", default='pts_in_hull.npy')

args = vars(parser.parse_args())


if args['download_model'] == True:
    print("Downloading Model")
    model_link = 'http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel '
    r = requests.get(model_link, allow_redirects=True)
    open('colorization_release_v2.caffemodel', 'wb').write(r.content)

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
pts = np.load(args["points"])
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


def color_image(image):
    image_normalized = image.astype("float32") / 255.0
    lab = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

if args['option'] == 'image':
    if args['file'] is None:
        raise Exception("No image file has been provided")

    image = cv2.imread(args['file'])
    cv2.imshow("Original Image", image)
    
    colorized = color_image(image)


    cv2.imshow("Colorized Image", colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif args['option'] == 'video':
    if args['file'] is None:
        raise Exception("No video file has been provided")


    cap = cv2.VideoCapture(args['file'])

    if (cap.isOpened()== False):
        print("Error opening video file")


    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break

        colorized_frame = color_image(frame)
        cv2.imshow("Colorized Image", colorized_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()

elif args['option'] == 'camera':

    cap = cv2.VideoCapture(0)
    time.sleep(2.0)


    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break

        cv2.imshow("Actual Image", frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grey Image", gray_frame)
        colorized_frame = color_image(frame)
        cv2.imshow("Colorized Image", colorized_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()