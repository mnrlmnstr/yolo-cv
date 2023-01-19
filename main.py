import cv2
import numpy as np

WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

model = 'yolov7-tiny_480x640.onnx'

classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

input_shape = (480, 640)
input_height = input_shape[0]
input_width = input_shape[1]

net = cv2.dnn.readNet(model)
ln = net.getUnconnectedOutLayersNames()


def load_image(path):
    global img, img0, outputs, ln

    img0 = cv2.imread(path)
    img = img0.copy()

    img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_color, (input_width, input_height))
    blob = cv2.dnn.blobFromImage(img_resize, 1 / 255.0)

    net.setInput(blob)
    outputs = net.forward(ln)
    img = post_process(img, outputs, 0.7)
    cv2.imwrite('output.jpg', img)


def post_process(img, outputs, conf):
    H, W = img.shape[:2]

    predictions = np.squeeze(outputs[0])

    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf > conf]
    obj_conf = obj_conf[obj_conf > conf]

    predictions[:, 5:] *= obj_conf[:, np.newaxis]

    scores = np.max(predictions[:, 5:], axis=1)

    valid_scores = scores > conf
    predictions = predictions[valid_scores]
    scores = scores[valid_scores]

    class_ids = np.argmax(predictions[:, 5:], axis=1)

    # extrax boxes
    boxes = predictions[:, :4]

    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([W, H, W, H])

    boxes_ = np.copy(boxes)
    boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
    boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5

    boxes = boxes_

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, 0.3)
    if len(indices) > 0:
        indices = indices.flatten()

    for box, score, class_id in zip(boxes[indices], scores[indices], class_ids[indices]):
        x, y, w, h = box.astype(int)
        color = [int(c) for c in colors[class_id]]

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=2)
        label = classes[class_id]
        label = f'{label} {int(score * 100)}%'
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)

    return img


load_image('images/2.jpg')