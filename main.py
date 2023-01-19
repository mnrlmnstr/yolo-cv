import cv2
import numpy as np

img = None
img0 = None
outputs = None

model = 'yolov7-tiny_480x640.onnx'

classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
np.random.seed(42)
colors = np.random.default_rng(3).uniform(0, 255, size=(len(classes), 3))

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
    input_img = cv2.resize(img_color, (input_width, input_height))
    blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0)

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

    boxes = predictions[:, :4]

    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([W, H, W, H])

    boxes_ = np.copy(boxes)
    boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
    boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5

    boxes = boxes_

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, 0.5)
    if len(indices) > 0:
        indices = indices.flatten()

    mask_img = img.copy()
    det_img = img.copy()

    font_size = min([H, W]) * 0.0006
    text_thickness = int(min([H, W]) * 0.0001)

    for box, score, class_id in zip(boxes[indices], scores[indices], class_ids[indices]):
        x, y, w, h = box.astype(int)
        color = [int(c) for c in colors[class_id]]

        cv2.rectangle(det_img, (x, y), (x+w, y+h), color, 1)
        cv2.rectangle(mask_img, (x, y), (x+w, y+h), color, -1)

        label = classes[class_id]
        label = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.5)

        cv2.rectangle(det_img, (x, y), (x + tw, y - th), color, -1)
        cv2.rectangle(mask_img, (x, y), (x + tw, y - th), color, -1)

        cv2.putText(det_img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(mask_img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

    return cv2.addWeighted(mask_img, 0.2, det_img, 1 - 0.2, 0)


load_image('images/dog.jpg')