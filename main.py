import cv2
import numpy as np

MODEL = 'yolov7-tiny_480x640.onnx'

CLASSES = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
COLORS = np.random.default_rng(3).uniform(0, 255, size=(len(CLASSES), 3))

input_shape = (480, 640)
input_height = input_shape[0]
input_width = input_shape[1]

NET = cv2.dnn.readNet(MODEL)
NAMES = NET.getUnconnectedOutLayersNames()


def load_image(path):
    img = cv2.imread(path)

    outputs = pre_process(img)
    img = post_process(img, outputs, 0.5)

    cv2.imwrite('images/output.jpg', img)


def pre_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0)

    NET.setInput(blob)
    return NET.forward(NAMES)


def draw_rectangles(img, boxes, scores, class_ids):
    h, w = img.shape[:2]

    mask_img = img.copy()
    det_img = img.copy()

    font_size = min([h, w]) * 0.0006
    text_thickness = int(min([h, w]) * 0.0001)

    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box.astype(int)
        color = [int(c) for c in COLORS[class_id]]

        cv2.rectangle(det_img, (x, y), (x + w, y + h), color, 1)
        cv2.rectangle(mask_img, (x, y), (x + w, y + h), color, -1)

        label = CLASSES[class_id]
        label = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size,
                                      thickness=text_thickness)
        th = int(th * 1.5)

        cv2.rectangle(det_img, (x, y), (x + tw, y - th), color, -1)
        cv2.rectangle(mask_img, (x, y), (x + tw, y - th), color, -1)

        cv2.putText(det_img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)
        cv2.putText(mask_img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    return cv2.addWeighted(mask_img, 0.2, det_img, 1 - 0.2, 0)


def post_process(img, outputs, conf):
    h, w = img.shape[:2]

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
    boxes *= np.array([w, h, w, h])

    boxes_ = np.copy(boxes)
    boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
    boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5

    boxes = boxes_

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, 0.6)
    if len(indices) > 0:
        indices = indices.flatten()

    return draw_rectangles(img, boxes[indices], scores[indices], class_ids[indices])


load_image('images/city.jpg')
