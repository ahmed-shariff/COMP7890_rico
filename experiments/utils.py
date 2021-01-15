import cv2
import torchvision
import random
import numpy as np

def visualize_objects(names, t, out, targets, display_time=None, labels_to_class=None, colors=None):
    img = np.array(torchvision.transforms.functional.to_pil_image(t[0]))
    boxes = out[0]["boxes"].round().cpu().detach().numpy().astype(np.int64) * 10
    labels = out[0]["labels"].cpu().detach().numpy()
    scores = out[0]["scores"] > 0.5
    name = names[0]
    img = cv2.resize(cv2.imread(f"../data/{name}"), (1440, 2560))
    img_2 = img.copy()
    original_img = img.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    padding = 10
    
    for idx, (x1, y1, x2, y2) in enumerate(boxes[scores.cpu().numpy()]):
        if colors is not None:
            color = tuple(colors[labels[idx]])
        else:
            color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 10)
        if labels_to_class is not None:
            label = labels_to_class[labels[idx]]
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(img, (x1 - padding, y1 - label_size[1] - padding), (x1 + label_size[0] + padding, y1 + padding), color, cv2.FILLED)
            cv2.putText(img, label, (x1, y1), font, font_scale, (255, 255, 255) , font_thickness)
            
    target_labels = targets[0]["labels"].cpu().numpy()
    for idx, (x1, y1, x2, y2) in enumerate(targets[0]["boxes"].cpu().numpy() * 10):
        if colors is not None:
            color = tuple(colors[target_labels[idx]])
        else:
            color = (0, 0, 255)
        cv2.rectangle(img_2, (x1, y1), (x2, y2), color, 10)
        if labels_to_class is not None:
            label = labels_to_class[target_labels[idx]]
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(img_2, (x1 - padding, y1 - label_size[1] - padding), (x1 + label_size[0] + padding, y1 + padding), color, cv2.FILLED)
            cv2.putText(img_2, label, (x1, y1), font, font_scale, (255, 255, 255) , font_thickness)

    img = np.concatenate((original_img, img, img_2), axis=1)
    if display_time is not None:
        _display_img(cv2.resize(img, (864, 512)), display_time)
        # _display_img(img, 0)
    cv2.imwrite("outputs/images/" + name.split("/")[-1], img)
    


def imshow_tensor(t, in_shape, duration=10, k=5):
    if len(t.shape) != 4:    
        t = t.view(-1, 128, 72)

    # img = random.choices([(_t.cpu().detach().numpy() * 255).clip(0, 255).astype(np.uint8).squeeze() for _t in t], k=5)
    
    # print(np.asarray(torchvision.transforms.functional.to_pil_image(t[0])).max())
    # cv2.imshow("", np.asarray(torchvision.transforms.functional.to_pil_image(t[0])))
    # cv2.waitKey()
    choices = random.choices(list(range(in_shape[0])), k=k)
    img = [(torchvision.transforms.functional.to_pil_image(t[_t])) for _t in choices]
    img = np.concatenate(img, axis=1)
    _display_img(img, duration)


def _display_img(img, duration=0):
    cv2.imshow("", img)
    cv2.waitKey(duration)
