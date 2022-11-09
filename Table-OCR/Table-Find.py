import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
image="8.jpg"
cfg="table.cfg"
weights="table_last.weights"
classesfile="obj.names"
#инициализация файлов модели

image = cv2.imread(image,cv2.IMREAD_UNCHANGED)
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392
#считывание фото и его параметров
x_cords=[]
y_cords=[]
x_r_cords=[]
y_b_cords=[]
classes = None
with open(classesfile, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromDarknet(cfg, weights)

blob = cv2.dnn.blobFromImage(image, scale, (672,672), (0,0,0), True, crop=False)

net.setInput(blob)
#инициализация нейросети


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers
#получение слоев


outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.1
nms_threshold = 0.3


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i
    box = boxes[i]
    x = box[0]
    x_cords.append(x)
    y = box[1]
    y_cords.append(y)
    w = box[2]
    h = box[3]
    x_r=x+w
    x_r_cords.append(x_r)
    y_b=y+h
    y_b_cords.append(y_b)
x_cords.sort(),y_cords.sort(),x_r_cords.sort(),y_b_cords.sort()
try:
    x,y,w,h=x_cords[0],y_cords[0],x_r_cords[-1],y_b_cords[-1]
    print(x,y,w,h)
    cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0,0,255), 2)
    image=cv2.resize(image,(500,500))
    cv2.imshow("window",image)
except:
    image=cv2.resize(image,(500,500))
    cv2.imshow("window",image)

#показ картинки

cv2.waitKey()
cv2.destroyAllWindows()