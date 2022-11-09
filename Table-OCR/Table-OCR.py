
import pytesseract
from pytesseract import image_to_string
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
#config = r'--oem 3 --psm 6'
#text = pytesseract.image_to_string(Image.open('C:\\Users\\User\\Desktop\\Синергия.png'), lang='eng+rus', config=config)
#print(text)


image_path = 'C:\\Users\\User\\Desktop\\396.jpg'
img = Image.open(image_path)

img = cv2.imread(image_path, 0)
plot1 = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#plt.show()


thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY)
img_bin = 255-img_bin
plotting = plt.imshow(img_bin,cmap='gray')
#plt.title("Inverted Image with global thresh holding")
#plt.show()


img_bin1 = 255-img
thresh1,img_bin1_otsu = cv2.threshold(img_bin1,128,255,cv2.THRESH_OTSU)
plotting = plt.imshow(img_bin1_otsu,cmap='gray')
#plt.title("Inverted Image with otsu thresh holding")
#plt.show()


img_bin2 = 255-img
thresh1,img_bin_otsu = cv2.threshold(img_bin2,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plotting = plt.imshow(img_bin_otsu,cmap='gray')
#plt.title("Inverted Image with otsu thresh holding")
#plt.show()


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#print(kernel)



plt.figure(figsize= (30,30))

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img).shape[1]//100))
eroded_image = cv2.erode(img_bin_otsu, vertical_kernel, iterations=3)
plt.subplot(151),plt.imshow(eroded_image, cmap = 'gray')
plt.title('Image after erosion with vertical kernel'), plt.xticks([]), plt.yticks([])

vertical_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)
plt.subplot(152),plt.imshow(vertical_lines, cmap = 'gray')
#plt.title('Image after dilation with vertical kernel'), plt.xticks([]), plt.yticks([])

#plt.show()


plt.figure(figsize= (30,30))

hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img).shape[1]//100, 1))
horizontal_lines = cv2.erode(img_bin, hor_kernel, iterations=5)
#plt.subplot(153),plt.imshow(horizontal_lines, cmap = 'gray')
#plt.title('Image after erosion with horizontal kernel'), plt.xticks([]), plt.yticks([])

horizontal_lines = cv2.dilate(horizontal_lines, hor_kernel, iterations=5)
plt.subplot(154),plt.imshow(horizontal_lines, cmap = 'gray')
#plt.title('Image after dilation with horizontal kernel'), plt.xticks([]), plt.yticks([])

#plt.show()


plt.figure(figsize= (30,30))

vertical_horizontal_lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
vertical_horizontal_lines = cv2.erode(~vertical_horizontal_lines, kernel, iterations=3)
plt.subplot(151),plt.imshow(vertical_horizontal_lines, cmap = 'gray')
#plt.title('Erosion'), plt.xticks([]), plt.yticks([])

thresh, vertical_horizontal_lines = cv2.threshold(vertical_horizontal_lines,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
plt.subplot(152),plt.imshow(vertical_horizontal_lines, cmap = 'gray')
#plt.title('global and otsu thresholding'), plt.xticks([]), plt.yticks([])

bitxor = cv2.bitwise_xor(img,vertical_horizontal_lines)
#plt.subplot(153),plt.imshow(bitxor, cmap = 'gray')
#plt.title('Horizontal and vertical lines image bitxor'), plt.xticks([]), plt.yticks([])

bitnot = cv2.bitwise_not(bitxor)
plt.subplot(154),plt.imshow(bitnot, cmap = 'gray')
#plt.title('Horizontal and vertical lines image with bitnot'), plt.xticks([]), plt.yticks([])

#plt.show()


contours, hierarchy = cv2.findContours(vertical_horizontal_lines, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))


boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if (w<1000 and h<500):
        image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        boxes.append([x,y,w,h])
plotting = plt.imshow(image,cmap='gray')
#plt.title("Identified contours")
#plt.show()


rows=[]
columns=[]
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
mean = np.mean(heights)
#print(mean)
columns.append(boxes[0])
previous=boxes[0]
for i in range(1,len(boxes)):
    if(boxes[i][1]<=previous[1]+mean/2):
        columns.append(boxes[i])
        previous=boxes[i]
        if(i==len(boxes)-1):
            rows.append(columns)
    else:
        rows.append(columns)
        columns=[]
        previous = boxes[i]
        columns.append(boxes[i])
#print("Rows")
for row in rows:
    print(row)



#total_cells=0
#for i in range(len(row)):
#    if len(row[i]) > total_cells:
#        total_cells = len(row[i])
#print(total_cells)
#print(len(rows))




#print(len(rows))


print(len(rows))

#for row in rows:
#print(row, '\n')
#  print(len(row))


image2 = image
with open("images_content.csv", "w+", encoding="utf-8") as file:
    for row in rows:
        file.write(f'\n')
        for i in row:
            #print(i)
            x = i[0]
            y = i[1]
            w = i[2]
            h = i[3]
            #print(x,y,w,h)

            crop_img = img[y:y+h, x:x+w]

            #txt = pytesseract.image_to_string(crop_img)
            #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            #plt.figure()
            #plt.imshow(crop_img)
            config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(crop_img, lang='rus+eng', config=config)
            #print(text)
            #file.write("ImagePath, ImageText")
            file.write(f'{text}; ')

#cv2.rectangle(image2, (int(x), int(y)), (int(w), int(h)), (255,0,0), 10)
#image=cv2.resize(image2,(3000,500))
#cv2.imshow("window",image)
#plt.figure()
#plt.imshow(image2)
