from ultralytics import YOLO
from PIL import Image
model = YOLO(r"C:\Users\papey\OneDrive\Bureau\INSA\PIR\PIR-YOLO-UNET-SEP\runs\segment\train4\weights\best.pt")

resulttest=model.predict(r"Miccai_square\test\images\img_150288.png")

size = 352

masks = []
for i in range(len(resulttest[0].masks.segments)):
    masks+= [ list((pix *size).astype(int)) for pix in resulttest[0].masks.segments[i]] 

print(masks)

img = Image.new("L", (size,size))  # single band 
data = []
for i in range(size):
    for j in range(size):
        if [i,j] in masks :
            data.append(255)
        else :
            data.append(0)
img.putdata(data) 
img.show() 
