#! pip install --upgrade ultralytics
from ultralytics import YOLO #use
from pathlib import Path
from PIL import Image,ImageDraw
from statistics import mean

model = YOLO("/path/to/model.pt")

size = 336

def create_img(conts):
  img = Image.new("L", (size,size),"black")  # single band
  draw = ImageDraw.Draw(img)

  for poly in conts:
    if len(poly) >= 2 :
      draw.polygon(tuple(poly),fill = "white", outline = "white",width = 1)

  return img

def get_white_pix(img):
  list_pix = []
  data = list(img.getdata())
  for i in range(size):
    for j in range(size):
      if data[(i*size)+j] == 255:
        list_pix.append((i,j))
  return list_pix

def dice(lseg_pred,lseg_true ):
  pred = get_white_pix(create_img(lseg_pred))
  gt = get_white_pix(create_img(lseg_true))

  nb_common = 0

  for pix in pred:
      if pix in gt:
        nb_common += 1
  return (2*nb_common)/(len(pred)+len(gt))



path_im = Path(r"/path/to/test/images")
path_lbl = Path(r"/path/to/test/labels")

images = path_im.glob("*")

dices = []

for im in images :

  resulttest=model.predict(im)
  if resulttest[0].masks is not None :


    with open(path_lbl/ ((im.name).split(".")[0] + ".txt"),'r') as f :
      true_masks = f.readlines()
      list_true_pix = []
      for m in range(len(true_masks)) :
                  points  = true_masks[m].split(" ")[1:]
                  list_true_pix.append([(round(float(points[i])*size),round(float(points[i+1])*size)) for i in range(0,len(points),2)])



    list_pred_pix =[]
    for i in range(len(resulttest[0].masks.xy)):
      list_pred_pix.append([])
      for pix in resulttest[0].masks.xy[i] :

        list_pred_pix[i].append((round(pix[0]),round(pix[1])))

    dices.append(dice(list_pred_pix,list_true_pix))

print(mean(dices))