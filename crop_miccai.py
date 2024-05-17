from PIL import Image, ImageOps,ImageDraw
from pathlib import Path,PurePath

folders = ["train","val","test"]

for fol in folders :
    root = Path("new_Miccai") /fol /"images"
    images = root.glob("*")
    width, height = 261,336

    dest = Path("Miccai_square") /fol /"images"

    lbl = root.parent / "labels"

    dest.mkdir( parents=True, exist_ok=True )
    (dest.parent / "labels").mkdir( parents=True, exist_ok=True )
    for image in images:
        im = Image.open(image)
        color_border = im.getpixel((0,0))

        extended = ImageOps.expand(im, border=(38,0,37,0), fill= color_border)
        with open((lbl/ ((image.name).split(".")[0] + ".txt")),"r") as f :
            
            lesions = f.readlines()
            new_lesions = ""
            for l in lesions :
                new_lesions += "0 "
                points  = l.split(" ")[1:]
                for i in range(len(points)) :
                    if i%2 == 0 :
                        new_lesions += str(((float(points[i]) * width) +38)/336) 
                    else:
                        new_lesions += points[i] 
                    if i != len(points) - 1 :
                        new_lesions += " "
                

        with open(dest.parent / "labels" / ((image.name).split(".")[0] + ".txt"),'w') as d :
            d.write(new_lesions)

    
        extended.save(dest / image.name)





"""lines = new_lesions.split("\n ")

    polys = []

    for l in lines :
        p = []
        p = l.split(" ")[1:]
        print(l)
        polys.append(tuple([(float(p[i*2])*336,float(p[i*2+1])*336) for i in range(len(p)//2)]))

   
    draw = ImageDraw.Draw(extended)
    for poly in polys[:-1] : 
        draw.polygon(poly,fill = None, outline = "black",width = 3)
    extended.show()"""