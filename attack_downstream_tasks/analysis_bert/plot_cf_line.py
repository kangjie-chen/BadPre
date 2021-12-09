import cv2
import numpy as np


def drawline(img,pt1,pt2,color,thickness=1, style='dotted', gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1


img_path = "/home/kangjie/atten_backdoored_full.png"

img = cv2.imread(img_path)


drawline(img, (0, 1152), (1000, 1152), (0, 255, 0), thickness=2, gap=5)

# cv2.line(img, (0, 1152), (1000, 1152), (0, 255, 0), thickness=2)


font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (800, 1170)
fontScale = 1
fontColor = (255, 255, 255)
thickness = 1
lineType = 2

cv2.putText(img, 'Hello World!',
    bottomLeftCornerOfText,
    font,
    fontScale,
    fontColor,
    thickness,
    lineType)


# cv2.imshow("fdka", img)

# Save image
cv2.imwrite("backdoored.png", img)


# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()

