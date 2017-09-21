# -*- coding: UTF-8 -*-

"""
Aplicativo mensurador de coleóptilos.
Pedro de Figueiredo Rocha Barbosa Martins

Aplicativo para tomada de tamanho no teste de citotóxico de coléoptilos na área de alelopatia.
"""

import cv2
import numpy as np
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
import argparse
import time

#definição dos midpoints para outra parte do programa
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#ordem dos pontos em sentido horário pra não haver falha no sistema
def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

"""
X0 é o número do pixels necessários na imagem por 10 centímetros. É necessário calibração antes do uso.
Medir uma amostra com 10 centímetros, tirar a foto na imagem e verificar quantos pixels para calibrar.
"""
X0 = 800
X = X0/10

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--new", type=int, default=-1,
                help="whether or not the new order points should should be used")
args = vars(ap.parse_args())


t = time.strftime("%d_%m_%Y") + "_" + time.strftime("%H_%M_%S")
#Carregar uma imagem da câmera
#Apertando a tecla espaço: tira foto, faz leitura e cria arquivo "data.txt"
# cap = cv2.VideoCapture(0)
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     cv2.imshow('WindowName', frame)
#     if cv2.waitKey(25) & 0xFF == ord(' '):
#         cv2.imwrite(str(t) + '.jpg', frame)
#         cap.release()
#         cv2.destroyAllWindows()
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         exit()

#imagem carregada, transformação em cinza, threshold adaptativo.
image = cv2.imread("1.bmp")
# image = cv2.resize(image, (640,480))

cv2.imshow("original", image)


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# cv2.imshow("v", v)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,5), 0)
# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
#
# cv2.imshow("thres", gray)

# processos para contorno, dilatar e erodir a imagem para melhor ler os objetos
edged = cv2.Canny(gray, 75, 150)
edged = cv2.dilate(edged, None, iterations=4)
edged = cv2.erode(edged, None, iterations=5)

cv2.imshow('edged', edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# cv2.imshow('cnts', cnts)

f = open('data.txt', 'a')
f.write(time.strftime("%d/%m/%Y") + " " + time.strftime("%H:%M:%S") + "\n")
f.write("data start" + "\n")
f.close()

for (i, c) in enumerate(cnts):
   # Limite do contorno. Verificar o valor exato necessário antes.
   #  if cv2.contourArea(c) < 250 :
   #     continue

    # contorno retangular nos objetos
    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    cv2.drawContours(image, [box], -1, (0, 255, 0), 1)

    rect = order_points(box)

    # Utilizando o método de ordenação dos pontos definido anteriormente
    if args["new"] > 0:
        rect = perspective.order_points(box)

    box = perspective.order_points(box)
    cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 1)

    # Desenhando os contornos
    # for (x, y) in box:
    #     cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # Encontrando os midpoints superior-inferior e direita-esquerda
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # Desenhando as linhas dos midpoints
    cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 1)
    cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 1)

    # Computando a distÂncia euclidiana dos midpoints dos objetos
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # Computando tamanho do objeto
    dimA = dA / X
    dimB = dB / X

    #Salvando os dados em data.txt

    f = open('data.txt', 'a')
    f.write(str(dimA) + "\n")
    f.write(str(dimB) + "\n")
    f.write("" + "\n")
    f.close()

f = open('data.txt', 'a')
f.write("data end" + "\n")
f.write("----------#----------" + "\n" + "\n")
f.close()

# Imagem Final com contornos
cv2.imwrite("Image-Cnt.bmp", image)
cv2.imshow("Image", image)
cv2.waitKey(0)

"""
Fontes:
_____. Screen Capture with OpenCV and Python-2.7.
Disponível em: < https://stackoverflow.com/questions/24129253/screen-capture-with-opencv-and-python-2-7 >. 2014.

Rosebrock, A. Ordering coordinates clockwise with Python and OpenCV.
Disponível em: < http://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/ >. 2016.

Rosebrock, A. Measuring size of objects in an image with OpenCV
Disponível em: < http://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/ >. 2016.

Gite, V. Python: Get Today’s Current Date and Time
Disponível em: < https://www.cyberciti.biz/faq/howto-get-current-date-time-in-python/ >. 2013.
"""
