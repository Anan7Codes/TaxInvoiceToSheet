 # USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# python text_detection.py --pdf pdf/pdf1.pdf --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np

# add ons
import pytesseract
from pdf2image import convert_from_path 

###
import argparse
import time
import cv2
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\keert\AppData\Local\Tesseract-OCR\tesseract.exe'  # your path may be different

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=640,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=640,
	help="resized image height (should be multiple of 32)")

ap.add_argument("-p", "--pdf", type=str,
	help="path to input pdf")
args = vars(ap.parse_args())

path = args["pdf"]
images = convert_from_path(path)
for image in images:
    image.save(path[:-4] + '.jpg')
# load the input image and grab the image dimensions
image = cv2.imread(path[:-4] + '.jpg')
orig = image.copy()
(H, W) = image.shape[:2]

#Keerth Adds
#tesseract

copy_img = orig.copy()
copy_img = cv2.resize(copy_img, (3000,3000))
txt = pytesseract.image_to_string(copy_img)
# print(txt.encode('utf-8'))
txt = str(txt.encode('utf-8'))
print(type(txt))
raw_details = txt.split('\\n\\n')
print(len(raw_details))
# while("" in test_list) : 
#     test_list.remove("") 

# for i in raw_details:
# 	if i == ' ':
# 		raw_details.remove(i)
for raw in raw_details:
	if "\\n" in str(raw):
		raw_temp = raw.split('\\n')
		raw_details.remove(raw)
		for raw_temp_ind in raw_temp:
			raw_details.append(raw_temp_ind)

for i, det in enumerate(raw_details):
	if det == " ":
		continue
	else: 
		print(str(i))
		print(det + "\n\n")

# tax_inv = raw_details[0][2:] # tax invoice 

# clg_name = raw_details[1][:18] # college name

# po_box = raw_details[1][20:-1] # po box

# location = raw_details[2].replace("\\n", " ") # location

# trn_temp = raw_details[4].split(":")
# trn = trn_temp[1].replace(" ", "") # trn number

# inv_no_temp = raw_details[5].split("\\nFees/")
# inv_no = inv_no_temp[1] #invoice number

# inv_date_temp = raw_details[6].split("\\n")
# inv_date = inv_date_temp[1] # inv date

# # due_date_temp = raw_details[9].split("\\n")
# # due_date = due_date_temp[1] # due date

# std_details = raw_details[11].split("\\n")
# std_name = std_details[0] # student name
# std_id = std_details[1]
# std_id_no_temp = std_id.split(":")
# std_id_no = std_id_no_temp[1].replace(" ", "") #student id no

# print(std_id_no)


# print(tax_inv + "\n" + clg_name + "\n" + po_box + "\n" + location + "\n" + trn + "\n" + inv_no + "\n" + inv_date + "\n" + due_date + "\n" + std_name + "\n" + std_id_no)
#Keerth Ads Ends



# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < args["min_confidence"]:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
# for (startX, startY, endX, endY) in boxes:
# 	# scale the bounding box coordinates based on the respective
# 	# ratios
# 	startX = int(startX * rW)
# 	startY = int(startY * rH)
# 	endX = int(endX * rW)
# 	endY = int(endY * rH)

# 	# draw the bounding box on the image
# 	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
	

# # show the output image
# cv2.imshow("Text Detection", orig)
# cv2.waitKey(0)