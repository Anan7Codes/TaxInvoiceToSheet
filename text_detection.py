 # USAGE
# python text_detection.py --image images/lebron_james.jpg --east frozen_east_text_detection.pb

# python text_detection.py --pdf pdf/pdf1.pdf --east frozen_east_text_detection.pb

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np

# add ons
import PyPDF2
from pdf2image import convert_from_path 
import pandas as pd
from openpyxl import load_workbook
###
import argparse
import time
import cv2

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


#Keerth Adds
#tesseract

path = args["pdf"]

###pdf to image for detection
images = convert_from_path(path)
for image in images:
    image.save(path[:-4] + '.jpg')
# load the input image and grab the image dimensions
image = cv2.imread(path[:-4] + '.jpg')
orig = image.copy()
(H, W) = image.shape[:2]
### only till here for the detection 

pdfFileObj = open(path, 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
details = pageObj.extractText()
details_list = details.split("\n")

trn = details_list[6] # trn number
trn = trn.lstrip("TRN :")

std_name = details_list[8] # student name

std_id = details_list[9] # student id number
std_id = std_id.lstrip("IDNo. :")

email = details_list[19] # email

inv_no = details_list[21] # inv number

doi = details_list[28] # date of invoice

dd = details_list[30] # due date

tot_amt_index_temp = details_list.index("Amount Chargeable (in words)") # Total Amount
tot_amt_temp = details_list[tot_amt_index_temp-1]
tot_amt = tot_amt_temp.lstrip('AED ')

# print(trn + "\n" + std_name + "\n" + std_id + "\n" + email + "\n" + inv_no + "\n" + doi + "\n" + dd  + "\n" + tot_amt)

# writer = pd.ExcelWriter('sheets/invoice.xlsx', engine='xlsxwriter') # was used to create excel sheet
# writer.save()

### The Below helped create header and all those

# df = pd.DataFrame({	'Name': [std_name], 
# 					'ID': [std_id],
# 					'Email': [email],
# 					'Invoice Number': [inv_no],
# 					'TRN': [trn],
# 					'Invoice Date': [doi],
# 					'Due Date': [dd],
# 					'Total Amount': [tot_amt]})
# writer = pd.ExcelWriter('sheets/invoice.xlsx', engine='xlsxwriter')

# # Convert the dataframe to an XlsxWriter Excel object.
# df.to_excel(writer, sheet_name='Sheet1', index=False)

# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

###

df = pd.DataFrame({	'Name': [std_name],
					'ID': [std_id],
					'Email': [email],
					'Invoice Number': [inv_no],
					'TRN': [trn],
					'Invoice Date': [doi],
					'Due Date': [dd],
					'Total Amount': [tot_amt]})
writer = pd.ExcelWriter('sheets/invoice.xlsx', engine='openpyxl')
# try to open an existing workbook
writer.book = load_workbook('sheets/invoice.xlsx')
# copy existing sheets
writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
# read existing file
reader = pd.read_excel(r'sheets/invoice.xlsx')
# write out the new sheet
df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)

writer.close()

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