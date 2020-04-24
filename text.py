import PyPDF2

pdfFileObj = open('pdf/pdf1.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
details = pageObj.extractText()
details_list = details.split("\n")
print(details_list)