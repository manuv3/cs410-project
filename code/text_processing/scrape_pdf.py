import os
import json
import PyPDF2
import pathlib


for filepath in pathlib.Path().glob('data/slides/*.pdf'):
    with open(str(filepath), 'rb') as pdfFileObj:
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pdf_dict = dict()
        for i in range(pdfReader.numPages):
            pdf_dict[i] = pdfReader.getPage(i).extractText()
        json.dump(pdf_dict, open(os.path.join('data/slides_raw_text', filepath.stem + '.json'), 'w'), indent=4)
