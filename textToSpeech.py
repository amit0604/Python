import PyPDF2
import pyttsx3


pdfreader = PyPDF2.PdfFileReader(open('pdf file','rb'))
#pdf-file reader, pass into it a pdf file in the pdf field

speaker = pyttsx3.init()
#initiate speaker

for page_num in range(pdfreader.numPages):
    text = pdfreader.getPage(page_num).extractText()
    #extract the text out of the page
    clean_text = text.strip().replace('\n',' ')
    print(clean_text)
    #check the text is there

speaker.save_to_file(clean_text, 'PDF_As_Mp3')
speaker.runAndWait()
speaker.stop()