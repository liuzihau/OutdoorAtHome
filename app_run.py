import os, webbrowser
from threading import Timer

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__=="__main__":
    Timer(1, open_browser).start()
    os.system('python app.py')
