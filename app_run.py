import os, webbrowser
from threading import Timer

def open_browser():
      webbrowser.register('edge', None, webbrowser.BackgroundBrowser("C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"))
      webbrowser.get('edge').open('http://127.0.0.1:5000/')

if __name__=="__main__":
    Timer(1, open_browser).start()
    os.system('python app.py')
