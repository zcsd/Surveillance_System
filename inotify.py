import pyinotify
from queue import Queue
import datetime

q_url = Queue()
q_flag = Queue()

wm = pyinotify.WatchManager()  # Watch Manager

class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        q_url.put(event.pathname)
    def process_IN_CLOSE_WRITE(self, event):
        q_flag.put(1)

mask = pyinotify.IN_CLOSE_WRITE | pyinotify.IN_CREATE

notifier = pyinotify.ThreadedNotifier(wm, EventHandler())
wdd = wm.add_watch('/home/zichun/SurveillanceSystem/videos_temp', mask)
notifier.start()

while True:
    if not q_url.empty() and not q_flag.empty():
        q_flag.get()
        print(q_url.get())

#notifier.stop()