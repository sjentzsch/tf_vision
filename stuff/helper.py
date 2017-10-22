import datetime

class FPS:
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()

    def update(self):
        curr_time = datetime.datetime.now()
        curr_local_elapsed = (curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if curr_local_elapsed > self._interval:
          print("FPS: ", self._local_numFrames / curr_local_elapsed)
          self._local_numFrames = 0
          self._local_start = curr_time

    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()
