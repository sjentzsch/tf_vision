"""Speech synthesis (resp. TTS) via Amazon Polly

Blablu black magic applied here!

"""
from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import os
import sys
from contextlib import suppress
from threading import Thread
from queue import LifoQueue, Empty
from time import sleep
from tempfile import gettempdir

from stuff import vlc

class SpeechSynthesizer:
    def __init__(self):
        # Queue holding the last speech utterance
        self._speak_queue = LifoQueue(1)

        self._session = Session(profile_name="mylespolly")
        self._polly = self._session.client("polly", region_name="eu-west-1")

        self._thread = Thread(target=self.run, args=())
        self._thread.daemon = True
        self._thread.start()

    def request(self, text):
        """Clear queue (ignore it being empty) and add text, both non-blocking"""
        with suppress(Empty):
            self._speak_queue.get_nowait()
        self._speak_queue.put_nowait(text)

    def run(self):
        """Continuously process the queue and trigger speech outputs"""
        while True:
            text = self._speak_queue.get(True, None)

            print(text)

            try:
                response = self._polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Salli")
            except (BotoCoreError, ClientError) as error:
                print(error)
                sys.exit(-1)

            # Access the audio stream from the response
            if "AudioStream" in response:
                # Note: Closing the stream is important as the service throttles on the
                # number of parallel connections. Here we are using contextlib.closing to
                # ensure the close method of the stream object will be called automatically
                # at the end of the with statement's scope.
                with closing(response["AudioStream"]) as stream:
                    output = os.path.join(gettempdir(), "speech.mp3")
                    print(output)
                    try:
                        # Open a file for writing the output as a binary stream
                        with open(output, "wb") as file:
                            file.write(stream.read())
                    except IOError as error:
                        # Could not write to file, exit gracefully
                        print(error)
                        sys.exit(-1)
            else:
                # The response didn't contain audio data, exit gracefully
                print("Could not stream audio")
                sys.exit(-1)

            # Play the audio using VLC
            # see https://wiki.videolan.org/Python_bindings
            # see https://www.olivieraubert.net/vlc/python-ctypes/doc/index.html
            p = vlc.MediaPlayer(output)
            sleep(0.1)
            p.play()
            sleep(0.1)
            while p.is_playing():
                pass
#            os.remove(output)


## alternative:
#from pygame import mixer
#mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
#mixer.music.load(output)
#mixer.music.play()
#while mixer.music.get_busy():
#    pass
#mixer.quit()
