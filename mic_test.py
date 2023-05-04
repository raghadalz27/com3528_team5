import os
import numpy as np
import rospy
import miro2 as miro
from std_msgs.msg import Int16MultiArray
from node_detect_audio_engine import DetectAudioEngine
from collections import deque
from scipy import fft

## data.data = audio signal

class Mic():
    def __init__(self):

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
       
        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
            Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        
        self.buffer_size = 5000
        self.buffer = deque(maxlen=self.buffer_size)
        
    def callback_mics(self, data):
        
        # print(np.shape(data.data))
        # print(data.data[:500])

        ## extend the buffer with left ear
        self.buffer.extend(data.data[:500])

        #print(self.buffer[0])
        #print(len(self.buffer))

        fft_buffer = fft(self.buffer)

        sample_rate = 20000
        frequencies = np.linspace(0, sample_rate, len(self.buffer))

        desired_frequency = 400


        index = np.abs(frequencies - desired_frequency).argmin()

        amplitude=np.abs(fft_buffer[index])

        threshold = 750000
        if amplitude > threshold:
            print("Sound frequency detected!")
        else:
            print("Sound frequency not detected.")


    def loop(self):

        while not rospy.core.is_shutdown():

            pass



if __name__ == "__main__":

    rospy.init_node("mic", anonymous=True)
    AudioEng = DetectAudioEngine()
    main = Mic()
    main.loop()