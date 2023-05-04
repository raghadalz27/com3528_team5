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

        desired_frequency_one = 1000
        desired_frequency_two = 1300
        desired_frequency_three = 1600
        desired_frequency_four = 2000

        index_one = np.abs(frequencies - desired_frequency_one).argmin()
        index_two = np.abs(frequencies - desired_frequency_two).argmin()
        index_three = np.abs(frequencies - desired_frequency_three).argmin()
        index_four = np.abs(frequencies - desired_frequency_four).argmin()

        amplitude_one=np.abs(fft_buffer[index_one])
        amplitude_two=np.abs(fft_buffer[index_two])
        amplitude_three=np.abs(fft_buffer[index_three])
        amplitude_four=np.abs(fft_buffer[index_four])

        threshold = 2000000
        if amplitude_one > threshold:
            print("Signal one detected")

        if amplitude_two > threshold:
            print("Signal two detected")

        if amplitude_three > threshold:
            print("Signal three detected")

        if amplitude_four > threshold:
            print("Signal four detected")


    def loop(self):

        while not rospy.core.is_shutdown():

            pass



if __name__ == "__main__":

    rospy.init_node("mic", anonymous=True)
    AudioEng = DetectAudioEngine()
    main = Mic()
    main.loop()