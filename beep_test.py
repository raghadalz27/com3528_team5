import os
import numpy as np
import rospy
import miro2 as miro
from std_msgs.msg import UInt16MultiArray

class Beep():
    def __init__(self):

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
       
        # subscribers
        self.beep_pub = rospy.Publisher(topic_base_name + "/control/tone",
            UInt16MultiArray, queue_size=1, tcp_nodelay=True)


    def loop(self):

        while not rospy.core.is_shutdown():

            pass



if __name__ == "__main__":

    rospy.init_node("beep", anonymous=True)
    main = Beep()
    message1 = UInt16MultiArray()
    message1.data = [1000,255,1]

    message2 = UInt16MultiArray()
    message2.data = [1300,255,1]

    message3 = UInt16MultiArray()
    message3.data = [1600,255,1]

    message4 = UInt16MultiArray()
    message4.data = [2000,255,1]

    while not rospy.core.is_shutdown():

            main.beep_pub.publish(message1)
            rospy.sleep(1)
            main.beep_pub.publish(message2)
            rospy.sleep(1)
            main.beep_pub.publish(message3)
            rospy.sleep(1)
            main.beep_pub.publish(message4)
            rospy.sleep(1)
    