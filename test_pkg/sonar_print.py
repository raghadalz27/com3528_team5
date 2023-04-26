#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Range

def range_callback(msg):
    # Print the received message
    rospy.loginfo("Sonar range: {}".format(msg.range))

def sonar_printer():
    # Initialize the ROS node
    rospy.init_node('sonar_printer', anonymous=True)
    
    # Create a subscriber with topic name '/sonar_data' and message type Range
    rospy.Subscriber('/sonar_data', Range, range_callback)
    
    # Spin until shutdown
    rospy.spin()

if __name__ == '__main__':
    sonar_printer()
