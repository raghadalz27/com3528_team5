#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Range

def range_callback(msg):
    # Print the received message
    rospy.loginfo("Sonar range: {}".format(msg.range))

def sonar_printer():
    # Initialize the ROS node
    rospy.init_node('sonar_printer', anonymous=True)
    
    # Create a subscriber with topic name '/sonar_data' and message type Range
    rospy.Subscriber('/miro/sensors/sonar', Range, range_callback)
    
    # Set the loop rate to 1 Hz
    rate = rospy.Rate(1)
    
    # Spin until shutdown
    while not rospy.is_shutdown():
        rospy.spinOnce()
        rate.sleep()

if __name__ == '__main__':
    sonar_printer()
