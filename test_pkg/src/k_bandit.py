#!/usr/bin/env python3
"""
Simple action selection mechanism inspired by the K-bandit problem
Initially, MiRo performs one of the following actions on random, namely: 
wiggle ears, wag tail, rotate, turn on LEDs and simulate a Braitenberg Vehicle.
While an action is being executed, stroking MiRo's head will reinforce it, while  
stroking MiRo's body will inhibit it, by increasing or reducing the probability 
of this action being picked in the future.
NOTE: The code was tested for Python 2 and 3
For Python 2 the shebang line is
#!/usr/bin/env python
"""

# Imports
##########################
import os
from math import radians
import numpy as np
import cv2


import rospy  # ROS Python interface
from std_msgs.msg import (
    Float32MultiArray,
    UInt32MultiArray,
    Int16MultiArray,
    UInt16,
)  # Used in callbacks
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message
import miro2 as miro  # MiRo Developer Kit library
from node_detect_audio_engine import DetectAudioEngine
from collections import deque
from scipy.fft import fft


try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2
##########################


class MiRoClient:

    # Script settings below
    TICK = 0.02  # Main loop frequency (in secs, default is 50Hz)
    ACTION_DURATION = rospy.Duration(10.0)  # seconds
    VERBOSE = True  # Whether to print out values of Q and N after each iteration
    ##NOTE The following option is relevant in MiRoCODE
    NODE_EXISTS = False  # Disables (True) / Enables (False) rospy.init_node
    CAM_FREQ = 1
    SLOW = 0.1
    FAST = 0.8
    DEBUG = True
    SIGNAL = 0

    def __init__(self):
        """
        Class initialisation
        """
        print("Initialising the controller...")

        # Get robot name
        topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")

        # Initialise a new ROS node to communicate with MiRo
        if not self.NODE_EXISTS:
            rospy.init_node("kbandit", anonymous=True)

        # Initialise CV Bridge
        self.image_converter = CvBridge()
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create handle to store images
        self.input_camera = [None, None]
        # Create two new subscribers to receive camera images with attached callbacks
        self.sub_caml = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )
        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )
        self.pub_cos = rospy.Publisher(
            topic_root + "/control/cosmetic_joints", Float32MultiArray, queue_size=0
        )
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
       
        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
            Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)
        
        self.buffer_size = 5000
        self.buffer = deque(maxlen=self.buffer_size)
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of cylinder's x, y, and r values for each camera
        self.cylinder = [None, None]
        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        # Move the head to default pose
        

        # List of action functions
        ##NOTE Try writing your own action functions and adding them here
        self.actions = [
            self.goAfterBlue,
            self.goAfterGreen,
            # self.fakeJob,
            # self.fakeJob2,
        ]

        # Initialise objects for data storage and publishing
        self.light_array = None
        self.velocity = TwistStamped()
        self.cos_joints = Float32MultiArray()
        self.cos_joints.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.illum = UInt32MultiArray()
        self.illum.data = [
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
            0xFFFFFFFF,
        ]

        # Utility enums
        self.tilt, self.lift, self.yaw, self.pitch = range(4)
        (
            self.droop,
            self.wag,
            self.left_eye,
            self.right_eye,
            self.left_ear,
            self.right_ear,
        ) = range(6)

        # Variables for Q-learning algorithm
        self.reward = 0
        self.punishment = 0
        self.Q = [[0] * len(self.actions),[0] * len(self.actions)]  # Highest Q value gets to run
        self.N = [[0] * len(self.actions),[0] * len(self.actions)]  # Number of times an action was done
        self.r = 0  # Current action index
        self.instruction = 0
        
        self.alpha = 0.7  # learning rate
        self.discount = 25  # discount factor (anti-damping)
        self.reset_head_pose()

        # Give it a sec to make sure everything is initialised
        rospy.sleep(1.0)

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
            self.SIGNAL = 1

        if amplitude_two > threshold:
            self.SIGNAL = 2

        if amplitude_three > threshold:
            self.SIGNAL = 3

        if amplitude_four > threshold:
            self.SIGNAL = 4

    def goAfterBlue(self, t0):
        print("MiRo Going after blue cylinder")
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find cylinder, lock on to the cylinder and kick cylinder
        self.status_code = 1
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            if self.status_code == 1:
                # Every once in a while, look for cylinder
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_cylinder(colour=0)

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_cylinder(colour = 0)

            # Step 3. Kick!
            elif self.status_code == 3:
                self.kick()

            # Fall back
            else:
                self.status_code = 0

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)
    
    def fakeJob(self, t0):
        print("faking job")
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find cylinder, lock on to the cylinder and kick cylinder
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.counter += 1
            rospy.sleep(self.TICK)

    def fakeJob2(self, t0):
        print("faking job: the sequel")
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find cylinder, lock on to the cylinder and kick cylinder
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            self.counter += 1
            rospy.sleep(self.TICK)

    def goAfterGreen(self, t0):
        print("MiRo Going after Green cylinder")
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find cylinder, lock on to the cylinder and kick cylinder
        self.status_code = 1
        while rospy.Time.now() < t0 + self.ACTION_DURATION:
            if self.status_code == 1:
                # Every once in a while, look for cylinder
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_cylinder(colour=2)

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_cylinder(colour= 2)

            # Step 3. Kick!
            elif self.status_code == 3:
                self.kick()

            # Fall back
            else:
                self.status_code = 0

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)

    def touchHeadListener(self, data):
        """
        Positive reinforcement comes from stroking the head
        """
        if data.data > 0:
            self.reward += 1

    def touchBodyListener(self, data):
        """
        Negative reinforcement comes from stroking the body
        """
        if data.data > 0:
            self.punishment -= 1

    def lightCallback(self, data):
        """
        Get the frontal illumination
        """
        if data.data:
            self.light_array = data.data
            
    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(35.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break

    def drive(self, speed_l=0.1, speed_r=0.1):  # (m/sec, m/sec)
        """
        Wrapper to simplify driving MiRo by converting wheel speeds to cmd_vel
        """
        # Prepare an empty velocity command message
        msg_cmd_vel = TwistStamped()

        # Desired wheel speed (m/sec)
        wheel_speed = [speed_l, speed_r]

        # Convert wheel speed to command velocity (m/sec, Rad/sec)
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)

        # Update the message with the desired speed
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta

        # Publish message to control/cmd_vel topic
        self.vel_pub.publish(msg_cmd_vel)

    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def detect_cylinder(self, frame, index, colour):
        if frame is None:  # Sanity check
            return

        # Debug window to show the frame
        if self.DEBUG:
            cv2.imshow("camera" + str(index), frame)
            cv2.waitKey(1)

        # Flag this frame as processed, so that it's not reused in case of lag
        self.new_frame[index] = False
        # Get image in HSV (hue, saturation, value) colour format
        im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # RGB values of target cylinder
        rgb_cylinder = [np.uint8([[[255, 0, 0]]]), np.uint8([[[0, 0, 255]]]), np.uint8([[[0, 255, 0]]])]  # e.g. Blue (Note: BGR)
        # Convert RGB values to HSV colour model
        hsv_cylinder = cv2.cvtColor(rgb_cylinder[colour], cv2.COLOR_RGB2HSV)        

        # Extract colour boundaries for masking image
        # Get the hue value from the numpy array containing target colour
        target_hue = hsv_cylinder[0, 0][0]
        if colour == 0:
            hsv_boundries = [np.array([target_hue - 20, 150, 70]), np.array([target_hue + 20, 255, 255])]
        elif colour == 1:
            hsv_boundries = [np.array([target_hue - 0, 70, 70]), np.array([target_hue + 0, 255, 255])]
        else:
            hsv_boundries = [np.array([target_hue - 20, 60, 60]), np.array([target_hue + 20, 255, 255])]

        # Generate the mask based on the desired hue range        
        mask = cv2.inRange(im_hsv, hsv_boundries[0], hsv_boundries[1])
        mask_on_image = cv2.bitwise_and(im_hsv, im_hsv, mask=mask)

        # Debug window to show the mask
        if self.DEBUG:
            cv2.imshow("mask" + str(index), mask_on_image)
            cv2.waitKey(1)

        # Clean up the image
        seg = mask 
        seg = cv2.GaussianBlur(seg, (5, 5), 0)
        seg = cv2.erode(seg, None, iterations=2)
        seg = cv2.dilate(seg, None, iterations=2)        
         
        contours, hierarchy = cv2.findContours(seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                

        if not contours:           
            return

        # Get the largest rectangle
        max_rectangle = None
        self.w = 0
        self.h = 0                
        dst = frame.copy()
        for cnt in contours:
            ## Get the straight bounding rect            
            bbox = cv2.boundingRect(cnt)                 
            x,y,w,h = bbox            

            ## Draw rect
            cv2.rectangle(dst, (x,y), (x+w,y+h), (255,0,0), 1, 16)                           

            ## Get the rotated rect
            rbox = cv2.minAreaRect(cnt)

            if  w*h >= self.w*self.h:
                self.w = w
                self.h = h
                max_rectangle = rbox 
                                 
        # This shouldn't happen, but you never know...
        if max_rectangle is None:                        
            return    

        # Append detected rectangle        
        box = cv2.boxPoints(max_rectangle)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)        
        if self.DEBUG:
            cv2.imshow("rectangles" + str(index), frame)
            cv2.waitKey(1)

        # Normalise values to: x,y = [-0.5, 0.5], r = [0, 1]           
        target_rectangle = np.array(max_rectangle[0]).astype("float32")        
        target_rectangle[0] -= self.x_centre
        target_rectangle[0] /= self.frame_width
        target_rectangle[1] -= self.y_centre
        target_rectangle[1] /= self.frame_width
        target_rectangle[1] *= -1.0       

        # Return a list values [cx, cy, w, h] for the largest rectangle        
        return [target_rectangle[0], target_rectangle[1]]

    def look_for_cylinder(self, colour):
        """
        [1 of 3] Rotate MiRo if it doesn't see a cylinder in its current
        position, until it sees one.
        """
        if self.just_switched:  # Print once
            print("MiRo is looking for the cylinder...")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect cylinder procedure
            self.cylinder[index] = self.detect_cylinder(image, index, colour)

        # If no cylinder has been detected
        if not self.cylinder[0] and not self.cylinder[1]:
            self.drive(self.SLOW, -self.SLOW)
        else:
            self.status_code = 2  # Switch to the second action
            self.just_switched = True

    def lock_onto_cylinder(self, colour, error=25):
        """
        [2 of 3] Once a cylinder has been detected, turn MiRo to face it
        """
        if self.just_switched:  # Print once
            print("MiRo is locking on to the cylinder")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect cylinder procedure
            self.cylinder[index] = self.detect_cylinder(image, index, colour)

        # If only the right camera sees the cylinder, rotate clockwise
        if not self.cylinder[0] and self.cylinder[1]:
            self.drive(self.SLOW, -self.SLOW)
        # Conversely, rotate counterclockwise
        elif self.cylinder[0] and not self.cylinder[1]:
            self.drive(-self.SLOW, self.SLOW)
        # Make the MiRo face the cylinder if it's visible with both cameras
        elif self.cylinder[0] and self.cylinder[1]:
            error = 0.05  # 5% of image width
            # Use the normalised values
            left_x = self.cylinder[0][0]  # should be in range [0.0, 0.5]
            right_x = self.cylinder[1][0]  # should be in range [-0.5, 0.0]
            rotation_speed = 0.03  # Turn even slower now
            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed)  # turn clockwise
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed)  # turn counterclockwise
            else:
                # Successfully turned to face the cylinder
                self.status_code = 3  # Switch to the third action
                self.just_switched = True
                self.bookmark = self.counter
        # Otherwise, the cylinder is lost :-(
        else:
            self.status_code = 0  # Go back to square 1...
            print("MiRo has lost the cylinder...")
            self.just_switched = True

    # GOAAAL
    def kick(self):
        """
        [3 of 3] Once MiRO is in position, this function should drive the MiRo
        forward until it kicks the cylinder!
        """
        if self.just_switched:
            print("MiRo is kicking the cylinder!")
            self.just_switched = False
        if self.counter <= self.bookmark + 2 / self.TICK:
            self.drive(self.FAST, self.FAST)
        else:
            self.status_code = 0  # Back to the default state after the kick
            self.just_switched = True

    def loop(self):
        """
        Main loop
        """
        print("Starting the loop")
        while not rospy.core.is_shutdown():
            if(self.SIGNAL == 1 or self.SIGNAL == 2):
                self.instruction = self.SIGNAL - 1
                self.reward = 0
                self.punishment = 0
                print("Instruction:")
                print(self.instruction)
                # Select next action randomly or via Q score with equal probability
                if np.random.random() >= 0.5:
                    print("Performing random action")
                    self.r = np.random.randint(0, len(self.actions))
                else:
                    print("Performing action with the highest Q score")
                    self.r = np.argmax(self.Q[self.instruction])

                # Run the selected action and update the action counter N accordingly
                start_time = rospy.Time.now()
                self.N[self.instruction][self.r] += 1
                self.actions[self.r](start_time)
                if self.VERBOSE:
                    print("Action finished, updating Q table")

                #start_of_break = rospy.Time.now()
                print("Waiting for Results")
                #while rospy.Time.now() < start_of_break + self.ACTION_DURATION :
                while self.SIGNAL != 3 and self.SIGNAL != 4 :
                    rospy.sleep(self.TICK)

                #reward_strength = self.reward + self.punishment
                if self.SIGNAL == 3:
                    final_reward = 1.0
                    print("This behaviour has been reinforced!")
                elif self.SIGNAL == 4:
                    final_reward = -1.0
                    print("This behaviour has been inhibited!")
                else:
                    final_reward = 0.0

                gamma = min(self.N[self.instruction][self.r], self.discount)
                self.Q[self.instruction][self.r] += self.alpha * (final_reward - self.Q[self.instruction][self.r]) / gamma
                if self.VERBOSE:
                    print("Q scores are: {}".format(self.Q))
                    print("N values are: {}".format(self.N))
                    print("----------------------------------------")
                self.SIGNAL = 0
                start_of_break = rospy.Time.now()
                print("Starting Break")
                while rospy.Time.now() < start_of_break + rospy.Duration(5.0):
                    rospy.sleep(self.TICK)
                print("Waiting for next instruction")
            
            


# This is run when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop
