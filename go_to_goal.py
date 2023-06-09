## COMP.4500- Mobile Robotics, SPR23
## Lab #5- Particle Filters (Part 2)
## Danielle Le & Matthew Bedard 

import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
from cozmo.util import degrees

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"

async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
#        cv2.imshow("Markers", opencv_image)

    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = []
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -math.atan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        # print('x =', x, 'y =', y,'theta =', yaw)
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list
    
#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch = True):
    global last_pose
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    if cvt_inch:
        last_x, last_y = last_x / 25.6, last_y / 25.6
        curr_x, curr_y = curr_x / 25.6, curr_y / 25.6

    return [[last_x, last_y, last_h],[curr_x, curr_y, curr_h]]

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# Code that is written for motion
async def run(robot: cozmo.robot.Robot):
    global last_pose
    global grid, gui

    curMarkers = []
    goalPos = False
    reset = False

    # Start streaming
    robot.camera.image_stream_enabled = True

    # Start particle filter
    pf = ParticleFilter(grid) 

    ######################### YOUR CODE HERE####################################

    await robot.set_head_angle(degrees(5)).wait_for_completed()
    await robot.set_lift_height(1.0).wait_for_completed()

    while True:
        
        if robot.is_picked_up == True:
            print("UNHAPPY!")
            pf = ParticleFilter(grid)
            await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()
            reset = True
            goalPos = False

        # Not a Goal
        if goalPos == False:
            # Throws arms up and down
            if reset == True:
                await robot.set_head_angle(degrees(5)).wait_for_completed()
                await robot.set_lift_height(1.0).wait_for_completed()

            pose = robot.pose
            curMarkers = await image_processing(robot)
            print(curMarkers)
            markerMeasure = cvt_2Dmarker_measurements(curMarkers)
            odom = compute_odometry(pose)
            newParticle = pf.update(odom, markerMeasure)
            gui.show_particles(pf.particles)
            gui.show_mean(newParticle[0], newParticle[1], newParticle[2], newParticle[3])
            gui.updated.set()

            # Check convergence
            if newParticle[3] == False:
                print("Not confident")
                last_pose = robot.pose
                await robot.turn_in_place(cozmo.util.degrees(-30)).wait_for_completed()
                await robot.drive_straight(cozmo.util.distance_mm(25),
                                           speed=cozmo.util.speed_mmps(40), should_play_anim = False).wait_for_completed()
            else:
                print("Converged")
                if robot.is_picked_up == True:
                    print("UNHAPPY!")
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()
                    reset = True
                    goalPos = False
                goal_pose = True
                robot.stop_all_motors()
                meanPose = compute_mean_pose(pf.particles)
                print(meanPose) 
                distance = math.sqrt((goal[0] - meanPose[0])**2 + (goal[1] - meanPose[1])**2)
                turn_angle = math.degrees(np.arctan2(goal[1] - meanPose[1], goal[0] - meanPose[0]))
                angle_dif = -(meanPose[2] - turn_angle)
                print(angle_dif)
            
                await robot.turn_in_place(cozmo.util.degrees(angle_dif)).wait_for_completed()
                goalPos = True
                move_dis = abs(distance)
                await robot.drive_straight(cozmo.util.distance_inches(move_dis), speed=cozmo.util.speed_mmps(20)).wait_for_completed()
                await robot.turn_in_place(cozmo.util.degrees(-turn_angle)).wait_for_completed()

                if robot.is_picked_up == True:
                    print("UNHAPPY!")
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()
                    reset = True
                    goalPos = False

        else:
            print("At Goal")
            for i in range(0,2):
                await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabReactHappy).wait_for_completed()
                if robot.is_picked_up == True:
                    print("UNHAPPY!")
                    pf = ParticleFilter(grid)
                    await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()
                    reset = True
                    goalPos = False
            if robot.is_picked_up == True:
                print("UNHAPPY!")
                pf = ParticleFilter(grid)
                await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()
                reset = True
                goalPos = False
            pf = ParticleFilter(grid)
            goalPos = False

    ############################################################################


class CozmoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()
