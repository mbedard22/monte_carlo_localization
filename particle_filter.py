## COMP.4500- Mobile Robotics, SPR23
## Lab #5- Particle Filters (Part 1, updated)
## Danielle Le & Matthew Bedard 
from grid import *
from particle import Particle
from utils import *
from setting import *
import math
import numpy as np

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments: 
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurement, a pair of robot pose, i.e. last time
                step pose and current time step pose

        Returns: the list of particle represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    alpha1 = 0.001
    alpha2 = 0.001
    alpha3 = 0.005
    alpha4 = 0.005
    #odom_noise =  add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA)

    result_particles = []

    rot1 = math.degrees(math.atan2(odom[1][1] - odom[0][1], odom[1][0] - odom[0][0])) - odom[0][2]
    Trans = np.sqrt((odom[0][0]- odom[1][0])**2 + (odom[0][1] - odom[1][1])**2)
    rot2 = odom[1][2] - odom[0][2] - rot1
    
    for i in particles:
        p1 = rot1 - add_gaussian_noise(alpha1* rot1 + alpha2 * Trans, ODOM_HEAD_SIGMA)
        p2 = Trans - add_gaussian_noise(alpha3 * Trans + alpha4 * (rot1 + rot2), ODOM_TRANS_SIGMA)
        p3 = rot2 - add_gaussian_noise(alpha1 * rot2 + alpha2 * Trans, ODOM_HEAD_SIGMA)

        x, y, h = i.xyh
        h = h + p1
        dx = math.cos(math.radians(h)) * p2
        dy = math.sin(math.radians(h)) * p2
        x += dx
        y += dy
        h = h + p3
        result_particles.append(Particle(x, y, h))
    return result_particles
# ------------------------------------------------------------------------
def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- a list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before measurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map containing the marker information. 
                see grid.py and CozGrid for definition

        Returns: the list of particle representing belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    weight = []

    for i in particles:
        match = []
        x,y, heading = i.xyh
        markersSeen = i.read_markers(grid)

        for marker in measured_marker_list:
            markerX, markerY, markerHeading = add_marker_measurement_noise(marker, MARKER_TRANS_SIGMA, MARKER_ROT_SIGMA)
            if len(markersSeen) == 0:
                break 
            closestMarker = None
            minDistance = 100006900069 # replaced distance

            for sm in markersSeen:
                dist = grid_distance(markerX, markerY, sm[0], sm[1])
                if dist < minDistance:
                        minDistance = dist
                        closestMarker = sm 
                        match.append([marker, closestMarker])

        if not grid.is_in(x,y):
                weight.append(0)
        elif (len(markersSeen) == 0) and (len(measured_marker_list) == 0):
                weight.append(1)
        elif len(markersSeen) != len(measured_marker_list):
                weight.append(0)
        elif len(match) != 0:
                probab = 1

                # Loops through matches     
                for k,l in match:
                        matchDistance = grid_distance(k[0], k[1], l[0], l[1])
                        matchAngle = diff_heading_deg(k[2], l[2])
                        matchDistance = ((matchDistance** 2 ) / ((2 * MARKER_TRANS_SIGMA)**2))
                        matchAngle = ((matchAngle** 2 ) / ((2 * MARKER_ROT_SIGMA)**2))
                        probab *= np.exp(-matchDistance - matchAngle)
                weight.append(probab) 

    # Normalize weights
    totalWeight = np.sum(weight) 
    normWeight = []
    if totalWeight != 0:
        for i in weight: 
                new = i / totalWeight 
                normWeight.append(new)

        if len(weight) != 0:
                measured_particles = np.random.choice(particles, PARTICLE_COUNT-5, replace = True, p = normWeight)    


    else:
        return Particle.create_random(PARTICLE_COUNT, grid)
    return measured_particles