'''
Created by Jiuyang Bai on 04/09/2021
baijiuyang@hotmail.com
This is an experiment on moving obstacle avoidance, in which
A moving obstacle intercepts participant from different angle
and at different speed. The differences from experitment A are:
(1) There is no ground floor. (2) All poles have a huge
vertical dimension to hide ground plane.
'''
import random
import os.path
import math
import viz
import viztracker
import vizconnect
import time
import vizact

def masterLoop(num):
    if conditions[ii][0] < 1 or not DATA_COLLECT:
        recorded = False
    else:
        recorded = True
    run_trial(int(conditions[ii][0]), conditions[ii][1], conditions[ii][2],
        float(conditions[ii][3]), float(conditions[ii][4]), recorded)
    
def run_trial(i_trial, angle, speed, dsize, ipd, recorded):
    global trial_time, alpha, print_flag, ii, timer_on, obst_v, data_buffer, recording, \
        IPD, eye_height
    time_elapsed = viz.getFrameElapsed()
    trial_time += time_elapsed
    # Current position and roation of the participant
    cur_pos = viz.get(viz.HEAD_POS)
    cur_rot = viz.get(viz.HEAD_ORI)
    # Pops up the Emergency Walls when participant is close to physical room edge.
    popWalls(cur_pos)
    
#    print(viz.MainWindow.getIPD())
    # Record data to buffer
    if recorded and recording:
        obst = models['obstPole'].getPosition()
        goal = models['goalPole'].getPosition()
        obstScale = models['obstPole'].getScale()
        data = [trial_time, cur_pos[0], cur_pos[1], cur_pos[2], cur_rot[0], cur_rot[1], cur_rot[2],
                obst[0], obst[1], obst[2], obstScale[0], obstScale[1], obstScale[2], goal[0], goal[1], goal[2]]
        data_buffer += ','.join([str(round(dat, 4)) for dat in data]) + '\n'
    
    if stage == 'ready':
        # Print start of trial, trial # and condition, initialize trial
        if print_flag:
            print_flag = False
            print('>>> Start Trial ' + str(i_trial) + ': ' + str(angle) + ', ' + str(speed) + ', ' + str(dsize) + ', ' + str(ipd))
            # Update IPD for this the trial
            IPD = ipd
            # Compute obstacle position and velocity for later use
            ang, dist = angle / 360 * 2 * math.pi, (DIAGONAL - 4) / 2
            obst_vi = rotate(DIAGONAL_UNIT[i_trial % 2], ang) # unit vector
            obst_p = [-i * dist for i in obst_vi]                
            obst_v = [vi * speed for vi in obst_vi]
            # Randomize position_y
            obst_p[1] = random.randint(-10, -3)
            models['obstPole'].setPosition(obst_p)
            models['obstPole'].setScale([0.5,1,0.5])
            HOME_POLE[i_trial % 2][1] = random.randint(-10, -3)
            models['homePole'].setPosition(HOME_POLE[i_trial % 2])
            ORI_POLE[i_trial % 2][1] = random.randint(-10, -3)
            models['orientPole'].setPosition(ORI_POLE[i_trial % 2])
            models['homePole'].visible(viz.ON)    
            models['orientPole'].visible(viz.ON)
            if recorded:
                data_buffer += '\n' + 'trial,' + str(i_trial).zfill(3) + ',angle,' + str(angle) + ',speed,' + str(speed) +\
                ',dsize,' + str(dsize) + ',ipd,' + str(ipd) + '\n'
                data_buffer += 'timeStamp,subjX,subjY,subjZ,subjYaw,subjPitch,subjRow,obstX,obstY,obstZ,obstScaleX,obstScaleY,obstScaleZ,goalX,goalY,goalZ\n'
        if alpha < 1.0:
            models['homePole'].alpha(alpha)
            models['orientPole'].alpha(alpha)
            alpha += time_elapsed
        if inRadiusAndFacing():
            if timer(ORIENT_TIME):
                goToStage('go')
        else:
            timer_on = False
    elif stage == 'go':
        # Turn off home and orient poles, turn on goal pole
        if models['homePole'].getVisible():
            # Read eye height
            eye_height = cur_pos[1]
            models['homePole'].visible(viz.OFF)
            models['orientPole'].visible(viz.OFF)
            models['goalPole'].setPosition(ORI_POLE[i_trial % 2])
            models['goalPole'].visible(viz.ON)
            sounds['begin'].play()
            trial_time = 0
            recording = True       
        # Is stage ended?
        if overTheLine(cur_pos, RAMP_DIST, i_trial):
            goToStage('avoid')
        else:
            # Check subject lateral deviation
            if offCourse(cur_pos):
                sounds['startover'].play()                
                reset_trial()
    # Obstacle appears    
    elif stage == 'avoid':
        if speed > 0:            
            if not models['obstPole'].getVisible():                           
                models['obstPole'].visible(viz.ON)
            # Move obstPole
            moveTarget(models['obstPole'], obst_v, time_elapsed)
            # Change the size of obstPole when it's on the opposite side of homePole
#            if models['obstPole'].getPosition()[0] * models['homePole'].getPosition()[0] < 0:
            changeSize(models['obstPole'], eye_height, cur_pos, dsize, time_elapsed)
        # Trial end
        if overTheLine(cur_pos, -END_DIST, i_trial):
            # Write data to disk
            if recorded:
                filename = os.path.abspath(os.path.join(OUTPUT_DIR, 'exp_b_Subj' + subject + '.csv'))
                with open(filename, 'a') as file:
                    file.write(data_buffer)
            reset_trial()
            if i_trial == TOTAL_TRIALS:
                stage == 'NULL'
                sounds['end'].play()
            else:
                ii += 1
                
def reset_trial():
    global stage, print_flag, alpha, timer_on, trial_time, timer_stamp, data_buffer, \
    recording
    # There are 4 stages: ready, go, avoid, NULL(end of exp)
    print_flag = True
    alpha = 0.0
    timer_on = False
    trial_time, timer_stamp = 0, 0
    data_buffer = ''
    recording = False    
    models['goalPole'].visible(viz.OFF)
    models['obstPole'].visible(viz.OFF)
    goToStage('ready')
    
def goToStage(next_stage):
	global stage
	stage = next_stage
	print('Going to stage: ' + stage)

def timer(t):
    global timer_on, trial_time, timer_stamp
    if not timer_on:
        print('timer start')
        timer_stamp = trial_time
        timer_on = True
    if trial_time - timer_stamp > t:
        timer_on = False
        return True
    return False

def rotate(vec, ang):
    '''
    Rotate the vector 'vec' clockwise on x-z plane by angle 'ang'.
    Args:
        vec (3d vector): [x, y, z] in meters.
        ang (float): Angle in radians.
    Return:
        (3d vector): The rotated angle [x', y, z'] in meters.
    '''
    x, y, z = vec
    return [x * math.cos(ang) + z * math.sin(ang), y, -x * math.sin(ang) + z * math.cos(ang)]
    
def overTheLine(cur_pos, dist, i_trial):
    '''
    Detect whether subject is over the line defined as
    y = k * x + b, which is perpendicular to the diagonal.
    Args:
        x, z (float): The x and z position of subject in meter.
        dist (float): The distance between the line and the closer
            corner in meter. Positive means distance to ending corner,
            negative means distance to starting corner.
        i_trial (int): Trial number starting from 1.
    Return:
        (boolean): Inidicating whether the subject is over the line.
    '''
    x, z = cur_pos[0], cur_pos[2]
    # Slope of the line, perpendicular to diagonal path
    k = -DIMENSION_X / DIMENSION_Z
    # Intercept of the line for two home poles respectively
    if dist > 0:
        b = [(DIAGONAL/2 - dist) / math.cos(ROOM_ANGLE), -(DIAGONAL/2 - dist) / math.cos(ROOM_ANGLE)]
    else:
        b = [-(DIAGONAL/2 + dist) / math.cos(ROOM_ANGLE), (DIAGONAL/2 + dist) / math.cos(ROOM_ANGLE)]
    return (z > x * k + b[i_trial % 2]) == (i_trial % 2)

def offCourse(cur_pos):
    # Compute the projection of cur_pos on diagonal path
    l = DIAGONAL_UNIT[1][0] * cur_pos[0] + DIAGONAL_UNIT[1][2] * cur_pos[2]
    p = [xi * l for xi in DIAGONAL_UNIT[1]]
    # Compute the distance between cur_pos and projection p
    d = distance(cur_pos[0], cur_pos[2], p[0], p[2])
    if d > COURSE_MARGIN:
        return True
    return False
    
def moveTarget(target, v, dt):
    # unit of spd is m/s
    p0 = target.getPosition()
    p1 = [p0[0] + v[0] * dt, p0[1], p0[2] + v[2] * dt]
    target.setPosition(p1)

def changeSize(target, center_y, cur_pos, dsize, dt):
    '''
    Change the size of an object on the x (width) and y (height) dimension while keeping
    the z dimension constant and towards the subject. 
    target (viz object): 3d model.
    center_y (float): The height of the center of expansion or contraction.
    cur_pos ([x,y,z]): The current position of hmd.
    dsize (float): The rate of change on size.
    dt (float): Time elapsed in a frame.
    '''
    sx, sy, sz = target.getScale()
    target.setScale([sx + sx * dsize * dt, sy + sy * dsize * dt, sz])
    # Move target vertically so that the height of center of expansion/contraction is constant.
    px, py, pz = target.getPosition()
    py = py - (center_y - py) * dsize * dt
    target.setPosition([px, py, pz])
    # Keep the z axis of the target facing the subject
    cur_pos_ground = [cur_pos[0], py, cur_pos[2]]
    target.lookAt(cur_pos_ground)
    

def relativeOrientation(pos1, pos2):
	xrel = round(pos2[0]-pos1[0],4)
	zrel = round(pos2[2]-pos1[2],4)
	theta = 0
	if zrel == 0.0 and xrel > 0:
		theta = math.pi/2
	elif zrel == 0.0:
		theta = math.pi/2*3
	else:
		theta = math.atan(round(xrel,4)/round(zrel,4))
		if zrel < 0:
			theta += math.pi
		if zrel > 0 and xrel < 0:
			theta += math.pi*2
	return theta
	
def facing(lookingObjectPosn, lookedAtObjectPosn, lookingObjectYaw, thresholdTheta):
	"""lookingObjectPosn: position of object that is looking
	lookedAtObjectPosn: position of object looked at
	lookingObjectYaw: yaw of the object that is looking (degrees)
	thresholdTheta: viewing angle must be +/- this amount in order to be considered 'looking at' the object. degrees

	return: bool, whether the looking object is facing the looked-at object
	>>> universals.facing([0,0,0],[1,0,5],0,20)
	True
	>>> universals.facing([3,0,3],[1,0,0],210,20)
	True
	"""
	degRelOrientation = 180.0/math.pi*relativeOrientation(lookingObjectPosn, lookedAtObjectPosn) #radians
	degRelOrientation = (degRelOrientation+180)%360-180
	return math.fabs(degRelOrientation-lookingObjectYaw)<thresholdTheta
	
def distance(x,y,a,b):
	return ((x-a)**2+(y-b)**2)**.5
	
def inRadius(pos, center, radius):
	#This method takes in two poadsitions in [x,y,z] form and then returns true if the distance between them is less than the radius given
	if pos == '' or center == '' or radius == '':
		return False
	return (distance(pos[0],pos[2],center[0],center[2]) <= radius)
	
def inRadiusAndFacing():
	cur_pos = viz.get(viz.HEAD_POS)
	cur_rot = viz.get(viz.HEAD_ORI)
	return inRadius(cur_pos, models['homePole'].getPosition(), POLE_TRIGGER_RADIUS) and facing(cur_pos, models['orientPole'].getPosition(), cur_rot[0], THRESHOLD_THETA)

##Now that we have the the wall models loaded we need to be able to activate them if a participant gets too close to one of them
def popWalls(pos):
	wallPopped = False
	for w in wall:
		popWall = False
		#For each potential threshold line that the wall could be check to see if the position is beyond that line
		if eval(w['threshLine']):
			popWall = True		
		#If the threshold for any particular wall has been crossed, display the wall and start playing the stop sound
		if popWall:
			w['object'].visible(viz.ON)
			sounds['stop'].play()
			wallPopped = True
		#IF not make sure that the visibility of the wall is not on
		else:
			w['object'].visible(viz.OFF)
	return wallPopped

def overwriteIPD():
    viz.ipd(IPD)
    
if __name__ == "__main__":
    # Settings
    # Set to True when ready to have the experiment write data
    DATA_COLLECT = True
    # If program should run practice trials
    DO_PRACTICE = True

    # If program crashes, start trials here
    START_ON_TRIAL = None
    if not START_ON_TRIAL:
        START_ON_TRIAL = -3 if DO_PRACTICE else 1
    TOTAL_TRIALS = 154 # 4 (angle) * 2 (speed) * 3 (expansion) * 2 (disparity) * 3 (reps) + 10 (freewalk) 

    # Set output directory for writing data
    INPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Input'))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Raw Data'))
    SOUND_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Stimuli', 'Sounds'))
    MODEL_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'Stimuli', 'Models'))


    # Orientation constants
    POLE_TRIGGER_RADIUS = 0.3 # How close participant must be to home pole
    THRESHOLD_THETA = 10 # Maximum angle participant can deviate when looking at orienting pole
    ORIENT_TIME = 3 # How long participant must orient onto pole
    COURSE_MARGIN = 0.2 # off-course threshold
    # The dimension of the room space used for experiment
    DIMENSION_X, DIMENSION_Z = 9.0, 11.0
    DIAGONAL = (DIMENSION_X**2 + DIMENSION_Z**2)**(1.0/2)# The length of the diagonal line of the experimental space
    ROOM_ANGLE = math.atan(DIMENSION_X/DIMENSION_Z) # the anger (in radian) between the diagonal and the shorter edge of the room
    DIAGONAL_UNIT = [[-math.sin(ROOM_ANGLE), 0, -math.cos(ROOM_ANGLE)], [math.sin(ROOM_ANGLE), 0, math.cos(ROOM_ANGLE)]]
    RAMP_DIST = 2 # acceleration distance before obstacle appears
    END_DIST = 2 # distance to the next home pole, within which the trial end.
    
    # Home and Orient Pole positions (x,z,y)
    HOME_POLE = [[DIMENSION_X/2, 0, DIMENSION_Z/2], [-DIMENSION_X/2, 0, -DIMENSION_Z/2]]
    ORI_POLE = [[-DIMENSION_X, 0, -DIMENSION_Z], [DIMENSION_X, 0, DIMENSION_Z]]
    
    # Emergency wall
    wall = []
    wall.append({'position': [6.5,0,0], 'rotation': [90,0,0], 'threshLine': 'pos[0] > 5.5'})
    wall.append({'position': [0,0,7.5], 'rotation': [0,0,0], 'threshLine': 'pos[2] > 6.5'})
    wall.append({'position': [-6.5,0,0], 'rotation': [90,0,0], 'threshLine': 'pos[0] < -5.5'})
    wall.append({'position': [0,0,-7.5], 'rotation': [0,0,0], 'threshLine': 'pos[2] < -6.5'})
    for w in wall:
        #For each entry in wall create a viz object that can be manipulated in the visible environment
        w['object'] = viz.add(os.path.abspath(os.path.join(MODEL_DIR, 'wall.3DS'))) #load the appropriate 3d model for the wall
        w['object'].setEuler(w['rotation'])
        w['object'].setPosition(w['position'])
        w['object'].visible(viz.OFF) # Turn the walls off initially

    # Load stimuli. Original pole size [0.4, 3, 0.4]
    viz.clearcolor(0,0.4,1.0) # blue world
    models = {}
    models['homePole'] = viz.add(os.path.abspath(os.path.join(MODEL_DIR, 'pole_blue_96-0.osgb')))
    models['orientPole'] = viz.add(os.path.abspath(os.path.join(MODEL_DIR, 'pole_yellow_96-0.osgb')))
    models['goalPole'] = viz.add(os.path.abspath(os.path.join(MODEL_DIR, 'pole_green_96-0.osgb')))
    models['obstPole'] = viz.add(os.path.abspath(os.path.join(MODEL_DIR, 'pole_red_96-0.osgb')))  
    # models['ground'] = viz.add(os.path.abspath(os.path.join(MODEL_DIR, 'ground.osgb')))  
    
    # Adjust models size turn visible off
    models['homePole'].setScale([0.6,1,0.6])
    models['orientPole'].setScale([1,1,1])
    models['goalPole'].setScale([1,1,1])
    models['obstPole'].setScale([0.5,1,0.5])
    models['homePole'].visible(viz.OFF)
    models['orientPole'].visible(viz.OFF)
    models['goalPole'].visible(viz.OFF)
    models['obstPole'].visible(viz.OFF)
    # Turn on the lights
    light1 = viz.addLight() #Add an overhead light
    light1.setEuler(0,90,0)
    light2 = viz.addLight() #Next four are lights from each direction to ensure even lighting
    light2.setEuler(90,0,0)
    light3 = viz.addLight()
    light3.setEuler(0,0,0)
    light4 = viz.addLight()
    light4.setEuler(180,0,0)
    light5 = viz.addLight()
    light5.setEuler(270,0,0)
    # Sounds
    sounds = {}
    sounds['end'] = viz.addAudio(os.path.abspath(os.path.join(SOUND_DIR, 'End.mp3')))
    sounds['begin'] = viz.addAudio(os.path.abspath(os.path.join(SOUND_DIR, 'Begin.mp3')))
    sounds['startover'] = viz.addAudio(os.path.abspath(os.path.join(SOUND_DIR, 'Startover.mp3')))
    sounds['stop'] = viz.addAudio(os.path.abspath(os.path.join(SOUND_DIR, 'Stop.wav')))
    # set up IO
    # Dialog box asking for type of control and subject number
    HMD = 'Odyssey'
    MONITOR = 'PC Monitor'
    controlOptions = [HMD, MONITOR]
    controlType = controlOptions[viz.choose('How would you like to explore? ', controlOptions)]

    subject = viz.input('Please enter the subject number:','')
    subject = str(subject).zfill(2)

    # Use keyboard controls
    # Controls:
    # q - Strafe L		w - Forward		e - Strafe R
    # a - Turn L		s - Back		d - Turn R
    # y - Face Up		r - Fly Up
    # h - Face Down		f - Fly Down
    if controlType == MONITOR:
        HZ = 60
        headTrack = viztracker.Keyboard6DOF()
        link = viz.link(headTrack, viz.MainView)
        headTrack.eyeheight(1.6)
        link.setEnabled(True)
        viz.go()
    elif controlType == HMD:
        HZ = 90
        vizconnect.go('vizconnect_config.py')
        # Overwrite headset ipd
        IPD = viz.MainWindow.getIPD()
        vizact.onupdate(viz.UPDATE_LINKS + 1, overwriteIPD)	
        # add Odyssey tracker
        ODTracker = vizconnect.getTracker('head_tracker')
    
    # Use a large size of the viewing frustum
    viz.clip(.001,1000)
    
    # loads experimental conditions
    inputFile = os.path.abspath(os.path.join(INPUT_DIR, 'exp_b_subj' + subject + '.csv'))
    
    with open(inputFile, 'r') as file:
        lines = file.read().split('\n')[1:-1]
        conditions = [[float(x) for x in line.split(',')] for line in lines]
        
    # Define the trial, with which the experiment starts
    ii = START_ON_TRIAL + 3 if START_ON_TRIAL else 0
    
    # Initailize trial variables
    reset_trial()
    
    # Restarts the loop, at a rate of Hz
    viz.callback(viz.TIMER_EVENT,masterLoop)
    viz.starttimer(0,1.0/HZ,viz.FOREVER)
    