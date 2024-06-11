##### Display Module (Server) Using TCP/IP Communication with ELID ANPR #######
#
# Current Author: Darshan Suresh
# Previous Author: Iqra
# Description:
# This program uses OpenCV to display information received from ELID ANPR
# No particular input is required but the port number, IP address, and commands must be known to ELID ANPR

# import libraries needed and other needed info and variables
from configparser import ConfigParser
from pathlib import Path
from datetime import datetime as date_t, date
from datetime import timedelta
from threading import Thread, Lock
import socket
import select
import cv2
import numpy as np
import os
import sys
import tkinter as tk
import subprocess
import time
import ipaddress
import glob
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from time import perf_counter

# bgr code of all colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (155, 0, 0)
SKYBLUE = (235, 206, 135)
RED = (0, 0, 155)
GREEN = (0, 155, 0)
TEAL = (128, 128, 0)
SCARLET = (0, 36, 255)
YELLOW = (0, 204, 255)
NEON = (20, 255, 57)
ORANGE = (0, 127, 255)
SILVER = (206, 202, 196)
ROSE = (127, 0, 255)
SAPPHIRE = (186, 82, 15)
EMERALD = (120, 200, 80)
GRAY = (169, 169, 169)
CRIMSON = (10, 15, 184)
GOLD = (0, 215, 255)
MAGENTA = (255, 0, 255)
PURPLE = (172, 13, 106)
BRIGHT_RED = (43, 75, 238)

# network variables
#host_ip = str(subprocess.check_output(['hostname', '-I'])).split(' ')[0].replace("b'", "")
#host_ip = "128.100.6.0"
#host_ip = str(subprocess.run(['hostname', '-I']))
def get_ip():
    while True:
        ip_output = subprocess.check_output(["hostname", "-I"]).decode("utf-8").strip()
        if ip_output.count('.') < 3:
            print("IP address has less than three dots. Rerunning command...")
        else:
            return ip_output
# Attempt to get an IP address with at least three dots
host_ip = get_ip()
port = 50005
root = tk.Tk()
buffersize = 4096

# screen width, height, and aspect ratio
width_px, height_px = root.winfo_screenwidth(), root.winfo_screenheight()  # this stores the display 0 width and height of the screen (e.g 1280x 720)
aspect_rat, screen_res = width_px / height_px, (width_px, height_px)  # aspect ratio : Width/Height

# stream width and height (fixed to HD)
cam_width, cam_height = 1280, 720  # HD is 720p

# find difference between monitor resolution and stream resolution
height_diff = int((height_px - 720) / 2)
width_diff = int((width_px - 1280) / 2)

# text width limit and y-position for mono mode
mid_mono_lim, mid_mono_y_pos = cam_height / height_px, 0.5

# fonts for text and number
font = cv2.FONT_HERSHEY_DUPLEX
font_comp = cv2.FONT_HERSHEY_TRIPLEX
font_size_percentage = 80 / 100

# pre-defined texts and images
established_text = "Network connection established"
imshow_title = "ANPR Display Monitor"
restart_command = "sudo systemctl restart anpr_network_monitor.service"
car_unrecognized = "VEHICLE UNRECOGNIZED"
car_blacklisted = "VEHICLE BLACKLISTED"
car_wrong_color = "WRONG VEHICLE COLOR"
car_wrong_make = "WRONG VEHICLE MAKE"
access_denied = "ACCESS DENIED"
under = "UNDER"
maintenance = "MAINTENANCE"
connecting = "Connecting to Camera..."
established = "Connection established"

# lists of commands - used in main thread
command_list = ['USER', 'STREAM', 'IPADDR', 'ENTRANCE', 'THEME', 'RECOG', 'TEST', 'NAME', 'MODE', 'MAINT', 'CAM',  'RESET']
stream_status_list = ['ON', 'OFF']
logo_mode_list = ['1', '2', '3']
recog_list = ['FAIL', 'BLACKLIST', 'COLOR', 'MAKE']
theme_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
entrance_check_list = ['YES', 'NO']

# to check the ip address of the camera, anpr and dms
ip_check = '//home//elid//anpr_display//network_monitor//network_monitor.sh'
FORMAT = 'utf-8'

# to check if noipcam txt file is present to kill the service after certain num of tries
noipcam = "/home/elid/noipcam.txt"

# =====VARIABLES defined in global to be used in main thread and display thread=====================#
camCanAccess = False  # set to false to let main thread to access configfile. Else let the display thread access the configfile. Ensure no deadlock
cmdSet = ""  # controls the readConfig function and the display/execution from the display thread
camStreamEnable = "OFF"  # checked from the config file ; default is off
userName = ""  # username
userNumPlate = ""  # car number plate
entrance_temp = ""  # to control the salutation message on the default screen.
camStreamStatus = False  # False==off, True==on ; we switch this to off if currently camera set to be off in config file and change to true otherwise.
recogMsg = ""  # to control which recognition error to be printed on the display screen
testContent = "none"  # to print the test message on the  display screen
frame = ""  # to display camera stream frame on the display screen - we grab all camera stream but only retrieved frame are used to display(better performance)

# variables to check connection
CamConnectStatus = -99 # set initial status value (can set any number, I just chose -99)
AttemptingConnection = 0 # connecting...
ShowingSuccessScreen = 1 # connection established
ConnectionAlreadyEstablished = 2 # after connection established

# defaultRead is used to allow the display server to read the currently configured config file once and display on the screen if it is restarted
defaultRead = True # set to true to read the config file once
if defaultRead == True:
    file = '//home//elid//anpr_display//config//config.ini'
    config = ConfigParser()
    config.read(file)
    FORMAT = 'utf-8'
    theme_chosen = config['Background']['num']
    entrance_check = config['Background']['entrance']
    compName = config['Logo']['compname']
    mode = config['Logo']['mode']
    logoname = config['Logo']['compname']
    maint_mode = config['Maintenance']['mode']
    maint_date_needed = config['Maintenance']['dateNeeded']
    startdate = config['Maintenance']['startdate']
    enddate = config['Maintenance']['enddate']
    startmonth = config['Maintenance']['startmonth']
    endmonth = config['Maintenance']['endmonth']
    startyear = config['Maintenance']['startyear']
    endyear = config['Maintenance']['endyear']
    ip_cam = config['Cam']['ip_cam']
    cam_username = config['Cam']['cam_username']
    cam_password = config['Cam']['cam_password']
    defaultRead = False # set False to stop reading

# create a display_screen image that will fill the screen with theme chosen in the config file when program is restarted
display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),
                            (width_px, height_px))  # let the display_screen to be any of theme chosen by user
convmonth =1

# ================ create or continue logging activities ================#
def logging(text):
    """
    input:
        text: a string of characters
    output:
        an updated logfile

    logging keep tracks of all the things that the display module executed and the connections.
    """
    path = "//home//elid//anpr_display//logfiles//"  # path to logfile
    filename = date_t.now().strftime('%Y-%m-%d.log')  # Name of logfile
    times = date_t.now().strftime('[%d-%m-%Y %H:%M:%S] ')  # Current time

    # write on a new logfile if the logfile for today does not exist
    if Path(path + filename).is_file():
        newlog = open(path + filename, "a")
        newlog.write(f'{times}{text}\n')
        newlog.close()
    else:
        newlog = open(path + filename, "w")
        newlog.write(f'{times}{text}\n')
        newlog.close()


# ================ encoding a string ================#
# simple decrypt function decrypts messages sent by user.
def simple_decrypt(ciphertext):
    """
    input:
        ciphertext: a list of numbers
    outout:
        return a string of decrypted characters
    """

    # data dictionary of common text and CLI chars
    encodeddict = {
        'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8,
        'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16,
        'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24,
        'y': 25, 'z': 26, ' ': 99, 'A': 100, 'B': 101, 'C': 102, 'D': 103, 'E': 104,
        'F': 105, 'G': 106, 'H': 107, 'I': 108, 'J': 109, 'K': 110, 'L': 111, 'M': 112,
        'N': 113, 'O': 114, 'P': 115, 'Q': 116, 'R': 117, 'S': 118, 'T': 119, 'U': 120,
        'V': 121, 'W': 122, 'X': 123, 'Y': 124, 'Z': 125, '.': 200, '/': 201, '\\': 202,
        '$': 203, '#': 204, '@': 205, '%': 206, '^': 207, '*': 208, '(': 209, ')': 210,
        '_': 211, '-': 212, '=': 213, '+': 214, '>': 215, '<': 216, '?': 217, ';': 218,
        ':': 219, '\'': 220, '\"': 221, '{': 222, '}': 223, '[': 224, ']': 225, '|': 226,
        '`': 227, '~': 228, '!': 229, '&': 230, '0': 300, '1': 301, '2': 302, '3': 303,
        '4': 304, '5': 306, '6': 307, '7': 308, '8': 309, '9': 310}

    # Using the eval built in to interpret the files line as a list instead string
    # Also utilizing the whitelist to only allow the list builtin class
    try:
        encodedbuffer = eval(ciphertext, {"__builtins__": {'list': list}})
    except SyntaxError as e:
        logging(f'Error: {e}')
        sys.exit()

    # Need to read the 'cleartext' IV in the first element of the list
    # The IV will combine with the user specified key to provide appropriate stream decrypt
    readiv = encodedbuffer[0]
    key = encodedbuffer[-1]

    # Use decryption algo which is inverse: (5x - 100 + key)^-1
    # Decryption algo: (x + 100 - k )/5
    decryptedsignal = []
    compositekey = int(readiv) + int(key)
    # Return the decrypted codes to the original ASCII equivalent
    for i in encodedbuffer:
        decryptedsignal.append(int((i + 100 - int(compositekey)) / 5))

    decryptedtext = []

    for i in decryptedsignal:
        # remember encodeddict is a dictionary using key value pairs
        # must access via .items method for value to key
        for k, v in encodeddict.items():
            if v == i:
                decryptedtext.append(k)

        # converting the list decryptedtext into original string form
        decryptedtextstring = ''
        for i in decryptedtext:
            decryptedtextstring = decryptedtextstring + str(i)

    return str(decryptedtextstring)


# ================ Check validity of IP Address ================#
def get_ip_type(addr1="", addr2="", addr3=""):
    """
    input:
        ip addresses
    output:
        True if both are valid addresses
        False if any is invalid
    """

    try:
        ip1 = ipaddress.ip_address(addr1)
        ip2 = ipaddress.ip_address(addr2)
        ip3 = ipaddress.ip_address(addr3)
        # checking the validity of both addresses using ipaddress library
        if isinstance(ip1, ipaddress.IPv4Address) and isinstance(ip2, ipaddress.IPv4Address) and isinstance(ip3, ipaddress.IPv4Address):
            return True
        else:
            return False
    except ValueError:
        return False


# ================ update ip address ================#

def replace_ip(elidanpr_ip="", camera_ip="", dms_ip=""):
    """
    input:
        new ip addresses for elid-anpr, ip camera and dms
    output:
        updated script with the new ip addresses
    """

    # read every lines
    lines = open(ip_check, 'r').readlines()

    # update the host ip address
    lines[2] = f"ELIDANPR_IP=\"{elidanpr_ip}\"\n"
    lines[3] = f"CAMERA_IP=\"{camera_ip}\"\n"
    lines[4] = f"DMS_IP=\"{dms_ip}\"\n"
    out = open(ip_check, 'w')
    out.writelines(lines)
    out.close

# ================================CAMERA STREAM THREAD===========================================#
'''
This thread controls the camera stream frame grabbing and retrieving operations.
Camera frames are grabbed all the time while it is only retrieved(decoded) per interval set.

'''
    
def delete_spawn_files():
    files = os.listdir("/home/elid/anpr_display/spawn/")
    for file in files:
        os.remove(os.path.join("/home/elid/anpr_display/spawn/", file))
        #print("removed " + str(file))

def CameraStreamThread(myLock):
    global frame
    global ip_cam
    global CamConnectStatus
    global AttemptingConnection
    global ShowingSuccessScreen
    global ConnectionAlreadyEstablished
    spawn_folder = '/home/elid/anpr_display/spawn'
    max_error_files = 49
    frCounter = 0
    intv = 5
    time.sleep(3)
    stream_link = "rtsp://{}:{}@{}:554/cam/realmonitor?channel=1&subtype=0".format(cam_username, cam_password, ip_cam)
    frCapture = cv2.VideoCapture(stream_link)
    if frCapture.isOpened():
        CamConnectStatus = ShowingSuccessScreen
        delete_spawn_files()
        while True:
            frCounter += 1
            frCapture.grab()

            if frCounter % intv == 0:
                ret, streamFrame = frCapture.retrieve()
                with myLock:
                    frame = streamFrame.copy()

            if frCounter > 1000:
                frCounter = 0
    else:
        error_files = glob.glob(os.path.join(spawn_folder, 'camera_error*.txt'))
        if len(error_files) >= max_error_files:
            delete_spawn_files()
            subprocess.run(['sudo', 'systemctl', 'mask', 'anpr_server_and_display.service'])
            subprocess.run(['sudo', 'systemctl', 'disable', 'anpr_server_and_display.service'])
            subprocess.run(['sudo', 'systemctl', 'stop', 'anpr_server_and_display.service'])
        else:
            error_filename = os.path.join(spawn_folder, f'camera_error_{len(error_files) + 1}.txt')
            with open(error_filename, 'w') as file:
                file.write('Failed to retrieve from IP camera.')
            os._exit(0)

# =================================DISPLAY THREAD===============================================#
'''
Thread to handle all display-related tasks that is shown on display screen.
This includes:
(a) Display error messages
(b) Display user information
(c) Show default screen
(d) Display maintenance mode
(e) Display testing screen

'''
def DisplayThread(myLock):
    global camCanAccess
    global cmdSet
    global camStreamEnable
    displayState = 0
    camStreamCount = 0
    textDisplayCount = 0
    userDisplayCount = 0
    recogDisplayCount = 0
    fps = 25
    camStreamDuration = 8
    textDisplayDuration = 3
    opFrCount = fps / 6
    camStreamTotal = opFrCount * camStreamDuration
    textDisplayTotal = opFrCount * textDisplayDuration
    userDisplayTotal = 30
    recogDisplayTotal = 30
    camStreamStatus = False  # False==off, True==on ; we switch this to off if currently camera set to be off in config file and change to true otherwise.
    global userName
    global userNumPlate
    global entrance_check
    global theme_chosen
    global mode
    global recogMsg
    global testContent
    global logoname
    global display_screen
    global entrance_temp
    global maint_mode
    global maint_date_needed
    global startdate
    global enddate
    global startmonth
    global endmonth
    global startyear
    global endyear
    global ip_cam
    global cam_username
    global cam_password
    global CamConnectStatus
    global AttemptingConnection
    global ShowingSuccessScreen
    global ConnectionAlreadyEstablished

    # Function to read from config file. Only values needed for display are read from the config file
    # The configParam below corresponds to the command sent by the user through recv[0]
    def readConfig(configParam):

        config = ConfigParser()
        config.read(file)
        global camStreamEnable
        global userName
        global userNumPlate
        global entrance_check
        global theme_chosen
        global mode
        global recogMsg
        global testContent
        global logoname
        global display_screen
        global entrance_temp
        global maint_mode
        global maint_date_needed
        global startdate
        global enddate
        global startmonth
        global endmonth
        global startyear
        global endyear
        global ip_cam
        global cam_username
        global cam_password

        # If command received is USER - to display user and user's car information
        if configParam == 'USER':
            userName = config['User']['name']
            userNumPlate = config['User']['numplate']
            theme_chosen = config['Background']['num']
            mode = config['Logo']['mode']
            entrance_check = config['Background']['entrance']
            if userName == "INFO UNAVAILABLE":
                logging(
                    "Failed to write username and user numplate info")  # write into logfile in case writing into configfile failed
            config.set('User', 'name', 'INFO UNAVAILABLE')
            config.set('User', 'numplate', 'INFO UNAVAILABLE')
            with open(file, 'w') as configfile:
                config.write(configfile)
            return True

        # If command received is STREAM - to turn on or off camera stream
        elif configParam == 'STREAM':
            # global camStreamEnable
            camStreamEnable = config['Stream']['status']
            camStreamEnable = str(camStreamEnable)
            ip_cam = config['Cam']['ip_cam']
            cam_username = config['Cam']['cam_username']
            cam_password = config['Cam']['cam_password']
            
            # print(camStreamEnable)
            return False

        # If command received is ENTRANCE - to set WELCOME/SEE YOU AGAIN greeting
        elif configParam == 'ENTRANCE':
            entrance_temp = config['Background']['entrance']
            # print(entrance_temp)
            theme_chosen = config['Background']['num']
            mode = config['Logo']['mode']
            return False

        # If command received is THEME - to set the theme for default screen
        elif configParam == 'THEME':
            theme_chosen = config['Background']['num']
            mode = config['Logo']['mode']
            entrance_check = config['Background']['entrance']
            return False

        # If command received is NAME - to set the company name
        elif configParam == 'NAME':
            logoname = config['Logo']['compname']
            return False

        # If command received is MODE - to set the logo mode option
        elif configParam == 'MODE':
            theme_chosen = config['Background']['num']
            mode = config['Logo']['mode']
            entrance_check = config['Background']['entrance']
            return False

        # If command received is RECOG - to display recognition error message
        elif configParam == 'RECOG':
            recogMsg = config['Recog']['type']
            if recogMsg != "FAIL":
                userName = config['User']['name']
                userNumPlate = config['User']['numplate']
            theme_chosen = config['Background']['num']
            mode = config['Logo']['mode']
            entrance_check = config['Background']['entrance']
            if userName == "INFO UNAVAILABLE":
                logging(
                    "Failed to write username and user numplate info")  # write into logfile in case writing into configfile failed
            config.set('User', 'name', 'INFO UNAVAILABLE')
            config.set('User', 'numplate', 'INFO UNAVAILABLE')
            with open(file, 'w') as configfile:
                config.write(configfile)
            return True

        # If command received is MAINT - to activate the maintenance mode message on the default screen
        elif configParam == 'MAINT':
            maint_mode = config['Maintenance']['mode']
            maint_date_needed = config['Maintenance']['dateNeeded']
            startdate = config['Maintenance']['startdate']
            enddate = config['Maintenance']['enddate']
            startmonth = config['Maintenance']['startmonth']
            endmonth = config['Maintenance']['endmonth']
            startyear = config['Maintenance']['startyear']
            endyear = config['Maintenance']['endyear']
            
        # If command received is CAM - to modify IP cam settings
        elif configParam == 'CAM':
            ip_cam = config['Cam']['ip_cam']
            cam_username = config['Cam']['cam_username']
            cam_password = config['Cam']['cam_password']
            
            # print(camStreamEnable)
            return True

        # If command received is RESET - reset all changes done to display to the factory setting
        elif configParam == 'RESET':
            camStreamEnable = config['Stream']['status']
            theme_chosen = config['Background']['num']
            entrance_check = config['Background']['entrance']
            logoname = config['Logo']['compname']
            mode = config['Logo']['mode']
            return True

        # If no command received, below information are read and used to show default screen
        elif configParam == "":
            camStreamEnable = config['Stream']['status']
            theme_chosen = config['Background']['num']
            entrance_check = config['Background']['entrance']
            logoname = config['Logo']['compname']
            mode = config['Logo']['mode']
            return False

    # ================ create one line of text on display ================#
    def create_text(text="", scale=0, x_lim=0.85, y_lim=0.15, thick_mul=0, x_pos=0, y_pos=0,
                    text_clr=BLACK, image=True, fnt=font):
        """
        input:
            text: string of characters
            scale: scale or size of text
            x_lim: width ratio of text over background width
            y_lim: height ratio of text over background height
            thick_mul: thickness of the text
            x_pos and y_pos: position of the text as a ratio of the image
            image: resolution of the image
            text_clr: text colour in BGR (blue, green, red) format
        output:
            displays input text on an image
        """

        screen_x, screen_y = image.shape[1], image.shape[0]
        font_scale = min(screen_x, screen_y) / (25 / scale)
        font_thick = int(font_scale * thick_mul)
        textsize = cv2.getTextSize(text, fnt, font_scale, font_thick)[0]
        textX = int(image.shape[1] * x_pos - textsize[0] / 2)  # x position
        textY = int(image.shape[0] * y_pos + textsize[1] / 2)  # y position
        return cv2.putText(image, text, (textX, textY), font, font_scale, text_clr,
                           font_thick, cv2.LINE_AA)

    # ================ Display function ================#
    def display(text1="", text2="", text3="", text4="", line1="false", line2="false",
                line3="false", line4="false", chosen_theme="", set_time=1, show=False, header=False, entrance=True,
                warning=False, mode_chosen="", logoName="ELID"):

        # image = background_image(theme_chosen)
        image = display_screen.copy() # create a fresh copy of the display_screen image initialized at startup
        font_size_percentage = 80 / 100 # font size scale percentage
        font_color_name = WHITE # font color set to default white

        # current date and time
        today = date.today().strftime("%c")
        # current_time = date_t.now().strftime("%-I:%M %p,")

        # ====DISPLAYING TEXT ON THE DISPLAY SCREEN=======#
        # 1A. writing the message in one line only for blinking text - done
        if line1 == "true" and line2 == "false" and warning == False:
            create_text(text=text1, scale=0.11, x_lim=0.5, y_lim=4, thick_mul=2.5, x_pos=0.5, y_pos=0.5, image=image,
                        text_clr=font_color_name)

        # 1B. writing the message in two line only for blinking text - done
        elif line1 == "true" and line2 == "true" and line3 == "false" and line4 == "false" and warning == False and header == False:
            create_text(text=text1, scale=0.118, x_lim=1.0, y_lim=4, thick_mul=2.5, x_pos=0.5, y_pos=0.4, image=image,
                        text_clr=font_color_name)
            create_text(text=text2, scale=0.118, x_lim=1.0, y_lim=4, thick_mul=2.5, x_pos=0.5, y_pos=0.6, image=image,
                        text_clr=font_color_name)

        # 2. writing the message in two lines for RECOG FAIL error message - done
        elif line1 == "true" and line2 == "true" and line3 == "false" and line4 == "false" and warning == True:
            create_text(text=text1, scale=0.118, x_lim=1.0, y_lim=4, thick_mul=2.5, x_pos=0.5, y_pos=0.4, image=image,
                        text_clr=font_color_name)
            create_text(text=text2, scale=0.118, x_lim=1.0, y_lim=4, thick_mul=2.5, x_pos=0.5, y_pos=0.6, image=image,
                        text_clr=font_color_name)

        # 3. writing the message in one line - backup
        elif line1 == "true" and line2 == "false" and header == True:
            create_text(text=text1, scale=0.5, x_lim=0.1, y_lim=0.1, x_pos=0.8, y_pos=0.2, image=image,
                        text_clr=font_color_name)

        # 4. writing the message in two lines for default screen - date and salutation message - done
        elif line1 == "true" and line2 == "true" and line3 == "false" and line4 == "false" and header == True:
            if aspect_rat >= 1.6:
                thick_, thick_2, y_1, y_2, y_l = 3.5, 3.0, 0.35, 0.65, 0.3
            else:
                thick_, thick_2, y_1, y_2, y_l = 2.2, 2.0, 0.45, 0.55, 0.3
            create_text(text=text1, scale=0.1, x_lim=0.25, y_lim=0.25, x_pos=0.8, y_pos=0.10, image=image,
                        text_clr=font_color_name, thick_mul=2)
            create_text(text=text2, scale=0.15, x_lim=0.55, y_lim=0.8, thick_mul=thick_2, x_pos=0.5, y_pos=0.3,
                        image=image, text_clr=font_color_name)

        # 5. writing the message in three lines for MAINT command with dateNeeded
        elif line1 == "true" and line2 == "true" and line3 == "true" and line4 == "false" and header == True and warning == False:
            if aspect_rat >= 1.6:
                thick_, thick_2, y_1, y_2, y_l = 3.5, 3.0, 0.35, 0.65, 0.3
            else:
                thick_, thick_2, y_1, y_2, y_l = 2.2, 2.0, 0.45, 0.55, 0.3
            create_text(text=text1, scale=0.15, x_lim=0.5, y_lim=0.3,  thick_mul=2,x_pos=0.5, y_pos=0.25, image=image,
                        text_clr=font_color_name)
            create_text(text=text2, scale=0.15, x_lim=0.5, y_lim=0.3, thick_mul=2, x_pos=0.5, y_pos=0.45, image=image,
                        text_clr=font_color_name)
            create_text(text=text3, scale=0.1, x_lim=0.5, y_lim=4,  thick_mul=2,x_pos=0.5,y_pos=0.75, image=image,
                        text_clr=font_color_name)

        # 6. writing the message in four lines for printing information from USER command
        elif line1 == "true" and line2 == "true" and line3 == "true" and line4 == "true" and header == True and warning == False:
            if aspect_rat >= 1.6:
                thick_, thick_2, y_1, y_2, y_l = 3.5, 3.0, 0.35, 0.65, 0.3
            else:
                thick_, thick_2, y_1, y_2, y_l = 2.2, 2.0, 0.45, 0.55, 0.3
            create_text(text=text1, scale=0.1, x_lim=0.25, y_lim=0.25, x_pos=0.8, y_pos=0.10, image=image,
                        text_clr=font_color_name, thick_mul=2)
            create_text(text=text2, scale=0.15, x_lim=0.55, y_lim=0.8, thick_mul=thick_2, x_pos=0.5, y_pos=0.3,
                        image=image, text_clr=font_color_name)
            create_text(text=text3, scale=0.18, x_lim=1, y_lim=4, thick_mul=2, x_pos=0.5, y_pos=0.6, image=image,
                        text_clr=font_color_name)
            create_text(text=text4, scale=0.18, x_lim=1, y_lim=4, thick_mul=2, x_pos=0.5, y_pos=0.85, image=image,
                        text_clr=font_color_name)

        # 7. writing the message in four lines to print other RECOG error message - done
        elif line1 == "true" and line2 == "true" and line3 == "true" and line4 == "true" and header == True and warning == True:
            if aspect_rat >= 1.6:
                thick_, thick_2, y_1, y_2, y_l = 3.5, 3.0, 0.35, 0.65, 0.3
            else:
                thick_, thick_2, y_1, y_2, y_l = 2.2, 2.0, 0.45, 0.55, 0.3

            create_text(text=text1, scale=0.15, x_lim=1, y_lim=4, x_pos=0.5, y_pos=0.15, thick_mul=2, image=image,
                        text_clr=font_color_name)
            create_text(text=text2, scale=0.15, x_lim=1, y_lim=4, thick_mul=2, x_pos=0.5, y_pos=0.35, image=image,
                        text_clr=font_color_name)
            create_text(text=text3, scale=0.12, x_lim=1, y_lim=4, thick_mul=2, x_pos=0.5, y_pos=0.65, image=image,
                        text_clr=font_color_name)
            create_text(text=text4, scale=0.12, x_lim=1, y_lim=4, thick_mul=2, x_pos=0.5, y_pos=0.85, image=image,
                        text_clr=font_color_name)

            # final = image.copy() # is this copy needed to show the logo - NOT needed, Use the image used to print userName and userNumPlate for Logo (refer logo part)
        final = image

        if show == True:
            return image  # This is when we turn on the camera stream, we replace all the display_screen background with the camera frame

        else:
            # display the final image
            cv2.namedWindow(imshow_title, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(imshow_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # cv2.startWindowThread()

            ##### LOGO SETUP ############################
            # mode 1 is no logo
            # mode 2 is text logo
            # mode 3 is picture logo
            if mode_chosen == "1":
                final = image  # just return the image with no logo
            if mode_chosen == "2":
                image = cv2.putText(image, logoName, (80, 100), cv2.FONT_HERSHEY_DUPLEX, 2, font_color_name, 2,
                                    cv2.LINE_AA)  # return a text-based logo
                final = image
            if mode_chosen == "3":
                logo = cv2.imread('/home/elid/anpr_display/company_logo/logo.jpg')  # read logo from the company_logo folder. The logo will be in logo.jpg format
                # image[15:logo_height,15:logo_width,:] = logo[15:logo_height,15:logo_width,:]  <--- initial attempt is not working properly

                logo = cv2.resize(logo, (150, 120))  # resize the logo to width 150px and height 120px
                logo_height, logo_width, _ = logo.shape
                # print("logoheight"+str(logo_height)+"widht"+str(logo_width))
                x = 20  # logo x position
                y = 20  # logo y position
                roi = image[y: y + logo_height,
                      x: x + logo_width]  # the region of interest where the logo will be placed
                logo_mask = logo[:, :, 2]  # Assuming the logo image has an alpha(transparency) channel
                logo_mask_inv = cv2.bitwise_not(logo_mask)
                logo_image_rgb = logo[:, :, 0:3]
                background_roi = cv2.bitwise_and(roi, roi, mask=logo_mask_inv)
                logo_roi = cv2.bitwise_and(logo_image_rgb, logo_image_rgb, mask=logo_mask)
                combined_roi = cv2.add(background_roi, logo_roi)
                image[y:y + logo_height, x:x + logo_width] = combined_roi
                final = image

            cv2.imshow(imshow_title, final)  # show the final image
            # Set the wait time to be 1 seconds
            cv2.waitKey(set_time)
            # cv2.destroyAllWindows()

        return 0

    while (True):
        current_time = date_t.now().strftime("%-I:%M %p")
        if camCanAccess == True and defaultRead == False:
            if readConfig(str(cmdSet)):
                camStreamEnable = "OFF"
                camStreamCount = 0
                textDisplayCount = 0
                camStreamStatus = False
                displayState = 0
            camCanAccess = False

        if camStreamStatus == True and camStreamEnable == "OFF":
            camStreamCount = 0
            textDisplayCount = 0
            camStreamStatus = False
            displayState = 0
        time3 = time.perf_counter()
        if camStreamEnable == "ON":
            if camStreamStatus == False:
                camStreamStatus = True
            if displayState == 0:
                camStreamCount += 1
                display_screen = display(show=True)
                with myLock:
                    display_frame = frame.copy()
                display_frame = cv2.rectangle(display_frame, (100, 100), (1200, 600), RED, 10)
                display_screen[height_diff:display_frame.shape[0] + height_diff,
                width_diff:display_frame.shape[1] + width_diff] = display_frame
                cv2.namedWindow(imshow_title, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(imshow_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.startWindowThread()
                cv2.imshow(imshow_title, display_screen)
                display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),
                                            (width_px, height_px))
                if camStreamCount >= camStreamTotal:
                    camStreamCount = 0
                    displayState = 1
            elif displayState == 1:
                textDisplayCount += 1
                display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),
                                            (width_px, height_px))
                # display_screen = display(show=False)
                display(text1="Put number plate", text2="inside red box", line1="true", line2="true", line3="false",
                        line4="false", warning=False, header=False, chosen_theme=theme_chosen)
                if textDisplayCount >= textDisplayTotal:
                    textDisplayCount = 0
                    displayState = 0
        time4 = time.perf_counter()
        
        if camStreamEnable == "OFF" and CamConnectStatus == AttemptingConnection:
            #display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px, height_px))
            display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/4.jpg"),(width_px, height_px))
            display(connecting, line1="true", chosen_theme=theme_chosen)
            
        if camStreamEnable == "OFF" and CamConnectStatus == ShowingSuccessScreen:
            #display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px, height_px))
            display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/4.jpg"),(width_px, height_px))
            display(established, line1="true", set_time=2500, chosen_theme=theme_chosen)
            CamConnectStatus = ConnectionAlreadyEstablished
            
        if camStreamEnable == "OFF" and CamConnectStatus == ConnectionAlreadyEstablished:
            # ========USER COMMAND==========#
            if cmdSet == "USER":
                logging("Displayed on the display screen : Username =" + str(recv[1]) + " Numplate = " + str(recv[2]))  # username and numplate displayed on the display screen
                userName = str(recv[1])
                userNumPlate = str(recv[2])
                if len(userName) > 10:
                    userName = userName[:10]

                if entrance_check == "YES":
                    userDisplayCount += 1
                    display(text1=str(current_time), text2="WELCOME", text3=str(userName), text4=str(userNumPlate),
                            line1="true",
                            line2="true", line3="true", line4="true", header=True, mode_chosen=mode, set_time=3500,
                            logoName=logoname, chosen_theme=theme_chosen)
                    cmdSet = ""
                elif entrance_check == "NO":
                    userDisplayCount += 1
                    display(text1=str(current_time), text2="SEE YOU AGAIN", text3=userName, text4=userNumPlate,
                            line1="true",
                            line2="true", line3="true", line4="true", header=True, mode_chosen=mode, set_time=3500,
                            logoName=logoname, chosen_theme=theme_chosen)
                    # display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px,height_px))
                    cmdSet = ""

            # ==========ENTRANCE COMMAND===========#
            elif cmdSet == "ENTRANCE":
                # print("from ifelseblock" + str(entrance_temp))
                if entrance_temp == "YES":
                    if entrance_check == "YES":
                        entrance_check = "YES"
                    elif entrance_check == "NO":
                        entrance_check = "YES"
                elif entrance_temp == "NO":
                    if entrance_check == "YES":
                        entrance_check = "NO"
                    elif entrance_check == "NO":
                        entrance_check = "NO"
                cmdSet = ""

            # =========TEST  COMMAND========#
            elif cmdSet == "TEST":
                display(testContent, line1="true", set_time=2500, chosen_theme=theme_chosen)
                cmdSet = ""

            # ===========RECOG COMMAND========#
            elif cmdSet == "RECOG":
                display_screen = np.zeros((height_px, width_px, 3), np.uint8)
                display_screen[:] = (0, 0, 180)

                if recv[1] != "FAIL":
                    userName = str(recv[2])
                    userNumPlate = str(recv[3])
                    if len(userName) > 10:
                        userName = userName[:10]

                # if user is not recognized
                if recogMsg == "FAIL":
                    display(access_denied, car_unrecognized, line1="true", line2="true", line3="false", line4="false",
                            warning=True, header=True, set_time=4000, logoName=logoname, chosen_theme='1')

                # if user is blacklisted
                elif recogMsg == "BLACKLIST":
                    display(userName, userNumPlate, access_denied, car_blacklisted, line1="true", line2="true",
                            line3="true", line4="true", warning=True, header=True, set_time=4000, logoName=logoname,
                            chosen_theme='1')

                # if user vehicle color is not the same as in database
                elif recogMsg == "COLOR":
                    display(userName, userNumPlate, access_denied, car_wrong_color, line1="true", line2="true",
                            line3="true", line4="true", warning=True, header=True, set_time=4000, logoName=logoname,
                            chosen_theme='1')

                # if user vehicle make is not the same as in database
                elif recogMsg == "MAKE":
                    display(userName, userNumPlate, access_denied, car_wrong_make, line1="true", line2="true",
                            line3="true", line4="true", warning=True, header=True, set_time=4000, logoName=logoname,
                            chosen_theme='1')

                # set cmdSet to "" to bring the screen back to default screen
                cmdSet = ""
            
            #========= CAM command ================#
            elif cmdSet == "CAM":
                #svcname = "anpr_server_and_display.service"
                subprocess.run(['sudo', 'systemctl', 'restart', 'anpr_server_and_display.service'] )
                #sys.exit(0)
                cmdSet = ""

            # ================DEFAULT SCREEN and MAINT COMMAND====================#
            else:
                if endmonth == "JAN":
                    convmonth = 1
                elif endmonth == "FEB":
                    convmonth = 2
                elif endmonth == "MAR":
                    convmonth = 3
                elif endmonth == "APR":
                    convmonth = 4
                elif endmonth == "MAY":
                    convmonth = 5
                elif endmonth == "JUN":
                    convmonth = 6
                elif endmonth == "JUL":
                    convmonth = 7
                elif endmonth == "AUG":
                    convmonth = 8
                elif endmonth == "SEP":
                    convmonth = 9
                elif endmonth == "OCT":
                    convmonth = 10
                elif endmonth == "NOV":
                    convmonth = 11
                else:
                    convmonth = 12
                    
                enddateadded1 = int(enddate) + 1 # to make sure the display shows maintenance mode on the last day of the maintenance
                endyearadded20 = "20" + str(endyear)
                #startyearadded20 = "20" + str(startyear)
                endperiod = (date_t(int(endyearadded20),int(convmonth),int(enddate))) + timedelta(days=1)
                currperiod = date_t.now()
                #startperiod = date_t(int(startyearadded20),int(convmonth),int(startdate))
                
                if entrance_check == "YES":
                    if maint_mode == "ON" and maint_date_needed == "NO":
                        display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)), (width_px, height_px))
                        display(under, maintenance, line1= "true", line2= "true", line3= "false", line4= "false", warning= True, header= True ,logoName= logoname, chosen_theme= theme_chosen)
                    elif maint_mode == "ON" and maint_date_needed == "YES":
                        if currperiod <= endperiod:
                            display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px, height_px))
                            display(under, maintenance, "{} {} {} - {} {} {}".format(startdate, startmonth, startyear, enddate, endmonth, endyear), line1= "true", line2= "true", line3= "true", line4= "false", header= True, warning= False, logoName= logoname, chosen_theme=theme_chosen)
                        if currperiod > endperiod:
                            display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)), (width_px, height_px))
                            display(under, maintenance, line1= "true", line2= "true", line3= "false", line4= "false", warning= True, header= True ,logoName= logoname, chosen_theme= theme_chosen)
                    else:
                        display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px, height_px))
                        display(text1=str(current_time), text2="WELCOME", line1="true", line2="true", header=True, mode_chosen=mode, logoName=logoname, chosen_theme=theme_chosen)

                elif entrance_check == "NO":
                    if maint_mode == "ON" and maint_date_needed == "NO":
                        display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)), (width_px, height_px))
                        display(under, maintenance, line1= "true", line2= "true", line3= "false", line4= "false", warning= True, header= True ,logoName= logoname, chosen_theme= theme_chosen)
                    elif maint_mode == "ON" and maint_date_needed == "YES":
                        if currperiod <= endperiod:
                            display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px, height_px))
                            display(under, maintenance, "{} {} {} - {} {} {}".format(startdate, startmonth, startyear, enddate, endmonth, endyear), line1= "true", line2= "true", line3= "true", line4= "false", header= True, warning= False, logoName= logoname, chosen_theme=theme_chosen)
                        elif currperiod > endperiod:
                            display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)), (width_px, height_px))
                            display(under, maintenance, line1= "true", line2= "true", line3= "false", line4= "false", warning= True, header= True ,logoName= logoname, chosen_theme= theme_chosen)
                    else:
                        display_screen = cv2.resize(cv2.imread("/home/elid/anpr_display/images/{}.jpg".format(theme_chosen)),(width_px, height_px))
                        display(text1=str(current_time), text2="SEE YOU AGAIN", line1="true", line2="true", header=True, mode_chosen=mode, logoName=logoname, chosen_theme=theme_chosen)

# ===============================MAIN THREAD===================================================#
#  writes into the config file and read the things that are inside the config file
# two threads are spawned here - display thread ( to handle display related tasks)
#                                                               - cam thread ( to handle grab and retrieving of camera frames)
if __name__ == '__main__':
    CamConnectStatus = AttemptingConnection
    subprocess.run(["/usr/bin/python3", "/home/elid/anpr_display/timeDateSync.py"])
    lock = Lock()
    displayThread = Thread(target=DisplayThread, args=(lock,))
    camThread = Thread(target=CameraStreamThread, args=(lock,))
    displayThread.start()
    camThread.start()
    # create the socket connection with TCP/IP
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_name = socket.gethostname()  # obtain the hostname
    socket_address = (host_ip, port)  # connect the socket to the port where the server is listening
    logging(f'Host ip used is {host_ip} and port used is {port}')
    server_socket.bind(socket_address)  # bind the local address to the socket
    server_socket.listen(5)  # listen to accept connections
    server_socket.setblocking(False)  # set the socket to non-blocking
    sockets_to_monitor = [server_socket]  # put the server socket inside the read list
    # logging('Connection established')  # update the logfile

    while (True):
        if camCanAccess == False and defaultRead == False:
            # to write into the configfile from the main thread
            file = '//home//elid//anpr_display//config//config.ini'
            config = ConfigParser()
            config.read(file)
            FORMAT = 'utf-8'
            try:
                ready_to_read, _, _ = select.select(sockets_to_monitor, [], [])
            except:
                print("Something went wrong  here")

            # continuously listening to client_socket
            for sock in ready_to_read:
                # if socket is received
                if sock == server_socket:
                    client_socket, address = server_socket.accept()
                    sockets_to_monitor.append(client_socket)
                else:
                    data = client_socket.recv(buffersize).decode(FORMAT)
                    if data:
                        recv0 = simple_decrypt(data)
                        recv = recv0.split('|')

                        # BELOW IS WHERE MAIN THREAD RECEIVES COMMAND FROM USER AND EDIT THE CONFIG FILE TO ADD THE INFORMATION
                        # ================USER command=======================================
                        if recv[0] == command_list[0]:
                            cmdSet = recv[0]
                            # shows the username in first line, car plate in second line
                            current_time = date_t.now().strftime("%-I:%M %p")
                            config.set('Stream', 'status', 'OFF')
                            config.set('User', 'name', recv[1])
                            config.set('User', 'numplate', recv[2])
                            with open(file, 'w') as configfile:
                                config.write(configfile)
                            logging("Written into config file : User = " + str(recv[1]) + " Numplate : " + str(
                                recv[2]))  # what is written into the config file
                            camCanAccess = True

                        # =================STREAM command============================================
                        elif recv[0] == command_list[1]:
                            # print("inside stream comm main thread")
                            cmdSet = recv[0]
                            # replace the stream status in config file
                            if os.path.isfile(file) == True and recv[1] in stream_status_list:
                                config.set('Stream', 'status', f'{recv[1]}')
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging("Config.ini does not exist")
                                pass
                            camCanAccess = True

                        # =====================IPADDR command========================================
                        # recv[1]= elidanpr ; recv[2] = ip cam address ; recv[3] = DMS ip
                        elif recv[0] == command_list[2] and get_ip_type(recv[1], recv[2], recv[3]) == True and len(recv) == 4:
                            cmdSet = recv[0]
                            # replace ip address in network_monitor.sh
                            if os.path.isfile(file) == True and isinstance(recv[3], str) == True:
                                replace_ip(recv[1], recv[2], recv[3])
                                subprocess.check_call(restart_command, shell=True)
                                #config.set('Values', 'cameraIP', recv[2])
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                                # exit()
                                # logging(f'Changed ELID ANPR IP address to {recv[1]}, Camera IP address to {recv[2]} and DMS IP to {recv[3]}')
                            else:
                                pass

                        # ==========================ENTRANCE command==========================
                        elif recv[0] == command_list[3] and recv[1] in entrance_check_list and len(recv) == 2:
                            # print('3')
                            cmdSet = recv[0]
                            # replace the entrance status in config file
                            if os.path.isfile(file) == True and isinstance(recv[1], str) == True:
                                config.set('Background', 'entrance', recv[1])
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging(f'Config.ini does not exist')
                                pass
                            camCanAccess = True
                            # print('3 done')

                        # ====================THEME command================================
                        elif recv[0] == command_list[4] and len(recv) == 2 and isinstance(recv[1], str) == True:
                            cmdSet = recv[0]
                            if os.path.isfile(file) == True:
                                config.set('Background', 'num', recv[1])
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging(f'Config.ini does not exist')
                                pass
                            camCanAccess = True

                        # =====================RECOG command===================================
                        elif recv[0] == command_list[5] and recv[1] in recog_list:
                            cmdSet = recv[0]
                            # replace the entrance status in config file
                            if os.path.isfile(file) == True and isinstance(recv[1], str) == True:
                                config.set('Recog', 'type', recv[1])
                                # print(recv[1])
                                if recv[1] != "FAIL":
                                    config.set('User', 'name', recv[2])
                                    config.set('User', 'numplate', recv[3])
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging(f'Config.ini does not exist')
                                pass
                            camCanAccess = True

                        # =========================TEST command==================================
                        elif recv[0] == command_list[6] and len(recv) == 2 and isinstance(recv[1], str) == True:
                            cmdSet = recv[0]
                            testContent = recv[1]
                            # display any text received

                        # =====================Name command - company logo============================
                        elif recv[0] == command_list[7] and len(recv) == 2 and isinstance(recv[1], str) == True:
                            cmdSet = recv[0]
                            # replace the stream status in config file
                            if os.path.isfile(file) == True and len(recv) == 2:
                                config.set('Logo', 'compname', f'{recv[1]}')
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging("Config.ini does not exist")
                                pass
                            camCanAccess = True

                        # ===========================MODE command for logo============================
                        elif recv[0] == command_list[8] and len(recv) == 2:
                            cmdSet = recv[0]
                            # replace the logo mode in config file
                            if os.path.isfile(file) == True and len(recv) == 2 and recv[1] in logo_mode_list:
                                config.set('Logo', 'mode', f'{recv[1]}')
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                # logging("Config.ini does not exist")
                                pass
                            camCanAccess = True

                        # =====================MAINT command======================================
                        elif recv[0] == command_list[9]:
                            cmdSet = recv[0]
                            # if MAINT mode with no date
                            if os.path.isfile(file) == True:
                                config.set('Maintenance', 'mode', recv[1])
                                config.set('Maintenance', 'dateNeeded', recv[2])
                            # if MAINT mode with date
                                if recv[2] == 'YES':
                                    config.set('Maintenance', 'startdate', recv[3])
                                    config.set('Maintenance', 'startmonth', recv[4])
                                    config.set('Maintenance', 'startyear', recv[5])
                                    config.set('Maintenance', 'enddate', recv[6])
                                    config.set('Maintenance', 'endmonth', recv[7])
                                    config.set('Maintenance', 'endyear', recv[8])
                                
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging(f'Config.ini does not exist')
                                pass
                            camCanAccess = True

                        # =========================CAM command==================================
                        elif recv[0] == command_list[10] and len(recv) == 4:
                            cmdSet = recv[0]
                            if os.path.isfile(file):
                                config.set('Cam', 'ip_cam', recv[1])
                                config.set('Cam', 'cam_username', recv[2])
                                config.set('Cam', 'cam_password', recv[3])
                                with open(file,'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging("Config.ini does not exist")
                                pass
                            camCanAccess = True
                            
                        # ======================RESET command for factory reset============================
                        elif recv[0] == command_list[-1]:
                            cmdSet = recv[0]
                            # replace the logo mode in config file
                            if os.path.isfile(file):
                                config.set('Stream', 'status', 'OFF')
                                config.set('Background', 'num', '1')
                                config.set('Background', 'entrance', 'YES')
                                config.set('Logo', 'compname', 'ELID')
                                config.set('Logo', 'mode', '3')
                                with open(file, 'w') as configfile:
                                    config.write(configfile)
                            else:
                                logging("Config.ini does not exist")
                                pass
                            camCanAccess = True
                    else:
                        sock.close()
                        sockets_to_monitor.remove(sock)
