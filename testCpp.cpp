/*
 * elidanpr.cpp
 *
 * Author: shan
 *
 */

//============================ ALL HEADER FILES [START] ===========================================

// standard header files
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <stdint.h>
#include <cmath>
#include <deque>

// application-specific files
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

// Developed header files
#include "anpr.h"
#include "wiegand.h"

//============================ ALL HEADER FILES [END] ===========================================

//============================== ALL GLOBAL VARIABLES [START] ========================================

int PIN_DATA_0; // GPIO pin number for data_0, as per wiring pi
int PIN_DATA_1; // GPIO pin number for data_1, as per wiring pi

//============================== ALL GLOBAL VARIABLES [END] ========================================

int IsLicensed(char mod,char *comm,char *sn); // function declaration of routine that checks for license. The object file (license.o) containing the function definition is provided by Mr. Ng.

int main (int argc, char *argv[])
{

//======================================== ALL VARIABLES [START] ======================================

	Mat frame,frameCopy; // original frame,  copy of frame for recording purposes
	Mat lpResult; // the recognized license plate in TEXT format
	Mat regionMask; // Binary mask specifying which regions are to be ignored!
	Mat lp; // License plate, obtained from applying the car & plate detector
	Mat lpGray; // Grayscale of lp
	Mat lpBinary; // Binarized lp obtained via adaptive foreground extraction
	Mat lpGrayWarped; // Warped grayscale image after applying perspective transformation
	Mat lpBinaryWarped; // Warped binary image after applying perspective transformation
	Mat tessBlock; // Binary image consisting of concatenation of segmented + sorted blobs, to be fed to OCR
	Mat car; // car region, obtained from applying the car & plate detector

	int cameraStreamCheckInterval = 25; // decodedFrame is written to disk as proof of stable camera stream only when frameCounter%cameraStreamCheckInterval=0. Default value = 25.
	int frameInterval = 1; // If the frame number divided by frameInterval equals zero, the frame is processed
	int samplingInterval = 1; // If the frame number divided by samplingInterval equals zero, the tesseract input is stored as a site-specific training image
	int fpsSet = 100; // frames per second set by system
	int frameCounter = 0; // this value indicates the counter of the frame being processed
	int databaseCheckCounter = 0; // incremental value used to determine if database change needs to be checked
	int wd_counter = 0; // counter for kicking the watchdog
	int counterToSaveCopy; // timeTillSaveCopy * frameRate
	int counterToIdentifyUnrecognized; //timeTillIdentifyUnrecognized * frameRate
	int counterToFlipRedLight; // stateChangeInterval * frameRate;
	int counterForStreamOn; // plateFoundTimeForStreamOn * frameRate;
	int counterForStreamOff; // plateMissTimeForStreamOff * frameRate;
	int noRecognitionCounter = 0; // number of consecutive times no recognition was performed
	int delayPostMatching = 0; // counter to track how many frames have passed after a recognition has been performed
	int delayPostPreviousMatch = 0; // counts the number of frames passed since previous recognition - but this is different from delayPostMaching as this is used to decide when the same number plate will be allowed to send wiegand again
	int plateFoundCounter = 0; // consecutive number of times plate was found
	int noPlateFoundCounter = 0; // consecutive number of times plate was NOT found
	int currentNumOfTrainingImages = 0; // number of site-specific training images already acquired
	int currentNumTrainOriginal = -1; // initialize the current number of original number plates captured
	int currentNumTrainWarped = -1; // initialize the current number of warped number plates captured
	int originalIntervalCount = 0; // number of frames passed since the last original image was saved
	int warpedIntervalCount = 0; // number of frames passed since the last warped image was saved
	bool didPerspectiveTransform = false; // boolean flag that indicates whether perspective transformation was done in the current frame. Used to determine if warped image should be saved or not
	fstream sampleCountLogger; // stream that records current image counts for both original and warped into file.

	// "DbMatch" in the plateFromLastDbMatch happens on each successive iteration where OCR output is matched with a db entry
	// But "AuthorizedPlate" in previousAuthorizedPlate might be done in a single iteration if only Plate recognition is required OR multiple iterations if make and color must also be verified
	string previousAuthorizedPlate = "Empty"; // this stores the previous matched license plate that was subjected to access status determination until a new one is found
	string plateFromLastDbMatch = "Empty"; // this stores the license plate from the previous db match
	string plateNumber; // license plate number that is output by the OCR

	bool redLightState = false; //keeps track whether the red light is on or off
	bool plateInferencePerformed = false; // flag that indicates whether plate recognition was attempted in the current iteration
	bool allow_recognition_update = true; // flag that decides if recognition should be allowed
	bool haveFrameCopyToSave = false; // flag that indicates whether a copy of frame exists to be saved. This is used for saving images of those cars which could not be sufficiently analyzed to determine access status.
									  //flag is set to false whenever a valid access status is determined
	bool recordTrainImsFinished = false; // flag that identifies if recording of training images is complete or not
	bool barrierActivated = false; // flag that indicates whether the relay for the barrier is turned ON or OFF
	bool showingLiveCamView = false; // flag that indicates whether the live camera view is currently being shown on the screen.

	string ocrSubfolder = "ocr/"; // subfolder name that is searched in the ocr model path name
	int modelNameBeginPos = -1; // position in the ocr model path name string, where the subfolder string is found
	string ocrExtension = ".traineddata"; // ocr model file extension
	int modelExtensionBeginPos = -1; // position in ocr model path name string, where the model extension is found

	FILE *ocrTried; // File stream to check for file that indicates ocr attempt
	FILE *trainDone; // File stream to check for file that indicates recording of site-specific training images is complete

	unsigned char wiegBits[3]={0,0,0}; // 24 bit wiegand number (2 parity bits) needs three characters

	string anprLogFile = "/home/elid/lpr/lprLogs/anpr/%04d-%02d-%02d.log"; // date-based file where all standard anpr events are logged
	ofstream anprLogger; // stream that writes the anpr events to the respective log file
	ofstream spawnLogger; // stream that logs the time whenever the anpr Program is spawned
	OperationSettings anprCfg; // struct that reads and contains all the anpr settings/parameters

	int databaseSize; // number of entries read from the database file
	vector<string>recentAttempts; // vector that stores the attempts made to recognize a license plate. Utilized for further processing only if recognition took longer than the specified duration

	double carPlateDetTime = 0; // time per frame for car & plate detection
	double makeDetTime = 0; // time per frame for make detection
	double colorClsTime = 0; // time per frame for color classification
	double ocrTime = 0; // time per frame for OCR

	AccessStatus access_status = STAT_UNDETERMINED; // set the access status as undetermined

	string displayContents = ""; // the contents to be sent to the display module
	string prefix_authorized = "USER|"; // prefix for authorized access
	string prefix_blocked_unrecognized = "RECOG|FAIL"; // prefix for blocked due to failed matching with database
	string prefix_blocked_blacklist = "RECOG|BLACKLIST|"; // prefix for blocked due to blacklisted
	string prefix_blocked_make = "RECOG|MAKE|"; // prefix for blocked due to make mismatch
	string prefix_blocked_color = "RECOG|COLOR|"; // prefix for blocked due to color mismatch
	string stream_ON = "STREAM|ON"; // command for turning on camera stream
	string stream_OFF = "STREAM|OFF"; // command for turning off camera stream
	ofstream displayContentsStream; // stream that writes to the disk the contents to be sent to display module
//======================================== ALL VARIABLES [END] ========================================

	logger_initialize(anprLogger, anprLogFile); // initialize stream for logging anpr events
	anprLogger << "========================================================================================================================================" << endl <<endl;
	anprLogger << "Setting up system for ANPR LIVE OPERATION..." << endl;

//================================LICENSE CHECKING [START]=============================================

	logger_addTimeStamp(anprLogger, 2);
	char anprSerial[1000];
	char licensePort[1000]="/dev/ttyS1";
	char mod='A';
	if(!IsLicensed(mod,licensePort,anprSerial))
		anprLogger<<"This unit of ELID-ANPR is licensed : "<<anprSerial<<endl;
	else
	{
		anprLogger<<"This unit of ELID-ANPR is NOT licensed! ANPR Program TERMINATED!"<<endl;
		return -1;
	}

//============================== LICENSE CHECKING [END]=================================================

//============================== READ PROGRAM SETTINGS [START] =========================================

	logger_addTimeStamp(anprLogger, 1);
	if(!readConfigurations(anprCfg,anprLogger))
		anprLogger<<"Reading of parameters complete" << endl;
	else
	{
		anprLogger<<"Error occurred while reading parameters" << endl;
		return -1;
	}

//============================== READ PROGRAM SETTINGS [END] ===========================================

//======================== RECORD THE ANPR PROGRAM SPAWN EVENT [START] =================================

	logger_initialize(spawnLogger,anprCfg.spawnLogFile);
	logger_addTimeStamp(spawnLogger, 2);
	spawnLogger << "ELID-ANPR started ..." << endl;
	logger_destroy(spawnLogger);

//======================== RECORD THE ANPR PROGRAM SPAWN EVENT [END] ===================================

//================== CREATE TODAY'S FOLDER IF IT DOES NOT EXIST [START] ================================

	time_t now = time(0);
	tm *t = localtime(&now);
	char *plateRecToday = (char *)malloc(sizeof(char)*200); // full path for today's directory to store saved images of recognized/non-recognized vehicles
	snprintf(plateRecToday,200,anprCfg.imageSaveFolder.c_str(),t->tm_year+1900,t->tm_mon+1,t->tm_mday);  // create the name of today's directory based on today's date
	struct stat info;
	logger_addTimeStamp(anprLogger, 1);
	if( stat( plateRecToday, &info ) != 0 ) // if the folder name pointed to by plateRecToday does not exist, execute the following block
	{
		anprLogger << plateRecToday << " does not exist. Creating now ..." <<endl;
		const int dir_err = mkdir(plateRecToday, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // create the directory
		if (-1 == dir_err)
		{
			anprLogger << "Error creating directory" << endl;
			exit(1);
		}
		else
			anprLogger << "Directory for saving images created successfully..."<<endl;
	}
	else if( info.st_mode & S_IFDIR ) // if folder already exists
		anprLogger << plateRecToday << " already exists. Accessing for recording recognized plates ..." <<endl;
	else // path might be an existing file!
	{
		anprLogger << plateRecToday << " might be a file. Please delete it to create the folder" <<endl;
		return -1;
	}

	char *nextPlatePath = (char *)malloc(sizeof(char)*200); // path name where next image will be stored
	char *plateTimeStamp = (char *)malloc(sizeof(char)*15); // time stamp for the image of vehicle to be stored

//======================CREATE TODAY'S FOLDER IF IT DOES NOT EXIST [END]===============================

//========================== GPIO INITIALIZATIONS [START] =============================================

	if(wiringPiSetup()==-1) // initialize the GPIO pins
	{
		logger_addTimeStamp(anprLogger,2);
		anprLogger<<"GPIO could not be initialized properly. Program Exiting ..."<<endl;
		exit(-1);
	}
	else
	{

		if(anprCfg.useStandardWiegand)
		{
			PIN_DATA_0 = anprCfg.PIN_DATA_0; // Assign the pin number for DATA 0 of wiegand
			PIN_DATA_1 = anprCfg.PIN_DATA_1; // Assign the pin number for DATA_1 of wiegand
		}
		else // connect the pins in reverse. Meaning, use DO pin for sending D1 signal and vice versa. This is done to cater for the situation where the wiring is erroneously connected in reverse.
		{
			PIN_DATA_0 = anprCfg.PIN_DATA_1;
			PIN_DATA_1 = anprCfg.PIN_DATA_0;
		}

		// set mode of indicator lights as OUTPUT
		pinMode(anprCfg.PIN_GREEN, OUTPUT);
		pinMode(anprCfg.PIN_RED, OUTPUT);

		pinMode(anprCfg.PIN_WDOG, OUTPUT); // set the watchdog pin as output

		if(anprCfg.controlIndicatorLights) // if LED light control is enabled
		{
			// turn ON both lights
			digitalWrite(anprCfg.PIN_GREEN,pin_HIGH);
			digitalWrite(anprCfg.PIN_RED,pin_HIGH);
		}

		pinMode(anprCfg.PIN_BARRIER_RELAY, OUTPUT); // set the barrier relay pin as output

		logger_addTimeStamp(anprLogger,2);
		anprLogger<<"GPIO initialized successfully."<<endl;
	}

	const char *wiegcommandstring = "sudo nice -n -20 /home/elid/lpr/wiegGen -n %d -s noshow &"; // command string that sends wiegand.
	char *sendwiegandcommand = (char *)malloc(sizeof(char)*100); // allocate memory where command string will be copied
//========================= GPIO INITIALIZATIONS [END] ===============================================

//======================= APPLY MASKS TO SPECIFY PROCESSING REGION [START] ===========================

	createRegionMasks(regionMask, anprCfg); // create a image that contains the regions to be masked filled with 0's and 1's everywhere else

//====================== APPLY MASKS TO SPECIFY PROCESSING REGION [END] ==============================

//======================= LOAD TENSORFLOW-LITE CAR & PLATE DETECTOR [START]===================================

	// Create an instance of the car and plate detector
	std::unique_ptr<CarAndPlateDetector>carPlusPlateDet= make_unique<CarAndPlateDetector>(anprCfg);
	logger_addTimeStamp(anprLogger, 2);
	anprLogger <<"Attempting to initialize the Car & Plate Detector..." << endl;
	logger_addTimeStamp(anprLogger, 2);
	if(logger_logModelInitStatus(anprLogger, carPlusPlateDet->initialize())!= DETECTOR_INIT_SUCCESS) // Abort program if detector initialization is unsuccessful ; note initialization member function is invoked within the logging function
		return -1;
	else
	{
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << carPlusPlateDet->printDetails() << endl; // Log the model name and class labels filename
	}

//======================= LOAD TENSORFLOW-LITE  CAR & PLATE DETECTOR  [END]=====================================

//====================== CREATE A CHARACTER SEGMENTER [START] ====================================================
	// Create an instance of the Character Segmenter
	std::unique_ptr<CharacterSegmenter> charSegm = make_unique<CharacterSegmenter>(anprCfg);
//====================== CREATE A CHARACTER SEGMENTER [END]=====================================================

//======================= LOAD TENSORFLOW-LITE MAKE DETECTOR MODEL [START]==========================================
	// Create an instance of the make detector
	std::unique_ptr<MakeDetector>makeDet = make_unique<MakeDetector>(anprCfg);
	logger_addTimeStamp(anprLogger, 2);
	anprLogger <<"Attempting to initialize the Make Detector..." << endl;
	logger_addTimeStamp(anprLogger, 2);
	if(logger_logModelInitStatus(anprLogger,makeDet->initialize())!= DETECTOR_INIT_SUCCESS) // Abort program if detector initialization is unsuccessful
		return -1;
	else
	{
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << makeDet->printDetails()<< endl; // Log the model name and class labels filename
	}
//======================= LOAD TENSORFLOW-LITE LOGO DETECTOR MODEL [END]============================================

//======================= LOAD TENSORFLOW-LITE COLOR CLASSIFIER MODEL [START]=======================================
	// Create an instance of the make detector
	std::unique_ptr<ColorClassifier>colorCls = make_unique<ColorClassifier>(anprCfg);
	logger_addTimeStamp(anprLogger, 2);
	anprLogger <<"Attempting to initialize the Color Classifier..." << endl;
	logger_addTimeStamp(anprLogger, 2);
	if(logger_logModelInitStatus(anprLogger,colorCls->initialize())!= CLASSIFIER_INIT_SUCCESS) // Abort program if detector initialization is unsuccessful
		return -1;
	else
	{
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << colorCls->printDetails() << endl; // Log the model name and class labels filename
	}
//======================= LOAD TENSORFLOW-LITE COLOR CLASSIFIER MODEL [END]=========================================

//======================= LOAD TESSERACT OCR MODEL [START]=====================================================

	tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI(); //tesseract ocr instance that will perform the recognition
	logger_addTimeStamp(anprLogger, 2);
	// Load the tesseract model
	anprLogger <<"Attempting to initialize the Optical Character Recognition (OCR) Model..." << endl;

	modelNameBeginPos = anprCfg.ocrModelPath.find(ocrSubfolder) + ocrSubfolder.length(); // Find the position in the ocr path name where ocr subfolder name starts
	modelExtensionBeginPos = anprCfg.ocrModelPath.find(ocrExtension); // Find the position in the ocr path name where the ocr Extension starts

	// OCR init functions requires _> input parameter 1 : path where to find the model, input parameter 2 : name of the model, excluding model extension, input parameter 3 : class of tesseract model used
	if(!ocr->Init(anprCfg.ocrModelPath.substr(0,modelNameBeginPos).c_str(), anprCfg.ocrModelPath.substr(modelNameBeginPos, modelExtensionBeginPos-modelNameBeginPos).c_str(),tesseract::OEM_LSTM_ONLY))
	{
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << "OCR model loaded. Initialization successful!" << endl;
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << "Model name : " << anprCfg.ocrModelPath.substr(modelNameBeginPos, anprCfg.ocrModelPath.length()-modelNameBeginPos) << endl;
	}
	else
	{
		anprLogger << "Failed to load ocr model. Program exiting" << endl;
		return -1;
	}

	ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE); // tesseract expects the characters to be recognized arranged in a single line

//======================= LOAD TESSERACT OCR  MODEL [END]======================================================

//================================== READ ANPR DATABASES [START] =======================================

	std::unique_ptr<DatabaseMatcher> dbMatcher = make_unique<DatabaseMatcher>(anprCfg); // create instance of database matcher - for reading database and matching against the license plate outputted by ocr
	logger_addTimeStamp(anprLogger, 2);
	anprLogger << "Reading the ANPR database ... "<< endl;
	databaseSize = dbMatcher->readDatabase(false); // Read the anpr database for the first time. Argument of false means this is NOT reading an updated database
	if( databaseSize == -1)
	{
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << "ANPR database reading failed! Database file may be empty, non-existent or corrupted"<< endl;
	}
	else
	{
		logger_addTimeStamp(anprLogger, 2);
		anprLogger << "ANPR database reading completed successfully. Total entries read = " << databaseSize << endl;
	}

	// Create an instance of the counter struct that holds all counts for various matching tasks and initialize it
	MatchingCounters matchCounts;
	matchCounts.resetCounts();
//=============================== READ ANPR DATABASES [END] ============================================

//====================== ESTABLISH CONNECTION WITH ANPR CAMERA [START]==================================

	VideoCapture anprCamera;// network ANPR camera stream
	logger_addTimeStamp(anprLogger, 1);
	anprLogger<< "Attempting to connect to the network ANPR camera" << endl;
	// live rtsp connection url to the LPC, based on username, password, IP address and rtsp port. subtype=0 accesses main stream (higher resolution), and subtype=1 accesses sub stream (lower resolution)
	string liveStreamUrl = "rtsp://" + anprCfg.LPC_USERNAME + ":" + anprCfg.LPC_PASSWORD + "@" + anprCfg.LPC_IP_ADDRESS + ":554/cam/realmonitor?channel=1&subtype=0";
	//string liveStreamUrl = "/home/elid/anprVids/elidVid_2.mp4";

	if (!anprCamera.open(liveStreamUrl)) // attempt to establish a connection to the camera and obtain a video stream
	{
		anprLogger << "Video stream could not be acquired! Program Exiting..." << endl;
		return -1;
	}
	else
		anprLogger<<"Video stream acquired successfully "<<endl;

	double fpsReal =  anprCamera.get(CAP_PROP_FPS); // get the original frame rate
	if(fpsReal > anprCfg.fpsNormalMax) // Abort the program if FPS of camera stream is abnormal
	{
		anprLogger <<"Aborting program due to abnormal FPS of camera stream!! Allowed maximum FPS = "<< anprCfg.fpsNormalMax << ". FPS of current camera stream = "<<fpsReal<<endl;
		return -1;
	}
	frameInterval = cvRound(fpsReal/anprCfg.fpsRequested); // the frame interval based on the requested frame rate
	if(frameInterval==0) // minimum frame rate is 1
		frameInterval=1;
	cameraStreamCheckInterval = frameInterval * anprCfg.cameraStreamCheckMultiplier; // interval at which a file will be written to disk indicating that this main program has a stable connection with the anpr camera
	fpsSet = cvRound(fpsReal/frameInterval); // determine the frame rate to be set as close as possible to the requested frame rate (using the previously calculated frameInterval)
	anprLogger<<"Resolution = "<<anprCamera.get(CAP_PROP_FRAME_WIDTH)<<" x "<<anprCamera.get(CAP_PROP_FRAME_HEIGHT);
	anprLogger<<" | Real Frame Rate = "<<anprCamera.get(CAP_PROP_FPS);
	anprLogger<<" | Requested Frame Rate = "<<anprCfg.fpsRequested;
	anprLogger<<" | Applied Frame Rate = "<<fpsSet;
	anprLogger<<" | Camera Stream Check Interval = "<<cameraStreamCheckInterval<<endl;


//====================== ESTABLISH CONNECTION WITH ANPR CAMERA [END]=====================================================

//====================== PREPARE FOR RECORDING SITE-SPECIFIC TRAINING IMAGES [START] =====================================
	if(anprCfg.recordTrainingImages)
	{
		logger_addTimeStamp(anprLogger, 1);
		struct stat infoTrain;
		if( stat(anprCfg.trainingImsFolder.c_str(), &infoTrain ) != 0 ) // if the folder for saving training images does not exist, create the folder
		{
			anprLogger << "Folder for saving site-speficic training images does not exist. Creating now ..." <<endl;
			const int dir_err = mkdir(anprCfg.trainingImsFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // create the directory
			if (-1 == dir_err)
				anprLogger << "Error creating directory" << endl;
			else
				anprLogger << "Directory for recording site-specific training images created successfully..."<<endl;
			currentNumOfTrainingImages = 0;
		}
		else if( infoTrain.st_mode & S_IFDIR ) // if folder already exists
		{
			anprLogger <<"Folder for saving site-speficic training images already exists. Attempting to determine the number of training images already saved ..." <<endl;
			struct dirent *entryTrain;
			DIR *dirTrain;
			if ((dirTrain = opendir(anprCfg.trainingImsFolder.c_str()))!=NULL)
			{
				while((entryTrain=readdir(dirTrain))!= NULL) // iterate through the folder contents to get file count
					++currentNumOfTrainingImages;
				closedir(dirTrain);
				if(currentNumOfTrainingImages>2)
					currentNumOfTrainingImages -= 2; // Reduce two because of '.' & '..'
				else
					currentNumOfTrainingImages = 0;
				anprLogger << "Required number of training images: "<< anprCfg.requiredNumOfTrainingImages << ". Existing number of training images: "<<currentNumOfTrainingImages<<endl;
			}
			else
				anprLogger << "Failed to determine the number of training images that already exist" << endl;
		}
		else // path might be an existing file!
		{
			anprLogger << anprCfg.trainingImsFolder << " might be a file. Please delete it to create the folder" <<endl;
		}

		samplingInterval = frameInterval * anprCfg.samplingInterval; // This is the interval at which training images will be saved to disk. Essentially converts time in seconds to number of frames
	}

	char *nextTrainImagePath = (char *)malloc(sizeof(char)*200); // path name where next image will be stored
	char *nextSampCounter = (char *)malloc(sizeof(char)*10); // string that will hold the value of the current total number of acquired site-specific training samples (with leading zeros)

//====================== PREPARE FOR RECORDING SITE-SPECIFIC TRAINING IMAGES [END] ==========================================


//====================== PREPARE FOR SAVING TRAINING SAMPLES OF PLATE IMAGES [START] ========================================
	if(anprCfg.recordPlateOriginal || anprCfg.recordPlateWarped)
	{
		logger_addTimeStamp(anprLogger, 1);
		struct stat infoTrain;
		if( stat(anprCfg.plateSamplesFolder.c_str(), &infoTrain ) != 0 ) // if the folder for saving training images does not exist, create the folder
		{
			anprLogger << "Folder for saving number plate images does not exist. Creating now ..." <<endl;
			const int dir_err = mkdir(anprCfg.plateSamplesFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // create the directory
			if (-1 == dir_err)
				anprLogger << "Error creating folder" << endl;
			else
				anprLogger << "Folder for saving number plate images created successfully..."<<endl;
		}
		else if( infoTrain.st_mode & S_IFDIR ) // if folder already exists
		{
			anprLogger <<"Folder for saving number plate images already exists..." <<endl;
		}
		else // path might be an existing file!
		{
			anprLogger << anprCfg.plateSamplesFolder << " might be a file. Please delete it to create the folder" <<endl;
		}

		fstream countReader;
		countReader.open(anprCfg.currentSampleCounts);
		if(countReader.good())
		{
			string nextLine;
			while(!countReader.eof())
			{
				getline(countReader,nextLine);
				if(nextLine.empty())
					continue;
				else
				{
					if(nextLine[0] == 'O')
					{
						currentNumTrainOriginal = stoi(nextLine.substr(nextLine.find('=')+1));
					}
					else if(nextLine[0] == 'W')
					{
						currentNumTrainWarped = stoi(nextLine.substr(nextLine.find('=')+1));
					}
				}
			}
		}

		if(currentNumTrainOriginal < 0)
			currentNumTrainOriginal = 0;
		if(currentNumTrainWarped < 0)
			currentNumTrainWarped = 0;

		samplingInterval = frameInterval * anprCfg.samplingInterval; // This is the interval at which training images will be saved to disk. Essentially converts time in seconds to number of frames

		if(anprCfg.recordPlateOriginal)
			anprLogger << "Required number of original number plate samples: "<< anprCfg.requiredNumOfPlateOriginal << ". Number of samples acquired: "<<currentNumTrainOriginal<<endl;
		else
			anprLogger <<"Recording of original number plate samples disabled"<<endl;

		if(anprCfg.recordPlateWarped)
			anprLogger << "Required number of warped number plate samples: "<< anprCfg.requiredNumOfPlateWarped << ". Number of samples acquired: "<<currentNumTrainWarped<<endl;
		else
			anprLogger <<"Recording of warped number plate samples disabled"<<endl;

	}

//====================== PREPARE FOR SAVING TRAINING SAMPLES OF PLATE IMAGES [END] ==========================================

	counterToSaveCopy = anprCfg.timeTillSaveCopy * fpsSet; // number of consecutive iterations car plate must be seen before making a copy of it for saving later
	counterToIdentifyUnrecognized = anprCfg.timeTillIdentifyUnrecognized * fpsSet; // number of consecutive iterations plate must not be found before deciding if unrecognized or not
	counterToFlipRedLight = anprCfg.stateChangeInterval*fpsSet; // number of iterations before red light changes state
	counterForStreamOn = cvRound(anprCfg.plateFoundTimeForStreamOn * (double)fpsSet); // number of consecutive frames that plate must be found in order to turn ON the live camera view
	counterForStreamOff = cvRound(anprCfg.plateMissTimeForStreamOff * (double)fpsSet); // number of consecutive frames that plate must be missed in order to turn OFF the live camera view

	if(anprCfg.discardInitialFrames)
	{
		for(int i=0;i<anprCfg.numOfDiscardFrames;i++) // if discarding of initial frames is activated, discard the specified number of frames
			anprCamera.grab();
	}

	if(anprCfg.controlIndicatorLights)
	{
		// turn OFF both lights - this is meant to indicate the end of the system initialization phase
		digitalWrite(anprCfg.PIN_GREEN,pin_LOW);
		digitalWrite(anprCfg.PIN_RED,pin_LOW);
	}

	if(anprCfg.displayFrames)
	{
		namedWindow("License Plate Detection");
		moveWindow("License Plate Detection",350,50);
	}

	logger_addTimeStamp(anprLogger, 2);
	anprLogger<< "All system setup steps completed. ELID-ANPR is now LIVE!" << endl;

	while(1) // The infinite program loop [START]
	{
	//===================KICK WATCHDOG ONCE EVERY 10 SECONDS FOR 80 MILLISECONDS [START]=====================

		if(wd_counter==anprCfg.wd_pulseEnd)
			wd_counter=0;
		if(wd_counter==0)
			digitalWrite(anprCfg.PIN_WDOG,pin_HIGH);
		if(wd_counter==anprCfg.wd_pulseStart)
			digitalWrite(anprCfg.PIN_WDOG,pin_LOW);
		++wd_counter;

	//===================KICK WATCHDOG ONCE EVERY 10 SECONDS FOR 80 MILLISECONDS [END]=======================

	//================CHECK THE LICENSE PLATE DATABASE HERE AND UPDATE IF REQUIRED [START]===================

		++databaseCheckCounter;
		if(databaseCheckCounter%anprCfg.databaseCheckInterval==0) // this checks for any change in the database at an interval of "databaseCheckInterval"
		{
			FILE *fileFinder; // File stream for opening/checking the file that is created to denote database modification
			fileFinder = fopen(anprCfg.databaseModifier.c_str() ,"r"); // databaseModifier is available(and therefore can be opened)whenever an entry is added or deleted.
			if(fileFinder!=NULL) // proceed to read the modified database, if the databaseModifier file could be successfully opened
			{
				fclose(fileFinder);
				databaseSize = dbMatcher->readDatabase(true); // Read the modified database. This is different from reading the database the first time, as specified by the boolean argument
				if( databaseSize == -1)
				{
					logger_addTimeStamp(anprLogger, 2);
					anprLogger << "ANPR database reading failed! Database file may be empty, non-existent or corrupted"<< endl;
				}
				else
				{
					logger_addTimeStamp(anprLogger, 2);
					anprLogger << "Modified ANPR database has been read successfully. Total entries read = " << databaseSize << endl;
				}

			}
		}

	//================CHECK THE LICENSE PLATE DATABASE HERE AND UPDATE IF REQUIRED [END]=====================

		if(anprCfg.displayFrames)
		{
			if(access_status == STAT_AUTHORIZED && !allow_recognition_update) // this displays the matched plate for a while after a database entry is succesfully matched, until enough frames have passed.
			{
				putText(lpResult, dbMatcher->matchedDbEntry.licensePlate.c_str(), cv::Point(500, 65), FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 3);
			}
			else
			{
				lpResult = Scalar(255,255,255);
				putText(lpResult, "License Plate : ", cv::Point(5, 65), FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2);
			}
		}

		++frameCounter; // increase the number of frames processed
		++delayPostMatching; // increase the number of frames passed after processing
		++delayPostPreviousMatch; // increase the number of frames passed since processing the same number plate
		++originalIntervalCount;
		++warpedIntervalCount;

		if(!allow_recognition_update && delayPostMatching > anprCfg.framesPassedAfterMatching) // After a short duration (if it's 100 frames at 25FPS, the duration is 4 seconds), allow recognition to resume
		{
			allow_recognition_update = true; // allow recognition again
			access_status = STAT_UNDETERMINED; // reset the access status
			delayPostMatching = 0; // reset the counter for post match counting
			if(anprCfg.controlIndicatorLights)
				digitalWrite(anprCfg.PIN_GREEN,pin_LOW); // turn off the green LED
		}

		if(barrierActivated && delayPostMatching > anprCfg.framesPassedAfterBarrierUp) // turn OFF the barrier relay after a short pulse
		{
				digitalWrite(anprCfg.PIN_BARRIER_RELAY,pin_LOW); // Turn off the relay
				barrierActivated = false; // disable the flag, so this block is not executed until the next barrier activation
		}

		double totalTimePerFrame = (double)getTickCount(); // this records the processing time per frame!

		anprCamera.grab();//grab next frame in the video stream
		if(frameCounter%frameInterval!=0) // skip the frames unless it is at the right interval
			continue;
		else
			anprCamera.retrieve(frame); // decode the frame


		if(!plateInferencePerformed) // if NO attempt was made to recognize plate number
		{
			++noRecognitionCounter; //increment the no recognition counter
			if(noRecognitionCounter>anprCfg.noRecognitionMaxCount) // if no-recognition counter is above the specified limit
			{
				// clear the list of recent attempts
				if(recentAttempts.size()>0)
					recentAttempts.clear();
			}
		}
		else // if recognition attempt was made, reinitialize the counter and the flag
		{
			noRecognitionCounter = 0;
			plateInferencePerformed = false;
		}

		if(frameCounter>0 && frameCounter%cameraStreamCheckInterval==0) // At the interval specified by cameraStreamCheckInterval, write a file that indicates the camera stream is working properly
		{
			FILE *frameFile;
			frameFile = fopen(anprCfg.frameDecoded.c_str(), "r");
			if(frameFile == NULL)
			{
				frameFile = fopen(anprCfg.frameDecoded.c_str(),"w");
				fclose(frameFile);
			}
			else
				fclose(frameFile);
		}

		if(allow_recognition_update) // [START] Proceed only if allow_recognition_update is enabled. It is disabled for a short period after a license plate is successfully recognized+matched
		{
			//============ CAR AND PLATE DETECTION [START] ====================

			carPlateDetTime = (double)getTickCount(); // this records the processing time for car & plate detection
			carPlusPlateDet->doInference(frame); // Execute inference to perform car and plate detection
			carPlusPlateDet->processResults(); // Process the inference outputs to obtain the correct car and plate locations
			carPlateDetTime = ((double)getTickCount()-carPlateDetTime)/getTickFrequency(); // find the difference in tick counts and divide by tick frequence to get the elapsed time in seconds

			//============ CAR AND PLATE DETECTION [END] ======================

			if(carPlusPlateDet->plate.classId != -1) // [START] Proceed further only if plate was found
			{
				// make the RED LED blink by turning it ON/OFF at specific interval
				if(anprCfg.controlIndicatorLights && plateFoundCounter%counterToFlipRedLight == 0)
				{
					if(redLightState)
					{
						digitalWrite(anprCfg.PIN_RED,pin_LOW); // turn off red LED
						redLightState = false;
					}
					else
					{
						digitalWrite(anprCfg.PIN_RED,pin_HIGH); // turn on red LED
						redLightState = true;
					}
				}
				// update the opposing counters
				noPlateFoundCounter = 0;
				++plateFoundCounter;

				if(plateFoundCounter > counterForStreamOn && !showingLiveCamView)
				{
					showingLiveCamView = true; // set to true to avoid this block unless this flag is false
					saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, stream_ON); // write display contents to disk that will turn on the live camera view
				}
				// Make a copy of the current frame if plate was found enough times, and a copy has not already been made
				if(plateFoundCounter>counterToSaveCopy && !haveFrameCopyToSave)
				{
					frame.copyTo(frameCopy);
					haveFrameCopyToSave = true;
				}

				//========================= CHARACTER SEGMENTATION [START] ========================
				lp = frame(carPlusPlateDet->plate.boundingBox).clone(); // Get a copy of the detected license plate region
				cvtColor(lp,lpGray,COLOR_BGR2GRAY); // Convert the license plate to grayscale

				charSegm->extractForeground(lpGray, lpBinary,false); // Extract the binary foreground from the grayscale image
				charSegm->findBlobs(lpBinary,false); // Extract blobs from the binary foreground
				if(anprCfg.displayDetailedLpInfo)
					charSegm->plotVarious(lp,true,false,false); // draw the BBs of the blobs on the BGR license plate
				charSegm->constructMinimumBoundingBox(lpBinary); // Determine the minimum bounding box that encapsulates all the extracted blobs

				if(charSegm->getPerspectiveSetting() == PERSPECTIVE_REQUIRED)
				{
					didPerspectiveTransform = true;
					charSegm->transformPerspective(lpGray, lpGrayWarped); // Execute perspective transformation
					if(anprCfg.displayDetailedLpInfo)
						charSegm->plotVarious(lp,false,true,false); // draw the minimum Rectangle encapsulating all blobs on the BGR license plate
					charSegm->extractForeground(lpGrayWarped, lpBinaryWarped, true); // Re-extract binary foreground from the transformed grayscale image
					charSegm->findBlobs(lpBinaryWarped,true); // Re-extract blobs from the transformed binary image

					if(anprCfg.displayDetailedLpInfo)
					{
						cvtColor(lpGrayWarped,lpGrayWarped,COLOR_GRAY2BGR); // convert the grayscale to 3-channel version for display purposes
						charSegm->plotVarious(lpGrayWarped,true,false,false); // draw the BBs of the segmented blobs on the grayscale image
						imshow("Warped",lpGrayWarped);
					}
				}
				else
				{
					didPerspectiveTransform = false;
					lpBinary.copyTo(lpBinaryWarped); // copy the license plate binary if perspective transformation is skipped
				}

				if(anprCfg.displayDetailedLpInfo)
					imshow("LP",lp);

				if(charSegm->segmentBlobs(lpBinaryWarped) == BLOBS_FOUND) // Perform standard and advanced character segmentation and proceed only if segmented blobs are found
				{
					charSegm->orderSegmentedBlobs(); // order the blobs according their position in the license plate, so that they can extracted in the correct order and fed to OCR
					charSegm->prepareOcrInput(tessBlock, lpBinaryWarped); // concatenate the segmented & ordered blobs
					if(anprCfg.displayOcrInput)
					{
						imshow("Ocr_Input",tessBlock);
						moveWindow("Ocr_Input",350,750);
					}
				}
				else
					continue;
				//========================= CHARACTER SEGMENTATION [END] ========================

				//=========================== OCR [START] =========================

				plateNumber.clear(); // Empty the contents of the plate number
				ocr->SetImage(tessBlock.data, tessBlock.cols,tessBlock.rows,1,tessBlock.step); // copy the row of characters to the ocr object
				ocr->SetSourceResolution(100);
				ocrTime= (double)getTickCount(); // this records the processing time for performing optical character recognition with tesseract
				plateNumber = string(ocr->GetUTF8Text()); // perform the OCR and store the result in plateNumber
				postProcessPlateNumber(plateNumber); // replace ambiguous characters with their likely alternatives. Eg, replace o(letter 0) with 0 or replace I with 1
				ocrTime = ((double)getTickCount()-ocrTime)/getTickFrequency(); // find the difference in tick counts and divide by tick frequence to get the elapsed time in seconds
				if(anprCfg.logRecognitionAttempts)
				{
					anprLogger<<endl;
					logger_addTimeStamp(anprLogger, 2);
					anprLogger<<"License Plate predicted by OCR: "<<plateNumber<<endl;
				}

				//=========================== OCR [END] =========================

				//================================== MAKE DETECTION [START] ================================

				makeDetTime = (double)getTickCount(); // this records the processing time for make detection
				makeDet->doInference(frame); // Execute inference to perform car and plate detection
				makeDet->processResults(); // Process the inference outputs to obtain the correct car and plate locations
				makeDetTime = ((double)getTickCount()-makeDetTime)/getTickFrequency(); // find the difference in tick counts and divide by tick frequence to get the elapsed time in seconds

				//================================== MAKE DETECTION [END] ================================

				//=================================== COLOR CLASSIFICAION [START] =================================

				if(carPlusPlateDet->car.classId!=-1) // Proceed to perform color classification only if a car was detected
				{

					car = frame(carPlusPlateDet->car.boundingBox).clone(); // get region of image corresponding to car presence

					colorClsTime = (double)getTickCount(); // this records the processing time for color classification
					colorCls->doInference(car); // perform car color classification
					if(anprCfg.debug_on_level_2_SEG)
					{
						for(unsigned int i=0; i<colorCls->outputClassConfidences.size(); i++)
							cout << colorCls->outputClassConfidences[i]<<endl;
					}

					colorCls->processResults(); // perform post processing
					colorClsTime = ((double)getTickCount()-colorClsTime)/getTickFrequency(); // find the difference in tick counts and divide by tick frequency to get the elapsed time in seconds
					if(anprCfg.debug_on_level_2_SEG)
					{
						cout << "Num of top scores: " << colorCls->topColors.size() << endl;
						for(unsigned int i = 0; i < colorCls->topColors.size(); i++)
							cout<<"Output # "<<i+1<<" : "<<colorCls->topColors[i].classLabel << "," << colorCls->topColors[i].confidenceScore << endl;
					}

				}

				//=================================== COLOR CLASSIFICAION [END] =================================

				//================================= DRAW ALL DETECTION/CLASSIFICATION OUTPUTS [START] =========================================

				if(anprCfg.displayFrames) // Display the frame with all detection and classification outputs, if displaying is allowed
				{
					if(carPlusPlateDet->plate.classId != -1) // If plate was found, draw its bounding box and print its confidence score
					{
						string displayStrPlate = "Plate : " + to_string(carPlusPlateDet->plate.predictionScore);
						cv::rectangle(frame, carPlusPlateDet->plate.boundingBox,Scalar(0,255,0), 3); //green BB
						cv::putText(frame, displayStrPlate, cv::Point( carPlusPlateDet->plate.boundingBox.x,  carPlusPlateDet->plate.boundingBox.y - 5),cv::FONT_HERSHEY_SIMPLEX, .6, cv::Scalar(0,255,0),2); // green text
					}


					if(carPlusPlateDet->car.classId != -1) // If car was found, draw its bounding box and print its confidence score
					{
						string displayStrCar= "Car : " + to_string(carPlusPlateDet->car.predictionScore);
						cv::rectangle(frame, carPlusPlateDet->car.boundingBox,Scalar(0,0,255), 3); //red BB
						cv::putText(frame, displayStrCar, cv::Point(carPlusPlateDet->car.boundingBox.x+5, carPlusPlateDet->car.boundingBox.y + carPlusPlateDet->car.boundingBox.height - 10),cv::FONT_HERSHEY_SIMPLEX, .9, cv::Scalar(0,0,255),2); //red text

						if(colorCls->carColor.classId != -1) // Print the color of the car and its confidence
						{
							string displayStrColor = colorCls->carColor.classLabel + " : " + to_string(colorCls->carColor.confidenceScore);
							cv::putText(frame, displayStrColor, cv::Point(carPlusPlateDet->car.boundingBox.x+5, carPlusPlateDet->car.boundingBox.y + carPlusPlateDet->car.boundingBox.height + 25),cv::FONT_HERSHEY_SIMPLEX, .9, cv::Scalar(0,255,255),2); //yellow text
						}
					}

					if(makeDet->carLogo.classId != -1) // If make was found, draw the bounding box and print its confidence score
					{
						string displayStrMake = makeDet->classLabels[makeDet->carLogo.classId] + " : " + to_string(makeDet->carLogo.predictionScore);
						cv::rectangle(frame, makeDet->carLogo.boundingBox,Scalar(255,255,0), 3); //cyan BB
						cv::putText(frame, displayStrMake, cv::Point( makeDet->carLogo.boundingBox.x,  makeDet->carLogo.boundingBox.y - 5),cv::FONT_HERSHEY_SIMPLEX, .6, cv::Scalar(255,255,0),2); //cyan text
					}

				}

				//================================= DRAW ALL DETECTION/CLASSIFICATION OUTPUTS [END] =========================================


				if(!plateNumber.empty()) // [START] If block to execute if the OCR output is not empty
				{

					//create file that indicates OCR was attempted
					ocrTried = fopen(anprCfg.ocrAttempted.c_str(), "r");
					if(ocrTried  == NULL)
					{
						ocrTried = fopen(anprCfg.ocrAttempted.c_str(),"w");
						fclose(ocrTried);
					}
					else
						fclose(ocrTried);
					//============================== SAVE TRAINING IMAGE IF RECORDING IS ENABLED [START] ==========================================

					if(anprCfg.recordTrainingImages && !recordTrainImsFinished) // Process the block only if recording is enabled and the required number of training images have not been reached
					{
						if(currentNumOfTrainingImages >= anprCfg.requiredNumOfTrainingImages)
						{
							recordTrainImsFinished = true; // specify that required number of training images have already been acquired

							//create file that indicates required number of training images have already been acquired
							trainDone = fopen(anprCfg.acquiredTrainingIms.c_str(), "r");
							if(trainDone  == NULL)
							{
								trainDone = fopen(anprCfg.acquiredTrainingIms.c_str(),"w");
								fclose(trainDone);
							}
							else
								fclose(trainDone);
						}
						else
						{
							if(frameCounter % samplingInterval == 0)
							{
								++currentNumOfTrainingImages; // increment the number of site-specific training images that have been acquired
								strcpy(nextTrainImagePath,anprCfg.trainingImsFolder.c_str()); // copy directory for saving the images to the file path
								strcat(nextTrainImagePath,"/");
								snprintf(nextSampCounter,10,"%05d_",currentNumOfTrainingImages); // convert the current counter for saved images to string
								strcat(nextTrainImagePath,nextSampCounter); // append the counter string as the first portion of the image file name
								strcat(nextTrainImagePath,plateNumber.c_str()); // append ocr output as the second portion of the image file name
								strcat(nextTrainImagePath,".tiff"); // append tiff extension as this is expected by tesseract training procedure
								imwrite(nextTrainImagePath,tessBlock); // save the tesseract input to the path
							}
						}
					}


				//============================== SAVE TRAINING IMAGE IF RECORDING IS ENABLED [END] ============================================

				//============================ SAVE ORIGINAL & WARPED PLATE SAMPLES [START] ====================================================
					if(anprCfg.recordPlateOriginal && currentNumTrainOriginal < anprCfg.requiredNumOfPlateOriginal && originalIntervalCount >= samplingInterval)
					{
						originalIntervalCount = 0;
						time_t now = time(0);
						tm *t = localtime(&now);
						++currentNumTrainOriginal;
						strcpy(nextTrainImagePath,anprCfg.plateSamplesFolder.c_str()); // copy directory for saving the images to the file path
						strcat(nextTrainImagePath,"/" );
						strcat(nextTrainImagePath,anprCfg.siteName.c_str());
						strcat(nextTrainImagePath,"_original_");
						strftime(plateTimeStamp,20,"%Y%m%d-%H%M%S_",t); // create time stamp
						strcat(nextTrainImagePath, plateTimeStamp);
						snprintf(nextSampCounter,10,"%05d_",currentNumTrainOriginal); // convert the current counter for saved images to string
						strcat(nextTrainImagePath,nextSampCounter); // append the counter string as the first portion of the image file name
						strcat(nextTrainImagePath,plateNumber.c_str()); // append ocr output as the second portion of the image file name
						strcat(nextTrainImagePath,".jpg"); // append tiff extension as this is expected by tesseract training procedure
						imwrite(nextTrainImagePath,lp); // save the tesseract input to the path
						saveCurrentSampleCount(sampleCountLogger, anprCfg.currentSampleCounts, currentNumTrainOriginal, "O");
					}
					if(anprCfg.recordPlateWarped && didPerspectiveTransform && currentNumTrainWarped < anprCfg.requiredNumOfPlateWarped && warpedIntervalCount >= samplingInterval)
					{
						warpedIntervalCount = 0;
						time_t now = time(0);
						tm *t = localtime(&now);
						++currentNumTrainWarped;
						strcpy(nextTrainImagePath,anprCfg.plateSamplesFolder.c_str()); // copy directory for saving the images to the file path
						strcat(nextTrainImagePath,"/" );
						strcat(nextTrainImagePath,anprCfg.siteName.c_str());
						strcat(nextTrainImagePath,"_warped_");
						strftime(plateTimeStamp,20,"%Y%m%d-%H%M%S_",t); // create time stamp
						strcat(nextTrainImagePath, plateTimeStamp);
						snprintf(nextSampCounter,10,"%05d_",currentNumTrainWarped); // convert the current counter for saved images to string
						strcat(nextTrainImagePath,nextSampCounter); // append the counter string as the first portion of the image file name
						strcat(nextTrainImagePath,plateNumber.c_str()); // append ocr output as the second portion of the image file name
						strcat(nextTrainImagePath,".jpg"); // append tiff extension as this is expected by tesseract training procedure
						imwrite(nextTrainImagePath,lpGrayWarped); // save the tesseract input to the path
						saveCurrentSampleCount(sampleCountLogger, anprCfg.currentSampleCounts, currentNumTrainWarped, "W");
					}


				//============================ SAVE ORIGINAL & WARPED PLATE SAMPLES [END] ======================================================

					if(anprCfg.engageSecondaryDatabase)
						recentAttempts.push_back(plateNumber); // save the license plate attempt
					plateInferencePerformed = true; // flag that indicates that "attempt" has been made to perform license plate recognition

				//============================= PERFORM MATCHING OF ALL THE EXTRACTED CAR INFORMATION [START] =========================================

					if(dbMatcher->compareLpToDb(plateNumber, anprLogger) == DB_MATCH_FOUND) // If the platenumber obtained from the OCR is matched with the license plate of any db entry
					{
						access_status = STAT_AUTHORIZED; // Set access to authorized first. Then process the remaining car information and update the access_status accordingly
						matchCounts.countDbNotMatched = 0; // Reset the counter for no database matches back to zero, since a match was successfully found
						if(anprCfg.engageSecondaryDatabase && recentAttempts.size()>=(unsigned int)anprCfg.numOfAttempts) // if secondary database is allowed and a pre-determined number of attempts are made, save these attempts
						{
							ofstream writeAttempts;
							writeAttempts.open(anprCfg.recognitionAttempts); // open file for saving the recognition attempts
							for(unsigned int i=0;i<recentAttempts.size()-1;i++)
								writeAttempts << recentAttempts[i] << endl; // save all the attempts made by the ocr to recognize the license plate (one attempt in each row)
							writeAttempts << dbMatcher->matchedDbEntry.licensePlate << " "<<dbMatcher->matchedDbEntry.cardNumber<<endl; // in the last row, save the actual license plate that was matched from the database, along with the corresponding card number
							writeAttempts.close();
						}
					}
					else // If the platenumber obtained from OCR failed to match with any of the license plates in the database
					{
						if(++matchCounts.countDbNotMatched >= anprCfg.numForDbMismatch) // Set access status to blocked if no database matches occurred a prespecified number of times consecutively
							access_status = STAT_BLOCKED_NoDbMatch;
						else // If the prespeficied number of database mismatches has not been reached, set the access status to undefined
							access_status = STAT_UNDETERMINED;
					}

					// At this point, STAT_AUTHORIZED means
					// 1) A db entry was found whose license plate matched the ocr output
					// Immediate set access status to blacklisted if it is defined as such in the mathced db entry
					if(access_status == STAT_AUTHORIZED && dbMatcher->matchedDbEntry.accessStatus == "Blacklisted")
						access_status = STAT_BLOCKED_Blacklisted;// Check if access_status is set to blocked due to make or color mismatch.

					// At this point, STAT_AUTHORIZED means
					// 1) A db entry was found whose license plate matched the ocr output
					// 2) The db entry is "Whitelisted", not "Blacklisted"
					// This subsequent block will check for the make and color simultaneously
					if(access_status == STAT_AUTHORIZED)
					{
						if(dbMatcher->matchedDbEntry.mustMatchMake) // Check if the matched db entry requires the make to be matched
						{
							//cout << "Entered make block...";
							if(dbMatcher->matchedDbEntry.carMake != makeDet->classLabels[makeDet->carLogo.classId]) // if the detected make did not match the make of the db entry
							{
								if(dbMatcher->matchedDbEntry.licensePlate != plateFromLastDbMatch) // if the current license plate is NOT the same as the previous license plate
									matchCounts.countMakeNotMatched = 1; // start the count for make mismatch
								else // if current license plate is the same as the previous one
									++matchCounts.countMakeNotMatched; // increment the count for make mismatch
								if(matchCounts.countMakeNotMatched >= anprCfg.numForMakeMismatch) // if the count for mismatch exceeds the prespecified total
									access_status = STAT_BLOCKED_MakeMismatch; // set access status as blocked due to make mismatch
							}
							else// if the detected make matched the make of the db entry
							{
								matchCounts.countMakeNotMatched = 0; // the counter for make not matched should be set to 0 anytime a successful make match is done
								if(dbMatcher->matchedDbEntry.licensePlate != plateFromLastDbMatch) // if the current license plate is NOT the same as the previous license plate
									matchCounts.countMakeMatched = 1; // start the count for make matched
								else // if current license plate is the same as the previous one
									++matchCounts.countMakeMatched; // increment the count for make matched
							}
							//cout << "Exiting make block"<<endl;
						}

						if(dbMatcher->matchedDbEntry.mustMatchColor) // Check if the matched db entry requires the color to be matched
						{
							//cout << "Entered color block...";
							if(dbMatcher->matchedDbEntry.carColor != colorCls->carColor.classLabel) // if the detected color did not match the color of the db entry
							{
								if(dbMatcher->matchedDbEntry.licensePlate != plateFromLastDbMatch) // if the current license plate is NOT the same as the previous license plate
									matchCounts.countColorNotMatched = 1; // start the count for color mismatch
								else // if current license plate is the same as the previous one
									++matchCounts.countColorNotMatched; // increment the count for color mismatch
								if(matchCounts.countColorNotMatched >= anprCfg.numForColorMismatch) // if the count for mismatch exceeds the prespecified total
									access_status = STAT_BLOCKED_ColorMismatch; // set access status as blocked due to color mismatch
							}
							else // if the detected color matched the make of the db entry
							{
								matchCounts.countColorNotMatched = 0; // the counter for color not matched should be set to 0 anytime a successful color match is done
								if(dbMatcher->matchedDbEntry.licensePlate != plateFromLastDbMatch) // if the current license plate is NOT the same as the previous license plate
									matchCounts.countColorMatched = 1; // start the count for color matched
								else // if current license plate is the same as the previous one
									++matchCounts.countColorMatched; // increment the count for color matched
							}
							//cout << "Exiting color block"<<endl;
						}

						// Check if access_status is set to blocked due to make or color mismatch.
						if(access_status!=STAT_BLOCKED_MakeMismatch && access_status!=STAT_BLOCKED_ColorMismatch)
						{
							if(dbMatcher->matchedDbEntry.mustMatchMake && !dbMatcher->matchedDbEntry.mustMatchColor) // Only make must be matched
							{
								if(matchCounts.countMakeMatched < anprCfg.numForMakeMatch) // if the count for make matches is below the total required number
									access_status = STAT_UNDETERMINED; // change the access status to undetermined.
							}
							else if (!dbMatcher->matchedDbEntry.mustMatchMake && dbMatcher->matchedDbEntry.mustMatchColor) // Only color must be matched
							{
								if(matchCounts.countColorMatched < anprCfg.numForColorMatch) // if the count for color matches is below the total required number
									access_status = STAT_UNDETERMINED; // change the access status to undetermined.
							}
							else if (dbMatcher->matchedDbEntry.mustMatchMake && dbMatcher->matchedDbEntry.mustMatchColor) // Both make and color must be matched
							{
								if(matchCounts.countMakeMatched < anprCfg.numForMakeMatch || matchCounts.countColorMatched < anprCfg.numForColorMatch) // if the count for make matches is below the total required number
									access_status = STAT_UNDETERMINED; // change the access status to undetermined.
							}
						}
					}

				//============================= PERFORM MATCHING OF ALL THE EXTRACTED CAR INFORMATION [END] ===========================================
					plateFromLastDbMatch.assign(dbMatcher->matchedDbEntry.licensePlate);
//					matchCounts.printCounts(); // print values of all matching counters
				} // [END] If block to execute if the OCR output is not empty

				totalTimePerFrame = ((double)getTickCount()-totalTimePerFrame )/getTickFrequency(); // find the difference in tick counts and divide by tick frequence to get the elapsed time in seconds
				if(anprCfg.logRecognitionAttempts)
				{
					logger_addTimeStamp(anprLogger, 2);
					anprLogger << "Frame #" << frameCounter <<". Run time frame rate = " << 1/totalTimePerFrame  << " FPS." << endl;
				}
			}  // [END] Proceed further only if plate was found
			else
			{
				plateFoundCounter = 0; // reset the counter of plate found
				++noPlateFoundCounter; // increase the counter for plate NOT found
			}

			//======================================== PROCESS THE ACCESS STATUS [START] ==============================================================
			if(access_status != STAT_UNDETERMINED) // No point of any processing if the status is UNDETERMINED
			{
				//cout<<"Current access status : "<< access_status<<endl;
				switch(access_status)
				{

					case STAT_AUTHORIZED:
						// Either
						// A) the current matched plate number is different from the previous matched plate number,  OR
						// B) the current matched plate number is same as the previous matched plate number, but the prespecified amount of time has passed
						if(dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)!=0 || (dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)==0 && delayPostPreviousMatch > anprCfg.framesPassedAfterPreviousMatch))
						{

							dbMatcher->saveImageHavingLp(frame, STAT_AUTHORIZED, anprLogger); // Save the image as a valid, authorized transaction
							displayContents = prefix_authorized + dbMatcher->matchedDbEntry.userName + "|" + dbMatcher->matchedDbEntry.licensePlate; // prepare the contents to be written to disk for sending to display module
							saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, displayContents); // write display contents to disk
							displayContents = ""; // reset the displayContents string
							haveFrameCopyToSave = false; // set to false since access status was determined. Later, frame copy should not be saved as "UNIDENTIFIED__UndeterminedStatus""

							//========================SEND WIEGAND SIGNAL OR TURN ON BARRIER RELAY [START] ==============================
							if(anprCfg.linkToAccessControl)
							{

								if(anprCfg.useSysCallForWiegand)
								{
									snprintf(sendwiegandcommand,100,wiegcommandstring,dbMatcher->matchedDbEntry.cardNumber); // print the wiegand number into the command string
									system(sendwiegandcommand); // send wiegand via system call
								}

								else
								{
									splitCardNumToChars(dbMatcher->matchedDbEntry.cardNumber,wiegBits); // convert the card number to bits
									calculateParity(wiegBits); // parity calculation
									sendWiegandSignal(wiegBits); // send out the wiegand signal to the controller
								}
							}
							else
							{
								digitalWrite(anprCfg.PIN_BARRIER_RELAY, pin_HIGH); // Turn on the barrier relay
								barrierActivated = true; // Turn on the flag that indicates the relay has been set high
							}

							if(anprCfg.logRecognitionAttempts)
							{
								logger_addTimeStamp(anprLogger, 2);
								anprLogger<< ">>>>>>>>>>>>>>>   ACCESS STATUS : AUTHORIZED. ||=========== GOING UP!"<<endl;
							}
							//========================SEND WIEGAND SIGNAL OR TURN ON BARRIER RELAY [END] ================================

							previousAuthorizedPlate.assign(dbMatcher->matchedDbEntry.licensePlate);// copy the matched plate to the current plate - this is necessary to evaluate if the next recognized license plate is the same as the current one
							delayPostMatching = 0; // reset the post match counter to 0
							delayPostPreviousMatch = 0; // reset the post previous match counter to 0
							showingLiveCamView = false; // reset the state fo the live camera view
							plateFoundCounter = 0; noPlateFoundCounter = 0; // reset both plate found and not found counters
							allow_recognition_update = false; //set to false so that the stream is not processed for the prespecified period of time after a successful authorized entry
							matchCounts.resetCounts(); // reset all the matching counters to 0
							access_status = STAT_UNDETERMINED; // Set status to undetermined once processing routine is done
						}
						else
						{
							access_status = STAT_UNDETERMINED; // set status to undetermined, since if conditions were not met. If not, the if block will be wrongly executed later!
							matchCounts.resetCounts(); // reset all the matching counters to 0
						}
						break;

					case STAT_BLOCKED_Blacklisted:
						if(dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)!=0 || (dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)==0 && delayPostPreviousMatch > anprCfg.framesPassedAfterPreviousMatch))
						{

							dbMatcher->saveImageHavingLp(frame, STAT_BLOCKED_Blacklisted, anprLogger); // Save the image as blocked due to blacklisted
							displayContents = prefix_blocked_blacklist + dbMatcher->matchedDbEntry.userName + "|" + dbMatcher->matchedDbEntry.licensePlate; // prepare the contents to be written to disk for sending to display module
							saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, displayContents); // write display contents to disk
							displayContents = ""; // reset the displayContents string
							haveFrameCopyToSave = false; // set to false since access status was determined. Later, frame copy should not be saved as "UNIDENTIFIED__UndeterminedStatus""

							if(anprCfg.logRecognitionAttempts)
							{
								logger_addTimeStamp(anprLogger, 2);
								anprLogger<< ">>>>>>>>>>>>>>>   ACCESS STATUS : BLOCKED. REASON = BLACKLISTED"<<endl;
							}

							previousAuthorizedPlate.assign(dbMatcher->matchedDbEntry.licensePlate);// copy the matched plate to the current plate - this is necessary to evaluate if the next recognized license plate is the same as the current one
							delayPostMatching = 0; // reset the post match counter to 0
							delayPostPreviousMatch = 0; // reset the post previous match counter to 0
							showingLiveCamView = false; // reset the state fo the live camera view
							plateFoundCounter = 0; noPlateFoundCounter = 0; // reset both plate found and not found counters
							allow_recognition_update = false; //set to false so that the stream is not processed for the prespecified period of time after a successful authorized entry
							matchCounts.resetCounts(); // reset all the matching counters to 0
							access_status = STAT_UNDETERMINED; // Set status to undetermined once processing routine is done
						}
						else
						{
							access_status = STAT_UNDETERMINED; // set status to undetermined, since if conditions were not met. If not, the if block will be wrongly executed later!
							matchCounts.resetCounts(); // reset all the matching counters to 0
						}
						break;

					case STAT_BLOCKED_MakeMismatch:
						if(dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)!=0 || (dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)==0 && delayPostPreviousMatch > anprCfg.framesPassedAfterPreviousMatch))
						{

							dbMatcher->saveImageHavingLp(frame, STAT_BLOCKED_MakeMismatch, anprLogger); // Save the image as blocked due to make mismatch
							displayContents = prefix_blocked_make + dbMatcher->matchedDbEntry.userName + "|" + dbMatcher->matchedDbEntry.licensePlate; // prepare the contents to be written to disk for sending to display module
							saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, displayContents); // write display contents to disk
							displayContents = ""; // reset the displayContents string
							haveFrameCopyToSave = false; // set to false since access status was determined. Later, frame copy should not be saved as "UNIDENTIFIED__UndeterminedStatus""

							if(anprCfg.logRecognitionAttempts)
							{
								logger_addTimeStamp(anprLogger, 2);
								anprLogger<< ">>>>>>>>>>>>>>>   ACCESS STATUS : BLOCKED. REASON = MAKE MISMATCH"<<endl;
							}

							previousAuthorizedPlate.assign(dbMatcher->matchedDbEntry.licensePlate);// copy the matched plate to the current plate - this is necessary to evaluate if the next recognized license plate is the same as the current one
							delayPostMatching = 0; // reset the post match counter to 0
							delayPostPreviousMatch = 0; // reset the post previous match counter to 0
							showingLiveCamView = false; // reset the state fo the live camera view
							plateFoundCounter = 0; noPlateFoundCounter = 0; // reset both plate found and not found counters
							allow_recognition_update = false; //set to false so that the stream is not processed for the prespecified period of time after a successful authorized entry
							matchCounts.resetCounts(); // reset all the matching counters to 0
							access_status = STAT_UNDETERMINED; // Set status to undetermined once processing routine is done
						}
						else
						{
							access_status = STAT_UNDETERMINED; // set status to undetermined, since if conditions were not met. If not, the if block will be wrongly executed later!
							matchCounts.resetCounts(); // reset all the matching counters to 0
						}
						break;

					case STAT_BLOCKED_ColorMismatch:
						if(dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)!=0 || (dbMatcher->matchedDbEntry.licensePlate.compare(previousAuthorizedPlate)==0 && delayPostPreviousMatch > anprCfg.framesPassedAfterPreviousMatch))
						{

							dbMatcher->saveImageHavingLp(frame, STAT_BLOCKED_ColorMismatch, anprLogger); // Save the image as blocked due to color mismatch
							displayContents = prefix_blocked_color + dbMatcher->matchedDbEntry.userName + "|" + dbMatcher->matchedDbEntry.licensePlate; // prepare the contents to be written to disk for sending to display module
							saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, displayContents); // write display contents to disk
							displayContents = ""; // reset the displayContents string
							haveFrameCopyToSave = false; // set to false since access status was determined. Later, frame copy should not be saved as "UNIDENTIFIED__UndeterminedStatus""

							if(anprCfg.logRecognitionAttempts)
							{
								logger_addTimeStamp(anprLogger, 2);
								anprLogger<< ">>>>>>>>>>>>>>>   ACCESS STATUS : BLOCKED. REASON = COLOR MISMATCH"<<endl;
							}

							previousAuthorizedPlate.assign(dbMatcher->matchedDbEntry.licensePlate);// copy the matched plate to the current plate - this is necessary to evaluate if the next recognized license plate is the same as the current one
							delayPostMatching = 0; // reset the post match counter to 0
							delayPostPreviousMatch = 0; // reset the post previous match counter to 0
							showingLiveCamView = false; // reset the state fo the live camera view
							plateFoundCounter = 0; noPlateFoundCounter = 0; // reset both plate found and not found counters
							allow_recognition_update = false; //set to false so that the stream is not processed for the prespecified period of time after a successful authorized entry
							matchCounts.resetCounts(); // reset all the matching counters to 0
							access_status = STAT_UNDETERMINED; // Set status to undetermined once processing routine is done
						}
						else
						{
							access_status = STAT_UNDETERMINED; // set status to undetermined, since if conditions were not met. If not, the if block will be wrongly executed later!
							matchCounts.resetCounts(); // reset all the matching counters to 0
						}
						break;

					case STAT_BLOCKED_NoDbMatch:
						if(plateNumber.compare(previousAuthorizedPlate)!=0 || (plateNumber.compare(previousAuthorizedPlate)==0 && delayPostPreviousMatch > anprCfg.framesPassedAfterPreviousMatch))
						{

							dbMatcher->saveImageHavingLp(frame, STAT_BLOCKED_NoDbMatch, anprLogger); // Save the image as blocked due to no database match
							displayContents = prefix_blocked_unrecognized; // prepare the contents to be written to disk for sending to display module
							saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, displayContents); // write display contents to disk
							displayContents = ""; // reset the displayContents string
							haveFrameCopyToSave = false; // set to false since access status was determined. Later, frame copy should not be saved as "UNIDENTIFIED__UndeterminedStatus""

							previousAuthorizedPlate.assign(plateNumber);// copy the recognized plate to the current plate - this is necessary to evaluate if the next recognized license plate is the same as the current one
							delayPostMatching = 0; // reset the post match counter to 0
							delayPostPreviousMatch = 0; // reset the post previous match counter to 0
							showingLiveCamView = false; // reset the state fo the live camera view
							plateFoundCounter = 0; noPlateFoundCounter = 0; // reset both plate found and not found counters
							allow_recognition_update = false; //set to false so that the stream is not processed for the prespecified period of time after a successful authorized entry
							matchCounts.resetCounts(); // reset all the matching counters to 0
							access_status = STAT_UNDETERMINED; // Set status to undetermined once processing routine is done
						}
						else
						{
							access_status = STAT_UNDETERMINED; // set status to undetermined, since if conditions were not met. If not, the if block will be wrongly executed later!
							matchCounts.resetCounts(); // reset all the matching counters to 0
						}
						break;

					default:
						// No default behavior exists
						break;
				}
			}
			//======================================== PROCESS THE ACCESS STATUS [END] ================================================================

		} // [END] Proceed only if allow_recognition_update is enabled. It is disabled for a short period after a license plate is successfully recognized+matched


		if(noPlateFoundCounter > counterForStreamOff && showingLiveCamView)
		{
			showingLiveCamView = false; // set to false to avoid this block unless this flag is false
			saveDisplayContent(displayContentsStream, anprCfg.sendToDisplayPath, stream_OFF); // write display contents to disk that will turn off the live camera view
		}

		//=========================================== SAVE IMAGES OF INSTANCES WHERE ACCESS STATUS COULD NOT BE DETERMINED [START] =====================================
		if(anprCfg.recordAfterCarPasses && noPlateFoundCounter > counterToIdentifyUnrecognized && haveFrameCopyToSave)
		{
			haveFrameCopyToSave = false; // reset the flag
			time_t now = time(0);
			tm *t = localtime(&now);

			snprintf(plateRecToday,200,anprCfg.imageSaveFolder.c_str(),t->tm_year+1900,t->tm_mon+1,t->tm_mday);  // create the name of today's directory based on today's date
			if( stat( plateRecToday, &info ) != 0 ) // if the folder name pointed to by plateRecToday does not exist, execute the following block
			{
				const int dir_err = mkdir(plateRecToday, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); // create the directory
				if (-1 == dir_err)
					exit(1);

				// Destroy the current logger stream and reinitialize for the next day
				logger_destroy(anprLogger);
				logger_initialize(anprLogger, anprLogFile);
			}

			//create the full path name that the image will be saved to, just like before. Except, append "UNIDENTIFIED" at the end
			strftime(plateTimeStamp,15,"%H-%M-%S__",t);
			strcpy(nextPlatePath,plateRecToday);
			strcat(nextPlatePath,"/");
			strcat(nextPlatePath,plateTimeStamp);
			strcat(nextPlatePath,"UNIDENTIFIED__UndeterminedStatus");
			strcat(nextPlatePath,".jpg");
			//write the image to that path
			imwrite(nextPlatePath,frameCopy);
		}
		//=========================================== SAVE IMAGES OF INSTANCES WHERE ACCESS STATUS COULD NOT BE DETERMINED [END] =====================================

		if(anprCfg.controlIndicatorLights && redLightState==true && noPlateFoundCounter>counterToFlipRedLight) // if red light is on, and plate is not found more than specified duration, turn it OFF
		{
				digitalWrite(anprCfg.PIN_RED,pin_LOW);
				redLightState = false;
		}

		if(anprCfg.displayFrames) // Display the frame with all the detection/classification outputs
		{
			imshow("License Plate Detection", frame);
			waitKey(anprCfg.displayPause);
		}

//=========================== RESET ALL COUNTERS IF THEY REACH 1 MILLION [START] ================================

		if(frameCounter>999999)
			frameCounter=0;
		if(delayPostMatching>999999)
			delayPostMatching=0;
		if(delayPostPreviousMatch>999999)
			delayPostPreviousMatch=0;
		if(databaseCheckCounter>999999)
			databaseCheckCounter=0;
		if(noPlateFoundCounter>999999)
			noPlateFoundCounter=0;
		if(noRecognitionCounter>999999)
			noRecognitionCounter = anprCfg.noRecognitionMaxCount+1;
		if(originalIntervalCount > 999999)
			originalIntervalCount = 0;
		if(warpedIntervalCount > 999999)
			warpedIntervalCount = 0;

//=========================== RESET ALL COUNTERS IF THEY REACH 1 MILLION [END] ================================

	} // The infinite program loop [END]
//======================= CODE TO RUN IF WHILE LOOP EXITS! [START] ====================================

	logger_destroy(anprLogger); // destroy stream, if ever infinite loop exits!
	ocr->End();

//======================= CODE TO RUN IF WHILE LOOP EXITS! [END] =======================================
    return 0;
}
