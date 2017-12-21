//Module: ViewController.mm
//Name: Robotic Concepts-Final Project
//Purpose: Facial Recognition and appropriate responses.
//Programmers: Juleyn Arnold, Samantha Glass, James Spahlinger, Kevin Coulman

#import "ViewController.h"

#ifdef DEBUG_3P
#define LOG_3P(str) NSLog(@"%s", str)
#else
#define LOG_3P(str) /* nothing */
#endif

#ifdef __cplusplus
#import <opencv2/opencv.hpp>
#import "opencv2/objdetect/objdetect.hpp"
#import "opencv2/imgproc/imgproc.hpp"
#import "opencv2/highgui/ios.h"
#import "opencv2/highgui/cap_ios.h"
#endif

#import "SMKDetectionCamera.h"
#import <GPUImage/GPUImage.h>
#import <DoubleControlSDK/DoubleControlSDK.h>
#import <AudioToolbox/AudioToolbox.h>
#import <OpenEars/OELanguageModelGenerator.h>
#import <OpenEars/OEAcousticModel.h>
#import <OpenEars/OEPocketsphinxController.h>
#import <opencv2/opencv.hpp>

#define NUMBER_OF_FACES 20
@interface ViewController() <DRDoubleDelegate> {
    GPUImageView* view;
    UIDevice *iOSDevice;    // Setup the view (this time using GPUImageView)
    SMKDetectionCamera* detector;
    GPUImageFilter* filter;
    //GPUImageView *cameraView_; //view on the screen
    //GPUImageVideoCamera *viewCapturer; //captures the view on the screen
    UIImage *capture; //a save for that capture
    cv::Mat captureMatrix;
    
    UIView *faceFeatureTrackingView_; // View for showing bounding box around the face
    CGAffineTransform cameraOutputToPreviewFrameTransform_;
    CGAffineTransform portraitRotationTransform_;
    CGAffineTransform texelToPixelTransform_;
    
    NSURL *audioFilePathURL;
    SystemSoundID soundSystemServicesId;
    NSTimer *tickTimer;
    int counter;
    
    CGFloat xOriginFace_;
    CGFloat xOriginCamera_;
    
    CGFloat heightFace;
    CGFloat widthFace;
    CGFloat heightCam;
    CGFloat widthCam;
    
    UILabel *tempLabel;
    UILabel *tmrLabel;
    cv::Mat FaceMatrix;
    
    //Juleyn Arnold added things
    cv::CascadeClassifier face_cascade;
    NSTimer *tmr;
    int tmrTicks;
    int lastTick;
    
    //Training face information
    UIImage *trainingFaces[NUMBER_OF_FACES];
    std::vector<UIImage*> trainingFacesVector;
    std::vector<cv::Mat> trainMat;
    std::vector<int> labels;
    NSUInteger numFaces; //counter to keep track of the number of training faces gathered so far
    
    //Face Recognition
    Boolean isRecognitionSetup; //Is face recognition setup or not, after recognition is setup is the face to be recognized in the screen
    NSUInteger facesFound;
    cv::Ptr<cv::FaceRecognizer> recognizer; //OpenCV facerecognizer utilizes eigenfaces algorthim
    bool goToFace, lookForFace;
    double amountTurned;
    NSTimer *calibrateDelay;
    bool calFinished;
    NSMethodSignature *sgn;
    NSInvocation *inv;
    
    UIImageView* newView;
    UIImageView* newView2;
    UIImage* test;
}
//@property (nonatomic, strong) SMKDetectionCamera *detector;
@end


@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    //Connect the double robot(Delegates this viewController to the
    [DRDouble sharedDouble].delegate = self;
    tmrTicks = 0;
    lastTick = 0;
    goToFace = false;
    lookForFace = false;
    calFinished = false;
    amountTurned = 0;
    self.view.frame = CGRectMake(0, 0, 640, 480);
    
    //Creates a camera(GPUImageView) to later be manipulated by the smerk detector from (dimensions are the size of the viewController in this case size of device)
    //Creates the detector sets it up for facial recognition
    detector = [[SMKDetectionCamera alloc] initWithSessionPreset:AVCaptureSessionPreset640x480 cameraPosition:AVCaptureDevicePositionFront];
    [detector setOutputImageOrientation:UIInterfaceOrientationPortrait]; // Set to portrait upside down
    
    filter = [[GPUImageFilter alloc] init];
    [detector addTarget:filter];
    
    view = [[GPUImageView alloc] initWithFrame:CGRectMake(0, 0, self.view.frame.size.height, self.view.frame.size.width)];
    NSLog(@"View Size w%f", self.view.frame.size.width); //View Size 640, 480
    NSLog(@"View Size h%f", self.view.frame.size.height);
    view.fillMode = kGPUImageFillModePreserveAspectRatioAndFill;
    
    [filter addTarget:view];
    
    
    
    //Adds the cameraView(GPUImageView) to this blank viewController
    //Calls custom methods that setup the face box view
    [self.view addSubview:view];
    [self setupFaceTrackingViews];
    [self calculateTransformations];
    
    //Setup camera that captures faces to be recognized and processed
    [filter useNextFrameForImageCapture];
    [detector useNextFrameForImageCapture];
    [self.view setBackgroundColor:[UIColor whiteColor]];
    //On voice command start face detection
    //Will be surrounded by a case or if statement that parse results of the voice command
    //Default Smerk: Setup for running face detector
    [detector beginDetecting:kFaceFeatures | kMachineAndFaceMetaData
                    codeTypes:@[AVMetadataObjectTypeQRCode]
           withDetectionBlock:^(SMKDetectionOptions detectionType, NSArray *detectedObjects, CGRect clapOrRectZero) {
               //Probably want something here like if face detection is started then do updateFaceFeatures
               //also if isFaceRecognized do another method that is a modified version of updateFaceFeatureTrackingViewWithObjects
               // Check if the kFaceFeatures have been discovered
               if (detectionType & kFaceFeatures) {
                   [self updateFaceFeatureTrackingViewWithObjects:detectedObjects];
               }
           }];
    //On voice command stop all detection
    //[detector_ stopAllDetection];
    
    // Finally start the camera]
    sgn = [self methodSignatureForSelector:@selector(calibrateTime:)];
    inv = [NSInvocation invocationWithMethodSignature: sgn];
    tmr = [NSTimer scheduledTimerWithTimeInterval:0.5 target:self selector:@selector(UpdateTimer:) userInfo:nil repeats:YES];//will change back time later
    calibrateDelay = [NSTimer timerWithTimeInterval:5 invocation:inv repeats:NO];

    [detector startCameraCapture];

    tempLabel = [[UILabel alloc] initWithFrame:CGRectMake(100, 100, 120, 30)]; //x, y, height, width
    [tempLabel setBackgroundColor:[UIColor clearColor]];
    [tempLabel setFont:[UIFont fontWithName:@"Courier" size: 18.0]];
    [tempLabel setTextColor:[UIColor redColor]];
    [tempLabel setHidden:YES];
    
    UIButton *overlayButton = [UIButton buttonWithType:UIButtonTypeCustom];
    [overlayButton setImage:[UIImage imageNamed:@"white_park.png"] forState:UIControlStateNormal];
    [overlayButton setFrame:CGRectMake(30, 30, 100, 100)];
    [overlayButton addTarget:self action:@selector(parkButtonPressed) forControlEvents:UIControlEventTouchUpInside];
    [[self view] addSubview:overlayButton];
    
    newView2 = [[UIImageView alloc] initWithImage:NULL];
    [self.view addSubview:newView2];
    
    newView = [[UIImageView alloc] initWithImage:NULL];
    [self.view addSubview:newView];
    
    
    counter = 0;
}
- (void) calibrateTime: (NSTimer*) t {
    calFinished = true;
}
- (void) UpdateTimer: (NSTimer *) t{
    tmrTicks++;
}

// Set up the view for facetracking
- (void)setupFaceTrackingViews {
    faceFeatureTrackingView_ = [[UIView  alloc] initWithFrame:CGRectZero];
    faceFeatureTrackingView_.layer.borderColor = [[UIColor redColor] CGColor];
    faceFeatureTrackingView_.layer.borderWidth = 3;
    faceFeatureTrackingView_.backgroundColor = [UIColor clearColor];
    faceFeatureTrackingView_.hidden = YES;
    faceFeatureTrackingView_.userInteractionEnabled = NO;
    
    [self.view addSubview:faceFeatureTrackingView_]; // Add as a sub-view
}

// Update the face feature tracking view
- (void)updateFaceFeatureTrackingViewWithObjects:(NSArray *)objects {
    
    if (!objects.count) {//if there are no objects(if count == 0 keep green face tracker hidden)
        faceFeatureTrackingView_.hidden = YES;
    }else {
        //Store the current screens image
        CIFaceFeature * feature = objects[0];
        CGRect face;
        
        //capture the image
        [filter useNextFrameForImageCapture];
        GPUImageFramebuffer *fb = filter.framebufferForOutput;//So far no overrelease problems
        [fb disableReferenceCounting];
        capture = filter.imageFromCurrentFramebuffer;
        
        //transform the image (crop to phone size)
        capture = [[UIImage alloc] initWithCGImage: capture.CGImage scale: 1.0 orientation: UIImageOrientationDown];
        
        if (capture == nil) return;
        //Try and gather training faces
        if (lastTick != tmrTicks) { //gives the person to be detected 5 seconds to gather training faces
            lastTick = tmrTicks;
            if (numFaces == NUMBER_OF_FACES) [tmr invalidate]; // stop the timer after time has elapsed or all faces have been gathered
            CGRect r = feature.bounds;
            r = CGRectApplyAffineTransform(r, portraitRotationTransform_);
            NSLog(@"Capturing images");
            [self gatherTrainingInformation:r];
        }
        
        //If training faces are gather(Recognition Setup) try to recognize
        int wantedFace = -1;
        bool isFaceRecognized = false;
        for (int i = 0; i < objects.count; i++) {
            CIFaceFeature* temp = objects[i];
            CGRect r2 = temp.bounds;
            r2 = CGRectApplyAffineTransform(r2, portraitRotationTransform_);
            if ([self trytoRecognizeFace:r2] == true) {
                wantedFace = i;
                isFaceRecognized = true;
                break;
            }
            
        }
        if (isFaceRecognized == false) {
            lookForFace = true;
            goToFace = false;
            if (isRecognitionSetup == true) NSLog(@"This is not the face");
            faceFeatureTrackingView_.hidden = YES;
            return;
        } else {
            if (isRecognitionSetup == true) NSLog(@"This is definitely the face");
        }
        lookForFace = false;
        goToFace = true;
    
        feature = objects[wantedFace];
        face = feature.bounds;
        
        //portrait transform to view
        face = CGRectApplyAffineTransform(face, portraitRotationTransform_);
        faceFeatureTrackingView_.frame = face;
        faceFeatureTrackingView_.hidden = NO;
    
        //set origins for tracking face in comparison to center of camera
        xOriginFace_ = faceFeatureTrackingView_.center.x;
        xOriginCamera_ = view.center.x;
    
        //face roughly 210 == 3 feet, 70
        heightFace = faceFeatureTrackingView_.frame.size.height;
        widthFace = faceFeatureTrackingView_.frame.size.width;
        heightCam = view.frame.size.height;
        widthCam = view.frame.size.width;
        
        NSNumber *myDoubleNumber = [NSNumber numberWithDouble:heightFace];
        [myDoubleNumber stringValue];
        NSNumber *xface = [NSNumber numberWithDouble:xOriginFace_];
        [xface stringValue];
    
        [tempLabel setText:xface.stringValue];
        [[self view] addSubview:tempLabel];
        
        // (< & >) reversed due to being flipped upside down
        if (xOriginFace_ > xOriginCamera_){
        }else if(xOriginFace_ == xOriginCamera_){
        }else{
        }

        // Finally check if I smile (change the color)
        if(feature.hasSmile) {
            faceFeatureTrackingView_.layer.borderColor = [[UIColor blueColor] CGColor];
            //set up counter to slow rate of speaking
            if (counter > 70) {
                counter = 0;
                [self playAudio];
                
                while (counter < 30) {
                    counter++;
                }
                counter = 0;
            }
            
        }else {
            faceFeatureTrackingView_.layer.borderColor = [[UIColor greenColor] CGColor];
        }
        counter++;
    }
}


// Calculate transformations for displaying output on the screen
- (void)calculateTransformations
{
    NSInteger outputHeight = [[detector.captureSession.outputs[0] videoSettings][@"Height"] integerValue];
    NSInteger outputWidth = [[detector.captureSession.outputs[0] videoSettings][@"Width"] integerValue];

    if (UIInterfaceOrientationIsPortrait(detector.outputImageOrientation)) {
        // Portrait mode, swap width & height
        NSInteger temp = outputWidth;
        outputWidth = outputHeight;
        outputHeight = temp;
    }
    
    // Use self.view because self.cameraView is not resized at this point (if 3.5" device)
    CGFloat viewHeight = self.view.frame.size.height;
    CGFloat viewWidth = self.view.frame.size.width;
    
    // Calculate the scale and offset of the view vs the camera output
    // This depends on the fillmode of the GPUImageView
    CGFloat scale;
    CGAffineTransform frameTransform;
    switch (view.fillMode) {
        case kGPUImageFillModePreserveAspectRatio:
            scale = MIN(viewWidth / outputWidth, viewHeight / outputHeight);
            frameTransform = CGAffineTransformMakeScale(scale, scale);
            frameTransform = CGAffineTransformTranslate(frameTransform, -(outputWidth * scale - viewWidth)/2, -(outputHeight * scale - viewHeight)/2 );
            break;
        case kGPUImageFillModePreserveAspectRatioAndFill:
            scale = MAX(viewWidth / outputWidth, viewHeight / outputHeight);
            frameTransform = CGAffineTransformMakeScale(scale, scale);
            frameTransform = CGAffineTransformTranslate(frameTransform, -(outputWidth * scale - viewWidth)/2, -(outputHeight * scale - viewHeight)/2 );
            break;
        case kGPUImageFillModeStretch:
            frameTransform = CGAffineTransformMakeScale(viewWidth / outputWidth, viewHeight / outputHeight);
            break;
    }
    cameraOutputToPreviewFrameTransform_ = frameTransform;
    
    
    //working
    //In portrait mode, need to swap x & y coordinates of the returned boxes
    if (UIInterfaceOrientationIsPortrait(detector.outputImageOrientation)) {
        // Interchange x & y
        portraitRotationTransform_ = CGAffineTransformMake(0, 1, 1, 0, 0, 0);
        
    }else {
        portraitRotationTransform_ = CGAffineTransformIdentity;
    }
    
    // AVMetaDataOutput works in texels (relative to the image size)
    // We need to transform this to pixels through simple scaling
    texelToPixelTransform_ = CGAffineTransformMakeScale(outputWidth, outputHeight);
}

//play recorded audio file "Hello I'm Double"
- (void)playAudio {
    CFBundleRef mainBundle = CFBundleGetMainBundle();
    CFURLRef soundFileURLRef;
    UInt32 soundID;
    
    //choose random number 0-2
    int randomPick = arc4random_uniform(3);
    
    //choose random greeting
    
    if(randomPick == 0){
        soundFileURLRef = CFBundleCopyResourceURL(mainBundle, (CFStringRef) @"doubleHi", CFSTR ("m4a"), NULL);
    }else if(randomPick == 1){
        soundFileURLRef = CFBundleCopyResourceURL(mainBundle, (CFStringRef) @"dsmiling", CFSTR ("m4a"), NULL);
    }else if(randomPick == 2){
        soundFileURLRef = CFBundleCopyResourceURL(mainBundle, (CFStringRef) @"meetyou", CFSTR ("m4a"), NULL);
    }
    
    
    
    AudioServicesCreateSystemSoundID(soundFileURLRef, &soundID);
    AudioServicesPlaySystemSound(soundID);
    
   }

//DOUBLE DELEGATE

- (void)doubleDidConnect:(DRDouble *)theDouble {
    NSLog(@"Double is Connected");

}

- (void)doubleDidDisconnect:(DRDouble *)theDouble {
    NSLog(@"Double is Disconnected");
}

- (void)doubleStatusDidUpdate:(DRDouble *)theDouble {
   
}

- (void)doubleDriveShouldUpdate:(DRDouble *)theDouble {
    //if face is found follow it
    if (isRecognitionSetup == true) {
        if (goToFace == true) {
            calFinished = false;
            amountTurned = 0;
            int kickstandState = [[DRDouble sharedDouble] kickstandState];
            int deployed = 1;
            
            if (kickstandState == deployed){
                [[DRDouble sharedDouble] retractKickstands];
            }

            CGFloat xOriginCamLeftBound;
            CGFloat xOriginCamRightBound;
            xOriginCamLeftBound = xOriginCamera_ - 120;
            xOriginCamRightBound = xOriginCamera_ + 120;
    
            //bool centered = xOriginCamera_ == xOriginFace_;
            bool left = xOriginFace_ < xOriginCamLeftBound  ? true : false;
            bool right = xOriginFace_ > xOriginCamRightBound ? true : false;
            float turn = (left) ? 1.0 : ((right) ? -1.0 : 0.0);
    
            //DRDriveDirection drive = centered ? kDRDriveDirectionForward : kDRDriveDirectionStop;
            bool forward = heightFace < 150 ? true : false;
    
            //drive if face is far enough away and face is centered on screen
            DRDriveDirection drive = forward && !left && !right ? kDRDriveDirectionForward : kDRDriveDirectionStop;
            [theDouble drive:drive turn:turn];
        }
    
        //if not search around the room
        if (lookForFace == true) [calibrateDelay fire]; //start the timer to look for face again
        if ((lookForFace == true) && (calFinished == true)) {
            NSLog(@"amount turned %f", amountTurned);
            int kickstandState = [[DRDouble sharedDouble] kickstandState];
            int deployed = 1;
            int retracted = 2;
            if(amountTurned >= 360) { //scanned entire room face could not be found
                if (kickstandState == retracted){
                    [[DRDouble sharedDouble] deployKickstands];
                }
            } else { //still scanning continue to turn and look around the room for the recognized face
                [theDouble turnByDegrees:1.0];
            }
            
            if (kickstandState != deployed) {
            } else {
                amountTurned = amountTurned + 2;
            }
        }
    }
}

- (void)doubleTravelDataDidUpdate:(DRDouble *)theDouble {

}

//actions

- (void) parkButtonPressed {
    
    int kickstandState = [[DRDouble sharedDouble] kickstandState];
    
    //integers associated with kickstand current states
    int deployed = 1;
    int retracted = 2;
    
    if(kickstandState == retracted){
        [[DRDouble sharedDouble] deployKickstands];
    }else if (kickstandState == deployed){
        [[DRDouble sharedDouble] retractKickstands];
    }
}



- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

-(UIImage *)UIImageFromCVMat:(cv::Mat)cvMat {
    NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
    CGColorSpaceRef colorSpace;
    
    if (cvMat.elemSize() == 1) {
        colorSpace = CGColorSpaceCreateDeviceGray();
    } else {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    
    // Creating CGImage from cv::Mat
    CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                        cvMat.rows,                                 //height
                                        8,                                          //bits per component
                                        8 * cvMat.elemSize(),                       //bits per pixel
                                        cvMat.step[0],                            //bytesPerRow
                                        colorSpace,                                 //colorspace
                                        kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                        provider,                                   //CGDataProviderRef
                                        NULL,                                       //decode
                                        false,                                      //should interpolate
                                        kCGRenderingIntentDefault                   //intent
                                        );
    
    
    // Getting UIImage from CGImage
    UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return finalImage;
}

- (UIImage* ) scaleUIImage: (UIImage *) img {
    CGRect rect = CGRectMake(0, 0, 50, 50);
    UIGraphicsBeginImageContext(rect.size);
    [img drawInRect:rect];
    UIImage* newImg = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    NSData* imgData = UIImagePNGRepresentation(newImg);
    return [UIImage imageWithData:imgData];
}

- (cv::Mat) UIImagetoMat: (UIImage*) cap {
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(cap.CGImage);
    CGFloat cols = cap.size.width;
    CGFloat rows = cap.size.height;
    cv::Mat cvMat(rows, cols, CV_8UC4); //
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 //Pointer to data
                                                    cols,                       // Width of bitmap
                                                    rows,                       // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), cap.CGImage);
    CGContextRelease(contextRef);
    cv::Mat greyMat;
    cv::cvtColor(cvMat, greyMat, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(greyMat, greyMat);
    return greyMat;
}

- (void) gatherTrainingInformation: (CGRect) r {
    if (isRecognitionSetup == true) return;
    //Get a subimage from the main screen image (NOTE: surround block with either threshold or time interval for differentiated training faces)
    CGImageRef ref = CGImageCreateWithImageInRect(capture.CGImage, r);
    trainingFaces[numFaces] = [UIImage imageWithCGImage:ref];
    trainingFaces[numFaces] = [self scaleUIImage:trainingFaces[numFaces]]; //scale the training faces to uniform size
    trainMat.push_back([self UIImagetoMat:trainingFaces[numFaces]]);
    numFaces++;
    CGImageRelease(ref);
    //if 12 faces are saved then and setup the recognizer
    if (numFaces == NUMBER_OF_FACES) {
        newView.frame = CGRectMake(0, 0, capture.size.width, capture.size.height);
        NSLog(@"capture size w%f", capture.size.width);
        NSLog(@"capture size h%f", capture.size.height);
        //[newView setImage:capture];
        //[newView setImage:trainingFaces[5]];
        NSLog(@"Recognition is Set");
        [self setUpRegcognition];
        isRecognitionSetup = true;
    }
}

//Face Recognition Function
- (void) setUpRegcognition {//TODO: NOTE we may need to lower the X/Y GRID to make it less skeptical and possibly increase the number of training faces //But so far these parameters are ok
    recognizer = cv::createLBPHFaceRecognizer(5,8,12,12);
    //recognizer = cv::createLBPHFaceRecognizer();
    labels = {5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5}; //label is abritrary value just for recognition result//additionally to labels may need to setup confidence scale for individual pictues
    recognizer->train(trainMat,labels);
}

- (bool) trytoRecognizeFace: (CGRect) r  {
    if (isRecognitionSetup == true) {
        CGImageRef faceOnScreen = CGImageCreateWithImageInRect(capture.CGImage, r);
        UIImage *img = [UIImage imageWithCGImage:faceOnScreen];
        UIImage *scaledImg = [self scaleUIImage:img];
        CGImageRelease(faceOnScreen);
        cv::Mat fosMat = [self UIImagetoMat:scaledImg];
        double confidence = 0;
        int myLabel = -1;
        recognizer->predict(fosMat, myLabel, confidence);
        NSLog(@"confidence: %f", confidence);
        if (myLabel == 5) {
            if (confidence > 120) return false;
            return true;
        } else {
            return false;
        }
            }
    return false;
}

- (void) response:(DRDouble *)theDouble {
    //pick a random number to determine the reaction
    int lowerBound = 1;
    int upperBound = 3;
    int randChoice = lowerBound + arc4random() % (upperBound - lowerBound);
    bool toTurn = false;
    CFBundleRef mainBundle = CFBundleGetMainBundle();
    CFURLRef soundFileURLRef;
    UInt32 soundID;
    
    switch(randChoice){
        case 1:
            //move poll up and down and say hello
            [[DRDouble sharedDouble] poleDown];
            [[DRDouble sharedDouble] poleUp];
            [[DRDouble sharedDouble] poleDown];
            [[DRDouble sharedDouble] poleUp];
            soundFileURLRef = CFBundleCopyResourceURL(mainBundle, (CFStringRef) @"doubleHi", CFSTR ("m4a"), NULL);
            AudioServicesCreateSystemSoundID(soundFileURLRef, &soundID);
            AudioServicesPlaySystemSound(soundID);
        case 2:
            //Smiling
            soundFileURLRef = CFBundleCopyResourceURL(mainBundle, (CFStringRef) @"dsmiling", CFSTR ("m4a"), NULL);
            AudioServicesCreateSystemSoundID(soundFileURLRef, &soundID);
            AudioServicesPlaySystemSound(soundID);
        case 3:
            //move poll up and down and say nice to meet you
            [[DRDouble sharedDouble] poleDown];
            [[DRDouble sharedDouble] poleUp];
            [[DRDouble sharedDouble] poleDown];
            [[DRDouble sharedDouble] poleUp];
            soundFileURLRef = CFBundleCopyResourceURL(mainBundle, (CFStringRef) @"meetyou", CFSTR ("m4a"), NULL);
            AudioServicesCreateSystemSoundID(soundFileURLRef, &soundID);
            AudioServicesPlaySystemSound(soundID);
    }
}
@end
