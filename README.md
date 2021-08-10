# OpenCV AI Competition 2021 :trophy:
## Hats'i Team :robot:
<img src="./img_readme/opencv_finalist.jpg" width="600" height="700" style="display:block; margin: 0 auto;">

---
### Automatic Medical Translator :syringe: | Mexican Sign Language - Spanish :mexico:

<p style="text-align: justify;">
The automatic medical translator recognizes a pre-established set of signs to generate a report with the users' symptoms. The physician will analyze the report and will be able to make the diagnosis and indicate the treatment to follow.
</p>

- The body keypoints were obtained using personalized MediaPipe models
- The prediction process was done using a LSTM network

<img src="./img_readme/modelDemo.gif" width="800" height="500">

---
### Install :gear:
Running on Python 3.9.2
Install all the necessary packages
```
pip install -r requirements.txt
```
---
### Usage :video_game:
Download the pre-trained model from https://drive.google.com/file/d/1Tf9y-uDrDdjj8IE2839PzHxtzGv0Dxhv/view?usp=sharing and put it into the `codes/` directory

There are multiple code files:
- `oakD.py` contains all the configuration process to setup the oakD camera
- `bodyPointsDetector.py` gets all the X,Y and Z values in meters of each keypoint
- `signCapture.py` captures a sliding window of 15 frames per sign and generates its CSV file
- `cleanData.py` cleans the dataframes with different operations 
- `modelTraining.py` trains the LSTM model with the sliding windows of 15 frames

#### Directories
```
hatsi/
└─ codes/
    ├─ videos/
    |   └── ...
    ├─ oakD.py
    ├─ bodyPointsDetector.py
    ├─ signCapture.py
    ├─ cleanData.py
    ├─ modelTrining.py
    └─ En_Enf_500_1.h5
```
#### Run an example
`python3 realTimeDemo.py`

---
### Landmarks Diagrams :dart:
#### Face landmarks <br><img src="./img_readme/rostro.jpg" width="500" height="500" style="display:block; margin: 0 auto;">
#### Body landmarks <br><img src="./img_readme/cuerpo.jpeg" width="500" height="500" style="display:block; margin: 0 auto;">
#### Hands landmarks <br><img src="./img_readme/mano.png" width="500" height="500" style="display:block; margin: 0 auto;">

