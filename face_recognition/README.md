# Face Recognition with Python and OpenCV

## DATA
Data obtained from [OpenCV tutorials](https://docs.opencv.org/3.4/da/d60/tutorial_face_main.html)
- [AT&T Facedatabase](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

Contains 10 different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).

- [Yale Facedatabase A](http://vision.ucsd.edu/content/yale-face-database)

The Yale Facedatabase A (also known as Yalefaces) is a more appropriate dataset for initial experiments, because the recognition problem is harder. The database consists of 15 people (14 male, 1 female) each with 11 grayscale images sized 320×243 pixel. There are changes in the light conditions (center light, left light, right light), facial expressions (happy, normal, sad, sleepy, surprised, wink) and glasses (glasses, no-glasses).

### Preparing the data
Write a simple CSV file where its lines are composed by a filename refering to an image and its label (integer).

Download the AT&T Facedatabase from AT&T Facedatabase and the corresponding CSV file from at.txt, which looks like this:

'''
./at/s1/1.pgm;0
./at/s1/2.pgm;0
...
./at/s2/1.pgm;1
./at/s2/2.pgm;1
...
./at/s40/1.pgm;39
./at/s40/2.pgm;39
'''

Once you have a CSV file with valid filenames and labels, you can run any of the demos by passing the path to the CSV file as parameter:

'''
facerec_demo.exe D:/data/at.txt
'''