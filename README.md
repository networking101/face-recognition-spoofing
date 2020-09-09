# face-recognition-spoofing
CIS 663 Biometrics project.  Generate face images that will spoof face authentication systems

A face recognition system is one of the important biometric information processes. It can be used to identify or verify a person from a digital image. Its applicability is easier and working range is larger than the other biometric modalities. Face recognition system is used in security which involves extracting its features and then recognizing it, regardless of lighting, expression, illumination, ageing, transformations (translate, rotate and scale image) and pose, which is a difficult task. While the face recognition system is being widely used in different security systems, there are challenges associated with this system such as spoofing. In one spoofing case, the attackers can modify the important features of a face to look like a victim in order to bypass the security systems. This research project looked at how modifying key facial features can be used to spoof a facial recognition and authentication system.  Three different cases were run against the OpenCV facial recognition library.  One where the original attacker image is run against a victim, another, where the size and position of the attacker’s features match the victim’s, and the last case where the victim’s features are copied onto the spoofed image.  The results of the tests showed the effectiveness of all three cases.  When the original attacker images were run, OpenCV correctly identified a facial recognition match 98.6 percent of the time.  When only size and position were modified there was minimal effect on face comparison and OpenCV still had a success rate of 93 percent.  When the victim’s features were translated over to the spoofed image, OpenCV only made the correct comparison 63.4 percent of the time.  This shows that minute details of key features have a large impact on the success of facial recognition in OpenCV, but size and position alone do not.  Also, OpenCV facial recognition favors a higher true positive rate over a higher true negative rate.  This means that accepting two different faces is favorable to not identifying a match between the same faces.

**NOTE**:  api.py must be modified in the face_recognition library to successfully run run_tests.py

```
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    ret = face_distance(known_face_encodings, face_encoding_to_check)[0]
    return (list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance), ret)
```

run_tests.py takes an attacker image and tests it against all files in the "victims" directory.  The attacker's image is spoofed to match each victim's image and compared.  A dictionary of the output is saved in the local directory, which contains a list of all calculated differences.

```python3 run_tests.py --encodings encodings.pickle --attacker Aberdeen/hin1.jpg --victims Aberdeen/```

face_adjustnment.py takes an attacker image and victim image, and modifies the attacker to spoof the victim.  The size and position of the eyes, nose, mouth, and jawline are modified and an image file is saved to the local directory.

```python3 face_adjustnment.py --image1 michael_pics/alec1.jpg --image2 michael_pics/adrian1.jpg```
