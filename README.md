# face-recognition-spoofing
CIS 663 Biometrics project.  Generate face images that will spoof face authentication systems


**NOTE**  api.py must be modified in the face_recognition library to successfully run run_tests.py

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