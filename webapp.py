import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av

# import pyttsx3
# from gtts import gTTS

import tensorflow as tf
import mediapipe as mp


st.set_page_config(page_title="Action Detection", page_icon="🤖")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


mp_drawing = mp.solutions.drawing_utils #drawing utilities
mp_holistic = mp.solutions.holistic #holistic models

actions = np.array(['namaste','hello','goodmorning', 'howareyou', 'IamFine', 'thankyou', 'Iamlearning', 'signs', 'iloveyou', 'byebye'])

 # 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #color conversion
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #color conversion
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(255,255,0),thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(255,255,204),thickness=1, circle_radius=1)
                             )
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,128,255),thickness=4, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,255,255),thickness=4, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(102,0,0),thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,102,102),thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(153,0,153),thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(255,153,255),thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    return np.concatenate([pose, face, lh, rh])

def load_model():
    model = tf.keras.models.load_model('actionmodel.h5')
    return model

colors = [(245,117,16), (117,245,16), (16,117,245), (235,215,0), (225,0,215), (60,225,50), (35,235,105), (235,20,155), (235,215,90), (225,115,80)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        
    return output_frame

class OpenCVVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.actions = actions
        self.seq = sequence
        self.sent = sentence
        self.pred = predictions
        self.threshold = threshold
        self.model = load_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")         

        ####################################################################################
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:

                # Make detections
                image, results = mediapipe_detection(img, holistic)
                print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                self.seq.append(keypoints)
                self.seq = self.seq[-20:]
                
                if len(self.seq) == 20:
                    res = self.model.predict(np.expand_dims(self.seq, axis=0))[0]
                    print(self.actions[np.argmax(res)])
                    self.pred.append(np.argmax(res))
                    
                    
                #3. Vizualization and Text To Speech logic
                    if np.unique(self.pred[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > self.threshold: 
                            if len(self.sent) > 0: 
                                if self.actions[np.argmax(res)] != self.sent[-1]:
                                    self.sent.append(self.actions[np.argmax(res)])
                                    # engine = pyttsx3.init()
                                    # engine.setProperty("rate", 90)
                                    # engine.say(self.sent[-1])
                                    # engine.runAndWait()
                        ########## using google text to speech ########
                                    # text = self.sent[-1]
                                    # tts = gTTS(text)
                                    # tts.save(self.sent[-1]+".wav")
                                    # file = self.sent[-1]+".wav"
                                    # os.system(file)
                            
                            else:
                                self.sent.append(self.actions[np.argmax(res)])
                                # engine = pyttsx3.init()
                                # engine.setProperty("rate", 100)
                                # engine.say(self.sent[-1])
                                # engine.runAndWait()

                    if len(self.sent) > 4: 
                        self.sent = self.sent[-4:]
                    
                    
                    image = prob_viz(res, actions, image, colors)
            
                cv2.rectangle(image, (0,0), (640, 40), (0, 5, 0), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

                return av.VideoFrame.from_ndarray(image, format="bgr24")


def main():
    # Action Detection Application #
    st.title("Real Time Sign Language Recognition Application")
    activities = ["Home", "Action Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(
        """ Developed by Nivedita Rani    
            Email : nivedita.rani2020@vitstudent.ac.in""")

    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Action Detection Application using OpenCV, LSTM model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time person detection using web cam feed.
                 2. Real time ASL/ISL sign language recognization.
                 """)

    elif choice == "Action Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your Signs")
        webrtc_streamer(key="key", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=OpenCVVideoProcessor, 
                        async_processing=True, media_stream_constraints={"video": True, "audio": False}
                        )

    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                    <h4 style="color:white;text-align:center;">
                                    Real time Action Detection Application using OpenCV, LSTM deep learning model and Streamlit.</h4>
                                    </div>
                                    </br>"""
        st.markdown(html_temp_about1, unsafe_allow_html=True)

        html_temp4 = """
                             		<div style="background-color:#98AFC7;padding:10px">
                             		<h4 style="color:white;text-align:center;">This Application is developed by Nivedita Rani using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you have any suggestion or want to comment just write a mail at nivedita.rani2020@vitstudent.ac.in . </h4>
                             		<h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                             		</div>
                             		<br></br>
                             		<br></br>"""

        st.markdown(html_temp4, unsafe_allow_html=True)

    else:
        pass


if __name__ == "__main__":
    main()