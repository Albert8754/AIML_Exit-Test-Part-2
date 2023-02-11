# Highlights extractor
the app uses OpenCV to extract goal, happy, and loss moments from a video. The app starts by uploading a video file using the file_uploader function from Streamlit. Once the file is uploaded, it is read using the cv2.VideoCapture function from OpenCV. The app then identifies the goal moments and extracts 30-second segments before and after each goal moment. Finally, the app displays the extracted segments using the st.image function from Streamlit.

