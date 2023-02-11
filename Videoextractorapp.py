import streamlit as st
import cv2
import numpy as np
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

st.title("Video Emotion Extractor")

# Upload video file
file = st.file_uploader("Upload video file", type=["mp4", "avi"])

if file is not None:
    # Read video file
    video = cv2.VideoCapture(file)

    # Set frame rate
    frame_rate = 30.0

    # Get total number of frames
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get frame height and width
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Identify scenarios
    # For simplicity, let's assume the goal moments start at minute 1, 5, and 9
    goal_moments = [60, 300, 540]

    # Extract segments
    segments = []
    for moment in goal_moments:
        start = int(moment - 30 * frame_rate)
        end = int(moment + 30 * frame_rate)
        if start >= 0 and end <= num_frames:
            segment = []
            video.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ret, frame = video.read()
                segment.append(frame)
            segments.append(segment)

    # Release video file
    video.release()

    # Show segments
    st.write("Extracted segments:")
    for segment in segments:
        st.write("Segment:")
        for frame in segment:
            st.image(frame)


st.title("Emotion Detection in Video")

# Upload video file
file = st.file_uploader("Upload video file", type=["mp4", "avi"])

if file is not None:
    # Read video file
    video = cv2.VideoCapture(file)

    # Set frame rate
    frame_rate = 30.0

    # Get total number of frames
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get frame height and width
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Initialize goal moments, happy moments, and loss moments
    goal_moments = []
    happy_moments = []
    loss_moments = []

    # Loop over all frames
    for i in range(num_frames):
        # Read next frame
        ret, frame = video.read()

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop over all detected faces
        for (x, y, w, h) in faces:
            # Crop face region
            face = frame[y:y+h, x:x+w]

            # Resize face region
            face = cv2.resize(face, (64, 64))

            # Convert face region to grayscale
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Predict emotion using deep learning model
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))
            # Compile the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Fit the model on the training data
            train_datagen = ImageDataGenerator(rescale=1./255)
            validation_datagen = ImageDataGenerator(rescale=1./255)

            train_generator = train_datagen.flow_from_directory(
                'train_set',
                target_size=(img_rows, img_cols),
                batch_size=batch_size,
                color_mode='grayscale',
                class_mode='categorical')

            validation_generator = validation_datagen.flow_from_directory(
                'validation_set',
                target_size=(img_rows, img_cols),
                batch_size=batch_size,
                color_mode='grayscale',
                class_mode='categorical')

            model.fit_generator(
            train_generator,
                steps_per_epoch=train_generator.samples//batch_size,
                epochs=10,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples//batch_size)

            # Save the model
            model.save('emotion_detection_model.h5')

            # Run the face through the model to predict the emotion
            emotion = model.predict(face)

            # Get the emotion label
            def emotion_label(prediction):
                return emotion

            # If emotion is goal, add frame to goal moments
            if emotion_label == "goal":
                goal_moments.append(frame)

            # If emotion is happy, add frame to happy moments
            if emotion_label == "happy":
                happy_moments.append(frame)

            # If emotion is loss, add frame to loss moments
            if emotion_label == "loss":
                loss_moments.append(frame)

# Release video file
video.release()

# Show goal moments
st.write("Goal moments:")
for frame in goal_moments:
    st.image(frame)

# Show happy moments
st.write("Happy moments:")
for frame in happy_mo
    st.image(frame)

# Show loss moments
st.write("Loss moments:")
for frame in loss_momments:
    st.image(frame)
