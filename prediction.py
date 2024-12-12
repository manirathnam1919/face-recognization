import cv2
import face_rec
import numpy as np
import pandas as pd
import db
import insightface
from insightface.app import FaceAnalysis
name = 'academy:register'
retrive_dict = face_rec.r.hgetall(name)
retrive_series = pd.Series(retrive_dict)
retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
index = retrive_series.index
index = list(map(lambda x: x.decode(), index))
retrive_series.index = index
retrive_df = retrive_series.to_frame().reset_index()
retrive_df.columns = ['name_role','facial_features']
retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x: x.split('@')).apply(pd.Series)
print("ppppppp",retrive_df)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(retrive_df) 

import cv2
import numpy as np
from sklearn.metrics import pairwise

def ml_search_algorithm(retrive_df, facial_features, test_vector, name_role=['Name', 'Role'], thresh=0.6):
    """
    Cosine similarity-based search algorithm.
    """
    # Step 1: Take the dataframe - collection of data
    retrive_df = retrive_df.copy()
    
    x_list = retrive_df[facial_features].to_list()
    print("First 5 elements in x_list:", x_list[:5]) 
    # Keep only valid embeddings (NumPy arrays of shape (512,))
    filtered_x_list = [
    embedding for embedding in x_list
    if isinstance(embedding, (np.ndarray, list)) and len(embedding) == 512
]

# Convert to NumPy array
    x = np.asarray(filtered_x_list)
    print(x) 
    valid_rows = retrive_df[facial_features].apply(lambda x: isinstance(x, (np.ndarray, list)) and len(x) == 512)
    filtered_dataframe = retrive_df[valid_rows].copy()
    x_list = filtered_dataframe[facial_features].to_list()
    x = np.asarray(x_list)
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten() 
    filtered_dataframe['cosine'] = similar_arr
    # filtered_dataframe['cosine'] = similar_arr
    # Add the cosine similarity to the filtered DataFrame
    filtered_dataframe = filtered_dataframe.copy()  # Ensure it is a copy, not a slice
    filtered_dataframe.loc[:, 'cosine'] = similar_arr  # Use .loc to safely assign


    
    # Filter results
    datafilter = filtered_dataframe.query(f'cosine > {thresh}')
    if len(datafilter) > 0:
        datafilter.reset_index(drop=True, inplace=True)
        argmax = datafilter['cosine'].argmax()
        person_name, person_role = datafilter.loc[argmax][name_role]
        return person_name, person_role
    else:
        return 'Unknown', 'Unknown'
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture video.")
        break
    faceapp = FaceAnalysis(name='buffalo_sc',root='buffalo_sc',providers=['CPUExecutionProvider'])
    faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.6)
    # faceapp = FaceAnalysis(allowed_modules=['detection', 'recognition'])
#     faceapp.prepare(ctx_id=0, det_size=(640, 640))
    # Detect faces in the frame
    results = faceapp.get(frame)

    for res in results:
        # Get bounding box and embeddings
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']

        # Perform prediction
        person_name, person_role = ml_search_algorithm(
            retrive_df,
            'facial_features',
            test_vector=embeddings,
            name_role=['Name', 'Role'],
            thresh=0.5
        )

        # Set colors for known/unknown
        color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{person_name} ({person_role})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # # Display the frame
    # pred_frame = face_rec.face_prediction(frame,retrive_df,'facial_features',['Name','Role'],thresh=0.5)
    cv2.imshow('Real-Time Face Recognition', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) == 27:
        break

# Release resources 
video_capture.release()
cv2.destroyAllWindows() 

#     # Step 2: Index face embeddings from the dataframe and convert into array
#     # x_list = dataframe[feature_column].to_list()
#     # x = np.asarray(x_list)
#     x_list = retrive_df[feature_column].to_list()
#     print(x_list[:5])  # Print the first 5 elements for inspection
#     x = np.asarray(x_list)  # This is where the error occurs

    
#     # Step 3: Calculate cosine similarity
#     similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
#     similar_arr = np.array(similar).flatten()
#     retrive_df['cosine'] = similar_arr
    
#     # Step 4: Filter the data
#     datafilter = retrive_df.query(f'cosine > {thresh}')
#     if len(datafilter) > 0:
#         # Step 5: Get the person's name and role
#         datafilter.reset_index(drop=True, inplace=True)
#         argmax = datafilter['cosine'].argmax()
#         person_name, person_role = datafilter.loc[argmax][name_role]
#         return person_name, person_role
#     else:
#         return 'Unknown', 'Unknown'


# # Initialize the video stream
# video_capture = cv2.VideoCapture(0)  # Use the default camera

# # Loop for real-time processing
# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break
#     faceapp = FaceAnalysis(allowed_modules=['detection', 'recognition'])
#     faceapp.prepare(ctx_id=0, det_size=(640, 640))
#     # Apply the InsightFace model to detect faces and extract embeddings
#     results = faceapp.get(frame)  # Assuming faceapp is already initialized
    
#     for res in results:
#         x1, y1, x2, y2 = res['bbox'].astype(int)
#         embeddings = res['embedding']
        
#         # Call the search algorithm
#         person_name, person_role = ml_search_algorithm(
#             retrive_df,
#             'facial_features',
#             test_vector=embeddings,
#             name_role=['Name', 'Role'],
#             thresh=0.5
#         )
        
#         # Set the color and text based on recognition results
#         color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
        
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
#         # Display the person's name
#         cv2.putText(frame, person_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
#     # Display the processed frame
#     cv2.imshow('Real-Time Face Recognition', frame)
    
#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# video_capture.release()
# cv2.destroyAllWindows()

# # cap = cv2.VideoCapture(0)

# # while True:
# #     ret, frame =cap.read()

# #     if ret == False:
# #          break
    
# #     pred_frame = face_rec.face_prediction(frame,retrive_df,'facial_features',['Name','Role'],thresh=0.8)

# #     cv2.imshow('frame',frame)
# #     cv2.imshow('prediction',pred_frame)

# #     if cv2.waitKey(1) == 27:
# #          break
    
# # cap.release()
# # cv2.destroyAllWindows() 