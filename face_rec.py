import numpy as np
import pandas as pd
import cv2
import redis

import db

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Connect to Redis Client

hostname = 'redis-11028.c264.ap-south-1-1.ec2.redns.redis-cloud.com'
portnumber = '11028'
password = 'V4v91HRFAef1alFXEIIwyvPw9gRiGKT3'


r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)  

# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.6)

# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector,name_role=['Name','Role'],thresh=0.6):
    """
    cosine similarity base search algorithm
    """
    
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    X_list = [np.array(x) for x in X_list if isinstance(x, np.ndarray) and x.shape == test_vector.shape]
    if not X_list:
        # If X_list is empty after filtering, return 'Unknown'
        return 'Unknown', 'Unknown'
    x = np.asarray(X_list)

    # # Convert to numpy array
    # x = np.vstack(X_list) 
    

    
    
    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,512))
    similar_arr = np.array(similar).flatten()
    dataframe = dataframe.iloc[:len(similar_arr)]
    dataframe['cosine'] = similar_arr 

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')

    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    return person_name, person_role
# def ml_search_algorithm(dataframe, feature_column, test_vector, name_role, thresh=0.5):
#     # Example logic
#     match_found = False  # Simulated condition
#     if match_found:
#         return "Person Name", "Person Role"
#     else:
#         return None, None  # Always return a tuple
def face_prediction(test_image, dataframe, feature_column, name_role=['Name', 'Role'], thresh=0.8):
    # Step-1: Take the test image and apply to insight face
    results = faceapp.get(test_image)
    test_copy = test_image.copy()

    # Step-2: Use for loop to extract each embedding and pass to ml_search_algorithm
    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res['embedding']

        # Call ml_search_algorithm and check result
        result = ml_search_algorithm(dataframe, feature_column, test_vector=embeddings, name_role=name_role, thresh=thresh)
        print(result)
        if result is None or len(result) != 2:
            person_name, person_role = "Unknown", "Unknown"
        else:
            person_name, person_role = result

        # Choose color based on the result
        if person_name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (0, 255, 0)  # Green for known

        # Draw rectangle and put text
        cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)
        text_gen = person_name
        cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

    return test_copy

#to detect multiple images
# def face_prediction(test_image,retrive_df,feature_column, name_role=['Name','Role'],thresh=0.5):
#     # step-1: take the test image and apply to insight face
#     results = faceapp.get(test_image)
#     test_copy = test_image.copy()
#     # step-2: use for loop and extract each embedding and pass to ml_search_algorithm

#     for res in results:
#         x1, y1, x2, y2 = res['bbox'].astype(int)
#         embeddings = res['embedding']
#         person_name,person_role = ml_search_algorithm(retrive_df,
#                                                       feature_column,
#                                                       test_vector=embeddings,
#                                                       name_role=name_role,
#                                                       thresh=thresh)
        
#         if person_name == 'Unknown':
#             color =(0,0,255) # bgr
#         else:
#             color = (0,255,0)


#         cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)

#         text_gen = person_name
#         cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)

#     return test_copy    





