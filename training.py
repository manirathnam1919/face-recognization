import numpy as np
import pandas as pd 
import cv2
import insightface
import matplotlib.pyplot as plt
import os 
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors
import db
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

#buffalo l model 
app_1 = FaceAnalysis(name='buffalo_l',root='buffalo_l',providers=['CPUExecutionProvider'])
app_1.prepare(ctx_id=0, det_size=(640, 640))
#buffalo sc model 
app_2 = FaceAnalysis(name='buffalo_sc',root = 'buffalo_sc',providers = ['CPUExecutionProvider'])
app_2.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread("/home/sathyadeepreddy/Downloads/FRAMS/test_image_2.jpg")
results_1  = app_1.get(img)
print(results_1)
print(type(results_1))
print(len(results_1))
print(results_1[0].keys())
results_1[0]['bbox']
results_1[0]['kps']
results_1[0]['det_score']
results_1[0]['gender']
results_1[0]['age']


#gender encoding
img_copy= img.copy()
gender_encode = ['Female','Male']
for res in results_1:
    x1,y1,x2,y2 = res['bbox'].astype('int')

    #draw rectangle
    cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)

    #Face key points
    kps = res['kps'].astype(int)
    for k1,k2 in kps:
       cv2.circle(img_copy,(k1,k2),2,(0,255,255),-1)

    #Face detection score
    score = "score: {}%".format(int(res["det_score"]*100))
    cv2.putText(img_copy,score,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,255))

    #age and gender
    gender = gender_encode[res['gender']]
    age = res['age']
    age_gender = f'{gender}::{age}'

    cv2.putText(img_copy,age_gender,(x1, y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,255))



#results using buffalo sc model 
results_2 = app_2.get(img)
print(results_2)
type(results_2), len(results_2)
results_2[0].keys()
results_2[0]['bbox']
results_2[0]['kps']
results_2[0]['det_score']

for res in results_2:
    #bounding box
    x1,y1,x2,y2 = res['bbox'].astype(int)
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    #Face key points
    kps = res['kps'].astype(int)
    for kp in kps:
        cv2.circle(img,kp,3,(0,255,255),-1)

    #Face detection score
    score = "score: {}%".format(int(res['det_score']*100))
    cv2.putText(img,score,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,255,255),2)


#configuring face analysis 
faceapp = FaceAnalysis(name='buffalo_sc',root='buffalo_sc',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5) #dont set the dtection threshold < 0.3
import re
def clean_name(string):
    string = re .sub(r'[^A-Za-z]',' ',string)
    substring = string.title()
    return string

import os
import cv2
import pandas as pd

# Function to clean name and role
def clean_name(name):
    return name.strip().replace(" ", "_")  # Modify as needed

person_info = []
base_path = "/home/sathyadeepreddy/Downloads/FRAMS/images"
listdir = os.listdir(base_path)

for folder_name in listdir:
    # Ensure folder_name has a valid format
    if '-' in folder_name and folder_name.count('-') == 1:
        role, name = folder_name.split('-')
        name = clean_name(name)
        role = clean_name(role)

        # Path to each image folder
        folder_path = os.path.join(base_path, folder_name)
        print(f"Processing folder: {folder_name}")

        # Process each image in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing image: {file_name}")

            # Read the image
            img_arr = cv2.imread(file_path)

            # Check if the image was successfully loaded
            if img_arr is None:
                print(f"Error loading image at path: {file_path}")
                continue  # Skip this image if it didn't load correctly

            # Get face embedding
            result = faceapp.get(img_arr, max_num=1)  # Assumes faceapp.get returns a list of results

            if result:
                # Extracting facial embedding
                res = result[0]
                embedding = res['embedding']

                # Save all info (name, role, embedding) in the list
                person_info.append([name, role, embedding])
    else:
        print(f"Skipping invalid folder name: {folder_name}")

# Convert to DataFrame
dataframe = pd.DataFrame(person_info, columns=['Name', 'Role', 'Facial_Features'])
print(dataframe)


#read test image
img_test = cv2.imread('/home/sathyadeepreddy/Downloads/FRAMS/test_images/test_6.jpeg')
# cv2.imshow('image test',img_test)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#extract feature 
res_test = faceapp.get(img_test,max_num=1)
for i, rt in enumerate(res_test):
    print('current loop =',i)
    bbox = rt['bbox'].astype(int)
    score = int(rt['det_score']*100)
    embed_test = rt['embedding']
print(len(embed_test))
x_list = dataframe['Facial_Features'].to_list()
x= np.asarray(x_list)
print(x.shape)

#ml search algorithm 
y=embed_test.reshape(1,512)
equilidian_distance = pairwise.euclidean_distances(x,y)
manhattan_distance = pairwise.manhattan_distances(x,y)
cosine_distance = pairwise.cosine_similarity(x,y)
data_algorithms = dataframe.copy()
data_algorithms['equilidian'] = equilidian_distance
data_algorithms['manhattan'] = manhattan_distance
data_algorithms['cosine'] = cosine_distance
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(data_algorithms)

print(data_algorithms)

#import os
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/path/to/your/qt/plugins'
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/sathyadeepreddy/.local/lib/python3.10/site-packages/cv2/qt/plugins'


# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


# plots of eqilidian, manhattan and cosine 
# plt.figure(figsize=(8,15))
# plt.subplot(3,1,1)
# plt.plot(data_algorithms['equilidian'])
# plt.xticks(ticks=list(range(len(data_algorithms['equilidian']))),labels=data_algorithms['Name'],rotation=90)
# plt.xlabel("Record Number")
# plt.ylabel("Equilidian Distance")
# plt.title("equilidian")
# plt.grid()

# plt.subplot(3,1,2)
# plt.plot(data_algorithms['manhattan'])
# plt.xticks(ticks=list(range(len(data_algorithms['manhattan']))),labels=data_algorithms['Name'],rotation=90)
# plt.xlabel("Record Number")
# plt.ylabel("manhattan Distance")
# plt.title("manhattan")
# plt.grid()

# plt.subplot(3,1,3)
# plt.plot(data_algorithms['cosine'])
# plt.xticks(ticks=list(range(len(data_algorithms['cosine']))),labels=data_algorithms['Name'],rotation=90)
# plt.xlabel("Record Number")
# plt.ylabel("cosine Distance")
# plt.title("cosine")
# plt.grid()
# print(plt.show())
#identifying the person name with the  equilidian distance 
d_eq_optimal = 25
datafilter = data_algorithms.query(f'equilidian < {d_eq_optimal}')
datafilter.reset_index(drop=True,inplace=True)

if len(datafilter) > 0:
    argmin = datafilter['equilidian'].argmin()
    name ,role = datafilter.loc[argmin][['Name','Role']]
    print(name,role)
else:
     name = 'Unknown'
     role = 'Unknown'

#identifying the person name with manhatan distance 
d_man_optimal = 450
datafilter = data_algorithms.query(f'manhattan < {d_man_optimal}')
datafilter.reset_index(drop=True,inplace=True)

if len(datafilter) > 0:
    argmin = datafilter['manhattan'].argmin()
    name_man ,role_man = datafilter.loc[argmin][['Name','Role']]
    print(name_man,role_man)
else:
     name_man = 'Unknown'
     role_man = 'Unknown'

#identifying the person name with cosine similarity  
d_cos_optimal = 0.4
datafilter = data_algorithms.query(f'cosine > {d_cos_optimal}')
datafilter.reset_index(drop=True,inplace=True)

if len(datafilter) > 0:
    argmax = datafilter['cosine'].argmax()
    name_cos ,role_cos = datafilter.loc[argmax][['Name','Role']]
    print(name_cos, role_cos)
else:
     name_cos = 'Unknown'
     role_cos = 'Unknown'
# print(datafilter)


#identify the multiple persons in an image 
def ml_search_algorithm(dataframe, feature_column, test_vector,name_role=['Name','Role'],thresh=0.5):
    """
    cosine similarity base search algorithm

    """
    # Step 1: take the dataframe - collection of data
    dataframe = dataframe.copy()  
    #step 2 : index face embedding from the dataframe and converting into array
    x_list = dataframe[feature_column].to_list()
    x = np.asarray(x_list)
    #step 3 : calaculatity cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, 512))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr
    #step 4 : filtering the data
    datafilter = dataframe.query(f'cosine > {thresh}')
    if len(datafilter) > 0:
    #step 5 : get the person name
        datafilter.reset_index(drop=True, inplace=True)
        argmax = datafilter['cosine'].argmax()
        person_name, person_role = datafilter.loc[argmax][name_role]
        return person_name, person_role
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        return person_name, person_role

test_img = cv2.imread('/home/sathyadeepreddy/Downloads/FRAMS/test_images/test_11.jpg')
# cv2.imshow('test img',test_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#step-1 take the test image and apply to insightface
results = faceapp.get(test_img)
test_copy = test_img.copy()

#step-2 use for loop and extract each embedding and pass to ml_search_algorithm
for res in results:
    x1, y1, x2, y2 = res['bbox'].astype(int)
    embeddings = res['embedding']
    person_name, person_role = ml_search_algorithm(dataframe,
                                                   'Facial_Features',
                                                   test_vector=embeddings,
                                                   name_role=['Name','Role'],
                                                   thresh = 0.5)
    if person_name == 'Unknown':
        color = (0,0,255)
    else:
        color = (0,255,0)

    cv2.rectangle(test_copy,(x1,y1),(x2,y2),(0,255,0),2)

    text_gen = person_name
    cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,(255,0,255),1)
    # print(person_name, person_role)
# cv2.imshow('test image',test_copy)
# cv2.waitKey()
# cv2.destroyAllWindows()

#compressing the dataset 
dataframe_compress = dataframe.groupby(by=['Name','Role']).mean()
dataframe_compress.reset_index(inplace=True)
print(dataframe_compress)

# # #converting  the dataframe into array and save in numpy zip format
xvalues = dataframe_compress.values
col_name = np.array(dataframe_compress.columns)
np.savez('dataframe_students_teacher.npz',xvalues,col_name)
file_np = np.load('dataframe_students_teacher.npz',allow_pickle=True)
print(pd.DataFrame(file_np['arr_0'],columns=file_np['arr_1']))

# #connecting to redis cloud 
import redis
import db 
hostname = 'redis-11028.c264.ap-south-1-1.ec2.redns.redis-cloud.com'
portnumber = '11028'
password = 'V4v91HRFAef1alFXEIIwyvPw9gRiGKT3'
r = redis.StrictRedis(host=hostname,port=portnumber,password=password)
print(r.ping())
 
np_file = np.load('dataframe_students_teacher.npz',allow_pickle=True)
x_values = np_file['arr_0']
col_names = np_file['arr_1']

df = pd.DataFrame(x_values,columns=col_names)
df['name_role'] = df['Name']+'@'+df['Role']
print(df)
records = df[['name_role','Facial_Features']].to_dict(orient = 'records')
len(records)

#saving data in redis -redis hashes
for record in records:
    name_role = record['name_role']
    vector = record['Facial_Features']
#converting numpy arrays into bytes
    vector_bytes = vector.tobytes()
#save data into redis cloud
    r.hset(name = 'academy:register',key = name_role,value = vector_bytes)

#configuring face analysis 
from insightface.app import FaceAnalysis
faceapp = FaceAnalysis(name='buffalo_l',root='insightface_model',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)

# #resgistration form  collecting name and role 

person_name = input('Enter your name: ')

trials = 3
for i in range(trials):
    role = input("""
    please choose
    1.developer or manager
    2.other


  enter number either 1 or 2
""")
    if role in ("1","2"):
         if role == "1":
              role = 'developer or manager'
         else:
             role = 'other'

         break
    else:
        print('invalid input')
    if i == 3:
          print('exceeds maximum trails')

key = person_name+'@'+role
print('Your name =', person_name)
print('Your role =', role)
print('Key=',key)
# print(person_name,role) 

import cv2
import insightface

# Initialize the video capture object
cap = cv2.VideoCapture(0)
face_embeddings = []
sample = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Unable to read from camera')
        break 

    # Perform face detection using insightface
    results = faceapp.get(frame, max_num=1)
    
    for res in results:
        sample += 1
        # Extract bounding box and convert to integers
        x1, y1, x2, y2 = res['bbox'].astype(int)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract facial features (embedding)
        embeddings = res['embedding']
        face_embeddings.append(embeddings)

    # Stop collecting embeddings after 100 samples
    if sample >= 200:
        break

    # Display the frame with bounding boxes
    cv2.imshow('frame', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()  
cv2.destroyAllWindows()

x_mean = np.asarray(face_embeddings).mean(axis=0)
print(x_mean.shape)
x_mean_bytes = x_mean.tobytes()
# print(x_mean_bytes)
r.hset(name= 'academy:register',key=key,value = x_mean_bytes)

# # if face_embeddings:
# #     averaged_embedding = np.mean(face_embeddings, axis=0)
# #     vector_bytes = averaged_embedding.tobytes()
# #     r.hset(name='academy:register', key=key,value=vector_bytes)
# #     print(f'data stored in redis for {key}.')
# # else: 
# #     print('no face embeddings collectes')
# # # Release resources
# # # cap.release()
# # # cv2.de stroyAllWindows() 

# # #importing a module with all requirements named face_rec.py

import face_rec
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



# # # cap.release()

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame =cap.read()

#     if ret == False:
#          break
    
#     pred_frame = face_rec.face_prediction(frame,retrive_df,'facial_features',['Name','Role'],thresh=0.5)

#     cv2.imshow('frame',frame)
#     cv2.imshow('prediction',pred_frame)

#     if cv2.waitKey(1) == 27:
#          break
    
# cap.release()
# cv2.destroyAllWindows()

