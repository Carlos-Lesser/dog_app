# load json and create model
from keras.optimizers import Adam
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as image1
import streamlit as st

###########################################################################################################################


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
optimizer = Adam(learning_rate=0.0001)
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer= optimizer, metrics=['accuracy'])
########################################################################################################################
label_map = {0: 'Affenpinscher',
 1: 'Afghan_hound',
 2: 'African_hunting_dog',
 3: 'Airedale',
 4: 'American_Staffordshire_terrier',
 5: 'Appenzeller',
 6: 'Australian_terrier',
 7: 'Basenji',
 8: 'Basset',
 9: 'Beagle',
 10: 'Bedlington_terrier',
 11: 'Bernese_mountain_dog',
 12: 'Black-and-tan_coonhound',
 13: 'Blenheim_spaniel',
 14: 'Bloodhound',
 15: 'Bluetick',
 16: 'Border_collie',
 17: 'Border_terrier',
 18: 'Borzoi',
 19: 'Boston_bull',
 20: 'Bouvier_des_Flandres',
 21: 'Boxer',
 22: 'Brabancon_griffon',
 23: 'Briard',
 24: 'Brittany_spaniel',
 25: 'Bull_mastiff',
 26: 'Cairn',
 27: 'Cardigan',
 28: 'Chesapeake_Bay_retriever',
 29: 'Chihuahua',
 30: 'Chow',
 31: 'Clumber',
 32: 'Cocker_spaniel',
 33: 'Collie',
 34: 'Curly-coated_retriever',
 35: 'Dandie_Dinmont',
 36: 'Dhole',
 37: 'Dingo',
 38: 'Doberman',
 39: 'English_foxhound',
 40: 'English_setter',
 41: 'English_springer',
 42: 'EntleBucher',
 43: 'Eskimo_dog',
 44: 'Flat-coated_retriever',
 45: 'French_bulldog',
 46: 'German_shepherd',
 47: 'German_short-haired_pointer',
 48: 'Giant_schnauzer',
 49: 'Golden_retriever',
 50: 'Gordon_setter',
 51: 'Great_Dane',
 52: 'Great_Pyrenees',
 53: 'Greater_Swiss_Mountain_dog',
 54: 'Groenendael',
 55: 'Ibizan_hound',
 56: 'Irish_setter',
 57: 'Irish_terrier',
 58: 'Irish_water_spaniel',
 59: 'Irish_wolfhound',
 60: 'Italian_greyhound',
 61: 'Japanese_spaniel',
 62: 'Keeshond',
 63: 'Kelpie',
 64: 'Kerry_blue_terrier',
 65: 'Komondor',
 66: 'Kuvasz',
 67: 'Labrador_retriever',
 68: 'Lakeland_terrier',
 69: 'Leonberg',
 70: 'Lhasa',
 71: 'Malamute',
 72: 'Malinois',
 73: 'Maltese_dog',
 74: 'Mexican_hairless',
 75: 'Miniature_pinscher',
 76: 'Miniature_poodle',
 77: 'Miniature_schnauzer',
 78: 'Newfoundland',
 79: 'Norfolk_terrier',
 80: 'Norwegian_elkhound',
 81: 'Norwich_terrier',
 82: 'Old_English_sheepdog',
 83: 'Papillon',
 84: 'Pekinese',
 85: 'Pembroke',
 86: 'Pomeranian',
 87: 'Pug',
 88: 'Redbone',
 89: 'Rhodesian_ridgeback',
 90: 'Rottweiler',
 91: 'Saint_Bernard',
 92: 'Saluki',
 93: 'Samoyed',
 94: 'Schipperke',
 95: 'Scotch_terrier',
 96: 'Scottish_deerhound',
 97: 'Sealyham_terrier',
 98: 'Shetland_sheepdog',
 99: 'Shih-Tzu',
 100: 'Siberian_husky',
 101: 'Silky_terrier',
 102: 'Soft-coated_wheaten_terrier',
 103: 'Staffordshire_bullterrier',
 104: 'Standard_poodle',
 105: 'Standard_schnauzer',
 106: 'Sussex_spaniel',
 107: 'Tibetan_mastiff',
 108: 'Tibetan_terrier',
 109: 'Toy_poodle',
 110: 'Toy_terrier',
 111: 'Vizsla',
 112: 'Walker_hound',
 113: 'Weimaraner',
 114: 'Welsh_springer_spaniel',
 115: 'West_Highland_white_terrier',
 116: 'Whippet',
 117: 'Wire-haired_fox_terrier',
 118: 'Yorkshire_terrier',
 119: 'otterhound'}




@st.cache
def predictor(im, top_results=3):
    """
    This function takes in an image and returns a prediction on dog breed.
    INPUT: the path to the image to be classified
    OUTPUT: returns dog breed
    """
    #im = image(img_path, target_size=(299,299)) # -> PIL image
    im = Image.open(uploaded_file) # -> PIL image
    im = im.resize((299,299))
    doc = image1.img_to_array(im) # -> numpy array
    doc = np.expand_dims(doc, axis=0)
    doc = doc/255.0
    ############display(image1.array_to_img(doc[0]))

    # make a prediction of dog_breed based on image
    prediction =loaded_model.predict(doc)[0]
    dog_breed_indexes = prediction.argsort()[-top_results:][::-1]
    probabilities = sorted(prediction, reverse=True)[:top_results]

    for i in range(top_results):
        s = "This dog looks like a {} with probability {:.2f} %".format(label_map.get(dog_breed_indexes[i]), (probabilities[i]*100))

    
    return  s


#########################################################################################################################



st.write("""
         # Dog Breed Image Classification
         """
         )
st.write("This is a image classification web app to predict different dog breeds")

uploaded_file = st.file_uploader("Please upload an image file", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.',width = 399)
    st.write("Classifying...")
    label = predictor(image,1)
    st.write(label)
    st.write(predictor(image,2))
    st.write(predictor(image,3))

  
   




