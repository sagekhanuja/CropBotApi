import requests
from flask import Flask, request, jsonify
from bs4 import BeautifulSoup
import io
import logging
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import calendar
import json

app = Flask(__name__)

######welcome########

@app.route("/")
def hello():
    return "Welcome To CropBot API!"



################returns live info such as temperature based on given latitude and longitude############
@app.route('/info', methods=['GET'])
def info():
    latitude = float(request.args.get('latitude'))
    longitude = float(request.args.get('longitude'))
    
    print("latitude = " + str(latitude))
    print("longitude = " + str(longitude))
    
    currMonth = calendar.month_name[datetime.now().month][:3]
    print(currMonth)
    
    
    weather_url = "https://www.wunderground.com/weather/" + str(latitude) + "," + str(longitude)

    print("url " + str(weather_url))

    r = requests.get(weather_url)

    soup = BeautifulSoup(r.text, 'html.parser')
    temperature = int(soup.find('div', attrs={'class' : 'current-temp'}).find('span', attrs={'class' : 'wu-value wu-value-to'}).text)
    city = soup.find('div', attrs={'class' : 'columns small-12 city-header ng-star-inserted'}).find('span', attrs={'class' : 'station-city'}).text
    elevation = int(soup.find('div', attrs={'class' : 'columns small-12 city-header ng-star-inserted'}).find('span', attrs={'class' : 'wx-value'}).text)
    weather = soup.find_all('a', attrs={'class' : 'module-link'})[-1].text
    pressure = float(soup.find('lib-display-unit', attrs={'type': 'pressure'}).find('span').text.split('\xa0')[0])
    visibility = float(soup.find('lib-display-unit', attrs={'type': 'distance'}).find('span').text.split('\xa0')[0])
    humidity = int(soup.find('lib-display-unit', attrs={'type': 'humidity'}).find('span').text.split('\xa0')[0])
    day_precipitation = float(soup.find('lib-display-unit', attrs={'type': 'rain'}).find('span').text.split('\xa0')[0])
    
    soil_data_url = "https://rest.soilgrids.org/query?lon=" + str(longitude) + "&lat=" + str(latitude)
    r = requests.get(soil_data_url)
    soil_data = json.loads(r.text)
    
    
    monthly_precipitation = soil_data['properties']['PREMRG']['M'][calendar.month_abbr[datetime.now().month]]
    
    
    ##PH
    if len(soil_data['properties']['PHIHOX']['M'].values()) == 0:
        phH20 = 0
    else:
        vals = soil_data['properties']['PHIHOX']['M'].values()
   
        phH20 = sum(vals)/ 10 / len(vals)
    
    
    ##CLAY
    if len(soil_data['properties']['CLYPPT']['M'].values()) == 0:
        clay = 0
    else:
        vals = soil_data['properties']['CLYPPT']['M'].values()
        clay = sum(vals) / len(vals)
        
        
    ##SILT
    if len(soil_data['properties']['SLTPPT']['M'].values()) == 0:
        silt = 0
    else:
        vals = soil_data['properties']['SLTPPT']['M'].values()
        silt = sum(vals) / len(vals)

        
    ##SAND
    if len(soil_data['properties']['SNDPPT']['M'].values()) == 0:
            sand = 0
    else:
        vals = soil_data['properties']['SNDPPT']['M'].values()
        sand = sum(vals) / len(vals)
    
    soil_list = ['clay', 'silt', 'sand', 'loam']
    climate_list = ['tropical', 'temperate', 'arctic']
    
    best_crop_labels = ['Almond Tree',
                        'Banana',
                        'Beans',
                        'Bell Peppers',
                        'Broccoli',
                        'Canola',
                        'Carrot',
                        'Corn',
                        'Cotton',
                        'Flax',
                        'Onion',
                        'Peanuts',
                        'Pecan Tree',
                        'Potato',
                        'Rice',
                        'Soybean',
                        'Sugarbeets',
                        'Sugarcane',
                        'Sunflower',
                        'Walnut Tree',
                        'Wheat']
    
    print(sand, silt, clay)
    
    ##creating and loading info into json which will be returned
    info = {}
    info['temperature'] = temperature
    info['city'] = city
    info['elevation'] = elevation
    info['weather'] = weather
    info['pressure'] = pressure
    info['visibility'] = visibility
    info['humidity'] = humidity
    info['day_precipitation'] = day_precipitation
    info['monthly_preciptation'] = monthly_precipitation
    info['phH20'] = phH20
    info['soil_type'] = soil_list[compute_soil_type(sand, silt, clay)]
    info['climate'] = climate_list[find_climate(latitude)]
    best_crop_predictions = bestCropModel.predict(np.asarray([temperature, find_climate(latitude), compute_soil_type(sand, silt, clay), phH20, monthly_precipitation, humidity]).reshape(1, 6, 1))
    info['best_crop'] = best_crop_labels[np.argmax(best_crop_predictions)]
    print(np.argmax(bestCropModel.predict(np.asarray([temperature, find_climate(latitude), compute_soil_type(sand, silt, clay), phH20, monthly_precipitation, humidity]).reshape(1, 6, 1))))
    print(info['best_crop'])
    
    return jsonify(info)


##route that takes post request file of an image-- reshapes it and passes
##it through a crop disease detection model
@app.route('/cropImageModel', methods=['POST'])
def cropImage():
    labels = ['Apple__Apple_scab',
               'Apple_Black_rot', 
               'Apple_Cedar_apple_rust', 
               'Apple_healthy', 
               'Blueberry_healthy', 
               'Cherry(including_sour)__healthy', 
               'Cherry(including_sour)__Powdery_mildew', 
               'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 
               'Corn(maize)__Common_rust', 'Corn_(maize)__healthy',
                'Corn(maize)__Northern_Leaf_Blight', 
                'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 
                'Grape__healthy', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 
                'Orange__Haunglongbing(Citrus_greening)', 
                'Peach__Bacterial_spot', 
                'Peach_healthy', 
                'Pepper,_bell_Bacterial_spot', 
                'Pepper,_bell_healthy',
                 'Potato_Early_blight', 
                 'Potato_healthy', 
                 'Potato_Late_blight', 
                 'Raspberry_healthy', 
                 'Soybean_healthy', 
                 'Squash_Powdery_mildew', 
                 'Strawberry_healthy', 
                 'Strawberry_Leaf_scorch', 
                 'Tomato_Bacterial_spot', 
                 'Tomato_Early_blight', 
                 'Tomato_healthy', 
                 'Tomato_Late_blight', 
                 'Tomato_Leaf_Mold', 
                 'Tomato_Septoria_leaf_spot', 
                 'Tomato_Spider_mites Two-spotted_spider_mite', 
                 'Tomato_Target_Spot', 
                 'Tomato_Tomato_mosaic_virus', 
                 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']
    try:
        data = request.files['file'].read()
        image = Image.open(io.BytesIO(data)).resize((256, 256))
        image_arr = np.asarray(image, dtype=np.float16)
        print('predicting')
        print(np.argmax(cropImageModel.predict(image_arr.reshape(1, 256, 256, 3))))
        print(labels[np.argmax(cropImageModel.predict(image_arr.reshape(1, 256, 256, 3)))])
    except Exception as e:
        print(e)
    output = {}
    disease = ' '.join(labels[np.argmax(cropImageModel.predict(image_arr.reshape(1, 256, 256, 3)))].split('_'))
    return disease


##route that takes post request file of an image-- reshapes it and passes
##it through a weed disease detection model
@app.route('/weedImageModel', methods=['POST'])
def weedImageModel():
    labels = ['Broadleaf_weed', 'Grass_weed']
    try:
        data = request.files['file'].read()
        image = Image.open(io.BytesIO(data)).resize((256, 256))
        image_arr = np.asarray(image, dtype=np.float16)
        print('predicting')
        print(np.argmax(weedImageModel.predict(image_arr.reshape(1, 256, 256, 3))))
        print(labels[np.argmax(weedImageModel.predict(image_arr.reshape(1, 256, 256, 3)))])
    except Exception as e:
        print(e)
    output = {}
    disease = ' '.join(labels[np.argmax(weedImageModel.predict(image_arr.reshape(1, 256, 256, 3)))].split('_'))
    return disease

##finds soil based on different soil types
def compute_soil_type(sand, silt, clay):
    if (sand>= 35 and sand <= 45) and (silt >= 35 and silt <= 45) and (clay >= 16 and clay <= 24):
        return 3
    else:
        return np.argmax([clay, silt, sand])

    
##finds climate based on latitude     
def find_climate(latitude):
    latitude = abs(latitude)
    if latitude >= 0 and latitude <= 23.5:
        return 0
    elif latitude > 23.5 and latitude <= 66.5:
        return 1
    else:
        return 2
    

if __name__ == '__main__':
    cropImageModel = load_model('CropDiseaseDetection.h5')
    weedImageModel = load_model('WeedDetection.h5')
    bestCropModel = load_model('bestCrop.h5')
    app.run(host='127.0.0.1', port=8080, debug=True)