 
from flask import Flask, jsonify,request,make_response,json,render_template
from PIL import Image
import cv2
from math import floor
from facial import Facial
import numpy as np
import urllib.request
import requests
import os
import string
import random

#import cStringIO
app = Flask(__name__,template_folder='template')
#from producto import producto
#/apiv1/validar_imagen
@app.route('/',methods = ['POST','GET'])
def RouteRaiz():      
    return jsonify({"success" : "OK", "message": "--PN--"}) 

@app.route('/api/v1/validar_imagen',methods = ['POST'])
def ver():
    if request.method == 'POST':#METODO POST
      data1 = request.get_data()
      facial =  Facial(data1)
      facial.Validar()  
      return jsonify({"success" : facial.success , "message": facial.message , "url_micro" : facial.url_micro}) 

@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('error404.html'), 404


if __name__ == '__main__':
    #app.run(debug=True,port=4000)
    app.run()