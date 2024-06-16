from flask import Flask, request,  flash, jsonify, Response, send_from_directory
from flask_cors import cross_origin
from Neural_netowrk_model import run_neural_network, returnColumn
import pandas as pd;
import io
import os
import json
import csv
from io import StringIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import base64

try:
    from base64 import encodebytes
except ImportError:  # 3+
    from base64 import encodestring as encodebytes
# this is for flask https://www.youtube.com/watch?v=zsYIw6RXjfM


UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route("/")
@cross_origin()
def home():
    return "Home"


@app.route("/nn/columns", methods=['GET', 'POST'])
def testingNNCol():
    print("faceeeeee")
    if request.method == 'POST':
        print(request.headers.get("file2"))
        
           
        col = returnColumn(request.data)
        print(col)
        data = [ {
        "columns" : col, 
        }]

        return data

@app.route("/nn/run", methods=['GET', 'POST'])
def runNN():

    if request.method == 'POST':

        target_area = request.headers.get("target")
        hidden_neurons = request.headers.get("neurons")
        epochs = request.headers.get("epochs")

    
        trn_sse, trn_mse, trn_rmse, photo, target_array, predictions, x = run_neural_network(target_area, int(hidden_neurons), int(epochs), request.data)
        
        output = io.BytesIO()

        FigureCanvas(photo).print_png(output)

        # dataStr = json.dumps(output.getvalue())
        base64EncodedStr = base64.b64encode(output.getvalue())

        r = []
        r2 = []
        for i in range(len(target_array.tolist())):
            re = {
                    "x": x.tolist()[i][0],
                    "y": target_array.tolist()[i][0],
                    "z": 200
                }
            r.append({
                        "x": x.tolist()[i][0],
                        "y": target_array.tolist()[i][0],
                        "z": 25
                    })
            r2.append({
                        "x": x.tolist()[i][0],
                        "y": predictions.tolist()[i][0],
                        "z": 150
                    })
            
       
        Response = {
            'trn_sse': trn_sse, 
            'trn_mse': trn_mse,
            'trn_rmse': trn_rmse,
            'photo':  base64EncodedStr.decode(),
            'target': target_array.tolist(), 
            'predictions': predictions.tolist(),
            'x': x.tolist(),
            'values': r,
            'values2': r2
        }

        return Response

    df = pd.read_csv('forestfires.csv')
    # return runNN('area', df)
    trn_sse, trn_mse, trn_rmse, photo, target_array, predictions = runNN('area', df)
    # b_array = bytearray(photo, encoding='utf8')
   
    output = io.BytesIO()

    FigureCanvas(photo).print_png(output)

    # dataStr = json.dumps(output.getvalue())
    base64EncodedStr = base64.b64encode(output.getvalue())
    print("heheheheheheheheheheh")
    Response = {
        'trn_sse': trn_sse, 
        'trn_mse': trn_mse,
        'trn_rmse': trn_rmse,
        'photo':  base64EncodedStr.decode(),
        'target': target_array.tolist(), 
        'predictions': predictions.tolist()
    }

    return Response






if __name__ == "__app__":
    app.run(host='0.0.0.0', port='5000', debug=True)