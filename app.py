from flask import Flask, render_template, request,jsonify
import pandas as pd
import pickle
import sklearn
import joblib

# file = open("rate_predict.pkl","rb")
# model = pickle.load(file)
label_encoder = joblib.load('required files\label_encoder.joblib')
label_encoder_cuisines = joblib.load('required files\label_encoder_cuisines.joblib')
label_encoder_location = joblib.load('required files\label_encoder_location.joblib')   
loaded_model = joblib.load('required files\model.joblib')



app = Flask(__name__, static_url_path='/static')

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predictor", methods=["GET", "POST"])
def predictor():

    if request.method == "POST":
        # Input features
        # location_input = 'Banashankari'
        # cuisine_input = 'North Indian'
        # approx_cost_input = 1000  # Enter the approximate cost here
        # online_order_input = "Yes"
        # book_table_input = "Yes"

        location_input = request.form['Location']
        cuisine_input = request.form['Cuisines']
        approx_cost_input = int(request.form['approx_cost'])
        online_order_input = request.form['online_order']
        book_table_input = request.form['book_table']
        

        # print(location_input)
        # print(cuisine_input)
        # print(type(cuisine_input))
        # print(approx_cost_input)
        # print(type(approx_cost_input))
        # print(online_order_input)
        # print(type(online_order_input))
        # print(book_table_input)

       

        # Transform input features using the loaded LabelEncoder
        location_encoded_input = label_encoder_location.transform([location_input])[0]
        cuisines_encoded_input = label_encoder_cuisines.transform([cuisine_input])[0]
        # location_encoded_input = label_encoder.transform([location_input])[0] if location_input in label_encoder.classes_ else len(label_encoder.classes_)
        # cuisines_encoded_input = label_encoder.transform([cuisine_input])[0] if cuisine_input in label_encoder.classes_ else len(label_encoder.classes_)
        online_order_encoded_input = label_encoder.transform([online_order_input])[0]
        book_table_encoded_input = label_encoder.transform([book_table_input])[0]

        # Make prediction using the loaded model
        input_features = [[location_encoded_input, cuisines_encoded_input, approx_cost_input, 
                        online_order_encoded_input, book_table_encoded_input]]
        predicted_rating = loaded_model.predict(input_features)
        # print(type(predicted_rating[0]))
        return render_template("predictor.html", prediction_text=format(predicted_rating[0],'.2f'))
    
    return render_template("predictor.html")


@app.route("/new")
def new():
    return render_template("new.html")



if __name__ == "__main__":
    app.run(debug = True)

