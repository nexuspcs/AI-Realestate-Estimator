from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('property_value_predictor.pkl','rb'))

if __name__ == "__main__":
    app.run(debug=True)

@app.route('/')
def home():
    return render_template('/templates/index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('/templates/index.html', evaluation_text='The property is worth approximately ${}.'.format(output))
    
