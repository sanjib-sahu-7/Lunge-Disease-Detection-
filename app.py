from flask import Flask, render_template, request, redirect, url_for, session
from pymongo import MongoClient
import os
from werkzeug.utils import secure_filename
import base64
from bson import ObjectId
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = '2e69e3cf3e2ae34c7b2a4718268a1ea7'  # Replace this with your actual secret key

# MongoDB connection string
# Replace <password> with your actual password
client = MongoClient("mongodb+srv://user1:12345@atlascluster.nq1qhdg.mongodb.net/?retryWrites=true&w=majority&ssl=true")

# Select the database to use
db = client["lunge"]
 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = db.users.find_one({'email': email, 'password': password})
        if user:
            session['email'] = email  # Store the email in the session
            return redirect(url_for('profile', message='Signin successful!'))
        else:
            return redirect(url_for('signin', message='Invalid email or password. Please try again.'))

    # Get the message from the query parameters
    message = request.args.get('message', None)
    return render_template('signin.html', message=message)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Extract name, email, and password from the form
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        # Check if email already exists in the database
        existing_user = db.users.find_one({'email': email})
        if existing_user:
            # Email already exists, return a message to the user
            return render_template('signup.html', message="Email already exists! Try again with a different email.")

        # Insert new user into the database
        db.users.insert_one({'name': name, 'email': email, 'password': password})
        # Signup successful, redirect to the sign-in page
        return redirect(url_for('signin', message='Signup successful!'))

    return render_template('signup.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    # Check if the user is logged in
    if 'email' in session:
        # Get the email of the logged-in user from the session
        email = session['email']
        
        # Query the database to fetch user data
        user = db.users.find_one({'email': email})

        # Check if user data exists
        if user:
            message = request.args.get('message', None)
            # User data exists, pass it to the template
            return render_template('profile.html', user=user, message=message)
    
    # Redirect to the sign-in page if the user is not logged in
    return redirect(url_for('signin'))

@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if request.method == 'POST':
        # Get the email of the logged-in user from the session
        email = session['email']
        
        # Get the form data
        name = request.form['name']
        address = request.form['address']
        profile_image = request.files['profile_image']
        
        # Convert the image file to base64 encoding
        if profile_image:
            profile_image_data = base64.b64encode(profile_image.read()).decode('utf-8')
        else:
            profile_image_data = None
        
        # Update user data in the database
        db.users.update_one({'email': email}, {'$set': {'name': name, 'address': address, 'profile_image': profile_image_data}})
        
        # Redirect to the profile page with a success message
        return redirect(url_for('profile', message='Profile updated successfully!'))

    # Render the update profile page
    return render_template('update_profile.html')

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    # Redirect to the sign-in page
    return redirect(url_for('signin'))

@app.route('/add_patient_details', methods=['GET', 'POST'])
def add_patient_details():
    if 'email' in session:
        user_email = session['email']
        if request.method == 'POST':
            name = request.form['name']
            age = request.form['age']
            gender = request.form['gender']
            disease = request.form['disease']
            disease_image = request.files['diseaseImage']

            save_patient_details(name, age, gender, disease, disease_image, user_email)

            return redirect(url_for('profile', message='Patient details added successfully!'))

        return render_template('add_patient_details.html')

    return redirect(url_for('signin'))

def save_patient_details(name, age, gender, disease, disease_image, user_email):
    # Convert the image file to base64 encoding
    if disease_image:
        disease_image_data = base64.b64encode(disease_image.read()).decode('utf-8')
    else:
        disease_image_data = None
        
    # Save patient details into the database
    patient_data = {
        'name': name,
        'age': age,
        'gender': gender,
        'disease': disease,
        'user_email': user_email,
        'disease_image':disease_image_data
    }
    db.patients.insert_one(patient_data)

@app.route('/update_patient/<patient_id>', methods=['POST'])
def update_patient_details_route(patient_id):
    # Extract updated patient details from the form
    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']
    disease = request.form['disease']
    disease_image = request.files['diseaseImage']
    print("Patient ID:", patient_id) 
    # Update the patient details in the database
    update_patient_details(patient_id, name, age, gender, disease,disease_image)
    # Optionally handle the update of the disease image here
    # Redirect to the view patients page or any other appropriate page
    return redirect(url_for('view_patients'))

@app.route('/update_patient/<patient_id>', methods=['GET'])
def update_patient(patient_id):
    print(patient_id)
    # Fetch the existing patient details from the database using patient_id
    patient = fetch_patient(patient_id)
    print(patient)
    # Pass the patient details to the update_patient template
    return render_template('update_patient.html', patient=patient)

@app.route('/view_patients')
def view_patients():
    # Check if the user is logged in
    if 'email' in session:
        # Get the email of the logged-in user from the session
        user_email = session['email']
        # Fetch patient details from the database
        patients = fetch_patient_details(user_email)
        # Render the view_patients.html template with patient details
        return render_template('view_patients.html', patients=patients)
    # Redirect to the sign-in page if the user is not logged in
    return redirect(url_for('signin'))

def fetch_patient_details(user_email):
    # Query the database to fetch patient details
    patients = db.patients.find({'user_email': user_email})
    # Convert the cursor to a list of dictionaries
    patients_list = list(patients)
    return patients_list

def fetch_patient(patient_id):
    # Convert the string patient_id to an ObjectId instance
    patient_object_id = ObjectId(patient_id)
    
    # Query the database to fetch patient details
    patient = db.patients.find_one({'_id': patient_object_id})
    
    # Return the patient details or None if not found
    return patient

def update_patient_details(patient_id, name, age, gender, disease,disease_image):
    if disease_image:
        disease_image_data = base64.b64encode(disease_image.read()).decode('utf-8')
    else:
        disease_image_data = None
    print(patient_id)    
    # Update the patient details in the database
    db.patients.update_one(
       {"_id": ObjectId(patient_id)},
       {"$set": {
           "name": name,
           "age": age,
           "gender": gender,
           "disease": disease,
           "disease_image": disease_image_data
       }}
      )
    
# Load the trained model
model = load_model('mnist_cnn_model.h5')
class_labels = ['Healthy', 'Pneumonia', 'COVID-19']

# Define a function to preprocess the input image
def preprocess_image(image):
    # Resize the image to match the input shape of the model (28x28)
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Normalize the pixel values to the range [0, 1]
    image = np.array(image) / 255.0
    # Expand the dimensions to match the model input shape
    image = np.expand_dims(image, axis=0)
    # Reshape the image to match the input shape of the model (batch size, height, width, channels)
    image = np.reshape(image, (1, 28, 28, 1))
    return image

# Function to perform disease detection using the loaded model
def detect_disease(image):
    # Preprocess the image (if needed)
    preprocessed_image = preprocess_image(image)

    # Perform inference using your model
    predicted_class = model.predict(preprocessed_image)

    print("Predicted class:", predicted_class)  # Print the predicted class for debugging

    # Convert the predicted class to an integer (if needed)
    if isinstance(predicted_class, np.ndarray):
        if predicted_class.size == 1:
            predicted_class = int(predicted_class[0])
        else:
            predicted_class = predicted_class.argmax()  # Choose the index of the highest probability
    else:
        predicted_class = int(predicted_class)

    # Map the predicted class to the corresponding disease label
    predicted_disease = class_labels[predicted_class]

    return predicted_disease

# Define a route for the disease detection page
@app.route('/disease_detection', methods=['GET', 'POST'])
def disease_detection():
    if request.method == 'POST':
        # Get the uploaded image file from the request
        uploaded_file = request.files['image']

        if uploaded_file.filename != '':
            # Read the image file
            image = Image.open(io.BytesIO(uploaded_file.read()))

            # Perform disease detection on the image
            predicted_disease = detect_disease(image)

            return render_template('disease_detection_result.html', predicted_disease=predicted_disease)

    return render_template('disease_detection.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
