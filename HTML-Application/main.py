import os
from app import app
import model
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from waitress import serve


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
        if 'file' not in request.files:
                flash('No file found')
                return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
                flash('No image selected for uploading')
                return redirect(request.url)
        if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                result=model.predict(app.config['UPLOAD_FOLDER']+filename)
                flash(result)
                return render_template('upload.html', filename=filename)
                
        else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    #app.run(host='192.168.1.16',port=8080)
    serve(app, host='192.168.42.218',port=8080)
    #os.remove(app.config['UPLOAD_FOLDER'])