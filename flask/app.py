from re import template
from flask import *
from flask_bootstrap import Bootstrap
import os
from werkzeug.utils import secure_filename
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2RGB
from MasterLib import detect_ripeness  # the image processing code

from datetime import datetime


app = Flask(__name__,
            static_url_path="",
            static_folder="static",
            template_folder="templates")
Bootstrap(app)
flaskfolder = os.getcwd()
uploadsfolder = os.path.join(flaskfolder, 'mysite', 'static', 'uploads')
app.config["UPLOAD_FOLDER"] = uploadsfolder
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/gallery/')
def gallery():
    uploads = os.listdir(uploadsfolder)
    return render_template('gallery.html', uploads=uploads)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        global img
        global r_state
        img = request.files.get("image")
        names = secure_filename(img.filename).split(".")
        now = datetime.now()
        filename = names[0]+"_" + str(now.year) + "_" + str(now.month) + "_" + str(
            now.day) + "_" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second) + "." + names[1]
        img.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        cvimg = imread(os.path.join(
            app.config["UPLOAD_FOLDER"], filename))  # open image
        ripeness_state, total_area, p_img = detect_ripeness(
            cvimg)  # analyze image
        if (ripeness_state == 0):
            r_state = "Ripe"
        elif (ripeness_state == 1):
            r_state = "Semi-Ripe"
        elif (ripeness_state == 2):
            r_state = "Unripe"
        elif (ripeness_state == 3):
            r_state = "None"
        p_img = cvtColor(p_img, COLOR_BGR2RGB)  # fix colorspace
        newname = "Rip" + names[0] + "." + names[1]
        imwrite(os.path.join(
            app.config["UPLOAD_FOLDER"], newname), p_img)
        return ("OKeee bee")
    return render_template("upload.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
