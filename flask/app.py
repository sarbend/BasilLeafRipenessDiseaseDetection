from re import template
from flask import *
from flask_bootstrap import Bootstrap
import os
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename


app = Flask(__name__,
            static_url_path="",
            static_folder="static",
            template_folder="templates")
Bootstrap(app)
app.config["SECRET_KEY"] = "SECRET_KEY"
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["MONGO_DBNAME"] = "gallery"
app.config["MONGO_URI"] = "mongodb://localhost:27017/gallery"

mongo = PyMongo(app)
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/gallery/')
def gallery():
    flaskfolder = os.getcwd()
    uploadsfolder = os.path.join(flaskfolder, 'static', 'uploads')
    uploads = os.listdir(uploadsfolder)
    print(uploads)
    return render_template('gallery.html', uploads=uploads)


@app.route("/upload/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        global img
        img = request.files.get("image")
        filename = secure_filename(img.filename)
        img.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return redirect(url_for("upload"))
    return render_template("upload.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
