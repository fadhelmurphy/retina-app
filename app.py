import os
import imghdr
from imagepro import preProcess,postProcess,removeBoarder,skeleton
from flask import Flask,render_template,request,redirect,flash,url_for, \
    send_from_directory, abort
from werkzeug.utils import secure_filename
app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.tif']
app.config['UPLOAD_PATH'] = 'uploads/input'

def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.route('/', methods=['GET', 'POST'])
def index():
   if request.method == 'GET':
      files = os.listdir(app.config['UPLOAD_PATH'])
      return render_template('page/home.html', files=files)
   else:
      
      uploaded_file = request.files['file']
      i = uploaded_file.filename.split('_')[0]
      filename = secure_filename(uploaded_file.filename)
      if filename != '':
         # file_ext = os.path.splitext(filename)[1]
         # if file_ext not in app.config['UPLOAD_EXTENSIONS'] 
         # or file_ext != validate_image(uploaded_file.stream):
            # abort(400)
         uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
      preProcess(i)
      postProcess(i)
      removeBoarder(i)
      skeleton(i)
      # uploaded_file = request.files['file']
      # filename = secure_filename(uploaded_file.filename)
      # if filename != '':
      #    file_ext = os.path.splitext(filename)[1]
      #    # if file_ext not in app.config['UPLOAD_EXTENSIONS'] 
      #    # or file_ext != validate_image(uploaded_file.stream):
      #       # abort(400)
      #    # uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
      return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


@app.route('/login')
def login():
   return render_template('page/login.html')

if __name__ == '__main__':
   app.run(debug=True)