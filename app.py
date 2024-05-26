from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
import cv2
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from foam_detection import detect_foam
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
from datetime import datetime

# GPUメモリの使用量を制限
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

# モデルの遅延読み込み
model = None
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('urine_color_model.h5')

# カテゴリ（ラベル）のリスト
categories = ["yellow", "coffee_milky", "light_pink", "red", "transparent_yellow", "white_milky", "brown"]

def preprocess_image(image):
    # 画像を128x128にリサイズ
    image = cv2.resize(image, (128, 128))
    
    # 画像の明るさとコントラストを調整
    alpha = 1.2  # コントラスト制御 (1.0-3.0)
    beta = 20    # 明るさ制御 (0-100)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 画像をnumpy配列に変換し、正規化
    image = image.astype('float32') / 255.0
    # バッチサイズの次元を追加
    image = np.expand_dims(image, axis=0)
    return image

def extract_center_region(image):
    # 画像の中央部分を切り取る
    h, w = image.shape[:2]
    center = image[h//4:3*h//4, w//4:3*w//4]
    return center

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'mysecret'

db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    birthdate = db.Column(db.Date, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Integer, nullable=False)
    results = db.relationship('Result', backref='user', lazy=True)

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    color = db.Column(db.String(20), nullable=False)
    foam = db.Column(db.String(20), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=200)])
    remember = BooleanField('Remember me')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=4, max=200)])
    birthdate = StringField('Birthdate', validators=[InputRequired()])
    height = StringField('Height (cm)', validators=[InputRequired()])
    weight = StringField('Weight (kg)', validators=[InputRequired()])

    def validate_username(self, username):
        existing_user = User.query.filter_by(username=username.data).first()
        if existing_user:
            raise ValidationError('Username already exists. Choose a different one.')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        birthdate = datetime.strptime(form.birthdate.data, '%Y%m%d').date()
        new_user = User(username=form.username.data, password=hashed_password,
                        birthdate=birthdate, height=form.height.data, weight=form.weight.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    latest_color = "黄"
    latest_foam = "なし"
    return render_template('home.html', latest_color=latest_color, latest_foam=latest_foam)

@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_image():
    file = request.files['image']
    if file:
        image = Image.open(io.BytesIO(file.read()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        urine_region = extract_center_region(image)
        load_model()
        processed_image = preprocess_image(urine_region)
        prediction = model.predict(processed_image)
        detected_color = categories[np.argmax(prediction)]
        foam_detected, result_image = detect_foam(urine_region)
        output_path = os.path.join('static', 'detected_foam_contours.png')
        if not os.path.exists('static'):
            os.makedirs('static')
        cv2.imwrite(output_path, result_image)
        result = {
            'detected_color': detected_color,
            'foam_detected': foam_detected,
            'image_path': url_for('static', filename='detected_foam_contours.png')
        }
        return jsonify(result)
    return jsonify({'error': 'No file uploaded'})

@app.route('/result')
@login_required
def result():
    return render_template('result.html')

@app.route('/history')
@login_required
def history():
    results = Result.query.order_by(Result.date.desc()).all()
    return render_template('history.html', results=results)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        user = User.query.get(current_user.id)
        user.birthdate = request.form['birthdate']
        user.height = request.form['height']
        user.weight = request.form['weight']
        db.session.commit()
        flash('Settings updated successfully.')
        return redirect(url_for('settings'))
    return render_template('settings.html', user=current_user)

if __name__ == '__main__':
    app.run(debug=True)