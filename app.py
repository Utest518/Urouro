from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
import cv2
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pytz
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
        model = tf.keras.models.load_model('urine_autoencoder_model.keras')

# データベースの設定
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'mysecret'

db = SQLAlchemy(app)
migrate = Migrate(app, db)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# データベースモデルの定義
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
    status = db.Column(db.String(20), nullable=False)
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
    today = datetime.today().date()  # 日付だけで比較するよう修正
    latest_result = Result.query.filter_by(user_id=current_user.id).filter(Result.date >= today).order_by(Result.date.desc()).first()
    
    print(f"Latest result: {latest_result}")  # デバッグ用ログ
    
    health_advice = ""
    if latest_result:
        print(f"Latest result status: {latest_result.status}, Latest result date: {latest_result.date}")  # デバッグ用ログ
        status = latest_result.status
        if status == "正常":
            health_advice = "健康な尿です。水分をしっかり摂りましょう。"
        else:
            health_advice = "異常な色です。医師の診察を受けてください。"
    
    return render_template('home.html', latest_result=latest_result, health_advice=health_advice)

@app.route('/upload')
@login_required
def upload():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    file = request.files['image']
    if file:
        try:
            image = Image.open(io.BytesIO(file.read()))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            load_model()
            processed_image = preprocess_image(image)
            reconstructed_image = model.predict(processed_image)
            mse = np.mean(np.power(processed_image - reconstructed_image, 2), axis=(1, 2, 3))
            
            threshold = 0.01  # 訓練データに基づいて設定
            status = "異常" if mse > threshold else "正常"
            
            # 日本のタイムゾーンを使用して現在時刻を取得
            jst = pytz.timezone('Asia/Tokyo')
            current_time = datetime.now(jst)

            # 結果をデータベースに保存
            new_result = Result(status=status, user_id=current_user.id, date=current_time)
            db.session.add(new_result)
            db.session.commit()

            result = {
                'status': status
            }
            
            return jsonify(result)
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Processing error'})
    return jsonify({'error': 'No file uploaded'})

def preprocess_image(image):
    img_size = 128
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/result')
@login_required
def result():
    return render_template('result.html')

@app.route('/history')
@login_required
def history():
    results = Result.query.filter_by(user_id=current_user.id).order_by(Result.date.desc()).all()
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
