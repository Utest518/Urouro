import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import cv2
import numpy as np
from PIL import Image
import io
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pytz
from datetime import datetime
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv

load_dotenv() 
app = Flask(__name__)

# データベースの設定
uri = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")
if uri and uri.startswith("postgres://"):
    uri = uri.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "default_secret_key")

# 環境変数の確認のためにプリントステートメントを追加
print("SECRET_KEY:", app.config['SECRET_KEY'])
print("SQLALCHEMY_DATABASE_URI:", app.config['SQLALCHEMY_DATABASE_URI'])


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
    today = datetime.today().date()
    latest_result = Result.query.filter_by(user_id=current_user.id).filter(Result.date >= today).order_by(Result.date.desc()).first()
    
    health_advice = ""
    if latest_result:
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
            image = cv2.resize(image, (128, 128)).flatten() / 255.0

            # モデルの読み込み
            pca = joblib.load('pca_model.pkl')
            iso_forest = joblib.load('iso_forest_model.pkl')

            # 画像データの前処理
            image_pca = pca.transform([image])
            prediction = iso_forest.predict(image_pca)

            status = "正常" if prediction == 1 else "異常"

            # ステータスに基づいたメッセージの設定
            if status == "正常":
                message = ("おめでとうございます！検査結果は正常です。<br><br>"
                           "健康な尿の色は淡黄色から濃い黄色の範囲です。尿の色が透明や淡い黄色である場合、水分をしっかり摂取している証拠です。"
                           "日中や運動後には、十分な水分補給を心がけてください。また、朝一番の尿が濃い黄色であっても心配いりません。"
                           "これは体が夜間に尿を濃縮して水分を保持しようとするためです。<br><br>"
                           "引き続き、バランスの取れた食事と規則正しい生活を心がけ、健康を維持してください。")
            else:
                message = ("検査結果は異常を示しています。<br><br>"
                           "尿の色が赤色、茶色、または異常に濃い色である場合、何らかの健康問題が考えられます。"
                           "例えば、赤色の尿は血尿の可能性があり、腎臓や尿路に問題があるかもしれません。"
                           "茶色の尿は肝臓や胆道の問題を示している可能性があります。<br><br>"
                           "このような結果が出た場合は、速やかに医師の診察を受けることを強くお勧めします。"
                           "また、日々の生活習慣を見直し、適切な水分摂取やバランスの取れた食事を心がけましょう。")

            # 日本のタイムゾーンを使用して現在時刻を取得
            jst = pytz.timezone('Asia/Tokyo')
            current_time = datetime.now(jst)

            # 結果をデータベースに保存
            new_result = Result(status=status, user_id=current_user.id, date=current_time)
            db.session.add(new_result)
            db.session.commit()

            result = {
                'status': status,
                'message': message
            }
            
            return jsonify(result)
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Processing error'})
    return jsonify({'error': 'No file uploaded'})

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
