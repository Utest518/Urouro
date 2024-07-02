from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, IntegerField
from wtforms.validators import DataRequired, InputRequired, Length, ValidationError
from models import User

class LoginForm(FlaskForm):
    username = StringField('ユーザー名', validators=[InputRequired(message='このフィールドは必須です。'), Length(min=4, max=80, message='ユーザー名は4文字以上80文字以下で入力してください。')])
    password = PasswordField('パスワード', validators=[InputRequired(message='このフィールドは必須です。'), Length(min=4, max=200, message='パスワードは4文字以上200文字以下で入力してください。')])
    remember = BooleanField('ログイン状態を保持する')

class RegisterForm(FlaskForm):
    username = StringField('ユーザー名', validators=[DataRequired(message='このフィールドは必須です。')])
    password = PasswordField('パスワード', validators=[DataRequired(message='このフィールドは必須です。'), Length(min=6, message='パスワードは最低6文字必要です。')])
    birthdate = StringField('生年月日', validators=[DataRequired(message='このフィールドは必須です。')])
    height = IntegerField('身長(cm)', validators=[DataRequired(message='このフィールドは必須です。')])
    weight = IntegerField('体重(kg)', validators=[DataRequired(message='このフィールドは必須です。')])

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('ユーザー名は既に存在します。別のものを選んでください。')
