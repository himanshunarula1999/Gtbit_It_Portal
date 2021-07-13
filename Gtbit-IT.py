from flask import Flask, render_template, request, session, redirect, url_for
from flask_mysqldb import MySQL,MySQLdb
import bcrypt 
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_mail import Mail
from werkzeug.utils import secure_filename
import json
import os
import time
import math
import warnings
warnings.filterwarnings('ignore')
import re
import nltk
import matplotlib.pyplot as plt

import pandas as pd
import collections

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
import joblib
tvec = TfidfVectorizer()
lr = LogisticRegression()


# test_data.head()
comparr = ['amazon' , 'josh' , 'zs' , 'new' , 'hashed' , 'traveloka' , 'aws', 'newgen', 'compro'  ]
# predictions = []
resarr = []

for i in range(len(comparr)):
    zero = int(0)
    one = int(0)
    test_data=pd.read_csv( comparr[i] + ".csv" , encoding='ISO-8859–1')
    model = joblib.load('model_predict.joblib')
    X_test = test_data['Review']
    # Y_test = test_data['Liked']

    predictions = model.predict(X_test)

    # from sklearn.metrics import confusion_matrix
    # confusion_matrix(predictions,test_data['Liked']) #Good model

    occurences = collections.Counter(predictions)
    zero = occurences[0]
    one = occurences[1]
    sum = one + zero
    res = (one/sum) * 100
    resarr.append(int(res))


with open('config.json', 'r') as c:
    params = json.load(c)["params"]


local_server = True

app = Flask(__name__)
app.secret_key = "caircocoders-ednalan-2020"
 
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'gtbit'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# app.secret_key = 'super-secret-key'
app.config['UPLOAD_FOLDER'] = params['upload_location']
app.config.update(
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = '465',
    MAIL_USE_SSL = True,
    MAIL_USERNAME = params['gmail-user'],
    MAIL_PASSWORD=  params['gmail-password']
)

mail = Mail(app)


if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/gtbit'
db = SQLAlchemy(app)


class Review(db.Model):
   
    sno = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    review = db.Column(db.String(1500), nullable=False)
    perks = db.Column(db.String(1000), nullable=False)
    ctc = db.Column(db.String(1000), nullable=False)
    # date = db.Column(db.String(12), nullable=True)
    email = db.Column(db.String(150), nullable=False)

class Contacts(db.Model):
   
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    phone_num = db.Column(db.String(12), nullable=False)
    msg = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)
    email = db.Column(db.String(20), nullable=False)


class Posts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    slug = db.Column(db.String(21), nullable=False)
    description = db.Column(db.String(120), nullable=False)
    content = db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(20), nullable=False)
    profile = db.Column(db.String(50), nullable=False)
    salary = db.Column(db.String(12), nullable=True)
    roles = db.Column(db.String(50), nullable=False)
    skills = db.Column(db.String(50), nullable=False)
    tagline = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)
    img_file = db.Column(db.String(12), nullable=True)
    button = db.Column(db.String(12), nullable=True)
    referral = db.Column(db.String(12), nullable=True)

    
@app.route("/")
def home():
    post = Posts.query.filter_by().all()
    return render_template('login1.html',params=params, post=post)
    
@app.route("/index")
def index():
    return render_template('index.html', params=params)

@app.route("/insights")
def insights():
    return render_template('insights.html', params=params)  

@app.route("/review")
def review():
    return render_template("review.html",your_list = resarr, params=params)

@app.route("/writereview", methods = ['GET', 'POST'] )
def writereview():
    if(request.method=='POST'):
        '''Add entry to the database'''
        company = request.form.get('company')
        name = request.form.get('name')
        email = request.form.get('email')
        review = request.form.get('review')
        perks = request.form.get('perks')
        ctc = request.form.get('ctc')
        entry = Review(company = company, name=name, review=review,email = email, ctc=ctc, perks = perks )
        db.session.add(entry)
        time.sleep(5)
        db.session.commit()

    return render_template("writereview.html", params=params)    

@app.route("/register",methods = ["GET","POST"])
def register():
    if request.method == 'GET':
        return render_template("login1.html", params=params)
    else:
        name = request.form['name']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt())
 
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s,%s,%s)",(name,email,hash_password,))
        mysql.connection.commit()
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        time.sleep(3)
        return redirect(url_for('login1'))

@app.route('/login1',methods=["GET","POST"])
def login1():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
 
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM users WHERE email=%s",(email,))
        user = curl.fetchone()
        curl.close()
 
        if len(user) > 0:
            if bcrypt.hashpw(password, user["password"].encode('utf-8')) == user["password"].encode('utf-8'):
                session['name'] = user['name']
                session['email'] = user['email']
                return redirect(url_for('index'))
            else:
                return "Error password and email not match"
        else:
            return "Error user not found"
    else:
        return render_template("login1.html", params = params)     

@app.route('/logout1')
def logout1():
    session.clear()
    return render_template("login1.html", params=params)           


@app.route("/dashboard", methods=['GET', 'POST'])
def dashboard():

    if ('user' in session and session['user'] == params['admin_user']):
        posts = Posts.query.all()
        return render_template('dashboard.html', params=params, posts = posts)


    if request.method=='POST':
        username = request.form.get('uname')
        userpass = request.form.get('pass')
        if (username == params['admin_user'] and userpass == params['admin_password']):
            #set the session variable
            session['user'] = username
            posts = Posts.query.all()
            return render_template('dashboard.html', params=params, posts = posts)

    return render_template('login.html', params=params)



@app.route("/edit/<string:sno>", methods = ['GET', 'POST'])
def edit(sno):
    if ('user' in session and session['user'] == params['admin_user']):
        if request.method == 'POST':
            box_title = request.form.get('title')
            tline = request.form.get('tline')
            slug = request.form.get('slug')
            desc = request.form.get('description')
            content = request.form.get('content')
            location = request.form.get('location')
            profile = request.form.get('profile')
            salary = request.form.get('salary')
            roles = request.form.get('roles')
            skills = request.form.get('skills')
            img_file = request.form.get('img_file')
            date = datetime.now()

            if sno=='0':
                post = Posts(title=box_title, slug=slug, content=content, tagline=tline, description=desc, location=location,profile=profile,salary=salary,roles=roles,skills=skills,img_file=img_file, date=date)
                db.session.add(post)
                db.session.commit()
            else:
                post = Posts.query.filter_by(sno=sno).first()
                post.title = box_title
                post.slug = slug
                post.description = desc
                post.location= location
                post.profile = profile
                post.roles = roles
                post.salary = salary
                post.skills = skills
                post.content = content
                post.tagline = tline
                post.img_file = img_file
                post.date = date
                db.session.commit()
                return redirect('/edit/'+sno)
        post = Posts.query.filter_by(sno=sno).first()
        return render_template('edit.html', params=params, post=post, sno=sno)



@app.route("/fresher")
def fresher():
    post = Posts.query.filter_by().all()
    last = math.ceil(len(post)/int(params['no_of_posts']))
    #[0: params['no_of_posts']]
    #posts = posts[]
    page = request.args.get('page')
    if(not str(page).isnumeric()):
        page = 1
    page= int(page)
    post = post[(page-1)*int(params['no_of_posts']): (page-1)*int(params['no_of_posts'])+ int(params['no_of_posts'])]
    #Pagination Logic
    #First
    if (page==1):
        prev = "#"
        next = "/fresher?page="+ str(page+1)
    elif(page==last):
        prev = "/fresher?page=" + str(page - 1)
        next = "#"
    else:
        prev = "/fresher?page=" + str(page - 1)
        next = "/fresher?page=" + str(page + 1)

    return render_template('fresher.html', params=params, post=post, prev=prev, next=next)        
   
@app.route("/experience")
def experience():
    post = Posts.query.filter_by().all()
    last = math.ceil(len(post)/int(params['no_of_posts']))
    #[0: params['no_of_posts']]
    #posts = posts[]
    page = request.args.get('page')
    if(not str(page).isnumeric()):
        page = 1
    page= int(page)
    post = post[(page-1)*int(params['no_of_posts']): (page-1)*int(params['no_of_posts'])+ int(params['no_of_posts'])]
    #Pagination Logic
    #First
    if (page==1):
        prev = "#"
        next = "/fresher?page="+ str(page+1)
    elif(page==last):
        prev = "/fresher?page=" + str(page - 1)
        next = "#"
    else:
        prev = "/fresher?page=" + str(page - 1)
        next = "/fresher?page=" + str(page + 1)

    return render_template('experience.html', params=params, post=post, prev=prev, next=next)

@app.route("/internship")
def internship():
    post = Posts.query.filter_by().all()
    last = math.ceil(len(post)/int(params['no_of_posts']))
    #[0: params['no_of_posts']]
    #posts = posts[]
    page = request.args.get('page')
    if(not str(page).isnumeric()):
        page = 1
    page= int(page)
    post = post[(page-1)*int(params['no_of_posts']): (page-1)*int(params['no_of_posts'])+ int(params['no_of_posts'])]
    #Pagination Logic
    #First
    if (page==1):
        prev = "#"
        next = "/fresher?page="+ str(page+1)
    elif(page==last):
        prev = "/fresher?page=" + str(page - 1)
        next = "#"
    else:
        prev = "/fresher?page=" + str(page - 1)
        next = "/fresher?page=" + str(page + 1)

    return render_template('internship.html', params=params, post=post, prev=prev, next=next)


@app.route("/post/<string:post_slug>", methods= ['GET'])
def post_route(post_slug):
        post = Posts.query.filter_by(slug=post_slug).first()
        
        return render_template('post.html', params=params,post=post)

@app.route("/uploader", methods = ['GET', 'POST'])
def uploader():
    if ('user' in session and session['user'] == params['admin_user']):
        if (request.method == 'POST'):
            f= request.files['file1']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename) ))
            return "Uploaded successfully"

@app.route("/logout")
def logout():
    session.pop('user')
    return redirect('/dashboard')


@app.route("/delete/<string:sno>", methods = ['GET', 'POST'])
def delete(sno):
    if ('user' in session and session['user'] == params['admin_user']):
        post = Posts.query.filter_by(sno=sno).first()
        db.session.delete(post)
        db.session.commit()
    return redirect('/dashboard')
    
        
@app.route("/contact", methods = ['GET', 'POST'])
def contact():
    if(request.method=='POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        entry = Contacts(name=name, phone_num = phone, msg = message, date= datetime.now(),email = email )
        db.session.add(entry)
        db.session.commit()
        mail.send_message('New message from ' + name,
                          sender=email,
                          recipients = [params['gmail-user']],
                          body = message + "\n" + phone
                          )
    return render_template('contact.html',params=params)


app.run(debug=True)




