from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory, Response
import boto3, botocore
import json
import os, shutil
from werkzeug.exceptions import abort
from werkzeug.wsgi import FileWrapper
from io import BytesIO
from base64 import b64encode, urlsafe_b64encode
import uuid
from datetime import date
import functions

sync_path = 'data/new/'
arch_path = 'data/archive/'
test_file = 'common_voice_en_38497561.mp3'

app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'
app.config['UPLOAD_FOLDER'] = sync_path

### Boto3 setup
boto3.setup_default_session(profile_name='AWSIntroTraining-408122842185')

if functions.check_login():
    print("AWS sso token is actief.")
else:
    os._exit(100)

s3 = boto3.resource('s3')
bucketname = 'humainr-aws-intro-bucket'
bucket = s3.Bucket(bucketname)

db = boto3.resource('dynamodb')
tablename = 'AWSIntroTable'
table = db.Table(tablename)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    items = table.scan()
    return render_template('index.html', items=items['Items'])

@app.route('/<string:id>')
def details(id):
    obj = s3.Object(bucket_name=bucketname, key=id)
    response = obj.get()
    audio_bytes = response['Body']
    w = audio_bytes.read()
    q = b64encode(w).decode("utf-8")
    items = table.scan()
    for item in items['Items']:
        if item['id'] == id:
            print(item)
            return render_template('details.html', items=item, w=q)

@app.route('/control', methods=('GET', 'POST'))
def control():
    if request.method == 'POST':
        if request.form.get('verwerk') == 'Verwerk':
            flash('Bestanden zijn verwerkt.')
            for entry in os.listdir(sync_path):
                if entry.endswith('.mp3'):
                    uuid_token = str(uuid.uuid4())
                    bucket.upload_file(sync_path + entry, uuid_token)
                    shutil.move(sync_path + entry, arch_path + entry)
                    try:
                        table.put_item(
                            Item={
                                'id': uuid_token,
                                'file_name': entry,
                                'framerate': '32000',
                                'transcript': '',
                                'create_date': 'Aanmaak datum bestand',
                                'upload_date': str(date.today()),
                                'lenght': 'Lengte van het fragment',
                                'description': 'Standard omschrijving',
                            }
                        )
                    except:
                        flash('Opslaan in DB is gefaald.')
            pass # do something

        elif  request.form.get('upload') == 'Upload':
            print('upload geklicked')
            file = request.files['file']
            if file.filename == '':
              flash('No selected file')
            if file:
                flash('Bestand ' + file.filename + ' is opgeslagen.')
                filename = file.filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  
            pass # do something else

        else:
            flash('Something wrong!')
            pass # unknown

    elif request.method == 'GET':
        return render_template('control.html')
    
    return render_template('control.html')

@app.route('/reset')
def reset():
    for obj in bucket.objects.all():
        obj.delete()
        print(obj)

    scan = table.scan(
        ProjectionExpression='#k',
        ExpressionAttributeNames={
            '#k': 'id'
        }
    )
    with table.batch_writer() as batch:
        for record in scan['Items']:
            batch.delete_item(Key=record)
            #print(record)

    return render_template('reset.html')

