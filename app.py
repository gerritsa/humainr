from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory, Response
import boto3
import os
from werkzeug.exceptions import abort
from werkzeug.wsgi import FileWrapper
import functions


### Configuratie voor de applicatie
sync_path = 'data/new/'
arch_path = 'data/archive/'
test_file = 'common_voice_en_38497561.mp3'

### Configuratie voor Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'
app.config['UPLOAD_FOLDER'] = sync_path

### Configuratie voor AWS
boto3.setup_default_session(profile_name='AWSIntroTraining-408122842185')
bucketname = 'humainr-aws-intro-bucket'
tablename = 'AWSIntroTable'

### Check of de SSO token actief is, zoniet stop de applicatie.
if not functions.check_login(): os._exit(1)

### Activeer de gebruikte clients van AWS
s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
db = boto3.resource('dynamodb')
bucket = s3.Bucket(bucketname)
table = db.Table(tablename)


### Route voor de index pagina
@app.route('/')
def index():
    items = functions.load_records(table)
    return render_template('index.html', items=items['Items'])

### Route voor het pagina icoon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path + "/static", 'favicon.ico', mimetype='image/vnd.microsoft.icon')

### Route voor de detail pagina
@app.route('/<string:id>')
def details(id):
    item = functions.load_record(table, id)
    file = functions.load_b64file(s3, bucketname, id)
    return render_template('details.html', items=item, audiofile=file)

### Route voor de control pagina
@app.route('/control', methods=('GET', 'POST'))
def control():
    if request.method == 'POST':
        if request.form.get('verwerk') == 'Verwerk':
            flash('Bestanden zijn verwerkt.')
            functions.process_server(s3, bucketname, bucket, table, sync_path, arch_path)
        elif request.form.get('reset') == 'Reset':
            flash('Weet u het zeker? <a href="/reset" class="alert-link">klik hier</a>')

            return redirect(url_for('control'))
        elif  request.form.get('upload') == 'Upload':
            file = request.files['file']
            if file.filename == '':
                flash('U moet een file selecteren voor de upload!')
            if file:
                filename = file.filename
                try:
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                except: 
                    flash('Upload mislukt, probeer het later opnieuw.')
                    return render_template('control.html')
                flash('Bestand ' + file.filename + ' is opgeslagen op de server.')

    elif request.method == 'GET':
        pass ### Boiler plate code voor toekomstige implementatie
    
    return render_template('control.html')

### Route voor de verborgen reset pagina
# Deze pagina zal alle gegevens resetten waardoor fouten tijdens het testen geen probleem zijn.
@app.route('/reset')
def reset():
    functions.reset_application(bucket, table)
    return render_template('reset.html')

### Route voor de debug pagina
# Deze pagina laat een overzicht van alle opgeslagen gegevens zien.
@app.route('/debug')
def debug():
    localfiles = functions.show_localfiles(sync_path)
    s3files = functions.show_s3files(s3_client, bucketname)
    records = functions.load_records(table)
    return render_template('debug.html', localfiles=localfiles, s3files=s3files['Contents'], records=records['Items'])
