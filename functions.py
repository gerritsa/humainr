### Standaard imports
import os, shutil, uuid
from base64 import b64encode
import datetime
from pydub.utils import mediainfo
from io import BytesIO

### Imports voor AWS
import boto3, botocore


### Transcibing trough ML
import torch
import torchaudio

### Voor transcribe Speech2Text
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

### Voor transcribe Wav2Vec2
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoProcessor, AutoModelForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder

### Controleer of de SSO token al is geauthoriseerd.
def check_login():
    sts = boto3.client('sts')
    try:
        sts.get_caller_identity()
        return True
    except botocore.exceptions.UnauthorizedSSOTokenError:
        print('We zijn niet geautoriseerd op AWS. Kijk of systeem wel goed geconfigureerd is.')
        return False
    except botocore.exceptions.SSOTokenLoadError:
        print('AWS SSO Token is niet actief. Graag aws sso login --profile \'AWSIntroTraining-408122842185\' uitvoeren.')
        return False
    
### Laad een bestand als bytestream
def load_s3file(s3, bucketname, id):
    obj = s3.Object(bucket_name=bucketname, key=id)
    response = obj.get()
    audio_bytes = response['Body'].read()
    return audio_bytes

### Zet het bytestreambestand om naar een base64 encoded string.
def load_b64file(s3, bucketname, id):
    b64file = b64encode(load_s3file(s3, bucketname, id)).decode("utf-8")
    return b64file

### Copieer een bestand naar de aangegevens s3 bucket
def upload_s3file(bucket, sync_path, entry, uuid_token):
    bucket.upload_file(sync_path + entry, uuid_token)
    return

### Haal alle gegevens op uit de AWS DynamoDB
def load_records(table):
    items = table.scan()
    return items

### Sla een record op in de AWS DynamoDB met de verwachte gegevens
def save_record(table, uuid_token, filename, samplerate, transcript, create_date, lenght):
    table.put_item(
        Item={
            'id': uuid_token,
            'file_name': filename,
            'framerate': samplerate,
            'transcript': transcript,
            'create_date': create_date,
            'upload_date': str(datetime.date.today()),
            'lenght': lenght,
            'description': 'work in progress',
        }
    )

### Haal een record op uit de AWS DynamoDB
def load_record(table, id):
    records = load_records(table)
    for record in records['Items']:
        if record['id'] == id:
            return record
    return

### Haal een overzicht van alle bestanden op uit de s3 bucket
def show_s3files(s3, bucketname):
    response = s3.list_objects_v2(
        Bucket=bucketname
    )
    if response['KeyCount'] == 0:
        response['Contents'] = {}
    return response

### Haal een overzicht van alle bestanden op uit de lokale server
def show_localfiles(sync_path):
    response = os.listdir(sync_path)
    return response


### Verwijder alle gegevens uit de applicatie zodat we een nieuwe start hebben.
def reset_application(bucket, table):
    for obj in bucket.objects.all():
        obj.delete()

    scan = table.scan(
        ProjectionExpression='#k',
        ExpressionAttributeNames={
            '#k': 'id'
        }
    )
    with table.batch_writer() as batch:
        for record in scan['Items']:
            batch.delete_item(Key=record)
    return

### Verwerk alle klaarstaande mp3 bestand op de lokale server
def process_server(s3, bucketname, bucket, table, sync_path, arch_path):
    for file in os.listdir(sync_path):
        if file.endswith('.mp3'):
            uuid_token = str(uuid.uuid4())
            timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(sync_path + file))
            create_date = timestamp.strftime('%Y-%m-%d')
            sampleinfo = mediainfo(sync_path + file)
            upload_s3file(bucket, sync_path, file, uuid_token)
            #transcriptie = transcribe_audiofile_beamlm_chunk(BytesIO(load_s3file(s3, bucketname, uuid_token)))
            transcriptie = transcribe_audiofile_chunk_beam_wordoff(BytesIO(load_s3file(s3, bucketname, uuid_token)), debug = True)
            shutil.move(sync_path + file, arch_path + file)
            save_record(table, uuid_token, file, str(sampleinfo['sample_rate']), transcriptie, create_date, str(sampleinfo['duration']))
    return

### Maak een transcriptie volgens het Speech2Text model
def transcribe_audiofile(audio_bytes):
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

    ### Zet de bytestream van de s3 opslag om naar een numberarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")

    ### Zet de sampling rate om naar 16000 voor het getrainde model
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    ### Genereer features uit de data
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

    ### [TODO] Uitzoeken wat deze functie precies doet.
    generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

    ### [TODO] Omzetten prediction naar transcriptie.
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(transcription[0])

    return transcription[0]

### Maak een transcriptie volgens het Wav2Vec2 model
def transcribe_audiofile1(audio_bytes):
    #Loading the pre-trained model and the tokenizer
    model_name = "facebook/wav2vec2-base-960h"
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to('cuda')
    
    ### Zet de bytestream van de s3 opslag om naar een numberarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")

    ### Zet de sampling rate om naar 16000 voor het getrainde model
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    waveform = waveform.squeeze().numpy()

    ### Genereer features uit de data
    input_values = tokenizer(waveform, sampling_rate=16000, return_tensors="pt").to('cuda')

    ### Genereer de logits
    with torch.no_grad():
        logits = model(**input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    ### Maak de transcriptie
    transcription = tokenizer.batch_decode(predicted_ids)

    return transcription[0].lower()

### Maak een transcriptie volgens het Wav2Vec2 model
# Met een chunk functie. Deze werkt hier niet goed.
def transcribe_audiofile_chunked(audio_bytes):
    fixed_samplerate = 16000

    #Loading the pre-trained model and the tokenizer
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    ### Zet de bytestream van de s3 opslag om naar een numberarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")

    ### Zet de sampling rate om naar 16000 voor het getrainde model
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=fixed_samplerate)
    waveform = waveform.squeeze().numpy()
    chunk_duration = 10
    padding_duration = 0

    chunk_len = chunk_duration*fixed_samplerate
    input_padding_len = int(padding_duration*fixed_samplerate)
    output_padding_len = model._get_feat_extract_output_lengths(input_padding_len)
    all_preds = []
    for start in range(input_padding_len, len(waveform)-input_padding_len, chunk_len):

        chunk = waveform[start-input_padding_len:start+chunk_len+input_padding_len]
        input_values = processor(chunk, sampling_rate=fixed_samplerate, return_tensors="pt")

        with torch.no_grad():
            logits = model(**input_values).logits
            logits = logits[output_padding_len:len(logits)-output_padding_len]

            predicted_ids = torch.argmax(logits, dim=-1)
            all_preds.append(predicted_ids)

        print('Test voor chunks.')

    print('Kom jij hier?')
 
    a0 = torch.cat(all_preds, dim=1)
    transcription= processor.batch_decode(a0.numpy())

    print(transcription[0].lower())

    return transcription[0].lower()

### Maak een transcriptie volgens het Wav2Vec2 model
# Met een chunk functie. Deze werkt hier nog niet goed.
def transcribe_audiofile_chunked_large(audio_bytes):
    fixed_samplerate = 16000

    #Loading the pre-trained model and the processor
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to('cuda')

    print(len(audio_bytes.getvalue()))

    ### Zet de bytestream van de s3 opslag om naar een tensorarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")

    ### Zet de sampling rate om naar 16000 voor het getrainde model
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=fixed_samplerate)
    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

    ### Configuratie voor de chunks
    chunk_duration = 30
    striding_duration = 0
    lenght_waveform = len(waveform[0])

    chunk_len = chunk_duration*fixed_samplerate
    stride_len = striding_duration*fixed_samplerate
    tensor_stride_len = model._get_feat_extract_output_lengths(stride_len)

    all_preds = torch.tensor(()).to('cuda')

    for start in range(0, lenght_waveform, chunk_len):
        print('Test voor chunk: ' + str(start))

        if start == 0:
            begin = 0
            padding_input = 0
        else:
            begin = start-stride_len
            padding_input = 1

        if start+chunk_len+stride_len > lenght_waveform:
            end = lenght_waveform
            padding_output = 0
        else:
            end = start + chunk_len + stride_len
            padding_output = 1

        print('Start = ' + str(start))
        print('Waveform = ' + str(lenght_waveform))
        print('Chunk = ' + str(chunk_len))
        print('Begin = ' + str(begin))
        print('End = ' + str(end))

        chunk = waveform[0][begin:end]

        input_values = processor(chunk, sampling_rate=fixed_samplerate, return_tensors="pt").to('cuda')

        with torch.no_grad():
            logits = model(**input_values).logits
            
        if padding_input == 0:
            begin_input = 0
        else:
            begin_input = tensor_stride_len

        if padding_output == 0:
            begin_output = len(logits[0])
        else:
            begin_output = len(logits[0])-tensor_stride_len

        print('Begin_input = '+ str(begin_input))
        print('Begin_output = '+ str(begin_output))
        print(len(logits[0]))
        print(logits)
        logits = logits[begin_input:begin_output]
        print(logits)

        all_preds = torch.cat((all_preds, logits), 1)


    predicted_ids = torch.argmax(all_preds, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    print(transcription[0].lower())

    return transcription[0].lower()

### Maak een transcriptie volgens het Wav2Vec2 model
# Met een pyctcdecoder beamfunctie en taalmodel
# Zonder chunk functie.
def transcribe_audiofile_beamlm(audio_bytes):
    # Laden van het pre-getrainde model en de tokenizer
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to('cuda')

    # Laden van de woordenlijst en het taalmodel voor decodering
    vocab_list = list(processor.tokenizer.get_vocab())
    decoder = build_ctcdecoder(
        vocab_list,
        unigrams='kenlm/language_model_unigrams.txt',
        kenlm_model_path='kenlm/4-gram.bin',
        alpha = 0.5,
        beta = 1.5,
        unk_score_offset=-10.0,
        lm_score_boundary=True
        )
    
    ### Zet de bytestream van de s3 opslag om naar een numberarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    waveform = waveform.squeeze().numpy()

    ### Genereer features uit de data
    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").to('cuda').input_values

    # Creeer de logits
    logits = model(input_values).logits.cpu().detach().numpy()[0]
    
    # Decode de logits
    transcription = decoder.decode(logits)

    return transcription.lower()

### Maak een transcriptie volgens het Wav2Vec2 model
# Met een pyctcdecoder beamfunctie en taalmodel
# Met chunk functie en striding.
# LET OP: Er zit nog een bug in deze functie waar 
# model._get_feat_extract_output_lengths(stride_len)
# Niet een correcte grote voor de striding weergeeft na
# transformatie naar logits. Waardoor en af en toe een frame
# Teveel word toegevoegd op de grans van de chunk. Dit kan een
# foutieve prediction geven in de beamsearch mogelijk.
def transcribe_audiofile_beamlm_chunk(audio_bytes, debug = False):
    # Laden van het pre-getrainde model en de tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (debug): print('Start transciptie')
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

    ### Geef hier de gewenste sample_rate van het model op.
    # TODO: Uitzoeken of we dit ook uit het model zelf kunnen halen.
    fixed_samplerate = 16000

    # Laden van de woordenlijst en het taalmodel voor decodering
    # Dit is de beamsearch decoder van pyctcdecorder
    vocab_list = list(processor.tokenizer.get_vocab())
    with open("kenlm/language_model_unigrams.txt") as f:
        unigram_list = [t for t in f.read().strip().split("\n")]
    decoder = build_ctcdecoder(
        vocab_list,
        kenlm_model_path='kenlm/4-gram.bin',
        unigrams=unigram_list,
        alpha = 0.7,
        beta = 1.5,
        unk_score_offset=-10.0,
        lm_score_boundary=True
        )
    
    # Omzetten van de audiobytes naar een tensorarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")

    # Resample waveform als de sample_rate niet overeen met de sample_rate waarop het model getrained is
    if (sample_rate != fixed_samplerate):
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=fixed_samplerate)
        sample_rate = fixed_samplerate
    
    # Zet waveform om naar mono
    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

    # Configuratie voor het verdelen van audiochunks
    chunk_duration = 30
    striding_duration = 3
    lenght_waveform = len(waveform[0])
    chunk_len = chunk_duration*sample_rate
    stride_len = striding_duration*sample_rate
    # TODO: Bug ui volgende regel halen (lijkt niet altijd juiste waarde te geven)
    tensor_stride_len = int(model._get_feat_extract_output_lengths(stride_len))
    all_preds = torch.tensor(()).to(device)

    if (debug): print()
    if (debug): print('------------------------------- DEBUG ----------------------------------')
    if (debug): print('waveform: ' + str(lenght_waveform) + ' samples.')
    if (debug): print('chunk_len: ' + str(chunk_len) + ' samples.')
    if (debug): print('stride_len: ' + str(stride_len) + ' samples.')
    if (debug): print('tensor_stride_len: ' + str(tensor_stride_len) + ' frames.') 
    if (debug): print('------------------------------- DEBUG ----------------------------------')
    if (debug): print()

    for start in range(0, lenght_waveform, chunk_len):
        # Bepaal de start en eindposities van de huidige chunk
        begin = start - stride_len if start != 0 else 0
        end = min(start + chunk_len + stride_len, lenght_waveform)

        # Extraheren van de huidige chunk van de waveform
        chunk = waveform[0][begin:end]
        input_values = processor(chunk, sampling_rate=sample_rate, return_tensors="pt").to(device)

        if (debug): print()
        if (debug): print('------------------------------- DEBUG ----------------------------------')
        if (debug): print('chunk: ' + str(len(chunk)) + ' samples.')
        if (debug): print('input_values: ' + str(len(input_values['input_values'][0])) + ' samples.')

        # Verkrijg de logits voor de huidige chunk
        with torch.no_grad():
            logits = model(**input_values).logits

        # Bepaal het gedeelte van de logits dat nodig is (afhankelijk van de stride)
        begin_input = tensor_stride_len if start != 0 else 0
        begin_output = len(logits[0]) - tensor_stride_len if end < lenght_waveform else len(logits[0])
        
        if (debug): print('begin: ' + str(begin) + ' samples.')
        if (debug): print('end: ' + str(end) + ' samples.')
        if (debug): print('begin_input: ' + str(begin_input) + ' frames.')
        if (debug): print('begin_output: ' + str(begin_output) + ' frames.')
        if (debug): print('logits: ' + str(len(logits[0])) + ' frames.')
        if (debug): print(logits[0][0])

        logits_stripped = logits[0][begin_input:begin_output]
        
        if (debug): print('logits_stripped: ' + str(len(logits_stripped)) + ' frames.')
        if (debug): print(logits_stripped)

        all_preds = torch.cat((all_preds, logits_stripped), 0)
        
        if (debug): print('all_preds: ' + str(len(all_preds[0])))
        if (debug): print('------------------------------- DEBUG ----------------------------------')
        if (debug): print()

    # Selecteer het relevante gedeelte van de logits en voeg ze samen
    decode_logits = all_preds.cpu().numpy() 
    transcription = decoder.decode(decode_logits)

    if (debug): print(transcription.lower())
    return transcription.lower()


### Maak een transcriptie volgens het Wav2Vec2 model
# Met een pyctcdecoder beamfunctie en taalmodel
# Met chunk functie en striding.
# LET OP: Er zit nog een bug in deze functie waar 
# model._get_feat_extract_output_lengths(stride_len)
# Niet een correcte grote voor de striding weergeeft na
# transformatie naar logits. Waardoor en af en toe een frame
# Teveel word toegevoegd op de grens van de chunk. Dit kan een
# foutieve prediction geven in de beamsearch mogelijk.
def transcribe_audiofile_chunk_beam_wordoff(audio_bytes, debug = False):
    # Laden van het pre-getrainde model en de tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (debug): print('Start transciptie')
    model_name = "facebook/wav2vec2-large-960h-lv60-self"
    model = AutoModelForCTC.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    # import model, feature extractor, tokenizer

    ### Haal de gewenste samplerate van het model op.
    fixed_samplerate = processor.feature_extractor.sampling_rate

    # Laden van de woordenlijst en het taalmodel voor decodering
    # Dit is de beamsearch decoder van pyctcdecorder
    vocab_list = list(processor.tokenizer.get_vocab())
    with open("kenlm/language_model_unigrams.txt") as f:
        unigram_list = [t for t in f.read().strip().split("\n")]
    decoder = build_ctcdecoder(
        vocab_list,
        kenlm_model_path='kenlm/4-gram.bin',
        unigrams=unigram_list,
        alpha = 1.5,
        beta = 1.5,
        unk_score_offset=-10.0,
        lm_score_boundary=True
        )

    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    # Omzetten van de audiobytes naar een tensorarray
    waveform, sample_rate = torchaudio.load(audio_bytes, format="mp3")

    # Resample waveform als de sample_rate niet overeen met de sample_rate waarop het model getrained is
    if (sample_rate != fixed_samplerate):
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=fixed_samplerate)
        sample_rate = fixed_samplerate
    
    # Zet waveform om naar mono
    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

    # Configuratie voor het verdelen van audiochunks
    chunk_duration = 30
    striding_duration = 3
    lenght_waveform = len(waveform[0])
    chunk_len = chunk_duration*sample_rate
    stride_len = striding_duration*sample_rate
    # TODO: Bug ui volgende regel halen (lijkt niet altijd juiste waarde te geven)
    tensor_stride_len = int(model._get_feat_extract_output_lengths(stride_len))
    all_preds = torch.tensor(()).to(device)

    if (debug): print()
    if (debug): print('------------------------------- DEBUG ----------------------------------')
    if (debug): print('waveform: ' + str(lenght_waveform) + ' samples.')
    if (debug): print('chunk_len: ' + str(chunk_len) + ' samples.')
    if (debug): print('stride_len: ' + str(stride_len) + ' samples.')
    if (debug): print('tensor_stride_len: ' + str(tensor_stride_len) + ' frames.') 
    if (debug): print('------------------------------- DEBUG ----------------------------------')
    if (debug): print()

    for start in range(0, lenght_waveform, chunk_len):
        # Bepaal de start en eindposities van de huidige chunk
        begin = start - stride_len if start != 0 else 0
        end = min(start + chunk_len + stride_len, lenght_waveform)

        # Extraheren van de huidige chunk van de waveform
        chunk = waveform[0][begin:end]
        input_values = processor_with_lm(chunk, sampling_rate=sample_rate, return_tensors="pt").to(device)

        if (debug): print()
        if (debug): print('------------------------------- DEBUG ----------------------------------')
        if (debug): print('chunk: ' + str(len(chunk)) + ' samples.')
        if (debug): print('input_values: ' + str(len(input_values['input_values'][0])) + ' samples.')

        # Verkrijg de logits voor de huidige chunk
        with torch.no_grad():
            logits = model(**input_values).logits
            #logits = model(input_values).logits[0].cpu().numpy()

        # Bepaal het gedeelte van de logits dat nodig is (afhankelijk van de stride)
        begin_input = tensor_stride_len if start != 0 else 0
        begin_output = len(logits[0]) - tensor_stride_len if end < lenght_waveform else len(logits[0])
        
        if (debug): print('begin: ' + str(begin) + ' samples.')
        if (debug): print('end: ' + str(end) + ' samples.')
        if (debug): print('begin_input: ' + str(begin_input) + ' frames.')
        if (debug): print('begin_output: ' + str(begin_output) + ' frames.')
        if (debug): print('logits: ' + str(len(logits[0])) + ' frames.')
        if (debug): print(logits[0][0])

        logits_stripped = logits[0][begin_input:begin_output]
        
        if (debug): print('logits_stripped: ' + str(len(logits_stripped)) + ' frames.')
        if (debug): print(logits_stripped)

        all_preds = torch.cat((all_preds, logits_stripped), 0)
        
        if (debug): print('all_preds: ' + str(len(all_preds[0])))
        if (debug): print('------------------------------- DEBUG ----------------------------------')
        if (debug): print()

    # Selecteer het relevante gedeelte van de logits en voeg ze samen
    decode_logits = all_preds.cpu().numpy() 
    #transcription = decoder.decode(decode_logits)
    # retrieve word stamps (analogous commands for `output_char_offsets`)
    outputs = processor_with_lm.decode(decode_logits, output_word_offsets=True)
    transcription = processor_with_lm.decode(decode_logits, output_word_offsets=False)
    # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
    time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

    word_offsets = [
        {
            "word": d["word"],
            "start_time": round(d["start_offset"] * time_offset, 2),
            "end_time": round(d["end_offset"] * time_offset, 2),
        }
        for d in outputs.word_offsets
    ]

    print(word_offsets)
    if (debug): print(transcription['text'].lower())
    return transcription['te1xt'].lower()
