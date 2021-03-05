from typing import Optional
import wave
import os
import pylab
import uuid
from fastapi import FastAPI, File, UploadFile
from fastai import *
from fastai.vision import *
from io import BytesIO
from PIL import Image

app = FastAPI()

learn = load_learner("model/", "stage-1.pth")

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


@app.get("/")
def read_root():
    return {"status": "online"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    print(file.filename)
    wav_f = file.file
    sound_info, frame_rate = get_wav_info(wav_f)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.specgram(sound_info, Fs=frame_rate)
    filename = str(uuid.uuid4())
    pylab.savefig('tmp/'+filename+".png")
    path = 'tmp/'+filename+".png"
    im = Image.open(path)
    img_bytes = await (im.os.read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return {'result': str(prediction)}
