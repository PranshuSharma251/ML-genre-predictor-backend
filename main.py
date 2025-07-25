from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, numpy as np, librosa, joblib
from pydub import AudioSegment
from pydub.utils import which
from fastapi.middleware.cors import CORSMiddleware



# ────────────────── FastAPI setup ──────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ml-music-genre-predictor.vercel.app/","https://ml-music-genre-predictor.vercel.app"],  # you can replace "*" with your frontend domain for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────── Load model & scaler ──────────────────
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ────────────────── Feature extractor (57 dims) ──────────────────
def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, duration=30)

    feats = []

    # 1‑2 chroma stft
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += [np.mean(chroma), np.var(chroma)]

    # 3‑4 RMS
    rms = librosa.feature.rms(y=y)[0]
    feats += [np.mean(rms), np.var(rms)]

    # 5‑6 spectral centroid
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats += [np.mean(sc), np.var(sc)]

    # 7‑8 bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats += [np.mean(bw), np.var(bw)]

    # 9‑10 roll‑off
    ro = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats += [np.mean(ro), np.var(ro)]

    # 11‑12 zero‑crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats += [np.mean(zcr), np.var(zcr)]

    # 13‑14 harmonic component
    harm = librosa.effects.harmonic(y)
    feats += [np.mean(harm), np.var(harm)]

    # 15‑16 spectral flatness (perceptual)
    flat = librosa.feature.spectral_flatness(y=y)[0]
    feats += [np.mean(flat), np.var(flat)]

    # 17 tempo
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    feats.append(tempo)

    # 18‑57: 20 MFCCs (mean & var)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        feats += [np.mean(mfcc[i]), np.var(mfcc[i])]

    return np.array(feats).reshape(1, -1)  # shape (1, 57)

# ────────────────── Prediction endpoint ──────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    tmp_path = f"temp/{file.filename}"

    # save upload
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # convert mp3→wav if needed
    if tmp_path.lower().endswith(".mp3"):
        wav_path = tmp_path.rsplit(".", 1)[0] + ".wav"
        AudioSegment.from_mp3(tmp_path).export(wav_path, format="wav")
        os.remove(tmp_path)
        tmp_path = wav_path

    try:
        X = extract_features(tmp_path)
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)[0]
        top3_indices = np.argsort(proba)[::-1][:3]
        class_names = model.classes_

        predictions = [
            {
                "genre": class_names[i],
                "confidence": round(float(proba[i]), 4)
            }
            for i in top3_indices
        ]
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"predictions": predictions}