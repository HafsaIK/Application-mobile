import os
from typing import List, Dict

from PIL import Image
import numpy as np
import torch

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from paddleocr import PaddleOCR

from gtts import gTTS
from playsound import playsound
import tempfile


# ==============================
# CONFIG
# ==============================

# BASE_DIR = r"C:\Users\User\Desktop\yolov8s_balanced_80epochs_6reco\REsultats test\amelioration ocr maroc"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_TEST_DIR = os.path.join(BASE_DIR, "Det_test")
# MODEL_DIR = os.path.join(BASE_DIR, "amelioration ocr maroc", "trocr_rec_maroc_finetuned", "epoch_5")
MODEL_DIR = os.path.join(BASE_DIR, "trocr_rec_maroc_finetuned", "epoch_5")
IMAGE_NAME = "img_80.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"📌 Device : {DEVICE}")
print(f"🧠 Chargement modèle EN : {MODEL_DIR}")


# ==============================
# MODELS INIT
# ==============================

processor_en = TrOCRProcessor.from_pretrained(MODEL_DIR)
# model_en = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE)
# model_en.eval()

# ocr_ar = PaddleOCR(lang="ar")


ocr_ar = PaddleOCR(lang="ar")


# ==============================
# UTILITAIRES
# ==============================

def clean_token(token: str) -> str:
    """Supprime ### et tokens vides."""
    t = token.strip()
    if not t:
        return ""
    if all(ch == "#" for ch in t):
        return ""
    return t


def parse_det_file(txt_path: str):
    entries = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            coords = list(map(int, parts[:8]))
            lang = parts[8].strip()
            text = clean_token(",".join(parts[9:]))
            entries.append({"coords": coords, "lang": lang, "text": text})
    return entries


def trocr_ocr_pil(img, processor_en, model_en):
    inputs = processor_en(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model_en.generate(**inputs)
    return processor_en.batch_decode(out, skip_special_tokens=True)[0].strip()


def paddle_ocr_pil(img, ocr_ar):
    np_img = np.array(img)
    result = ocr_ar.ocr(np_img)
    texts = []
    for line in result:
        for item in line:
            if len(item) >= 2:
                txt, score = item[1][0], item[1][1]
                txt = clean_token(txt)
                if txt and score > 0.5:
                    texts.append(txt)
    return " ".join(texts).strip()


# ==============================
# TTS
# ==============================

def play_tts(text, lang):
    text = text.strip()
    if not text:
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = tmp.name
    tmp.close()

    gTTS(text=text, lang=lang).save(tmp_path)

    print(f"🔊 Lecture ({lang}) :", text)
    playsound(tmp_path)

    os.remove(tmp_path)


def speak_ar_en(text_ar, text_en):
    print("\n🧾 TEXTE LU :")
    print("AR :", text_ar)
    print("EN :", text_en)

    if text_ar.strip():
        print("\n🔊 Lecture AR...")
        play_tts(text_ar, lang="ar")

    if text_en.strip():
        print("\n🔊 Lecture EN...")
        play_tts(text_en, lang="en")


# ==============================
# MAIN
# ==============================

def analyze_image_text(img, processor_en, model_en, ocr_ar):
    """
    Prend une image PIL, retourne (resume_ar, resume_en)
    """
    # Pour l'API, on va faire simple : on applique l'OCR sur toute l'image
    # Ou alors on simule la détection de zones si on n'a pas le fichier .txt
    # ICI : On va supposer qu'on applique Paddle sur toute l'image pour l'Arabe
    # et TrOCR sur toute l'image (ou des crops si on avait un détecteur de texte) pour l'Anglais.
    
    # NOTE: Le script original utilisait un fichier .txt de détection (Det_test).
    # Sans ce fichier (cas temps réel), on doit soit :
    # 1. Utiliser PaddleOCR pour détecter les boîtes (il le fait bien).
    # 2. Utiliser ces boîtes pour nourrir TrOCR si c'est de l'anglais.
    
    # Approche simplifiée pour l'API :
    # 1. PaddleOCR pour tout (détection + reco Arabe + reco Anglais basique)
    # 2. Si on veut vraiment TrOCR, on peut l'appliquer sur les crops trouvés par Paddle.
    
    # On va utiliser PaddleOCR pour détecter les zones de texte
    np_img = np.array(img)
    result = ocr_ar.ocr(np_img)
    
    ar_words = []
    en_words = []
    
    if result:
        lines = []
        print(f"DEBUG: PaddleOCR result type: {type(result)}")
        if isinstance(result, list) and len(result) > 0:
             print(f"DEBUG: result[0] type: {type(result[0])}")

        # Robust parsing
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list):
                 # Standard nested list
                 try:
                    if isinstance(result[0][0], list):
                         lines = result[0]
                    else:
                         lines = result
                 except (IndexError, TypeError):
                    lines = result
            elif len(result) > 0 and hasattr(result[0], 'dt_polys') and hasattr(result[0], 'rec_text'):
                # PaddleX OCRResult object list
                lines = result
            else:
                lines = result
        
        for line in lines:
            coords = []
            text_paddle = ""
            score = 0.0

            # Check if line is PaddleX OCRResult object
            if hasattr(line, 'dt_polys') and hasattr(line, 'rec_text'):
                # PaddleX OCRResult object
                # dt_polys: list of points [[x,y], [x,y], ...]
                # rec_text: string
                # rec_score: float
                coords = line.dt_polys if hasattr(line, 'dt_polys') else []
                text_paddle = line.rec_text if hasattr(line, 'rec_text') else ""
                score = line.rec_score if hasattr(line, 'rec_score') else 0.0
            elif isinstance(line, dict):
                 # Maybe dict?
                 coords = line.get('points', [])
                 text_paddle = line.get('text', "")
                 score = line.get('score', 0.0)
            elif isinstance(line, list):
                # Standard list format: [[x1,y1, x2,y2, ...], (text, score)]
                if len(line) >= 2:
                    coords = line[0]
                    if isinstance(line[1], tuple) or isinstance(line[1], list):
                        text_paddle = line[1][0]
                        score = line[1][1] if len(line[1]) > 1 else 0.0
            else:
                # Try string conversion as fallback or skip
                print(f"DEBUG: Unknown line format: {type(line)}")
                continue

            # ... rest of logic ...
            
            # Normalize coords to list of [x,y] if needed
            # If coords is flat list [x1,y1, x2,y2], convert to [[x1,y1], [x2,y2]]
            # If coords is already [[x,y], ...], good.
            
            final_coords = []
            if coords and isinstance(coords[0], (int, float)):
                 # Flat list
                 for i in range(0, len(coords), 2):
                     final_coords.append([coords[i], coords[i+1]])
            else:
                 final_coords = coords
            
            coords = final_coords

            # On peut essayer de deviner la langue ou juste tout garder
            # Pour l'instant, on renvoie tout ce que Paddle trouve
            # (TrOCR est lent si on doit l'appeler sur chaque crop sans savoir si c'est EN)
            
            # Simple heuristique : si contient des caractères arabes -> AR, sinon EN
            if any("\u0600" <= c <= "\u06FF" for c in text_paddle):
                # C'est de l'arabe
                if not coords:
                    continue
                # Centre de la boite
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x_c = sum(xs) / 4
                y_c = sum(ys) / 4
                ar_words.append({"text": text_paddle, "x": x_c, "y": y_c})
            else:
                # C'est probablement de l'anglais (ou autre)
                # On peut utiliser TrOCR ici pour améliorer la reco si on veut
                if not coords:
                    continue
                # Crop
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
                
                # Ensure crop is valid
                if x_max > x_min and y_max > y_min:
                    crop = img.crop((x_min, y_min, x_max, y_max))
                    try:
                        text_trocr = trocr_ocr_pil(crop, processor_en, model_en)
                    except:
                        text_trocr = text_paddle # Fallback
                else:
                    text_trocr = text_paddle

                x_c = sum(xs) / 4
                y_c = sum(ys) / 4
                en_words.append({"text": text_trocr, "x": x_c, "y": y_c})

    # Tri
    ar_sorted = sorted(ar_words, key=lambda w: (w["y"], -w["x"]))
    en_sorted = sorted(en_words, key=lambda w: (w["y"], w["x"]))

    resume_ar = " ".join([w["text"] for w in ar_sorted])
    resume_en = " ".join([w["text"] for w in en_sorted])
    
    return resume_ar, resume_en


def main():
    img_path = os.path.join(DET_TEST_DIR, IMAGE_NAME)
    txt_path = img_path.replace(".jpg", ".txt")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    entries = parse_det_file(txt_path)

    print(f"🖼️ Image : {img_path}")
    print(f"📄 Fichier : {txt_path}")
    print(f"📌 {len(entries)} zones détectées\n")

    ar_words = []
    en_words = []

    for e in entries:
        lang = e["lang"]
        text_gt = e["text"]

        coords = e["coords"]
        xs = coords[0::2]
        ys = coords[1::2]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        crop = img.crop((x_min, y_min, x_max, y_max))

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        if lang.lower().startswith("arabic"):
            # GT propre (pas la prédiction)
            if text_gt:
                ar_words.append({"text": text_gt, "x": x_center, "y": y_center})

        elif lang.lower().startswith("english"):
            pred = trocr_ocr_pil(crop)
            pred = clean_token(pred)
            tok = pred if pred else text_gt
            if tok:
                en_words.append({"text": tok, "x": x_center, "y": y_center})

        else:
            # fallback EN
            pred = trocr_ocr_pil(crop)
            pred = clean_token(pred)
            tok = pred if pred else text_gt
            if tok:
                en_words.append({"text": tok, "x": x_center, "y": y_center})

    # =====================================
    # 🔥 TRI : AR -> droite→gauche, EN -> gauche→droite
    # =====================================

    # Arabe : y croissant, x décroissant
    ar_sorted = sorted(ar_words, key=lambda w: (w["y"], -w["x"]))

    # Anglais : y croissant, x croissant
    en_sorted = sorted(en_words, key=lambda w: (w["y"], w["x"]))

    resume_ar = " ".join([w["text"] for w in ar_sorted])
    resume_en = " ".join([w["text"] for w in en_sorted])

    print("\n🧾 Résumé AR :", resume_ar)
    print("🧾 Résumé EN :", resume_en)

    speak_ar_en(resume_ar, resume_en)

    print("\n✅ FIN")


if __name__ == "__main__":
    main()
