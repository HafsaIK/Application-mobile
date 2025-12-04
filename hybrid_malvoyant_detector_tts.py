# fichier : hybrid_malvoyant_detector_tts.py
# Détecteur hybride (custom 35 classes + YOLOv8 COCO) + annonce vocale des objets détectés

import os
import argparse
import cv2
import torch
from ultralytics import YOLO
import pyttsx3

# ==========================
# 1) Configuration
# ==========================

# Ton modèle custom (35 classes)
# Ton modèle custom (35 classes)
# CUSTOM_MODEL_PATH = r"C:\Users\User\Desktop\yolov8s_balanced_80epochs_6reco\weight et resultats de entrainement\weights\best.pt"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "weights", "best.pt")

# Modèle COCO générique (80 classes)
COCO_MODEL_PATH = "yolov8n.pt"   # téléchargé auto s'il n'existe pas

# Dossier de sortie (images annotées + audio)
# OUTPUT_DIR = r"C:\Users\User\Desktop\yolov8s_balanced_80epochs_6reco\REsultats test\hybrid_tts"
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Seuil confiance
CONF_TH = 0.35

# Tes 35 classes custom (en anglais, comme dans ton YAML)
CUSTOM_CLASSES = {
    "person", "dog", "cat", "car", "bus", "truck",
    "bicycle", "motorcycle", "train",
    "traffic light", "stop sign",
    "chair", "couch", "bed", "dining table", "tv",
    "potted plant", "clock", "book", "laptop", "cell phone",
    "cup", "bottle", "bowl",
    "knife", "fork", "spoon",
    "apple", "banana", "orange",
    "handbag", "backpack", "umbrella", "tie", "bench"
}

# (optionnel) traduction française pour la voix
FR_TRANSLATION = {
    "person": "personne",
    "dog": "chien",
    "cat": "chat",
    "car": "voiture",
    "bus": "bus",
    "truck": "camion",
    "bicycle": "vélo",
    "motorcycle": "moto",
    "train": "train",
    "traffic light": "feu de circulation",
    "stop sign": "panneau stop",
    "chair": "chaise",
    "couch": "canapé",
    "bed": "lit",
    "dining table": "table à manger",
    "tv": "télévision",
    "potted plant": "plante en pot",
    "clock": "horloge",
    "book": "livre",
    "laptop": "ordinateur portable",
    "cell phone": "téléphone",
    "cup": "tasse",
    "bottle": "bouteille",
    "bowl": "bol",
    "knife": "couteau",
    "fork": "fourchette",
    "spoon": "cuillère",
    "apple": "pomme",
    "banana": "banane",
    "orange": "orange",
    "handbag": "sac à main",
    "backpack": "sac à dos",
    "umbrella": "parapluie",
    "tie": "cravate",
    "bench": "banc",
}


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}


# ==========================
# 2) Fusion simple
# ==========================

def merge_simple(res_custom, res_coco):
    """
    RÈGLE :

    - Pour les 35 classes custom : on garde UNIQUEMENT le modèle custom.
    - Pour les autres classes : on garde les prédictions COCO.

    Résultat : liste de dicts {box, name, conf, source}
    """
    merged = []

    # --- Prédictions custom ---
    b_c = res_custom.boxes
    if b_c is not None and len(b_c) > 0:
        xyxy_c = b_c.xyxy.cpu().numpy()
        conf_c = b_c.conf.cpu().numpy()
        cls_c = b_c.cls.cpu().numpy().astype(int)
        names_c = res_custom.names

        for box, cf, cl in zip(xyxy_c, conf_c, cls_c):
            merged.append({
                "box": box,
                "conf": float(cf),
                "name": names_c[cl],
                "source": "custom"
            })

    # --- Prédictions COCO ---
    b_o = res_coco.boxes
    if b_o is None or len(b_o) == 0:
        return merged

    xyxy_o = b_o.xyxy.cpu().numpy()
    conf_o = b_o.conf.cpu().numpy()
    cls_o = b_o.cls.cpu().numpy().astype(int)
    names_o = res_coco.names

    for box, cf, cl in zip(xyxy_o, conf_o, cls_o):
        name_o = names_o[cl]

        # ⚠️ Si la classe COCO est dans CUSTOM_CLASSES,
        # on l'ignore (on fait confiance à ton modèle).
        if name_o in CUSTOM_CLASSES:
            continue

        merged.append({
            "box": box,
            "conf": float(cf),
            "name": name_o,
            "source": "coco"
        })

    return merged


def draw_merged(img, merged):
    """
    Dessine les boxes fusionnées :
      - Vert : modèle custom
      - Bleu : modèle COCO (classes supplémentaires)
    """
    for det in merged:
        x1, y1, x2, y2 = map(int, det["box"])
        label = f'{det["name"]} {det["conf"]:.2f}'
        if det["source"] == "custom":
            color = (0, 255, 0)   # vert
        else:
            color = (255, 0, 0)   # bleu

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )
    return img


# ==========================
# 3) Synthèse vocale
# ==========================

def build_sentence_from_detections(merged, filename):
    """
    Construit une phrase française du type :
    "Attention, dans l'image X, il y a : personne, voiture, bus."
    """
    if not merged:
        return ""

    # classes uniques, source peu importe
    names = []
    for det in merged:
        name = det["name"]
        # traduction en français si dispo
        fr_name = FR_TRANSLATION.get(name, name)
        if fr_name not in names:
            names.append(fr_name)

    # Sort names to ensure consistent order (e.g. "car, person" == "person, car")
    names.sort()
    listed = ", ".join(names)
    sentence = f"Attention, dans cette image il y a : {listed}."
    return sentence


def tts_save(sentence, out_audio_path):
    """
    Lis la phrase avec pyttsx3 et la sauvegarde dans un fichier audio (wav).
    """
    engine = pyttsx3.init()
    # (Optionnel) Essayer de mettre une voix française si dispo
    # Tu peux commenter ce bloc si ça pose problème.
    for v in engine.getProperty('voices'):
        if "fr" in v.id.lower() or "french" in v.name.lower():
            engine.setProperty('voice', v.id)
            break

    engine.save_to_file(sentence, out_audio_path)
    engine.runAndWait()


# ==========================
# 4) Traitement image / vidéo
# ==========================

# ==========================
# 4) Traitement image / vidéo
# ==========================

def process_image_data(img_array, model_custom, model_coco, device):
    """
    Version API : prend une image (numpy array BGR), retourne :
    - image annotée (numpy array)
    - phrase (str)
    """
    # 1. Prédictions
    res_cus = model_custom.predict(
        source=img_array,
        conf=CONF_TH,
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    res_coco = model_coco.predict(
        source=img_array,
        conf=CONF_TH,
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    # 2. Fusion
    merged = merge_simple(res_cus, res_coco)

    # 3. Annotation
    img_annot = res_cus.orig_img.copy()
    img_annot = draw_merged(img_annot, merged)

    # 4. Phrase
    sentence = build_sentence_from_detections(merged, "image_api")
    
    return img_annot, sentence


def process_image(path_img, model_custom, model_coco, device):
    print(f"\n🖼️ Image : {os.path.basename(path_img)}")

    res_cus = model_custom.predict(
        source=path_img,
        conf=CONF_TH,
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    res_coco = model_coco.predict(
        source=path_img,
        conf=CONF_TH,
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False
    )[0]

    merged = merge_simple(res_cus, res_coco)

    img = res_cus.orig_img.copy()
    img_annot = draw_merged(img, merged)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path_img))[0]

    # Image annotée
    out_img = os.path.join(OUTPUT_DIR, base + "_hybrid.jpg")
    cv2.imwrite(out_img, img_annot)

    # Phrase + audio
    sentence = build_sentence_from_detections(merged, os.path.basename(path_img))
    out_wav = os.path.join(OUTPUT_DIR, base + "_hybrid.wav")
    tts_save(sentence, out_wav)

    print(f"✅ Image sauvegardée : {out_img}")
    print(f"🔊 Audio sauvegardé : {out_wav}")
    print("   Phrase dite :")
    print("   ", sentence)


def process_video(path_vid, model_custom, model_coco, device):
    """
    Pour simplifier : on génère une seule phrase pour la vidéo,
    en se basant sur un échantillon de frames (ex. ~10 premières).
    Et on enregistre la vidéo annotée + audio.
    """
    print(f"\n🎥 Vidéo : {os.path.basename(path_vid)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path_vid))[0]
    out_video = os.path.join(OUTPUT_DIR, base + "_hybrid.mp4")

    stream_cus = model_custom.predict(
        source=path_vid,
        conf=CONF_TH,
        iou=0.5,
        imgsz=640,
        device=device,
        stream=True,
        verbose=False
    )
    stream_coco = model_coco.predict(
        source=path_vid,
        conf=CONF_TH,
        iou=0.5,
        imgsz=640,
        device=device,
        stream=True,
        verbose=False
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    # Pour construire une phrase globale, on accumule les classes sur les premières frames
    all_dets_for_sentence = []

    for idx, (res_cus, res_coco) in enumerate(zip(stream_cus, stream_coco)):
        frame = res_cus.orig_img
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_video, fourcc, 25, (w, h))

        merged = merge_simple(res_cus, res_coco)
        frame_annot = draw_merged(frame, merged)
        writer.write(frame_annot)

        # on accumule les détections des ~10 premières frames
        if idx < 10:
            all_dets_for_sentence.extend(merged)

    if writer is not None:
        writer.release()

    # Construire phrase globale + audio
    sentence = build_sentence_from_detections(all_dets_for_sentence, os.path.basename(path_vid))
    out_wav = os.path.join(OUTPUT_DIR, base + "_hybrid.wav")
    tts_save(sentence, out_wav)

    print(f"✅ Vidéo hybride sauvegardée : {out_video}")
    print(f"🔊 Audio sauvegardé : {out_wav}")
    print("   Phrase dite :")
    print("   ", sentence)


# ==========================
# 5) Main
# ==========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image / vidéo / dossier (images+vidéos) à tester"
    )
    args = parser.parse_args()
    src = args.source

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🔹 Device utilisé : {device}")

    print(f"🔹 Chargement modèle custom : {CUSTOM_MODEL_PATH}")
    model_custom = YOLO(CUSTOM_MODEL_PATH)

    print(f"🔹 Chargement modèle COCO : {COCO_MODEL_PATH}")
    model_coco = YOLO(COCO_MODEL_PATH)

    if os.path.isdir(src):
        files = sorted(os.listdir(src))
        for f in files:
            path = os.path.join(src, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in IMG_EXT:
                process_image(path, model_custom, model_coco, device)
            elif ext in VID_EXT:
                process_video(path, model_custom, model_coco, device)
    else:
        ext = os.path.splitext(src)[1].lower()
        if ext in IMG_EXT:
            process_image(src, model_custom, model_coco, device)
        elif ext in VID_EXT:
            process_video(src, model_custom, model_coco, device)
        else:
            print("❌ Format non supporté :", src)


if __name__ == "__main__":
    main()
