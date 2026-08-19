import os
import base64

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHEMIN_MODELE = os.path.join(
    BASE_DIR,
    "app",
    "modele_tri_dechets.keras"
)

CHEMIN_CLASSES = os.path.join(
    BASE_DIR,
    "app",
    "classes.txt"
)

TAILLE_IMAGE = (224, 224)
# SEUIL_CONFIANCE = 0.55
SEUIL_CONFIANCE = 0.70


# ============================================================
# Application Flask
# ============================================================

app = Flask(__name__)


# ============================================================
# Chargement du modèle
# ============================================================

print("Chargement du modèle...")

modele = tf.keras.models.load_model(CHEMIN_MODELE)

print("Modèle chargé avec succès !")


# ============================================================
# Chargement des classes
# ============================================================

with open(CHEMIN_CLASSES, "r") as fichier:
    classes = [
        ligne.strip()
        for ligne in fichier
        if ligne.strip()
    ]

print("Classes :", classes)


# ============================================================
# Correspondance classe → poubelle
# ============================================================

POUBELLE_RECOMMANDEE = {
    "cardboard": "Poubelle jaune — recyclable (carton)",
    "glass": "Poubelle verte — verre",
    "metal": "Poubelle jaune — recyclable (métal)",
    "paper": "Poubelle jaune — recyclable (papier)",
    "plastic": "Poubelle jaune — recyclable (plastique)",
    "trash": "Poubelle grise — non recyclable",
}


# ============================================================
# Préparation de l'image
# ============================================================

def preparer_image(image_bgr):

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_redim = cv2.resize(
        image_rgb,
        TAILLE_IMAGE
    )

    image_array = np.expand_dims(
        image_redim,
        axis=0
    ).astype("float32")

    return preprocess_input(image_array)


# ============================================================
# Page principale
# ============================================================

@app.route("/")
def accueil():
    return render_template("index.html")


# ============================================================
# API de prédiction
# ============================================================

@app.route("/predict", methods=["POST"])
def predire():

    try:

        # ----------------------------------------------------
        # Vérifier qu'une image est reçue
        # ----------------------------------------------------

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "message": "Aucune image reçue."
            }), 400

        fichier = request.files["image"]

        # ----------------------------------------------------
        # Lire l'image
        # ----------------------------------------------------

        image_bytes = fichier.read()

        image_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        image_bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image_bgr is None:
            return jsonify({
                "success": False,
                "message": "Impossible de lire l'image."
            }), 400

        # ----------------------------------------------------
        # Prétraitement
        # ----------------------------------------------------

        entree = preparer_image(image_bgr)

        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        predictions = modele.predict(
            entree,
            verbose=0
        )[0]

        index_classe = int(
            np.argmax(predictions)
        )

        classe = classes[index_classe]

        confiance = float(
            predictions[index_classe]
        )

        # ----------------------------------------------------
        # Seuil de confiance
        # ----------------------------------------------------

        if confiance >= SEUIL_CONFIANCE:

            resultat = classe

            poubelle = POUBELLE_RECOMMANDEE.get(
                classe,
                "Poubelle inconnue"
            )

            certain = True

        else:

            resultat = "Incertain"

            poubelle = (
                "Rapprochez l'objet ou améliorez "
                "les conditions d'éclairage."
            )

            certain = False

        # ----------------------------------------------------
        # Réponse JSON
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "classe": resultat,

            "confiance": round(
                confiance * 100,
                2
            ),

            "poubelle": poubelle,

            "certain": certain

        })

    except Exception as erreur:

        print("Erreur :", erreur)

        return jsonify({

            "success": False,

            "message": str(erreur)

        }), 500


# ============================================================
# Lancement
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )