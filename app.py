import os

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# CONFIGURATION
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

SEUIL_CONFIANCE = 0.70


# ============================================================
# APPLICATION FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print("========================================")
print("Chargement du modèle...")
print("========================================")

modele = tf.keras.models.load_model(
    CHEMIN_MODELE
)

print("Modèle chargé avec succès !")


# ============================================================
# CHARGEMENT DES CLASSES
# ============================================================

with open(
    CHEMIN_CLASSES,
    "r",
    encoding="utf-8"
) as fichier:

    classes = [
        ligne.strip()
        for ligne in fichier
        if ligne.strip()
    ]


print("Classes :", classes)


# ============================================================
# CORRESPONDANCE CLASSE -> POUBELLE
# ============================================================

POUBELLE_RECOMMANDEE = {

    "cardboard":
        "Poubelle jaune — recyclable (carton)",

    "glass":
        "Poubelle verte — verre",

    "metal":
        "Poubelle jaune — recyclable (métal)",

    "paper":
        "Poubelle jaune — recyclable (papier)",

    "plastic":
        "Poubelle jaune — recyclable (plastique)",

    "trash":
        "Poubelle grise — non recyclable"
}


# ============================================================
# PREPARATION IMAGE
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

    image_pretraitee = preprocess_input(
        image_array
    )

    return image_pretraitee


# ============================================================
# PAGE PRINCIPALE
# ============================================================

@app.route("/")
def accueil():

    return render_template(
        "index.html"
    )


# ============================================================
# TEST SERVEUR
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "success": True,

        "message":
            "Serveur opérationnel",

        "modele_charge":
            modele is not None,

        "classes":
            classes

    })


@app.route("/test-post", methods=["POST"])
def test_post():

    print("========================================")
    print("TEST POST RECU")
    print("========================================")

    return jsonify({
        "success": True,
        "message": "POST fonctionne correctement"
    })

# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predire():

    print("")
    print("========================================")
    print("1 - REQUETE /predict RECUE")
    print("========================================")

    try:

        # ====================================================
        # ETAPE 1 : fichier
        # ====================================================

        print("2 - Vérification du fichier...")

        if "image" not in request.files:

            print("ERREUR : image absente")

            return jsonify({
                "success": False,
                "message": "Aucune image reçue."
            }), 400

        fichier = request.files["image"]

        print(
            "3 - Fichier reçu :",
            fichier.filename
        )

        # ====================================================
        # ETAPE 2 : lecture
        # ====================================================

        image_bytes = fichier.read()

        print(
            "4 - Taille image :",
            len(image_bytes),
            "bytes"
        )

        if not image_bytes:

            print("ERREUR : image vide")

            return jsonify({
                "success": False,
                "message": "Image vide."
            }), 400

        # ====================================================
        # ETAPE 3 : OpenCV
        # ====================================================

        print("5 - Décodage OpenCV...")

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image_bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image_bgr is None:

            print("ERREUR : cv2.imdecode() a échoué")

            return jsonify({
                "success": False,
                "message": "Impossible de décoder l'image."
            }), 400

        print(
            "6 - Image décodée :",
            image_bgr.shape
        )

        # ====================================================
        # ETAPE 4 : préparation
        # ====================================================

        print("7 - Préparation de l'image...")

        entree = preparer_image(
            image_bgr
        )

        print(
            "8 - Image préparée :",
            entree.shape,
            entree.dtype
        )

        # ====================================================
        # ETAPE 5 : modèle
        # ====================================================

        print("9 - AVANT modele.predict()")
        print("========================================")

        predictions = modele.predict(
            entree,
            verbose=0
        )

        print("========================================")
        print("10 - APRES modele.predict()")

        print(
            "Predictions :",
            predictions
        )

        # ====================================================
        # ETAPE 6 : résultat
        # ====================================================

        predictions = predictions[0]

        index_classe = int(
            np.argmax(predictions)
        )

        if index_classe >= len(classes):

            raise ValueError(
                "L'indice de classe retourné par le modèle "
                "ne correspond pas au fichier classes.txt."
            )


        classe = classes[index_classe]


        confiance = float(
            predictions[index_classe]
        )


        print(
            "11 - Classe :",
            classe
        )

        print(
            "12 - Confiance :",
            confiance
        )

        # ====================================================
        # ETAPE 7 : poubelle
        # ====================================================

        if confiance >= SEUIL_CONFIANCE:

            resultat = classe

            poubelle = (
                POUBELLE_RECOMMANDEE.get(
                    classe,
                    "Poubelle inconnue"
                )
            )

            certain = True

        else:

            resultat = "Incertain"

            poubelle = (
                "Rapprochez l'objet ou améliorez "
                "les conditions d'éclairage."
            )

            certain = False


        # ====================================================
        # ETAPE 8 : réponse
        # ====================================================

        reponse = {

            "success": True,

            "classe": resultat,

            "confiance": round(
                confiance * 100,
                2
            ),

            "poubelle": poubelle,

            "certain": certain

        }


        print(
            "13 - Réponse finale :",
            reponse
        )

        print(
            "========================================"
        )


        return jsonify(reponse)


    except Exception as erreur:

        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERREUR PYTHON DANS /predict")
        print("Type :", type(erreur).__name__)
        print("Message :", str(erreur))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return jsonify({
            "success": False,
            "message": str(erreur)
        }), 500


# ============================================================
# DEMARRAGE
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print(
        "Lancement de l'application Flask..."
    )

    print(
        "Port :",
        port
    )

    app.run(

        host="0.0.0.0",

        port=port

    )