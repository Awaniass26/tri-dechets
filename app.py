import os

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

SEUIL_CONFIANCE = 0.70


# ============================================================
# Configuration TensorFlow
# ============================================================

# Évite que TensorFlow essaie d'utiliser une quantité excessive
# de threads sur le serveur Render.

try:

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

except Exception as erreur:

    print(
        "Configuration TensorFlow :",
        erreur
    )


# ============================================================
# Application Flask
# ============================================================

app = Flask(__name__)


# Limite la taille maximale d'une requête
# 10 Mo suffisent largement pour une image webcam.

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# Chargement du modèle
# ============================================================

print("==========================================")
print("Chargement du modèle...")
print("==========================================")

modele = tf.keras.models.load_model(
    CHEMIN_MODELE,
    compile=False
)

print("Modèle chargé avec succès !")


# ============================================================
# Chargement des classes
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
# Correspondance classe → poubelle
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
        "Poubelle grise — non recyclable",
}


# ============================================================
# Préparation de l'image
# ============================================================

def preparer_image(image_bgr):

    # Conversion BGR → RGB

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Redimensionnement

    image_redim = cv2.resize(
        image_rgb,
        TAILLE_IMAGE,
        interpolation=cv2.INTER_AREA
    )

    # Conversion float32

    image_array = image_redim.astype(
        np.float32
    )

    # Ajout de la dimension batch

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prétraitement MobileNetV2

    image_array = preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# Page principale
# ============================================================

@app.route("/")
def accueil():

    return render_template(
        "index.html"
    )


# ============================================================
# Route de test
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "success": True,

        "message": "Serveur opérationnel",

        "modele": "chargé",

        "classes": classes

    })


# ============================================================
# API de prédiction
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predire():

    try:

        print("")
        print("==========================================")
        print("Nouvelle demande de prédiction")
        print("==========================================")


        # ----------------------------------------------------
        # Vérifier la présence de l'image
        # ----------------------------------------------------

        if "image" not in request.files:

            print(
                "ERREUR : aucune image reçue"
            )

            return jsonify({

                "success": False,

                "message":
                    "Aucune image reçue."

            }), 400


        fichier = request.files["image"]


        # ----------------------------------------------------
        # Vérifier le nom du fichier
        # ----------------------------------------------------

        if fichier.filename == "":

            return jsonify({

                "success": False,

                "message":
                    "Le fichier image est vide."

            }), 400


        # ----------------------------------------------------
        # Lire l'image
        # ----------------------------------------------------

        image_bytes = fichier.read()


        print(
            "Taille image reçue :",
            len(image_bytes),
            "octets"
        )


        if len(image_bytes) == 0:

            return jsonify({

                "success": False,

                "message":
                    "L'image reçue est vide."

            }), 400


        # ----------------------------------------------------
        # Transformer en tableau NumPy
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # Décoder l'image
        # ----------------------------------------------------

        image_bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image_bgr is None:

            print(
                "ERREUR : OpenCV ne peut pas lire l'image"
            )

            return jsonify({

                "success": False,

                "message":
                    "Impossible de lire l'image."

            }), 400


        print(
            "Image décodée :",
            image_bgr.shape
        )


        # ----------------------------------------------------
        # Prétraitement
        # ----------------------------------------------------

        entree = preparer_image(
            image_bgr
        )


        print(
            "Image préparée pour le modèle."
        )


        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        print(
            "Début de la prédiction..."
        )


        predictions = modele.predict(
            entree,
            verbose=0
        )[0]


        print(
            "Prédiction terminée."
        )


        # ----------------------------------------------------
        # Trouver la classe
        # ----------------------------------------------------

        index_classe = int(
            np.argmax(predictions)
        )


        classe = classes[
            index_classe
        ]


        confiance = float(
            predictions[index_classe]
        )


        print(
            "Classe :",
            classe
        )

        print(
            "Confiance :",
            confiance
        )


        # ----------------------------------------------------
        # Vérifier le seuil
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Réponse
        # ----------------------------------------------------

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
            "Réponse :",
            reponse
        )


        return jsonify(
            reponse
        )


    except Exception as erreur:

        print("")
        print("==========================================")
        print("ERREUR PREDICTION")
        print("==========================================")
        print(
            repr(erreur)
        )


        return jsonify({

            "success": False,

            "message":
                "Erreur pendant la prédiction : "
                + str(erreur)

        }), 500


# ============================================================
# Gestion fichier trop volumineux
# ============================================================

@app.errorhandler(413)
def fichier_trop_grand(erreur):

    return jsonify({

        "success": False,

        "message":
            "Image trop volumineuse."

    }), 413


# ============================================================
# Lancement local
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False
    )