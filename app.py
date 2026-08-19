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

@app.route(
    "/predict",
    methods=["POST"]
)
def predire():

    print("")
    print("========================================")
    print("REQUETE /predict RECUE")
    print("========================================")

    try:

        # ----------------------------------------------------
        # Vérifier le fichier
        # ----------------------------------------------------

        print(
            "Fichiers reçus :",
            list(request.files.keys())
        )

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


        print(
            "Nom du fichier :",
            fichier.filename
        )


        # ----------------------------------------------------
        # Lire les bytes
        # ----------------------------------------------------

        image_bytes = fichier.read()


        print(
            "Taille image reçue :",
            len(image_bytes),
            "bytes"
        )


        if not image_bytes:

            print(
                "ERREUR : image vide"
            )

            return jsonify({

                "success": False,

                "message":
                    "L'image reçue est vide."

            }), 400


        # ----------------------------------------------------
        # Convertir en tableau numpy
        # ----------------------------------------------------

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # Décoder JPEG
        # ----------------------------------------------------

        image_bgr = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image_bgr is None:

            print(
                "ERREUR : impossible de décoder l'image"
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

        print(
            "Prétraitement..."
        )

        entree = preparer_image(
            image_bgr
        )


        print(
            "Entrée modèle :",
            entree.shape
        )


        # ----------------------------------------------------
        # Prédiction
        # ----------------------------------------------------

        print(
            "Lancement de la prédiction..."
        )

        predictions = modele.predict(
            entree,
            verbose=0
        )[0]


        print(
            "Prédiction terminée."
        )

        print(
            "Predictions :",
            predictions
        )


        # ----------------------------------------------------
        # Classe
        # ----------------------------------------------------

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
            "Classe :",
            classe
        )

        print(
            "Confiance :",
            confiance
        )


        # ----------------------------------------------------
        # Seuil de confiance
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
            "Réponse envoyée :",
            reponse
        )

        print(
            "========================================"
        )


        return jsonify(reponse)


    except Exception as erreur:

        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERREUR /predict")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print(
            type(erreur).__name__
        )

        print(
            str(erreur)
        )

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )


        return jsonify({

            "success": False,

            "message":
                "Erreur pendant la prédiction : "
                + str(erreur)

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