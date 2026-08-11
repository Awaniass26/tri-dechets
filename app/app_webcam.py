"""
Démonstrateur webcam — Tri automatique des déchets (Sujet 11)
================================================================
Classe en direct le déchet présenté devant la webcam, affiche la confiance,
la poubelle recommandée, et un compteur de session par classe.

Utilisation :
    1. Place ce fichier dans le même dossier que :
       - modele_tri_dechets.keras
       - classes.txt
    2. Installe les dépendances : pip install -r requirements.txt
    3. Lance : python app_webcam.py
    4. Appuie sur 'q' pour quitter, 'r' pour réinitialiser le compteur.
"""

import collections
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
chemin_modele = "modele_tri_dechets.keras"
chemin_classes = "classes.txt"
taille_image = (224, 224)
seuil_confiance = 0.55          # en dessous, on affiche "incertain"
intervalle_prediction = 0.4     # secondes entre deux prédictions (évite de surcharger le CPU)

# Indication de la poubelle associée à chaque classe TrashNet.
# À adapter si besoin selon les consignes de tri locales.
poubelle_recommandee = {
    "cardboard": "Poubelle jaune (recyclable - carton)",
    "glass": "Poubelle verte (verre)",
    "metal": "Poubelle jaune (recyclable - métal)",
    "paper": "Poubelle jaune (recyclable - papier)",
    "plastic": "Poubelle jaune (recyclable - plastique)",
    "trash": "Poubelle grise (non recyclable)",
}


def charger_classes(chemin):
    with open(chemin, "r") as fichier:
        return [ligne.strip() for ligne in fichier if ligne.strip()]


def preparer_image(image_bgr):
    """Convertit une image OpenCV (BGR) en entrée prête pour le modèle."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_redim = cv2.resize(image_rgb, taille_image)
    image_array = np.expand_dims(image_redim, axis=0).astype("float32")
    return preprocess_input(image_array)


def dessiner_overlay(frame, classe, confiance, compteur_session):
    """Dessine les informations de prédiction et le compteur sur l'image."""
    hauteur, largeur = frame.shape[:2]

    # Bandeau semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (largeur, 90), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if confiance >= seuil_confiance:
        texte_classe = f"{classe.upper()} ({confiance:.0%})"
        texte_poubelle = poubelle_recommandee.get(classe, "Poubelle inconnue")
        couleur = (0, 200, 0)
    else:
        texte_classe = "Incertain..."
        texte_poubelle = "Rapproche l'objet ou améliore l'éclairage"
        couleur = (0, 165, 255)

    cv2.putText(frame, texte_classe, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, couleur, 2)
    cv2.putText(frame, texte_poubelle, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Compteur de session, en bas de l'image
    y = hauteur - 15 * len(compteur_session) - 10
    for nom_classe, total in compteur_session.items():
        cv2.putText(frame, f"{nom_classe}: {total}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 15

    cv2.putText(frame, "q: quitter   r: reinitialiser", (largeur - 260, hauteur - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return frame


def main():
    print("Chargement du modèle...")
    modele = tf.keras.models.load_model(chemin_modele)
    class_names = charger_classes(chemin_classes)
    print("Classes chargées :", class_names)

    compteur_session = collections.OrderedDict((c, 0) for c in class_names)
    derniere_classe_comptee = None
    derniere_prediction_time = 0.0
    classe_actuelle, confiance_actuelle = "", 0.0

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Impossible d'accéder à la webcam. Vérifie qu'aucune autre application ne l'utilise.")

    print("Démo lancée. Appuie sur 'q' pour quitter.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Erreur de lecture webcam.")
            break

        maintenant = time.time()
        if maintenant - derniere_prediction_time >= intervalle_prediction:
            entree = preparer_image(frame)
            predictions = modele.predict(entree, verbose=0)[0]
            index_classe = int(np.argmax(predictions))
            classe_actuelle = class_names[index_classe]
            confiance_actuelle = float(predictions[index_classe])
            derniere_prediction_time = maintenant

            # On incrémente le compteur seulement si la classe change,
            # pour éviter de compter plusieurs fois le même objet immobile.
            if confiance_actuelle >= seuil_confiance and classe_actuelle != derniere_classe_comptee:
                compteur_session[classe_actuelle] += 1
                derniere_classe_comptee = classe_actuelle

        frame_affichee = dessiner_overlay(frame, classe_actuelle, confiance_actuelle, compteur_session)
        cv2.imshow("Tri automatique des dechets - Demo", frame_affichee)

        touche = cv2.waitKey(1) & 0xFF
        if touche == ord("q"):
            break
        elif touche == ord("r"):
            compteur_session = collections.OrderedDict((c, 0) for c in class_names)
            derniere_classe_comptee = None
            print("Compteur réinitialisé.")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
