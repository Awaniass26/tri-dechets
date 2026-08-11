# Sujet 11 — Tri automatique des déchets (recyclage)

## Contenu de ce dossier

| Fichier | Rôle | Où l'utiliser |
|---|---|---|
| `tri_dechets_entrainement.ipynb` | Notebook complet : dataset → modèle → évaluation | Google Colab (GPU) |
| `app_webcam.py` | Démonstrateur webcam en temps réel | Ta machine locale |
| `requirements.txt` | Dépendances Python pour la démo locale | Ta machine locale |

## Étapes

### 1. Entraînement (Google Colab)
1. Va sur https://colab.research.google.com et importe `tri_dechets_entrainement.ipynb`
2. Active le GPU : *Exécution > Modifier le type d'exécution > GPU (T4)*
3. Exécute les cellules dans l'ordre (Exécution > Tout exécuter). Compte 20-40 min au total.
4. À la fin, télécharge deux fichiers depuis le panneau de fichiers à gauche de Colab :
   - `modele_tri_dechets.keras`
   - `classes.txt`
   - Garde aussi `courbes_apprentissage.png` et `matrice_confusion.png` : tu en auras besoin pour le mémoire (section Résultats).

### 2. Démo locale (ta machine)
1. Place `modele_tri_dechets.keras` et `classes.txt` téléchargés dans le même dossier que `app_webcam.py`
2. Installe les dépendances :
   ```
   pip install -r requirements.txt
   ```
3. Lance la démo :
   ```
   python app_webcam.py
   ```
4. Présente un objet devant la webcam (carton, bouteille plastique, canette, feuille de papier...). L'app affiche la classe prédite, la confiance, et la poubelle recommandée, avec un compteur en bas de l'écran.

### 3. Dépôt Git
Structure conseillée pour ton repo :
```
tri-dechets/
├── notebook/
│   └── tri_dechets_entrainement.ipynb
├── app/
│   ├── app_webcam.py
│   ├── requirements.txt
│   ├── modele_tri_dechets.keras
│   └── classes.txt
├── resultats/
│   ├── courbes_apprentissage.png
│   └── matrice_confusion.png
├── memoire/
│   └── memoire_tri_dechets.pdf
└── README.md
```

## Ce qu'il reste à faire pour le mémoire

Le plan type du sujet (voir le mini-mémoire) attend 9 parties. Voici ce que tu peux déjà remplir avec ce que ce pipeline te donne :

- **Données** : le notebook affiche le nombre d'images par classe/split → utilise ces chiffres
- **Méthodologie** : architecture MobileNetV2 + tête dense, deux phases d'entraînement (tête gelée puis fine-tuning), Adam, categorical crossentropy, poids de classe pour le déséquilibre
- **Résultats** : `courbes_apprentissage.png`, `matrice_confusion.png`, le `classification_report` (précision/rappel/F1 par classe)
- **Discussion** : regarde quelles classes se confondent le plus dans la matrice de confusion (souvent glass/plastic ou metal/plastic) — c'est une bonne base d'analyse d'erreurs
- **État de l'art** : à faire à part — 8 à 12 références sur le tri de déchets par deep learning (je peux t'aider à en trouver si tu veux)
- **L'outil** : capture d'écran de `app_webcam.py` en fonctionnement + explication de l'architecture logicielle

Dis-moi quand tu veux qu'on attaque la rédaction du mémoire (Word) ou les slides de soutenance (PowerPoint) — je peux te les structurer section par section.
