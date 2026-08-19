# Sujet 11 - Tri automatique des déchets

Application de classification d'images basée sur MobileNetV2. Le modèle reconnaît
les catégories `cardboard`, `glass`, `metal`, `paper`, `plastic` et `trash`, puis
propose la poubelle correspondante.

## Structure du projet

| Élément | Rôle |
|---|---|
| `app.py` | Application web Flask avec caméra du navigateur |
| `templates/index.html` | Interface web de capture et de prédiction |
| `app/app_webcam.py` | Démonstrateur OpenCV avec webcam locale |
| `app/modele_tri_dechets.keras` | Modèle entraîné |
| `app/classes.txt` | Liste des classes dans l'ordre du modèle |
| `notebook/` | Notebook d'entraînement et d'évaluation |
| `resultats/` | Courbes et résultats d'évaluation |
| `requirements.txt` | Dépendances de l'application |

## Installation

Depuis la racine du projet :

```bash
cd /home/hawa-niass/Documents/tri-dechets
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

L'installation peut être longue, car TensorFlow est une dépendance volumineuse.
Utilise toujours le même interpréteur Python pour installer les paquets et lancer
l'application.

## Application web

L'application web utilise la webcam du navigateur et envoie l'image au endpoint
`/predict`.

```bash
cd /home/hawa-niass/Documents/tri-dechets
source .venv/bin/activate
python app.py
```

Ouvre ensuite <http://127.0.0.1:5000> dans le navigateur. Le endpoint
<http://127.0.0.1:5000/health> permet de vérifier que le serveur et le modèle sont
chargés.

La webcam du navigateur nécessite une autorisation. Sur certains navigateurs,
l'accès est autorisé uniquement depuis `localhost` ou `127.0.0.1`.

## Démonstrateur webcam OpenCV

Le script webcam se trouve dans `app/`. Lance-le depuis ce dossier afin que les
chemins vers le modèle et `classes.txt` soient trouvés :

```bash
cd /home/hawa-niass/Documents/tri-dechets/app
source ../.venv/bin/activate
python app_webcam.py
```

Appuie sur `q` pour quitter et sur `r` pour remettre le compteur à zéro.

## Entraînement

Le notebook peut être exécuté dans Google Colab avec un GPU :

1. Importer le notebook présent dans `notebook/`.
2. Activer un GPU dans les paramètres d'exécution.
3. Exécuter les cellules dans l'ordre.
4. Copier `modele_tri_dechets.keras` et `classes.txt` dans `app/`.

Les fichiers `courbes_apprentissage.png` et `matrice_confusion.png` peuvent être
conservés dans `resultats/` pour le rapport.

## Dépannage

### `ModuleNotFoundError: No module named 'cv2'`

L'environnement Python actif ne contient pas les dépendances. Active `.venv`,
puis relance :

```bash
python -m pip install -r requirements.txt
```

Vérifie ensuite :

```bash
python -c "import flask, cv2, numpy, tensorflow; print('Dépendances OK')"
```

### Fichier introuvable

`app.py` doit être lancé depuis la racine du projet. `app_webcam.py` doit être
lancé depuis le dossier `app/`, car ses chemins de modèle sont relatifs.

### Webcam OpenCV indisponible

La version `opencv-python-headless` convient au serveur web, mais ne fournit pas
les fenêtres graphiques nécessaires à `cv2.imshow`. Si le démonstrateur OpenCV
échoue sur `imshow`, installe la variante desktop :

```bash
python -m pip uninstall opencv-python-headless
python -m pip install opencv-python
```

## Préparation du mémoire

- **Données** : effectifs par classe et par split affichés dans le notebook.
- **Méthodologie** : MobileNetV2, tête dense, gel initial puis fine-tuning.
- **Résultats** : courbes, matrice de confusion et classification report.
- **Discussion** : analyser les confusions entre classes proches, notamment
  `glass`/`plastic` et `metal`/`plastic`.
- **Outil** : présenter l'interface Flask et le démonstrateur webcam.

## Ce qu'il reste à faire pour le mémoire

Le plan type du sujet (voir le mini-mémoire) attend 9 parties. Voici ce que tu peux déjà remplir avec ce que ce pipeline te donne :

- **Données** : le notebook affiche le nombre d'images par classe/split → utilise ces chiffres
- **Méthodologie** : architecture MobileNetV2 + tête dense, deux phases d'entraînement (tête gelée puis fine-tuning), Adam, categorical crossentropy, poids de classe pour le déséquilibre
- **Résultats** : `courbes_apprentissage.png`, `matrice_confusion.png`, le `classification_report` (précision/rappel/F1 par classe)
- **Discussion** : regarde quelles classes se confondent le plus dans la matrice de confusion (souvent glass/plastic ou metal/plastic) — c'est une bonne base d'analyse d'erreurs
- **État de l'art** : à faire à part — 8 à 12 références sur le tri de déchets par deep learning (je peux t'aider à en trouver si tu veux)
- **L'outil** : capture d'écran de `app_webcam.py` en fonctionnement + explication de l'architecture logicielle

Dis-moi quand tu veux qu'on attaque la rédaction du mémoire (Word) ou les slides de soutenance (PowerPoint) — je peux te les structurer section par section.
