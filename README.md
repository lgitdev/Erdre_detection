# Projet PI2 – Fédération des Amis de l’Erdre

Dans le cadre de notre projet PI² avec la Fédération des Amis de l'Erdre, nous avons réalisé ces codes qui permettent de traiter des images aériennes de l'Erdre et d'en extraire les contours. L'assemblage des photos entre elles (matching + stitching) nous permet d'étudier l'évolution de l'érosion au fur et à mesure des années.

Le code fourni est principalement divisé en trois modules :

1. **`Erdre_1 1.py`** – Définit les classes et structures de données pour gérer un ensemble d’images (classe `Image` et classe `Dataset`).
2. **`matching 2.py`** – Contient les fonctions dédiées à la détection de points clés, au filtrage, et au calcul d’homographies (ORB/SIFT, RANSAC, etc.).
3. **`Test programme de Louis - mask + contours.py`** – Programme pour générer des masques de la rivière, tracer les contours.

Ce document a pour but de décrire le fonctionnement et l'usage de ces différents scripts de la manière la plus détaillée possible. 
Vous retrouverez également des indications sur l’organisation du projet, les prérequis et la méthode d’installation.

---

## Table des matières

1. Architecture générale
2. Prérequis & Installation
3. Détail des modules
   1. Module `Erdre_1 1.py`
   2. Module `matching 2.py`
   3. Script `Test programme de Louis - mask + contours.py`
4. FLux de travail
5. Remerciements

---

## Architecture générale

Notre projet se base sur un ensemble d'images aériennes numérisées. Nous suivons les étapes suivantes : 

1. **Chargement des métadonnées** depuis un fichier Excel (`database.xlsx`) : angles de rotation, ID de la mission, date, etc.
2. **Création d’instances `Image`** pour chaque fichier `.tif` détecté dans un répertoire, en associant les informations de rotation, l’année, etc.
3. **Organisation de ces images** au sein d’une classe `Dataset` :
    - Possibilité de prétraiter chaque image (rotation, recadrage, suppression de bordures noires).
    - Lancement de la procédure de *matching* entre paires d’images (ORB/SIFT + RANSAC, calcul des homographies, etc.).
4. **Scripts de test et visualisation** :
    - Génération de *masks* permettant d’isoler la zone de la rivière (scripts basés sur le traitement d’images via Sobel, Laplacien, seuillage, etc.).
    - Calcul des contours et superposition sur l’image d’origine.

**Schéma résumé** :

![image](https://github.com/user-attachments/assets/dea00e3e-f6b1-432d-abb7-2afaf5a41615)


# Prérequis & Installation

## Environnement Python

Le code est écrit en Python 3. Pour le lancer, nous avons besoin d'installer plusieurs bibliothèques

## Bibliothèques requises

- **OpenCV (cv2)** – Traitement d’images, détection de caractéristiques, RANSAC, etc.
- **NumPy** – Manipulation de tableaux numériques.
- **pandas** – Gestion des données tabulaires (lecture du fichier Excel des métadonnées).
- **matplotlib** – Visualisation (pour certains tracés intermédiaires).
- **scipy** – Fonctions avancées (ex. `ndimage.rotate`).
- **scikit-image (skimage)** – Pour le calcul de matrices de co-occurrence ou d’autres transformations avancées.

Vous pouvez installer ces librairies manuellement, par exemple :

```bash
pip install opencv-python numpy pandas matplotlib scipy scikit-image
```

# Détails des modules

## Module `Erdre_1 1.py`

Dans ce module on retrouve deux classes majeures : **Image** et **Dataset**.
On va ici chercher à charger des images '.tif' (qui sont les images disponibles sur IGN) et la prétraiter pour avoir toutes les informations nécessaires pour le traitement. 

---

### 1. Classe `Image`

Représente les images chargées

#### Attributs principaux :
- **`self.PATH`** : Chemin complet vers le fichier image `.tif`.
- **`self.name`** : Nom du fichier sans extension.
- **`self.year`**, **`self.mission`**, **`self.id`** : Informations extraites du nom de fichier (format attendu : `AAAA_MISSION_ID`).
- **`self.angle`** : Angle de rotation récupéré dans le fichier Excel.
- **`self.image`** : L’image chargée en mémoire (matrice OpenCV).

#### Méthodes :
- **`filter_file(file_name)`** : Retire l’extension `.tif` ou `.tiff`.
- **`dismantle_name()`** : Décompose le nom en année, mission et id.
- **`get_infos(path_to_db)`** : Lit l’Excel pour récupérer l’angle de rotation correspondant.
- **`image_prep(...)`** : Pipeline complet de préparation comprenant :
    - `scaling` : redimensionnement.
    - `rotating` : rotation de l’image selon l’angle.
    - `cleaning_borders` : suppression des bordures noires.
- **`scaling(scale, inplace)`** : Redimensionne l’image d’un facteur `scale`.
- **`rotating(angle, inplace)`** : Tourne l’image autour de son centre selon l'angle.
- **`cleaning_borders(threshold, inplace)`** : Supprime les pixels trop sombres sur les bordures selon un seuil.

---

### 2. Classe `Dataset`

Représente un ensemble d’images à analyser.

#### Attributs principaux :
- **`self.DIR`** : Chemin vers le dossier contenant les images `.tif`.
- **`self.path_to_db`** : Chemin vers l’Excel des métadonnées.
- **`self.images`** : Liste d’objets `Image`.
- **`self.names`** : Liste des noms des images.

#### Méthodes :
- **`load()`** : Parcourt le répertoire `self.DIR` pour charger toutes les images `.tif` en tant qu’objets `Image`.
- **`preprocessing(homo_scale, border_threshold)`** : Applique à toutes les images un même facteur de redimensionnement, nettoyage des bordures, et rotation selon leur angle spécifique (`Image.angle`).
- **`match(seuil_ransac)`** :
    - Parcourt toutes les paires d’images du dataset.
    - Exécute les fonctions du module `matching` pour détecter les points clés (ORB), les filtrer, puis calcule l’homographie via RANSAC.
    - Retourne un dictionnaire contenant les homographies et les masques associés à chaque paire.
- **`get_image(name)`** : Recherche l’image par son nom et retourne la matrice correspondante (OpenCV).

![image](https://github.com/user-attachments/assets/e71a9584-5af8-4676-b678-195f3a1fe46f)


---

## Module `matching 2.py`

On regroupe ici tout ce qui touche au matching, c'est à dire à la détection de points de similitude entre les 2 images. Le matching nous permet de savoir automatiquement si 2 images sont côté à côté (c'est à dire savoir si on peut ou non les superposer - dans le stitching - avec un décalage)
Il contient :

### Fonctions de détection de points clés :

- **`init_matching_orb(img1, img2)`** : Utilise la méthode ORB pour détecter et matcher les points clés entre deux images.
- **`init_matching_sift(img1, img2)`** : Utilise la méthode SIFT pour détecter et matcher les points clés entre deux images.  

### Fonctions de filtrage des correspondances :

- **`filtre_distance(matches)`** : Élimine les correspondances trop éloignées de la distance moyenne.
- **`find_parallel_groups(angles, tolerance)`** : Regroupe les angles similaires (considérés parallèles) selon une tolérance spécifiée.
- **`filtre_parallel_matches(kpt1, kpt2, matches, tolerance)`** :  Conserve seulement les correspondances ayant des angles jugés parallèles.
- **`create_artificial_match(kpt1, kpt2, matches, match_number)`** :  Génère artificiellement des correspondances supplémentaires lorsqu’il n’y en a pas assez pour calculer une homographie.

![image](https://github.com/user-attachments/assets/1f2d452d-0f11-4a4c-82ae-da6a8b752f5b)


### Fonctions de calcul d’homographie / transformations :

- **`ransac(kpt1, kpt2, matches, seuil_ransac)`** :  Tente de trouver la matrice d’homographie `H` via `cv.findHomography` avec la méthode **RANSAC**.
- **`Affine(kpt1, kpt2, matches)`** :  Calcule une transformation affine (nécessite au minimum 3 points).

### Fonctions utilitaires :

- **`extract_rotation_angle(H)`** :  Extrait l’angle de rotation depuis une matrice d’homographie calculée.
- **`matrice_angles(ANGLES, names)`** :  Crée et affiche une matrice de chaleur (*heatmap*) représentant les angles estimés entre les paires d’images.

**NB :** Les algorithmes **RANSAC** et **ORB** sont particulièrement utiles pour assembler des images présentant des décalages ou rotations significatives.

![image](https://github.com/user-attachments/assets/1d28be8a-a285-4274-8ce7-8509c1388296)

---

## Script Test : `programme de Louis - mask + contours.py`

Ce script permet de détection les contours (segmentation) autour de la rivière dans une image. 

### Fonctions principales :

#### 1. **`remove_black_borders(image, threshold)`**
- Détecte et supprime les bandes noires (ou très sombres) entourant une image.
- Calcule la proportion de pixels sombres par ligne et colonne afin de déterminer les limites à conserver.

#### 2. **`create_river_mask(image, kernel, min_area)`**
- Convertit l’image en niveaux de gris.
- Applique des filtres Sobel et Laplacian pour détecter les bords et reliefs.
- Combine ces informations, applique un seuillage inversé, puis effectue des opérations morphologiques (érosion, dilatation) pour affiner le masque.
- Élimine les petites zones (inférieures à `min_area` pixels).

#### 3. **`process_image(input_dir, output_dir, new_width, new_height, year)`**
- Parcourt le dossier `input_dir`, charge chaque image et retire ses bordures noires.
- Potentiellement applique une rotation spécifique à l’image (via une liste prédéfinie d’angles : `rot_angle`).
- Calcule ensuite le masque binaire de la rivière (`create_river_mask`) et dessine les contours détectés sur l’image originale.
- Sauvegarde les résultats (image annotée et masque binaire correspondant) dans le dossier `output_dir`.

![image](https://github.com/user-attachments/assets/f613af04-3d27-4ce4-a8fc-5a8251704d85)

---

# Exemple de flux de travail

Vous pouvez suivre les instructions du fichier `.ipynb` pour éxécuter tous ces codes et avoir les résultats attendus, en fonction de l'année choisie.

---

# Remerciements

Merci à la Fédération des Amis de l'Erdre pour ce projet enrichissant et l'accompagnement tout au long de l'année dans sa réalisation, en particulier à Gwendoline MONNIER, coordinatrice du projet.
Merci également à Catherine BASKIOTIS, notre référente ESILV, pour son support technique et son accompagnement dans les grandes étapes de résolution du projet.

**Contributeurs du projet :**

- Soline VIRICEL, ESILV A4 EVD
- Romane LAURENT, ESILV A4 EVD
- Marc MONIN, ESILV A4 CCC
- Léo DEMELLE, ESILV A4 DIA
- Louis GINDRE, ESILV A4 DIA

