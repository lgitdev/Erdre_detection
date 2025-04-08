import cv2 as cv # OpenCV, bibliothèque spécialisée dans le traitement des images
import numpy as np
import matplotlib.pyplot as plt


        ############
        ##Matching##
        ############

def init_matching_orb(img1, img2):
    """
    Prend en argument deux images cv2, trouvent des points particuliers dans chacune d'entre elles puis essaie de coupler les paires
    de points clés qui se ressemblent le plus.
    Renvoie les points clés des deux images et les pairs couplées.
    """
    # Détecter les points clés et les descripteurs avec ORB
    orb = cv.ORB_create()
    kpt1, desc1 = orb.detectAndCompute(img1, None)
    kpt2, desc2 = orb.detectAndCompute(img2, None)

    # Matcher les points clés avec BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kpt1, kpt2, matches

def init_matching_sift(img1, img2):
    """
    Prend en argument deux images cv2, trouvent des points particuliers dans chacune d'entre elles puis essaie de coupler les paires
    de points clés qui se ressemblent le plus.
    Renvoie les points clés des deux images et les pairs couplées.
    """
    # Détecter les points clés et les descripteurs avec SIFT
    sift = cv.SIFT_create()
    kpt1, desc1 = sift.detectAndCompute(img1, None)
    kpt2, desc2 = sift.detectAndCompute(img2, None)

    # Matcher les points clés avec BFMatcher
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kpt1, kpt2, matches




        ###########
        ##Filtres##
        ###########



def filtre_distance(matches):
    """
    Prend en entrée des matchs BFMatcher et renvoie seulement ceux qui passe le fiiltre.
    Ici on définie une distance maximal pour accepter des couples de points clés comme pertinents dans BFMatcher.
    """
    # Exemple pour calculer la distance moyenne et l'écart type
    distances = [m.distance for m in matches]
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    # Définir un seuil
    threshold = mean_distance + std_distance
    # Filtrer les matches
    return [m for m in matches if m.distance < threshold]

def find_parallel_groups(angles, tolerance=3):
    """
    Prends en entrée une liste des angles des couples et éventuellement une tolérance du parallélisme.
    Crée des groupes pour chaque angle unique (sous le seuil de tolérance) pour obtenir le plus fréquent.
    Renvoie le groupe avec le plus d'occurences
    """
    angle_groups = {}
    
    for angle in angles:
        found_group = False
        for key in angle_groups.keys():
            if abs(angle - key) < tolerance:
                angle_groups[key].append(angle)
                found_group = True
                break
        if not found_group:
            angle_groups[angle] = [angle]

    best_group = max(angle_groups.values(), key=len, default=[])
    return best_group

def filtre_parallel_matches(kpt1, kpt2, matches, tolerance=0.1):
    """
    Prends en entrée les points clés, les couples et éventuellement une tolérance sur la qualité du parallélisme.
    Fouille tous les couples pour trouver un groupe de couples parallèles.
    Renvoie le groupe de couples filtré.
    """
    angles = []

    for match in matches:
        pt1 = np.array(kpt1[match.queryIdx].pt)
        pt2 = np.array(kpt2[match.trainIdx].pt)
        delta = pt2 - pt1
        angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
        angles.append((angle, match))

    best_group = find_parallel_groups([a[0] for a in angles], tolerance)
    
    if len(best_group) < 2:
        return []

    filtered_matches = [match for angle, match in angles if angle in best_group]
    return filtered_matches

def create_artificial_match(kpt1, kpt2, matches, match_number=4):
    """
    Prends en entrée les points clés, les couples et éventuellement le nombre final de couples.
    Duplique le premier couple en le décalant de quelques pixels dans chaque image juqu'au nombre désiré de couples.
    Renvoie les points clés et les couples.
    """
    original_match = matches[0]
    original_kp1 = kpt1[original_match.queryIdx]
    original_kp2 = kpt2[original_match.trainIdx]
    i=2

    while len(matches) < match_number:
        new_kp1 = cv.KeyPoint(original_kp1.pt[0] + i, original_kp1.pt[1] + i, 
                            original_kp1.size, original_kp1.angle, original_kp1.response, 
                            original_kp1.octave, original_kp1.class_id)
        
        new_kp2 = cv.KeyPoint(original_kp2.pt[0] + i, original_kp2.pt[1] + i, 
                            original_kp2.size, original_kp2.angle, original_kp2.response, 
                            original_kp2.octave, original_kp2.class_id)
        
        i += 2
        kpt1 = kpt1 + (new_kp1,)
        kpt2 = kpt2 + (new_kp2,)
        new_match = cv.DMatch((len(kpt1)-1), (len(kpt2)-1), original_match.distance)
        matches.append(new_match)
    
    return (kpt1, kpt2, matches)

def ransac(kpt1, kpt2, matches, seuil_ransac = 1.0):
    """
    Prend en entrée les points clés et les couples (minimum 4).
    Essaie de trouver une homographie (ie une fonction qui conserve les angles, donc translation/rotation/changement d'échelle) 
    qui convienne à un maximum de couples et supprime les autres. Prend aussi en compte les changements de perspective 
    Renvoie l'homographie sous forme :
    D'une matrice 3x3 avec le carré 2x2 en haut à gauche responsable de la rotation, scaling et shearing, 
    les deux points [1,3] et [2,3] sont les translations selon x et y, les deux points [3,1], [3,2] sont les changements de perspectives
    et [3,3] est normalisé à 1.
    Un masque qui indique quels couples ont été filtré par l'algorithme (basé sur la distance)
    """
    x = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    y = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(x, y, cv.RANSAC, seuil_ransac)
    return H, mask

def Affine(kpt1, kpt2, matches):
    """
    Prend en entrée les points clés et les couples (minimum 3).
    Trouve une homographie (ie une fonction qui conserve les angles, donc translation/rotation/changement d'échelle) 
    qui convienne aux 3 premiers couples.
    Renvoie l'homographie sous forme d'une matrice 2x3 avec les deux points [1,3] et [2,3] sont les translations selon x et y 
    et le reste qui est responsable de la rotation, scaling et shearing.
    """
    x = np.float32([kpt1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    y = np.float32([kpt2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    x = x[:3]
    y = y[:3]
    H = cv.getAffineTransform(x, y)
    return H


        ##############
        ##Evaluateur##
        ##############


def extract_rotation_angle(H):
    """
    Prends en entrée la matrice de transformation.
    Extrait la rotation (en degrées) de l'homographie entre les deux images.
    """
    if H is None:
        return None  # No valid homography

    r11, r12 = H[0, 0], H[0, 1]
    r21, r22 = H[1, 0], H[1, 1]

    theta = np.arctan2(r21, r11)  # Compute rotation in radians
    angle = np.degrees(theta)  # Convert to degrees

    return angle

def matrice_angles(ANGLES, names):
    """
    Prends en entrée un dictionnaire des valeurs d'angles et la liste des noms des images.
    Crée une matrice a partir des valeurs d'angles extraites du matching des images.
    """
    # Get the number of images
    n = len(names)
    
    # Create a matrix to hold the angles
    angle_matrix = np.zeros((n, n))
    
    # Populate the angle matrix with the values from the ANGLES dictionary
    for i in range(n):
        for j in range(i + 1, n):
            image_1_name = names[i]
            image_2_name = names[j]
            if (image_1_name, image_2_name) in ANGLES:
                angle_matrix[i, j] = ANGLES[(image_1_name, image_2_name)]
                angle_matrix[j, i] = ANGLES[(image_1_name, image_2_name)]  # Symmetric
    
    # Plot the matrix using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(angle_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Rotation Angle')
    plt.title('Rotation Angle Matrix')

    # Set the labels for the x and y axes (image names)
    plt.xticks(ticks=np.arange(n), labels=names, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(n), labels=names)

    for i in range(n):
        for j in range(n):
            if angle_matrix[i, j] != 0:  # Only display values where there's an angle
                 plt.text(j, i, f"{angle_matrix[i, j]:.2f}", ha='center', va='center', color='black')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

