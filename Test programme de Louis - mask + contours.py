import cv2
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from skimage.feature import graycomatrix, graycoprops
#Base du code : etudiant ESILV année 4 2024-2025

########____________CROP IMAGE__________#######
def remove_black_borders(image, threshold=0.5):
    '''
    Supprime les bordures noires d'une image en détectant les lignes et colonnes
    où plus de `threshold` proportion de pixels sont noirs.

    Paramètres :
    - image (ndarray) : image d'entrée au format BGR.
    - threshold (float) : proportion maximale de noir tolérée (0.0 - 1.0).

    Retour :
    - ndarray : image recadrée sans les bordures noires.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Détection des lignes noires
    row_black_ratio = np.sum(gray == 0, axis=1) / width
    col_black_ratio = np.sum(gray == 0, axis=0) / height

    # Trouver les indices des premières et dernières lignes/colonnes non noires
    top = np.argmax(row_black_ratio < threshold)
    bottom = height - np.argmax(row_black_ratio[::-1] < threshold)
    left = np.argmax(col_black_ratio < threshold)
    right = width - np.argmax(col_black_ratio[::-1] < threshold)

    # Recadrer l'image
    cropped_image = image[top:bottom, left:right]
    return cropped_image


########____________RIVER MASK__________#######
def create_river_mask(image, kernel, min_area=15000):
    '''
    Crée un masque binaire correspondant aux contours de la rivière sur une image.

    La fonction utilise Sobel et Laplacien pour accentuer les reliefs,
    puis applique un seuillage double avec ouverture et fermeture morphologique.

    Paramètres :
    - image (ndarray) : image en couleur.
    - kernel (ndarray) : noyau morphologique à utiliser.
    - min_area (int) : surface minimale pour conserver un composant connexe.

    Retour :
    - ndarray : masque binaire de la rivière.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sobel & Laplacien pour augmenter le contraste
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) 
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    combined_relief = cv2.addWeighted(sobel_magnitude, 5, laplacian, 1, 0)
    
    river_mask = combined_relief
    
    #"""
    list_seuil = np.asarray([175, 178])
    stock_matrix = np.zeros((river_mask.shape[0],river_mask.shape[1],len(list_seuil)))
    
    for j in range(len(list_seuil)):
        # Création du mask
        _, river_mask = cv2.threshold(combined_relief, list_seuil[j], 255, cv2.THRESH_BINARY_INV)
            
        # Erosion & Dilatation pour affiner les contours
    
        # Ouverture 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        dilated_mask = river_mask
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        river_mask = dilated_mask
  
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(river_mask, connectivity=8) #Connectivity 4 ou 8 ne change rien
        cleaned_mask = np.zeros_like(river_mask)

        # Supprimer zones isolées
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255

        river_mask = cleaned_mask
    
        # # Fermeture
        eroded_mask = river_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        river_mask = eroded_mask
        stock_matrix[:,:,j] = river_mask
    
    #Sud = 1er seuil, Nord = 2eme seuil
    river_mask[300:,:] = stock_matrix[300:,:,0]
    river_mask[:300,:] = stock_matrix[:300,:,1]
    #"""
    
    return river_mask

########____________PROCESS IMAGE__________#######
def process_image(input_dir, output_dir, new_width=800, new_height=600, year=2007):
    '''
    Traite toutes les images d'un dossier pour générer les masques de rivières.

    Étapes :
    - Chargement et recadrage de l'image
    - Rotation (selon liste `rot_angle`)
    - Détection des rivières et génération du masque
    - Sauvegarde du masque et de la photo avec contours tracés

    Paramètres :
    - input_dir (str) : dossier contenant les images originales.
    - output_dir (str) : dossier de sortie pour sauvegarder les résultats.
    - new_width, new_height (int) : dimensions cibles (non utilisé ici).
    - year (int) : année utilisée pour nommer les fichiers en sortie.

    Retour :
    - ndarray : dernier masque généré (utile pour tests).
    '''
    os.makedirs(output_dir, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    compteur = 1
    
    # Initialize river_mask with a default value (None or an empty array)
    river_mask = None
    # Adapter ici les bons angles si besoin -> retirer le commentaire de l'année en question 
    #1923 rot_angle = [57, 64, 68, 0, 170]
    #1943 rot_angle = [-76, -83, -93, -95, -94]
    #1944 rot_angle = [89, 89, 89, 89, 89, -87, -87, -87]
    #1947 rot_angle = [3, 3, 5, 5]
    #1951 rot_angle = [86, 86, 86]
    #1953 rot_angle = [-89, -88]
    #1955 rot_angle = [88, 88, -91, -90]
    #1958 rot_angle = [-86, -86, -86]
    #rot_angle = [-93, -93, -93]
    #1963 rot_angle = [-94, 87, 87, 175, 175, -94, -94, -94, -94]
    #1966 rot_angle = [-90, -91, -83, 22, 22, 22]
    #1967 rot_angle = [-91, -91, -90, -90]
    #1969 rot_angle = [3, 3, 3, 3, 3, 3, -178, -178]
    #1970 rot_angle = [-93, -93, -86]
    #1971 rot_angle = [105, 105, 105, 107, 107, 106, 106, 106, 106, 106, 106, 106, 106, 106]
    #1974 rot_angle = [-179, -179]
    #1975 rot_angle = [-89, -89]
    #1977 rot_angle = [-91, -91, 85]
    #1978 rot_angle = [-90, -90, -90, 86, 86, 86]
    #1980 rot_angle = [87, 87, 87, -93, -93]
    #1981 rot_angle = [-73, -73, -73]
    #1983 rot_angle = [87, 87, -94, -94, -94, -180, -180, 0, 0, 0]
    #1984 rot_angle = [89, 89, 89, -93]
    #1988 rot_angle = [-94, 86, 86, 86, -91, -91, -91, -91, -91, -94, -94, -94, 86, 86, 86]
    rot_angle = [0]

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        if not os.path.exists(image_path):
            print(f"Le fichier {image_path} n'existe pas.")
            continue
       
        image = cv2.imread(image_path)

        if image is not None:
            # Redimensionner l'image
            #image = cv2.resize(image, (new_width, new_height))
            image = remove_black_borders(image)
            image = ndimage.rotate(image, rot_angle[compteur-1])           
            
            '''
            plt.figure()
            imshow(image, cmap='gray', vmin=0, vmax=255)
            '''

            # Créer le masque pour la rivière
            river_mask = create_river_mask(image, kernel)
                                   
            plt.figure()
            imshow(river_mask, cmap='jet', vmin=0, vmax=255)
            plt.title(compteur)
            plt.colorbar()
            
            #"""
            # Image avec uniquement le masque de la rivière
            river_only = np.zeros_like(image, dtype=np.uint8)
            river_only[river_mask > 0] = [255, 255, 200]  # Bleu pour la rivière
            river_only_path = os.path.join(output_dir, f"{year}masque{compteur}.tif")
            cv2.imwrite(river_only_path, river_mask)
            
            contours, _ = cv2.findContours(river_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE) 
            
            for contour in contours:
                cv2.drawContours(image, [contour], 0, 255, 3)
                plt.close('all')

            
            plt.figure()
            imshow(image, cmap='gray', vmin=0, vmax=255)
            
            
            plt.colorbar()
            photo_contour_path = os.path.join(output_dir, f"{year}photocontour{compteur}.tif")
            cv2.imwrite(photo_contour_path,image)
            
            # alpha = 0.5
            # if len(image.shape) == 2:
            #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # output_image = cv2.addWeighted(image, 1, overlay, alpha, 0)

            # # Sauvegarder l'image combinée
            # 
            # # cv2.imwrite(photo_masked_path, output_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            compteur+=1
            print("fin photo")
            
    return river_mask 
            

####__________MAIN__________###            
input_dir = 'Port_Boyer_2007'
output_dir = 'Masks_2007'

test = process_image(input_dir, output_dir)

print('terminé')
    