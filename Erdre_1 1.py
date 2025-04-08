import cv2 as cv
import pandas as pd
import numpy as np
import os
import matching

# BIBLIOTHEQUE DU PROJET PI2 DE LA FEDERATION DES AMIS DE L'ERDRE
# DEFINITION DES CLASSES IMAGES, DATASET

class Image:
    def __init__(self, name, path, path_to_db):
        self.PATH = os.path.join(path, name)
        self.name = self.filter_file(name)
        self.year = self.dismantle_name()[0]
        self.mission = self.dismantle_name()[1]
        self.id = self.dismantle_name()[2]
        self.angle = self.get_infos(path_to_db)
        self.date = [0,0]
        self.image = cv.imread(self.PATH)

    def filter_file(self, file_name):
        if file_name[-4:] == '.tif':
            name = file_name[:-4]
        elif file_name[-5:] == '.tiff':
            name = file_name[:-5]
        else:
            print("Le fichier n'est pas au format .tif/.tiff")
            return None
        return name
    
    def dismantle_name(self):
        i,j = 0, 0
        #print(self.name)
        n = len(self.name)
        while i < n and self.name[i] != "_" :
            i += 1
        j = i+1
        while j < n and self.name[j] != "_" :
            j += 1
        year, mission, id = self.name[:i], self.name[i+1:j], self.name[j+1:]
        return int(year), mission, int(id)

    def get_infos(self, path_to_db) :
        db = pd.read_excel(path_to_db, str(self.year), header=0, skiprows=[1], usecols=['ID_mission', 'Numéro', 'angle'])
        db_mission = db[db["ID_mission"] == self.mission]
        row = db_mission[db_mission["Numéro"] == self.id]
        return -int(db_mission["angle"].values[0])

    def image_prep(self, border_threshold=5, angle=None, scale=None) :
        """Fonction qui prépare l'image en 3 étapes : 
         le scaling, la rotation et le nettoyages des bordures noires. """
        preped = self.image

        if scale:
            print(f"Scaling {self.name}")
            preped = self.scaling(scale, inplace=True)
            print("Done scaling")
        if angle:
            print(f"Rotating {self.name}")
            preped = self.rotating(angle, inplace=True)
            print("Done rotating")
        if border_threshold:
           print(f"Cutting border from {self.name}")
           preped = self.cleaning_borders(border_threshold, inplace=True)
           print("✅ Done processing")
        return preped
    
    def scaling(self, scale, inplace=False):
        (h, w) = self.image.shape[:2]
        if scale:
            new_size = (round(w * scale), round(h * scale))
            scaled = cv.resize(self.image, new_size, interpolation=cv.INTER_AREA)
        if inplace:
            self.image = scaled
        return scaled
    
    def rotating(self, angle, inplace=False):
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)

        # Obtenir la matrice de transformation
        M = cv.getRotationMatrix2D(center, self.angle, 1.0)
        # Appliquer la rotation
        rotated = cv.warpAffine(self.image, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        if inplace:
            self.image = rotated
        return rotated
    
    def cleaning_borders(self, threshold=5, inplace=False):
        image = self.image
        image_gris = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        y_max, x_max = image_gris.shape
        x_min, y_min = 0, 0
        while np.any(image_gris[y_min:y_max, x_min:x_max] <= threshold) :
            y_min += 50
            x_min += 50
            y_max -= 50
            x_max -= 50
        cropped_image = image[y_min:y_max, x_min:x_max]
        if inplace:
            self.image = cropped_image
        return cropped_image


class Dataset:
    def __init__(self, directory, path_to_db='database.xlsx'):
        self.DIR = directory
        self.path_to_db = path_to_db
        self.images = self.load()
        self.names = [image.name for image in self.images]
    
    def load(self):
        NAMES,IMG = [],[]
        for filename in os.listdir(self.DIR):
            if '.tif' in filename: # obligatoire pour filtrer tout autre fichier/dossier dans ce répertoire
                image_path = os.path.join(self.DIR, filename)
                image = Image(filename, self.DIR, self.path_to_db)    
                IMG.append(image)
        return IMG
    
    def preprocessing(self, homo_scale, border_threshold=None):
        print("PROCESSING...")
        for i in range(len(self.images)):
            angle = self.images[i].angle
            print(f"Processing {self.images[i].name}")
            self.images[i].image_prep(border_threshold=border_threshold, angle=angle, scale=homo_scale)
            print("Done")

    def match(self, seuil_ransac=5.0):
        N = len(self.images)
        Hs = {}
        masks = {}
        matcheses = {}
        for i in range(N):
            for j in range(i+1,N):
                img_i, img_j = self.images[i], self.images[j]
                kpt_i, kpt_j, matches = matching.init_matching_orb(img_i.image, img_j.image)
                cv.imwrite("test_lines.jpg", cv.drawMatches(img_i.image, kpt_i, img_j.image, kpt_j, matches, None))
                filtered_matches = matching.filtre_parallel_matches(kpt_i, kpt_j, matches)
                if len(filtered_matches) < 4:
                    matches = filtered_matches
                    kpt_i, kpt_j, matches = matching.create_artificial_match(kpt_i,kpt_j,matches)
                    H, mask= matching.ransac(kpt_i, kpt_j, matches)
                    mask = []
                    #matches = matching.filtre_distance(matches)
                    #H, mask = matching.ransac(kpt_i, kpt_j, matches, seuil_ransac)
                else : 
                    matches = filtered_matches
                    H, mask = matching.ransac(kpt_i, kpt_j, matches, seuil_ransac)
                cv.imwrite("test_filtering1.jpg", cv.drawMatches(img_i.image, kpt_i, img_j.image, kpt_j, matches, None))
                matcheses[(self.names[i],self.names[j])] = matches
                Hs[(self.names[i],self.names[j])] = H
                masks[(self.names[i],self.names[j])] = mask
        return Hs, masks, matcheses

    def get_image(self, name):
        for image in self.images:
            if image.name == name:
                return image.image
        print(f"Image with name {name} not found.")
        return None