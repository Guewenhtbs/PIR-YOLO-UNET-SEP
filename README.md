# PIR-YOLO-UNET-SEP

## Contributeurs

Hautbois Guewen, Bartosik Johan, Peyrie Pierre-Angelo, Duthilleul Maïlys, Peng Zixin

## Préparation des données

Pour obtenir nos données nous avons utiliser 2 jeux de données : Miccai et Muslim.

### Miccai

Nous avons passé les images Flair et leur segmentation dans le script [data_setup_Miccai.py](data_setup_Muslim.py).

Ce script permet de normaliser les niveaux de gris avec le N4 Bias Field Corecction Filter, de garder seulement les coupes avec segmentation, de refaire l'échelle des niveaux de gris entre 0 et 255 sur les coupes, puis de les enregistrer au bon format.

### Muslim

Nous avons tout d'abord retiré les cranes des images grâce au [projet_SIR](https://github.com/GLucas01/projet_SIR.git), puis nous avons passé les images Flair sans crane et leur segmentation dans le script [data_setup_Miccai.py](data_setup_Muslim.py).

Ce script permet de rescale et resampler les images en taille 512x512x30. Puis il permet ensuite de normaliser les niveaux de gris avec le N4 Bias Field Corecction Filter, de garder seulement les coupes avec segmentation, de refaire l'échelle des niveaux de gris entre 0 et 255 sur les coupes, et enfin de les enregistrer au bon format.

### YOLO
Le fichier [training_yolo.py](https://github.com/Guewenhtbs/PIR-YOLO-UNET-SEP/blob/main/training_yolo.py) contient la configuration d'entrainement utilisée pour notre YOLO.  
Le fichier [dice_yolo.py](https://github.com/Guewenhtbs/PIR-YOLO-UNET-SEP/blob/main/dice_yolo.py) contient le code permettant de mesurer les performence d'un modèle YOLO sur un jeu de donnée de test avec l'indice DICE.  
Le fichier [testperf.py](https://github.com/Guewenhtbs/PIR-YOLO-UNET-SEP/blob/main/testperf.py) permet d'aficher la segmentation d'une image produite par un modèle YOLO.
