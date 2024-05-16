# PIR-YOLO-UNET-SEP

## Contributeurs

Hautbois Guewen, Bartosik Johan, Peyrie Pierre-Angelo, Duthilleul Maïlys, Peng Zixin

## Préparation des données

Pour obtenir nos données nous avons utiliser 2 jeux de données : Miccai et Muslim 

### Miccai

Nous avons passé les images Flair et leur segmentation dans le script [data_setup_Miccai.py](data_setup_Muslim.py).

Ce script permet de garder seulement les coupes avec segmentation, de normaliser les niveaux de gris, puis d'enregistrer les images au bon format

### Muslim

Nous avons tout d'abord retiré les cranes des images grâce au [projet_SIR](https://github.com/GLucas01/projet_SIR.git), puis nous avons passé les images Flair sans crane et leur segmentation dans le script [data_setup_Miccai.py](data_setup_Muslim.py).

Ce script permet de rescale et resampler les images en taille 512x512x30. Puis de garder seulement les coupes avec segmentation, de normaliser les niveaux de gris, et enfin d'enregistrer les images au bon format.