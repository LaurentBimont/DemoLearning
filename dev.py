
def aire_rectangle2(**kwargs):  # les arguments passes en parametre sont paquetes dans kwargs qui se comporte comme un dictionnaire
    if len(kwargs) == 2:
        result = 1
        for key, value in kwargs.items():
            result *=value
        return result
    else:
        print('Merci de stipuler deux parametres')

aire_rectangle2({'ff':[1,2], 'fs':[4,4]})
