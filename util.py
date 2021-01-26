import os
import pickle

parent_dir = '/Users/septem/Downloads/com_data'
def save(obj, filename, parent_dir = parent_dir):
    with open(os.path.join(parent_dir, filename), 'wb') as f:
        pickle.dump(obj, f)

def load(filename):
    with open(os.path.join(parent_dir, filename), 'rb') as f:
        return pickle.load(f)


cls2code = {'Landbouw, bosbouw en visserij': ['01', '02', '03'], 
'Winning van delfstoffen': ['06', '08', '09'], 
'Industrie': [str(e) for e in list(range(10,34))],
"Productie en distributie van en handel in elektriciteit, aardgas, stoom en gekoelde lucht": ["35"],
"Winning en distributie van water; afval- en afvalwaterbeheer en sanering": ["36", "37","38","39"],
"Bouwnijverheid": ["41","42","43"],
"Groot- en detailhandel; reparatie van auto’s": ["45","46","47"],
"Vervoer en opslag": [str(e) for e in list(range(49,54))],
"Logies-, maaltijd- en drankverstrekking": ["55","56"],
"Informatie en communicatie": [str(e) for e in list(range(58,64))],
"Financiële instellingen": ["64","65","66"],
"Verhuur van en handel in onroerend goed": ["68"],
"Advisering, onderzoek en overige specialistische zakelijke dienstverlening": [str(e) for e in list(range(69,76))],
"Verhuur van roerende goederen en overige zakelijke dienstverlening": [str(e) for e in list(range(77,83))],
"Openbaar bestuur, overheidsdiensten en verplichte sociale verzekeringen": ["84"],
"Onderwijs": ["85"],
"Gezondheids- en welzijnszorg": ["86","87","88"],
"Cultuur, sport en recreatie": ["90","91","92","93"],
"Overige dienstverlening": ["94","95","96"],
"Huishoudens als werkgever; niet-gedifferentieerde productie van goederen en diensten door huishoudens voor eigen gebruik": ["97","98"],
"Extraterritoriale organisaties en lichamen": ["99"]

}