import tkinter as tk
from tkinter import ttk, scrolledtext
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords, words
from nltk import pos_tag
import string, re
import joblib
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
from transfo_word_embbeding import DLWithEmbeddings

# Initialisation NLP
dictionnary = set(words.words())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocessing(text):
    text = re.sub(r'\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,7}\b', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = text.strip().lower()
    text = ''.join(c for c in text if not c.isdigit())
    for p in string.punctuation:
        text = text.replace(p, ' ')
    tokenized = word_tokenize(text)
    words_only = [w for w in tokenized if w.isalpha()]
    good_words = [w for w in words_only if w in dictionnary]
    without_stopwords = [w for w in good_words if w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in without_stopwords]
    return " ".join(lemmatized)

list_map1 = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Technology"
}

list_map12 = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Technology"
}

list_map2= {
    'FilmFestival': 'SocietalEvent',
    'Convention': 'SocietalEvent',
    'FootballMatch': 'SportsEvent',
    'OlympicEvent': 'Olympics',
    'GrandPrix': 'SportsEvent',
    'GolfTournament': 'Tournament',
    'WomensTennisAssociationTournament': 'Tournament',
    'TennisTournament': 'Tournament',
    'SoccerTournament': 'Tournament',
    'WrestlingEvent': 'SportsEvent',
    'HorseRace': 'Race',
    'CyclingRace': 'Race',
    'MixedMartialArtsEvent': 'SportsEvent',
    'Election': 'SocietalEvent',
    'SoccerClubSeason': 'SportsTeamSeason',
    'NationalFootballLeagueSeason': 'FootballLeagueSeason',
    'NCAATeamSeason': 'SportsTeamSeason',
    'BaseballSeason': 'SportsTeamSeason',
    'VideoGame': 'Software',
    'BiologicalDatabase': 'Database',
    'EurovisionSongContestEntry': 'Song',
    'Album': 'MusicalWork',
    'Musical': 'MusicalWork',
    'ClassicalMusicComposition': 'MusicalWork',
    'ArtistDiscography': 'MusicalWork',
    'Single': 'MusicalWork',
    'Poem': 'WrittenWork',
    'Magazine': 'PeriodicalLiterature',
    'Newspaper': 'PeriodicalLiterature',
    'AcademicJournal': 'PeriodicalLiterature',
    'Play': 'WrittenWork',
    'Manga': 'Comic',
    'ComicStrip': 'Comic',
    'Anime': 'Cartoon',
    'HollywoodCartoon': 'Cartoon',
    'MusicGenre': 'Genre',
    'Grape': 'FloweringPlant',
    'Conifer': 'Plant',
    'Fern': 'Plant',
    'Moss': 'Plant',
    'GreenAlga': 'Plant',
    'CultivatedVariety': 'Plant',
    'Cycad': 'Plant',
    'Arachnid': 'Animal',
    'Fish': 'Animal',
    'Insect': 'Animal',
    'Reptile': 'Animal',
    'Mollusca': 'Animal',
    'Bird': 'Animal',
    'Amphibian': 'Animal',
    'RaceHorse': 'Horse',
    'Crustacean': 'Animal',
    'Fungus': 'Eukaryote',
    'Lighthouse': 'Tower',
    'Theatre': 'Venue',
    'RollerCoaster': 'AmusementParkAttraction',
    'Airport': 'Infrastructure',
    'RailwayStation': 'Station',
    'Road': 'RouteOfTransportation',
    'RailwayLine': 'RouteOfTransportation',
    'Bridge': 'RouteOfTransportation',
    'RoadTunnel': 'RouteOfTransportation',
    'Dam': 'Infrastructure',
    'CricketGround': 'SportFacility',
    'Stadium': 'SportFacility',
    'Racecourse': 'RaceTrack',
    'GolfCourse': 'SportFacility',
    'Prison': 'Building',
    'Hospital': 'Building',
    'Museum': 'Building',
    'Hotel': 'Building',
    'Library': 'EducationalInstitution',
    'Restaurant': 'Building',
    'ShoppingMall': 'Building',
    'HistoricBuilding': 'Building',
    'Castle': 'Building',
    'Volcano': 'NaturalPlace',
    'MountainPass': 'NaturalPlace',
    'Glacier': 'NaturalPlace',
    'Canal': 'Stream',
    'River': 'Stream',
    'Lake': 'BodyOfWater',
    'Mountain': 'NaturalPlace',
    'Cave': 'NaturalPlace',
    'MountainRange': 'NaturalPlace',
    'Galaxy': 'CelestialBody',
    'ArtificialSatellite': 'Satellite',
    'Planet': 'CelestialBody',
    'Town': 'Settlement',
    'Village': 'Settlement',
    'Diocese': 'ClericalAdministrativeRegion',
    'AutomobileEngine': 'Engine',
    'SupremeCourtOfTheUnitedStatesCase': 'LegalCase',
    'MilitaryPerson': 'Person',
    'Religious': 'Person',
    'Engineer': 'Person',
    'BusinessPerson': 'Person',
    'SportsTeamMember': 'OrganisationMember',
    'SoccerManager': 'SportsManager',
    'Chef': 'Person',
    'Philosopher': 'Person',
    'CollegeCoach': 'Coach',
    'ScreenWriter': 'Writer',
    'Historian': 'Writer',
    'Poet': 'Writer',
    'President': 'Politician',
    'PrimeMinister': 'Politician',
    'Congressman': 'Politician',
    'Senator': 'Politician',
    'Mayor': 'Politician',
    'MemberOfParliament': 'Politician',
    'Governor': 'Politician',
    'Monarch': 'Person',
    'PlayboyPlaymate': 'Person',
    'Cardinal': 'Cleric',
    'Saint': 'Cleric',
    'Pope': 'Cleric',
    'ChristianBishop': 'Cleric',
    'BeautyQueen': 'Person',
    'RadioHost': 'Presenter',
    'HandballPlayer': 'Athlete',
    'Cricketer': 'Athlete',
    'Jockey': 'Athlete',
    'SumoWrestler': 'Wrestler',
    'AmericanFootballPlayer': 'GridironFootballPlayer',
    'LacrossePlayer': 'Athlete',
    'TennisPlayer': 'Athlete',
    'AmateurBoxer': 'Boxer',
    'SoccerPlayer': 'Athlete',
    'Rower': 'Athlete',
    'TableTennisPlayer': 'Athlete',
    'BeachVolleyballPlayer': 'VolleyballPlayer',
    'SpeedwayRider': 'MotorcycleRider',
    'FormulaOneRacer': 'RacingDriver',
    'NascarDriver': 'RacingDriver',
    'Swimmer': 'Athlete',
    'IceHockeyPlayer': 'WinterSportPlayer',
    'FigureSkater': 'WinterSportPlayer',
    'Skater': 'WinterSportPlayer',
    'Curler': 'WinterSportPlayer',
    'Skier': 'WinterSportPlayer',
    'GolfPlayer': 'Athlete',
    'SquashPlayer': 'Athlete',
    'PokerPlayer': 'Athlete',
    'BadmintonPlayer': 'Athlete',
    'ChessPlayer': 'Athlete',
    'RugbyPlayer': 'Athlete',
    'DartsPlayer': 'Athlete',
    'NetballPlayer': 'Athlete',
    'MartialArtist': 'Athlete',
    'Gymnast': 'Athlete',
    'Canoeist': 'Athlete',
    'GaelicGamesPlayer': 'Athlete',
    'HorseRider': 'Athlete',
    'BaseballPlayer': 'Athlete',
    'Cyclist': 'Athlete',
    'Bodybuilder': 'Athlete',
    'AustralianRulesFootballPlayer': 'Athlete',
    'BasketballPlayer': 'Athlete',
    'Ambassador': 'Person',
    'Baronet': 'BritishRoyalty',
    'Model': 'Person',
    'Architect': 'Person',
    'Judge': 'Person',
    'Economist': 'Person',
    'Journalist': 'Person',
    'Painter': 'Artist',
    'Comedian': 'Artist',
    'ComicsCreator': 'Artist',
    'ClassicalMusicArtist': 'MusicalArtist',
    'FashionDesigner': 'Artist',
    'AdultActor': 'Actor',
    'VoiceActor': 'Actor',
    'Photographer': 'Artist',
    'HorseTrainer': 'Person',
    'Entomologist': 'Scientist',
    'Medician': 'Scientist',
    'SoapCharacter': 'FictionalCharacter',
    'AnimangaCharacter': 'ComicsCharacter',
    'MythologicalFigure': 'FictionalCharacter',
    'Noble': 'Person',
    'Astronaut': 'Person',
    'OfficeHolder': 'Person',
    'PublicTransitSystem': 'Organisation',
    'BusCompany': 'Company',
    'LawFirm': 'Company',
    'Winery': 'Company',
    'RecordLabel': 'Company',
    'Brewery': 'Company',
    'Airline': 'Company',
    'Publisher': 'Company',
    'Bank': 'Company',
    'PoliticalParty': 'Organisation',
    'Legislature': 'Organisation',
    'Band': 'Group',
    'BasketballLeague': 'SportsLeague',
    'SoccerLeague': 'SportsLeague',
    'IceHockeyLeague': 'SportsLeague',
    'BaseballLeague': 'SportsLeague',
    'RugbyLeague': 'SportsLeague',
    'MilitaryUnit': 'Organisation',
    'University': 'EducationalInstitution',
    'School': 'EducationalInstitution',
    'CyclingTeam': 'SportsTeam',
    'CanadianFootballTeam': 'SportsTeam',
    'BasketballTeam': 'SportsTeam',
    'AustralianFootballTeam': 'SportsTeam',
    'HockeyTeam': 'SportsTeam',
    'HandballTeam': 'SportsTeam',
    'CricketTeam': 'SportsTeam',
    'RugbyClub': 'SportsTeam',
    'TradeUnion': 'Organisation',
    'RadioStation': 'Broadcaster',
    'BroadcastNetwork': 'Broadcaster',
    'TelevisionStation': 'Broadcaster'
}  

list_map3= {
    1: 'AcademicJournal',
    2: 'AdultActor',
    3: 'Airline',
    4: 'Airport',
    5: 'Album',
    6: 'AmateurBoxer',
    7: 'Ambassador',
    8: 'AmericanFootballPlayer',
    9: 'Amphibian',
    10: 'AnimangaCharacter',
    11: 'Anime',
    12: 'Arachnid',
    13: 'Architect',
    14: 'ArtificialSatellite',
    15: 'ArtistDiscography',
    16: 'Astronaut',
    17: 'AustralianFootballTeam',
    18: 'AustralianRulesFootballPlayer',
    19: 'AutomobileEngine',
    20: 'BadmintonPlayer',
    21: 'Band',
    22: 'Bank',
    23: 'Baronet',
    24: 'BaseballLeague',
    25: 'BaseballPlayer',
    26: 'BaseballSeason',
    27: 'BasketballLeague',
    28: 'BasketballPlayer',
    29: 'BasketballTeam',
    30: 'BeachVolleyballPlayer',
    31: 'BeautyQueen',
    32: 'BiologicalDatabase',
    33: 'Bird',
    34: 'Bodybuilder',
    35: 'Brewery',
    36: 'Bridge',
    37: 'BroadcastNetwork',
    38: 'BusCompany',
    39: 'BusinessPerson',
    40: 'CanadianFootballTeam',
    41: 'Canal',
    42: 'Canoeist',
    43: 'Cardinal',
    44: 'Castle',
    45: 'Cave',
    46: 'Chef',
    47: 'ChessPlayer',
    48: 'ChristianBishop',
    49: 'ClassicalMusicArtist',
    50: 'ClassicalMusicComposition',
    51: 'CollegeCoach',
    52: 'Comedian',
    53: 'ComicStrip',
    54: 'ComicsCreator',
    55: 'Congressman',
    56: 'Conifer',
    57: 'Convention',
    58: 'CricketGround',
    59: 'CricketTeam',
    60: 'Cricketer',
    61: 'Crustacean',
    62: 'CultivatedVariety',
    63: 'Curler',
    64: 'Cycad',
    65: 'CyclingRace',
    66: 'CyclingTeam',
    67: 'Cyclist',
    68: 'Dam',
    69: 'DartsPlayer',
    70: 'Diocese',
    71: 'Earthquake',
    72: 'Economist',
    73: 'Election',
    74: 'Engineer',
    75: 'Entomologist',
    76: 'EurovisionSongContestEntry',
    77: 'FashionDesigner',
    78: 'Fern',
    79: 'FigureSkater',
    80: 'FilmFestival',
    81: 'Fish',
    82: 'FootballMatch',
    83: 'FormulaOneRacer',
    84: 'Fungus',
    85: 'GaelicGamesPlayer',
    86: 'Galaxy',
    87: 'Glacier',
    88: 'GolfCourse',
    89: 'GolfPlayer',
    90: 'GolfTournament',
    91: 'Governor',
    92: 'GrandPrix',
    93: 'Grape',
    94: 'GreenAlga',
    95: 'Gymnast',
    96: 'HandballPlayer',
    97: 'HandballTeam',
    98: 'Historian',
    99: 'HistoricBuilding',
    100: 'HockeyTeam',
    101: 'HollywoodCartoon',
    102: 'HorseRace',
    103: 'HorseRider',
    104: 'HorseTrainer',
    105: 'Hospital',
    106: 'Hotel',
    107: 'IceHockeyLeague',
    108: 'IceHockeyPlayer',
    109: 'Insect',
    110: 'Jockey',
    111: 'Journalist',
    112: 'Judge',
    113: 'LacrossePlayer',
    114: 'Lake',
    115: 'LawFirm',
    116: 'Legislature',
    117: 'Library',
    118: 'Lighthouse',
    119: 'Magazine',
    120: 'Manga',
    121: 'MartialArtist',
    122: 'Mayor',
    123: 'Medician',
    124: 'MemberOfParliament',
    125: 'MilitaryConflict',
    126: 'MilitaryPerson',
    127: 'MilitaryUnit',
    128: 'MixedMartialArtsEvent',
    129: 'Model',
    130: 'Mollusca',
    131: 'Monarch',
    132: 'Moss',
    133: 'Mountain',
    134: 'MountainPass',
    135: 'MountainRange',
    136: 'Museum',
    137: 'MusicFestival',
    138: 'MusicGenre',
    139: 'Musical',
    140: 'MythologicalFigure',
    141: 'NCAATeamSeason',
    142: 'NascarDriver',
    143: 'NationalFootballLeagueSeason',
    144: 'NetballPlayer',
    145: 'Newspaper',
    146: 'Noble',
    147: 'OfficeHolder',
    148: 'OlympicEvent',
    149: 'Painter',
    150: 'Philosopher',
    151: 'Photographer',
    152: 'Planet',
    153: 'Play',
    154: 'PlayboyPlaymate',
    155: 'Poem',
    156: 'Poet',
    157: 'PokerPlayer',
    158: 'PoliticalParty',
    159: 'Pope',
    160: 'President',
    161: 'PrimeMinister',
    162: 'Prison',
    163: 'PublicTransitSystem',
    164: 'Publisher',
    165: 'RaceHorse',
    166: 'Racecourse',
    167: 'RadioHost',
    168: 'RadioStation',
    169: 'RailwayLine',
    170: 'RailwayStation',
    171: 'RecordLabel',
    172: 'Religious',
    173: 'Reptile',
    174: 'Restaurant',
    175: 'River',
    176: 'Road',
    177: 'RoadTunnel',
    178: 'RollerCoaster',
    179: 'Rower',
    180: 'RugbyClub',
    181: 'RugbyLeague',
    182: 'RugbyPlayer',
    183: 'Saint',
    184: 'School',
    185: 'ScreenWriter',
    186: 'Senator',
    187: 'ShoppingMall',
    188: 'Single',
    189: 'Skater',
    190: 'Skier',
    191: 'SoapCharacter',
    192: 'SoccerClubSeason',
    193: 'SoccerLeague',
    194: 'SoccerManager',
    195: 'SoccerPlayer',
    196: 'SoccerTournament',
    197: 'SolarEclipse',
    198: 'SpeedwayRider',
    199: 'SportsTeamMember',
    200: 'SquashPlayer',
    201: 'Stadium',
    202: 'SumoWrestler',
    203: 'SupremeCourtOfTheUnitedStatesCase',
    204: 'Swimmer',
    205: 'TableTennisPlayer',
    206: 'TelevisionStation',
    207: 'TennisPlayer',
    208: 'TennisTournament',
    209: 'Theatre',
    210: 'Town',
    211: 'TradeUnion',
    212: 'University',
    213: 'VideoGame',
    214: 'Village',
    215: 'VoiceActor',
    216: 'Volcano',
    217: 'Winery',
    218: 'WomensTennisAssociationTournament',
    219: 'WrestlingEvent'
}

class ChatbotGUI_class:
    def __init__(self, root):
        self.root = root
        self.root.title("Unintelligent classification")
        self.root.geometry("700x500")
        self.root.configure(bg="#2C3E50")
        self.root.minsize(400, 200)

        # Chargement des modèles
        self.modelG = joblib.load("Model\Classification\Général\Vest_model3.pkl")
        self.modelP = joblib.load("Model\Classification\Précis\model22.pkl")
        
        model_dl_G = load_model("Model\Classification\Général\model_dl.keras")
        word2vec_model = Word2Vec.load("Model\Classification\Général\word2vec.model")
        max_len = 73
        self.model_dl_G = DLWithEmbeddings(model_dl=model_dl_G, word2vec_model=word2vec_model, maxlen=max_len)

        model_dl_P = load_model("Model\Classification\Précis\model_dl.keras")
        word2vec_model_new_data = Word2Vec.load("Model\Classification\Précis\word2vec.model")
        max_len = 200
        self.model_dl_P = DLWithEmbeddings(model_dl=model_dl_P, word2vec_model=word2vec_model_new_data, maxlen=max_len)

        # Style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12), padding=5)
        self.style.configure("TLabel", font=("Helvetica", 12), background="#2C3E50", foreground="white")
        self.style.configure("TEntry", font=("Helvetica", 12))
        self.style.configure("Custom.TButton", background="#3498DB", foreground="#2C3E50")

        # Titre
        self.title_label = ttk.Label(root, text="Le chat qui ne GPT pas bien", font=("Helvetica", 16, "bold"), foreground="#ECF0F1")
        self.title_label.pack(pady=10)

        # Zone de chat
        self.chat_frame = tk.Frame(root, bg="#34495E", bd=2)
        self.chat_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, bg="#ECF0F1", fg="#2C3E50", font=("Helvetica", 11), bd=0)
        self.chat_display.pack(padx=10, pady=10, fill="both", expand=True)

        # Zone d'entrée
        self.input_frame = tk.Frame(root, bg="#2C3E50")
        self.input_frame.pack(padx=20, pady=10, fill="x")

        self.input_label = ttk.Label(self.input_frame, text="Your message :")
        self.input_label.pack(side="left", padx=5)

        self.input_field = ttk.Entry(self.input_frame, width=50)
        self.input_field.pack(side="left", padx=5, fill="x", expand=True)
        self.input_field.bind("<Return>", self.process_input)

        self.send_button = ttk.Button(self.input_frame, text="Send", command=self.process_input, style="Custom.TButton")
        self.send_button.pack(side="right", padx=5)

        self.configure_tags()

    def process_input(self, event=None):
        user_input = self.input_field.get().strip()
        if user_input:
            self.chat_display.insert(tk.END, "You : ", ("user",))
            self.chat_display.insert(tk.END, user_input + "\n")

            # Prédiction ML
            try:
                cleaned_input = preprocessing(user_input)
                pred_ml_G = self.modelG.predict([cleaned_input])[0]
                pred_ml_P = self.modelP.predict([cleaned_input])[0]
                self.chat_display.insert(tk.END, "Bot ML: ", ("bot",))
                self.chat_display.insert(tk.END, "Générique ", ("bot",))
                self.chat_display.insert(tk.END, str(list_map1.get(pred_ml_G, "Inconnu")))
                self.chat_display.insert(tk.END, " Précis ", ("bot",))
                self.chat_display.insert(tk.END, f"{list_map2.get(pred_ml_P, 'Inconnu')} - {pred_ml_P}\n\n")
            except Exception as e:
                self.chat_display.insert(tk.END, f"Error ML: {e}\n\n", ("bot",))
                print("Error ML :", e)

            # Prédiction DL
            try:
                cleaned_input = preprocessing(user_input)
                pred_dl_G = self.model_dl_G.predict([cleaned_input])[0]
                pred_dl_P = self.model_dl_P.predict([cleaned_input])[0]
                self.chat_display.insert(tk.END, "Bot DL: ", ("bot",))
                self.chat_display.insert(tk.END, "Générique ", ("bot",))
                self.chat_display.insert(tk.END, str(list_map12.get(pred_dl_G, "Inconnu")))
                self.chat_display.insert(tk.END, " Précis ", ("bot",))
                self.chat_display.insert(tk.END, f"{str(list_map2.get(str(list_map3.get(pred_dl_P, "Inconnu")), "Inconnu"))} - {str(list_map3.get(pred_dl_P, "Inconnu"))} "+"\n\n")
                
            except Exception as e:
                self.chat_display.insert(tk.END, f"Error DL: {e}\n\n", ("bot",))
                print("Error DL :", e)

            self.input_field.delete(0, tk.END)
            self.chat_display.see(tk.END)
    def configure_tags(self):
        self.chat_display.tag_configure("user", foreground="#3498DB", font=("Helvetica", 11, "bold"))
        self.chat_display.tag_configure("bot", foreground="#E74C3C", font=("Helvetica", 11, "bold"))

def main_class():
    root = tk.Tk()
    app = ChatbotGUI_class(root)
    root.mainloop()

if __name__ == "__main__":
    main_class()
