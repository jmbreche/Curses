import os
import re
import json
import nltk
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
from keras import backend
from nltk.corpus import wordnet
from alive_progress import alive_bar
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")

STOPWORDS = [
    "x", "y", "your", "yours", "yourself", "yourselves", "you", "yond", "yonder", "yon", "ye", "yet", "z", "zillion", "j", "u", "umpteen", "usually", "us", "username", "uponed", "upons", "uponing", "upon", "ups", "upping", "upped", "up", "unto", "until", "unless", "unlike", "unliker", "unlikest", "under", "underneath", "use", "used", "usedest", "r", "rath", "rather", "rathest", "rathe", "re", "relate", "related", "relatively", "regarding", "really", "res", "respecting", "respectively", "q", "quite", "que", "qua", "n", "neither", "neaths", "neath", "nethe", "nethermost", "necessary", "necessariest", "necessarier", "never", "nevertheless", "nigh", "nighest", "nigher", "nine", "noone", "nobody", "nobodies", "nowhere", "nowheres", "no", "noes", "nor", "nos", "no-one", "none", "not", "notwithstanding", "nothings", "nothing", "nathless", "natheless", "t", "ten", "tills", "till", "tilled", "tilling", "to", "towards", "toward", "towardest", "towarder", "together", "too", "thy", "thyself", "thus", "than", "that", "those", "thou", "though", "thous", "thouses", "thoroughest", "thorougher", "thorough", "thoroughly", "thru", "thruer", "thruest", "thro", "through", "throughout", "throughest", "througher", "thine", "this", "thises", "they", "thee", "the", "then", "thence", "thenest", "thener", "them", "themselves", "these", "therer", "there", "thereby", "therest", "thereafter", "therein", "thereupon", "therefore", "their", "theirs", "thing", "things", "three", "two", "o", "oh", "owt", "owning", "owned", "own", "owns", "others", "other", "otherwise", "otherwisest", "otherwiser", "of", "often", "oftener", "oftenest", "off", "offs", "offest", "one", "ought", "oughts", "our", "ours", "ourselves", "ourself", "out", "outest", "outed", "outwith", "outs", "outside", "over", "overallest", "overaller", "overalls", "overall", "overs", "or", "orer", "orest", "on", "oneself", "onest", "ons", "onto", "a", "atween", "at", "athwart", "atop", "afore", "afterward", "afterwards", "after", "afterest", "afterer", "ain", "an", "any", "anything", "anybody", "anyone", "anyhow", "anywhere", "anent", "anear", "and", "andor", "another", "around", "ares", "are", "aest", "aer", "against", "again", "accordingly", "abaft", "abafter", "abaftest", "abovest", "above", "abover", "abouter", "aboutest", "about", "aid", "amidst", "amid", "among", "amongst", "apartest", "aparter", "apart", "appeared", "appears", "appear", "appearing", "appropriating", "appropriate", "appropriatest", "appropriates", "appropriater", "appropriated", "already", "always", "also", "along", "alongside", "although", "almost", "all", "allest", "aller", "allyou", "alls", "albeit", "awfully", "as", "aside", "asides", "aslant", "ases", "astrider", "astride", "astridest", "astraddlest", "astraddler", "astraddle", "availablest", "availabler", "available", "aughts", "aught", "vs", "v", "variousest", "variouser", "various", "via", "vis-a-vis", "vis-a-viser", "vis-a-visest", "viz", "very", "veriest", "verier", "versus", "k", "g", "go", "gone", "good", "got", "gotta", "gotten", "get", "gets", "getting", "b", "by", "byandby", "by-and-by", "bist", "both", "but", "buts", "be", "beyond", "because", "became", "becomes", "become", "becoming", "becomings", "becominger", "becomingest", "behind", "behinds", "before", "beforehand", "beforehandest", "beforehander", "bettered", "betters", "better", "bettering", "betwixt", "between", "beneath", "been", "below", "besides", "beside", "m", "my", "myself", "mucher", "muchest", "much", "must", "musts", "musths", "musth", "main", "make", "mayest", "many", "mauger", "maugre", "me", "meanwhiles", "meanwhile", "mostly", "most", "moreover", "more", "might", "mights", "midst", "midsts", "h", "huh", "humph", "he", "hers", "herself", "her", "hereby", "herein", "hereafters", "hereafter", "hereupon", "hence", "hadst", "had", "having", "haves", "have", "has", "hast", "hardly", "hae", "hath", "him", "himself", "hither", "hitherest", "hitherer", "his", "how-do-you-do", "however", "how", "howbeit", "howdoyoudo", "hoos", "hoo", "w", "woulded", "woulding", "would", "woulds", "was", "wast", "we", "wert", "were", "with", "withal", "without", "within", "why", "what", "whatever", "whateverer", "whateverest", "whatsoeverer", "whatsoeverest", "whatsoever", "whence", "whencesoever", "whenever", "whensoever", "when", "whenas", "whether", "wheen", "whereto", "whereupon", "wherever", "whereon", "whereof", "where", "whereby", "wherewithal", "wherewith", "whereinto", "wherein", "whereafter", "whereas", "wheresoever", "wherefrom", "which", "whichever", "whichsoever", "whilst", "while", "whiles", "whithersoever", "whither", "whoever", "whosoever", "whoso", "whose", "whomever", "s", "syne", "syn", "shalling", "shall", "shalled", "shalls", "shoulding", "should", "shoulded", "shoulds", "she", "sayyid", "sayid", "said", "saider", "saidest", "same", "samest", "sames", "samer", "saved", "sans", "sanses", "sanserifs", "sanserif", "so", "soer", "soest", "sobeit", "someone", "somebody", "somehow", "some", "somewhere", "somewhat", "something", "sometimest", "sometimes", "sometimer", "sometime", "several", "severaler", "severalest", "serious", "seriousest", "seriouser", "senza", "send", "sent", "seem", "seems", "seemed", "seemingest", "seeminger", "seemings", "seven", "summat", "sups", "sup", "supping", "supped", "such", "since", "sine", "sines", "sith", "six", "stop", "stopped", "p", "plaintiff", "plenty", "plenties", "please", "pleased", "pleases", "per", "perhaps", "particulars", "particularly", "particular", "particularest", "particularer", "pro", "providing", "provides", "provided", "provide", "probably", "l", "layabout", "layabouts", "latter", "latterest", "latterer", "latterly", "latters", "lots", "lotting", "lotted", "lot", "lest", "less", "ie", "ifs", "if", "i", "info", "information", "itself", "its", "it", "is", "idem", "idemer", "idemest", "immediate", "immediately", "immediatest", "immediater", "in", "inwards", "inwardest", "inwarder", "inward", "inasmuch", "into", "instead", "insofar", "indicates", "indicated", "indicate", "indicating", "indeed", "inc", "f", "fact", "facts", "fs", "figupon", "figupons", "figuponing", "figuponed", "few", "fewer", "fewest", "frae", "from", "failing", "failings", "five", "furthers", "furtherer", "furthered", "furtherest", "further", "furthering", "furthermore", "fourscore", "followthrough", "for", "forwhy", "fornenst", "formerly", "former", "formerer", "formerest", "formers", "forbye", "forby", "fore", "forever", "forer", "fores", "four", "d", "ddays", "dday", "do", "doing", "doings", "doe", "does", "doth", "downwarder", "downwardest", "downward", "downwards", "downs", "done", "doner", "dones", "donest", "dos", "dost", "did", "differentest", "differenter", "different", "describing", "describe", "describes", "described", "despiting", "despites", "despited", "despite", "during", "c", "cum", "circa", "chez", "cer", "certain", "certainest", "certainer", "cest", "canst", "cannot", "cant", "cants", "canting", "cantest", "canted", "co", "could", "couldst", "comeon", "comeons", "come-ons", "come-on", "concerning", "concerninger", "concerningest", "consequently", "considering", "e", "eg", "eight", "either", "even", "evens", "evenser", "evensest", "evened", "evenest", "ever", "everyone", "everything", "everybody", "everywhere", "every", "ere", "each", "et", "etc", "elsewhere", "else", "ex", "excepted", "excepts", "except", "excepting", "exes", "enough",
    "assimilationism", "assimilationist", "associationisms", "associativities", "associationists", "assignabilities", "assiduousnesses", "assertivenesses", "ambassadorships", "disembarrassing", "disassociations", "dispassionately", "classifications", "counterassaults", "compassionately", "compassionating", "classlessnesses", "impassibilities", "impassabilities", "immunoassayable", "impassivenesses", "overclassifying", "overassessments", "uncompassionate", "unassailability", "thalassocracies", "classification", "unclassifiable", "associationism", "associationist", "associateships", "assistantships", "assimilability", "assassinations", "assaultiveness", "ambassadresses", "ambassadorship", "audiocassettes", "assumabilities", "classificatory", "classicalities", "contrabassoons", "counterassault", "contrabassists", "compassionless", "compassionates", "compassionated", "disassociating", "disassociation", "disembarrassed", "disembarrasses", "embarrassments", "embarrassingly", "encompassments", "johnsongrasses", "impassableness", "hardinggrasses", "unassumingness", "thoroughbasses", "videocassettes", "weatherglasses", "subclassifying", "pseudoclassics", "overassessment", "overclassified", "overclassifies", "overassertions", "neoclassicists", "passementeries", "passionflowers", "passionateness", "misclassifying", "misassumptions", "microcassettes", "neoclassicisms", "embarrassment", "compassionate", "assemblywoman", "dispassionate", "videocassette", "audiocassette", "neoclassicism", "underclassman", "unsurpassable", "upperclassman", "unassimilated", "postclassical", "unembarrassed", "assistantship", "passionflower", "semiclassical", "contrabassoon", "passementerie", "unimpassioned", "thalassocracy", "gyrocompasses", "glasspapering", "impassibility", "impassiveness", "impassivities", "impassability", "contrabassist", "classlessness", "embarrassedly", "embarrassable", "disassemblies", "disassembling", "disassociated", "disassociates", "declassifying", "gallowglasses", "glassblowings", "fiberglassing", "encompassment", "assimilations", "associational", "associateship", "associativity", "associatively", "assertiveness", "asseverations", "assignability", "assiduousness", "assemblywomen", "assemblagists", "assassinating", "assassination", "assassinators", "anticlassical", "antimacassars", "ambassadorial", "assurednesses", "assortatively", "thalassocrats", "upperclassmen", "unassimilable", "unassertively", "underclassmen", "unassuageable", "passivenesses", "neoclassicist", "nonassociated", "nonclassified", "overasserting", "overassertion", "overassertive", "microcassette", "misassembling", "misassumption", "misclassified", "misclassifies", "massivenesses", "subassemblies", "subclassified", "subclassifies", "switchgrasses", "thalassaemias", "pseudoclassic", "peppergrasses", "reclassifying", "reassemblages", "reassessments", "reassignments", "assimilation", "unassailable", "unclassified", "disassociate", "unassociated", "supermassive", "assimilative", "thalassaemia", "antimacassar", "ambassadress", "weatherglass", "thoroughbass", "disembarrass", "declassifies", "disassembles", "detasselling", "disassembled", "declassified", "embarrassing", "fricasseeing", "glassblowers", "glassblowing", "encompassing", "fiberglassed", "fiberglasses", "fibreglasses", "classicality", "classinesses", "classicistic", "classicizing", "classifiable", "contrabasses", "groundmasses", "grasshoppers", "glassinesses", "glassmakings", "glasspapered", "glassworkers", "goosegrasses", "hardinggrass", "impassioning", "immunoassays", "lemongrasses", "johnsongrass", "assaultively", "assassinator", "assassinates", "assassinated", "assentations", "assemblagist", "assimilators", "assimilatory", "assimilating", "associations", "asseverative", "assignations", "asseverating", "asseveration", "assumability", "assuagements", "bromegrasses", "bunchgrasses", "brassinesses", "cassiterites", "thalassocrat", "unassailably", "underclasses", "witchgrasses", "thalassemics", "thalassemias", "superclasses", "surpassingly", "semiclassics", "reassurances", "reassuringly", "reclassified", "reclassifies", "reassignment", "reassessment", "reassertions", "reassemblies", "reassembling", "reassemblage", "preassembled", "preassigning", "passageworks", "passionately", "passivations", "nonassertive", "nonclassroom", "nonpasserine", "nonclassical", "passacaglias", "overclassify", "overasserted", "neoclassical", "misassembled", "misassembles", "association", "unsurpassed", "associative", "impassioned", "assemblyman", "assassinate", "immunoassay", "subassembly", "grasshopper", "disassemble", "assignation", "passionless", "thalassemia", "switchgrass", "unassembled", "assimilable", "passacaglia", "misclassify", "unassertive", "gyrocompass", "passagework", "cassiterite", "peppergrass", "assentation", "gallowglass", "glassblower", "gassinesses", "encompassed", "encompasses", "detasseling", "detasselled", "disassembly", "dispassions", "embarrassed", "embarrasses", "compassable", "compassions", "coassisting", "crassnesses", "crassitudes", "crabgrasses", "cordgrasses", "cuirassiers", "classicists", "classicized", "classicizes", "classicisms", "classifying", "classifiers", "classically", "cassowaries", "glassworker", "glasspapers", "glassmakers", "glassmaking", "glasshouses", "impassivity", "impassively", "hourglasses", "harassments", "isinglasses", "knotgrasses", "assemblymen", "assemblages", "ambassadors", "assimilated", "assimilates", "assimilator", "assistances", "assoilments", "associating", "assignments", "assiduities", "assiduously", "assertively", "assessments", "asseverated", "asseverates", "carnassials", "beargrasses", "bentgrasses", "bassoonists", "bluegrasses", "assortative", "assortments", "assuagement", "assumptions", "assuredness", "preassigned", "preassuring", "postclassic", "reassurance", "reassorting", "sandglasses", "reassailing", "reassembled", "reassembles", "reasserting", "reassertion", "reassessing", "assessment", "assistance", "assumption", "assignment", "classified", "passionate", "ambassador", "compassion", "assortment", "fiberglass", "assimilate", "assemblage", "surpassing", "underclass", "unassuming", "passageway", "classicism", "unassisted", "impassable", "reassemble", "unassigned", "fibreglass", "reclassify", "classicist", "glasshouse", "lemongrass", "classifier", "declassify", "assaultive", "superclass", "dispassion", "glassmaker", "impassible", "groundmass", "assumptive", "goosegrass", "contrabass", "massasauga", "unassuaged", "cuirassier", "classicize", "bromegrass", "brassbound", "carnassial", "asseverate", "glasspaper", "glassworks", "glassworts", "glasswares", "glassiness", "grassplots", "grassroots", "grasslands", "impassably", "impassibly", "impassions", "harassment", "landmasses", "lassitudes", "madrassahs", "massacring", "massacrers", "interclass", "classiness", "classifies", "classicals", "chassepots", "cassoulets", "crevassing", "cutgrasses", "crassitude", "compassing", "coassuming", "classrooms", "classworks", "classmates", "coassisted", "embassages", "eelgrasses", "declassing", "cuirassing", "demitasses", "detasseled", "forepassed", "eyeglasses", "fricassees", "galleasses", "fricasseed", "galliasses", "assessable", "assertedly", "assertions", "assignable", "assistants", "assonances", "assoilment", "associated", "associates", "assaulters", "assistant", "assurance", "classroom", "associate", "passenger", "classical", "assertion", "encompass", "potassium", "assertive", "assistive", "embarrass", "casserole", "passivity", "assailant", "grassland", "assembler", "glassware", "classmate", "impassive", "classless", "brasserie", "bluegrass", "assiduous", "underpass", "hourglass", "lassitude", "brassiere", "sassafras", "assonance", "cassowary", "crabgrass", "demitasse", "passerine", "wineglass", "wiregrass", "assiduity", "cassoulet", "glasswork", "fricassee", "vassalage", "assumpsit", "brassware", "cassation", "overclass", "passional", "cordgrass", "madrassah", "beargrass", "impassion", "passivism", "passivate", "matelasse", "thalassic", "sargassum", "passepied", "isinglass", "sandglass", "glasswort", "grassplot", "knotgrass", "chassepot", "embassage", "cassimere", "cassingle", "assurgent", "assuasive", "paillasse", "palliasse", "outpassed", "outpasses", "outgassed", "outgasses", "passbands", "passbooks", "passadoes", "passaging", "passalong", "passively", "passivist", "passingly", "passersby", "matrasses", "massicots", "massiness", "massively", "masscults", "masseuses", "masseters", "misassign", "misassays", "repassing", "reassigns", "reasserts", "reassumed", "reassumes", "reassured", "reassures", "reassorts", "repassage", "assembly", "assuming", "password", "asserted", "assemble", "passport", "cassette", "massacre", "classify", "reassure", "assorted", "assassin", "assessor", "reassess", "trespass", "sunglass", "reassert", "assignee", "overpass", "passable", "reassign", "molasses", "passbook", "bioassay", "subclass", "spyglass", "eyeglass", "passerby", "landmass", "brassard", "classism", "reassume", "masseuse", "crevasse", "sargasso", "assignor", "basswood", "bassinet", "windlass", "brassica", "passband", "ryegrass", "declasse", "outclass", "massless", "masseter", "glassful", "glassine", "chasseur", "eelgrass", "assignat", "passible", "massicot", "ribgrass", "piassava", "sasswood", "galleass", "galliass", "curassow", "cutgrass", "crassest", "classist", "classons", "classing", "coassist", "coassume", "cassocks", "classily", "classico", "classics", "classier", "classers", "gassiest", "gassings", "degassed", "degasser", "degasses", "detassel", "glassier", "glassies", "glassily", "glassing", "glassman", "grassier", "grassily", "grassing", "harassed", "impasses", "hassling", "hassocks", "hassiums", "harasser", "harasses", "lassoers", "lassoing", "massaged", "massager", "massages", "madrassa", "kolbassi", "classes", "massive", "classic", "assumed", "passing", "passage", "passion", "assured", "assault", "passive", "embassy", "chassis", "compass", "massage", "surpass", "biomass", "impasse", "bassist", "carcass", "assuage", "canvass", "bassoon", "cassava", "cutlass", "bagasse", "cassock", "passkey", "masseur", "cassino", "cassina", "brassie", "hassock", "passant", "classon", "cassata", "wassail", "classis", "hassium", "assegai", "passado", "assagai", "babassu", "cuirass", "declass", "quassia", "cassena", "cassaba", "cassene", "rubasse", "subbass", "sassaby", "passade", "oquassa", "vinasse", "vassals", "wrasses", "wrassle", "trasses", "outpass", "passels", "passers", "massier", "massing", "massifs", "matrass", "megasse", "morassy", "sassing", "sassily", "sassier", "sassies", "tassies", "tassets", "tassels", "rassled", "rassles", "quasses", "quassin", "cassias", "cassine", "brasses", "brassed", "biassed", "biasses", "bassets", "bassett", "assorts", "assurer", "assures", "assumer", "assumes", "assuror", "asswage", "amassed", "amasser", "amasses", "assayed", "assume", "assist", "assess", "assure", "bypass", "assert", "assign", "hassle", "grassy", "assent", "classy", "glassy", "harass", "morass", "lassie", "passim", "gassed", "brassy", "massif", "vassal", "basset", "chasse", "assail", "passel", "tassel", "cassis", "wrasse", "gasser", "assize", "cassia", "assort", "repass", "strass", "tassie", "passus", "dassie", "rassle", "assoil", "assets", "assais", "admass", "assays", "basses", "camass", "bassly", "bassos", "gasses", "lasses", "lassis", "lassos", "kavass", "jassid", "hassel", "tasses", "tasset", "sasses", "sassed", "passed", "passee", "passer", "passes", "masses", "massas", "massed", "megass", "class", "asset", "glass", "grass", "brass", "assay", "masse", "crass", "amass", "sassy", "passe", "massa", "lasso", "basso", "bassi", "gassy", "massy", "tasse", "assai", "lassi", "frass", "trass", "kvass", "eyass", "bassy", "quass", "mass", "pass", "bass", "lass", "sass", "tass", "shuttlecocking", "cocksurenesses", "cockeyednesses", "cockfightings", "shuttlecocked", "shuttlecocks", "weathercocks", "cocksureness", "cockeyedness", "cockneyfying", "cockleshells", "shuttlecock", "weathercock", "cockleshell", "cockneyfied", "cockneyfies", "cockneyisms", "cockroaches", "cockinesses", "cockalorums", "cockatrices", "cockbilling", "cockchafers", "cocktailing", "coldcocking", "peacockiest", "cockamamie", "cockatrice", "cockchafer", "cockalorum", "cockneyism", "cockleburs", "cockneyish", "cockhorses", "cockeyedly", "billycocks", "cockbilled", "cockatiels", "cockateels", "cocksurely", "cocksfoots", "cockscombs", "cockswains", "cocktailed", "peacockier", "coldcocked", "poppycocks", "pinchcocks", "peacocking", "peacockish", "cockroach", "poppycock", "cockatiel", "cockscomb", "cocklebur", "billycock", "cockhorse", "cocksfoot", "pinchcock", "princocks", "recocking", "stopcocks", "uncocking", "woodcocks", "cockshuts", "cockspurs", "cockshies", "cocktails", "cockswain", "peacocked", "moorcocks", "gamecocks", "coldcocks", "cockiness", "cockerels", "cockering", "cocklofts", "cockneyfy", "cockamamy", "cockatoos", "cockapoos", "cockateel", "cockboats", "cockcrows", "cockbills", "cocktail", "woodcock", "cockeyed", "cockatoo", "cocksure", "cockerel", "stopcock", "moorcock", "gamecock", "cockspur", "cockapoo", "cockcrow", "cockboat", "cockloft", "coldcock", "gorcocks", "haycocks", "cockneys", "cockshut", "uncocked", "peacocks", "peacocky", "princock", "recocked", "seacocks", "cockpits", "cockered", "cockiest", "cockeyes", "cockbill", "bibcocks", "bawcocks", "cockaded", "cockades", "cockpit", "peacock", "cockney", "haycock", "cockade", "seacock", "cockeye", "bibcock", "gorcock", "cockups", "recocks", "uncocks", "bawcock", "cockers", "cockier", "cockily", "cocking", "cockled", "cockles", "cockle", "cocked", "cockup", "recock", "uncock", "cocky", "damnablenesses", "damnableness", "damnations", "damnation", "damnatory", "damnified", "damnifies", "bedamning", "damnable", "bedamned", "damnably", "damneder", "damnify", "damning", "bedamns", "damners", "damned", "damner", "bedamn", "dickcissels", "dickcissel", "dickenses", "dickering", "benedicks", "dickered", "benedick", "medicks", "zaddick", "medick", "inspissations", "inspissators", "inspissating", "inspissation", "inspissated", "inspissates", "inspissator", "inspissate", "pissoirs", "pissants", "pissant", "pissoir", "sourpusses", "pussytoes", "sourpuss", "pussleys", "pusslies", "pussley", "pussly"
]

html = re.compile("<(.*?)>")
url = re.compile("https?://\\s+|www\\.\\s+")
nonalpha = re.compile("[^a-z]+")
multispace = re.compile("\\s\\s+")
multichar = re.compile("(.)\\1{2,}")


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def clean(text):
    text = text.lower()
    text = html.sub(" ", text)
    text = url.sub(" ", text)
    text = nonalpha.sub(" ", text)
    text = multispace.sub(" ", text)
    text = multichar.sub("\\1", text)

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    lemmatizer = WordNetLemmatizer()
    lemmatized_text = []

    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_text.append(word)
        else:
            lemmatized_text.append(lemmatizer.lemmatize(word, tag))

    return " ".join([word for word in lemmatized_text if word not in STOPWORDS])


def get_embeddings(sentences):
    preprocessed_text = preprocessor(sentences)

    return encoder(preprocessed_text)['pooled_output']


def balanced_recall(y_true, y_pred):
    recall_by_class = 0

    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]

        true_positives = backend.sum(backend.round(backend.clip(y_true_class * y_pred_class, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true_class, 0, 1)))

        recall = true_positives / (possible_positives + backend.epsilon())
        recall_by_class = recall_by_class + recall

    return recall_by_class / y_pred.shape[1]


def balanced_precision(y_true, y_pred):
    precision_by_class = 0

    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        y_true_class = y_true[:, i]

        true_positives = backend.sum(backend.round(backend.clip(y_true_class * y_pred_class, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred_class, 0, 1)))

        precision = true_positives / (predicted_positives + backend.epsilon())
        precision_by_class = precision_by_class + precision

    return precision_by_class / y_pred.shape[1]


def balanced_f1_score(y_true, y_pred):
    precision = balanced_precision(y_true, y_pred)
    recall = balanced_recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + backend.epsilon()))


def train():
    if args.nobs:
        df = pd.read_csv("comments_in.csv", index_col=0, usecols=["id", "comment_text", "obscene"], nrows=args.nobs)
    else:
        df = pd.read_csv("comments_in.csv", index_col=0, usecols=["id", "comment_text", "obscene"])

    curses = list(pd.read_csv("curses_in.csv", names=["curses"])["curses"])

    if args.clean:
        with alive_bar(len(df.index), title="Cleaning") as bar:
            for i, comment in df["comment_text"].iteritems():
                df.at[i, "comment_text"] = clean(comment)

                bar()

        for curse in curses:
            df[curse] = df["comment_text"].str.contains(curse)

        df = df[df[curses].sum(axis=1) == 1]
        df["Label"] = 0

        with alive_bar(len(df.index), title="Analyzing") as bar:
            for i, row in df.iterrows():
                if row["obscene"]:
                    df.at[i, "Label"] = list(row)[2:].index(True) + 1

                bar()

        df[["Label", "comment_text"]].to_csv("comments_out.csv")

    if args.train:
        if args.nobs:
            df = pd.read_csv("comments_out.csv", names=["id", "Label", "Text"], skiprows=1, index_col="id", nrows=args.nobs)
        else:
            df = pd.read_csv("comments_out.csv", names=["id", "Label", "Text"], skiprows=1, index_col="id")

        num_classes = len(df["Label"].value_counts())

        y = tf.keras.utils.to_categorical(df["Label"].values, num_classes=num_classes)
        x_train, x_test, y_train, y_test = train_test_split(df["Text"], y, test_size=0.25)

        i = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
        x = preprocessor(i)
        x = encoder(x)
        x = tf.keras.layers.Dropout(.2, name="dropout")(x["pooled_output"])
        x = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

        n_epochs = 100

        model = tf.keras.Model(i, x)

        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            balanced_recall,
            balanced_precision,
            balanced_f1_score
        ]

        earlystop_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_balanced_f1_score",
            patience=3,
            restore_best_weights=True,
            mode="max",
            min_delta=.01
        )

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=metrics
        )

        model_fit = model.fit(
            x_train,
            y_train,
            epochs=n_epochs,
            validation_data=(x_test, y_test),
            callbacks=[earlystop_callback]
        )

        metric_list = list(model_fit.history.keys())
        num_metrics = int(len(metric_list)/2)

        fig, ax = plt.subplots(nrows=1, ncols=num_metrics, figsize=(30, 5))

        for i in range(num_metrics):
            ax[i].plot(model_fit.history[metric_list[i]], marker="o", label=metric_list[i].replace("_", " "))
            ax[i].plot(model_fit.history[metric_list[i+num_metrics]], marker="o", label=metric_list[i+num_metrics].replace("_", " "))
            ax[i].set_xlabel("epochs", fontsize=14)
            ax[i].set_title(metric_list[i].replace("_", " "), fontsize=20)
            ax[i].legend(loc="lower left")

        plt.show()

        model.save("classifier")

    model = tf.keras.models.load_model(
        "classifier",
        custom_objects={
            "balanced_recall": balanced_recall,
            "balanced_precision": balanced_precision,
            "balanced_f1_score": balanced_f1_score
        }
    )

    test_strings = [
        clean(string) for string in [
            "fuck Jonathan and all of his friends",
            "i wish you would go suck a cock",
            "of course hes just another dick and harry",
            "she got dicked down last night"
        ]
    ]

    np.set_printoptions(suppress=True)

    print(
        json.dumps(
            dict(
                zip(
                    test_strings,
                    [
                        dict(
                            zip(
                                ["clean"] + curses, np.round(pred, 2).astype(float)
                            )
                        ) for pred in model.predict(test_strings)
                    ]
                )
            ),
            indent=2
        )
    )


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", help="reclean the dataset", action="store_true")
    parser.add_argument("--train", help="retrain the model", action="store_true")
    parser.add_argument("--nobs", help="number of observations to consider", type=int)
    args = parser.parse_args()

    train()
