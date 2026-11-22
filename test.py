import json
import easyocr
import ollama

CHAMPS = ["nom", "prenom", "date", "numero_document", "organisme_emetteur"]

# Le modèle OCR est chargé une seule fois
reader = easyocr.Reader(["fr", "en"], gpu=False)


def lire_texte(chemin_image):
    """Lit l'image telle quelle et renvoie tout le texte, une ligne par morceau."""
    textes = reader.readtext(chemin_image, detail=0)   # detail=0 : uniquement les textes
    return "\n".join(textes)


def extraire_champs(texte, modele="llama3.2"):
    """Demande au LLM d'extraire les 5 champs du texte OCR et renvoie un dict."""

    prompt = f"""Tu extrais des informations d'un document administratif à partir de son texte OCR.
Réponds uniquement avec un objet JSON contenant exactement ces clés :
"nom", "prenom", "date", "numero_document", "organisme_emetteur".

Règles :
- Recopie les valeurs exactement comme elles apparaissent dans le texte, sans les corriger.
- N'invente rien. Si une information est absente, mets null.
- "date" : la date du document, au format jj/mm/aaaa.
- "organisme_emetteur" : l'entreprise ou l'administration qui a émis le document (souvent en en-tête).
- Si tu ne peux pas séparer nom et prénom avec certitude, mets le nom complet dans "nom" et null dans "prenom".

Exemple :
Texte :
REÇU DE PAIEMENT
Banque Exemple - Sfax
Réf : REC-2019-00123
Date : 02/03/2019
Client : Trabelsi Sara
Réponse :
{{"nom": "Trabelsi", "prenom": "Sara", "date": "02/03/2019", "numero_document": "REC-2019-00123", "organisme_emetteur": "Banque Exemple"}}

Texte :
{texte}
Réponse :"""

    reponse = ollama.generate(
        model=modele,
        prompt=prompt,
        format="json",                                   # force une sortie JSON valide
        options={"temperature": 0, "num_predict": 300},  # déterministe, assez de place pour le JSON
    )

    try:
        brut = json.loads(reponse["response"])
        if not isinstance(brut, dict):                   # le modèle n'a pas renvoyé un objet
            brut = {}
    except json.JSONDecodeError:
        brut = {}

    champs = {}
    for cle in CHAMPS:
        valeur = brut.get(cle)                           # None si la clé manque
        if valeur is not None:
            valeur = str(valeur).strip()
            if valeur == "" or valeur.lower() in ("null", "none", "n/a"):
                valeur = None
        champs[cle] = valeur
    return champs


# --- Utilisation ---
texte = lire_texte("image.png")
print("Texte OCR :\n", texte, "\n")

champs = extraire_champs(texte)
print("Champs extraits :")
for cle, valeur in champs.items():
    print(f"  {cle} : {valeur}")
