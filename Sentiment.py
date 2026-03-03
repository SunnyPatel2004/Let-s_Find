#GLOBAL LIBRARY IMPORTS

# Core
import pandas as pd
import numpy as np
import re
import string
import ast

# NLP utilities
import nltk
import contractions
import emoji
from num2words import num2words

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Deep Learning / Transformers
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertForSequenceClassification


#NLTK RESOURCE DOWNLOADS

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

#CLEANING FUNCTION
def Cleaning(csv_path: str) -> pd.DataFrame:
    """
    Cleans raw college review data.
    """

    df = pd.read_csv(csv_path)

    # Extract year
    df['Enrolled Year'] = df['Enrolled Year'].astype(str).str.extract(r'(\d{4})')
    df['Enrolled Year'] = df['Enrolled Year'].fillna('NA')

    # Drop missing section data
    df = df.dropna(subset=['Section Data'])

    # Fake reviewers
    fake_reviewers = {
        "College Ninja": "False review",
        "Review guru": "False review",
        "Review Ninja": "False review",
        "Review Rebel": "False review",
        "Rating Sage": "False review",
        "Review Ranger": "False review"
    }

    df['Student Name'] = df['Student Name'].replace(fake_reviewers)
    df = df[df['Student Name'] != 'False review']

    df = df.drop_duplicates()

    return df



#PREPROCESSING FUNCTION


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Text normalization + token generation.
    """

    # Expand contractions
    df["Section Data"] = df["Section Data"].astype(str).apply(contractions.fix)

    # Remove noise patterns
    df["Section Data"] = (
        df["Section Data"]
        .str.replace(r'(\nLikes\n)|(\nDislikes\n)', ' ', regex=True)
        .str.replace(r'(\n)+', ' ', regex=True)
        .str.replace(r'^What Students Say\s*', '', regex=True)
    )

    # Emoji & special char removal
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\U00002600-\U000026FF"
        "]+", flags=re.UNICODE
    )

    def clean_text(text):
        text = emoji_pattern.sub("", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    df["Corrected_Text"] = df["Section Data"].apply(clean_text).str.lower()

    # Convert numbers → words
    def numbers_to_text(text):
        def replace(match):
            num = match.group()
            return num2words(float(num)) if "." in num else num2words(int(num))
        return re.sub(r'\d+(\.\d+)?', replace, text)

    df["Corrected_Text"] = df["Corrected_Text"].apply(numbers_to_text)

    # Tokenization
    df["Summary_Token"] = df["Corrected_Text"].apply(word_tokenize)

    # Stopword removal
    stop_words = set(stopwords.words("english"))
    df["Summary_Token"] = df["Summary_Token"].apply(
        lambda x: [w for w in x if w not in stop_words]
    )

    # Lemmatization
    lemm = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    def lemmatize(tokens):
        return [
            lemm.lemmatize(w, get_wordnet_pos(pos))
            for w, pos in pos_tag(tokens)
        ]

    df["Summary_Token"] = df["Summary_Token"].apply(lemmatize)

    # Cleanup
    df["Summary_Token"] = df["Summary_Token"].apply(
        lambda x: [t for t in x if len(t) > 1]
    )

    df = df[df["Summary_Token"].apply(len) > 1].reset_index(drop=True)
    df.drop(columns=["Section Data"], inplace=True)

    return df


#SENTIMENT SCORING

def sentiment_score(
    df: pd.DataFrame,
    text_column: str = "Corrected_Text",
    model_dir: str = "C:/Users/patel/Desktop/Python/Model/Sentiment/sentiment_model",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Generates signed sentiment score ∈ [-1, 1]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    texts = df[text_column].astype(str).tolist()
    scores = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1)

        scores.extend(((probs[:, 1] * 2) - 1).cpu().numpy())

    df["Sentiment_Score"] = scores
    return df

# Post-Sentiment Processing
def post_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-processing after sentiment scoring.
    Extracts structured fields required for the recommendation system.
    """

    Special = {
    #  COMPUTER CORE 
    "COMPUTER SCIENCE": "COMPUTER SCIENCE AND ENGINEERING",
    "COMPUTER SCIENCE ENGINEERING": "COMPUTER SCIENCE AND ENGINEERING",
    "COMPUTER SCIENCE AND ENGINEERING": "COMPUTER SCIENCE AND ENGINEERING",
    "COMPUTER ENGINEERING": "COMPUTER SCIENCE AND ENGINEERING",
    "INFORMATION SCIENCE": "COMPUTER SCIENCE AND ENGINEERING",
    "CSE": "COMPUTER SCIENCE AND ENGINEERING",
    "C.S.E": "COMPUTER SCIENCE AND ENGINEERING",
    "COMPUTER ENGG": "COMPUTER SCIENCE AND ENGINEERING",

    #  INFORMATION TECHNOLOGY 
    "INFORMATION TECHNOLOGY": "INFORMATION TECHNOLOGY",
    "INFORMATION TECH": "INFORMATION TECHNOLOGY",
    "INFO TECH": "INFORMATION TECHNOLOGY",
    "IT": "INFORMATION TECHNOLOGY",

    #  AI / DATA 
    "ARTIFICIAL INTELLIGENCE": "ARTIFICIAL INTELLIGENCE",
    "AI": "ARTIFICIAL INTELLIGENCE",

    "MACHINE LEARNING": "ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING",
    "ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING": "ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING",
    "AI ML": "ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING",

    "DATA SCIENCE": "DATA SCIENCE",
    "DATA ANALYTICS": "DATA SCIENCE",
    "BIG DATA": "DATA SCIENCE",
    "DS": "DATA SCIENCE",

    "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE": "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
    "AI AND DATA SCIENCE": "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
    "AI & DS": "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",
    "AI DS": "ARTIFICIAL INTELLIGENCE AND DATA SCIENCE",

    #  CYBER / IOT 
    "CYBER SECURITY": "CYBER SECURITY",
    "CYBER FORENSICS": "CYBER SECURITY",
    "INFORMATION SECURITY": "CYBER SECURITY",

    "INTERNET OF THINGS": "INTERNET OF THINGS",
    "IOT": "INTERNET OF THINGS",

    "BLOCKCHAIN": "BLOCKCHAIN",

    #  ELECTRONICS 
    "ELECTRONICS AND COMMUNICATION": "ELECTRONICS AND COMMUNICATION ENGINEERING",
    "ELECTRONICS AND COMMUNICATION ENGINEERING": "ELECTRONICS AND COMMUNICATION ENGINEERING",
    "ELECTRONICS & COMMUNICATION ENGINEERING": "ELECTRONICS AND COMMUNICATION ENGINEERING",
    "ECE": "ELECTRONICS AND COMMUNICATION ENGINEERING",
    "ELECTRONICS AND TELE COMMUNICATION ENGINEERING":"ELECTRONICS AND TELECOMMUNICATION ENGINEERING",
    "ELECTRONICS AND TELECOMMUNICATION ENGINEERING": "ELECTRONICS AND TELECOM ENGINEERING",
    "ELECTRONICS AND COMMUNICATION TECHNOLOGY":"ELECTRONICS AND COMMUNICATION ENGINEERING",


    "VLSI": "VLSI DESIGN",
    "EMBEDDED SYSTEMS": "EMBEDDED SYSTEMS",
    "INSTRUMENTATION": "INSTRUMENTATION ENGINEERING",

    #  ELECTRICAL 
    "ELECTRICAL ENGINEERING": "ELECTRICAL AND ELECTRONICS ENGINEERING",
    "ELECTRICAL AND ELECTRONICS": "ELECTRICAL AND ELECTRONICS ENGINEERING",
    "ELECTRICAL AND ELECTRONICS ENGINEERING": "ELECTRICAL AND ELECTRONICS ENGINEERING",
    "EEE": "ELECTRICAL AND ELECTRONICS ENGINEERING",

    #  MECHANICAL 
    "MECHANICAL ENGINEERING": "MECHANICAL ENGINEERING",
    "MECHANICAL ENGG": "MECHANICAL ENGINEERING",
    "ME": "MECHANICAL ENGINEERING",

    "AUTOMOBILE": "AUTOMOBILE ENGINEERING",
    "PRODUCTION": "PRODUCTION ENGINEERING",
    "MANUFACTURING": "MANUFACTURING ENGINEERING",
    "MECHATRONICS": "MECHATRONICS ENGINEERING",

    #  CIVIL 
    "CIVIL ENGINEERING": "CIVIL ENGINEERING",
    "CIVIL ENGG": "CIVIL ENGINEERING",
    "CE": "CIVIL ENGINEERING",

    "STRUCTURAL": "STRUCTURAL ENGINEERING",
    "TRANSPORTATION": "TRANSPORTATION ENGINEERING",
    "GEOTECHNICAL": "GEOTECHNICAL ENGINEERING",
    "ENVIRONMENTAL": "ENVIRONMENTAL ENGINEERING",

    #  SCIENCES
    "PHYSICS": "PHYSICS",
    "CHEMISTRY": "CHEMISTRY",
    "MATHEMATICS": "MATHEMATICS",
    "STATISTICS": "STATISTICS",
    "BIOLOGY": "BIOLOGY",
    "BIOTECHNOLOGY": "BIOTECHNOLOGY",
    "BIOTECH": "BIOTECHNOLOGY",

    #  MANAGEMENT 
    "BUSINESS ADMINISTRATION": "BUSINESS ADMINISTRATION",
    "BUSINESS MANAGEMENT": "BUSINESS ADMINISTRATION",
    "MANAGEMENT STUDIES": "BUSINESS ADMINISTRATION",
    "MANAGEMENT": "BUSINESS ADMINISTRATION",
    "BBA": "BUSINESS ADMINISTRATION",
    "MBA": "BUSINESS ADMINISTRATION",

    "MARKETING": "MARKETING",
    "FINANCE": "FINANCE",
    "HUMAN RESOURCE": "HUMAN RESOURCE MANAGEMENT",

    #  MEDICAL / HEALTH 
    "PHARMACY": "PHARMACY",
    "PHARMACEUTICAL": "PHARMACY",
    "PHARMACEUTICAL SCIENCES": "PHARMACY",
    "PHARMA": "PHARMACY",

    "NURSING": "NURSING",
    "PHYSIOTHERAPY": "PHYSIOTHERAPY",
    "PHYSICAL THERAPY": "PHYSIOTHERAPY",
    "MEDICAL": "MEDICAL",
}

    DEGREES = {

        "BACHELOR OF TECHNOLOGY (B.Tech)": [
            "b.tech", "b tech", "btech",
            "bachelor of technology",
            "bachelor of technology b tech"
        ],

        "BACHELOR OF ENGINEERING (B.E)": [
            "be", "b.e", "b e",
            "bachelor of engineering",
            "bachelor engineering"
        ],

        "MASTER OF TECHNOLOGY (M.Tech)": [
            "m.tech", "mtech",
            "master of technology"
        ],

        "MASTER OF ENGINEERING (M.E)": [
            "m.e", "me",
            "master of engineering"
        ],

        "BACHELOR OF SCIENCE (B.Sc)": [
            "b.sc", "bsc", "b.sc.",
            "bs", "b.s",
            "bachelor of science",
            "bachelor of science bs",
            "bsc hons", "bsc honours",
            "bsc it", "bsc cs", "bsc physics",
            "bsc chemistry", "bsc biology",
            "bsc agriculture", "bsc nursing"
        ],

        "MASTER OF SCIENCE (M.Sc)": [
            "m.sc", "msc",
            "ms", "m.s",
            "master of science",
            "master of science ms",
            "msc physics", "msc chemistry",
            "msc botany", "msc zoology",
            "msc maths"
        ],

        "BACHELOR OF ARTS (B.A)": [
            "ba", "b.a",
            "bachelor of arts",
            "ba hons", "ba honours",
            "ba programme", "ba program"
        ],

        "MASTER OF ARTS (M.A)": [
            "ma", "m.a",
            "master of arts"
        ],

        "BACHELOR OF COMMERCE (B.Com)": [
            "b.com", "bcom", "b com",
            "bachelor of commerce",
            "bcom hons"
        ],

        "MASTER OF COMMERCE (M.Com)": [
            "m.com", "mcom",
            "master of commerce"
        ],

        "BACHELOR OF BUSINESS ADMINISTRATION (BBA)": [
            "bba",
            "bachelor of business administration",
            "bba hons"
        ],

        "MASTER OF BUSINESS ADMINISTRATION (MBA)": [
            "mba",
            "master of business administration",
            "executive mba",
            "international mba"
        ],

        "BACHELOR OF COMPUTER APPLICATIONS (BCA)": [
            "bca", "b.c.a",
            "bachelor of computer application",
            "bachelor of computer applications"
        ],

        "MASTER OF COMPUTER APPLICATIONS (MCA)": [
            "mca",
            "master of computer application",
            "master of computer applications"
        ],

        "BACHELOR OF LAW (LLB)": [
            "llb", "l.l.b",
            "bachelor of law"
        ],

        "INTEGRATED LAW (BA LLB / B.Com LLB)": [
            "ba llb", "ballb", "b.a llb",
            "b.com llb", "bcom llb",
            "bachelor of arts and bachelor of law"
        ],

        "MASTER OF LAW (LLM)": [
            "llm", "l.l.m",
            "master of law"
        ],

        "BACHELOR OF PHARMACY (B.Pharm)": [
            "b.pharm", "bpharm", "b pharma",
            "bpharmacy", "bachelor of pharmacy"
        ],

        "MASTER OF PHARMACY (M.Pharm)": [
            "m.pharm", "mpharm",
            "master of pharmacy"
        ],

        "BACHELOR OF MEDICINE & SURGERY (MBBS)": [
            "mbbs", "bachelor of medicine"
        ],

        "BACHELOR OF DENTAL SURGERY (BDS)": [
            "bds", "bachelor of dental surgery"
        ],

        "BACHELOR OF ARCHITECTURE (B.Arch)": [
            "b.arch", "bachelor of architecture"
        ],

        "BACHELOR OF DESIGN (B.Des)": [
            "b.des", "bdes",
            "bachelor of design"
        ],

        "DIPLOMA / POLYTECHNIC": [
            "diploma", "polytechnic",
            "advanced diploma"
        ],

        "BACHELOR OF VOCATIONAL STUDIES (B.Voc)": [
            "b.voc", "bvoc",
            "bachelor of vocational"
        ],

        "BACHELOR OF EDUCATION (B.Ed)": [
            "b.ed", "bed",
            "bachelor of education"
        ],

        "MASTER OF EDUCATION (M.Ed)": [
            "m.ed", "med",
            "master of education"
        ],

        "DOCTOR OF PHILOSOPHY (Ph.D)": [
            "phd", "ph.d",
            "doctor of philosophy"
        ],

        "POST GRADUATE DIPLOMA / PROGRAM IN MANAGEMENT (PGDM / PGPM)": [
            "pgdm", "pgpm", "pgdbm",
            "pgdm in marketing", "pgdm in finance",
            "pgdm in banking", "pgdm in human resource",
            "pgdm in operations", "pgdm in analytics",
            "pgdm in international business",
            "pgdm retail", "pgdm ib",
            "pg program",
            "executive pg program",
            "ipm", "mhrm", "mhrd", "mdp", "mfm"
        ],

        "BACHELOR / MASTER OF PHYSIOTHERAPY (BPT / MPT)": [
            "bpt", "b.p.t",
            "mpt", "m.p.t",
            "physiotherapy",
            "sports physiotherapy",
            "orthopaedics physiotherapy",
            "musculoskeletal physiotherapy",
            "community physiotherapy"
        ],

        "DOCTOR OF PHARMACY (Pharm.D)": [
            "pharm.d", "pharmd", "pharm.d (pb)"
        ],

        "DIPLOMA IN PHARMACY(D.Pharm)": [
            "d.pharm", "dpharm", "d.p.h",
        ],

        "POSTGRADUATE MEDICAL DEGREE (MD / MDS)": [
            "md ", "m.d",
            "md in pathology",
            "md in pharmacology",
            "md in radiodiagnosis",
            "mds", "m.d.s",
            "mds orthodontics",
            "mds oral medicine"
        ],

        "ALLIED HEALTH SCIENCES": [
            "allied health science",
            "operation theatre technician",
            "dotat", "bmlt", "bott", "ott",
            "cvt", "bmr it", "medical radiotherapy"
        ],

        "JOURNALISM & MASS COMMUNICATION (BJMC / MJMC)": [
            "bjmc", "mjmc",
            "bjmc and mjmc",
            "bjmc and dmlp",
            "mjmc and dmlp",
            "bmm", "b.m.m", "bmc", "mj"
        ],

        "HOTEL MANAGEMENT & HOSPITALITY STUDIES": [
            "bhm", "b.h.m", "bhmct",
            "hospitality and hotel administration",
            "bttm", "mttm", "bha"
        ],

        "COMPUTER SCIENCE & DATA SCIENCE (NON-B.Tech)": [
            "computer science",
            "data science",
            "artificial intelligence",
            "artificial intelligence and data science",
            "aids",
            "computer eng",
            "electronics and computer science"
        ],

        "VETERINARY SCIENCES (B.VSc / M.VSc)": [
            "b.v.sc", "bvsc",
            "veterinary", "animal husbandry",
            "veterinary public health",
            "m.v.sc"
        ],

        "TEACHER EDUCATION (D.El.Ed / B.P.Ed / M.P.Ed)": [
            "d.el.ed", "d.ed",
            "b.p.ed", "m.p.ed", "bteach"
        ],

        "FINE ARTS & DESIGN": [
            "bfa", "b.f.a",
            "bva", "b.v.a",
            "animation",
            "applied arts",
            "drawing and painting",
            "textile design",
            "b.des", "m.des"
        ],

        "PERFORMING ARTS": [
            "bpa", "dance",
            "music", "vocal", "instrumental"
        ],

        "LAW (GENERAL)": [
            "law"
        ],

        "CERTIFICATE / FOUNDATION": [
            "certificate", "certification",
            "hsc", "puc", "pre university",
            "cec", "rs-cit","fellowship"
        ],

    }



    # CLASS SIZE & FEE EXTRACTION
    def extract_class_size_and_fee(text):
        class_size, fee = None, None
        m1 = re.search(r"Class Size:\s*(\d+)", text)
        m2 = re.search(r"(Course Fees?|Hostel Fee):\s*INR\s*([\d,]+)", text)

        if m1:
            class_size = int(m1.group(1))
        if m2:
            fee = int(m2.group(2).replace(",", ""))

        return class_size, fee

    df[["Class Size", "Course Fee"]] = df["Review"].astype(str).apply(
        lambda x: pd.Series(extract_class_size_and_fee(x))
    )

    
    # CATEGORY RATINGS
    rating_patterns = {
        "Campus Life": r"Campus Life\n(\d+(?:\.\d+)?)",
        "Faculty": r"Faculty\n(\d+(?:\.\d+)?)",
        "Placement Experience": r"Placement Experience\n(\d+(?:\.\d+)?)",
        "Hostel Facilities": r"Hostel Facilities\n(\d+(?:\.\d+)?)"
    }

    for col in rating_patterns:
        df[col] = np.nan

    for idx, review in df["Review"].astype(str).items():
        for col, pattern in rating_patterns.items():
            match = re.search(pattern, review)
            if match:
                df.at[idx, col] = float(match.group(1))

    # NORMALIZATION
    df["Rating"] = df["Rating"].astype(str).str.split().str[0].astype(float)
    df["College"] = df["College"].astype(str).str.upper().str.strip()
    df["Course"] = df["Course"].astype(str).str.upper().str.strip()

    # DEGREE / SPECIALIZATION SPLIT
    df["Course"] = (
        df["Course"]
        .str.replace(r"\s*\+\s*", " AND ", regex=True)
        .str.replace(r"\s*,\s*", " IN ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    df[["Degree", "Specialization"]] = df["Course"].str.split(" IN ", n=1, expand=True)

    def normalize_degree(text):
        t = str(text).lower()
        for degree, variants in DEGREES.items():
            if any(v in t for v in variants):
                return degree
        return "OTHER"

    df["Degree"] = df["Degree"].apply(normalize_degree)
    df = df[df["Degree"] != "OTHER"]

    df["Specialization"] = (
        df["Specialization"]
        .fillna("GENERAL")
        .str.upper()
        .map(Special)
        .fillna("GENERAL")
    )

    # FINAL CLEANUP
    drop_cols = [
        "Student Name", "Enrolled Year", "Date Reviewed",
        "Sentiment_Encoded", "Course", "Unnamed: 0"
    ]

    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df.reset_index(drop=True)
