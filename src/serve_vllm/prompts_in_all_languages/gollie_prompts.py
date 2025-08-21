"""
GoLLIE prompt headers for NER in multiple languages.
Each variable is a string containing the code scaffold
to be prepended before `text = ...; result = [` when building prompts.
"""

# ---------------- German ----------------
GERMAN_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Deutsche Personennamen. Schließt individuelle Menschen ein (Vorname + Nachname, ggf. Titel).
    Schließe Organisationen und Orte aus.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Organisationen in deutscher Sprache (Behörden, Firmen, Parteien, Vereine, internationale Org.).
    Schließe Personen- und Ortsnamen aus.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Ortsnamen in deutscher Sprache (Städte, Länder, Regionen, geografische Orte).
    Schließe Personen und Organisationen aus.\"\"\"
    mention: str
"""

# ---------------- English ----------------
ENGLISH_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"English personal names. Includes individual people (first + last name, possibly titles).
    Exclude organizations and places.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Organizations in English (companies, government bodies, parties, NGOs, international orgs).
    Exclude persons and locations.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Geographical names in English (cities, countries, regions, landmarks).
    Exclude persons and organizations.\"\"\"
    mention: str
"""

# ---------------- Chinese ----------------
CHINESE_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"中文的人名，包括个人姓名（姓和名），可能包含头衔。
    不包括组织和地点。\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"中文的组织名称（公司、政府机构、政党、非政府组织、国际组织等）。
    不包括人名和地名。\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"中文的地名（城市、国家、地区、地理位置）。
    不包括人名和组织。\"\"\"
    mention: str
"""

# ---------------- Arabic ----------------
ARABIC_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"الأسماء الشخصية باللغة العربية. تشمل أسماء الأفراد (الاسم الأول + اسم العائلة، وقد تتضمن الألقاب).
    استبعاد أسماء المنظمات والأماكن.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"أسماء المنظمات باللغة العربية (شركات، هيئات حكومية، أحزاب، منظمات غير حكومية، منظمات دولية).
    استبعاد أسماء الأشخاص والأماكن.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"أسماء الأماكن باللغة العربية (مدن، دول، مناطق، مواقع جغرافية).
    استبعاد الأشخاص والمنظمات.\"\"\"
    mention: str
"""

# ---------------- Bulgarian ----------------
BULGARIAN_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Лични имена на български език. Включва индивидуални хора (собствено и фамилно име, възможни титли).
    Изключва организации и места.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Организации на български език (компании, държавни институции, партии, НПО, международни организации).
    Изключва личности и географски имена.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Географски имена на български език (градове, държави, региони, географски обекти).
    Изключва личности и организации.\"\"\"
    mention: str
"""
