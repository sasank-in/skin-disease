import os
from typing import List, Tuple

import requests

DEFAULT_HOSPITALS = [
    {
        "name": "Central Dermatology Center",
        "specialty": "Dermatology & Skin Oncology",
        "contact": "Front desk: (000) 000-0000",
    },
    {
        "name": "Riverside Skin Clinic",
        "specialty": "General Dermatology",
        "contact": "Appointments: (000) 000-0000",
    },
]

HOSPITALS_BY_DISEASE = {
    "Acne": [
        {
            "name": "ClearSkin Institute",
            "specialty": "Acne & Cosmetic Dermatology",
            "contact": "Appointments: (000) 000-0000",
        }
    ],
    "Basal Cell Carcinoma": [
        {
            "name": "Allergy & Derm Care",
            "specialty": "Basal Cell Carcinoma",
            "contact": "Appointments: (000) 000-0000",
        }
    ],
    "Melanoma": [
        {
            "name": "Onco-Derm Center",
            "specialty": "Skin Oncology",
            "contact": "Cancer care: (000) 000-0000",
        }
    ],
    "Psoriasis": [
        {
            "name": "Psoriasis Treatment Hub",
            "specialty": "Chronic Skin Conditions",
            "contact": "Front desk: (000) 000-0000",
        }
    ],
    "Rosacea": [
        {
            "name": "Vascular Derm Clinic",
            "specialty": "Rosacea & Redness",
            "contact": "Appointments: (000) 000-0000",
        }
    ],
}


def get_hospitals(disease: str):
    return HOSPITALS_BY_DISEASE.get(disease, DEFAULT_HOSPITALS)


def build_practo_skin_clinics_url(location: str) -> str:
    base = "https://www.practo.com"
    loc = location.strip().lower()
    if not loc:
        return f"{base}/location/clinics/skin-clinics"
    slug = "-".join(loc.split())
    return f"{base}/{slug}/clinics/skin-clinics"


def fetch_practo_clinics(location: str) -> Tuple[List[dict], str | None]:
    url = build_practo_skin_clinics_url(location)
    user_agent = os.getenv(
        "PRACTO_USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.practo.com/",
    }
    cookie = os.getenv("PRACTO_COOKIE")
    if cookie:
        headers["Cookie"] = cookie

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200:
            return [], f"Practo responded with status {resp.status_code}."

        # Best-effort parse without external dependencies
        text = resp.text
        clinics = []
        # Very lightweight extraction to avoid brittle selectors
        for line in text.splitlines():
            if "clinic-card" in line or "clinic-name" in line:
                name = line.strip()
                if name and len(name) < 120:
                    clinics.append(
                        {
                            "name": name,
                            "specialty": "Skin Clinic",
                            "clinic": "Practo Listing",
                            "location": location or "location",
                            "fee": "",
                        }
                    )
            if len(clinics) >= 15:
                break

        if not clinics:
            return [], "No listings could be parsed from Practo. Access may be blocked."
        return clinics[:15], None
    except Exception as exc:
        return [], f"Unable to load live Practo listings: {exc}"


def get_top_cities_specialists():
    # Curated sample data for Indian cities (10 specialists per city)
    names = [
        "Dr. Aarav Mehta",
        "Dr. Kavya Rao",
        "Dr. Rohan Iyer",
        "Dr. Sneha Kulkarni",
        "Dr. Vivek Nair",
        "Dr. Nisha Bhat",
        "Dr. Priya Menon",
        "Dr. Suresh Rao",
        "Dr. Rahul Jain",
        "Dr. Aisha Khan",
    ]
    cities = {
        "Bangalore": ["Indiranagar", "Koramangala", "Whitefield", "Jayanagar", "HSR Layout", "MG Road", "BTM Layout", "Hebbal", "JP Nagar", "Yelahanka"],
        "Mumbai": ["Andheri West", "Bandra", "Juhu", "Powai", "Thane", "Dadar", "Borivali", "Chembur", "Malad", "Colaba"],
        "Delhi": ["South Delhi", "Dwarka", "Rohini", "Saket", "Lajpat Nagar", "Karol Bagh", "Noida", "Gurgaon", "Pitampura", "Mayur Vihar"],
        "Hyderabad": ["HITEC City", "Banjara Hills", "Jubilee Hills", "Gachibowli", "Madhapur", "Kondapur", "Begumpet", "Secunderabad", "Kukatpally", "LB Nagar"],
        "Chennai": ["Adyar", "T Nagar", "Velachery", "Anna Nagar", "OMR", "Porur", "Tambaram", "Nungambakkam", "Mylapore", "Guindy"],
        "Kolkata": ["Salt Lake", "Park Street", "New Town", "Garia", "Howrah", "Behala", "Dum Dum", "Jadavpur", "Ballygunge", "Kasba"],
        "Pune": ["Koregaon Park", "Hinjewadi", "Baner", "Aundh", "Kothrud", "Viman Nagar", "Hadapsar", "Wakad", "Shivajinagar", "Kharadi"],
        "Ahmedabad": ["SG Highway", "Navrangpura", "Prahladnagar", "Bodakdev", "Satellite", "Paldi", "Maninagar", "Ghatlodia", "Vastrapur", "Chandkheda"],
        "Jaipur": ["C Scheme", "Malviya Nagar", "Vaishali Nagar", "Tonk Road", "Jagatpura", "Mansarovar", "Bani Park", "MI Road", "Ajmer Road", "Raja Park"],
        "Lucknow": ["Gomti Nagar", "Hazratganj", "Indira Nagar", "Aliganj", "Rajajipuram", "Mahanagar", "Vikas Nagar", "Chowk", "Alambagh", "Jankipuram"],
        "Chandigarh": ["Sector 17", "Sector 35", "Sector 22", "Sector 44", "Sector 8", "Sector 15", "Sector 11", "Sector 9", "Sector 21", "Sector 10"],
        "Indore": ["Vijay Nagar", "Palasia", "Bhawarkua", "Rajwada", "New Palasia", "Saket", "Rau", "Bengali Square", "Choithram", "Annapurna"],
        "Bhopal": ["Arera Colony", "MP Nagar", "Kolar", "Bawadiya Kalan", "Habibganj", "TT Nagar", "Piplani", "Ashoka Garden", "Kotra", "Shahpura"],
        "Surat": ["Adajan", "Vesu", "City Light", "Piplod", "Varachha", "Katargam", "Udhna", "Rander", "Nanpura", "Athwa"],
    }
    data = {}
    for city, areas in cities.items():
        specialists = []
        for i in range(10):
            specialists.append(
                {
                    "name": names[i],
                    "specialty": "Dermatologist",
                    "clinic": f"{areas[i]} Skin Clinic",
                    "experience": f"{6 + i} yrs",
                    "address": areas[i],
                    "hours": "Mon-Sat 10am-6pm",
                    "phone": f"+91 90000 {city[:2].upper()}{i:02d}".replace(" ", ""),
                }
            )
        data[city] = specialists
    return data


def get_city_specialists(city: str):
    if not city:
        return []
    data = get_top_cities_specialists()
    key = city.strip().lower()
    for city_name, specialists in data.items():
        if city_name.lower() == key:
            return specialists
    return []
