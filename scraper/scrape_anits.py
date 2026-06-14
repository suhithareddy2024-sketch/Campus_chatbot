import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import io

# -----------------------------
# ANITS URLs
# -----------------------------

urls = [
    "https://www.anits.org/",
    "https://www.anits.org/department/chemical",
    "https://www.anits.org/department/civil",
    "https://www.anits.org/department/cse",
    "https://www.anits.org/department/cse-ai-ml",
    "https://www.anits.org/department/cse-ds",
    "https://www.anits.org/department/eee",
    "https://www.anits.org/department/ece",
    "https://www.anits.org/department/it",
    "https://www.anits.org/department/mech",
    "https://www.anits.org/department/mba",
    "https://www.anits.org/fed",
    "https://www.anits.org/department/chemistry",
    "https://www.anits.org/department/english",
    "https://www.anits.org/department/maths",
    "https://www.anits.org/department/physics",
    "https://www.anits.org/anits_glance",
    "https://www.anits.org/about_principal",
    "https://www.anits.org/our_team",
    "https://www.anits.org/placements",
    "https://www.anits.org/canteen"
]

pdf_urls = [
    "https://anits.edu.in/po/20.pdf"
]

# -----------------------------
# Storage
# -----------------------------

all_text = ""

headers = {
    "User-Agent": "Mozilla/5.0"
}

# -----------------------------
# Website Scraping
# -----------------------------

print("=" * 50)
print("SCRAPING WEBSITE PAGES")
print("=" * 50)

for url in urls:

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        print(f"{response.status_code} -> {url}")

        if response.status_code != 200:
            continue

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        if len(text) < 50:
            continue

        all_text += "\n\n"
        all_text += "=" * 100
        all_text += "\n"
        all_text += f"SOURCE PAGE: {url}\n"
        all_text += "=" * 100
        all_text += "\n\n"

        all_text += text

        print("✓ Added")

    except Exception as e:

        print(f"ERROR -> {url}")
        print(e)

# -----------------------------
# PDF Scraping
# -----------------------------

print("\n")
print("=" * 50)
print("SCRAPING PDF FILES")
print("=" * 50)

for pdf_url in pdf_urls:

    try:

        response = requests.get(
            pdf_url,
            headers=headers,
            timeout=20
        )

        print(f"PDF -> {pdf_url}")

        pdf_reader = PdfReader(
            io.BytesIO(response.content)
        )

        pdf_text = ""

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                pdf_text += page_text + "\n"

        all_text += "\n\n"
        all_text += "=" * 100
        all_text += "\n"
        all_text += f"PDF SOURCE: {pdf_url}\n"
        all_text += "=" * 100
        all_text += "\n\n"

        all_text += pdf_text

        print("✓ PDF Added")

    except Exception as e:

        print("PDF ERROR")
        print(e)

# -----------------------------
# Manual Campus Facts
# -----------------------------

campus_facts = """

======================================================================
ANITS CAMPUS FACTS
======================================================================

College Name:
Anil Neerukonda Institute of Technology and Sciences (ANITS)

Location:
Sangivalasa,
Bheemunipatnam Mandal,
Visakhapatnam,
Andhra Pradesh,
India

Principal Email:
principal@anits.edu.in

Admissions Contacts:
8712005999
8712008222

Website:
https://www.anits.org

Placement Information:
92% Placement Rate
150 Recruiting Partners

Departments:
Chemical Engineering
Civil Engineering
Computer Science and Engineering
Computer Science and Engineering (AI & ML)
Computer Science and Engineering (Data Science)
Electrical and Electronics Engineering
Electronics and Communication Engineering
Information Technology
Mechanical Engineering
MBA
Chemistry
Physics
Mathematics
English and Humanities

Facilities:
Library
Canteen
Hostel
Transportation
Training and Placement Cell
NCC
NSS

"""

all_text += campus_facts

# -----------------------------
# Save File
# -----------------------------

output_file = "scraper/anits_content.txt"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(all_text)

print("\n")
print("=" * 50)
print("SCRAPING COMPLETED")
print("=" * 50)

print(f"Saved to: {output_file}")
print(f"Total Characters: {len(all_text)}")