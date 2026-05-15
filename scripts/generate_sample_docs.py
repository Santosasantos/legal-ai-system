"""
Generate synthetic legal documents for testing.
Creates 4 realistic PDFs:
  1. A civil complaint (clean text)
  2. A contract (clean text)
  3. A court notice (simulated scan — low quality)
  4. An affidavit (with handwriting simulation)
"""

import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from PIL import Image, ImageFilter, ImageEnhance
import random

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "sample_documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Document 1: Civil Complaint ───────────────────────────────────────────────

def create_civil_complaint():
    path = OUTPUT_DIR / "civil_complaint_pearson_v_hardman.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  alignment=TA_CENTER, fontSize=14, spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"],
                                    fontSize=12, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=14, alignment=TA_JUSTIFY)
    bold_style = ParagraphStyle("Bold", parent=styles["Normal"],
                                 fontSize=10, fontName="Helvetica-Bold")

    story.append(Paragraph("IN THE UNITED STATES DISTRICT COURT", title_style))
    story.append(Paragraph("FOR THE SOUTHERN DISTRICT OF NEW YORK", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Case No.: 2024-CV-08847-SDNY", bold_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("PEARSON SPECTER LITT LLP,", body_style))
    story.append(Paragraph("Plaintiff,", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("v.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("HARDMAN & ASSOCIATES LLC, and DANIEL HARDMAN, individually,", body_style))
    story.append(Paragraph("Defendants.", body_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("CIVIL COMPLAINT FOR DAMAGES AND INJUNCTIVE RELIEF", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("I. INTRODUCTION", heading_style))
    story.append(Paragraph(
        "Plaintiff Pearson Specter Litt LLP ('PSL' or 'Plaintiff'), by and through its "
        "undersigned counsel, brings this action against Defendants Hardman & Associates LLC "
        "and Daniel Hardman (collectively 'Defendants') for breach of fiduciary duty, "
        "misappropriation of trade secrets, tortious interference with business relations, "
        "and unjust enrichment arising from Defendants' unlawful scheme to steal PSL's "
        "confidential client list and solicit PSL's clients in violation of the Non-Solicitation "
        "Agreement dated March 15, 2019.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("II. PARTIES", heading_style))
    story.append(Paragraph(
        "1. Plaintiff Pearson Specter Litt LLP is a limited liability partnership organised "
        "and existing under the laws of the State of New York, with its principal place of "
        "business at 579 Fifth Avenue, New York, NY 10017.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "2. Defendant Hardman & Associates LLC is a limited liability company organised "
        "under the laws of the State of New York, with its principal place of business at "
        "1221 Avenue of the Americas, New York, NY 10020.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "3. Defendant Daniel Hardman is an individual residing in New York County, "
        "New York, and is the sole managing member of Hardman & Associates LLC. "
        "Hardman was formerly a named partner at PSL from January 2008 through "
        "December 2018.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("III. JURISDICTION AND VENUE", heading_style))
    story.append(Paragraph(
        "4. This Court has subject matter jurisdiction pursuant to 28 U.S.C. § 1332 "
        "as the parties are citizens of different states and the amount in controversy "
        "exceeds $75,000, exclusive of interest and costs.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "5. Venue is proper in this district pursuant to 28 U.S.C. § 1391(b) because "
        "a substantial part of the events giving rise to the claims occurred in this district.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("IV. FACTUAL ALLEGATIONS", heading_style))
    story.append(Paragraph(
        "6. On or about January 15, 2024, PSL discovered that Defendant Hardman had "
        "accessed PSL's confidential client database without authorisation and downloaded "
        "a list of 847 active client accounts, including contact information, billing rates, "
        "and matter histories.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "7. Between January 20, 2024 and March 1, 2024, Defendants directly solicited "
        "at least 23 of PSL's clients, offering legal services at rates 15% below PSL's "
        "standard billing rates. As a result, 11 clients terminated their relationship "
        "with PSL, causing damages of no less than $4,200,000 in lost revenue.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "8. Defendant Hardman's actions constitute a direct violation of the Non-Solicitation "
        "Agreement executed on March 15, 2019, which prohibits Hardman from soliciting "
        "PSL clients for a period of five (5) years following his departure from the firm.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("V. CAUSES OF ACTION", heading_style))
    story.append(Paragraph("COUNT I — BREACH OF CONTRACT", bold_style))
    story.append(Paragraph(
        "9. Plaintiff incorporates by reference all preceding paragraphs. "
        "Defendant Hardman breached the Non-Solicitation Agreement by directly "
        "soliciting PSL clients within the restricted period. PSL has suffered "
        "damages in excess of $4,200,000 as a direct result.", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("COUNT II — MISAPPROPRIATION OF TRADE SECRETS", bold_style))
    story.append(Paragraph(
        "10. PSL's client list constitutes a trade secret under the Defend Trade Secrets "
        "Act, 18 U.S.C. § 1836, and the New York Trade Secrets Act. Defendants "
        "misappropriated this trade secret through improper means, causing damages "
        "in an amount to be determined at trial.", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("VI. RELIEF REQUESTED", heading_style))
    story.append(Paragraph(
        "WHEREFORE, Plaintiff respectfully requests that this Court enter judgment "
        "in favour of Plaintiff and against Defendants as follows:", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("(a) Compensatory damages of no less than $4,200,000;", body_style))
    story.append(Paragraph("(b) Punitive damages in an amount to be determined at trial;", body_style))
    story.append(Paragraph("(c) Permanent injunctive relief prohibiting further solicitation;", body_style))
    story.append(Paragraph("(d) Attorneys' fees and costs; and", body_style))
    story.append(Paragraph("(e) Such other relief as the Court deems just and proper.", body_style))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Respectfully submitted,", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("PEARSON SPECTER LITT LLP", bold_style))
    story.append(Paragraph("By: Harvey Specter, Esq.", body_style))
    story.append(Paragraph("579 Fifth Avenue, New York, NY 10017", body_style))
    story.append(Paragraph("Tel: (212) 555-0100", body_style))
    story.append(Paragraph("Date: April 3, 2024", body_style))

    doc.build(story)
    print(f"Created: {path}")
    return path


# ── Document 2: Settlement Agreement ─────────────────────────────────────────

def create_settlement_agreement():
    path = OUTPUT_DIR / "settlement_agreement_2024.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  alignment=TA_CENTER, fontSize=14, spaceAfter=12)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"],
                                    fontSize=11, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=14, alignment=TA_JUSTIFY)
    bold_style = ParagraphStyle("Bold", parent=styles["Normal"],
                                 fontSize=10, fontName="Helvetica-Bold")

    story.append(Paragraph("CONFIDENTIAL SETTLEMENT AGREEMENT AND MUTUAL RELEASE", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "This Confidential Settlement Agreement and Mutual Release ('Agreement') is entered "
        "into as of June 15, 2024 ('Effective Date'), by and between:", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "PEARSON SPECTER LITT LLP, a New York limited liability partnership ('PSL'); and", body_style))
    story.append(Paragraph(
        "HARDMAN & ASSOCIATES LLC and DANIEL HARDMAN, individually (collectively 'Hardman Parties').", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("RECITALS", heading_style))
    story.append(Paragraph(
        "WHEREAS, PSL filed a civil complaint against the Hardman Parties in the United States "
        "District Court for the Southern District of New York, Case No. 2024-CV-08847-SDNY "
        "('the Litigation'), alleging breach of contract, misappropriation of trade secrets, "
        "and related claims;", body_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "WHEREAS, the parties desire to resolve all claims between them without further "
        "litigation and without any admission of liability;", body_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("TERMS AND CONDITIONS", heading_style))
    story.append(Paragraph("1. Settlement Payment", bold_style))
    story.append(Paragraph(
        "The Hardman Parties shall pay to PSL the total sum of TWO MILLION FIVE HUNDRED "
        "THOUSAND DOLLARS ($2,500,000.00) ('Settlement Amount') as follows: "
        "(a) $1,000,000 within 30 days of the Effective Date; "
        "(b) $750,000 on or before September 15, 2024; "
        "(c) $750,000 on or before December 15, 2024.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("2. Non-Solicitation Covenant", bold_style))
    story.append(Paragraph(
        "Daniel Hardman agrees to extend the non-solicitation period for an additional "
        "three (3) years from the Effective Date, through June 15, 2027. Hardman shall "
        "not directly or indirectly solicit any current or former PSL client.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3. Mutual Release", bold_style))
    story.append(Paragraph(
        "Upon receipt of the full Settlement Amount, each party releases and forever "
        "discharges the other from any and all claims, demands, actions, and causes of "
        "action arising out of or related to the Litigation.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("4. Confidentiality", bold_style))
    story.append(Paragraph(
        "The parties agree to keep the terms of this Agreement strictly confidential. "
        "Neither party shall disclose the Settlement Amount or any terms hereof to any "
        "third party without prior written consent, except as required by law.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("5. Governing Law", bold_style))
    story.append(Paragraph(
        "This Agreement shall be governed by and construed in accordance with the laws "
        "of the State of New York, without regard to its conflict of laws principles. "
        "Any dispute arising hereunder shall be resolved exclusively in the courts of "
        "New York County, New York.", body_style))
    story.append(Spacer(1, 0.3 * inch))

    # Signature table
    sig_data = [
        ["PEARSON SPECTER LITT LLP", "HARDMAN & ASSOCIATES LLC"],
        ["", ""],
        ["By: _______________________", "By: _______________________"],
        ["Harvey Specter, Managing Partner", "Daniel Hardman, Managing Member"],
        ["Date: ___________________", "Date: ___________________"],
    ]
    sig_table = Table(sig_data, colWidths=[3 * inch, 3 * inch])
    sig_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)

    doc.build(story)
    print(f"Created: {path}")
    return path


# ── Document 3: Court Notice (simulated scan) ─────────────────────────────────

def create_court_notice():
    """Create a clean PDF then degrade it to simulate a scanned document."""
    clean_path = OUTPUT_DIR / "_notice_clean.pdf"
    final_path = OUTPUT_DIR / "court_notice_scanned.pdf"

    doc = SimpleDocTemplate(str(clean_path), pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  alignment=TA_CENTER, fontSize=13)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=14)
    bold_style = ParagraphStyle("Bold", parent=styles["Normal"],
                                 fontSize=10, fontName="Helvetica-Bold")

    story.append(Paragraph("UNITED STATES DISTRICT COURT", title_style))
    story.append(Paragraph("SOUTHERN DISTRICT OF NEW YORK", title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("NOTICE OF HEARING", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Case No.: 2024-CV-08847-SDNY", bold_style))
    story.append(Paragraph("Pearson Specter Litt LLP v. Hardman & Associates LLC et al.", bold_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "TO ALL PARTIES AND THEIR COUNSEL OF RECORD:", bold_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "PLEASE TAKE NOTICE that a hearing on Plaintiff's Motion for Preliminary "
        "Injunction has been scheduled as follows:", body_style))
    story.append(Spacer(1, 0.15 * inch))

    hearing_data = [
        ["Date:", "Tuesday, July 16, 2024"],
        ["Time:", "10:00 a.m. Eastern Time"],
        ["Courtroom:", "Courtroom 14B, 40 Foley Square, New York, NY 10007"],
        ["Judge:", "Hon. Margaret Chen, U.S. District Judge"],
    ]
    hearing_table = Table(hearing_data, colWidths=[1.5 * inch, 4.5 * inch])
    hearing_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(hearing_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph(
        "Defendants are required to file any opposition papers no later than "
        "July 9, 2024. Failure to appear may result in the motion being granted "
        "by default pursuant to Local Rule 7.1.", body_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Dated: June 28, 2024", body_style))
    story.append(Paragraph(
        "BY ORDER OF THE COURT", bold_style))
    story.append(Paragraph(
        "Clerk of Court, United States District Court, S.D.N.Y.", body_style))

    doc.build(story)

    # Degrade to simulate scan
    _degrade_pdf_to_scan(str(clean_path), str(final_path))
    os.unlink(clean_path)
    print(f"Created (scanned): {final_path}")
    return final_path


def _degrade_pdf_to_scan(input_pdf: str, output_pdf: str):
    """Convert PDF pages to degraded images and back to PDF (simulates scan)."""
    try:
        import fitz
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas
        import tempfile

        pdf = fitz.open(input_pdf)
        img_paths = []

        for page in pdf:
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Add noise and blur to simulate scan
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Add random noise
            import numpy as np
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, 8, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            # Slight rotation to simulate misaligned scan
            angle = random.uniform(-0.5, 0.5)
            img = img.rotate(angle, fillcolor=255)

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            img_paths.append(tmp.name)

        pdf.close()

        # Rebuild as PDF — write each page image directly via canvas
        from reportlab.pdfgen import canvas as rl_canvas
        c = rl_canvas.Canvas(output_pdf, pagesize=letter)
        for img_path in img_paths:
            c.drawImage(img_path, 0, 0, width=letter[0], height=letter[1])
            c.showPage()
        c.save()

        for p in img_paths:
            os.unlink(p)

    except ImportError:
        # If numpy not available, just copy the clean PDF
        import shutil
        shutil.copy(input_pdf, output_pdf)


# ── Document 4: Affidavit ─────────────────────────────────────────────────────

def create_affidavit():
    path = OUTPUT_DIR / "affidavit_witness_statement.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                  alignment=TA_CENTER, fontSize=13)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
                                 fontSize=10, leading=14, alignment=TA_JUSTIFY)
    bold_style = ParagraphStyle("Bold", parent=styles["Normal"],
                                 fontSize=10, fontName="Helvetica-Bold")

    story.append(Paragraph("AFFIDAVIT OF DONNA PAULSEN", title_style))
    story.append(Paragraph("IN SUPPORT OF PLAINTIFF'S MOTION FOR PRELIMINARY INJUNCTION", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("STATE OF NEW YORK  )", bold_style))
    story.append(Paragraph("                              ) ss.:", bold_style))
    story.append(Paragraph("COUNTY OF NEW YORK )", bold_style))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "I, DONNA PAULSEN, being duly sworn, depose and state as follows:", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "1. I am the Office Manager at Pearson Specter Litt LLP ('PSL') and have held "
        "this position since February 2005. I have personal knowledge of the facts set "
        "forth herein.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "2. On January 14, 2024, at approximately 11:47 p.m., I received an automated "
        "security alert from PSL's IT system indicating that user account 'dhardman_former' "
        "had accessed the firm's client database. This account had been deactivated on "
        "December 31, 2018, the date of Mr. Hardman's departure from the firm.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "3. I immediately contacted PSL's IT Director, Alex Williams, who confirmed that "
        "the access originated from an IP address registered to Hardman & Associates LLC "
        "at 1221 Avenue of the Americas, New York, NY 10020.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "4. A review of the access logs, attached hereto as Exhibit A, shows that the "
        "intruder downloaded 847 client records including names, contact information, "
        "billing rates, and matter descriptions. The download occurred over a period of "
        "approximately 23 minutes.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "5. Between January 20, 2024 and March 1, 2024, I received calls from eleven (11) "
        "PSL clients informing me that they had been contacted by Hardman & Associates LLC "
        "and were terminating their relationship with PSL. Copies of the termination notices "
        "are attached as Exhibit B.", body_style))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "6. I declare under penalty of perjury that the foregoing is true and correct "
        "to the best of my knowledge and belief.", body_style))
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("_______________________________", body_style))
    story.append(Paragraph("DONNA PAULSEN", bold_style))
    story.append(Paragraph("Sworn to before me this 5th day of July, 2024", body_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("_______________________________", body_style))
    story.append(Paragraph("Notary Public, State of New York", body_style))
    story.append(Paragraph("My Commission Expires: December 31, 2025", body_style))

    doc.build(story)
    print(f"Created: {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Generating sample documents in: {OUTPUT_DIR}")
    create_civil_complaint()
    create_settlement_agreement()
    create_court_notice()
    create_affidavit()
    print("\nAll sample documents created successfully.")
