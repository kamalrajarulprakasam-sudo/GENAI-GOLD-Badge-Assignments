"""Generate 3 sample PDF documents used to demonstrate the RAG pipeline.

Run once before `ingest.py`:

    python create_sample_pdfs.py

This creates `documents/company_handbook.pdf`, `documents/product_manual.pdf`
and `documents/it_security_policy.pdf`. Feel free to replace these with your
own real PDFs -- the ingestion script picks up every *.pdf file it finds in
the `documents/` folder.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

DOCS_DIR = Path(__file__).parent / "documents"

DOCUMENTS: dict[str, tuple[str, str]] = {
    "company_handbook.pdf": (
        "Acme Corp Employee Handbook",
        """
        Welcome to Acme Corp! This handbook explains our policies.

        Working Hours: Standard working hours are 9:00 AM to 6:00 PM,
        Monday through Friday, with a one hour lunch break. Employees may
        request flexible hours through their manager.

        Leave Policy: Full-time employees accrue 18 days of paid time off
        (PTO) per year, plus 10 public holidays. Sick leave of up to 12 days
        per year does not require advance notice, but employees should
        inform their manager as soon as possible.

        Remote Work: Employees may work remotely up to 3 days per week
        after completing a 90 day probation period, subject to manager
        approval. Fully remote arrangements require VP-level sign-off.

        Code of Conduct: All employees must treat colleagues, customers,
        and partners with respect. Harassment, discrimination, and
        retaliation of any kind are strictly prohibited and will result in
        disciplinary action up to and including termination.

        Expense Reimbursement: Business expenses (travel, client meals,
        software subscriptions) are reimbursed within 15 business days of
        submitting a receipt through the Expensify portal. The approval
        limit for a direct manager is $500 per expense; anything above that
        requires Finance approval.

        Performance Reviews: Formal performance reviews happen twice a
        year, in June and December. Employees set quarterly OKRs with their
        manager and receive a written review summarizing progress against
        those goals.
        """,
    ),
    "product_manual.pdf": (
        "NovaBrew Coffee Machine - User Manual",
        """
        NovaBrew Coffee Machine Model NB-200 User Manual.

        Setup: Remove all packaging materials, place the machine on a flat
        surface, and fill the removable 1.5 liter water tank with fresh
        cold water. Do not exceed the MAX fill line.

        Brewing a Cup: Press the power button and wait for the ready light
        to turn solid blue (approximately 30 seconds). Select your brew
        size (Small 150ml, Medium 250ml, Large 350ml) using the dial, then
        press Start. The machine automatically stops when the selected
        volume is reached.

        Descaling: The NovaBrew should be descaled every 2 months, or when
        the descale indicator light flashes amber. Use the included
        descaling solution mixed with water in a 1:4 ratio, run it through
        the machine using the Descale cycle, then run two cycles of plain
        water to rinse.

        Cleaning: The drip tray and water tank are dishwasher safe on the
        top rack. Wipe the exterior with a damp cloth only; never immerse
        the base unit in water.

        Troubleshooting: If the machine will not turn on, check that it is
        plugged into a working outlet and that the water tank is properly
        seated -- the tank has a safety sensor that blocks power when it is
        missing. If brewing is unusually slow, this usually indicates
        mineral buildup and the unit should be descaled.

        Warranty: The NovaBrew NB-200 carries a 2 year limited warranty
        covering manufacturing defects. Warranty does not cover damage from
        failure to descale the unit as recommended.
        """,
    ),
    "it_security_policy.pdf": (
        "Acme Corp IT Security Policy",
        """
        Acme Corp Information Security Policy, Version 3.

        Password Requirements: All corporate accounts must use passwords of
        at least 14 characters, combining upper case, lower case, numbers,
        and symbols. Passwords must be rotated every 180 days and must not
        be reused across the last 10 passwords.

        Multi-Factor Authentication: MFA is mandatory for all access to
        email, VPN, and cloud infrastructure (AWS, GCP, Azure) accounts.
        Approved MFA methods are the Okta Verify app or a hardware security
        key; SMS-based MFA is disallowed for administrative accounts.

        Device Policy: Only company-issued or MDM-enrolled personal
        devices may access corporate email and internal systems. Devices
        must have disk encryption enabled and a screen lock timeout of 5
        minutes or less.

        Data Classification: Data is classified as Public, Internal,
        Confidential, or Restricted. Restricted data (customer PII,
        payment information) may only be stored in approved, encrypted
        systems and may never be emailed or copied to USB drives.

        Incident Reporting: Any suspected security incident -- lost
        device, phishing email, suspicious login -- must be reported to
        security@acmecorp.example within 1 hour of discovery by emailing
        the security team or filing a ticket in the #security-incidents
        channel.

        Vendor Access: Third-party vendors requiring system access must
        sign a Data Processing Agreement and be provisioned with
        time-limited accounts that expire automatically after 90 days
        unless renewed.
        """,
    ),
}


def build_pdf(path: Path, title: str, body: str) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER)
    story = [Paragraph(title, styles["Title"]), Spacer(1, 0.3 * inch)]

    for raw_paragraph in textwrap.dedent(body).strip().split("\n\n"):
        clean = " ".join(line.strip() for line in raw_paragraph.splitlines()).strip()
        if clean:
            story.append(Paragraph(clean, styles["BodyText"]))
            story.append(Spacer(1, 0.15 * inch))

    doc.build(story)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, body) in DOCUMENTS.items():
        out_path = DOCS_DIR / filename
        build_pdf(out_path, title, body)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
