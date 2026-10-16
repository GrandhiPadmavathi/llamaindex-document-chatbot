"""Generate a small sample PDF under sample_docs/ (fake content only)."""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_docs" / "quick_start_guide.pdf"


def main() -> None:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("reportlab not installed; skip PDF generation")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "OrbitMobile Quick Start Guide")
    c.setFont("Helvetica", 11)
    lines = [
        "Welcome to OrbitMobile prepaid service.",
        "",
        "Activation: Insert SIM, power on, dial *228# to activate.",
        "Default voicemail PIN is 0000 — change it within 24 hours.",
        "",
        "Self-care URL: https://example.invalid/orbitmobile (placeholder).",
        "Support chat is available 24/7 from the mobile app.",
        "",
        "Page 1 — Getting started",
    ]
    y = 690
    for line in lines:
        c.drawString(72, y, line)
        y -= 16

    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(72, 720, "Page 2 — Safety and SIM")
    c.setFont("Helvetica", 11)
    lines2 = [
        "Do not share your SIM PIN or account OTP with anyone.",
        "Report lost devices within 2 hours to lock the line.",
        "eSIM users: keep backup codes in a secure password manager.",
    ]
    y = 690
    for line in lines2:
        c.drawString(72, y, line)
        y -= 16

    c.save()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
