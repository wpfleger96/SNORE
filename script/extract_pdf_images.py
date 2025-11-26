#!/usr/bin/env python3
"""
Extract Images from OSCAR Guide PDF

Extracts all images from the OSCAR Guide PDF and saves them to a temporary
directory for manual review and organization.

Usage:
    python scripts/extract_pdf_images.py
    python scripts/extract_pdf_images.py --output data/guidelines/images/temp
"""

import argparse
import logging
from pathlib import Path
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_images_from_pdf(pdf_path: Path, output_dir: Path):
    """
    Extract all images from PDF.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images
    """
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)

    image_count = 0
    images_by_page = {}

    # Extract images from each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()

        if image_list:
            images_by_page[page_num + 1] = len(image_list)

        for img_index, img in enumerate(image_list):
            # Get image XREF
            xref = img[0]

            # Extract pixmap
            try:
                pix = fitz.Pixmap(doc, xref)

                # Convert CMYK to RGB if necessary
                if pix.n - pix.alpha > 3:  # CMYK
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Save only if RGB or GRAY
                if pix.n - pix.alpha <= 3:
                    image_filename = f"page_{page_num + 1:02d}_img_{img_index + 1:02d}.png"
                    image_path = output_dir / image_filename

                    pix.save(str(image_path))
                    image_count += 1

                    logger.info(f"  Extracted: {image_filename} (page {page_num + 1})")

                pix = None  # Free memory

            except Exception as e:
                logger.warning(
                    f"  Failed to extract image {img_index + 1} from page {page_num + 1}: {e}"
                )

    doc.close()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total images extracted: {image_count}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("\nImages by page:")
    for page_num, count in sorted(images_by_page.items()):
        logger.info(f"  Page {page_num:2d}: {count} images")

    # Print next steps
    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info("1. Review images in the output directory")
    logger.info("2. Identify which images correspond to:")
    logger.info("   - Flow limitation classes (1-7)")
    logger.info("   - Respiratory events (OA, CA, Hypopnea, RERA)")
    logger.info("   - Breathing patterns (CSR, periodic breathing)")
    logger.info("3. Rename and organize:")
    logger.info("   data/guidelines/images/")
    logger.info("   ├── flow_limitation/")
    logger.info("   │   ├── class_1_sinusoidal.png")
    logger.info("   │   ├── class_2_double_peak.png")
    logger.info("   │   └── ...")
    logger.info("   ├── events/")
    logger.info("   │   ├── obstructive_apnea.png")
    logger.info("   │   └── ...")
    logger.info("   └── patterns/")
    logger.info("       └── csr_breathing.png")
    logger.info("4. Update patterns.py with image paths")
    logger.info("5. Delete the temp directory")
    logger.info("=" * 70)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract images from OSCAR Guide PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/guidelines/documents/OSCAR_The_Guide_Apnea_Board_Wiki.pdf"),
        help="Path to PDF file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/guidelines/images/temp"),
        help="Output directory for extracted images",
    )

    args = parser.parse_args()

    try:
        extract_images_from_pdf(args.pdf, args.output)
        logger.info("\n✓ Image extraction complete!")

    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
