# OSCAR Reference Images

## Overview

This directory contains reference images extracted from the OSCAR Guide and organized by function. These images support:
- Medical pattern recognition (flow limitation classes, respiratory events)
- Multimodal LLM analysis (comparing user waveforms with references)
- Documentation and user education
- UI reference for developers

**Total Images: 32**

## Directory Structure

```
images/
├── flow_limitation/  (2 images)
│   ├── OSCAR_flow_limitation_classes.png  ★ KEY IMAGE
│   └── OSCAR_flow_limitation_chart.png
├── events/  (1 image)
│   └── OSCAR_daily_standard_charts_event_flags_chart.png
├── patterns/  (8 images)
│   ├── OSCAR_arousal_recovery_breathing_chart_example.png
│   ├── OSCAR_arousal_recovery_breathing_chart_example_2.png
│   ├── OSCAR_palatal_prolapse_graph_example.png
│   ├── OSCAR_periodic_breathing_chart_example.png
│   ├── OSCAR_periodic_breathing_chart_example_resmed.png
│   ├── OSCAR_periodic_leg_movement_chart_example.png
│   ├── OSCAR_positional_apnea_chart_example.png
│   └── OSCAR_positional_apnea_chart_example_detailed.png
├── charts/  (10 images)
│   ├── OSCAR_flow_rate_graph_example.png
│   ├── OSCAR_flow_rate_detail_view.png
│   ├── OSCAR_leak_rate_graph_phillips.png
│   ├── OSCAR_leak_rate_graph_resmed.png
│   ├── OSCAR_mask_pressure_example_resmed_asv.png
│   ├── OSCAR_pressure_graph.png
│   ├── OSCAR_pressure_graph_detailed.png
│   ├── OSCAR_snore_rate_graph_resmed.png
│   ├── OSCAR_time_at_pressure_graph.png
│   └── OSCAR_timte_at_pressure_graph_APAP.png
└── ui/  (11 images)
    ├── OCAR_daily_screen_calendar_details.png
    ├── OSCAR_custom_event_flagging_settings.png
    ├── OSCAR_daily_AHI_graph.png
    ├── OSCAR_daily_screen.png
    ├── OSCAR_daily_screen_bookmarks_breakdown_tab.png
    ├── OSCAR_daily_screen_detailed_graphs_how_to_change_y_axis.png
    ├── OSCAR_daily_screen_event_breakdown_pie_chart.png
    ├── OSCAR_daily_screen_events_breakdown_tab.png
    ├── OSCAR_daily_screen_left_sidebar_details.png
    ├── OSCAR_optional_zeo_data_import.png
    └── OSCAR_sidebar_summary.png
```

## Key Images

### Flow Limitation Classes Diagram ★
**File:** `flow_limitation/OSCAR_flow_limitation_classes.png`

THE most important reference image showing all 7 flow limitation classes:
- Class 1: Sinusoidal (normal)
- Class 2: Double peak
- Class 3: Multiple tiny peaks
- Class 4: Early peak with plateau
- Class 5: Mid-peak with plateaus
- Class 6: Late peak
- Class 7: Plateau throughout

All 7 classes in `src/snore/constants.py` (`FLOW_LIMITATION_CLASSES`) reference this single diagram with different `reference_section` values.

### Complex Pattern Examples
**Directory:** `patterns/`

- **Periodic Breathing**: Two manufacturer variants showing waxing/waning amplitude
- **Positional Apnea**: Shows event clustering, detailed and overview versions
- **Palatal Prolapse**: Expiratory flow cutoff pattern
- **Arousal Recovery**: Post-arousal breathing changes (2 examples)
- **Periodic Leg Movement (PLM)**: Limb movement markers

### Chart Examples
**Directory:** `charts/`

Standard OSCAR visualizations for each data channel:
- Flow rate waveforms (general + detail view)
- Pressure graphs (CPAP/APAP)
- Leak rate (ResMed vs Philips manufacturers)
- Time at pressure histograms (CPAP vs APAP)
- Snore rate detection
- Mask pressure (ASV/VPAP devices)

## Usage in Code

### Access flow limitation class images
```python
from snore.constants import FLOW_LIMITATION_CLASSES

# Get image for Class 1
class_1 = FLOW_LIMITATION_CLASSES[1]
image_path = class_1["reference_image"]
section = class_1["reference_section"]
# → "data/guidelines/images/flow_limitation/OSCAR_flow_limitation_classes.png", "Class 1"
```

### Access image directories
```python
from snore.constants import IMAGE_DIRS

# Get directory paths
flow_limitation_dir = IMAGE_DIRS["flow_limitation"]
# → PosixPath("data/guidelines/images/flow_limitation")

patterns_dir = IMAGE_DIRS["patterns"]
# → PosixPath("data/guidelines/images/patterns")
```

### Load images for multimodal LLM
```python
from pathlib import Path
from PIL import Image
from snore.constants import FLOW_LIMITATION_CLASSES

# Load flow limitation reference diagram
image_path = Path(FLOW_LIMITATION_CLASSES[1]["reference_image"])
reference_image = Image.open(image_path)

# Send to Claude with user's waveform for comparison
# (Implementation in Phase 3-5)
```

## Image Metadata

### Format
- **Type**: PNG
- **Color**: RGB (converted from CMYK if needed during extraction)
- **Source**: OSCAR Guide PDF (26 pages)
- **Extraction Tool**: PyMuPDF (fitz)

### Naming Convention
`OSCAR_{category}_{description}_{variant}.png`

Examples:
- `OSCAR_flow_limitation_classes.png` - The main classification diagram
- `OSCAR_periodic_breathing_chart_example_resmed.png` - Manufacturer-specific
- `OSCAR_positional_apnea_chart_example_detailed.png` - Detail level variant

### Image Categories

1. **Reference Diagrams** (`flow_limitation/OSCAR_flow_limitation_classes.png`)
   - Authoritative classification diagrams
   - Multiple patterns in single image
   - Used for pattern matching and validation

2. **Chart Examples** (`charts/*.png`)
   - Standard OSCAR channel visualizations
   - Show normal/expected appearances
   - May have manufacturer variants

3. **Pattern Examples** (`patterns/*.png`)
   - Real-world examples of complex patterns
   - May have detailed/overview variants
   - Annotated to show key features

4. **Event Markers** (`events/*.png`)
   - Event flag displays
   - Temporal distribution examples

5. **UI References** (`ui/*.png`)
   - Screenshot documentation
   - User interface guides
   - Settings and configuration examples

## Adding New Images

1. **Extract from PDF**:
   ```bash
   python scripts/extract_pdf_images.py --pdf path/to/document.pdf
   ```

2. **Review and Rename**:
   - Check images in `temp/` directory
   - Delete irrelevant images (logos, borders)
   - Rename with descriptive names following convention

3. **Organize by Category**:
   - Medical patterns → `patterns/`
   - Flow limitation diagrams → `flow_limitation/`
   - Chart examples → `charts/`
   - UI screenshots → `ui/`
   - Event displays → `events/`

4. **Update Code References**:
   ```python
   # In patterns.py or chart_examples.py
   "reference_images": [
       "data/guidelines/images/category/filename.png"
   ]
   ```

5. **Commit to Git**:
   ```bash
   git add data/guidelines/images/
   git commit -m "Add reference images for [description]"
   ```

## Best Practices

1. **Keep Images Small**: Optimize PNG files if needed
   ```bash
   optipng -o5 image.png  # Lossless compression
   ```

2. **Descriptive Names**: Include context in filename
   - ✅ `OSCAR_flow_limitation_class_3_example.png`
   - ❌ `img_003.png`

3. **Version Control**: Commit images to git
   - Small PNGs are fine for git
   - If images become large, consider Git LFS

4. **Attribution**: Track source in filenames
   - `OSCAR_*.png` - From OSCAR Guide
   - `ApneaBoard_*.png` - From forum
   - `Paper_Author_*.png` - From research papers

5. **Multiple Variants**: Use suffixes for variations
   - `_resmed.png` vs `_phillips.png` - Manufacturer
   - `_detailed.png` vs `_overview.png` - Detail level
   - `_example_1.png`, `_example_2.png` - Multiple examples

## Image Usage Scenarios

### 1. Pattern Recognition (Phase 3)
Compare user waveforms against reference diagrams:
```python
# Load reference flow limitation classes
ref_image = load_image(FLOW_LIMITATION_CLASSES[1]["reference_image"])
# Compare user breath waveform to Class 1 reference
```

### 2. Multimodal LLM Analysis (Phase 5)
Send images to Claude for visual analysis:
```python
# Include reference image in LLM prompt
prompt = f"Compare this waveform to the reference Class {class_num} pattern"
# Attach both user waveform screenshot and reference image
```

### 3. Documentation Generation
Include images in analysis reports:
```python
from pathlib import Path

# Generate report with embedded images
image_path = Path("data/guidelines/images/patterns/OSCAR_positional_apnea_chart_example.png")
report.add_image(image_path)
report.add_caption("Example of positional apnea clustering")
```

### 4. UI Development
Show reference images in user interface:
```python
from pathlib import Path

# Display appropriate reference when user's data shows pattern
if detected_pattern == "periodic_breathing":
    image_path = Path("data/guidelines/images/patterns/OSCAR_periodic_breathing_chart_example.png")
    show_reference_image(image_path)
```

## Maintenance

### Periodic Review
- Verify all image paths in code are valid
- Check for broken links when reorganizing
- Update if OSCAR Guide releases new version

### Adding New Sources
When adding images from new sources:
1. Follow naming convention with source prefix
2. Add to appropriate category directory
3. Update `FLOW_LIMITATION_CLASSES` in constants.py if relevant
4. Document in this IMAGES.md file

### Image Validation
```python
# Verify all referenced images exist
from pathlib import Path
from snore.constants import FLOW_LIMITATION_CLASSES, IMAGE_DIRS

# Validate flow limitation class images
for class_data in FLOW_LIMITATION_CLASSES.values():
    image_path = Path(class_data["reference_image"])
    assert image_path.exists(), f"Missing: {image_path}"

# Validate image directories exist
for category, dir_path in IMAGE_DIRS.items():
    assert dir_path.exists(), f"Missing directory: {category} at {dir_path}"

print("✓ All image references valid")
```

## License and Attribution

Images extracted from "OSCAR - The Guide" (Apnea Board Wiki)
- Source: https://www.apneaboard.com/wiki/index.php/OSCAR_The_Guide
- License: Check Apnea Board Wiki license terms
- Use: Educational and analysis purposes for SNORE

Additional images added from other sources should include appropriate attribution.
