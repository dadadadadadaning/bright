"""Safe superpixel-relation ablations for the frozen P3 occurrence path."""

# V2 fixes the shuffled-region control, building-boundary contamination, and
# train/evaluation superpixel-scale mismatch.  Keep it distinct from V1 so an
# old checkpoint cannot silently enter the corrected experiment.
PROTOCOL = "SPI_SUPERPIXEL_RELATION_OCCURRENCE_V2"
