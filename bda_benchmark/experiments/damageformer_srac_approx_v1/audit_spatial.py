"""Independent Val audit entry. Never changes S3 stride or trains models."""
from .audit import main

if __name__ == '__main__':
    main(spatial=True)
