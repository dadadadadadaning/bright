import sys

sys.dont_write_bytecode = True

from probe import main

raise SystemExit(main())
