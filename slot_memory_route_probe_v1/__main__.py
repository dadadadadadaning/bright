import sys

sys.dont_write_bytecode = True

from route_probe import main

raise SystemExit(main())
