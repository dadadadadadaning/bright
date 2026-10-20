import sys

sys.dont_write_bytecode = True

from replay import main

raise SystemExit(main())
