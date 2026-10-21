"""Zipapp command dispatcher; inventory mode has no third-party dependencies."""
import sys
sys.dont_write_bytecode = True

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print("Commands: inventory [--root RUN] [--output NEW_DIR]; check-reader --run RUN [--device cpu|cuda:0]")
    print("This package does NOT launch or resume training.")
    raise SystemExit(0)
command = sys.argv.pop(1)
if command == "inventory":
    from review import main
elif command == "check-reader":
    from check_reader import main
else:
    raise SystemExit("Unknown command: " + command)
result = main()
raise SystemExit(result or 0)
