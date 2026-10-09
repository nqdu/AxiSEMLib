"""Allow ``python -m axisemlib`` to use the package CLI."""

from .cli import main


raise SystemExit(main())
