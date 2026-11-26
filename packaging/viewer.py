"""Entry point for standalone desktop builds."""

from multiprocessing import freeze_support

from serveutil.desktop import main

if __name__ == "__main__":
    freeze_support()
    main()
