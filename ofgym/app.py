"""진입점. `uv run main` 이 여기로 온다."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ofgym.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("optical_flow_gym")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
