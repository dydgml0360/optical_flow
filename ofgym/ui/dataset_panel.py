"""데이터셋 트리: 세션 > 샘플."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ofgym.config import PATHS
from ofgym.data import Sample, scan_sessions
from ofgym.recon import find_output

SAMPLE_ROLE = Qt.ItemDataRole.UserRole


class DatasetPanel(QWidget):
    sampleSelected = Signal(object)  # Sample
    statusMessage = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._root_label = QLabel()
        self._root_label.setWordWrap(True)
        self._root_label.setStyleSheet("color: palette(mid);")

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["세션 / 샘플"])
        self._tree.setUniformRowHeights(True)
        self._tree.currentItemChanged.connect(self._on_current_changed)

        refresh = QPushButton("다시 읽기")
        refresh.clicked.connect(self.reload)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self._root_label)
        layout.addWidget(self._tree, 1)
        layout.addWidget(refresh)
        # reload() 는 MainWindow 가 시그널을 연결한 뒤에 부른다.
        # 여기서 부르면 첫 선택의 sampleSelected 가 허공으로 날아간다.

    def reload(self) -> None:
        self._tree.clear()
        self._root_label.setText(str(PATHS.raw))

        sessions = scan_sessions()
        if not sessions:
            placeholder = QTreeWidgetItem([f"{PATHS.raw} 에 세션이 없습니다"])
            placeholder.setDisabled(True)
            self._tree.addTopLevelItem(placeholder)
            self.statusMessage.emit(f"데이터셋 없음: {PATHS.raw}")
            return

        total = 0
        for session in sessions:
            node = QTreeWidgetItem([f"{session.session_id}  ({len(session.samples)})"])
            for sample in session.samples:
                child = QTreeWidgetItem([""])
                child.setData(0, SAMPLE_ROLE, sample)
                child.setToolTip(0, str(sample.path))
                self._label_item(child, sample)
                node.addChild(child)
                total += 1
            self._tree.addTopLevelItem(node)
            node.setExpanded(True)

        self.statusMessage.emit(f"세션 {len(sessions)}개 / 샘플 {total}개")

        first = self._first_sample_item()
        if first is not None:
            self._tree.setCurrentItem(first)

    def _first_sample_item(self) -> Optional[QTreeWidgetItem]:
        for i in range(self._tree.topLevelItemCount()):
            node = self._tree.topLevelItem(i)
            if node.childCount():
                return node.child(0)
        return None

    @staticmethod
    def _label_item(item: QTreeWidgetItem, sample: Sample) -> None:
        """재구성 결과가 있으면 ● 를 붙인다."""
        marker = "● " if find_output(sample) is not None else "○ "
        item.setText(0, marker + sample.label)

    def _iter_sample_items(self):
        for i in range(self._tree.topLevelItemCount()):
            node = self._tree.topLevelItem(i)
            for j in range(node.childCount()):
                child = node.child(j)
                sample = child.data(0, SAMPLE_ROLE)
                if sample is not None:
                    yield child, sample

    def all_samples(self) -> List[Sample]:
        """지금 트리에 있는 모든 샘플 (일괄 처리용)."""
        return [sample for _item, sample in self._iter_sample_items()]

    def mark_recon(self, sample: Sample) -> None:
        """한 샘플의 3D 유무 표시만 갱신한다 (트리 전체를 다시 읽지 않는다)."""
        for item, existing in self._iter_sample_items():
            if existing.sample_id == sample.sample_id:
                self._label_item(item, existing)
                return

    def _on_current_changed(
        self, current: Optional[QTreeWidgetItem], _previous: Optional[QTreeWidgetItem]
    ) -> None:
        if current is None:
            return
        sample = current.data(0, SAMPLE_ROLE)
        if sample is not None:
            self.sampleSelected.emit(sample)
