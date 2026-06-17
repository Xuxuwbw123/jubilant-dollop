

try:
    from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                                 QTableWidget, QTableWidgetItem,
                                 QHeaderView)
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QColor
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

from common.audio_config import Alert


class HistoryWidget(QWidget):
    """报警历史记录面板 (增强版: 排序 + 暗色主题)"""

    COLUMNS = ["时间", "级别", "类别", "严重度", "描述"]
    MAX_ROWS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    # ═══════════════════════════════════════════════════════════
    # 表格初始化 — QTableWidget 列定义、表头、缩放模式
    # ═══════════════════════════════════════════════════════════

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # 标题
        title = QLabel("报警历史")
        title.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # 表格
        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)

        # 列宽
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 时间
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 级别
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 类别
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 严重度
        header.setSectionResizeMode(4, QHeaderView.Stretch)           # 描述
        header.setSortIndicatorShown(True)  # 显示排序指示器

        layout.addWidget(self._table)

    # ═══════════════════════════════════════════════════════════
    # 报警记录追加 — 颜色编码严重度（红/橙/黄），自动滚动
    # Auto-scroll and max row limit enforcement
    # ═══════════════════════════════════════════════════════════

    # ---- 数据 ----

    def add_alert(self, alert: Alert):
        """添加一条报警记录"""
        # 插入到第一行
        self._table.insertRow(0)

        # 时间
        from common.utils import timestamp_str
        time_item = QTableWidgetItem(timestamp_str(alert.timestamp))
        time_item.setToolTip(f"帧: {alert.frame_index}")
        self._table.setItem(0, 0, time_item)

        # 级别 (三级颜色: 危急红/警告橙/注意黄)
        level_cn = {"critical": "危急", "warning": "警告", "pre_alert": "注意"}
        level_item = QTableWidgetItem(level_cn.get(alert.level, alert.level.upper()))
        level_colors = {
            "critical": QColor(255, 60, 60),
            "warning": QColor(240, 160, 48),
            "pre_alert": QColor(200, 180, 50),
        }
        level_item.setForeground(level_colors.get(alert.level, QColor(200, 150, 0)))
        if alert.level == "critical":
            level_item.setFont(QFont(level_item.font().family(), -1, QFont.Bold))
        self._table.setItem(0, 1, level_item)

        # 类别
        class_cn = {"normal": "正常", "scream": "尖叫", "cry": "大哭",
                    "laugh": "大笑"}
        class_item = QTableWidgetItem(class_cn.get(alert.class_name, alert.class_name))
        if alert.class_name in ("scream", "cry"):
            class_item.setForeground(QColor(255, 100, 100))
        elif alert.class_name in ("laugh",):
            class_item.setForeground(QColor(240, 160, 48))
        self._table.setItem(0, 2, class_item)

        # 严重度
        sev = round(getattr(alert, 'severity', 0), 2)
        sev_item = QTableWidgetItem(f"{sev:.2f}")
        sev_item.setToolTip(f"严重度评分: {sev:.2f}")
        # 颜色: 越高越红
        if sev >= 0.7:
            sev_item.setForeground(QColor(255, 60, 60))
        elif sev >= 0.4:
            sev_item.setForeground(QColor(240, 160, 48))
        self._table.setItem(0, 3, sev_item)

        # 描述
        desc_item = QTableWidgetItem(alert.message)
        self._table.setItem(0, 4, desc_item)

        # 限制最大行数
        while self._table.rowCount() > self.MAX_ROWS:
            self._table.removeRow(self._table.rowCount() - 1)

    def clear(self):
        """清空历史"""
        self._table.setRowCount(0)
