"""Editing interface for trianimations.


"""
import sys
from os.path import abspath, join, split
from trianimate.triangulate import (
    get_triangle_means,
    triangulate,
    warp_colours,
)

import cv2 as cv
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QColor, QIcon, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

WHITE = QColor(255, 255, 255)
BLACK = QColor(0, 0, 0)
RED = QColor(255, 0, 0)
PRIMARY = QColor(53, 53, 53)
SECONDARY = QColor(35, 35, 35)
TERTIARY = QColor(42, 130, 218)


def css_rgb(color, a=False):
    """Get a CSS `rgb` or `rgba` string from a `QtGui.QColor`."""
    return ("rgba({}, {}, {}, {})" if a else "rgb({}, {}, {})").format(*color.getRgb())


def set_stylesheet(app):
    """Static method to set the tooltip stylesheet to a `QtWidgets.QApplication`."""
    app.setStyleSheet(
        "QToolTip {{"
        "color: {white};"
        "background-color: {tertiary};"
        "border: 1px solid {white};"
        "}}".format(white=css_rgb(WHITE), tertiary=css_rgb(TERTIARY))
    )


class QDarkPalette(QPalette):
    """Dark palette for a Qt application meant to be used with the Fusion theme."""

    def __init__(self, *__args):
        super().__init__(*__args)

        # Set all the colors based on the constants in globals
        self.setColor(QPalette.Window, PRIMARY)
        self.setColor(QPalette.WindowText, WHITE)
        self.setColor(QPalette.Base, SECONDARY)
        self.setColor(QPalette.AlternateBase, PRIMARY)
        self.setColor(QPalette.ToolTipBase, WHITE)
        self.setColor(QPalette.ToolTipText, WHITE)
        self.setColor(QPalette.Text, WHITE)
        self.setColor(QPalette.Button, PRIMARY)
        self.setColor(QPalette.ButtonText, WHITE)
        self.setColor(QPalette.BrightText, RED)
        self.setColor(QPalette.Link, TERTIARY)
        self.setColor(QPalette.Highlight, TERTIARY)
        self.setColor(QPalette.HighlightedText, BLACK)
        self.setColor(QPalette.Disabled, QPalette.WindowText, SECONDARY)
        self.setColor(QPalette.Disabled, QPalette.WindowText, SECONDARY)
        self.setColor(QPalette.Disabled, QPalette.Text, SECONDARY)
        self.setColor(QPalette.Disabled, QPalette.Light, PRIMARY)

    def set_app(self, app):
        """Set the Fusion theme and this palette to a `QtWidgets.QApplication`."""
        app.setStyle("Fusion")
        app.setPalette(self)
        set_stylesheet(app)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # initiate window members
        self.menu = QMenuBar()
        self.file_menu = QAction()

        self.toolbar = QToolBar()
        self.export_btn = QAction()
        self.import_btn = QAction()

        self.central = QWidget()
        self.grid = QGridLayout()
        self.tabs = QTabWidget()
        self.image = QImage()
        self.img_label = QLabel()

        self.triangle_tab = QWidget()
        self.animate_tab = QWidget()
        self.three_d_tab = QWidget()

        self.detail_lbl = QLabel()
        self.detail_sld = QSlider(Qt.Horizontal)
        self.thresh_lbl = QLabel()
        self.thresh_sld = QSlider(Qt.Horizontal)
        self.colour_lbl = QLabel()
        self.colour_sld = QSlider(Qt.Horizontal)
        self.bright_lbl = QLabel()
        self.bright_sld = QSlider(Qt.Horizontal)
        self.triangle_btn = QPushButton()

        self.animate_btn = QPushButton()

        self.three_d_btn = QPushButton()

        self.img_frame = QFrame()
        self.img_btn = QPushButton()

        # initiate state variables
        self.path = split(abspath(__file__))[0]
        self.filepath = ""
        self.saved = True
        self.img = np.zeros((720, 720, 3), dtype=np.uint8)
        self.preview = self.img.copy()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Trianimate - Timeline")
        self.setWindowIcon(QIcon(join(self.path, "icons", "trianimate.png")))
        self.setCentralWidget(self.central)
        self.central.setLayout(self.grid)

        self.setMenuBar(self.menu)
        self.file_menu = self.menu.addMenu("&File")

        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

        self.export_btn.setIcon(QIcon(join(self.path, "icons", "export.png")))
        self.export_btn.setIconText("Export Video")
        self.export_btn.setShortcut("Ctrl+E")
        self.export_btn.setStatusTip("Export Video")

        self.import_btn.setIcon(QIcon(join(self.path, "icons", "import.png")))
        self.import_btn.setIconText("Import Image")
        self.import_btn.setShortcut("Ctrl+I")
        self.import_btn.setStatusTip("Import Image")

        self.toolbar.addAction(self.export_btn)
        self.toolbar.addAction(self.import_btn)

        self.import_btn.triggered.connect(self.import_img)

        self.grid.setSpacing(10)
        self.img_frame.setFrameStyle(1)

        self.tabs.addTab(self.triangle_tab, "Triangulate")
        self.tabs.addTab(self.animate_tab, "Animate")
        self.tabs.addTab(self.three_d_tab, "3D")

        self.triangle_tab.layout = QVBoxLayout(self.triangle_tab)

        self.detail_lbl.setText("Detail")
        self.detail_sld.setMinimum(0)
        self.detail_sld.setMaximum(100)
        self.detail_sld.setValue(50)
        self.triangle_tab.layout.addWidget(self.detail_lbl)
        self.triangle_tab.layout.addWidget(self.detail_sld)

        self.thresh_lbl.setText("Threshold")
        self.thresh_sld.setMinimum(0)
        self.thresh_sld.setMaximum(100)
        self.thresh_sld.setValue(50)
        self.triangle_tab.layout.addWidget(self.thresh_lbl)
        self.triangle_tab.layout.addWidget(self.thresh_sld)

        self.colour_lbl.setText("Colour")
        self.colour_sld.setMinimum(-100)
        self.colour_sld.setMaximum(100)
        self.colour_sld.setValue(50)
        self.triangle_tab.layout.addWidget(self.colour_lbl)
        self.triangle_tab.layout.addWidget(self.colour_sld)

        self.bright_lbl.setText("Brightness")
        self.bright_sld.setMinimum(-100)
        self.bright_sld.setMaximum(100)
        self.bright_sld.setValue(50)
        self.triangle_tab.layout.addWidget(self.bright_lbl)
        self.triangle_tab.layout.addWidget(self.bright_sld)

        self.triangle_btn.setText("Triangulate")
        self.triangle_btn.clicked.connect(self.triangulate)
        self.triangle_tab.layout.addWidget(self.triangle_btn)
        self.triangle_tab.layout.addStretch(1)
        self.triangle_tab.layout.setSpacing(15)

        self.animate_tab.layout = QVBoxLayout(self.animate_tab)
        self.animate_tab.layout.addWidget(self.animate_btn)

        self.three_d_tab.layout = QVBoxLayout(self.three_d_tab)
        self.three_d_tab.layout.addWidget(self.three_d_btn)

        self.img_frame.layout = QVBoxLayout(self.img_frame)

        self.image = QImage(self.preview.data, 720, 720, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(self.image))
        self.img_frame.layout.addWidget(self.img_label)
        self.img_frame.layout.setAlignment(self.img_label, Qt.AlignCenter)

        self.grid.addWidget(self.tabs, 0, 0)
        self.grid.addWidget(self.img_frame, 0, 1)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 2)

        self.showMaximized()

    def import_img(self, _):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            "Image Files (*.jpg *.jpeg *.gif *.bmp *.png)",
            options=options,
        )
        if filename:
            self.img: np.ndarray = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
            oh, ow = self.img.shape[:2]
            maxd = 720
            if oh > maxd or ow > maxd:
                h, w = int(oh * maxd / max(oh, ow)), int(ow * maxd / max(oh, ow))
                self.preview = cv.resize(self.img, (w, h), interpolation=cv.INTER_AREA)
            else:
                h, w = oh, ow
                self.preview = self.img.copy()
            self.image = QImage(
                self.preview.data,
                w,
                h,
                int(self.preview.data.nbytes / h),
                QImage.Format_RGB888,
            )
            self.img_label.setPixmap(QPixmap.fromImage(self.image))

    def triangulate(self, _):
        det_val = self.detail_sld.value() / 100.0
        thr_val = self.thresh_sld.value() / 100.0
        col_val = self.colour_sld.value() / 100.0
        brt_val = self.bright_sld.value() / 100.0
        img = warp_colours(self.img, col_val, brt_val)
        vertices, faces = triangulate(img, det_val, thr_val)
        cols = get_triangle_means(img, vertices, faces)
        cols = warp_colours(cols, col_val, 0.0)

        h, w, d = self.preview.shape
        prev = np.zeros((h * 2, w * 2, d), dtype=np.uint8)

        for fdx, pts in enumerate(faces):
            curr_pts = vertices[pts, :] * np.array([[w * 2, h * 2]])
            prev = cv.fillConvexPoly(
                prev,
                np.int32([curr_pts]),
                color=cols[fdx].tolist(),
                lineType=cv.LINE_AA,
            )

        self.preview = cv.resize(prev, (w, h), interpolation=cv.INTER_AREA)
        self.image = QImage(
            self.preview.data,
            w,
            h,
            int(self.preview.data.nbytes / h),
            QImage.Format_RGB888,
        )
        self.img_label.setPixmap(QPixmap.fromImage(self.image))

    def save_work(self):
        if not self.filepath:
            pass
        self.saved = True

    def closeEvent(self, event: QCloseEvent):
        if self.saved:
            event.accept()
        else:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "Are you sure you want to quit? You have unsaved changes.",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self.save_work()
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()


def main():

    app = QApplication(sys.argv)
    palette = QDarkPalette()
    palette.set_app(app)

    t = MainWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
