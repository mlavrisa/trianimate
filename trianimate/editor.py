"""Editing interface for trianimations.

Run this module from the command line, or python -m trianimate
"""
import re
import sys
from os.path import abspath, join, split
from typing import Callable

import cv2 as cv
import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (QCloseEvent, QColor, QIcon, QImage, QMouseEvent,
                         QPalette, QPixmap, QValidator)
from PyQt5.QtWidgets import (QAction, QApplication, QCheckBox, QFileDialog,
                             QFrame, QGridLayout, QLabel, QLineEdit,
                             QMainWindow, QMenuBar, QMessageBox, QPushButton,
                             QSlider, QTabWidget, QToolBar, QVBoxLayout,
                             QWidget)

from trianimate.render import TriangleShader
from trianimate.triangulate import (find_colours, find_faces, find_vertices,
                                    warp_colours)

WHITE = QColor(255, 255, 255)
BLACK = QColor(0, 0, 0)
RED = QColor(255, 0, 0)
PRIMARY = QColor(53, 53, 53)
SECONDARY = QColor(35, 35, 35)
TERTIARY = QColor(42, 130, 218)
QUATERNARY = QColor(78, 78, 78)


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
        self.setColor(QPalette.Disabled, QPalette.ButtonText, QUATERNARY)
        self.setColor(QPalette.Disabled, QPalette.Button, SECONDARY)

    def set_app(self, app):
        """Set the Fusion theme and this palette to a `QtWidgets.QApplication`."""
        app.setStyle("Fusion")
        app.setPalette(self)
        set_stylesheet(app)


class CallableValidator(QValidator):
    """Provides a `QValidator` object which will call a function to test for validity.
    
    Specifically used for a QLineEdit validator, the arguments won't work for other
    widgets. The function in question returns a boolean value. Useful if validation
    requires accessing one or more of the scopes within the application in order to
    validate the content or not.
    """

    def __init__(self, validator: Callable):
        super().__init__()
        self.validator = validator

    def validate(self, txt: str, pos: int) -> int:
        """Calls the validator callable. If True, sends the Acceptable signal."""
        return (
            QValidator.Acceptable if self.validator(txt) else QValidator.Invalid,
            txt,
            pos,
        )


class TriangulateWorker(QObject):
    """Worker object for processing triangulations done in numba in parallel."""

    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    progress = pyqtSignal(int)

    def __init__(
        self,
        img: np.ndarray,
        detail: float,
        threshold: float,
        colour_boost: float,
        brightness_boost: float,
        maxd: int,
    ):
        """Create a new worker, pass in all the relevant values to triangulate.
        
        Args:
            img: `np.ndarray` (dtype: uint8, ndim: 3) the image
            detail: `float` value between 0 and 1 indicating the level of detail
            threshold: `float` value between 0 and 1 indicating background fraction
            colour_boost: `float` value between -1 and 1 indicating saturation change
            brightness_boost: `float` value between -1 and 1 indicating value change
        """
        super().__init__()
        self.img = img
        self.detail = detail
        self.threshold = threshold
        self.colour_boost = colour_boost
        self.brightness_boost = brightness_boost
        self.maxd = maxd

    def run(self):
        """Runs the triangulation, generates a preview image, emits finished signal."""
        img = warp_colours(self.img, self.colour_boost, self.brightness_boost)
        vertices = find_vertices(img, self.detail, self.threshold)
        faces = find_faces(vertices)
        cols = find_colours(img, vertices, faces)
        colours = warp_colours(cols, self.colour_boost, 0.0)

        # longest dimension of preview should always be self.maxd px
        aspect = self.img.shape[0] / self.img.shape[1]
        if aspect >= 1:
            h = self.maxd
            w = round(self.maxd / aspect)
        else:
            w = self.maxd
            h = round(self.maxd * aspect)

        # render the preview
        with TriangleShader().render_2d(w, h) as render:
            preview = render(vertices, faces, colours)

        # emit the finished signal, pass along the calculated values
        self.finished.emit(preview, vertices, faces, colours)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """Main application window for triangulating and animating images"""
        super().__init__()

        # initiate window members
        # menubar
        self.menu = QMenuBar()
        self.file_menu = QAction()

        # toolbar
        self.toolbar = QToolBar()
        self.import_btn = QAction()

        # main components and layouts
        self.central = QWidget()
        self.grid = QGridLayout()
        self.tabs = QTabWidget()
        self.image = QImage()
        self.img_label = QLabel()

        # tabs
        self.triangle_tab = QWidget()
        self.animate_tab = QWidget()
        self.three_d_tab = QWidget()
        self.export_tab = QWidget()

        # triangulation tab
        self.detail_lbl = QLabel()
        self.detail_sld = QSlider(Qt.Horizontal)
        self.thresh_lbl = QLabel()
        self.thresh_sld = QSlider(Qt.Horizontal)
        self.colour_lbl = QLabel()
        self.colour_sld = QSlider(Qt.Horizontal)
        self.bright_lbl = QLabel()
        self.bright_sld = QSlider(Qt.Horizontal)
        self.triangle_btn = QPushButton()

        # animation tab
        self.animate_btn = QPushButton()

        # 3d animations tab
        self.three_d_btn = QPushButton()

        # export tab
        self.export_width_txt = QLineEdit()
        self.export_height_txt = QLineEdit()
        self.export_aspect_chk = QCheckBox()
        self.export_frame_btn = QPushButton()
        self.export_anim_btn = QPushButton()

        # preview image/animation
        self.img_frame = QFrame()
        self.img_btn = QPushButton()

        # Initiate state variables
        self.path = split(abspath(__file__))[0]
        self.filepath = ""
        self.saved = True

        self.maxd = 720

        self.export_width = self.maxd
        self.export_height = self.maxd

        self.img = np.zeros((self.export_height, self.export_width, 3), dtype=np.uint8)
        self.preview = self.img.copy()

        self.vertices = None
        self.faces = None
        self.colours = None

        self.animations = []
        self.depths = None

        self.preview_thread = QThread()
        self.preview_worker = None
        self.numba_compiled = False
        self.imported = False

        # set everything up
        self.init_ui()
        self._init_numba_jit()

    def init_ui(self):
        """Initialize the user interface, set up events, etc."""

        # Basic winndow setup
        self.setWindowTitle("Trianimate - Timeline")
        self.setWindowIcon(QIcon(join(self.path, "icons", "trianimate.png")))
        self.setCentralWidget(self.central)
        self.central.setLayout(self.grid)

        # Menu bar
        self.setMenuBar(self.menu)
        self.file_menu = self.menu.addMenu("&File")

        # toolbar
        self.addToolBar(Qt.LeftToolBarArea, self.toolbar)

        self.import_btn.setIcon(QIcon(join(self.path, "icons", "import.png")))
        self.import_btn.setIconText("Import Image (Ctrl+I)")
        self.import_btn.setShortcut("Ctrl+I")

        self.toolbar.addAction(self.import_btn)

        # events for toolbar
        self.import_btn.triggered.connect(self.import_img)

        # main window layout is a grid
        self.grid.setSpacing(10)
        self.img_frame.setFrameStyle(1)

        # set up tabs for editing
        self.tabs.addTab(self.triangle_tab, "Triangulate")
        self.tabs.addTab(self.animate_tab, "Animate")
        self.tabs.addTab(self.three_d_tab, "3D")
        self.tabs.addTab(self.export_tab, "Export")

        # set up triangulation tab - load an image, set triangulation params, calculate
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
        self.bright_sld.setValue(25)
        self.triangle_tab.layout.addWidget(self.bright_lbl)
        self.triangle_tab.layout.addWidget(self.bright_sld)

        self.triangle_btn.setText("Triangulate")
        self.triangle_btn.setEnabled(False)  # can't triangulate without an image
        self.triangle_btn.clicked.connect(self.triangulate)
        self.triangle_tab.layout.addWidget(self.triangle_btn)
        self.triangle_tab.layout.addStretch(1)
        self.triangle_tab.layout.setSpacing(15)

        # animations tab - select groups of points and apply animations
        self.animate_tab.layout = QVBoxLayout(self.animate_tab)
        self.animate_tab.layout.addWidget(self.animate_btn)

        # 3D animations tab - move points into or out of frame to allow 3D effects
        self.three_d_tab.layout = QVBoxLayout(self.three_d_tab)
        self.three_d_tab.layout.addWidget(self.three_d_btn)

        # Export tab - export static frames, or video
        self.export_tab.layout = QVBoxLayout(self.export_tab)

        # Use CallableValidator to get around issues with blank fields not triggering
        # editingFinished when using other Validators, but also to calculate if the
        # aspect ratio will put one of the dimensions over size
        self.export_width_txt.setValidator(CallableValidator(self.validate_width))
        self.export_height_txt.setValidator(CallableValidator(self.validate_height))
        self.export_width_txt.setText(str(self.export_width))
        self.export_height_txt.setText(str(self.export_height))
        self.export_width_txt.textEdited.connect(self.export_width_changed)
        self.export_width_txt.editingFinished.connect(self.export_width_deselect)
        self.export_height_txt.textEdited.connect(self.export_height_changed)
        self.export_height_txt.editingFinished.connect(self.export_height_deselect)

        self.export_aspect_chk.setText("Maintain Original Aspect Ratio")
        self.export_aspect_chk.setChecked(True)
        self.export_aspect_chk.stateChanged.connect(self.export_aspect_changed)

        self.export_frame_btn.setText("Export Frame")
        self.export_frame_btn.setEnabled(False)
        self.export_frame_btn.clicked.connect(self.export_frame)

        self.export_anim_btn.setText("Export Animation")
        self.export_anim_btn.setEnabled(False)

        self.export_tab.layout.addWidget(self.export_width_txt)
        self.export_tab.layout.addWidget(self.export_height_txt)
        self.export_tab.layout.addWidget(self.export_aspect_chk)
        self.export_tab.layout.addWidget(self.export_frame_btn)
        self.export_tab.layout.addWidget(self.export_anim_btn)

        self.export_tab.layout.addStretch(1)
        self.export_tab.layout.setSpacing(15)

        # Preview window
        self.img_frame.layout = QVBoxLayout(self.img_frame)

        # set up the image, starts out black
        self.image = QImage(
            self.preview.data,
            self.export_width,
            self.export_height,
            QImage.Format_RGB888,
        )
        self.img_label.setPixmap(QPixmap.fromImage(self.image))
        self.img_frame.layout.addWidget(self.img_label)
        self.img_frame.layout.setAlignment(self.img_label, Qt.AlignCenter)

        # set up the grid, show the window
        self.grid.addWidget(self.tabs, 0, 0)
        self.grid.addWidget(self.img_frame, 0, 1)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 2)

        self.showMaximized()

    def _init_numba_jit(self):
        """Runs numba pre-compilation on those functions which use it."""
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        self.preview_worker = TriangulateWorker(img, 0.0, 0.0, 0.0, 0.0, self.maxd,)
        self.preview_worker.moveToThread(self.preview_thread)

        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.finished.connect(self._end_numba_jit)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)

        self.preview_thread.start()
        self.triangle_btn.setText("Compiling Scripts...")

    def _end_numba_jit(self, *args):
        """Callback for numba pre-compilation."""
        self.numba_compiled = True
        self.triangle_btn.setText("Triangulate")
        if self.imported:
            self.triangle_btn.setEnabled(True)

        self.preview_thread.quit()
        self.preview_worker.deleteLater()

    # TODO: when animations are implemented, ask if you want to save your work first
    def import_img(self, _):
        """Handler for import action - load an image into the tool.
        
        First, makes sure that anything currently being worked on is saved. If it isn't
        prompts the user to save before importing, lest they lose all their work. Opens
        a file dialog to select an image to load in. Preview is scaled to a maximum size
        but internally the full resolution is used."""

        # find a file
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
            # open the image using opencv, convert to RGB.
            # self.img is the full resolution version, self.preview is scaled down
            self.img: np.ndarray = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
            oh, ow = self.img.shape[:2]

            # set the size of the export if the size exceeds the maximum
            export_factor = min(4320 / oh, 7680 / ow)
            if export_factor < 1.0:
                self.export_height = int(oh * export_factor)
                self.export_width = int(ow * export_factor)
                self.export_height_txt.setText(str(self.export_height))
                self.export_width_txt.setText(str(self.export_width))
            else:
                self.export_height = oh
                self.export_width = ow
                self.export_height_txt.setText(str(self.export_height))
                self.export_width_txt.setText(str(self.export_width))

            # rescale the image down to self.maxd if necessary
            if oh > self.maxd or ow > self.maxd:
                h, w = (
                    int(oh * self.maxd / max(oh, ow)),
                    int(ow * self.maxd / max(oh, ow)),
                )
                self.preview = cv.resize(self.img, (w, h), interpolation=cv.INTER_AREA)
            else:
                h, w = oh, ow
                self.preview = self.img.copy()

            # display self.preview
            self.image = QImage(
                self.preview.data,
                w,
                h,
                int(self.preview.data.nbytes / h),
                QImage.Format_RGB888,
            )
            self.img_label.setPixmap(QPixmap.fromImage(self.image))

            # if this is our first image, we can now triangulate it!
            self.imported = True
            if self.numba_compiled:
                self.triangle_btn.setEnabled(True)
            # but our triangulation is gone, so we cannot export
            self.export_frame_btn.setEnabled(False)
            self.vertices = None
            self.faces = None
            self.colours = None

    def validate_width(self, txt: str) -> bool:
        """Would the resulting resolution be less than 8k?"""
        # allows you to type in anything in the range 1-7680
        w_rgx = (
            "^$|^[1-9][0-9]{0,2}$|^[1-6][0-9]{3}$|"
            "^7[0-5][0-9]{2}$|^76[0-7][0-9]$|^7680$"
        )
        if re.match(w_rgx, txt):
            if not txt:
                # blank text needs a special case - it is validated, but don't update
                return True
            elif self.export_aspect_chk.isChecked():
                # otherwise, calculate the resulting width: if it's unnder 4320, success
                width = int(txt)
                aspect = self.img.shape[0] / self.img.shape[1]
                height = round(width * aspect)
                if height > 4320 or height < 1:
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False

    def validate_height(self, txt: str) -> bool:
        """Would the resulting resolution be less than 8k?"""
        # allows you to type in anything in the range 1-4320
        h_rgx = (
            "^$|^[1-9][0-9]{0,2}$|^[1-3][0-9]{3}$|"
            "^4[0-2][0-9]{2}$|^43[0-1][0-9]$|^4320$"
        )
        if re.match(h_rgx, txt):
            if not txt:
                # blank text needs a special case - it is validated, but don't update
                return True
            elif self.export_aspect_chk.isChecked():
                # otherwise, calculate the resulting width: if it's unnder 7680, success
                height = int(txt)
                aspect = self.img.shape[1] / self.img.shape[0]
                width = round(height * aspect)
                if width > 7680 or width < 1:
                    return False
                else:
                    return True
            else:
                return True
        else:
            return False

    def export_width_changed(self, txt: str):
        """Update export width and height as appropriate when user changes values."""
        if txt:
            self.export_width = int(txt)
            if self.export_aspect_chk.isChecked():
                aspect = self.img.shape[0] / self.img.shape[1]
                self.export_height = round(self.export_width * aspect)
                self.export_height_txt.setText(str(self.export_height))

    def export_width_deselect(self):
        """Fill in width field if user deselects it and it was blank."""
        txt: str = self.export_width_txt.text()
        if not txt:
            self.export_width_txt.setText(str(self.export_width))

    def export_height_changed(self, txt: str):
        """Update export width and height as appropriate when user changes values."""
        if txt:
            self.export_height = int(txt)
            if self.export_aspect_chk.isChecked():
                aspect = self.img.shape[1] / self.img.shape[0]
                self.export_width = round(self.export_height * aspect)
                self.export_width_txt.setText(str(self.export_width))

    def export_height_deselect(self):
        """Fill in width field if user deselects it and it was blank."""
        txt: str = self.export_height_txt.text()
        if not txt:
            self.export_height_txt.setText(str(self.export_height))

    def export_aspect_changed(self, checked):
        """Figure out the new export dimensions when checkbox is re-checked."""
        if checked:
            # basically, pick the dimension that will give the largest result
            # but obviously don't go over the max export resolution of 7680x4320
            orig_aspect = self.img.shape[0] / self.img.shape[1]
            curr_aspect = self.export_height / self.export_width
            if curr_aspect > orig_aspect:
                th = self.export_height
                tw = th / orig_aspect
            else:
                tw = self.export_width
                th = tw * orig_aspect
            export_factor = min(7680 / tw, 4320 / th)
            if export_factor < 1.0:
                tw = int(export_factor * tw)
                th = int(export_factor * th)
            self.export_height = th
            self.export_width = tw
            self.export_height_txt.setText(str(th))
            self.export_width_txt.setText(str(tw))

    # TODO: call the triangulation asynchronously so the interface doesn't hang
    # or at least a status indicator to show it's working.
    def triangulate(self, _):
        """Handler for triangulate button - triangulate an image according to sliders.
        
        See triangulate.py for full information on triangulation parameters. Runs
        through standard triangulation procedure to change the image colouration,
        calculate the critical points of the image and run a triangulation on them, and
        finally calculate the colours of each of the triangles. Currently uses openCV to
        generate the preview image, but intend to use openGL with multisampling to do it
        instead, which should be significantly faster, especially for video generation
        which is yet to come."""

        # read off slider values, normalize appropriately, run triangulation procedure
        det_val = self.detail_sld.value() / 100.0
        thr_val = self.thresh_sld.value() / 100.0
        col_val = self.colour_sld.value() / 100.0
        brt_val = self.bright_sld.value() / 100.0

        self.triangle_btn.setDisabled(True)
        self.triangle_btn.setText("Working...")

        self.preview_worker = TriangulateWorker(
            self.img, det_val, thr_val, col_val, brt_val, self.maxd,
        )
        self.preview_thread = QThread()
        self.preview_worker.moveToThread(self.preview_thread)

        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.finished.connect(self.show_triangulate_preview)
        self.preview_thread.finished.connect(self.preview_thread.deleteLater)
        self.preview_thread.start()

    def show_triangulate_preview(
        self,
        preview: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray,
        colours: np.ndarray,
    ):
        """Collects the triangulation result and shows the preview."""
        self.preview = preview
        self.vertices = vertices
        self.faces = faces
        self.colours = colours
        h, w, _ = self.preview.shape

        # set the preview image
        self.image = QImage(
            self.preview.data,
            w,
            h,
            int(self.preview.data.nbytes / h),
            QImage.Format_RGB888,
        )
        self.img_label.setPixmap(QPixmap.fromImage(self.image))

        # Once we have run a triangulation, we can export!
        self.export_frame_btn.setEnabled(True)
        self.triangle_btn.setEnabled(True)
        self.triangle_btn.setText("Triangulate")

        # clean up thread
        self.preview_thread.quit()
        self.preview_worker.deleteLater()

    def export_frame(self, _):
        """Exports a static image of a triangulation."""
        # open file picker to save the image
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save to Image",
            "",
            "Image Files (*.jpg *.jpeg *.gif *.bmp *.png)",
            options=options,
        )
        if not filename:
            return

        # if vertices is 2D, it's a static image, if 3D, it's an animation
        # TODO: If it's an animation, provide a selector for the frame
        if len(self.vertices.shape) == 2:
            with TriangleShader().render_2d(
                self.export_width, self.export_height
            ) as render:
                result = render(self.vertices, self.faces, self.colours)
        elif len(self.vertices.shape) == 3:
            with TriangleShader().render_2d(
                self.export_width, self.export_height
            ) as render:
                result = render(self.vertices[0], self.faces[0], self.colours[0])

        # opencv likes to write BGR for whatever reason :/
        cv.imwrite(filename, cv.cvtColor(result, cv.COLOR_RGB2BGR))

    def save_work(self):
        """Write a project file for each image which can be reopened later."""
        # Not implemented
        self.saved = True

    def close_event(self, event: QCloseEvent):
        """Handler for the window being closed - prompts the user to save if needed."""
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
    """Set up an application and the main window and run it."""
    app = QApplication(sys.argv)
    palette = QDarkPalette()
    palette.set_app(app)

    t = MainWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
