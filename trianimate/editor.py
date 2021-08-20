"""Editing interface for trianimations.

Run this module from the command line, or python -m trianimate
"""
import re
import sys
from os.path import abspath, join, split
from trianimate.palette import QDarkPalette
from trianimate.animation import AnimationEditor
from trianimate.triangulate_utils import _points_in_polygon
from typing import Callable

import cv2 as cv
import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import (
    QCloseEvent,
    QIcon,
    QImage,
    QMouseEvent,
    QPixmap,
    QValidator,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
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

from trianimate.render import TriangleShader
from trianimate.triangulate import find_colours, find_faces, find_vertices, warp_colours


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


class LassoWorker(QObject):
    """Worker object for processing lasso selection done in numba in parallel."""

    finished = pyqtSignal(np.ndarray, int)

    def __init__(
        self, vertices: np.ndarray, polygon: np.ndarray, modifier: int,
    ):
        """Create a new worker, pass in all the relevant values to select points.
        
        Args:
            vertices: `np.ndarray` (dtype: float32, ndim: 2) vertices of triangulation
            polygon: `np.ndarray` (dtype: float32, ndim: 2) vertices defining polygon
        """
        super().__init__()
        self.vertices = vertices
        self.polygon = polygon
        self.modifier = modifier

    def run(self):
        """Runs the numba function to find the vertices in the selection."""
        in_poly = _points_in_polygon(self.vertices, self.polygon)

        # emit the finished signal, pass along the calculated values
        self.finished.emit(in_poly, self.modifier)


class ImageLabel(QLabel):

    lasso_finished = pyqtSignal(np.ndarray, int)
    lasso_progress = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.lasso_enabled = False
        self._lasso_started = False
        self._lasso_poly = np.zeros((10000, 2), dtype=np.int32)
        self._poly_size = 0

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        super().mousePressEvent(ev)
        if self.lasso_enabled:
            self._lasso_started = True

            # polygon defined to always loop back to the start
            self._lasso_poly[:] = np.array([ev.x(), ev.y()])
            self._poly_size = 1

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        super().mouseMoveEvent(ev)
        if self._lasso_started:
            self._lasso_poly[self._poly_size] = np.array([ev.x(), ev.y()])
            self._poly_size += 1
            if self._poly_size == self._lasso_poly.shape[0] - 1:
                self._lasso_started = False
                self._poly_size = 0
                modifiers = ev.modifiers()
                ctrl = int(bool(modifiers & Qt.ControlModifier))
                alt = int(bool(modifiers & Qt.AltModifier))
                self.lasso_finished.emit(self._lasso_poly, ctrl - alt)
            else:
                self.lasso_progress.emit(self._lasso_poly[: self._poly_size + 1])

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)
        if len(self._lasso_poly) > 2 and self._lasso_started:
            self._lasso_started = False
            modifiers = ev.modifiers()
            ctrl = int(bool(modifiers & Qt.ControlModifier))
            alt = int(bool(modifiers & Qt.AltModifier))
            self.lasso_finished.emit(self._lasso_poly[: self._poly_size + 1], ctrl - alt)
            self._poly_size = 0


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
        self.img_label = ImageLabel()

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
        self.animate_editor = None

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
        self.preview_overlay = self.img.copy()

        self.vertices = None
        self.faces = None
        self.colours = None

        self.animations = []
        self.depths = None

        self.preview_thread = QThread()
        self.lasso_thread = QThread()
        self.preview_worker = None
        self.lasso_worker = None
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
        self.tabs.currentChanged.connect(self.changed_tab)

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
        self.animate_btn.setText("ANIMATE!")
        self.animate_btn.clicked.connect(self.open_animation_editor)
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
            self.preview_overlay.data,
            self.export_width,
            self.export_height,
            QImage.Format_RGB888,
        )
        self.img_label.setPixmap(QPixmap.fromImage(self.image))
        self.img_frame.layout.addWidget(self.img_label)
        self.img_frame.layout.setAlignment(self.img_label, Qt.AlignCenter)

        self.img_label.lasso_progress.connect(self.draw_lasso)
        self.img_label.lasso_finished.connect(self.finalize_lasso)

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

        # start a new thread for the points_in_polygon function, smaller
        test_poly = np.array(
            [[10.0, 0.0], [10.0, 10.0], [20.0, 5.0], [10.0, 0.0]], dtype=np.float32
        )
        test_points = np.array([[0.0, 0.0], [11.0, 10.0], [11.0, 5.0]], dtype=np.float32)
        self.lasso_worker = LassoWorker(test_points, test_poly, 0)
        self.lasso_worker.moveToThread(self.lasso_thread)

        self.lasso_thread.started.connect(self.lasso_worker.run)
        self.lasso_worker.finished.connect(self._end_numba_lasso_jit)
        self.lasso_thread.finished.connect(self.lasso_thread.deleteLater)

        self.lasso_thread.start()

    def _end_numba_lasso_jit(self, *args):
        """Callback for numba pre-compilation."""
        self.lasso_thread.quit()
        self.lasso_worker.deleteLater()

    def changed_tab(self, new_idx):
        """Call back for changing tab - if animate or 3D tab, allow lasso selection."""
        if new_idx in [1, 2] and self.vertices is not None:
            self.img_label.lasso_enabled = True
        else:
            self.img_label.lasso_enabled = False

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
                self.preview_overlay = self.preview.copy()
            else:
                h, w = oh, ow
                self.preview = self.img.copy()
                self.preview_overlay = self.img.copy()

            # display self.preview
            self.image = QImage(
                self.preview_overlay.data,
                w,
                h,
                int(self.preview_overlay.data.nbytes / h),
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
        self.preview_overlay = preview.copy()
        self.vertices = vertices
        self.faces = faces
        self.colours = colours
        h, w, _ = self.preview.shape

        # set the preview image
        self.image = QImage(
            self.preview_overlay.data,
            w,
            h,
            int(self.preview_overlay.data.nbytes / h),
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

    def draw_lasso(self, polygon: np.ndarray):
        if polygon.shape[0] > 3:
            vdx = polygon.shape[0] - 4
            c = 255 * (vdx % 2)
            cv.line(
                self.preview_overlay,
                tuple(polygon[vdx]),
                tuple(polygon[vdx + 1]),
                (c, c, c),
            )
        self.image = QImage(
            self.preview_overlay.data,
            self.preview_overlay.shape[1],
            self.preview_overlay.shape[0],
            int(self.preview_overlay.data.nbytes / self.preview_overlay.shape[0]),
            QImage.Format_RGB888,
        )
        self.img_label.setPixmap(QPixmap.fromImage(self.image))

    def finalize_lasso(self, polygon: np.ndarray, modifier: int):
        h, w, d = self.preview_overlay.shape
        self.lasso_thread = QThread()
        self.lasso_worker = LassoWorker(
            self.vertices * np.array((w - 1, h - 1), dtype=np.float32),
            polygon.astype(np.float32),
            modifier,
        )
        self.lasso_worker.moveToThread(self.lasso_thread)

        self.lasso_thread.started.connect(self.lasso_worker.run)
        self.lasso_worker.finished.connect(self.show_selected_points)

        self.lasso_thread.start()

    def show_selected_points(self, selected: np.ndarray, modifier):
        self.preview_overlay = self.preview.copy()
        h, w, d = self.preview_overlay.shape
        if modifier == 0:
            self.selected = selected
        elif modifier == 1:
            self.selected = np.logical_or(self.selected, selected)
        else:
            self.selected = np.logical_and(self.selected, np.logical_not(selected))
        for vdx in range(self.selected.size):
            if self.selected[vdx]:
                x, y = tuple(
                    (
                        self.vertices[vdx] * np.array((w - 1, h - 1), dtype=np.float32)
                    ).astype(np.int32)
                )
                self.preview_overlay[
                    max(0, min(h - 1, y - 1)) : max(0, min(h - 1, y + 2)),
                    max(0, min(w - 1, x - 1)) : max(0, min(w - 1, x + 2)),
                ] = 255
                self.preview_overlay[
                    max(0, min(h - 1, y)) : max(0, min(h - 1, y + 1)),
                    max(0, min(w - 1, x)) : max(0, min(w - 1, x + 1)),
                ] = 0

        self.image = QImage(
            self.preview_overlay.data,
            self.preview_overlay.shape[1],
            self.preview_overlay.shape[0],
            int(self.preview_overlay.data.nbytes / self.preview_overlay.shape[0]),
            QImage.Format_RGB888,
        )
        self.img_label.setPixmap(QPixmap.fromImage(self.image))

        self.lasso_worker.deleteLater()
        self.lasso_thread.quit()
        self.lasso_thread.deleteLater()

    def open_animation_editor(self, *args):
        self.animate_editor = AnimationEditor()

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
