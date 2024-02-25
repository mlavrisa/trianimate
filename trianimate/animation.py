import json
import sys
from hashlib import blake2b
from typing import Any, Callable, Dict, Generator, List, Set, Tuple, Union

import numpy as np
from numpy import linalg
from PyQt5.QtCore import QPoint, QPointF, QRectF
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QGridLayout,
    QStyleOptionGraphicsItem,
    QWidget,
)

from trianimate.palette import PRIMARY, QDarkPalette


class ParamType:
    TIME = 1
    POS = 2
    DELTA = 4
    DIST = 8
    ANGLE = 16
    PARAM = 32

    def __init__(
        self,
        name: str,
        accept_types: int,
        parent: str = "",
        connections: List[Tuple[str, str]] = [],
    ):
        self.name = name
        if type(accept_types) is not int or accept_types > 63 or accept_types < 1:
            raise ValueError("Parameter accept types must be an integer [1...63]")
        self.type = accept_types
        self.parent = parent
        self.connections = connections
        self.index = -1

    def to_json(self) -> str:
        connection_json = (
            "["
            + ", ".join([f"""["{c[0]}", "{c[1]}"]""" for c in self.connections])
            + "]"
        )
        json = f"""            "{self.name}": {{
                "type": {self.type},
                "connections": {connection_json}
            }}"""
        return json

    @classmethod
    def from_json(
        cls, name: str, parent: str, json_dict: Dict[str, Union[int, List[List[str]]]]
    ) -> "ParamType":
        connections = [(c[0], c[1]) for c in json_dict["connections"]]
        return ParamType(name, json_dict["type"], parent, connections)

    def __and__(self, o: object) -> bool:
        if type(o) is ParamType:
            return bool(self.type & o.type)
        elif type(o) is int:
            return bool(self.type & o)
        else:
            raise TypeError("ParamType cannot be compared against type " + str(type(o)))

    def __eq__(self, o: object) -> bool:
        if type(o) is ParamType:
            return bool(self.type & o.type)
        elif type(o) is int:
            return bool(self.type & o)
        else:
            raise TypeError("ParamType cannot be compared against type " + str(type(o)))


class ParamList:
    def __init__(self, params: Tuple[ParamType], parent: str):
        self.keys = []
        for idx, inp in enumerate(params):
            if hasattr(self, inp.name):
                raise ValueError(f"Attribute {inp.name} already exists.")
            else:
                inp.parent = parent
                inp.index = idx
                setattr(
                    self, inp.name, inp,
                )
                self.keys.append(inp.name)

    def __getitem__(self, key) -> ParamType:
        return getattr(self, key)

    def __len__(self):
        return len(self.keys)

    def iter(self) -> Generator[ParamType, None, None]:
        for key in self.keys:
            yield self[key]


class ComponentConnector(QGraphicsItem):
    def __init__(self, start_x: int, start_y: int):
        super().__init__()
        self.setX(start_x)
        self.setY(start_y)

        self.start_x = start_x
        self.start_y = start_y

        self.end_x = 1.0
        self.end_y = 1.0
        self.new_end_x = 1.0
        self.new_end_y = 1.0
        self.dim = 1

        self.setAcceptHoverEvents(True)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ):
        col = QColor(255, 255, 255, 255)
        painter.setPen(col)
        path = QPainterPath(QPoint(0, 0))

        dim_x = self.dim * (1 if self.new_end_x > 0 else -1)
        dim_y = self.dim * (1 if self.new_end_y > 0 else -1)
        path.lineTo(QPointF(self.new_end_x - dim_x, 0))
        pos = path.currentPosition()
        path.cubicTo(
            pos + QPoint(int(dim_x * 0.707), 0),
            pos + QPoint(dim_x, int(dim_y * 0.293)),
            pos + QPoint(dim_x, dim_y),
        )
        path.lineTo(QPointF(self.new_end_x, self.new_end_y))
        painter.drawPath(path)
        self.end_x = self.new_end_x
        self.end_y = self.new_end_y

    def boundingRect(self) -> QRectF:
        return QRectF(
            min(0.0, self.end_x, self.new_end_x) - 1.0,
            min(0.0, self.end_y, self.new_end_y) - 1.0,
            max(0.0, self.end_x, self.new_end_x)
            - min(0.0, self.end_x, self.new_end_x)
            + 2.0,
            max(0.0, self.end_y, self.new_end_y)
            - min(0.0, self.end_y, self.new_end_y)
            + 2.0,
        )

    def mouse_has_moved(self, x, y):
        self.prepareGeometryChange()
        self.new_end_x = x - self.start_x
        self.new_end_y = y - self.start_y
        self.dim = int(min(abs(self.end_x), abs(self.end_y), 12))
        self.update(self.boundingRect())


class AnimatorComponent(QGraphicsItem):
    start_input_connection: Callable = None
    start_output_connection: Callable = None

    def __init__(
        self,
        name: str,
        inputs: Tuple[ParamType],
        outputs: Tuple[ParamType],
        hash_id: str,
        level: int = -1,
    ):
        super().__init__()
        self.inputs = ParamList(inputs, hash_id)
        self.outputs = ParamList(outputs, hash_id)
        self.name = name
        self.id = hash_id
        self.level = level

        self.padding = 10
        self.font_size = 12
        self.target_rad = 5

        self.setAcceptHoverEvents(True)

        self.title_font = QFont("arial", self.font_size + 2, 1)
        self.title_metric = QFontMetrics(self.title_font)
        self.w_title = self.title_metric.width(self.name)
        self.h_title = -self.padding if self.name == "" else self.title_metric.ascent()

        self.label_font = QFont("arial", self.font_size, 0)
        label_metric = QFontMetrics(self.label_font)
        self.w_inputs = [label_metric.width(lbl) for lbl in self.inputs.keys]
        w_outputs = [label_metric.width(lbl) for lbl in self.outputs.keys]
        self.h_lbl = label_metric.ascent()

        self.n_inputs = len(self.inputs)
        self.n_outputs = len(self.outputs)

        max_w_inputs = -self.padding if self.n_inputs == 0 else max(self.w_inputs)
        max_w_outputs = -self.padding if self.n_outputs == 0 else max(w_outputs)

        self.height = (
            (self.padding + self.h_lbl) * max(self.n_inputs, self.n_outputs)
            + self.h_title
            + self.padding * 2
        )

        self.left = -max(
            max_w_inputs + self.padding * 1.5, self.w_title * 0.5 + self.padding
        )
        self.right = max(
            max_w_outputs + self.padding * 1.5, self.w_title * 0.5 + self.padding
        )
        self.width = self.right - self.left

        self._input_targets = np.zeros((self.n_inputs, 3))
        self._output_targets = np.zeros((self.n_outputs, 3))
        self.__initiated = False
        self.input_hit_allowed = np.array([True] * self.n_inputs)
        self.output_hit_allowed = np.array([True] * self.n_outputs)

    def paint(
        self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget
    ):

        col = QColor(0, 0, 0, 0)
        painter.setPen(col)
        bkg = PRIMARY
        bkg.setAlpha(128)
        painter.setBrush(bkg)
        painter.drawRoundedRect(
            int(self.left),
            int(-self.height / 2 + 2),
            int(self.width),
            int(self.height),
            5,
            5,
        )

        col = QColor(255, 255, 255)
        col.setAlpha(0 if self.name == "" else 40)
        painter.setPen(col)
        painter.drawLine(
            int(self.left + 2),
            int(-self.height / 2 + self.padding * 1.5 + self.h_title + 2),
            int(self.right - 3),
            int(-self.height / 2 + self.padding * 1.5 + self.h_title + 2),
        )

        col = QColor(255, 255, 255, 255)
        painter.setPen(col)
        painter.setFont(self.title_font)
        painter.drawText(
            int(-self.w_title / 2),
            int(-self.height * 0.5 + self.h_title + self.padding),
            self.name,
        )

        painter.setFont(self.label_font)
        input_start = (
            self.h_lbl
            + self.padding
            - (self.h_lbl + self.padding) * self.n_inputs / 2
            + self.h_title / 2
        )
        output_start = (
            self.h_lbl
            + self.padding
            - (self.h_lbl + self.padding) * self.n_outputs / 2
            + self.h_title / 2
        )

        for kdx, name in enumerate(self.inputs.keys):
            painter.setPen(QColor(255, 255, 255, 255))
            y = int(kdx * (self.h_lbl + self.padding) + input_start)
            painter.drawText(
                int(-self.w_inputs[kdx] - self.padding * 0.5), y, name,
            )
            painter.setPen(QPen(QColor(0, 0, 0), 3.0))
            painter.setBrush(QColor(255, 255, 255))
            target_size = self.target_rad + self._input_targets[kdx, 2]
            target_x = self.left - target_size
            target_y = y - target_size - int(self.h_lbl * 0.35)
            painter.drawEllipse(
                QRectF(target_x, target_y, target_size * 2, target_size * 2)
            )
            if not self.__initiated:
                self._input_targets[kdx, :2] = np.array(
                    [self.left, y - int(self.h_lbl * 0.35)]
                )

        for kdx, name in enumerate(self.outputs.keys):
            painter.setPen(QColor(255, 255, 255, 255))
            y = int(kdx * (self.h_lbl + self.padding) + output_start)
            painter.drawText(
                int(self.padding * 0.5), y, name,
            )
            painter.setPen(QPen(QColor(0, 0, 0), 3.0))
            painter.setBrush(QColor(255, 255, 255))
            target_size = self.target_rad + self._output_targets[kdx, 2]
            target_x = self.right - target_size
            target_y = y - target_size - int(self.h_lbl * 0.35)
            painter.drawEllipse(
                QRectF(target_x, target_y, target_size * 2, target_size * 2)
            )
            if not self.__initiated:
                self._output_targets[kdx, :2] = np.array(
                    [self.right, y - int(self.h_lbl * 0.35)]
                )

        self.__initiated = True

    def boundingRect(self) -> QRectF:
        return QRectF(
            int(self.left - self.target_rad - 2),
            int(-self.height / 2 + 2 - self.target_rad - 2),
            int(self.width + self.target_rad * 2 + 4),
            int(self.height + self.target_rad * 2 + 4),
        )

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        pos = np.array([event.pos().x(), event.pos().y()])
        inp_hit = np.logical_and(
            linalg.norm(self._input_targets[:, :2] - pos, axis=1) <= self.target_rad,
            self.input_hit_allowed,
        ).astype("float")
        outp_hit = np.logical_and(
            linalg.norm(self._output_targets[:, :2] - pos, axis=1) <= self.target_rad,
            self.output_hit_allowed,
        ).astype("float")
        inp_change = np.any(self._input_targets[:, 2] != inp_hit)
        outp_change = np.any(self._output_targets[:, 2] != outp_hit)
        if inp_change or outp_change:
            self.update(self.boundingRect())
        self._input_targets[:, 2] = inp_hit
        self._output_targets[:, 2] = outp_hit

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = np.array([event.pos().x(), event.pos().y()])
        inp_hit = np.logical_and(
            linalg.norm(self._input_targets[:, :2] - pos, axis=1) <= self.target_rad,
            self.input_hit_allowed,
        ).astype("float")
        outp_hit = np.logical_and(
            linalg.norm(self._output_targets[:, :2] - pos, axis=1) <= self.target_rad,
            self.output_hit_allowed,
        ).astype("float")
        inp_change = np.any(inp_hit)
        outp_change = np.any(outp_hit)
        if inp_change:
            kdx = np.nonzero(inp_hit)[0][0]
            key = self.inputs.keys[kdx]
            self.start_input_connection(
                self.id,
                kdx,
                key,
                *self._input_targets[kdx, :2],
                self.pos().x(),
                self.pos().y(),
            )
        elif outp_change:
            kdx = np.nonzero(outp_hit)[0][0]
            key = self.outputs.keys[kdx]
            self.start_output_connection(
                self.id,
                kdx,
                key,
                *self._output_targets[kdx, :2],
                self.pos().x(),
                self.pos().y(),
            )

    def get_input_targets(
        self, param_type: int
    ) -> List[Tuple[str, bool, float, float]]:
        x, y = self.pos().x(), self.pos().y()
        result = []
        for kdx, name in enumerate(self.inputs.keys):
            if self.inputs[name].type & param_type and not self.inputs[name].connection:
                target_x = self._input_targets[kdx][0]
                target_y = self._input_targets[kdx][1]
                result.append((name, True, target_x + x, target_y + y))
            else:
                result.append((name, False, 0.0, 0.0))

        return result

    def get_output_targets(
        self, param_type: int
    ) -> List[Tuple[str, bool, float, float]]:
        x, y = self.pos().x(), self.pos().y()
        result = []
        for kdx, name in enumerate(self.outputs.keys):
            if self.outputs[name].type & param_type:
                target_x = self._output_targets[kdx][0]
                target_y = self._output_targets[kdx][1]
                result.append((name, True, target_x + x, target_y + y))
            else:
                result.append((name, False, 0.0, 0.0))

        return result

    @classmethod
    def from_json(cls, hash_id: str, json_dict: Dict[str, Any]):
        """Called when an AnimatorComponent is loaded from a save file."""
        raise NotImplementedError("Subclasses must override the from_json method.")

    def to_json(self) -> str:
        """Called when an AnimatorNetwork is being saved."""
        raise NotImplementedError("Subclasses must override the to_json method.")

    def get_result(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must override the get_result method.")


class AnimatorNetwork(QGraphicsScene):
    def __init__(self, network_json: str = ""):
        super().__init__()
        self.members: Dict[str, AnimatorComponent] = {}
        self.outputs_to: Dict[str, str] = {}
        self.inputs_from: Dict[str, str] = {}
        self.layers: List[List[str]] = []
        self._hash = blake2b(digest_size=5)

        AnimatorComponent.start_input_connection = self.start_input_connection
        AnimatorComponent.start_output_connection = self.start_output_connection

        if network_json:
            self.load_network(network_json)
        else:
            self.init_simple()

        self.make_layout(True)

        self.dragging = None

    def init_simple(self):
        self.layers = [[] for _ in range(2)]
        self._hash.update("BaseInputComponent#pos".encode("utf-8"))
        pos_str = "pos"
        pos_comp = BaseInputComponent(ParamType("pos", ParamType.POS), pos_str)
        self.members[pos_str] = pos_comp
        self.layers[0].append(pos_str)
        self._hash.update("BaseInputComponent#time".encode("utf-8"))
        time_str = "time"
        time_comp = BaseInputComponent(ParamType("time", ParamType.TIME), time_str)
        self.members[time_str] = time_comp
        self.layers[0].append(time_str)
        self._hash.update("OutputDeltaComponent#delta".encode("utf-8"))
        delta_str = "delta"
        delta_comp = OutputDeltaComponent()
        self.members[delta_str] = delta_comp
        self.layers[1].append(delta_str)
        print(self.save_network())

    def start_input_connection(
        self,
        id: str,
        key_idx: int,
        conn_name: str,
        internal_x: int,
        internal_y: int,
        pos_x: int,
        pos_y: int,
    ):
        test = ComponentConnector(internal_x + pos_x, internal_y + pos_y)
        self.addItem(test)
        self.dragging = test

    def start_output_connection(
        self,
        id: str,
        key_idx: int,
        conn_name: str,
        internal_x: int,
        internal_y: int,
        pos_x: int,
        pos_y: int,
    ):
        test = ComponentConnector(internal_x + pos_x, internal_y + pos_y)
        self.addItem(test)
        self.dragging = test

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)
        if not self.dragging is None:
            self.dragging.mouse_has_moved(event.scenePos().x(), event.scenePos().y())

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseReleaseEvent(event)
        self.removeItem(self.dragging)
        del self.dragging
        self.dragging = None

    def sort_levels(self):
        # first walk up the graph, determine the max layer
        # from that, instantiate the layers list
        # anything without connected inputs is automatically layer 1
        # otherwise, number by going down the list, making a note of complete nodes
        # the final output is always one greater than the rest of the components
        # this is true even when it's not connected
        pass

    def make_layout(self, init: bool = False):
        total_width = 0.0
        heights = []
        for ldx, clist in enumerate(self.layers):
            layer_height = 0.0
            y_pos = []
            min_left = 0.0
            max_right = 0.0
            for cdx, comp_id in enumerate(clist):
                comp = self.members[comp_id]
                min_left = min(min_left, comp.left)
                max_right = max(max_right, comp.right)
                y_pos.append(layer_height)
                layer_height += comp.height + 10.0
            layer_height -= 10.0
            total_width -= min_left if ldx else 0.0
            for cdx, comp_id in enumerate(clist):
                comp = self.members[comp_id]
                if init:
                    self.addItem(comp)
                comp.setX(-0.5 * comp.width + total_width)
                comp.setY(y_pos[cdx] - 0.5 * layer_height)
            total_width += max_right + 50.0
            heights.append(layer_height)

    def load_network(self, network_json: str):
        loaded = json.loads(network_json)
        for ldx, layer_dict in enumerate(loaded):
            for key in layer_dict.keys():
                c_type, hash_id = key.split("#")
                cls = globals()[c_type]
                test = cls.from_json(hash_id, layer_dict[key])
                # add to layer ldx, update the hash generator
                print(test)

    def save_network(self):
        json_layers = []
        for ldx, layer_list in enumerate(self.layers):
            member_json = ",\n".join([self.members[c].to_json() for c in layer_list])
            layer_json = f"""    {{
{member_json}
    }}"""
            json_layers.append(layer_json)
        json_join = ",\n".join(json_layers)
        saved_json = f"""[
{json_join}
]"""
        self.load_network(saved_json)
        return saved_json

    def find_all_downstream(self, start: str) -> Set[str]:
        """Used to check which connnections' outputs cannot be connected to."""
        downstream = set()
        new_members = set()
        new_members.update(self.outputs_to[start])
        while True:
            temp = set()
            if len(new_members):
                for member in new_members:
                    temp.update(self.outputs_to[member])
                downstream.update(new_members)
                new_members = temp
            else:
                break
        return downstream

    def find_all_upstream(self, start: str) -> Set[str]:
        """Used to check which connnections' inputs cannot be connected to."""
        upstream = set()
        new_members = {self.inputs_from[start]}
        while True:
            temp = set()
            if len(new_members):
                for member in new_members:
                    temp.update(self.inputs_from[member])
                upstream.update(new_members)
                new_members = temp
            else:
                break
        return upstream


class BaseInputComponent(AnimatorComponent):
    def __init__(
        self, param: ParamType, hash_id: str,
    ):
        super().__init__("", (), (param,), hash_id, 0)

    def set_param(self, value: Union[np.ndarray, float]):
        if not (type(value) is np.ndarray or type(value) is float):
            raise ValueError("value must be of type np.ndarray, or float")
        self.param = value

    def get_result(self) -> Union[np.ndarray, float]:
        if self.param is None:
            raise RuntimeError("set_param must be called before get_result")
        return self.param

    def to_json(self) -> str:
        name = self.outputs.keys[0]
        json_output = f"""        "BaseInputComponent#{self.id}": {{
{self.outputs[name].to_json()}
        }}"""
        return json_output

    @classmethod
    def from_json(cls, hash_id: str, json_dict: Dict[str, Any]) -> "BaseInputComponent":
        param_name = next(iter(json_dict.keys()))
        param = ParamType.from_json(param_name, hash_id, json_dict[param_name])
        return BaseInputComponent(param, hash_id)


class OutputDeltaComponent(AnimatorComponent):
    def __init__(self, connection: Union[None, Tuple[str, str]] = None):
        if connection is None:
            param = ParamType("delta", ParamType.DELTA, "delta")
        elif (
            type(connection) is not tuple
            or not len(connection) == 2
            or type(connection[0]) is not str
            or type(connection[1]) is not str
        ):
            raise ValueError("OutputDeltaComponent can only have one input connection.")
        else:
            param = ParamType("delta", ParamType.DELTA, "delta", [connection])
        super().__init__("", (param,), (), "delta")

    @classmethod
    def from_json(cls, hash_id: str, json_dict: Dict[str, Any]):
        param = ParamType.from_json("delta", "delta", json_dict["delta"])
        if param.connections:
            return OutputDeltaComponent(param.connections[0])
        else:
            return OutputDeltaComponent()

    def to_json(self) -> str:
        json_output = f"""        "OutputDeltaComponent#{self.id}": {{
{self.inputs["delta"].to_json()}
        }}"""
        return json_output


class LissajousComponent(AnimatorComponent):
    def __init__(
        self,
        name: str,
        inputs: Tuple[ParamType],
        outputs: Tuple[ParamType],
        sin_amp: float = 10.0,
        sin_period: float = 2.0,
        sin_phase: float = 0.0,
        cos_amp: float = 0.0,
        cos_period: float = 2.0,
        cos_phase: float = 0.0,
    ):
        super().__init__(name, inputs, outputs, ())
        if (
            not hasattr(self.outputs, "delta")
            or not self.outputs.delta.type & ParamType.DELTA
            or not len(outputs) == 1
        ):
            raise ValueError(
                'LissajousComponent must have output "delta" of type DELTA'
            )
        if (
            not hasattr(self.inputs, "time")
            or not self.inputs.time.type & ParamType.TIME
            or not len(inputs) == 1
        ):
            raise ValueError(
                'LissajousComponent must have output "delta" of type DELTA'
            )
        self.sin_amp = sin_amp
        self.sin_period = max(sin_period, 0.01)
        self.sin_phase = sin_phase
        self.cos_amp = cos_amp
        self.cos_period = max(cos_period, 0.01)
        self.cos_phase = cos_phase

    # TODO: 3D not supported, is this important?
    def get_result(self, time: np.ndarray):
        delta = np.zeros(time.shape[0], 2)
        delta[:, 0] = self.cos_amp * np.cos(
            6.28318531 * time / self.cos_period + self.cos_phase
        )
        delta[:, 1] = self.sin_amp * np.sin(
            6.28318531 * time / self.sin_period + self.sin_phase
        )
        output = {"delta": delta}
        return output

    def from_json(self, json):
        pass

    def to_json(self):
        pass


class AnimationEditor(QWidget):
    def __init__(self):
        super().__init__()

        self.grid = QGridLayout()
        self.settings = QWidget()
        self.detail_viewer = QGraphicsView()
        self.detail_scene = QGraphicsScene()
        self.node_viewer = QGraphicsView()
        self.node_scene: AnimatorNetwork = None

        self.init_ui()

    def init_ui(self):

        self.grid.setSpacing(10)
        # self.node_viewer.setFrameStyle(1)

        self.grid.addWidget(self.settings, 0, 0)
        self.grid.addWidget(self.node_viewer, 0, 1)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 2)

        # self.t_comp = BaseInputComponent(ParamType("time", ParamType.TIME))
        # self.t_comp.set_param(np.random.rand(10, 2))
        # self.t_comp.setPos(0.0, -20.0)
        # self.node_scene.addItem(self.t_comp)
        # self.x_comp = BaseInputComponent(ParamType("pos", ParamType.POS))
        # self.x_comp.set_param(np.random.rand(10, 2))
        # self.x_comp.setPos(0.0, 20.0)
        # self.node_scene.addItem(self.x_comp)
        self.node_scene = AnimatorNetwork(self.node_scene)
        self.node_viewer.setScene(self.node_scene)

        # inputs = (ParamType("time", ParamType.TIME), ParamType("space", ParamType.POS))
        # outputs = (ParamType("t_warp", ParamType.TIME),)
        # self.test = AnimatorComponent("likeaboss", inputs=inputs, outputs=outputs)
        # self.test.setPos(140.0, 0.0)
        # self.node_scene.addItem(self.test)

        # self.final_comp = BaseOuputComponent(ParamType("delta", ParamType.DELTA))
        # self.final_comp.setPos(280.0, 0.0)
        # self.node_scene.addItem(self.final_comp)

        self.setMinimumSize(800, 600)
        self.setWindowTitle("Animation Station")
        self.setLayout(self.grid)
        self.show()


def main():
    """Set up an application and the main window and run it."""
    app = QApplication(sys.argv)
    palette = QDarkPalette()
    palette.set_app(app)

    t = AnimationEditor()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
