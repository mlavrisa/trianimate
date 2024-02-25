import contextlib
import logging
import threading

import numpy as np

from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QWidget

import qtpynodeeditor as nodeeditor
from qtpynodeeditor import (
    NodeData,
    NodeDataModel,
    NodeDataType,
    NodeValidationState,
    Port,
    PortType,
)
from qtpynodeeditor.type_converter import TypeConverter


def sizes_can_broadcast(arr1: np.ndarray, arr2: np.ndarray, exceptions, strict) -> bool:
    if arr1.size == 1 or arr2.size == 2:
        return True

    if strict and not arr1.ndim == arr2.ndim:
        return False

    dim = min(arr1.ndim, arr2.ndim) if strict else arr1.ndim

    return not any(
        arr1.shape[idx] == arr2.shape[idx]
        for idx in range(dim)
        if not (idx in exceptions or arr1.shape[idx] == 1 or arr2.shape[idx] == 1)
    )


class ArrayData(NodeData):
    """Data holding an ndarray"""

    data_type = NodeDataType("array", "Array")

    def __init__(self, array: np.ndarray) -> None:
        self._arr = array
        self._lock = threading.RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def array(self):
        return self._arr

    def number_as_text(self) -> str:
        "Number as a string"
        return "%g" % self._arr.ravel()[0]


class BinaryBroadcastOps(NodeDataModel):
    caption_visible = True
    num_ports = {
        "input": 3,
        "output": 1,
    }
    port_caption_visible = True
    data_type = ArrayData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._array1: ArrayData = None
        self._array2: ArrayData = None
        self._axes: ArrayData = None
        self._result: ArrayData = None
        self._validation_state = NodeValidationState.warning
        self._validation_message = "Uninitialized"

    @property
    def caption(self):
        return self.name

    def _check_inputs(self):
        array1_ok = self._array1 is not None and self._array1.data_type.id == "array"
        array2_ok = self._array2 is not None and self._array2.data_type.id == "array"
        axes_ok = self._axes is None or self._axes.data_type.id == "array"

        strict, dim_exc = self.get_broadcast_rules()

        if (
            not array1_ok
            or not array2_ok
            or not axes_ok
            or not sizes_can_broadcast(
                self._array1.array, self._array2.array, dim_exc, strict
            )
        ):
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            self._result = None
            self.data_updated.emit(0)
            return False

        passes, state, msg = self._additional_checks()
        if not passes:
            self._validation_state = state
            self._validation_message = msg
            self._result = None
            self.data_updated.emit(0)
            return False

        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        return True

    @contextlib.contextmanager
    def _compute_lock(self):
        if not self._array1 or not self._array2:
            raise RuntimeError("inputs unset")

        with self._array1.lock:
            with self._array2.lock:
                if self._axes is not None:
                    with self._axes.lock:
                        yield
                else:
                    yield

        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        """
        The output data as a result of this calculation

        Parameters
        ----------
        port : int

        Returns
        -------
        value : NodeData
        """
        return self._result

    def set_in_data(self, data: NodeData, port: Port):
        """
        New data at the input of the node

        Parameters
        ----------
        data : NodeData
        port_index : int
        """
        if port.index == 0:
            self._array1 = data
        elif port.index == 1:
            self._array2 = data
        elif port.index == 2:
            self._axes = data

        if self._check_inputs():
            with self._compute_lock():
                self.compute()

    def validation_state(self) -> NodeValidationState:
        return self._validation_state

    def validation_message(self) -> str:
        return self._validation_message

    def _additional_checks(self):
        return True, NodeValidationState.valid, ""

    def get_broadcast_rules(self):
        if self._axes is None or not self._axes.data_type.id == "array":
            return False, []
        return False, self._axes.array.ravel().astype(int).tolist()

    def compute(self):
        raise NotImplementedError("Must subclass and override this function.")


class DistancesToPoint(BinaryBroadcastOps):
    name = "Distances to a point"
    port_caption = {
        "input": {
            0: "Sources",
            1: "Reference",
            2: "Vector Axis",
        },
        "output": {0: "Distances"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        if self._axes is not None:
            dim = self._axes.array.ravel().astype(int)
        else:
            dim = 1
        res = np.linalg.norm(self._array1.array - self._array2.array, axis=dim)
        self._result = ArrayData(res)

    def _additional_checks(self):
        if not np.count_nonzero(np.array(self._array2.array.shape) > 1) == 1:
            return False, NodeValidationState.warning, "'Reference' must be a vector."
        if self._axes is None:
            return super()._additional_checks()
        dim = self._axes.array.ravel().astype(int)
        if self._axes.array.size > 1:
            return False, NodeValidationState.warning, "'Vector Axis' size must be 1."
        elif self._array2.array.shape[dim[0]] < 2:
            return (
                False,
                NodeValidationState.warning,
                "'Reference' is a vector in the wrong axis.",
            )
        return super()._additional_checks()


class AnglesToPoint(BinaryBroadcastOps):
    name = "Angles to a point"
    port_caption = {
        "input": {
            0: "Sources",
            1: "Reference",
            2: "N/A",
        },
        "output": {0: "Angles"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        vec = self._array1.array - self._array2.array
        res = np.arctan2(vec[:, 1], vec[:, 0])
        self._result = ArrayData(res)

    def _additional_checks(self):
        if self._array1.array.ndim == 2 and self._array1.array.shape[1] == 2:
            return True, NodeValidationState.valid, ""
        return (
            False,
            NodeValidationState.warning,
            "Angle only makes sense for 2D vectors. Input arrays "
            f"had dimensions {self._array1.array.shape} and {self._array2.array.shape}",
        )


class NormedToPoint(BinaryBroadcastOps):
    name = "Normalized vectors to a point"
    port_caption = {
        "input": {
            0: "Sources",
            1: "Reference",
            2: "Vector Axis",
        },
        "output": {0: "Normalized"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        ax = 1 if self._axes is None else self._axes.array.ravel().astype(int)[0]
        diff = self._array2.array - self._array1.array
        norms = np.linalg.norm(diff, axis=ax)
        res = diff / np.maximum(norms, 1e-9)
        self._result = ArrayData(res)


class MultiplyArrays(BinaryBroadcastOps):
    name = "Multiply"
    port_caption = {
        "input": {
            0: "Array",
            1: "Scalar/Array",
            2: "N/A",
        },
        "output": {0: "Product"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        res = self._array1.array * self._array2.array
        self._result = ArrayData(res)


class AddArrays(BinaryBroadcastOps):
    name = "Add"
    port_caption = {
        "input": {
            0: "Array",
            1: "Scalar/Array",
            2: "N/A",
        },
        "output": {0: "Sum"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        res = self._array1.array + self._array2.array
        self._result = ArrayData(res)


class ReductionOps(NodeDataModel):
    caption_visible = True
    num_ports = {
        "input": 2,
        "output": 1,
    }
    port_caption_visible = True
    data_type = ArrayData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._array: ArrayData = None
        self._axes: ArrayData = None
        self._result: ArrayData = None
        self._validation_state = NodeValidationState.warning
        self._validation_message = "Uninitialized"

    @property
    def caption(self):
        return self.name

    def _check_inputs(self):
        array_ok = self._array is not None and self._array.data_type.id == "array"
        axes_ok = self._axes is None or self._axes.data_type.id == "array"

        if not array_ok or not axes_ok:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            self._result = None
            self.data_updated.emit(0)
            return False

        passes, state, msg = self._additional_checks()
        if not passes:
            self._validation_state = state
            self._validation_message = msg
            self._result = None
            self.data_updated.emit(0)
            return False

        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        return True

    @contextlib.contextmanager
    def _compute_lock(self):
        if not self._array1 or not self._array2:
            raise RuntimeError("inputs unset")

        with self._array.lock:
            if self._axes is None:
                yield
            else:
                with self._axes.lock:
                    yield

        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        """
        The output data as a result of this calculation

        Parameters
        ----------
        port : int

        Returns
        -------
        value : NodeData
        """
        return self._result

    def set_in_data(self, data: NodeData, port: Port):
        """
        New data at the input of the node

        Parameters
        ----------
        data : NodeData
        port_index : int
        """
        if port.index == 0:
            self._array = data
        elif port.index == 1:
            self._axes = data

        if self._check_inputs():
            with self._compute_lock():
                self.compute()

    def validation_state(self) -> NodeValidationState:
        return self._validation_state

    def validation_message(self) -> str:
        return self._validation_message

    def _additional_checks(self):
        return True, NodeValidationState.valid, ""

    def compute(self):
        raise NotImplementedError("Must subclass and override this function.")


class UnitaryOps(NodeDataModel):
    caption_visible = True
    num_ports = {
        "input": 1,
        "output": 1,
    }
    port_caption_visible = True
    data_type = ArrayData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._array: ArrayData = None
        self._result: ArrayData = None
        self._validation_state = NodeValidationState.warning
        self._validation_message = "Uninitialized"

    @property
    def caption(self):
        return self.name

    def _check_inputs(self):
        array_ok = self._array is not None and self._array.data_type.id == "array"

        if not array_ok:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            self._result = None
            self.data_updated.emit(0)
            return False

        passes, state, msg = self._additional_checks()
        if not passes:
            self._validation_state = state
            self._validation_message = msg
            self._result = None
            self.data_updated.emit(0)
            return False

        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        return True

    @contextlib.contextmanager
    def _compute_lock(self):
        if not self._array1 or not self._array2:
            raise RuntimeError("inputs unset")

        with self._array.lock:
            yield

        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        """
        The output data as a result of this calculation

        Parameters
        ----------
        port : int

        Returns
        -------
        value : NodeData
        """
        return self._result

    def set_in_data(self, data: NodeData, port: Port):
        """
        New data at the input of the node

        Parameters
        ----------
        data : NodeData
        port_index : int
        """
        if port.index == 0:
            self._array = data

        if self._check_inputs():
            with self._compute_lock():
                self.compute()

    def validation_state(self) -> NodeValidationState:
        return self._validation_state

    def validation_message(self) -> str:
        return self._validation_message

    def _additional_checks(self):
        return True, NodeValidationState.valid, ""

    def compute(self):
        raise NotImplementedError("Must subclass and override this function.")


class SineOp(UnitaryOps):
    name = "Sine"
    port_caption = {
        "input": {
            0: "Array",
        },
        "output": {0: "Array"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        res = np.sin(self._array.array)
        self._result = ArrayData(res)


class ScalarInput(NodeDataModel):
    name = "Scalar"
    caption_visible = False
    num_ports = {
        PortType.input: 0,
        PortType.output: 1,
    }
    port_caption = {"output": {0: "Result"}}
    data_type = ArrayData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._number: ArrayData = None
        self._line_edit = QLineEdit()
        self._line_edit.setValidator(QDoubleValidator())
        self._line_edit.setMaximumSize(self._line_edit.sizeHint())
        self._line_edit.textChanged.connect(self.on_text_edited)
        self._line_edit.setText("0.0")

    @property
    def number(self):
        return self._number

    def save(self) -> dict:
        "Add to the JSON dictionary to save the state of the NumberSource"
        doc = super().save()
        if self._number:
            doc["number"] = self._number.array.ravel()[0]
        return doc

    def restore(self, state: dict):
        "Restore the number from the JSON dictionary"
        try:
            value = float(state["number"])
        except Exception:
            ...
        else:
            self._number = ArrayData(np.array([value]))
            self._line_edit.setText(self._number.number_as_text())

    def out_data(self, port: int) -> NodeData:
        """
        The data output from this node

        Parameters
        ----------
        port : int

        Returns
        -------
        value : NodeData
        """
        return self._number

    def embedded_widget(self) -> QWidget:
        "The number source has a line edit widget for the user to type in"
        return self._line_edit

    def on_text_edited(self, string: str):
        """
        Line edit text has changed

        Parameters
        ----------
        string : str
        """
        try:
            number = float(self._line_edit.text())
        except ValueError:
            self.data_invalidated.emit(0)
        else:
            self._number = ArrayData(np.array([number]))
            self.data_updated.emit(0)


class VerticesInput(NodeDataModel):
    name = "Vertices"
    caption_visible = False
    num_ports = {
        PortType.input: 0,
        PortType.output: 1,
    }
    port_caption = {"output": {0: "Vertices (x, y)"}}
    data_type = ArrayData.data_type
    vertices = None

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._array: ArrayData = VerticesInput.vertices

    @property
    def number(self):
        return self._array

    def save(self) -> dict:
        "Add to the JSON dictionary to save the state of the NumberSource"
        doc = super().save()
        if self._array:
            doc["array"] = self._array.array.tolist()
        return doc

    def restore(self, state: dict):
        "Restore the number from the JSON dictionary"
        try:
            value = state["array"]
        except Exception:
            ...
        else:
            self._array = ArrayData(np.array(value))

    def out_data(self, port: int) -> NodeData:
        """
        The data output from this node

        Parameters
        ----------
        port : int

        Returns
        -------
        value : NodeData
        """
        return self._array


class Concatenate(BinaryBroadcastOps):
    name = "Concatenate"
    port_caption = {
        "input": {
            0: "Array 1",
            1: "Array 2",
            2: "Concat Axis",
        },
        "output": {0: "Array"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        dim = 0 if self._axes is None else self._axes.array.ravel().astype(int)[0]
        res = np.concatenate((self._array1.array, self._array2.array), axis=dim)
        self._result = ArrayData(res)

    def _additional_checks(self):
        if self._axes is not None and self._axes.array.size > 1:
            return False, NodeValidationState.warning, "Can only concat on one axis."
        return super()._additional_checks()

    def get_broadcast_rules(self):
        if self._axes is not None:
            return True, self._axes.array.ravel().astype(int).tolist()
        return True, [0]


class ExpandDims(ReductionOps):
    name = "Concatenate"
    port_caption = {
        "input": {
            0: "Array",
            1: "New Axis",
        },
        "output": {0: "Array"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        dim = (
            -1 if self._axes is None else self._axes.array.ravel().astype(int).tolist()
        )
        res = np.expand_dims(self._array.array, axis=dim)
        self._result = ArrayData(res)


class InitOp(NodeDataModel):
    caption_visible = True
    num_ports = {
        "input": 1,
        "output": 1,
    }
    port_caption_visible = True
    data_type = ArrayData.data_type

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._dimensions: ArrayData = None
        self._result: ArrayData = None
        self._validation_state = NodeValidationState.warning
        self._validation_message = "Uninitialized"

    @property
    def caption(self):
        return self.name

    def _check_inputs(self):
        dims_ok = (
            self._dimensions is not None and self._dimensions.data_type.id == "array"
        )

        if not dims_ok:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect input"
            self._result = None
            self.data_updated.emit(0)
            return False

        passes, state, msg = self._additional_checks()
        if not passes:
            self._validation_state = state
            self._validation_message = msg
            self._result = None
            self.data_updated.emit(0)
            return False

        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        return True

    @contextlib.contextmanager
    def _compute_lock(self):
        if not self._dimensions:
            raise RuntimeError("inputs unset")

        with self._dimensions.lock:
            yield

        self.data_updated.emit(0)

    def out_data(self, port: int) -> NodeData:
        """
        The output data as a result of this calculation

        Parameters
        ----------
        port : int

        Returns
        -------
        value : NodeData
        """
        return self._result

    def set_in_data(self, data: NodeData, port: Port):
        """
        New data at the input of the node

        Parameters
        ----------
        data : NodeData
        port_index : int
        """
        if port.index == 0:
            self._dimensions = data

        if self._check_inputs():
            with self._compute_lock():
                self.compute()

    def validation_state(self) -> NodeValidationState:
        return self._validation_state

    def validation_message(self) -> str:
        return self._validation_message

    def _additional_checks(self):
        return True, NodeValidationState.valid, ""

    def compute(self):
        raise NotImplementedError("Must subclass and override this function.")


class UniformRandomInit(InitOp):
    name = "Uniform"
    port_caption = {
        "input": {
            0: "Dimensions",
        },
        "output": {0: "Array"},
    }

    def compute(self):
        self._validation_state = NodeValidationState.valid
        self._validation_message = ""
        res = np.random.rand(*self._dimensions.array.ravel().astype(int).tolist())
        self._result = ArrayData(res)


class NumberDisplayModel(NodeDataModel):
    name = "NumberDisplay"
    data_type = ArrayData.data_type
    caption_visible = False
    num_ports = {
        PortType.input: 1,
        PortType.output: 0,
    }
    port_caption = {"input": {0: "Number"}}

    def __init__(self, style=None, parent=None):
        super().__init__(style=style, parent=parent)
        self._number: ArrayData = None
        self._label = QLabel()
        self._label.setMargin(3)
        self._validation_state = NodeValidationState.warning
        self._validation_message = "Uninitialized"

    def set_in_data(self, data: ArrayData, port: Port):
        """
        New data propagated to the input

        Parameters
        ----------
        data : NodeData
        int : int
        """
        self._number = data
        number_ok = self._number is not None and self._number.data_type.id == "array"

        if number_ok:
            self._validation_state = NodeValidationState.valid
            self._validation_message = ""
            self._label.setText(self._number.number_as_text())
        else:
            self._validation_state = NodeValidationState.warning
            self._validation_message = "Missing or incorrect inputs"
            self._label.clear()

        self._label.adjustSize()

    def embedded_widget(self) -> QWidget:
        "The number display has a label"
        return self._label


def init_animation_editor(vertices, callback):
    registry = nodeeditor.DataModelRegistry()

    VerticesInput.vertices = vertices

    models = (
        UniformRandomInit,
        Concatenate,
        ScalarInput,
        VerticesInput,
        AddArrays,
        MultiplyArrays,
        DistancesToPoint,
        AnglesToPoint,
        NormedToPoint,
        NumberDisplayModel,
        SineOp,
    )
    for model in models:
        registry.register_model(model, category="Operations", style=None)

    scene = nodeeditor.FlowScene(registry=registry)

    view = nodeeditor.FlowView(scene)
    view.setWindowTitle("Basic Example")
    view.resize(800, 600)
    view.show()

    # inputs = []
    # node_add = scene.create_node(AdditionModel)
    # node_sub = scene.create_node(SubtractionModel)
    # node_mul = scene.create_node(MultiplicationModel)
    # node_div = scene.create_node(DivisionModel)
    # node_mod = scene.create_node(ModuloModel)

    # for node_operation in (node_add, node_sub, node_mul, node_div, node_mod):
    #     node_a = scene.create_node(NumberSourceDataModel)
    #     node_a.model.embedded_widget().setText("1.0")
    #     inputs.append(node_a)

    #     node_b = scene.create_node(NumberSourceDataModel)
    #     node_b.model.embedded_widget().setText("2.0")
    #     inputs.append(node_b)

    #     scene.create_connection(
    #         node_a[PortType.output][0],
    #         node_operation[PortType.input][0],
    #     )

    #     scene.create_connection(
    #         node_b[PortType.output][0],
    #         node_operation[PortType.input][1],
    #     )

    #     node_display = scene.create_node(NumberDisplayModel)

    #     scene.create_connection(
    #         node_operation[PortType.output][0],
    #         node_display[PortType.input][0],
    #     )

    # try:
    #     scene.auto_arrange(nodes=inputs, layout="bipartite")
    # except ImportError as e:
    #     print(e)

    return scene, view, []


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    app = QApplication([])
    scene, view, nodes = init_animation_editor(np.random.rand(15, 2), None)
    view.show()
    app.exec_()
