from __future__ import annotations

import napari
"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import ast

import numpy as np
from napari.qt import thread_worker
import napari
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import widgets, magic_factory
from .predict import u, predict_output_chunks


def predict_output_chunks_widget(
        napari_viewer,
        input_volume_layer: napari.layers.Image,
        chunk_size: str = '(10, 256, 256)',
        state: dict = None,
        ):
    if type(chunk_size) is str:
        chunk_size = ast.literal_eval(chunk_size)
    viewer = napari_viewer
    layer = input_volume_layer
    ndim = len(chunk_size)
    slicing = viewer.dims.current_step[:-ndim]
    input_volume = layer.data
    if 'unet-output' in state:
        state['unet-worker'].quit()  # in case we are running on another slice
        output_volume = state['unet-output']
        output_volume[:] = 0
        layerlist = state['unet-output-layers']
    else:
        output_volume = np.zeros((5,) + input_volume.shape, dtype=np.float32)
        state['unet-output'] = output_volume
        scale = np.asarray(layer.scale)[-ndim:]
        offsets = -0.5 * scale * np.eye(5, 3)  # offset affinities, not masks
        layerlist = viewer.add_image(
                output_volume,
                channel_axis=0,
                name=['z-aff', 'y-aff', 'x-aff', 'mask', 'centroids'],
                scale=scale,
                translate=list(np.asarray(layer.translate[-ndim:]) + offsets),
                colormap=[
                        'bop purple',
                        'bop orange',
                        'bop orange',
                        'bop blue',
                        'gray',
                        ],
                visible=[False] * 4 + [True],
                )
        state['unet-output-layers'] = layerlist
    prediction_worker = thread_worker(
            predict_output_chunks,
            connect={'yielded': [ly.refresh for ly in layerlist]}
            )
    prediction_worker(u, input_volume, chunk_size, output_volume)
    state['unet-worker'] = prediction_worker


class UNetPredictWidget(widgets.FunctionGui):
    def __init__(self, napari_viewer):
        self._state = {}
        super().__init__(
                predict_output_chunks_widget,
                param_options=dict(
                        napari_viewer={'visible': False},
                        chunk_size={'widget_type': 'LiteralEvalLineEdit'},
                        state={'visible': False},
                        )
                )
        self.state.bind(self._state)
        self.napari_viewer.bind(napari_viewer)


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


@magic_factory
def example_magic_widget(img_layer: napari.layers.Image):
    print(f"you have selected {img_layer}")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [UNetPredictWidget, ExampleQWidget]
