"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import ast
from typing import Optional, List, Dict
import functools

import numpy as np
import napari
from napari.qt import thread_worker
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import widgets, magic_factory

from .predict import throttle_function, u, predict_output_chunks
from . import watershed as ws


def predict_output_chunks_widget(
        napari_viewer,
        input_volume_layer: napari.layers.Image,
        chunk_size: str = '(10, 256, 256)',
        margin: str = '(0, 0, 0)',
        state: Dict = None,
        ):
    if type(chunk_size) is str:
        chunk_size = ast.literal_eval(chunk_size)
    if type(margin) is str:
        margin = ast.literal_eval(margin)
    if state is None:
        state = {}
    viewer = napari_viewer
    layer = input_volume_layer
    ndim = len(chunk_size)
    slicing = viewer.dims.current_step[:-ndim]
    state['slicing'] = slicing
    input_volume = np.asarray(layer.data[slicing]).astype(np.float32)
    input_volume /= np.max(input_volume)
    if 'unet-output' in state:  # not our first rodeo
        state['unet-worker'].quit()  # in case we are running on another slice
        if state['self'].call_watershed is not None:
            state['self'].call_watershed.enabled = False
        output_volume = state['unet-output']
        output_volume[:] = 0
        layerlist = state['unet-output-layers']
        for layer in layerlist:
            layer.refresh()
    else:
        output_volume = np.zeros((5,) + input_volume.shape, dtype=np.float32)
        state['unet-output'] = output_volume
        scale = np.asarray(layer.scale)[-ndim:]
        translate = np.asarray(layer.translate[-ndim:])
        offsets = -0.5 * scale * np.eye(5, 3)  # offset affinities, not masks
        layerlist = viewer.add_image(
                output_volume,
                channel_axis=0,
                name=['z-aff', 'y-aff', 'x-aff', 'mask', 'centroids'],
                scale=scale,
                translate=list(translate + offsets),
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
        state['scale'] = scale
        state['translate'] = translate
    launch_prediction_worker = thread_worker(
            predict_output_chunks,
            connect={
                    'yielded': [ly.refresh for ly in layerlist],
                    'returned': state['self'].add_watershed_widgets,
                    }
            )
    worker = launch_prediction_worker(
            u, input_volume, chunk_size, output_volume, margin=margin
            )
    state['unet-worker'] = worker


def copy_data_to_sliced_layer(ws_result, *, layer, slice_):
    segmentation, _, _ = ws_result
    ndim_in = segmentation.ndim
    ndim_out = layer.data.ndim
    slicing = slice_[:ndim_out - ndim_in]
    layer.data[slicing] = segmentation


def segment_from_prediction_widget(
        napari_viewer: napari.viewer.Viewer,
        prediction: np.ndarray,
        visualize: napari.layers.Labels = None,
        copy_to: napari.layers.Labels = None,
        state: Optional[Dict] = None,
        ):
    viewer = napari_viewer
    output = np.pad(
            np.zeros(prediction.shape[1:], dtype=np.uint32),
            1,
            mode='constant',
            constant_values=0,
            )
    crop = tuple([slice(1, -1),] * output.ndim)
    if visualize is None:
        visualize = viewer.add_labels(
                output[crop],
                name='watershed',
                scale=state['scale'],
                translate=state['translate'],
                )
    else:
        visualize.data = output[crop]
    refresh_vis = throttle_function(visualize.refresh, every_n=10_000)
    if copy_to is not None:
        return_callback = functools.partial(
                copy_data_to_sliced_layer,
                layer=copy_to,
                slice_=state['slicing'],
                )
    else:
        def do_nothing(*args, **kwargs): pass
        return_callback = do_nothing

    launch_segmentation = thread_worker(
            ws.segment_output_image,
            connect={
                'yielded': refresh_vis,
                'returned': return_callback,
                },
            )
    worker = launch_segmentation(
            prediction,
            affinities_channels=(0, 1, 2),
            thresholding_channel=3,
            centroids_channel=4,
            out=output.ravel(),
            )
    viewer.dims.events.current_step.connect(lambda ev: worker.quit())


class UNetPredictWidget(widgets.Container):
    def __init__(self, napari_viewer):
        self._state = {'self': self}
        super().__init__()
        self.predict_widget = widgets.FunctionGui(
                predict_output_chunks_widget,
                param_options=dict(
                        napari_viewer={'visible': False},
                        chunk_size={'widget_type': 'LiteralEvalLineEdit'},
                        state={'visible': False},
                        )
                )
        self.append(self.predict_widget)
        self.predict_widget.state.bind(self._state)
        self.predict_widget.napari_viewer.bind(napari_viewer)
        self.viewer = napari_viewer
        self.call_watershed = None
    
    def add_watershed_widgets(self, volume):
        if self.call_watershed is None:
            self.call_watershed = widgets.FunctionGui(
                segment_from_prediction_widget,
                call_button='Run Watershed',
                param_options=dict(
                        prediction={'visible': False},
                        state={'visible': False},
                )
            )
            self.append(self.call_watershed)
        self.call_watershed.prediction.bind(volume)
        self.call_watershed.state.bind(self._state)
        self.call_watershed.enabled = True


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
