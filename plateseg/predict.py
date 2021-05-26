# coding: utf-8
import itertools
import torch
from tqdm import tqdm
import numpy as np
import nd2_dask as nd2
import napari
from napari.qt import thread_worker
from skimage.exposure import rescale_intensity

import unet
import watershed as ws

#u_state_fn = '/data/platelets-deep/210525_141407_basic_z-1_y-1_x-1_m_centg/212505_142127_unet_210525_141407_basic_z-1_y-1_x-1_m_centg.pt'
u_state_fn = '/Users/jni/data/platelets-deep/212505_142127_unet_210525_141407_basic_z-1_y-1_x-1_m_centg.pt'
#data_fn = '/data/platelets/200519_IVMTR69_Inj4_dmso_exp3.nd2'
data_fn = '/Users/jni/Dropbox/share-files/200519_IVMTR69_Inj4_dmso_exp3.nd2'

u = unet.UNet(in_channels=1, out_channels=5)
u.load_state_dict(torch.load(u_state_fn, map_location=torch.device('cpu')))
layer_list = nd2.nd2_reader.nd2_reader(data_fn)

IGNORE_CUDA = False

if torch.cuda.is_available() and not IGNORE_CUDA:
    u.cuda()

t_idx = 114

source_vol = layer_list[2][0]
vol2predict = rescale_intensity(
        np.asarray(source_vol[t_idx])
        ).astype(np.float32)
prediction_output = np.zeros((5,) + vol2predict.shape, dtype=np.float32)

chunk_start_0 = (0, 6, 12, 18, 23)
chunk_start_1 = (0, 128, 256)
chunk_start_2 = (0, 128, 256)
crops_0 = [(0, 8), (2, 8), (2, 8), (2, 8), (3, 10)]
crops_1 = [(0, 192), (64, 192), (64, 256)]
crops_2 = [(0, 192), (64, 192), (64, 256)]
chunk_starts = list(itertools.product(chunk_start_0, chunk_start_1, chunk_start_2))
chunk_crops = list(itertools.product(crops_0, crops_1, crops_2))
size = (10, 256, 256)

def predict_output_chunks(
    unet, input_volume, chunk_size, chunk_starts, chunk_crops, output_volume
    ):
    u = unet
    for start, crop in tqdm(list(zip(chunk_starts, chunk_crops))):
        sl = tuple(slice(start0, start0+step) for start0, step
                in zip(start, chunk_size))
        tensor = torch.from_numpy(input_volume[sl][np.newaxis, np.newaxis])
        if torch.cuda.is_available() and not IGNORE_CUDA:
            tensor = tensor.cuda()
        predicted_array = u(tensor).cpu().detach().numpy()
        # add slice(None) for the 5 channels
        cr = (slice(None),) + tuple(slice(i, j) for i, j in crop)
        output_volume[(slice(None),) + sl][cr] = predicted_array[(0,) + cr]
        # print(f'output volume is prediction output', output_volume is prediction_output)
        yield
    return output_volume

viewer = napari.Viewer(ndisplay=3)
l0 = viewer._add_layer_from_data(*layer_list[0])[0]
l1 = viewer._add_layer_from_data(*layer_list[1])[0]
l2 = viewer._add_layer_from_data(*layer_list[2])[0]

offsets = -0.5 * np.asarray(l0.scale)[-3:] * np.eye(5, 3)
prediction_layers = viewer.add_image(
        prediction_output,
        channel_axis=0,
        name=['z-aff', 'y-aff', 'x-aff', 'mask', 'centroids'],
        scale=l0.scale[-3:],
        translate=list(np.asarray(l0.translate[-3:]) + offsets),
        colormap=['bop purple', 'bop orange', 'bop orange', 'gray', 'gray'],
        visible=[False, False, False, True, False],
        )
viewer.dims.set_point(0, t_idx)


def refresh_prediction_layers():
    for layer in prediction_layers:
        layer.refresh()


labels = np.pad(
    np.zeros(prediction_output.shape[1:], dtype=np.uint32),
    1,
    mode='constant',
    constant_values=0,
)
labels_layer = viewer.add_labels(
        labels[1:-1, 1:-1, 1:-1],
        name='watershed',
        scale=prediction_layers[-1].scale,
        translate=prediction_layers[-1].translate,
        )

counter = 0
def refresh_labels():
    # we throttle labels refreshes because they take a long time
    global counter
    counter += 1
    if counter % 10000 == 1:
        labels_layer.refresh()


# closure to connect to threadworker signal
def segment(prediction):
    yield from ws.segment_output_image(
        prediction,
        affinities_channels=(0, 1, 2),
        centroids_channel=4,
        thresholding_channel=3,
        out=labels.ravel()
    )

segment_worker = thread_worker(
    segment,
    connect={'yielded': refresh_labels}
)

prediction_worker = thread_worker(
    predict_output_chunks,
    connect={
        'yielded': refresh_prediction_layers,
        'returned': segment_worker
        },
)
prediction_worker(u, vol2predict, size, chunk_starts, chunk_crops, prediction_output)

napari.run()
