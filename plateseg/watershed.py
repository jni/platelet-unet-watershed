from heapq import heappop, heappush
import time

from scipy import ndimage as ndi
import numpy as np
import napari
from skimage import filters, morphology, feature
from skimage.morphology._util import _offsets_to_raveled_neighbors, _validate_connectivity


SLEEP_PER_PIX = 0
# ---------
# Watershed
# ---------

def watershed(image, marker_coords, mask,
                   compactness=0, affinities=True, scale=None, out=None):
    dim_weights = _prep_anisotropy(scale, marker_coords)
    prepped_data = _prep_data(
            image, marker_coords, mask, affinities, output=out
            )
    image_raveled, marker_coords, offsets, mask, output, strides = prepped_data
    yield from slow_raveled_watershed(image_raveled, marker_coords,
                                    offsets, mask, strides, compactness,
                                    output, affinities, dim_weights)
    if affinities:
        shape = image.shape[1:]
    else:
        shape = image.shape
    output = output.reshape(shape)
    return output



def _prep_data(image, marker_coords, mask=None, affinities=False, output=None):
    # INTENSITY VALUES
    if affinities:
        im_ndim = image.ndim - 1 # the first dim should represent affinities
        image_shape = image.shape[1:]
        image_strides = image[0].strides
        image_itemsize = image[0].itemsize
        raveled_image = np.zeros((image.shape[0], image[0].size), dtype=image.dtype)
        for i in range(image.shape[0]):
            raveled_image[i] = image[i].ravel()
    else:
        im_ndim = image.ndim
        image_shape = image.shape
        image_strides = image.strides
        image_itemsize = image.itemsize
        raveled_image = image.ravel()
    # NEIGHBORS
    selem, centre = _validate_connectivity(im_ndim, 1, None)
    if affinities:
        # array of shape (ndim * 2, 2) giving the indicies of neighbor affinities
        offsets = _indices_to_raveled_affinities(image_shape, selem, centre)
    else:
        offsets = _offsets_to_raveled_neighbors(image_shape, selem, centre)
    raveled_markers = np.apply_along_axis(_raveled_coordinate, 1, 
                                          marker_coords, **{'shape':image_shape})
    if mask is None:
        small_shape = [s - 2 for s in image_shape]
        mask = np.ones(small_shape, dtype=bool)
        mask = np.pad(mask, 1, constant_values=0)
        assert image_shape == mask.shape
    mask_raveled = mask.ravel()
    if output is None:
        output = np.zeros(mask_raveled.shape, dtype=raveled_image.dtype)
    labels = np.arange(len(raveled_markers)) + 1
    output[raveled_markers] = labels
    strides = np.array(image_strides, dtype=np.intp) // image_itemsize
    return raveled_image, raveled_markers, offsets, mask_raveled, output, strides


def _raveled_coordinate(coordinate, shape):
    # array[z, y, x] = array.ravel()[z * array.shape[1] * array.shape[2] + y * array.shape[2] + x]
    raveled_coord = 0
    for i in range(len(coordinate)):
        to_add = coordinate[i]
        for j in range(len(shape)):
            if j > i:
                to_add *= shape[j]
        raveled_coord += to_add
    return raveled_coord
    

def _indices_to_raveled_affinities(image_shape, selem, centre):
    im_offsets = _offsets_to_raveled_neighbors(image_shape, selem, centre)
    #im_offsets[-len(image_shape):] = 0
    affs = np.concatenate([np.arange(len(image_shape)), 
                           np.arange(len(image_shape))[::-1]])
    indices = np.stack([affs, im_offsets], axis=1)
    return indices


def _prep_anisotropy(scale, marker_coords):
    dim_weights = None
    if scale is not None:
        # validate that the scale is appropriate for coordinates
        assert len(scale) == marker_coords.shape[1] 
        dim_weights = list(scale) + list(scale)[::-1]
        dim_weights = list(map(abs, dim_weights))
    return dim_weights


def slow_raveled_watershed(image_raveled, marker_coords, offsets, mask,
                           strides, compactness, output, affinities,
                           dim_weights):
    '''
    Parameters
    ----------

    '''
    heap = Heap()
    n_neighbors = offsets.shape[0]
    age = 1
    compact = compactness > 0
    anisotropic = dim_weights is not None
    aff_offsets = offsets.copy()
    aff_offsets[:int(len(offsets) / 2), 1] = 0
    # add each seed to the stack
    for i in range(marker_coords.shape[0]):
        elem = Element()
        index = marker_coords[i]
        if affinities:
            elem.value = 0.
        else:
            elem.value = image_raveled[index]
        elem.source = index
        elem.index = index
        elem.age = 0
        heap.push(elem)
    # remove from stack until empty
    while not heap.is_empty:
        elem = heap.pop()
        if compact: # or anisotropic:
            if output[elem.index] and elem.index != elem.source:
                # non-marker, already visited, move on to next item
                continue
            output[elem.index] = output[elem.source]
            yield
            time.sleep(SLEEP_PER_PIX)
        for i in range(n_neighbors):
            # get the flattened address of the neighbor
            if affinities:
                # offsets are 2d (size, 2) with columns 0 and 1 corresponding to 
                # affinities and image neighbour indices respectively
                neighbor_index = offsets[i, 1] + elem.index
                # in this case the index used to find elem.value will be 2d tuple
                affinity_index = tuple(aff_offsets[i] + np.array([0, elem.index]))
            else:
                neighbor_index = offsets[i] + elem.index
            if not mask[neighbor_index]:
                # neighbor is not in mask, move on to next neighbor
                continue
            if output[neighbor_index]:
                # if there is a non-zero value in output, move on to next neighbor
                continue
            # if the neighbor is in the mask and not already labeled, add to queue
            age += 1
            new_elem = Element()
            if affinities:
                new_elem.value = image_raveled[affinity_index]
            else:
                new_elem.value = image_raveled[neighbor_index]
            if anisotropic:
                dim_weight = dim_weights[i]
                new_elem.value = new_elem.value * dim_weight
            if compact:
                # weight values according to distance from source voxel
                new_elem.value += (compactness *
                                       _euclid_dist(neighbor_index, elem.source,
                                                        strides))
                # weight the value according to scale 
                # (may need to introduce a scaling hyperparameter)
            else:
                output[neighbor_index] = output[elem.index]
            yield
            time.sleep(SLEEP_PER_PIX)
            new_elem.age = age
            new_elem.index = neighbor_index
            new_elem.source = elem.source
            heap.push(new_elem)
    return output


def _euclid_dist(pt0, pt1, strides):
    result, curr = 0, 0
    for i in range(strides.shape[0]):
        curr = (pt0 // strides[i]) - (pt1 // strides[i])
        result += curr * curr
        pt0 = pt0 % strides[i]
        pt1 = pt1 % strides[i]
    return np.sqrt(result)


class Element:
    def __init__(self, value=None, index=None, age=None, source=None):
        self._value = value
        self._index = index
        self._age = age
        self._source = source
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, v):
        self._value = v
    @property
    def index(self):
        return self._index
    @index.setter
    def index(self, i):
        self._index = i
    @property
    def age(self):
        return self._age
    @age.setter
    def age(self, a):
        self._age = a
    @property
    def source(self):
        return self._source
    @source.setter
    def source(self, s):
        self._source = s


class Heap:
    def __init__(self):
        self.items = {}
        self.values = []
        self.id = 0

    @property
    def is_empty(self):
        return len(self.items) == 0

    def push(self, item: Element):
        '''
        Add an element to the heap

        Parameters
        ----------
        item: Element
        '''
        # add the item to the new ID
        self.items[self.id] = item
        new_id = self.id
        heappush(self.values, (item.value, new_id))
        # new ID
        self.id += 1

    def pop(self):
        '''
        Remove the highest value element from the heap
        '''
        _, ID = heappop(self.values)
        elem = self.items.pop(ID)
        return elem

    def size(self):
        return len(self.items)


def segment_output_image(
        unet_output,
        affinities_channels,
        centroids_channel,
        thresholding_channel,
        scale=None,
        compactness=0.,
        absolute_thresh=None,
        out=None,
    ):
    '''
    Parameters
    ----------
    unet_output: np.ndarray or dask.array.core.Array
        Output from U-net inclusive of all channels. If there is an extra
        dim of size 1, this will be squeezed out. Therefore shape may be
        (1, c, z, y, x) or (c, z, y, x).
    affinities_channels: tuple of int
        Ints, in order (z, y, x) describe the channel indicies to which
        the z, y, and x short-range affinities belong.
    centroids_channel: int
        Describes the channel index for the channel that is used to find
        centroids.
    thresholding_channel: in
        Describes the channel index for the channel that is used to find
        the mask for watershed.
    '''
    unet_output = np.asarray(np.squeeze(unet_output))
    # Get the affinities image (a, z, y, x)
    affinities = unet_output[list(affinities_channels)]
    affinities /= np.max(affinities, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
    affinities = np.pad(
        affinities,
        ((0, 0), (1, 1), (1, 1), (1, 1)),
        mode='constant',
        constant_values=0,
    )
    # Get the image for finding centroids
    centroids_img = unet_output[centroids_channel]
    # find the centroids
    centroids = _get_centroids(centroids_img) + 1  # account for padding
    # Get the image for finding the mask
    masking_img = unet_output[thresholding_channel]
    # find the mask for use with watershed
    if absolute_thresh is None:
        mask = _get_mask(masking_img)
    else:
        mask = masking_img > absolute_thresh
    mask = np.pad(mask, 1, constant_values=0) # edge voxels must be 0
    mask, centroids = _remove_unwanted_objects(
        mask, centroids, min_area=10, max_area=10000
        )
    # affinity-based watershed
    segmentation = yield from watershed(
            affinities, centroids, mask,
            affinities=True, scale=scale,
            compactness=compactness, out=out
            )
    segmentation = segmentation[1:-1, 1:-1, 1:-1]
    seeds = centroids - 1
    return segmentation, seeds, mask


def _get_mask(img, sigma=2):
    thresh = filters.threshold_otsu(
        filters.gaussian(img, sigma=(sigma/4, sigma, sigma))
        )
    mask = img > thresh
    return mask


def _get_centroids(cent, gaussian=True):
    if gaussian:
        cent = filters.gaussian(cent, sigma=(0, 1, 1))
    centroids = feature.peak_local_max(cent, threshold_abs=.04) #* c_scale
    #centroids = blob_log(cent, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    return centroids


def _remove_unwanted_objects(mask, centroids, min_area=0, max_area=10000):
    labels, _ = ndi.label(mask)
    labels_no_small = morphology.remove_small_objects(labels, min_size=min_area)
    labels_large = morphology.remove_small_objects(labels_no_small, min_size=max_area)
    labels_goldilocks = labels_no_small ^ labels_large
    centroid_labels = labels_goldilocks[tuple(centroids.T)]
    new_centroids = centroids[centroid_labels > 0]
    new_mask = labels_goldilocks.astype(bool)
    return new_mask, new_centroids


if __name__ == '__main__':
    #import skimage.io as io
    #labs = io.imread('/Users/amcg0011/Data/pia-tracking/cang_training/191113_IVMTR26_I3_E3_t58_cang_training_labels.tif')
    #img = io.imread('/Users/amcg0011/Data/pia-tracking/cang_training/191113_IVMTR26_I3_E3_t58_cang_training_image.tif')
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max


    # Generate an initial image with two overlapping circles
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    image = np.logical_or(mask_circle1, mask_circle2)
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    distance = 1 - (distance / distance.max())
    out = watershed(distance, coords, image, 0, affinities=False)
    from train_io import get_affinities
    affs = get_affinities(out.copy())
    from skimage.filters import gaussian
    a_out = watershed(affs.copy(), coords, image, 0, affinities=True)
    affs_g = np.stack([gaussian(affs[i], sigma=1) for i in range(affs.shape[0])])
    ag_out = watershed(affs_g.copy(), coords, image, 0, affinities=True)
    import napari
    v = napari.view_labels(image, name='mask', blending='additive', visible=False)
    v.add_labels(out, name='watershed', blending='additive', visible=False)
    v.add_image(affs[0], name='y affinities', blending='additive', colormap='green', visible=False)
    v.add_image(affs[1], name='x affinities', blending='additive', colormap='magenta', visible=False)
    v.add_labels(a_out, name='affinity watershed', blending='additive', visible=False)
    v.add_image(affs_g[0], name='y affinities (Gaussian)', blending='additive', colormap='green', visible=False)
    v.add_image(affs_g[1], name='x affinities (Gaussian)', blending='additive', colormap='magenta', visible=False)
    v.add_labels(ag_out, name='affinity watershed (Gaussian)', blending='additive')
    v.add_points(coords, size=2)
    napari.run()


    #from helpers import get_files, get_paths

    #data_dir = '/Users/amcg0011/Data/pia-tracking/cang_training'
    #train_dir = os.path.join(data_dir, 'cang_training_data')
    #prediction_dir = os.path.join(data_dir, '210314_training_0')

    # Get the file paths
    #image_files, affinities_files = get_files(train_dir)
    #prediction_files = get_paths(prediction_dir)

    # Get some sample images
    #i0 = imread(image_files[0])
    #a0 = imread(affinities_files[0])
    #p0 = imread(prediction_files[0])


# ----------------
# This Didn't Work
# ----------------

#from elf.segmentation.workflows import simple_multicut_workflow
#from elf.segmentation.mutex_watershed import mutex_watershed
#from helpers import get_files, get_paths
# GOT SOME INTERESTING TRACEBACKS!!
#seg_a0 = simple_multicut_workflow(a0, True, 'blockwise-multicut') 
#seg_p0 = simple_multicut_workflow(p0, False, 'greedy-additive') 


# ----------------------------
# Playing With Raveled Indices
# ----------------------------

#OFFSETS = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
#STRIDES = [1, 1, 1]
# CAN'T GET AFFORGATO
#mw_a0 = mutex_watershed(a0, OFFSETS, STRIDES)
#offsets = _offsets_to_raveled_neighbors((10, 256, 256), selem, (1, 1, 1))
#image = np.random.random((100, 100))
#offsets = _offsets_to_raveled_neighbors((100, 100), np.ones((3, 3)), (1, 1))
#image_raveled = image.raveled()
#pixel_pos = [49, 49]
#selem = np.ones((3, 3))
#selem_indices = np.stack(np.nonzero(selem), axis=-1)
#offsets = selem_indices - (1,1)
#ravel_factors = image.shape[1:] + (1,)
#raveled_offsets = (offsets * ravel_factors).sum(axis=1)    
# similarly
#i = np.array([[12, 23, 31], [43, 39, 24]])
#ir = np.ravel_multi_index(i, (100, 100))
