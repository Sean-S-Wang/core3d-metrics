
import numpy as np

import scipy.ndimage as ndimage
import time
import multiprocessing
from .metrics_util import getUnitWidth
from .threshold_geometry_metrics import run_threshold_geometry_metrics
from .relative_accuracy_metrics import run_relative_accuracy_metrics
from core3dmetrics.instancemetrics.instance_metrics import eval_instance_metrics


def eval_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, plot=None, testCONF=None,
                 verbose=True):

    # Evaluate threshold geometry metrics using refDTM as the testDTM to mitigate effects of terrain modeling
    # uncertainty
    result_geo, unitArea, _, _ = run_threshold_geometry_metrics(refDSM, refDTM, refMask, testDSM, refDTM, testMask, tform, ignoreMask,
                                                plot=plot, for_objectwise=True, testCONF=testCONF, verbose=verbose)

    # Run the relative accuracy metrics and report results.
    result_acc = run_relative_accuracy_metrics(refDSM, testDSM, refMask, testMask, ignoreMask,
                                               getUnitWidth(tform), for_objectwise=True, plot=plot)

    return result_geo, result_acc, unitArea


# Compute statistics on a list of values
def metric_stats(val):
    s = dict()
    s['values'] = val.tolist()
    s['mean'] = np.mean(val)
    s['stddev'] = np.std(val)
    s['pctl'] = {}
    s['pctl']['rank'] = [0, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 91, 92, 93, 94, 95, 96, 96, 98, 99, 100]
    try:
        s['pctl']['value'] = np.percentile(val, s['pctl']['rank']).tolist()
    except IndexError:
        s['pctl']['value'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    return s


def multiprocessing_fun(ref_ndx, loop_region, refMask, test_ndx, ref_ndx_orig,
                        ref_use_counter, testMask, test_use_counter, refDSM,
                        refDTM, testDSM, testDTM, tform,
                        ignoreMask, plot, verbose, max_area, min_area,
                        max_volume, min_volume):
    # Reference region under evaluation
    ref_objs = (ref_ndx == loop_region) & refMask

    # Find test regions overlapping with ref
    test_regions = np.unique(test_ndx[ref_ndx == loop_region])

    # Find test regions overlapping with ref
    ref_regions = np.unique(ref_ndx_orig[ref_ndx == loop_region])

    # Remove background region, '0'
    if np.any(test_regions == 0):
        test_regions = test_regions.tolist()
        test_regions.remove(0)
        test_regions = np.array(test_regions)

    if np.any(ref_regions == 0):
        ref_regions = ref_regions.tolist()
        ref_regions.remove(0)
        ref_regions = np.array(ref_regions)

    if len(test_regions) == 0:
        return None

    for refRegion in ref_regions:
        # Increment counter for ref region used
        ref_use_counter[refRegion - 1] = ref_use_counter[refRegion - 1] + 1

    # Make mask of overlapping test regions
    test_objs = np.zeros_like(testMask)
    for test_region in test_regions:
        test_objs = test_objs | (test_ndx == test_region)
        # Increment counter for test region used
        test_use_counter[test_region - 1] = test_use_counter[test_region - 1] + 1

    # TODO:  Not practical as implemented to enable plots. plots is forced to false.
    [result_geo, result_acc, unitArea] = eval_metrics(refDSM, refDTM, ref_objs, testDSM, testDTM, test_objs, tform,
                                                      ignoreMask, plot=plot, verbose=verbose)

    this_metric = dict()
    this_metric['ref_objects'] = ref_regions.tolist()
    this_metric['test_objects'] = test_regions.tolist()
    this_metric['threshold_geometry'] = result_geo
    this_metric['relative_accuracy'] = result_acc

    # Calculate min and max area/volume
    if this_metric['threshold_geometry']['area']['test_area'] > max_area or loop_region == 1:
        max_area = this_metric['threshold_geometry']['area']['test_area']
    if this_metric['threshold_geometry']['area']['test_area'] < min_area or loop_region == 1:
        min_area = this_metric['threshold_geometry']['area']['test_area']
    if this_metric['threshold_geometry']['volume']['test_volume'] > max_volume or loop_region == 1:
        max_volume = this_metric['threshold_geometry']['volume']['test_volume']
    if this_metric['threshold_geometry']['volume']['test_volume'] < min_volume or loop_region == 1:
        min_volume = this_metric['threshold_geometry']['volume']['test_volume']

    return this_metric, result_geo, result_acc, unitArea, ref_regions


def run_objectwise_metrics(refDSM, refDTM, refMask, testDSM, testDTM, testMask, tform, ignoreMask, merge_radius=2,
                           plot=None, verbose=True, geotiff_filename=None, use_multiprocessing=False):

    # parse plot input
    if plot is None:
        PLOTS_ENABLE = False
    else:
        PLOTS_ENABLE = True
        PLOTS_SAVE_PREFIX = "objectwise_"

    # Number of pixels to dilate reference object mask
    padding_pixels = np.round(merge_radius / getUnitWidth(tform))
    strel = ndimage.generate_binary_structure(2, 1)

    # Dilate reference object mask to combine closely spaced objects
    ref_ndx_orig = np.copy(refMask)
    #ref_ndx = ndimage.binary_dilation(ref_ndx_orig, structure=strel,  iterations=padding_pixels.astype(int))
    ref_ndx = ref_ndx_orig

    # Create index regions
    ref_ndx, num_ref_regions = ndimage.label(ref_ndx)
    ref_ndx_orig, num_ref_regions = ndimage.label(ref_ndx_orig)
    test_ndx, num_test_regions = ndimage.label(testMask)
    result = None
    return result, test_ndx, ref_ndx
