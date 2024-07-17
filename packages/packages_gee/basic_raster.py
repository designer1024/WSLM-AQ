'''
author：Boyi Li
Upgrate Date: 20240116
This module provides commonly used functions for processing GEE raster data, including:
- `get_cf`: Calculate water frequency from the water index collection.
- `ndwi_col`: Calculate NDWI for the collection.

import ee
from ee.batch import Export, Task

def get_cf(ndwicollection,threshold,bandname):
    '''
    Calculate cumulative frequency from the index collection.
    :param ndwicollection: Index image collection
    :param threshold: Index threshold
    :param bandname: Name of the band to set
    :return: Cumulative frequency image
    '''
    def getrecord(tempimg):
        image1=tempimg.where(tempimg.lt(10000),1)
        return image1
    recordcollection=ndwicollection.map(getrecord)
    def threshold_seg(img):
        waterimg = img.where(img.gte(threshold), 1).where(img.lt(threshold), 0)
        return waterimg
    watercollection=ndwicollection.map(threshold_seg)

    record_sum=recordcollection.sum()
    water_sum=watercollection.sum()
    wf=water_sum.multiply(1.0).divide(record_sum)
    wf=wf.select(0).rename(bandname)
    return wf

def ndwi_col(collection,nir_num,green_num):
    '''
    Calculate NDWI for the collection.
    :param collection: Image collection
    :return: NDWI image collection
    '''
    def ndwi(img):
        nir = img.select(nir_num)
        green = img.select(green_num)
        ndwiImg = green.subtract(nir).divide(green.add(nir))#.add(0.00000001)
        return ndwiImg.select(0).rename('ndwi')
    ndwis=collection.map(ndwi)
    return ndwis
