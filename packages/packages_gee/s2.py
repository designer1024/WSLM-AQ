'''
Update: 2023-04-21
This module provides basic processing functions for Sentinel-2 2A data ("COPERNICUS/S2_SR" or "COPERNICUS/S2_SR_HARMONIZED"). 
The functionalities include:
- s2_rc_qa: cloud removal (based on QA band)

import ee
from ee.batch import Export, Task

def s2_rc_qa(startDate,endDate,cloudy_pixel_percentage,area=False,flag=True,cloud_shadow_percentage=100,s2_data=1):
    '''
    Get an ImageCollection filtered for cloud cover and cloud removal (based on QA band).
    :param area: Study area
    :param startDate: Start date, e.g., '2020-01-01'
    :param endDate: The day after the end date, e.g., '2021-01-01'
    :param flag: Sorting order of the collection by acquisition date
    :param cloudy_pixel_percentage: Cloud cover percentage, recommended to set to 50 (the lower the value, the fewer images retained)
    :param cloud_shadow_percentage: Cloud shadow percentage
    :param s2_data: Which type of S2 data to select
    :return: ImageCollection filtered for cloud cover and cloud removal
    '''
    if s2_data==0:
        s2=ee.ImageCollection("COPERNICUS/S2_SR")
    else:
        s2=ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    if area==False:
        collection=s2.filterDate(startDate, endDate).sort('system:time_start',flag)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
        # .filter(ee.Filter.lt('CLOUD_SHADOW_PERCENTAGE',cloud_shadow_percentage))
        # .filter(ee.Filter.lt('CLOUD_SHADOW_PERCENTAGE',0.1))
    else:
         collection=s2.filterDate(startDate, endDate).filterBounds(area).sort('system:time_start',flag)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_pixel_percentage))
        # .filter(ee.Filter.lt('CLOUD_SHADOW_PERCENTAGE',cloud_shadow_percentage))
        # .filter(ee.Filter.lt('CLOUD_SHADOW_PERCENTAGE',0.1))     
    #cloud removal (based on QA band)
    def rmCloud(image):
        qa=image.select('QA60')
        cloudBitMask=1<<10
        cirrusBitMask = 1 << 11
        mask1 = qa.bitwiseAnd(cloudBitMask).eq(0)
        mask2=qa.bitwiseAnd(cirrusBitMask).eq(0)
        mask=(mask1.add(mask2)).eq(2)
        return image.updateMask(mask).divide(10000).set('system:time_start',image.get('system:time_start'))#乘以比例因子scale
    collection=collection.map(rmCloud)
    return collection
