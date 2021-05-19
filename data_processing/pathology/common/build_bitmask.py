import xml.etree.ElementTree as et
import numpy as np
import cv2

def xml2mask(xml_fn, shape):
    """ Convert a sparse XML annotation file (polygons) to a dense bitmask of shape <shape> """

    ret = dict()
    board_pos = None
    board_neg = None
    # Annotations >> 
    e = et.parse(xml_fn).getroot()
    e = e.findall('Annotation')
    for ann in e:
        if ann.get('Name') != 'Tumor':
                continue
        board_pos = np.zeros(shape, dtype=np.uint8)
        board_neg = np.zeros(shape, dtype=np.uint8)
        regions = ann.findall('Regions')
        assert(len(regions) == 1)
        rs = regions[0].findall('Region')
        plistlist = list()
        nlistlist = list()
        #print('rs:', len(rs))
        for i, r in enumerate(rs):
            ylist = list()
            xlist = list()
            plist, nlist = list(), list()
            negative_flag = int(r.get('NegativeROA'))
            assert negative_flag == 0 or negative_flag == 1
            negative_flag = bool(negative_flag)
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            vs.append(vs[0]) # last dot should be linked to the first dot
            for v in vs:
                y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
                ylist.append(y)
                xlist.append(x)
                if negative_flag:
                    nlist.append((x, y))
                else:
                    plist.append((x, y))
            if plist:
                plistlist.append(plist)
            else:
                nlistlist.append(nlist)
        for plist in plistlist:
            thiscontour = np.array(plist, dtype=np.int32)
            board_pos = cv2.drawContours(board_pos, [np.array(plist, dtype=np.int32)], -1, [255, 0, 0], -1)
        for nlist in nlistlist:
            board_neg = cv2.drawContours(board_neg, [np.array(nlist, dtype=np.int32)], -1, [255, 0, 0], -1)
        ret = (board_pos>0) * (board_neg==0)
    return ret


def xml2roi(xml_fn):
    """ Read the rectangle ROI of a halo XML annotation file """
    
    ylist = list()
    xlist = list()
        
    e = et.parse(xml_fn).getroot()
    for ann in e.findall('Annotation'):

        regions = ann.findall('Regions')[0]
        if not regions[0].get('Type')=='Rectangle': continue
        
        for i, r in enumerate(regions):
            vs = r.findall('Vertices')[0]
            vs = vs.findall('V')
            for v in vs:
                y, x = int(v.get('Y').split('.')[0]), int(v.get('X').split('.')[0])
                ylist.append(y)
                xlist.append(x)
    return xlist, ylist



