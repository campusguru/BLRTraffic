def pascal_voc_to_coco(x1y1x2y2):
    x1, y1, x2, y2 = x1y1x2y2
    return [x1, y1, x2 - x1, y2 - y1]

def get_best_ocr(preds, rec_conf, ocr_res, track_id):
    for info in preds:
    # Check if it is current track id
        if info['track_id'] == track_id:
          # Check if the ocr confidence is maximum or not
            if info['ocr_conf'] < rec_conf:
                info['ocr_conf'] = rec_conf
                info['ocr_txt'] = ocr_res
            else:
                rec_conf = info['ocr_conf']
                ocr_res = info['ocr_txt']
            break
    return preds, rec_conf, ocr_res

