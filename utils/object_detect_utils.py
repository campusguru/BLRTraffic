from ultralytics import YOLO

model = YOLO(model='model/anpr.pt', task='detect')

def detect_plates(src):
    predictions = model.predict(src, verbose=False)
    results = []
    for prediction in predictions:
        for box in prediction.boxes:
            det_confidence = box.conf.item()
            if det_confidence < 0.6:
                continue
            coords = [int(position) for position in (box.xyxy.view(1, 4)).tolist()[0]]
            results.append({"coords": coords, "det_conf": det_confidence})
    return results

