from ultralytics import YOLO

model = YOLO(model='anpr_weights/anpr.pt', task='detect')


def get_predictions(input_data):
    # print("input_data", input_data)
    return model.predict(input_data, device='mps', verbose=True)


def detect_plates(src):
    print("Get Predictions...")
    predictions = get_predictions(src)
    print(f"Obtained {predictions}")
    results = []
    for prediction in predictions:
        for box in prediction.boxes:
            det_confidence = box.conf.item()
            if det_confidence < 0.6:
                continue
            coords = [int(position) for position in (box.xyxy.view(1, 4)).tolist()[0]]
            results.append({"coords": coords, "det_conf": det_confidence})
    return results

