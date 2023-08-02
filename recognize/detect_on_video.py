import os
import cv2
from tqdm import trange
from pathlib import Path
from utils.ocr_utils import ocr_plate
from utils.img_utils import crop, plot_box
from utils.object_detect_utils import detect_plates
from utils.misc_utils import pascal_voc_to_coco, get_best_ocr
from deep_sort_realtime.deepsort_tracker import DeepSort


def get_plates_from_video(source, output):
    # Create a VideoCapture object
    video = cv2.VideoCapture(source)

    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object.
    temp = f'{Path(output).stem}_temp{Path(output).suffix}'
    export = cv2.VideoWriter(temp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Intializing tracker
    tracker = DeepSort()

    # Initializing some helper variables.
    preds = []
    total_obj = 0

    for i in trange(frames_total):
        ret, frame = video.read()
        if ret is True:
            # Run the ANPR algorithm
            det_predictions = detect_plates(frame)
            # Convert Pascal VOC detections to COCO
            bboxes = list(map(lambda bbox: pascal_voc_to_coco(bbox), [det_prediction['coords'] for det_prediction in det_predictions]))

            if len(bboxes) > 0:
                # Storing all the required info in a list.
                detections = [(bbox, score, 'number_plate') for bbox, score in zip(bboxes, [det_prediction['det_conf'] for det_prediction in det_predictions])]

                # Applying tracker.
                # The tracker code flow: kalman filter -> target association(using hungarian algorithm) and appearance descriptor.
                tracks = tracker.update_tracks(detections, frame=frame)

                # Checking if tracks exist.
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # Changing track bbox to top left, bottom right coordinates
                    bbox = [int(position) for position in list(track.to_tlbr())]

                    for i in range(len(bbox)):
                        if bbox[i] < 0:
                            bbox[i] = 0

                    # Cropping the license plate and applying the OCR.
                    plate_region = crop(frame, bbox)
                    ocr_prediction = ocr_plate(plate_region)
                    plate_text, ocr_confidence = (
                        ocr_prediction["plate"],
                        ocr_prediction["ocr_conf"],
                    )

                    # Storing the ocr output for corresponding track id.
                    output_frame = {'track_id': track.track_id, 'ocr_txt': plate_text, 'ocr_conf': ocr_confidence}

                    # Appending track_id to list only if it does not exist in the list
                    # else looking for the current track in the list and updating the highest confidence of it.
                    if track.track_id not in list(set(pred['track_id'] for pred in preds)):
                        total_obj += 1
                        preds.append(output_frame)
                    else:
                        preds, ocr_confidence, plate_text = get_best_ocr(preds, ocr_confidence, plate_text, track.track_id)

                    # Plotting the prediction.
                    plot_box(frame, bbox, f'{str(track.track_id)}. {plate_text}', color=[255, 150, 0])

            # Write the frame into the output file
            export.write(frame)
        else:
            break

    # When everything done, release the video capture and video write objects
    video.release()
    export.release()

    # Compressing the video for smaller size and web compatibility.
    os.system(f'ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 1 -c:a aac -f mp4 /dev/null && ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 2 -c:a aac -movflags faststart {output}')
    os.system(f'rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')
