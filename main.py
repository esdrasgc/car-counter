import click
import cv2
import numpy as np
from tqdm import tqdm

from draw_utils import CountPlotter, draw_panel, draw_text
from tracker import Tracker
from yolo_detector import ObjectDetector

WIN_TITLE = 'Contador de veiculos'
PANEL_WIDTH = 300


@click.command()
@click.argument('filename')
@click.option("--nopreview", is_flag=True, show_default=True, default=False, help="Hide preview (will probably run faster)")
@click.option("--height", show_default=True, default=None, help="Scale image to this height before processing")
def main(filename, nopreview, height):
    """Detect and count vehicles and people in videos."""

    detector = ObjectDetector()
    tracker = Tracker()
    plotter = CountPlotter(tracker)
    cap = cv2.VideoCapture(filename=filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    roi = None
    
    # Define the target resolution (example: width=1280, height=720)
    target_width = 1280
    target_height = 720

    count = 0 ## gambi, tirar depois

    with tqdm(total=total_frames) as progress:
        while cap.isOpened:
            ret, frame = cap.read()
            count += 1
            if count % 3 != 0:
                continue
            if not ret:
                break


            # Get original frame dimensions
            original_height, original_width = frame.shape[:2]

            # Resize the frame if it's larger than the target resolution
            if original_width > target_width or original_height > target_height:
                scale_width = target_width / original_width
                scale_height = target_height / original_height
                scale = min(scale_width, scale_height)  # Maintain aspect ratio
                new_dimensions = (int(original_width * scale), int(original_height * scale))
                
                frame = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_AREA)

            # if height:
            #     scale = height / frame.shape[0]
            #     frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

            frame_height, frame_width, _ = frame.shape
            objects = detector.detect(frame)

            if not nopreview:
                full_win = np.ones((frame_height, frame_width + PANEL_WIDTH, 3), dtype=frame.dtype) * 255

            if not roi:
                if nopreview:
                    full_win = np.ones((frame_height, frame_width + PANEL_WIDTH, 3), dtype=frame.dtype) * 255
                full_win[:frame_height, :frame_width] = frame
                cv2.rectangle(full_win, (frame_width, 0), (full_win.shape[1], full_win.shape[0]), (0, 0, 0), -1)
                draw_text(full_win, 'Selecione a regiao de interesse\ne pressione [Enter]', frame_width + 10, 20, bg_cfg={'alpha': 0}, font_cfg={'color': (255, 255, 255)})
                roi = cv2.selectROI(WIN_TITLE, full_win, False, printNotice=False)
                if nopreview:
                    cv2.destroyWindow(WIN_TITLE)
                    cv2.waitKey(1)
            if roi == (0, 0, 0, 0):
                roi = (0, 0, frame_width, frame_height)

            objects = tracker.update(objects, roi)

            if not nopreview:
                for obj in objects:
                    obj.draw(frame)
                tracker.draw(frame, objects)

                full_win[:frame_height, :frame_width] = frame
                plotter.plot(full_win, frame_width)
                draw_panel(full_win, PANEL_WIDTH, tracker)

                cv2.imshow(WIN_TITLE, full_win)
                if cv2.waitKey(1) == 27:
                    break

            progress.update(1)

if __name__ == '__main__':
    main()
