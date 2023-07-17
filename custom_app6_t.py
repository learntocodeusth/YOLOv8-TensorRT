import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from config import CLASSES, COLORS
from models.utils import blob, det_postprocess, letterbox, path_to_list


def main(args: argparse.Namespace) -> None:
    if args.method == 'cudart':
        from models.cudart_api import TRTEngine
    elif args.method == 'pycuda':
        from models.pycuda_api import TRTEngine
    else:
        raise NotImplementedError

    Engine = TRTEngine(args.engine)
    H, W = Engine.inp_info[0].shape[-2:]

    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'Camera FPS: {fps:.2f}')
    start_time = time.time()
    frame_num = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out_file, fourcc, fps, (args.width, args.height))

    while True:
        ret, bgr = cap.read()
        if not ret:
            continue

        draw = bgr.copy()
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)
        # inference
        start_inference_time = time.time()
        data = Engine(tensor)
        end_inference_time = time.time()

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.size == 0:
            # if no bounding box
            print('No object!')
            continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)
        
        # write FPS on each frame
        frame_num += 1
        elapsed_time = time.time() - start_time
        inference_time = end_inference_time - start_inference_time
        fps_text = f'FPS: {frame_num/elapsed_time:.2f} | Inference Time: {inference_time:.4f}s'
        cv2.putText(draw, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if args.show:
            cv2.imshow('result', draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out.write(draw)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--device', type=int, default=0, help='Camera device number')
    parser.add_argument('--width', type=int, default=640, help='Video frame width')
    parser.add_argument('--height', type=int, default=480, help='Video frame height')
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-file',
                        type=str,
                        default='./output.mp4',
                        help='Path to output file')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output directory')
    parser.add_argument('--method',
                        type=str,
                        default='cudart',
                        help='CUDART pipeline')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)