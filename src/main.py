import cv2
import argparse

parser = argparse.ArgumentParser(description='Augmented reality application')
parser.add_argument('-s', '--source', help='Determine the source for the video', default=0, type=str)
parser.add_argument('-d', '--dog', help='Display a mask with a dog mask', action='store_true')
args = parser.parse_args()


def main():
    glasses_names = ['glasses_01.png', 'glasses_02.png', 'glasses_03.png']
    glasses = []
    glasses_choice = 0
    dog = cv2.imread('../files/dog.png')
    for glasses_name in glasses_names:
        glasses.append(cv2.imread('../files/' + glasses_name, -1))

    face_cascade = cv2.CascadeClassifier('../files/haarcascade_frontalface_default.xml')
    glasses_ori = glasses[glasses_choice]
    cigar_ori = cv2.imread('../files/joint.png', -1)

    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ret, frame = cap.read()

        faces = face_cascade.detectMultiScale(frame, 1.2, 5, 0)

        if args.dog:
            for x, y, w, h in faces:
                y0, y1 = int(y - 0.25 * h), int(y + 0.75 * h)
                x0, x1 = x, x + w

                frame[y0: y1, x0: x1] = insert_dog_face_on_faces(frame[y0: y1, x0: x1], dog)
        else:
            frame = insert_images_on_faces(faces, frame, glasses_ori, cigar_ori)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(30)

        if key == 113:
            break
        elif key == 83:
            glasses_choice += 1
            if glasses_choice > len(glasses) - 1:
                glasses_choice = 0
            glasses_ori = glasses[glasses_choice]

    cap.release()
    cv2.destroyAllWindows()


def transparent_overlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src


def insert_images_on_faces(faces, frame, glasses, cigar):
    for (x, y, w, h) in faces:
        glass_symin = int(y + 1.5 * h / 5)
        glass_symax = int(y + 2.5 * h / 5)
        sh_glass = glass_symax - glass_symin

        cigar_symin = int(y + 4 * h / 6)
        cigar_symax = int(y + 5.5 * h / 6)
        sh_cigar = cigar_symax - cigar_symin

        face_glass_roi_color = frame[glass_symin:glass_symax, x:x + w]
        face_cigar_roi_color = frame[cigar_symin:cigar_symax, x:x + w]

        specs = cv2.resize(glasses, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
        cigar = cv2.resize(cigar, (int(w / 2), sh_cigar), interpolation=cv2.INTER_CUBIC)
        transparent_overlay(face_glass_roi_color, specs)
        transparent_overlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 5)))

    return frame


def insert_dog_face_on_faces(face, mask):
    """Add the mask to the provided face, and return the face with mask."""
    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)

    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    off_h = int((face_h - new_mask_h) / 2)
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h + new_mask_h, off_w: off_w + new_mask_w][non_white_pixels] = resized_mask[non_white_pixels]

    return face_with_mask


if __name__ == '__main__':
    main()
