import cv2

face_cascade = cv2.CascadeClassifier('../files/haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('../files/lunettes.png', -1)
cigar_ori = cv2.imread('../files/joint.png', -1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)


def main():
    while True:
        ret, video = cap.read()
        faces = face_cascade.detectMultiScale(video, 1.2, 5, 0, (120, 120), (350, 350))
        for (x, y, w, h) in faces:
            glass_symin = int(y + 1.5 * h / 5)
            glass_symax = int(y + 2.5 * h / 5)
            sh_glass = glass_symax - glass_symin

            cigar_symin = int(y + 4 * h / 6)
            cigar_symax = int(y + 5.5 * h / 6)
            sh_cigar = cigar_symax - cigar_symin

            face_glass_roi_color = video[glass_symin:glass_symax, x:x + w]
            face_cigar_roi_color = video[cigar_symin:cigar_symax, x:x + w]

            specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
            cigar = cv2.resize(cigar_ori, (int(w / 2), sh_cigar), interpolation=cv2.INTER_CUBIC)
            transparent_overlay(face_glass_roi_color, specs)
            transparent_overlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 5)))

        cv2.imshow('Video', video)

        if cv2.waitKey(30) & 0xff == 27:
            break

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


if __name__ == '__main__':
    main()
