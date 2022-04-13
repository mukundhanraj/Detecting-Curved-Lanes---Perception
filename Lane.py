#Importing Libraries
import cv2
import numpy as np
import math
import functions

if __name__ == '__main__':
    cap = cv2.VideoCapture('suply/whiteline.mp4')
    while cap.isOpened():
        ret, frm = cap.read()
        if ret:
            cv2.imshow('Input Video', frm)
            thresh = functions.noise_elim(frm)
            lns = cv2.HoughLinesP(thresh, 4, np.pi/180, 100, None, 20, 60)

            ln_1, ln_2 = functions.ln_gen(lns)
            if (ln_1[0] > ln_2[0]):
                pts = functions.pt_ln(frm, ln_1)
                cv2.line(frm, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 255, 0], 3)
                pts = functions.pt_ln(frm, ln_2)
                cv2.line(frm, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 0, 255], 3)
            else:
                pts = functions.pt_ln(frm, ln_1)
                cv2.line(frm, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 0, 255], 3)
                pts = functions.pt_ln(frm, ln_2)
                cv2.line(frm, (pts[0], pts[1]),
                         (pts[2], pts[3]), [0, 255, 0], 3)

            cv2.imshow('Output of Lane Detection', frm)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
