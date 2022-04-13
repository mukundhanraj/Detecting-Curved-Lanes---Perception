#importing Libraries
import numpy as np
import cv2 as cv
# Function creates a sliding wdow on warped img
def sliding_wdow(img):

    lft_a = []; lft_b = []; lft_c = []
    rgt_a = []; rgt_b = []; rgt_c = []
    histo = hist(img)
    histo_ctr = int(histo.shape[0]/2)
    lft_histo = np.argmax(histo[:100])
    rgt_histo = np.argmax(histo[70:]) + histo_ctr
    w_h = int(img.shape[0]/24)
    lft_crnt = lft_histo
    rgt_crnt = rgt_histo
    lft_ln_indx = []
    rgt_ln_indx = []
    non_zro_x = np.array(img.nonzero()[1])
    non_zro_y = np.array(img.nonzero()[0])

    for w in range(25):

        y_lw = img.shape[0] - ((w+1)*w_h)
        y_hgh = img.shape[0] - (w*w_h)
        x_lft_lw = lft_crnt - 15
        x_lft_hgh = lft_crnt + 15
        x_rgt_lw = rgt_crnt - 15
        x_rgt_hgh = rgt_crnt + 15

        cv.rectangle(img, (x_lft_lw,y_lw),(x_lft_hgh, y_hgh), (255), 1)
        cv.rectangle(img, (x_rgt_lw,y_lw),(x_rgt_hgh, y_hgh), (255), 1)
        better_lft_w_indx = ((non_zro_x < x_lft_hgh) & (non_zro_x >= x_lft_lw) & (non_zro_y < y_hgh) & (non_zro_y >= y_lw)).nonzero()[0]
        better_rgt_w_indx =((non_zro_x < x_rgt_hgh) & (non_zro_x >= x_rgt_lw) & (non_zro_y < y_hgh) & (non_zro_y >= y_lw)).nonzero()[0]
        lft_ln_indx.append(better_lft_w_indx)
        rgt_ln_indx.append(better_rgt_w_indx)

        if len(better_lft_w_indx)>1:
            lft_crnt = (np.mean(non_zro_x[better_lft_w_indx])).astype(int)
        if len(better_rgt_w_indx)>1:
            rgt_crnt = (np.mean(non_zro_x[better_rgt_w_indx])).astype(int)

    lft_ln_indx = np.concatenate(lft_ln_indx)
    rgt_ln_indx = np.concatenate(rgt_ln_indx)

    lft_x = non_zro_x[lft_ln_indx]
    lft_y = non_zro_y[lft_ln_indx]
    rgt_x = non_zro_x[rgt_ln_indx]
    rgt_y = non_zro_y[rgt_ln_indx]

    if len(rgt_x) != 0:

        lft_fit = np.polyfit(lft_y, lft_x, 2)
        rgt_fit = np.polyfit(rgt_y, rgt_x, 2)

        lft_a.append(lft_fit[0])
        lft_b.append(lft_fit[1])
        lft_c.append(lft_fit[2])

        rgt_a.append(rgt_fit[0])
        rgt_b.append(rgt_fit[1])
        rgt_c.append(rgt_fit[2])

        lft_fit[0] = np.mean(lft_a[-10:])
        lft_fit[1] = np.mean(lft_b[-10:])
        lft_fit[2] = np.mean(lft_c[-10:])

        rgt_fit[0] = np.mean(rgt_a[-10:])
        rgt_fit[1] = np.mean(rgt_b[-10:])
        rgt_fit[2] = np.mean(rgt_c[-10:])

        plt_y = np.linspace(0, img.shape[0]-1, img.shape[0] )
        lft_fitx = lft_fit[0]*plt_y**2 + lft_fit[1]*plt_y + lft_fit[2]
        rgt_fitx = rgt_fit[0]*plt_y**2 + rgt_fit[1]*plt_y + rgt_fit[2]

        return img, (lft_fitx, rgt_fitx), (lft_fit, rgt_fit), plt_y
    else:
        return None

# Function to return the warpped img
def warping_image(img, source_points, destination_points, size):
    M = cv.getPerspectiveTransform(source_points, destination_points)
    wrpd_img = cv.warpPerspective(img, M, size)
    return wrpd_img

# Function to return the hist of img
def hist(img):
    histo = np.sum(img[:,:], axis=0)
    return histo

# Function to calculate radius of lft and rgt curve
def curve_ap(img, lft_x, rgt_x):

    plt_y = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(plt_y)

    lft_fit_cr = np.polyfit(plt_y, lft_x, 2)
    rgt_fit_cr = np.polyfit(plt_y, rgt_x, 2)

    lft_crve_rad = ((1 + (2*lft_fit_cr[0]*y_eval + lft_fit_cr[1])**2)**1.5) / np.absolute(2*lft_fit_cr[0])
    rgt_crve_rad = ((1 + (2*rgt_fit_cr[0]*y_eval + rgt_fit_cr[1])**2)**1.5) / np.absolute(2*rgt_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = lft_fit_cr[0]*img.shape[0]**2 + lft_fit_cr[1]*img.shape[0] + lft_fit_cr[2]
    r_fit_x_int = rgt_fit_cr[0]*img.shape[0]**2 + rgt_fit_cr[1]*img.shape[0] + rgt_fit_cr[2]
    ln_cntr_position = (r_fit_x_int + l_fit_x_int) /2
    cntr = (car_pos - ln_cntr_position)  / 10

    return (lft_crve_rad, rgt_crve_rad, cntr)


# Function to draw lines on the lft and rgt side and color the path of travel
def ln_drw(img, lft_fit, rgt_fit):

    plt_y = np.linspace(0, img.shape[0]-1, img.shape[0])

    for i in range(len(lft_fit)):

        cv.circle(img,(int(lft_fit[i]), int(plt_y[i])),0,(0,255,0),5)

    for i in range(len(rgt_fit)):

        cv.circle(img,(int(rgt_fit[i]), int(plt_y[i])),0,(0,0,255),5)

    lft = np.array([np.transpose(np.vstack([lft_fit, plt_y]))])
    rgt = np.array([np.flipud(np.transpose(np.vstack([rgt_fit, plt_y])))])
    points = np.hstack((lft, rgt))
    cv.fillPoly(img, np.int_(points), (0, 0, 255))
    dst = np.float32([[150,680],[630,425],[730,425],[1180,680]])
    src = np.float32([[0,400],[0,0],[200,0],[200,400]])
    img_size = (1280,720)
    frame_back = warping_image(img,src,dst,img_size)

    return frame_back


# Function to overlay imgs
def generateFinalimg(orig_img, warping_image, w_img, overlayed_img, curve_rad):

    cv.putText(orig_img,'(1)', (10, 70),cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)
    cv.putText(warping_image,'(2)', (10, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv.putText(w_img,'(3)', (10, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    if curve_rad[0] >0 and curve_rad[1] >0:
        cv.putText(overlayed_img,'Turn Right', (10, 50),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    warped_imgs = np.concatenate((warping_image, w_img), axis=1)
    mrge = cv.merge((warped_imgs,warped_imgs,warped_imgs))
    org_frame = cv.resize(orig_img,(400,320),interpolation = cv.INTER_AREA)
    rgt_stack = np.concatenate((org_frame,mrge),axis=0)
    img_stack = np.concatenate((overlayed,rgt_stack), axis=1)
    text_img = np.empty((150,1680,3),dtype= np.uint8)
    text_img[:,:]= (200,124,124)
    text = '(1) : Undistorted Image  (2) : Warped and filtered white and yellow ln'+' '+ '(3) : Sliding windows detecting the curve'
    curvature = 'Left Curvature = '+str(curve_rad[0]) + '  '+ 'Right Curvature = ' + str(curve_rad[1])

    cv.putText(text_img,text, (10, 50),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    cv.putText(text_img,curvature, (10, 100),cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)
    final_img = np.concatenate((img_stack,text_img),axis=0)

    return final_img

if __name__=='__main__':

    cap = cv.VideoCapture('suply\challenge.mp4')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    rows, columns = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    shape = (columns, rows)
    out = cv.VideoWriter('Outputs\Problem_3.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, shape)

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret == True:

            pts1 = np.float32([[150,680],[630,425],[730,425],[1180,680]])
            pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
            img_size = (200,400)
            frame_warp =warping_image(frame, pts1, pts2,img_size)
            frame_hsv = cv.cvtColor(frame_warp, cv.COLOR_BGR2HSV)
            mask_yl = cv.inRange(frame_hsv, (12, 124, 125), (33, 255, 255))
            frame_gray = cv.cvtColor(frame_warp, cv.COLOR_BGR2GRAY)
            ret,frame_bin = cv.threshold(frame_gray,200,255,cv.THRESH_BINARY)
            mskd = cv.bitwise_or(frame_bin, mask_yl)
            ftng = sliding_wdow(mskd)

            if ftng is not None:

                img_w, curves, lns, plt_y = ftng
                curve_radii = curve_ap(img_w, curves[0],curves[1])
                ln_detect = ln_drw(frame_warp, curves[0], curves[1])
                overlayed = cv.addWeighted(frame,1,ln_detect,0.5,0)

                final_img =generateFinalimg(frame,mskd,img_w,overlayed, curve_radii)
                cv.imshow("The lane detected",final_img)
                out.write(final_img)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllwindows()
