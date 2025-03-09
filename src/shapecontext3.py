
# Copyright (C) 2024 Brendan Murphy - All Rights Reserved
# This file is part of the "Shape Matchers (3 Methods)" project.
# See the LICENSE.TXT file which should be included in the same directory as this file.


import numpy as np
import math
import cv2
import imutils as im
import time


def fan_delete_resize(larger_array, smaller_array, larger_axis=0, smaller_axis=0, keep_end=0):
    '''
    Returns a ndarray of larger_array evenly deleted down to the size of smaller_array.

    * larger_array: the ndarray or list to be downsized
    * smaller_array: the ndarray, list, or int to downsized larger_array too
    * larger_axis: the axis of the larger_array to downsize
    * smaller_axis: the axis of the smaller_array to base the size of the larger array on
    * keep_end: keep the start, or the end of larger_array
    * keep_end=-1 indexes kept will be shifted towards the end, and the last index will be keep
    * keep_end=0 will delete evenly as possible, but maybe shift towards the front
    * keep_end=1 indexes kept will be shifted towards the start, and the first index will be keep
    '''

    if type(smaller_array) == int:
        shrink_to = smaller_array
    else:
        shrink_to = np.size(smaller_array, axis=smaller_axis)

    num_of_steps = np.size(larger_array, axis=larger_axis) - shrink_to

    step = np.size(larger_array, axis=larger_axis) / num_of_steps

    leftovr = 0
    idx_array = np.array([])
    applied_step = 0
    step2 = 0
    ends = 0

    for i in range(0, num_of_steps):

        if keep_end != 0:
            if step2 == 0:
                ends = step / 2
            else:
                ends = 0

        step2 = step + applied_step + leftovr - ends
        applied_step = round(step2)

        leftovr = step2 - applied_step

        idx_array = np.append(idx_array, applied_step)

    if keep_end <= 0:
        idx_array = np.subtract(idx_array, np.ones_like(idx_array))

    reduced_array = np.delete(larger_array, idx_array.astype("int32"), axis=larger_axis)

    return reduced_array



def find_cont(img):

    # _, thresh = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(img, contours, -1, (0, 25, 255), 3)
    for c in contours:
        if cv2.contourArea(c) < 20:
            continue
        # find the biggest countour (c) by the area
        # max_c = max(contours, key=cv2.contourArea)
    #     # --- find center of the contour ----
    #     M = cv2.moments(c)
    #     cX = int(M["m10"] / M["m00"])
    #     cY = int(M["m01"] / M["m00"])

    # print("max c shape ", max_c.shape)
    # print("contours shape ", contours[0].shape)

    return contours


def center_contour(img, cont, contour_thickness=-1, center_based_on=0, scale_value=1):

    '''

    (Seems to need full cont, changing to cont[0], in if statement, messes up the moving)

    Params
    ------
    * img = a gray image to draw centered contours on a return.
    * cont = contours, or contour[0]
    * contour_thickness = the thickness of the drawn centered contours (-1 to fill)
    * scale_value = a measured feature of the contour, which can later be used approximately scale
         the contour to that of a contour from a simular image, such as a template.

    center_based_on =
    -------
        center the contours based on...
    * 0 - center of mass
    * 1 - center of circle
    * 2 - average of mass and circle
    * 3 - center of triangle
    * 4 - center of min area rec
    * 5 - average of mass and rect

    scale_value =
    -------
        return a scale value that is...
    * 0 - contour area
    * 1 - circle radius
    * 2 - average of contour area and circle radius

    Returns:
    -------
    image, centered contour, scale value

    (Note: uses cvtools.tri_center())
    '''

    if len(cont) == 1:
        cont = cont[0]

    # xx Faults when contour is just a line, and area is 0.0, so return inputted.. may not be best way fixme
    if cv2.contourArea(cont) == 0:
        return img, cont, 1

    img = np.zeros_like(img)

    if scale_value == 0 or scale_value == 2 or center_based_on == 0 or center_based_on == 2 or center_based_on ==5:
        # Find center of the contour  "An image moment is a particular weighted average of image pixel intensities"
        M = cv2.moments(cont)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])  # M["m00"] is contour area

        scale_v_a = M["m00"]

        # area = cv2.contourArea(cont[0])

    if scale_value == 1 or scale_value == 2 or center_based_on == 1 or center_based_on == 2:
        # Scaling and centering based on min enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cont)
        circle_center = (int(x), int(y))
        scale_v_r = int(radius)

    # If using average of area and circle radius scale value
    if scale_value == 2:
        scale_v = (scale_v_a + scale_v_r) / 2

    elif scale_value == 0:
        scale_v = scale_v_a
    elif scale_value == 1:
        scale_v = scale_v_r

    # Can add more options for scale value...

    # Centering and maybe scaling based on min area rect, see ContShapeAn4.py for more
    rect = cv2.minAreaRect(cont)

    # Centering and maybe scaling based on min enclosing triangle    gives area and 3 points
    area, tri = cv2.minEnclosingTriangle(cont)

    tri_cent = tri_center(tri[0, 0], tri[1, 0], tri[2, 0])


    # Img center
    img_cent_x = (img.shape[1] // 2)
    img_cent_y = (img.shape[0] // 2)

    if center_based_on == 0:
        # Move to center of image based on center of contour mass
        move_x = img_cent_x - cX
        move_y = img_cent_y - cY

    elif center_based_on == 1:
        # Move to center based on center of min enclosing circle
        move_x = img_cent_x - circle_center[0]
        move_y = img_cent_y - circle_center[1]

    elif center_based_on == 2:
        # Move to center based on average or median of center of min enclosing circle, and center of contour volume
        move_x = (img_cent_x - circle_center[0] + img_cent_x - cX) // 2
        move_y = (img_cent_y - circle_center[1] + img_cent_y - cY) // 2

    elif center_based_on == 3:
        # Move to center based on center of min enclosing triangle
        move_x = img_cent_x - tri_cent[0]
        move_y = img_cent_y - tri_cent[1]

    elif center_based_on == 4:
        # Move to center based on center of min area rect
        move_x = img_cent_x - rect[0][0]
        move_y = img_cent_y - rect[0][1]

    elif center_based_on == 5:
        # Move to center based on average or median of center of min area rect, and center of contour volume
        move_x = (img_cent_x - rect[0][0] + img_cent_x - cX) // 2
        move_y = (img_cent_y - rect[0][1] + img_cent_y - cY) // 2

    centered_cnt = np.asarray(cont)

    # Centers contour. Can be used to move, scale, and try rotate
    centered_cnt[:, :, 0] = centered_cnt[:, :, 0] + move_x
    centered_cnt[:, :, 1] = centered_cnt[:, :, 1] + move_y

    # Expand dims so that contours can be drawn filled
    centered_cnt = centered_cnt[np.newaxis]

    # Can use cont Idx to draw different sections
    cv2.drawContours(img, centered_cnt, -1, 255, contour_thickness)   # for Gray img

    # cv2.imshow("testing for min closing", img)

    return img, centered_cnt, scale_v

# # xx Scaling do in loop if using ContShapeAn way, do here if going to freeman_chain
# # Scale sample to template based any returned scale_value
# scale_ratio = temp_SV / samp_SV  # cmaxc * x = tmaxc     Works when base on min enclosing circle
#
# # Now sample image or contours can be scaled to approximately match that of template
# # To be used as:
# scaled_sample = cvt.rotate(img, None, None, scale_ratio)   # scale before loop doesn't really speed it up?
# # or
# cvt.scale_contour(cont[0], scale_ratio)


def match_best_rotation(template_img, sample_img, step=1, break_accuracy=101, subtract=True, view=False):
    # todo consider np subtracting
    '''
    Rotates a sample image by "step" amount of degrees up to 370 degrees,
    and cv2.subtracts the template image for each step. Calculates and
    returns, a best accuracy value, and the degrees at which the best
    accuracy occurred. cvt.rotate() can then be used to rotate the
    sample image to approximately, or best match the template image.

    Originally designed to be used after cvt.center_contour(), to
    match the sample rotation to the templates rotation,
    use the flowing code (example) and pass in samp_cent_img which

    is the sample image scaled to near template image size:
    ---------
    * # Scale sample image to template image
    * scale_ratio = temp_scale_value / samp_scale_value
    * samp_cent_img = cvt.rotate(sample_centered_img, None, None, scale_ratio)
    :param template_img: The gray template image (for most cases a binary image).
    :param sample_img: The gray sample image (for most cases a binary image).
    :param step: How many degrees to step per rotation of sample_img, a higher number makes it faster,
        but lowers accuracy.
    :param break_accuracy: if this percentage value of accuracy is reached, break the loop. Is used for speed up.
    :param subtract: if True then the template is subtracted from the sample, otherwise cv2.absdiff is used.
    :param view: View as sample rotates with template subtracted, in cv2 window. Press or hold any key.
    :return: best_accuracy (a percentage), best_acc_degrees.
    '''

    best_accuracy = 0
    best_acc_degrees = 0
    samp_sum = int(np.sum(sample_img))

    for idx in range(0, 370, step):

        sample_rot = im.rotate(sample_img, idx)

        if subtract == True:
            diff = cv2.subtract(sample_rot, template_img)  # To check to 100% if samp_rot is within template
        else:
            diff = cv2.absdiff(sample_rot, template_img)

        diff_sum = int(np.sum(diff))  # // 10000   int for numpy long scalars problem
        saved_accuracy = best_accuracy

        # accuracy = round((temp_sum / diff_sum * 100), 2)      # for absdiff, not 0-100
        accuracy = round(((samp_sum - diff_sum) / samp_sum * 100), 2)   # For subtract  is 0 - 100%

        if saved_accuracy >= accuracy:
            best_accuracy = saved_accuracy
        else:
            best_accuracy = accuracy  # return these 2,
            best_acc_degrees = idx

        if best_accuracy >= break_accuracy:
            break

        if view == True:
            cv2.waitKey(0)
            cv2.imshow("match_best_rotation view", diff)

    return best_accuracy, best_acc_degrees



def freeman_chain(img, cnt, front_start=40, middle_start=20, step=2, level=1, rad_or_thk=4, thres=0.8, mode=0,
                  subtract=False, view=False, sleep=0.02):
    '''
    It's not really a freeman chain
    '''

    fs = front_start
    ms = middle_start
    idx = 0
    fs_switch = False
    ms_switch = False
    mrs_switch = False
    mre_switch = False
    last_f = 0
    b_paused = 0
    corn_switch = True
    m_range_end = 0
    m_range_start = 0
    save_dist = 0
    corner = np.array([])

    # temp_cnt = np.roll(cnt, -650)
    temp_cnt = cnt
    save_draw = img.copy()
    cv2.drawContours(save_draw, temp_cnt, -1, 255, 0)

    # save_draw = cv2.cvtColor(save_draw, cv2.COLOR_BGR2GRAY)

    while True:

        if fs >= len(temp_cnt):
            break

        if idx >= len(temp_cnt):
            break

        if idx + fs >= len(temp_cnt) and fs_switch == False:
            fs = fs - len(temp_cnt)
            fs_switch = True

        if idx + ms >= len(temp_cnt) and ms_switch == False:
            ms = ms - len(temp_cnt)
            ms_switch = True

        temp_cent_img_draw = img.copy()
        cv2.drawContours(temp_cent_img_draw, temp_cnt, -1, 255, 0)  # has to be full contours to fill, will aslo draw
                                                                       # lines between real points
        f = temp_cnt[:][idx + fs][0]
        m = temp_cnt[:][idx + ms][0]
        b = temp_cnt[:][idx][0]

        # f = temp_cnt[:][fs][0]    # for np.roll
        # m = temp_cnt[:][ms][0]
        # b = temp_cnt[:][idx][0]

        cv2.circle(temp_cent_img_draw, f, 5, 85, -1)
        cv2.circle(temp_cent_img_draw, m, 5, 110, -1)
        cv2.circle(temp_cent_img_draw, b, 5, 85, -1)

        fm = math.hypot((f[0] - m[0]), (f[1] - m[1]))
        mb = math.hypot((m[0] - b[0]), (m[1] - b[1]))
        bf = math.hypot((b[0] - f[0]), (b[1] - f[1]))

        try:
            height = tri_height(bf, fm, mb) #* level
        except:
            height = 0
            print("Failed height calc, set height to 0 and continued")

        # xx ------- Circle dilate or erode on curves ------
        if mode == 0 or mode == 2:
            if height > bf * thres:
                if rad_or_thk == None or rad_or_thk == 0:
                    if subtract == True:
                        cv2.circle(save_draw, m, round(height * level), 0, -1)   # use "level" to adjust radius of circle
                    else:
                        cv2.circle(save_draw, m, round(height * level), 150, -1)

                else:
                    if subtract == True:
                        cv2.circle(save_draw, m, rad_or_thk, 0, -1)
                    else:
                        cv2.circle(save_draw, m, rad_or_thk, 150, -1)

        # xx -------- Line extend or cut on curves --------
        if mode == 1 or mode == 2:
            if height > bf * thres:
                x_m_pt = int((f[0] + b[0]) / 2)    # finding middle on straight between f, and b
                y_m_pt = int((f[1] + b[1]) / 2)
                mid = (x_m_pt, y_m_pt)
                # cc = (m - mid) * round(height)    #  <---- can also be settable?
                cc = (m - mid) * level                 # multiply by num to get longer or shorter lines
                cd = np.add((round(m[0]), round(m[1])), (round(cc[0]), round(cc[1])))

                if subtract == True:
                    cv2.line(save_draw, m, cd, 0, rad_or_thk)
                    # cv2.line(save_draw, m, cd, 0, round(height))   #  <---- can also be settable?
                else:
                    cv2.line(save_draw, m, cd, 130, rad_or_thk)
                    # cv2.line(save_draw, m, cd, 130, round(height))

        # mode 3 find corners by hypot from outside, draw one corner circle, on sharp corners. May need for loop or cycler
        if mode == 3 or mode == 4:
            if height > bf * thres:    # this seems better, but still misses some corners on certain thres values

                # Grab b, at first instance of entering corner
                if corn_switch == True:
                    # b_paused = b
                    b_paused = idx
                    m_range_start = idx + ms    # use ms instead of middle_start, they are the same, but ms updates and end
                    corn_switch = False             # of cont index

                last_f = idx + fs
                m_range_end = idx + ms

            else:
                corn_switch = True

                if b_paused > 0 and last_f > 0:

                    # mid is middle of f and b. Farthest m pt from mid is corner.
                    x_m_pt = int((temp_cnt[last_f, 0, 0] + temp_cnt[b_paused, 0, 0]) / 2)  # finding middle on straight between f, and b
                    y_m_pt = int((temp_cnt[last_f, 0, 1] + temp_cnt[b_paused, 0, 1]) / 2)
                    mid = np.asarray([x_m_pt, y_m_pt])

                    range_len = abs(m_range_end - m_range_start)
                    if m_range_start > m_range_end:
                        start = m_range_start - len(temp_cnt)
                        range_len = abs(m_range_end - start)

                    for idx4, i in enumerate(range(range_len + 1)):

                        # stops from going out of range at end, but seems to loop around again
                        if m_range_start + idx4 >= len(temp_cnt) and mrs_switch == False:
                            m_range_start = m_range_start - len(temp_cnt)
                            mrs_switch = True

                        if m_range_end + idx4 >= len(temp_cnt) and mre_switch == False:
                            m_range_end = m_range_end - len(temp_cnt)
                            mre_switch = True

                        current = temp_cnt[m_range_start + idx4, 0]

                        if mode == 3:
                            dist = math.hypot((mid[0] - current[0]), (mid[1] - current[1]))

                            if dist > save_dist:
                                corner = temp_cnt[m_range_start + idx4, 0]
                                save_dist = dist

                        if mode == 4:
                            cc = (current - mid) * level  # multiply by num to get longer or shorter lines
                            cd = np.add((round(current[0]), round(current[1])), (round(cc[0]), round(cc[1])))

                            dist = math.hypot((cd[0] - current[0]), (cd[1] - current[1]))

                            if dist > save_dist:
                                cd_final = cd
                                corner = current
                                save_dist = dist

                    if corner.any():
                        if mode == 3:
                            if subtract == True:
                                cv2.circle(save_draw, corner, rad_or_thk, 0, -1)
                            else:
                                cv2.circle(save_draw, corner, rad_or_thk, 120, -1)
                        if mode == 4:
                            if subtract == True:
                                cv2.line(save_draw, corner, cd_final, 0, rad_or_thk)
                            else:
                                cv2.line(save_draw, corner, cd_final, 120, rad_or_thk)

                    last_f = 0
                    b_paused = 0
                    m_range_start = 0
                    m_range_end = 0
                    save_dist = 0

        cv2.line(temp_cent_img_draw, f, m, 222, 2)
        cv2.line(temp_cent_img_draw, m, b, 222, 2)
        cv2.line(temp_cent_img_draw, b, f, 222, 2)

        idx += step

        # xx If np.roll used, no need to step
        # temp_cnt = np.roll(temp_cnt, 2)    # another way to change the step, at least visually. Rolling 2 is stepping 1
        cv2.circle(temp_cent_img_draw, temp_cnt[0][0], 5, 255, -1)

        if view == True:

            cv2.imshow("save_draw", save_draw)
            cv2.imshow("temp draw on 2", temp_cent_img_draw)
            time.sleep(sleep)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return save_draw


# Templates need to be larger then sample images/contours
# Used in template set-up and chart matching
def contour_chain_prop(img, cnt, retur="height", front_start=40, middle_start=20, step=2, view=False):

    # has temporary code search for *****

    fs = front_start
    ms = middle_start
    idx = 0
    fs_switch = False
    ms_switch = False

    temp_cnt = cnt

    save_draw = img.copy()

    hi_list = np.array([])

    fidx = 0
    midx = 0
    bidx = 0
    sf = temp_cnt[:][fs][0]
    sm = temp_cnt[:][ms][0]
    sb = temp_cnt[:][bidx][0]

    # Method 2
    negf = 0
    negm = 0
    negb = 0
    accf = 0
    accm = 0
    accb = 0

    if view == True:
        temp_cent_img_draw = img.copy()  # moved out of loop put in view also
        temp_cent_img_draw = cv2.cvtColor(temp_cent_img_draw, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(temp_cent_img_draw, temp_cnt, -1, (255, 255, 255))  # has to be full contours to fill, will aslo draw
                    # lines between real points

    # for circle # fixme experimental, remove when removing circle part

    img5 = np.zeros_like(img)
    cv2.drawContours(img5, temp_cnt, -1, 255, -1)
    # cv2.imshow("img5 first draw cont", img5)
    co = find_cont(img5)
    cv2.drawContours(img5, co, -1, 255, -1)
    # cv2.imshow("img5 2nd draw cont", img5)

    imgSum = np.sum(img5)
    # imgSum = cv2.contourArea(co[0])
    # print("imgSum ", imgSum)
    # print("sum of img5 with cont drawn filled ", np.sum(img5))


    while True:

        if fs >= len(temp_cnt):
            break

        if bidx >= len(temp_cnt):
            # cv2.waitKey(0)
            break

        if fidx + fs >= len(temp_cnt) and fs_switch == False:
            fs = fs - len(temp_cnt)
            fs_switch = True

        if midx + ms >= len(temp_cnt) and ms_switch == False:
            ms = ms - len(temp_cnt)
            ms_switch = True

        f = temp_cnt[:][fidx + fs][0]
        m = temp_cnt[:][midx + ms][0]    # fixme find out why middle falls back, ..seems to be working fine as is
        b = temp_cnt[:][bidx][0]

        # <----- Method 2 - so can handle steps greater then 1
        # How much over and down the pt moved this step
        movfx = abs(sf[0] - f[0])
        movfy = abs(sf[1] - f[1])

        diaf = min(movfx, movfy)

        # Accumulate sqrt of 2 for each dia step
        accf = accf + (diaf * 0.41421356)

        # Take out of accum. full numbers, and nefg will be subtracted from the next f step
        # if accf >= 1:
        if round(accf) >= 1:
            negf = round(accf)     # using round is slightly better or worse, should be tested on more imgs
            # negf = int(accf)
            accf = accf - negf

        # Middle pt
        movmx = abs(sm[0] - m[0])
        movmy = abs(sm[1] - m[1])

        diam = min(movmx, movmy)  # this should work for all step amounts xx best way

        accm = accm + (diam * 0.41421356)

        # if accm >= 1:
        if round(accm) >= 1:
            negm = round(accm)
            # negm = int(accm)
            accm = accm - negm

        # Back pt
        movbx = abs(sb[0] - b[0])
        movby = abs(sb[1] - b[1])

        diab = min(movbx, movby)  # xx best way found

        accb = accb + (diab * 0.41421356)

        # if accb >= 1:
        if round(accb) >= 1:
            negb = round(accb)
            # negb = int(accb)
            accb = accb - negb

        sf = f  # save f
        sm = m
        sb = b

        fm = math.hypot((f[0] - m[0]), (f[1] - m[1]))
        mb = math.hypot((m[0] - b[0]), (m[1] - b[1]))
        bf = math.hypot((b[0] - f[0]), (b[1] - f[1]))

        # xx check if pt or pts along bf are inside contour, or what color, to determine if height should be negative
        # Using pointPolygonTest to check if inside

        # cd will be the pt that is opposite m, this also done below so if using this here can remove below
        x_m_pt = int((f[0] + b[0]) / 2)  # finding middle on straight between f, and b
        y_m_pt = int((f[1] + b[1]) / 2)
        mid = (x_m_pt, y_m_pt)
        cc = (m - mid) #* level  # multiply by num to get longer or shorter lines       # * level below
        cd = np.add((round(m[0]), round(m[1])), (round(cc[0]), round(cc[1])))

        # Color is used for determining if negatve, for triangle height, others will also be counted as negative
        color = img[int(cd[1]), int(cd[0])]    # seems to work need to change chart, do in another file
        # fixme faults here, maybe when template img rotates out of bounds

        if retur == "height":
            height = tri_height(bf, fm, mb) #* level
            if color > 127:  # has some jitter, 127 in case img is not properly normed, cv2.thres is not perfect, used np.where
                height *= -1
            # print("height is  ", height)

            # Possible new parameter or new func  this if  ****** makes list shorter so will speed up, but so far works
            # about the same. Can remove any under threshold after this func also, or at end.
            # if height > 1.0 or height < -1.0:
            hi_list = np.append(hi_list, height)

        elif retur == "area":
            area = tri_area(bf, fm, mb)
            if color > 127:         # fixme area is extreme goes off charts negative?
                area *= -1
            print("area is  ", area)
            hi_list = np.append(hi_list, area)

        elif retur == "base":
            # is base of triangle, if font_s is halt of len contours, then only half
            # of this array is needed, rest will repeat,
            if color > 127:                     # using negative here to tell weather bend is concave or convex
                bf *= -1
            hi_list = np.append(hi_list, bf)  # base length

        elif retur == "median":
            med = tri_median(bf, fm, mb)
            if color > 127:
                med *= -1
            hi_list = np.append(hi_list, med)


        elif retur == "angle":
            ang = tri_ang_from_side_lens(fm, mb, bf)
            if color > 127:
                ang *= -1
            hi_list = np.append(hi_list, ang)


        elif retur == "rect":    # fixme remove? not working good but interesting
            # won't work,
            # rect_p1y = min(f[0], b[0])
            # rect_p1x = min(f[1], b[1])
            #
            # rect_p2y = max(f[0], b[0])
            # rect_p2x = max(f[1], b[1])
            #
            # rect_sum = np.sum(img[rect_p1y:rect_p2y, rect_p1x:rect_p2x]) / 1000
            #
            # rect_area = (rect_p2y - rect_p1y) * (rect_p2x - rect_p1x) / 100
            # cv2.imshow("img", img)

            # if bidx >= len(temp_cnt):
            #     # cv2.waitKey(0)
            #     break
            # print("ms ", ms)
            # print("b ", b)

            # try summing area around mid pt, should do a circle mask
            # img5 = img.copy()
            # isum = np.sum(img5)      # move to start
            mask = np.zeros_like(img5)
            cv2.circle(mask, b, middle_start, 255, 1)      # cicle is not showing good results either
            # bit = cv2.bitwise_not(mask, img5, mask=img5)
            bit = cv2.absdiff(mask, img5)
            # bit[:, :] = np.where(bit[:, :] > 1, 1, 0)
            # print("imgSum ", imgSum)
            # print(int(imgSum) - int(np.sum(bit)))
            # print(img.shape, "  ", bit.shape, "  ", mask.shape)
            # cir_sum = np.sum(bit) / imgSum * 100 #// 100 #+ (np.sum(mask) / 2)

            cir_sum = (int(np.sum(bit)) - int(imgSum)) // 1000
            #xx Got nice results when using this for set up, and grow about 2, but "height" in chart match
            #xx probably because the circle size isn't scaling right, middle_start needs to be right when passed into
            #xx chart match ---- Need to get sum of just 1 cont at a time

            #xx working ok, has a scaling problem, not on high grow though, Maybe make this it's own func, and use np.roll
            # sum just area that needs to be. Can use lines or even imgs instead of circles.

            # cv2.imshow("mask", mask)
            # cv2.imshow("bit", bit)
            # cv2.imshow("img5", img5)

            hi_list = np.append(hi_list, cir_sum)

            # if view == True:
            #     cv2.rectangle(temp_cent_img_draw, (rect_p1y, rect_p1x), (rect_p2y, rect_p2x), 180, 2)

        # <----- Method 2
        fidx += step - negf
        midx += step - negm
        bidx += step - negb

        negf = 0
        negm = 0
        negb = 0
        # ------>

        idx += step

        # xx If np.roll used, no need to step
        # temp_cnt = np.roll(temp_cnt, 2)    # another way to change the step, at least visually. Rolling 2 is stepping 1
        # cv2.circle(temp_cent_img_draw, temp_cnt[0][0], 5, 255, -1)

        if view == True:

            temp_cent_img_draw = img.copy()  # moved out of loop put in view also
            temp_cent_img_draw = cv2.cvtColor(temp_cent_img_draw, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(temp_cent_img_draw, temp_cnt, -1,
                             (255, 255, 255))  # has to be full contours to fill, will also draw
            # lines between real points

            # # draw rect
            # cv2.rectangle(temp_cent_img_draw, (rect_p1y, rect_p1x), (rect_p2y, rect_p2x), 180, 2)

            # The triangle lines
            cv2.line(temp_cent_img_draw, f, m, (0, 0, 220), 2)
            cv2.line(temp_cent_img_draw, m, b, (0, 0, 220), 2)
            cv2.line(temp_cent_img_draw, b, f, (0, 0, 220), 2)

            # The triangle circles
            cv2.circle(temp_cent_img_draw, f, 4, (245, 76, 40), -1)
            cv2.circle(temp_cent_img_draw, m, 4, (10, 93, 201), -1)
            cv2.circle(temp_cent_img_draw, b, 4, (100, 210, 24), -1)

            # cv2.imshow("save_draw This img is returned", save_draw)
            cv2.imshow("Just for Triangle view", temp_cent_img_draw)

            # Making a graph with cv2, inside freeman func
            pad = 100
            border = 80

            posi_h = round(max(hi_list) + pad)
            neg_h = round(min(hi_list) + pad)

            chart_h = posi_h + neg_h

            chart_w = round(len(hi_list) + pad)

            chart = np.zeros((int(chart_h), int(chart_w), 3), dtype=np.uint8)

            for idx, i in enumerate(hi_list):
                # Chart line
                cv2.circle(chart, (idx + pad, chart.shape[0] - neg_h - int(round(i))), 1, (255, 255, 255))

            cv2.line(chart, (pad, chart.shape[0] - neg_h), (pad + len(hi_list), chart.shape[0] - neg_h), 180, 1)

            # y axis marker line
            cv2.line(chart, (pad - 30, chart.shape[0] - neg_h),
                     (pad - 20, chart.shape[0] - neg_h), (255, 255, 255), 1)

            cv2.putText(chart, "0", (pad - 60, chart.shape[0] - neg_h + 5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255, 255, 255))

            chart = cv2.copyMakeBorder(chart, border, border, border, border, cv2.BORDER_CONSTANT)

            title = "Triangle *Property* Along Contours"
            tit_txtfont = cv2.FONT_HERSHEY_COMPLEX_SMALL
            tit_txt_scale = 1
            tit_txt_Thik = None

            tit_txt_size = cv2.getTextSize(title, tit_txtfont, tit_txt_scale, tit_txt_Thik)

            tit_txt_xpos = (chart.shape[1] // 2) - (tit_txt_size[0][0] // 2)

            cv2.putText(chart, title, (tit_txt_xpos, 50), tit_txtfont, tit_txt_scale, (200, 200, 200))

            cv2.imshow("Prop Chart", chart)
            # time.sleep(sleep)
            cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    # # Possible new parameter or new func  this if  ****** makes list shorter so will speed up, but so far works
    # # about the same. Can remove any under threshold after this func also, or at end.
    # # hi_list = np.asarray(hi_list)
    # # print("hi list shape  ", hi_list.shape)
    # span = abs(max(hi_list) - min(hi_list)) * 0.1
    # # hi_list[hi_list > span or hi_list < -span] = hi_list
    #
    # # hi_list = [x for x in hi_list if hi_list > span or hi_list < -span]
    # # hi_list = np.where(hi_list > span)
    # # print("spannn ", span)
    # hi_list2 = []
    # for i in hi_list:
    #     if i > span or i < span * -1:
    #         hi_list2.append(i)
    #         # print("i   ", i)

    hi_list = np.asarray(hi_list)    # fixme, temporary, convert this func to work with array instead of list,
                                                # returning a list causes faults in chart matchers

    # print(hi_list2)
    # print("hi list shape after   ", hi_list2.shape)

    return save_draw, hi_list


# Part of template set-up
def chart_match_set_template(prop_list, contours, save_image_as, save_text_as, save_front_start):
    global x_pos, highlight_s, highlight_e, clear_last, grow, btn_y

    x_pos = 0

    def mouse_event(event, x, y, flags, param):
        # see mousecallback and window stuff in tkinter project
        global x_pos, highlight_s, highlight_e, clear_last, grow, btn_y
        # print(f"{x}, {y} - {flags} - {param} - event {event}")

        if event == cv2.EVENT_MOUSEMOVE:
            x_pos = x
        if event == cv2.EVENT_LBUTTONDOWN:  # not being used right now, can edit graph best in drawing program, but thickening graph
            highlight_s = x
            btn_y = y  # for btn's
        if event == cv2.EVENT_LBUTTONUP:
            highlight_e = x
            # highlight_s = 0           # There is a bug when clicking a lot on border, this didn't fix it
            # btn_y = y
        if event == cv2.EVENT_RBUTTONDOWN:
            clear_last = True
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                grow += 1
            else:
                if grow > 1:
                    grow -= 1

    cn = contours
    hi_lis = prop_list

    pad = 100
    border = 80
    pa_bor = pad + border
    highlight_s = 0
    highlight_e = 0
    highlight_list = []
    clear_last = False
    grow = 1
    chart_line_color = (255, 255, 255)
    chart_HL_color = (22, 244, 244)

    btn_pad = 20
    btn_y = -1

    st = 0

    template_PNG = save_image_as

    #xx <---- inter_graph
    posi_h = round(max(prop_list) + pad)
    neg_h = round(min(prop_list) + pad)


    chart_h = posi_h + neg_h

    # chart_h = max(posi_h, neg_h) * 2 + 0  # a bit of pad on height? #######   made worse here

    chart_w = round(len(prop_list) + pad)

    # Make sure the chart w, h are even, makes it easier to compare them later.
    if chart_h % 2 != 0:
        chart_h += 1
        neg_h += 1
    if chart_w % 2 != 0:
        chart_w += 1

    chart = np.zeros((int(chart_h), int(chart_w), 3), dtype=np.uint8)

    # print(chart.shape)
    cv2.line(chart, (pad, chart.shape[0] - neg_h), (pad + len(prop_list), chart.shape[0] - neg_h), 180, 1)

    for idx, i in enumerate(prop_list):
        # Chart line  Too low
        # cv2.circle(chart, (idx + pad, chart.shape[0] - neg_h - int(round(i))), 1,
        #            (255, 255, 255), -1)
        chart[chart.shape[0] - neg_h - int(round(i)), idx + pad] = 255    # Working fixed at curImg = curImg ...


    chart2 = cv2.copyMakeBorder(chart, border, border, border, border, cv2.BORDER_CONSTANT)

    title = f"Triangle Property Along Contours"
    tit_txtfont = cv2.FONT_HERSHEY_COMPLEX_SMALL
    tit_txt_scale = 1
    tit_txt_Thik = None

    tit_txt_size = cv2.getTextSize(title, tit_txtfont, tit_txt_scale, tit_txt_Thik)

    tit_txt_xpos = (chart2.shape[1] // 2) - (tit_txt_size[0][0] // 2)

    cv2.putText(chart2, title, (tit_txt_xpos, 50), tit_txtfont, tit_txt_scale, (200, 200, 200))

    # white by "0" y axis marker line
    cv2.line(chart2, (border + pad - 30, chart.shape[0] + border - neg_h),
             (border + pad - 20, chart.shape[0] + border - neg_h), (200, 200, 200), 1)

    # y axis text
    cv2.putText(chart2, "0", (border + 40, chart.shape[0] + border + 5 - neg_h), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (200, 200, 200))
    chart = chart2
    #xx ----->

    # xx Use Mouse callback to put values at certain point on graph like stock charts
    while True:

        time.sleep(0.002)

        chart3 = chart.copy()

        h_at_x = np.where(chart3[:, x_pos] == (255, 255, 255))

        h_at_x = np.asarray(h_at_x)[0]

        # print('h at x ', h_at_x)
        if len(h_at_x) != 0 and x_pos - pad - border >= 0:     # make sure it does not go into negative idx

            h_at = int(np.median(h_at_x))

            # <------ Highlight / Draw on graph
            if highlight_s > 0:
                hl_from = min(highlight_s, x_pos)
                hl_to = max(highlight_s, x_pos)

                # chart3[chart3[:, hl_from:hl_to] == (255, 255, 255)] = chart3[:, hl_from:hl_to] = (22, 210, 200)
                chart3[:, hl_from:hl_to] = np.where(chart3[:, hl_from:hl_to] == chart_line_color, chart_HL_color,
                                                    chart3[:, hl_from:hl_to])

                if highlight_e > 0:
                    highlight_s = 0          # s is L btn down
                    highlight_e = 0          # e is L btn up
                    highlight_list.append([hl_from, hl_to])

            if len(highlight_list) > 0:

                # Works, don't need clear_all, might anciently double right click
                if clear_last == True:
                    highlight_list.pop()
                    clear_last = False

                for i in highlight_list:
                    chart3[:, i[0]:i[1]] = np.where(chart3[:, i[0]:i[1]] == chart_line_color, (22, 244, 244),
                                                        chart3[:, i[0]:i[1]])

                where_yellow = np.where(chart3[pa_bor:, :] == chart_HL_color)
                where_yellow = np.asarray(where_yellow)
                # print(where_yellow)
                # print("where yellow shapee   ", where_yellow.shape)

                for idx, i in enumerate(where_yellow[0].tolist()):
                    cv2.circle(chart3, (where_yellow[1][idx], i + pa_bor), grow, chart_HL_color, -1)  # yellow
                    # cv2.circle(chart3, (where_yellow[1][idx], i + pa_bor), grow, (0, 153, 255), -1)  # orange

                # Stop inflating from going outside of bounds, now can get unaltered chart x value later
                chart3[:, :pad + border] = chart[:, :pad + border]
                chart3[:, chart3.shape[1] - border:] = chart[:, chart3.shape[1] - border:]
            # ------->

            if len(hi_lis) % 2 == 0:
                val = hi_lis[x_pos - pad - border]
            else:
                val = hi_lis[x_pos - pad - border - 1]    # incase the img shape was change, to make even num
            # print('            val ', val)

            hi_lis_idx = x_pos - pad - border

            # Chart ready for compare  curImg is current chart, teImg is template chart
            curImg = chart3.copy()
            curImg[:, :] = np.where((curImg[:, :] == chart_HL_color) | (curImg[:, :] == chart_line_color),
                                   curImg[:, :], 0)


            # Removing pad
            # curImg = curImg[pad:curImg.shape[0] - border, pad:curImg.shape[1]]   # Was the problem
            # curImg = curImg[pad + posi_h - neg_h - 20:curImg.shape[0] - border, pad:curImg.shape[1]]
            # Or, the same..      FIXED
            curImg = curImg[border + posi_h - neg_h:curImg.shape[0] - border, pad:curImg.shape[1]]  # use now

            # pad = 100
            # border = 80

            # print(f'posi h {posi_h}, neg h {neg_h}, h at {h_at}')


            view_curImg = curImg.copy()
            cv2.line(view_curImg, (20, view_curImg.shape[0] // 2), (view_curImg.shape[1] - 20, view_curImg.shape[0] // 2),
                     166, 1)
            cv2.imshow("curImg IN SET UP", view_curImg)   # For testing




            txt = f"Height {str(val)}, idx {hi_lis_idx}"
            txtfont = cv2.FONT_HERSHEY_COMPLEX_SMALL    # These are also used for btn
            txtfont_scale = 1
            txtThik = None

            txt_size = cv2.getTextSize(txt, txtfont, txtfont_scale, txtThik)  # [0][0] is width
            txt_w = txt_size[0][0]
            # print(txt_w)

            txt_xpos = x_pos - txt_w // 2

            if txt_xpos + txt_w > chart3.shape[1] - border:
                txt_xpos = chart3.shape[1] - border - txt_w

            if txt_xpos < border:
                txt_xpos = border

            cv2.putText(chart3, txt, (txt_xpos, h_at - 70), txtfont, txtfont_scale, (220, 210, 0), txtThik)

            # <------ Comparing graphs vs template graph, interface, and saving
            # A cv2 button

            exit_btn_txt = "Exit(q)"
            exit_btn_txt_size = cv2.getTextSize(exit_btn_txt, txtfont, txtfont_scale, txtThik)  # [0][0] is width
            exit_btn_txt_w = exit_btn_txt_size[0][0]
            exit_btn_txt_h = exit_btn_txt_size[0][1]

            save_btn_txt = "Save As Template"
            save_btn_txt_size = cv2.getTextSize(save_btn_txt, txtfont, txtfont_scale, txtThik)
            save_btn_txt_w = save_btn_txt_size[0][0]
            save_btn_txt_h = save_btn_txt_size[0][1]

            # TL = Top Left, (x, y)         img btn would be simpler
            tmp_btn_txt_TL = (chart3.shape[1] // 2 - exit_btn_txt_w // 2, chart3.shape[0] - 20)
            save_btn_txt_TL = (chart3.shape[1] // 2 - save_btn_txt_w // 2, chart3.shape[0] - 60)

            exit_rect_x_from = tmp_btn_txt_TL[0] - btn_pad
            exit_rect_x_to = tmp_btn_txt_TL[0] + exit_btn_txt_w + btn_pad
            exit_rect_y_from = tmp_btn_txt_TL[1] - btn_pad
            exit_rect_y_to = tmp_btn_txt_TL[1] + exit_btn_txt_h

            save_rect_x_from = save_btn_txt_TL[0] - btn_pad
            save_rect_x_to = save_btn_txt_TL[0] + save_btn_txt_w + btn_pad
            save_rect_y_from = save_btn_txt_TL[1] - btn_pad
            save_rect_y_to = save_btn_txt_TL[1] + save_btn_txt_h

            if time.time() < st:
                s_btn_txt_color = (65, 65, 65)
                s_btn_rec_color = (65, 65, 65)
            else:
                s_btn_txt_color = (220, 210, 0)
                s_btn_rec_color = (155, 30, 0)

            cv2.putText(chart3, exit_btn_txt, tmp_btn_txt_TL, txtfont, txtfont_scale, (220, 210, 0), txtThik)
            cv2.putText(chart3, save_btn_txt, save_btn_txt_TL, txtfont, txtfont_scale, s_btn_txt_color, txtThik)

            cv2.rectangle(chart3, (exit_rect_x_from, exit_rect_y_from), (exit_rect_x_to, exit_rect_y_to), (155, 30, 0))
            cv2.rectangle(chart3, (save_rect_x_from, save_rect_y_from), (save_rect_x_to, save_rect_y_to), s_btn_rec_color)

    # --------------------------
            # Compare To Template Button pressed  -- Removed was old way, see TestMyCont.py

                        # # Visualize the Zones
                        # # cv2.circle(sub, (border + zone_len * 4, 80), 5, (255, 255, 78), -1)
                        # cv2.line(sub, (bor, 80), (bor + zone_len, 80), (255, 255, 78), 1)   # only correct when width resized above
                        # cv2.line(sub, (bor + zone_len, 80), (bor + zone_len * 2, 80), (255, 0, 78), 1)
                        # cv2.line(sub, (bor + zone_len * 2, 80), (bor + zone_len * 3, 80), (0, 255, 78), 1)
                        # cv2.line(sub, (bor + zone_len * 3, 80), (bor + zone_len * 4, 80), (55, 0, 255), 1)
                        # ------->

            if highlight_s > exit_rect_x_from and highlight_s < exit_rect_x_to and btn_y > exit_rect_y_from and \
                    btn_y < exit_rect_y_to:
                break

            # Save Button Pressed, curImg will come back as teImg
            if highlight_s > save_rect_x_from and highlight_s < save_rect_x_to and btn_y > save_rect_y_from and \
                    btn_y < save_rect_y_to and time.time() > st:

                # highlight_s = 1
                btn_y = 0

                st = time.time() + 1.3

                # Get the prop list aka "hi_list" length, max and min, and save to a txt file, for later scaling the
                # sample img.   Added len of cn, minEclosingCirle radius, contour area, maybe add front_start too, but
                # is not in this func.

                cont_area = cv2.contourArea(cn[0])
                (x, y), radius = cv2.minEnclosingCircle(cn[0])

                prop_len = len(hi_lis)
                prop_min = min(hi_lis)
                prop_max = max(hi_lis)

                ps = open(save_text_as, "w+")
                ps.write(str(prop_len) + "\n")
                ps.write(str(prop_min) + "\n")
                ps.write(str(prop_max) + "\n")
                ps.write(str(len(cn[0])) + "\n")
                ps.write(str(cont_area) + "\n")
                ps.write(str(radius) + "\n")
                ps.write(str(save_front_start))
                ps.close()

                # Saves the current chart as template, will overwrite
                curImg[:, :] = np.where(curImg[:, :] != [0, 0, 0], [255, 255, 255], [0, 0, 0])
                cv2.imwrite(template_PNG, curImg)
        # -------->

            # Mouse following line on chart, must be after "Comparing graphs vs template graph"
            cv2.line(chart3, (x_pos, 0), (x_pos, chart.shape[0]), (20, 0, 220), 1)

            # Moving circle on chart
            cv2.circle(chart3, (x_pos, h_at), 5, (220, 210, 0), -1)

        cv2.imshow("Shape Data Chart - drag over, scroll mouse wheel", chart3)
        cv2.setMouseCallback("Shape Data Chart - drag over, scroll mouse wheel", mouse_event)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Part of template set-up
def chart_match_set_template_overlay(prop_lists_aligned, contours, save_image_as, save_text_as, save_front_start):
    '''
    Same as chart_match_set_template, but takes in prop_lists_aligned (and cut down the same length) instead of
    prop_lists.
    Contours can be the contours of any list in the array, because the inner arrays are the same image just rotated.
    Conts are only used for minEnclosingCircle, and contourArea.
    '''

    global x_pos, highlight_s, highlight_e, clear_last, grow, btn_y

    x_pos = 0

    def mouse_event(event, x, y, flags, param):
        # see mousecallback and window stuff in tkinter project
        global x_pos, highlight_s, highlight_e, clear_last, grow, btn_y
        # print(f"{x}, {y} - {flags} - {param} - event {event}")

        if event == cv2.EVENT_MOUSEMOVE:
            x_pos = x
        if event == cv2.EVENT_LBUTTONDOWN:  # not being used right now, can edit graph best in drawing program, but thickening graph
            highlight_s = x
            btn_y = y  # for btn's
        if event == cv2.EVENT_LBUTTONUP:
            highlight_e = x
            # highlight_s = 0           # There is a bug when clicking a lot on border, this didn't fix it
            # btn_y = y
        if event == cv2.EVENT_RBUTTONDOWN:
            clear_last = True
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                grow += 1
            else:
                if grow > 1:
                    grow -= 1

    cn = contours
    hi_lis = prop_lists_aligned

    pad = 100
    border = 80
    pa_bor = pad + border
    highlight_s = 0
    highlight_e = 0
    highlight_list = []
    clear_last = False
    grow = 1
    chart_line_color = (255, 255, 255)
    chart_HL_color = (22, 244, 244)

    btn_pad = 20
    btn_y = -1

    st = 0

    template_PNG = save_image_as

    # xx <---- inter_graph
    # posi_h = round(max(prop_dict["0"]) + pad)
    maxes = []
    for i in prop_lists_aligned:
        maxes.append(max(i))
    posi_h = round(max(maxes) + pad)

    mines = []
    for i in prop_lists_aligned:
        mines.append(min(i))
    neg_h = round(min(mines) + pad)

    # neg_h = round(min(prop_dict["0"]) + pad)

    chart_h = posi_h + neg_h

    # chart_h = max(posi_h, neg_h) * 2 + 0  # a bit of pad on height? #######   made worse here

    chart_w = round(len(prop_lists_aligned[0]) + pad)

    # Make sure the chart w, h are even, makes it easier to compare them later.
    if chart_h % 2 != 0:
        chart_h += 1
        neg_h += 1
    if chart_w % 2 != 0:
        chart_w += 1

    chart = np.zeros((int(chart_h), int(chart_w), 3), dtype=np.uint8)

    # print(chart.shape)
    cv2.line(chart, (pad, chart.shape[0] - neg_h), (pad + len(prop_lists_aligned[0]), chart.shape[0] - neg_h), 180, 1)

    for idx1, i in enumerate(prop_lists_aligned):
        for idx, j in enumerate(i):
            # Chart line  Too low
            # cv2.circle(chart, (idx + pad, chart.shape[0] - neg_h - int(round(i))), 1,
            #            (255, 255, 255), -1)
            chart[chart.shape[0] - neg_h - int(round(j)), idx + pad] = 255  # Working fixed at curImg = curImg ...

    chart2 = cv2.copyMakeBorder(chart, border, border, border, border, cv2.BORDER_CONSTANT)

    title = f"Triangle Property Along Contours"
    tit_txtfont = cv2.FONT_HERSHEY_COMPLEX_SMALL
    tit_txt_scale = 1
    tit_txt_Thik = None

    tit_txt_size = cv2.getTextSize(title, tit_txtfont, tit_txt_scale, tit_txt_Thik)

    tit_txt_xpos = (chart2.shape[1] // 2) - (tit_txt_size[0][0] // 2)

    cv2.putText(chart2, title, (tit_txt_xpos, 50), tit_txtfont, tit_txt_scale, (200, 200, 200))

    # white by "0" y axis marker line
    cv2.line(chart2, (border + pad - 30, chart.shape[0] + border - neg_h),
             (border + pad - 20, chart.shape[0] + border - neg_h), (200, 200, 200), 1)

    # y axis text
    cv2.putText(chart2, "0", (border + 40, chart.shape[0] + border + 5 - neg_h), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (200, 200, 200))
    chart = chart2
    # xx ----->

    # xx Use Mouse callback to put values at certain point on graph like stock charts
    while True:

        time.sleep(0.002)

        chart3 = chart.copy()

        h_at_x = np.where(chart3[:, x_pos] == (255, 255, 255))

        h_at_x = np.asarray(h_at_x)[0]

        # print('h at x ', h_at_x)
        if len(h_at_x) != 0 and x_pos - pad - border >= 0:  # make sure it does not go into negative idx

            h_at = int(np.median(h_at_x))

            # <------ Highlight / Draw on graph
            if highlight_s > 0:
                hl_from = min(highlight_s, x_pos)
                hl_to = max(highlight_s, x_pos)

                # chart3[chart3[:, hl_from:hl_to] == (255, 255, 255)] = chart3[:, hl_from:hl_to] = (22, 210, 200)
                chart3[:, hl_from:hl_to] = np.where(chart3[:, hl_from:hl_to] == chart_line_color, chart_HL_color,
                                                    chart3[:, hl_from:hl_to])

                if highlight_e > 0:
                    highlight_s = 0  # s is L btn down
                    highlight_e = 0  # e is L btn up
                    highlight_list.append([hl_from, hl_to])

            if len(highlight_list) > 0:

                # Works, don't need clear_all, might anciently double right click
                if clear_last == True:
                    highlight_list.pop()
                    clear_last = False

                for i in highlight_list:
                    chart3[:, i[0]:i[1]] = np.where(chart3[:, i[0]:i[1]] == chart_line_color, (22, 244, 244),
                                                    chart3[:, i[0]:i[1]])

                where_yellow = np.where(chart3[pa_bor:, :] == chart_HL_color)
                where_yellow = np.asarray(where_yellow)
                # print(where_yellow)
                # print("where yellow shapee   ", where_yellow.shape)

                for idx, i in enumerate(where_yellow[0].tolist()):
                    cv2.circle(chart3, (where_yellow[1][idx], i + pa_bor), grow, chart_HL_color, -1)  # yellow
                    # cv2.circle(chart3, (where_yellow[1][idx], i + pa_bor), grow, (0, 153, 255), -1)  # orange

                # Stop inflating from going outside of bounds, now can get unaltered chart x value later
                chart3[:, :pad + border] = chart[:, :pad + border]
                chart3[:, chart3.shape[1] - border:] = chart[:, chart3.shape[1] - border:]
            # ------->

            if len([hi_lis[0]]) % 2 == 0:
                val = hi_lis[0][x_pos - pad - border]
            else:
                val = hi_lis[0][x_pos - pad - border - 1]  # incase the img shape was change, to make even num
            # print('            val ', val)

            hi_lis_idx = x_pos - pad - border

            # Chart ready for compare  curImg is current chart, teImg is template chart
            curImg = chart3.copy()
            curImg[:, :] = np.where((curImg[:, :] == chart_HL_color) | (curImg[:, :] == chart_line_color),
                                    curImg[:, :], 0)

            # Removing pad
            # curImg = curImg[pad:curImg.shape[0] - border, pad:curImg.shape[1]]   # Was the problem
            # curImg = curImg[pad + posi_h - neg_h - 20:curImg.shape[0] - border, pad:curImg.shape[1]]
            # Or, the same..      FIXED
            curImg = curImg[border + posi_h - neg_h:curImg.shape[0] - border, pad:curImg.shape[1]]  # use now

            # pad = 100
            # border = 80

            # print(f'posi h {posi_h}, neg h {neg_h}, h at {h_at}')

            view_curImg = curImg.copy()
            cv2.line(view_curImg, (20, view_curImg.shape[0] // 2),
                     (view_curImg.shape[1] - 20, view_curImg.shape[0] // 2),
                     166, 1)
            cv2.imshow("curImg IN SET UP", view_curImg)  # For testing

            txt = f"Height {str(val)}, idx {hi_lis_idx}"
            txtfont = cv2.FONT_HERSHEY_COMPLEX_SMALL  # These are also used for btn
            txtfont_scale = 1
            txtThik = None

            txt_size = cv2.getTextSize(txt, txtfont, txtfont_scale, txtThik)  # [0][0] is width
            txt_w = txt_size[0][0]
            # print(txt_w)

            txt_xpos = x_pos - txt_w // 2

            if txt_xpos + txt_w > chart3.shape[1] - border:
                txt_xpos = chart3.shape[1] - border - txt_w

            if txt_xpos < border:
                txt_xpos = border

            cv2.putText(chart3, txt, (txt_xpos, h_at - 70), txtfont, txtfont_scale, (220, 210, 0), txtThik)

            # <------ Comparing graphs vs template graph, interface, and saving
            # A cv2 button

            exit_btn_txt = "Exit(q)"
            exit_btn_txt_size = cv2.getTextSize(exit_btn_txt, txtfont, txtfont_scale, txtThik)  # [0][0] is width
            exit_btn_txt_w = exit_btn_txt_size[0][0]
            exit_btn_txt_h = exit_btn_txt_size[0][1]

            save_btn_txt = "Save As Template"
            save_btn_txt_size = cv2.getTextSize(save_btn_txt, txtfont, txtfont_scale, txtThik)
            save_btn_txt_w = save_btn_txt_size[0][0]
            save_btn_txt_h = save_btn_txt_size[0][1]

            # TL = Top Left, (x, y)         img btn would be simpler
            tmp_btn_txt_TL = (chart3.shape[1] // 2 - exit_btn_txt_w // 2, chart3.shape[0] - 20)
            save_btn_txt_TL = (chart3.shape[1] // 2 - save_btn_txt_w // 2, chart3.shape[0] - 60)

            exit_rect_x_from = tmp_btn_txt_TL[0] - btn_pad
            exit_rect_x_to = tmp_btn_txt_TL[0] + exit_btn_txt_w + btn_pad
            exit_rect_y_from = tmp_btn_txt_TL[1] - btn_pad
            exit_rect_y_to = tmp_btn_txt_TL[1] + exit_btn_txt_h

            save_rect_x_from = save_btn_txt_TL[0] - btn_pad
            save_rect_x_to = save_btn_txt_TL[0] + save_btn_txt_w + btn_pad
            save_rect_y_from = save_btn_txt_TL[1] - btn_pad
            save_rect_y_to = save_btn_txt_TL[1] + save_btn_txt_h

            if time.time() < st:
                s_btn_txt_color = (65, 65, 65)
                s_btn_rec_color = (65, 65, 65)
            else:
                s_btn_txt_color = (220, 210, 0)
                s_btn_rec_color = (155, 30, 0)

            cv2.putText(chart3, exit_btn_txt, tmp_btn_txt_TL, txtfont, txtfont_scale, (220, 210, 0), txtThik)
            cv2.putText(chart3, save_btn_txt, save_btn_txt_TL, txtfont, txtfont_scale, s_btn_txt_color, txtThik)

            cv2.rectangle(chart3, (exit_rect_x_from, exit_rect_y_from), (exit_rect_x_to, exit_rect_y_to), (155, 30, 0))
            cv2.rectangle(chart3, (save_rect_x_from, save_rect_y_from), (save_rect_x_to, save_rect_y_to),
                          s_btn_rec_color)

            # --------------------------
            # Compare To Template Button pressed  -- Removed was old way, see TestMyCont.py

            # # Visualize the Zones
            # # cv2.circle(sub, (border + zone_len * 4, 80), 5, (255, 255, 78), -1)
            # cv2.line(sub, (bor, 80), (bor + zone_len, 80), (255, 255, 78), 1)   # only correct when width resized above
            # cv2.line(sub, (bor + zone_len, 80), (bor + zone_len * 2, 80), (255, 0, 78), 1)
            # cv2.line(sub, (bor + zone_len * 2, 80), (bor + zone_len * 3, 80), (0, 255, 78), 1)
            # cv2.line(sub, (bor + zone_len * 3, 80), (bor + zone_len * 4, 80), (55, 0, 255), 1)
            # ------->

            if highlight_s > exit_rect_x_from and highlight_s < exit_rect_x_to and btn_y > exit_rect_y_from and \
                    btn_y < exit_rect_y_to:
                break

            # Save Button Pressed, curImg will come back as teImg
            if highlight_s > save_rect_x_from and highlight_s < save_rect_x_to and btn_y > save_rect_y_from and \
                    btn_y < save_rect_y_to and time.time() > st:

                # highlight_s = 1
                btn_y = 0

                st = time.time() + 1.3

                # Get the prop list aka "hi_list" length, max and min, and save to a txt file, for later scaling the
                # sample img.   Added len of cn, minEclosingCirle radius, contour area, maybe add front_start too, but
                # is not in this func.

                cont_area = cv2.contourArea(cn[0])
                (x, y), radius = cv2.minEnclosingCircle(cn[0])

                prop_len = len(hi_lis[0])
                # prop_min = min(hi_lis)
                mins = []
                for i in prop_lists_aligned:
                    mins.append(min(i))
                prop_min = min(mins)

                # prop_max = max(hi_lis)
                maxs = []
                for i in prop_lists_aligned:
                    maxs.append(max(i))
                prop_max = max(maxs)

                ps = open(save_text_as, "w+")
                ps.write(str(prop_len) + "\n")
                ps.write(str(prop_min) + "\n")
                ps.write(str(prop_max) + "\n")
                ps.write(str(len(cn[0])) + "\n")
                ps.write(str(cont_area) + "\n")
                ps.write(str(radius) + "\n")
                ps.write(str(save_front_start))
                ps.close()

                # Saves the current chart as template, will overwrite
                curImg[:, :] = np.where(curImg[:, :] != [0, 0, 0], [255, 255, 255], [0, 0, 0])
                cv2.imwrite(template_PNG, curImg)
            # -------->

            # Mouse following line on chart, must be after "Comparing graphs vs template graph"
            cv2.line(chart3, (x_pos, 0), (x_pos, chart.shape[0]), (20, 0, 220), 1)

            # Moving circle on chart
            cv2.circle(chart3, (x_pos, h_at), 5, (220, 210, 0), -1)

        cv2.imshow("Shape Data Chart - drag over, scroll mouse wheel", chart3)
        cv2.setMouseCallback("Shape Data Chart - drag over, scroll mouse wheel", mouse_event)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



def prop_list_match(prop_list, temp_prop_list, template_text, roll):

    pg = open(template_text, "r")      # temp_prop_list should come from file, not this one though
    te_data = pg.read().splitlines()     # should not be done inside here because is not sample
    pg.close()
    temp_posi_h = float(te_data[2])
    posi_h = max(prop_list)          # current sample list

    #  #xx either an if statement here to find max shapes of both charts/imgs, or require that templates are always larger


    span_ratio = temp_posi_h / posi_h
    # if prop_list.size > temp_prop_list.size:
    #     temp_prop_list = fan_delete_resize(prop_list, temp_prop_list)
    # else:
    temp_prop_list = fan_delete_resize(temp_prop_list, prop_list)

    samp_prop_list = prop_list * span_ratio

    # make another func that finds a template feature in the sample

    best_cos = -100

    # print(type(samp_prop_list))

    for i in range(samp_prop_list.size // roll):
        # fixme faults when small arrays, or contours on edge of image
        cos_sim = np.dot(temp_prop_list, samp_prop_list) / (np.linalg.norm(temp_prop_list) * np.linalg.norm(samp_prop_list))
        best_cos = max(best_cos, cos_sim)
        samp_prop_list = np.roll(samp_prop_list, roll)

    best_cos *= 100

    # print(type(samp_prop_list))
    # print("temp prop list type: ", type(temp_prop_list))

    return best_cos


def prop_list_match_best_roll(prop_list, temp_prop_list, template_text, roll):
    '''
    Same as prop_list_match(), but just returns prop_list rolled to the best match,
    and needs arrays of same size.
    '''

    pg = open(template_text, "r")      # temp_prop_list should come from file, not this one though
    te_data = pg.read().splitlines()     # should not be done inside here because is not sample
    pg.close()
    temp_posi_h = float(te_data[2])
    posi_h = max(prop_list)          # current sample list

    #  #xx either an if statement here to find max shapes of both charts/imgs, or require that templates are always larger


    span_ratio = temp_posi_h / posi_h

    # print("span ratio ", span_ratio)
    # print(temp_posi_h)
    # print(posi_h)
    # print(type(span_ratio))
    # print(type(prop_list))
    # if prop_list.size > temp_prop_list.size:
    #     temp_prop_list = fan_delete_resize(prop_list, temp_prop_list)
    # else:
    # temp_prop_list = fan_delete_resize(temp_prop_list, prop_list)   don't need to downsize here, done before

    samp_prop_list = prop_list * span_ratio   # faults here make sure prop_list is a np.array

    # make another func that finds a template feature in the sample


    best_cos = -100
    best_cos_save = 0
    best_idx = 0
    # print(type(samp_prop_list))

    for idx, i in enumerate(range(samp_prop_list.size // roll)):
        cos_sim = np.dot(temp_prop_list, samp_prop_list) / (np.linalg.norm(temp_prop_list) * np.linalg.norm(samp_prop_list))
        best_cos = max(best_cos, cos_sim)
        if best_cos != best_cos_save:
            best_idx = idx
        best_cos_save = best_cos
        samp_prop_list = np.roll(samp_prop_list, roll)
        # print("samp: ", samp_prop_list[:5])
        # print("b r:  ", best_roll[:5])

    # best_cqos *= 100
    best_roll_amt = best_idx * roll
    # print("++++++++++++++++++++++++++++++++++++++++++")

    # print(type(samp_prop_list))
    # print("temp prop list type: ", type(temp_prop_list))

    return best_roll_amt


def chart_match(prop_list, template_image, template_text, roll=4, warp_match_x=True, warp_match_y=True, zones=0,
                 view=False):

    template_PNG = template_image
    chart_line_color = (255, 255, 255)

    temp_img = cv2.imread(template_PNG, 0)
    temp_img = temp_img[:, 80:temp_img.shape[1] - 80]     #xx taking off border on x, chart should be saved without it
    # print("temp img shape ", temp_img.shape)  # (188, 558)



    # posi_h = round(max(prop_list))
    # neg_h = round(min(prop_list))        # might be wrongly used in inter_graph is a negative number
    #
    # print("prop list posi h ", posi_h)
    # print(" and neg h  ", neg_h)
    #
    #
    # chart_h = max(posi_h, neg_h) * 2 + 0   # a bit of pad on height?
    #
    # chart_w = round(len(prop_list))    # this area not being used for now
    #
    # # Make sure the chart w, h are even, makes it easier to compare them later.
    # if chart_h % 2 != 0:
    #     chart_h += 1           # actually all chart imgs h should be odd numbers, so 0 is right in center
    #     neg_h += 1
    # if chart_w % 2 != 0:
    #     chart_w += 1

     #xx either an if statement here to find max shapes of both charts/imgs, or require that templates are always larger
    # fixme


    cur_img = np.zeros_like(temp_img)


    # # -- for testing --
    # start_temp = temp_img.copy()
    # start_cur = cur_img.copy()
    #
    # cv2.line(start_temp, (20, start_temp.shape[0] // 2), (start_temp.shape[1] - 20, start_temp.shape[0] // 2), 166, 1)
    # cv2.line(start_cur, (20, start_cur.shape[0] // 2), (start_cur.shape[1] - 20, start_cur.shape[0] // 2), 166, 1)
    #
    # cv2.imshow("temp img start of chart match", start_temp)
    # cv2.imshow("cur img start of chart match", start_cur)

    thickness = 1   # can always be
    idx = 0

    if warp_match_x == True and warp_match_y == True:
        mov = temp_img.shape[1] / len(prop_list)
        mv_acc = 0

        # for warp y, i * ratio of abs of max and min, for each list
        pg = open(template_text, "r")
        te_data = pg.read().splitlines()
        pg.close()
        temp_posi_h = float(te_data[2])
        temp_neg_h = float(te_data[1])
        posi_h = max(prop_list)          # current sample list
        neg_h = min(prop_list)


        # span_ratio = temp_h_span / cur_h_span   # working but might base ratio on something else, or make option
        span_ratio = temp_posi_h / posi_h
        # span_ratio = 1

        # print(len(prop_list), "  mov is  ", mov)
        for i in prop_list:
            # Chart line
            # cv2.circle(cur_img, (idx, cur_img.shape[0] // 2 - int(round(i * span_ratio)) + 4), thickness, (255, 255, 255), -1)
            cur_img[cur_img.shape[0] // 2 - int(round(i * span_ratio)), idx] = 1  # maybe use works about same
            # 1px draw work very good on alphabet, still works on f-mix, is likely more precise

            #fixme found out that different rotated templates zero lines are not matching up properly **  FIXED in set up

            mv = int(mov)
            # mv = round(mov)
            mv_acc = mov - mv + mv_acc
            if mv_acc >= 1:
                mvover = int(mv_acc)
                # mvover = round(mv_acc)
                idx += mvover
                mv_acc = mv_acc - mvover
            idx += mv

    elif warp_match_x == True:
        mov = temp_img.shape[1] / len(prop_list)
        mv_acc = 0
        # print(len(prop_list), "  mov is  ", mov)
        for i in prop_list:
            # Chart line
            # cv2.circle(cur_img, (idx, cur_img.shape[0] // 2 - int(round(i))), thickness, (255, 255, 255), -1)
            cur_img[cur_img.shape[0] // 2 - int(round(i)), idx] = 1

            mv = int(mov)
            mv_acc = mov - mv + mv_acc
            if mv_acc >= 1:
                mvover = int(mv_acc)
                idx += mvover
                mv_acc = mv_acc - mvover
            idx += mv

    elif warp_match_y == True:

        # for warp y, i * ratio of abs of max and min, for each list
        pg = open("template_data.txt", "r")
        te_data = pg.read().splitlines()
        pg.close()
        temp_posi_h = float(te_data[2])
        temp_neg_h = float(te_data[1])
        posi_h = max(prop_list)          # current sample list
        neg_h = min(prop_list)

        temp_h_span = abs(temp_posi_h - temp_neg_h)
        cur_h_span = abs(posi_h - neg_h)

        span_ratio = temp_h_span / cur_h_span   # working but might base ratio on something else, or make option

        for idx, i in enumerate(prop_list):
            # Chart line
            # cv2.circle(cur_img, (idx, cur_img.shape[0] // 2 - int(round(i * span_ratio))), thickness, (255, 255, 255), -1)
            cur_img[cur_img.shape[0] // 2 - int(round(i * span_ratio)), idx] = 1

    else:
        for idx, i in enumerate(prop_list):
        # for i in prop_list:
            # Chart line
            # cv2.circle(cur_img, (idx, cur_img.shape[0] // 2 - int(round(i))), thickness, (255, 255, 255), -1)
            cur_img[cur_img.shape[0] // 2 - int(round(i)), idx] = 1


    if view == True:
        cur_view = cur_img * 255
        # cur_view = cur_img.copy()   # remove and turn above back on
        # Zero y axis marker line for testing
        # cv2.line(cur_view, (20, cur_view.shape[0] // 2), (cur_view.shape[1] - 20, cur_view.shape[0] // 2), 166, 1)
        cv2.imshow("Template Prop List Chart", temp_img)           #####
        cv2.imshow("Sample Prop List Chart", cur_view)   ######
        # cv2.waitKey(0)

    temp_img[temp_img == 255] = 1       # go back to this

    # # For many charts combined in Inkscape         not working
    # # temp_img[temp_img > 0] = 1
    # wc = np.where(temp_img > 0)
    # wc = np.asarray(wc)
    # print(wc.shape)
    # print('0000------', wc[0][:20])
    # print(wc[1][:20])
    # for idx, i in enumerate(wc[0]):  # draws it pritty filled in but not good results
    #     cv2.circle(temp_img, (wc[1][idx], wc[0][idx]), 1, 122, -1)
    #
    # # view it
    # tempTest = temp_img.copy()
    # # tempTest[tempTest > 0] = 255
    # cv2.imshow("tempTest", tempTest)
    # cv2.waitKey(0)


    # cv2.imshow("temp imghhh", temp_img)

    while True:

        count = 0   # Option? or add along with zone if over 0

        acc_list = []
        zone_1_list = []
        zone_2_list = []
        zone_3_list = []
        zone_4_list = []


# Optional ---- # Fine tune with blur and dialate?
#         temp_img = cv2.dilate(temp_img, (7, 7), iterations=5)  # like grow but probably worse
                # n_teImg = cv2.medianBlur(n_teImg, 15)
        # temp_img = cv2.GaussianBlur(temp_img, (7, 7), 37)  # gaussian then median blur seems to improve
        # xx Option    medianBlur makes worse on chart match2 - Try different grow values
        # temp_img = cv2.medianBlur(temp_img, 17)               # adds about 0.02s to time, can be done to saved temp

        # curImg = cv2.dilate(curImg, (7, 7))  # like grow but probably worse
                # curImg = cv2.medianBlur(curImg, 15)
                # curImg = cv2.GaussianBlur(curImg, (7, 7), 11)


        # Downsize Imgs for speed up -- cuts time in half, but greatly diminishes accuracy
        # temp_img = cvt.resize(temp_img, temp_img.shape[1] // 2)
        # cur_img = cvt.resize(cur_img, cur_img.shape[1] // 2)

                # Now to roll curImg over teImg and diff or subtract, this could be a func
                # for i in range(teImg.shape[1] // rol):  # old over roll way
        for i in range((temp_img.shape[1]) // roll):  # changed now only goes over one full time
        # plus some to

# Optional?  Remove middle of y of curImg or n_curImg below, zero line on chart is not always half way down
            # xx Option    Getting good result with +-5 px, and median blur off, gap of 9, chart_match, same
            # xx Better to add white on template img in drawing program, try various blurred black zero lines.
                # with +- 8 gap is 8% pts,   .. better without this on full alphabet and grow 1
            # cur_img[cur_img.shape[0] // 2 - 5:cur_img.shape[0] // 2 + 5, :] = 0  # improves gap by 1% on F imgs
                                                                            # gap from 1% to 5% on F and alhpabet
            # # has to be inside, figure out, use simpleir chart imgs
            # n_curImg[n_curImg.shape[0] // 2 - curImg.shape[0] // 2:n_curImg.shape[0] // 2 + curImg.shape[0] // 2,
            #         n_curImg.shape[1] // 2 - curImg.shape[1] // 2:n_curImg.shape[1] // 2 + curImg.shape[1] // 2] \
            #     = curImg

            # n_curImg[n_curImg.shape[0] // 2 - 2:n_curImg.shape[0] // 2 + 2, :] = 0

            # cv2.imshow("n curImg in chart match", n_curImg)
            # cv2.imshow("curImg in chart match", curImg)

            # cv2.imshow('cur_img in chart match2', cur_img)     #####
            # cv2.imshow('temp_img in chart match2', temp_img)    #####
            # cv2.waitKey(0)

            sub = cv2.subtract(cur_img, temp_img)
            # sub = cv2.absdiff(cur_img, temp_img)

            # print("n teImg shape -----  ", n_teImg.shape)
            # print("n curImg shape ----  ", n_curImg.shape)

            # sub = cv2.addWeighted(n_curImg, 0.5, n_teImg, 0.5, 0)

            sub_sum = np.sum(sub)
            px_out = sub_sum #/ 255
            # px_out = sub_sum   # for cur_img drawn with value of 1 instead of 255

            acc_list.append([count, px_out])

            # if zones == 4:
            #     # <------ Feature Zone Matching, sum up different zones of the chart, get the best match for that
            #     # zone, average them together, and find their order.
            #     bor = (sub.shape[1] - curImg.shape[1]) // 2
            #
            #     zone_len = curImg.shape[1] // 4    # going with 4 zones, make Optional
            #     zone_1 = np.sum(sub[:, bor:bor + zone_len]) / 255
            #     zone_2 = np.sum(sub[:, bor + zone_len: bor + zone_len * 2]) / 255
            #     zone_3 = np.sum(sub[:, bor + zone_len * 2: bor + zone_len * 3]) / 255
            #     zone_4 = np.sum(sub[:, bor + zone_len * 3: bor + zone_len * 4]) / 255
            #
            #     zone_1_list.append([count, zone_1])
            #     zone_2_list.append([count, zone_2])
            #     zone_3_list.append([count, zone_3])
            #     zone_4_list.append([count, zone_4])


                # # Visualize the Zones
                # # cv2.circle(sub, (border + zone_len * 4, 80), 5, (255, 255, 78), -1)
                # cv2.line(sub, (bor, 80), (bor + zone_len, 80), (255, 255, 78), 1)   # only correct when width resized above
                # cv2.line(sub, (bor + zone_len, 80), (bor + zone_len * 2, 80), (255, 0, 78), 1)
                # cv2.line(sub, (bor + zone_len * 2, 80), (bor + zone_len * 3, 80), (0, 255, 78), 1)
                # cv2.line(sub, (bor + zone_len * 3, 80), (bor + zone_len * 4, 80), (55, 0, 255), 1)
                # ------->


            # print(f"{count}. Pixels out of bounds: {px_out}")
            cur_img = np.roll(cur_img, roll, axis=1)   # old way works

            # curImg = np.roll(n_curImg, rol, axis=1)
            # count += 1            #$$$$$$$$$$$$$$$$$$$$ not using for now

            if view == True:
                view_sub = sub * 255     # turn back on
                # Zero y axis marker line for testing
                # cv2.line(view_sub, (20, view_sub.shape[0] // 2), (view_sub.shape[1] - 20, view_sub.shape[0] // 2), 166, 1)
                cv2.imshow("Subtracting In Chart Match ", view_sub)     #########   # For some reason absdiff is accurate, but sub is not
                # time.sleep(sleep)
                cv2.waitKey(0)                                                        # may 5 3:30pm

            # if clear_last == True:
            #     clear_last = False
            #     count = 0
            #     break

            # print(np.sum(curImg))
            # print(np.sum(n_curImg))

            # if zones == 4:
            #     cur_img_sum = np.sum(cur_img / 255)  # // vs / actually speeds up loop on F-mixed by 0.02s
            #     # cur_img_sum = np.sum(cur_img)   # for cur_img drawn with values of 1 instead of 255

            #     best_acc_zone1 = (sorted(zone_1_list, key=lambda x: x[1]))
            #     best_acc_zone2 = (sorted(zone_2_list, key=lambda x: x[1]))
            #     best_acc_zone3 = (sorted(zone_3_list, key=lambda x: x[1]))
            #     best_acc_zone4 = (sorted(zone_4_list, key=lambda x: x[1]))
            #
            #     # curImg_sum isn't the best to use here, and dividing it by 4 wouldn't be perfect either
            #     # If the "shift stages" are in order low to high for high to low, it's more likely the imgs are the same.
            #     #   The imge features are in the same order along the contours.
            #     zone_sum = curImg_sum // 4
            #     zone1_overall = round(100 - best_acc_zone1[0][1] / zone_sum * 100, 2)
            #     # print(
            #     #     f"IN ZONE 1, the Template graph has covered {zone1_overall}% of the samples graphs pixels, "
            #     #     f"// 4, at the best point. At shift stage {best_acc_zone1[0][0]}")
            #     zone2_overall = round(100 - best_acc_zone2[0][1] / zone_sum * 100, 2)
            #     # print(
            #     #     f"IN ZONE 2, the Template graph has covered {zone2_overall}% of the samples graphs pixels, "
            #     #     f"// 4, at the best point. At shift stage {best_acc_zone2[0][0]}")
            #     zone3_overall = round(100 - best_acc_zone3[0][1] / zone_sum * 100, 2)
            #     # print(
            #     #     f"IN ZONE 3, the Template graph has covered {zone3_overall}% of the samples graphs pixels, "
            #     #     f"// 4, at the best point. At shift stage {best_acc_zone3[0][0]}")
            #     zone4_overall = round(100 - best_acc_zone4[0][1] / zone_sum * 100, 2)
            #     # print(
            #     #     f"IN ZONE 4, the Template graph has covered {zone4_overall}% of the samples graphs pixels, "
            #     #     f"// 4, at the best point. At shift stage {best_acc_zone4[0][0]}")


        cur_img_sum = np.sum(cur_img) #/ 255)  # // vs / actually speeds up loop on F-mixed by 0.02s
        # cur_img_sum = np.sum(cur_img)   # for cur_img drawn with values of 1 instead of 255

        best_acc = (sorted(acc_list, key=lambda x: x[1]))
        # print("Best 10 Accuracy readings, proceeded by their roll count: \n\t", best_acc[:10])

        # match_percent = round(best_acc[0][1] / best_acc[-1][1] * 100, 2)
        # median = best_acc[len(best_acc) // 2][1]
        # median_percent = round(best_acc[0][1] / median * 100, 2)
        # print(f'The best match is {median_percent}% of the median ({median}), and {match_percent}% of the worst match')
        # print('The worst match is: ', best_acc[-1])

        overall = round(100 - best_acc[0][1] / cur_img_sum * 100, 2)
        # print(f"The Template graph has covered {overall}% of the samples graphs pixels, at the best point")

        # if overall == 96.15:       # just for inspecting in inkscape
        #     cv2.imwrite("cur_img_crockedF.png", cur_img)

        # # For apr1 no rotate
        # if best_acc[0][1] < 4000:
        #     print("Template is apr1, matched")

        # <<>>>  for Not viewing   commenting waitkey ord q below to view again
        # break

        # else:
        #     no_img_txt = "Error: No template image found"
        #     cv2.putText(chart3, no_img_txt, (30, 30), txtfont, 2, (0, 0, 210), txtThik)
        #     time.sleep(0.5)
        # print("btn press")

    # cv2.imshow("Shape Data Chart - drag over, scroll mouse wheel", chart3)
    # cv2.imshow("Centered Scaled and Rotated Sample", templ)

    # cv2.imshow("chart2", chart2)

    # cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

        if zones == 4:
            return overall #, zone1_overall, zone2_overall, zone3_overall, zone4_overall
        else:
            return overall
            # --------graph----------->



def tri_area_and_height(base, sb, sc):   # 3 sides
    s = (base + sb + sc) / 2
    area = math.sqrt((s*(s-base)*(s-sb)*(s-sc)))
    h = 2*(area/base)
    return area, h

def tri_area(base, sb, sc):   # 3 sides, can use area and compare it to base length
    try:
        s = (base + sb + sc) / 2
        area = math.sqrt((s*(s-base)*(s-sb)*(s-sc)))
        return area
    except:
        return 0.0

def tri_height(base, sb, sc):   # 3 sides, finds area first
    try:
        s = (base + sb + sc) / 2
        area = math.sqrt((s*(s-base)*(s-sb)*(s-sc)))
        h = 2*(area/base)
        return h
    except:
        return 0.0     # If height is 0 or negative


def tri_median(base, sb, sc):
    n = (1 / 2)*math.sqrt(2*(sb**2) + 2*(sc**2) - base**2)
    return n


def tri_ang_from_side_lens(side_a, side_b, far_side):
    '''
    Find angle of base and side_b, in degrees, of a triangle, knowing just length of all sides.
    Not well tested was getting math domain error,
    '''
    # print(2.0 * side_a * side_b)
    try:
        angle = math.degrees(math.acos((side_a * side_a + side_b * side_b - far_side * far_side) / (2.0 * side_a * side_b)))
        return angle
    except:
        return 0.


def tri_center(pt1xy, pt2xy, pt3xy):
    x = (pt1xy[0] + pt2xy[0] + pt3xy[0]) / 3
    y = (pt1xy[1] + pt2xy[1] + pt3xy[1]) / 3

    return x, y

