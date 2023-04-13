from references import *


def piece_of_map(map):
    max_width = map.shape[1]
    max_hight = map.shape[0]
    left_w = random.randint(0, max_width - MAP_SLICE) 
    left_h = random.randint(0, max_hight - MAP_SLICE) 
    top_left = (left_w, left_h)
    bottom_right = (top_left[0] + MAP_SLICE, top_left[1] + MAP_SLICE)
    cv2.rectangle(map , top_left, bottom_right, 255, 2)
    cropped_image = map[left_h:left_h+MAP_SLICE, left_w:left_w+MAP_SLICE]

    # show_result(cropped_image)
    # cv2.imshow("cropped", cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 
    return cropped_image


def experiment():
    ''' meth = 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'    '''
    method = eval('cv2.TM_CCOEFF')
    for i in range(EXPERIMENT_COUNT):
        crop_img = piece_of_map(image1)
        result, t_l, b_r, minv, maxv = use_cv_match_template(image1, crop_img, method)

        res_coppy = np.array(result).flatten()

        extrema = argrelextrema(res_coppy, np.greater)
        e = sorted(extrema[0], reverse=True)[0:5]

        peaks, _ = find_peaks(res_coppy, distance=150)
        print(i, e, peaks)

        # plt.subplot(121),plt.imshow(result,cmap = 'gray')
        # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122),plt.imshow(image1,cmap = 'gray')
        # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        # plt.suptitle(method)

        # plt.show()



if __name__ == "__main__":
    init_time = time.time()
    image1 = cv2.imread(IMAGE_1_PATH,0)
    image2 = cv2.imread(IMAGE_2_PATH,0)
    experiment()