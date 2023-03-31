import cv2
import mediapipe as mp
import numpy as np
import math

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)
vid = cv2.VideoCapture('img_vid/light.mp4')

video.set(3, 640)
video.set(4, 340)

img_1 = cv2.imread('img_vid/xoay_xanh2.png', -1)
img_2 = cv2.imread('img_vid/xoay_xanh.png', -1)
img_3 = cv2.imread('img_vid/xoay_do2.png', -1)
img_4 = cv2.imread('img_vid/xoay_do.png', -1)

deg=0

# tọa độ điểm trên ngón tay
def position_data(lmlist):
    global wrist, thumb_top, index_top, index_root, midle_top, ring_top, pinky_top
    wrist = (lmlist[0][0], lmlist[0][1])        # cổ tay
    thumb_top = (lmlist[4][0], lmlist[4][1])    # đầu ngón cái
    index_root = (lmlist[5][0], lmlist[5][1])
    index_top = (lmlist[8][0], lmlist[8][1])    # đầu ngón chỏ
    midle_top = (lmlist[12][0], lmlist[12][1])  # đầu ngón giữa
    ring_top  = (lmlist[16][0], lmlist[16][1])  # đầu ngón nhẫn
    pinky_top = (lmlist[20][0], lmlist[20][1])  # đầu nhón út

# tính khoảng cách
def length2point(p1,p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght

# Tìm cos đầu 3 điểm 
def cos3(a, b, c):
    xa, ya, xb, yb, xc, yc = a[0], a[1], b[0], b[1], c[0], c[1]
    x1 = xb - xa
    y1 = yb - ya
    x2 = xc - xa
    y2 = yc - ya
    cosin = (x1*x2 + y1*y2) / (math.sqrt(x1*x1 + y1*y1)*math.sqrt(x2*x2 + y2*y2))
    return cosin
    
 # Tăng sáng ảnh   
def brightup(img_, value = 0):
    img = img_.copy()

    lim = 255 - value
    img[:,:,0][img[:,:,0] > lim] = 255
    img[:,:,1][img[:,:,1] > lim] = 255
    img[:,:,2][img[:,:,2] > lim] = 255

    img[:,:,0][(img[:,:,0] <= lim) & (img[:,:,0] != 0)] += value
    img[:,:,1][(img[:,:,1] <= lim) & (img[:,:,0] != 0)] += value
    img[:,:,2][(img[:,:,2] <= lim) & (img[:,:,0] != 0)] += value

    return img

# Cộng 2 ảnh theo int
def sum2img(img1, img2):

    img = np.int64(img1) + np.int64(img2)
    img[img > 255] = 255
    img = np.uint8(img)
    return img

#Cộng 2 ảnh theo uint8
def sum2img_uint8(img1, img2):
    img[:,:,0] = img1[:,:,0] + img2[:,:,0]
    img[:,:,1] = img1[:,:,1] + img2[:,:,1] * 2
    img[:,:,2] = img1[:,:,2] + img2[:,:,2] * 3
    return img

# Đổi Xanh sang đỏ
def blue2red(img_, value = 0):
    img = img_.copy()

    lim = 255 - value
    
    # Cộng
    # img[:,:,0][img[:,:,0] > lim] = 255
    # img[:,:,1][img[:,:,1] > lim] = 255
    img[:,:,2][img[:,:,2] > lim] = 255

    # Trừ
    img[:,:,2][img[:,:,2] < value] = value
    # img[:,:,1][img[:,:,1] < value] = value
    # img[:,:,0][img[:,:,0] < value] = value
    
    img[:,:,0][(img[:,:,0] <= lim) & (img[:,:,0] != 0)] -= value
    # img[:,:,1][(img[:,:,1] <= lim) & (img[:,:,1] != 0)] += value
    img[:,:,2][(img[:,:,2] <= lim) & (img[:,:,2] != 0)] += value

    return img

#Thay đổi fg vs bg
def transparent(targetImg, x, y, size=None):
    if size is not None:
        targetImg = cv2.resize(targetImg, size)

    # Tạo ảnh buffer có tỉ lệ tương tự như ảnh gốc
    newFrame = img.copy()
    
    # Tách lớp từ ảnh target
    b, g, r, a = cv2.split(targetImg)
    # Tạo mask từ lớp alpha
    mask = a

    # Bỏ hệ số alpha
    target_color = cv2.merge((b, g, r))
    h, w, _ = target_color.shape

    # Chọn vùng của ảnh gốc để chèn ảnh
    region = newFrame[y:y + h, x:x + w]

    
    # Ảnh FG và BG
    img1_bg = cv2.bitwise_and(region.copy(), region.copy(), mask = cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(target_color, target_color, mask = mask)
    # Thay BG bằng FG
    newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return newFrame

count = 0
countTime = 0
countEnergy = 0
FullEnergy = check_effect_1 = check_effect_2 = False
effect = 0

while True:
    ret, frame = video.read()
    img = cv2.flip(frame, 1)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)
    if result.multi_hand_landmarks:
        count = 0
        xr = yr = dr = cosr = xl = yl = dl = cosl = 0 
        
        #Hand1
        thumbr, indexr = (0, 0)

        for hand in result.multi_hand_landmarks:
            count +=1
            lmList=[]

            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                x_, y_=int(lm.x*w), int(lm.y*h)
                lmList.append([x_, y_])

            # mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
            position_data(lmList)

            wrist_index = length2point(wrist, index_top)
            wrist_index2 = length2point(wrist, index_root)
            wrist_midle = length2point(wrist, midle_top)
            wrist_ring = length2point(wrist, ring_top)
            wrist_pinky = length2point(wrist, pinky_top)
            wrist_3tail = (wrist_midle + wrist_ring + wrist_pinky) / 3
            ratio = wrist_index2/wrist_3tail
            
            if ((ratio > 1) & (wrist_index2 < wrist_index)):
                shield_size = 1.25
                diameter = round(length2point(thumb_top, index_top) * shield_size)
                x = round((thumb_top[0] + index_top[0])/2 - (diameter / 2))
                y = round((thumb_top[1] + index_top[1])/2 - (diameter / 2))
                h, w, c = img.shape

                #xử lý hình bàn tay ở góc hình
                if x < 0:   x = 0
                elif x > w: x = w
                if y < 0:   y = 0
                elif y > h: y = h

                if x + diameter > w:    diameter = w - x
                if y + diameter > h:    diameter = h - y

                #Đếm time 
                
                
                if (xr == x) & (yr == y):
                    xr == 0
                    yr == 0

                if count == 1:
                    xr = x
                    yr = y
                    dr = diameter
                    thumbr = thumb_top
                    indexr = index_top
                else: 
                    xl = x
                    yl = y
                    dl = diameter
                    if (dr != 0) & (dl != 0):
                        cosl = cos3(thumb_top, index_top, (xr + dr, yr + dr))
                        cosr = cos3(thumbr, indexr, (xl + dl, yl + dl))
            else: countEnergy = 0

            xx = round((xr + xl)/2)
            yy = round((yr + yl)/2)
            dd = round((dr + dl)/2)
            addpix = 0
            if count == 2:
                coss = (abs(cosl) + abs(cosr))/2 - 0.1
                if coss > 0:
                    addpix = round(110 * coss)
                    # print(cosr, cosl, coss, addpix)

            # Thiết kế xoay
            shield_size = dd, dd
            ang_vel = 2.0 #Tốc độ xoay
            deg = deg + ang_vel
            if deg > 360: deg = 0
            height, width, col = img_1.shape # Quả cầu
            cen = (width // 2, height // 2) # điểm ở trung tâm
            M1 = cv2.getRotationMatrix2D(cen, round(deg), 1.0) # xoay thuận
            M2 = cv2.getRotationMatrix2D(cen, round(360 - deg), 1.0) #xoay nghịch

            if (dd != 0) & (xr != 0 | yr != 0) & (xl != 0 | yl != 0) & (count == 2):
                # Tạo quả cầu khi chưa đủ năng lượng
                if (countEnergy <50):
                    countEnergy += 1
                    print(countEnergy)
                    rotated1 = cv2.warpAffine(blue2red(img_1, addpix), M1, (width, height))
                    rotated2 = cv2.warpAffine(blue2red(img_2, addpix), M2, (width, height))

                    img = transparent(rotated1, xx, yy, shield_size)
                    img = transparent(rotated2, xx, yy, shield_size)  
                
                #Tạo quả cầu khi đã đủ năng lượng
                if (countEnergy >= 50):
                    FullEnergy = False
                    if (addpix < 90):
                        rotated1 = cv2.warpAffine(blue2red(img_1, addpix), M1, (width, height))
                        rotated2 = cv2.warpAffine(blue2red(img_2, addpix), M2, (width, height))

                        img = transparent(rotated1, xx, yy, shield_size)
                        img = transparent(rotated2, xx, yy, shield_size)  
                    if (addpix > 89):
                        rotated3 = cv2.warpAffine(brightup(img_3, addpix), M1, (width, height))
                        rotated4 = cv2.warpAffine(brightup(img_4, addpix), M2, (width, height))

                        img = transparent(rotated3, xx, yy, shield_size)
                        img = transparent(rotated4, xx, yy, shield_size)  
                        FullEnergy = True

    
    #Hiệu ứng flash
    if (FullEnergy == True) & (countTime < 107):
        if (wrist_index2 > wrist_index) & (check_effect_2 == False): check_effect_1 = True
        if (wrist_index2 * 1.25 < wrist_pinky) &  (check_effect_1 == False): check_effect_2 = True
        
        if check_effect_1 == True:
            h, w, c = img.shape
            countTime += 1
            ret_, flash = vid.read()
            flash = cv2.resize(flash, (w,h))
            flash = cv2.cvtColor(flash, cv2.COLOR_BGR2RGB)
            img = sum2img(img, flash)

        if check_effect_2 == True:
            h, w, c = img.shape
            countTime += 1
            ret_, flash = vid.read()
            flash = cv2.resize(flash, (w,h))
            flash = cv2.cvtColor(flash, cv2.COLOR_BGR2RGB)
            img = sum2img(img, flash)
            img = sum2img_uint8(img, flash)

    if countTime == 106:
        countTime = 0
        FullEnergy = False
        check_effect_1 = False
        check_effect_2 = False
        effect += 1
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow("Image",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()