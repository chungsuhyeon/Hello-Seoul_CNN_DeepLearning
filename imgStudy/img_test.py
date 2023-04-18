import cnn_model
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

def check_photo(path):
    im_rows = 32  # 이미지의 높이
    im_cols = 32  # 이미지의 너비
    im_color = 3  # 이미지의 색공간
    in_shape = (im_rows, im_cols, im_color)
    nb_classes = len(["김밥", "떡볶이", "불고기", "비빔밥", "삼겹살", "치킨"])

    # 저장한 CNN 모델 읽어 들이기
    model = cnn_model.get_model(in_shape, nb_classes)
    model.load_weights('./photos-model.hdf5')

    # 이미지 읽어 들이기
    img = Image.open(path)
    img = img.convert("RGB") # 색공간 변환하기
    img = img.resize((im_cols, im_rows)) # 크기 변경하기
    # plt.imshow(img)
    # plt.show()
    # 데이터 변환하기
    x = np.asarray(img)
    x = x.reshape(-1, im_rows, im_cols, im_color)
    x = x / 255

    # 예측하기
    pre = model.predict([x])[0]
    idx = pre.argmax()
    per = int(pre[idx] * 100)
    return (idx, per)

def check_photo_str(path, filename):
    LABELS = ["김밥", "떡볶이", "불고기", "비빔밥", "삼겹살", "치킨"]

    idx, per = check_photo(path)

    # 응답하기
    print("이 사진은", LABELS[idx],"입니다.")
    print("가능성은", per, "%")

    # answer = {"foodname":LABELS[idx],"accuracy":per}
    # with open('./test.json', 'w', encoding="utf-8") as file :
    #     json.dump(answer, file, indent='\t', ensure_ascii=False)

    return {"foodname":LABELS[idx], "accuracy":per, "imgName":filename}

# if __name__ == '__main__':
#     check_photo_str('C:/Python_code/imgStudy/img/searchImg/e5c11e62.jpg', "e5c11e62.jpg")