import numpy as np
import json
import os
import cv2


if __name__=='__main__':


    label_dict_fpath = './json/test.json'
    with open(label_dict_fpath, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
    print(label_dict)




    id_arr = [key.split('_')[1] for key in label_dict.keys()]
    id_label_dict = {}
    id_label_dict['133'] = '朱浩南'
    id_label_dict['177'] = '麦智铭'
    id_label_dict['187'] = '任明龙'
    id_label_dict['190'] = '叶德清'
    id_label_dict['217'] = '黎亮'
    id_label_dict['247'] = '武文超'
    id_label_dict['318'] = '姚志广'
    id_label_dict['325'] = '姚虎'
    id_label_dict['353'] = '芦芝堂'
    id_label_dict['35'] = '徐子博'
    id_label_dict['372'] = '张艳红'
    id_label_dict['387'] = '邱发有'
    id_label_dict['436'] = '陈倩'
    id_label_dict['467'] = '邓豪兴'
    id_label_dict['48'] = '李博文'
    id_label_dict['58'] = '李华益'
    id_label_dict['63'] = '蔡一鹏'
    id_label_dict['67'] = '孙吉祥'
    id_label_dict['92'] = '王磊'








    # read crop_result
    img_dir = './cut_images'
    img_fn_list = os.listdir(img_dir)
    # crop_label_fpath = './crop_id_label_dict.json'
    # crop_label_dict = {}
    for img_fn in img_fn_list:
        # img_id = img_fn.split('_')[1]
        if img_fn not in label_dict.keys():
            print('-----------> img_id_not_found: ', img_fn)
            idx = img_fn.split('_')[0]
            label = id_label_dict[idx]
            print(label)
            label_dict[img_fn] = label

    print(label_dict)
    #     else:
    #         img_label = label_dict[img_id]
    #     print(img_fn, img_label)
    #     crop_label_dict[img_fn] = img_label
    # print(crop_label_dict)
    with open('./test.json', 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False)
    print('final_label_len: ', len(label_dict.keys()))
