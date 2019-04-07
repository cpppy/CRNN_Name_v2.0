import numpy as np
import json
import os


if __name__=='__main__':

    hegui_labels_fpath = './label_from_hegui.txt'
    label_dict = {}
    with open(hegui_labels_fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('label_length: ', len(lines))
        for line in lines:
            piece_list = line.split(' ')
            print(str(piece_list[0]), str(piece_list[4]))
            img_id = str(piece_list[0])
            label = str(piece_list[4]).replace('\'', '')
            label_dict[img_id] = label
    print(label_dict)
    # add
    label_dict['55995944'] = '杨亮'
    label_dict['55997183'] = '董明'
    label_dict['55997284'] = '孙皓'
    # label_dict['55997284'] = '孙皓'
    
    id_label_dict_fpath = './id_label_dict.json'
    with open(id_label_dict_fpath, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False)


    # read crop_result
    img_dir = './name_crop_result'
    img_fn_list = os.listdir(img_dir)
    crop_label_fpath = './crop_id_label_dict.json'
    crop_label_dict = {}
    for img_fn in img_fn_list:
        img_id = img_fn.split('_')[1]
        if img_id not in label_dict.keys():
            print('-----------> img_id_not_found: ', img_fn)
            continue
        else:
            img_label = label_dict[img_id]
        print(img_fn, img_label)
        crop_label_dict[img_fn] = img_label
    print(crop_label_dict)
    with open(crop_label_fpath, 'w', encoding='utf-8') as f:
        json.dump(crop_label_dict, f, ensure_ascii=False)
