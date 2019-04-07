
import hparams
_hparams = hparams.HParams()
# load char_dict
key_path = _hparams.key_fpath #'/data/CRNN_Name_v2.0/key/keys.txt'
char_to_int = {}
int_to_char = {}
with open(key_path, 'r', encoding='utf-8') as key_f:
    chars = key_f.read()
    for idx, char in enumerate(chars):
        char_to_int[char] = idx
        int_to_char[idx] = char
    num_classes = len(chars) + 1