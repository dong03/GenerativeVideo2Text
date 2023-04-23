
import re
import os
import jieba
from eval_utils import calculate_metrics, suppress_stdout_stderr, HiddenPrints

def clean(caption):
    caption = re.sub(
        "[_.!+-=——,$%^，。｡？:･?、~@#ಡω￥%……&*￣▽∇ﾉ♡セナスペシャルブ《》……&*（）～×￥%『』<>「」{}·：【】()/\\\[\]］'\"]",
        '',
        caption.lower(),
    )
    return caption


def eval(ref_f, hyp_f, jieba_flag=True, en=False):
    bv2ix = {}
    ref_anno = []
    ref = open(ref_f).readlines()
    ref = [each.strip().split('\t') for each in ref]
    for ix, line in enumerate(ref):
        bv, caps = line[0], line[1:]
        bv2ix[bv] = ix
        caps = caps[:1]

        for cap in caps:
            cap = clean(cap)
            if jieba_flag:
                cap_ = ' '.join(jieba.cut(clean(cap.replace(' ',''))))
            elif en:
                cap_ = cap
            else:
                cap_ =  ' '.join(list(cap.replace(' ','')))
            ref_anno.append({u'image_id': bv2ix[bv], u'caption': cap_})

    datasetGTS = {
        'annotations':ref_anno}

    
    hypo = open(hyp_f).readlines()
    hypo = [each.strip().split('\t') for each in hypo]
    
    hyp_anno = []
    for ix, line in enumerate(hypo):
        bv, caps = line[0], line[1:]
        if bv not in bv2ix:
            continue

        for cap in caps[:1]:
            cap = clean(cap)
            if jieba_flag:
                cap_ = ' '.join(jieba.cut(clean(cap.replace(' ',''))))
            elif en:
                cap_ = cap
            else:
                cap_ =  ' '.join(list(cap.replace(' ','')))
            hyp_anno.append({u'image_id': bv2ix[bv], u'caption':cap_})

    datasetRES = {
        'annotations':hyp_anno}
    rng = range(len(bv2ix))

    with suppress_stdout_stderr():
        with HiddenPrints():
            temp = calculate_metrics(rng, datasetGTS, datasetRES)
    temp['BLEU'] = (temp['Bleu_1'] + temp['Bleu_2'] + temp['Bleu_3'] + temp['Bleu_4']) / 4.0
    del temp['Bleu_1']
    del temp['Bleu_2']
    del temp['Bleu_3']
    del temp['Bleu_4']
    for key in temp:
        temp[key] = "%.3f"%(100*temp[key])
    return temp

if __name__ == '__main__':
    cn_pd_files = [
        'ChinaOpenResults/GIT_ft_ChinaOpen.txt',
        'ChinaOpenResults/GIT_ft_RandomBV.txt',
        'ChinaOpenResults/GIT_ft_Vatex.txt',
        'ChinaOpenResults/GVT_ft_ChinaOpen_woVTM.txt',
        'ChinaOpenResults/GVT_ft_ChinaOpen.txt'
    ]
    for hyp_f in cn_pd_files:
        print(os.path.split(hyp_f)[-1])
        # character, content-based
        print(eval(ref_f='data/BV_0417_caption_cn.txt', hyp_f=hyp_f,jieba_flag=False))
        # word, content-based
        print(eval(ref_f='data/BV_0417_caption_cn.txt', hyp_f=hyp_f,jieba_flag=True))
        # character, content-beyond
        print(eval(ref_f='data/BV_0417_title_cn.txt', hyp_f=hyp_f,jieba_flag=False))
        # word, content-beyond
        print(eval(ref_f='data/BV_0417_title_cn.txt', hyp_f=hyp_f,jieba_flag=True))
        
    
    en_pd_files = [
        'ChinaOpenResults/GIT.txt',
    ]
    for hyp_f in cn_pd_files:
        print(os.path.split(hyp_f)[-1])
        # content-based
        print(eval(ref_f='data/BV_0417_caption_en.txt', hyp_f=hyp_f,jieba_flag=False, en=True))
        # content-beyond
        print(eval(ref_f='data/BV_0417_title_en.txt', hyp_f=hyp_f,jieba_flag=False, en=True))