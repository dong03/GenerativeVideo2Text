import language_evaluation
import json
import jieba
result_file = '/home/dcb/code/bv/git_aimc/ckpt/caption_git_large_dcb_bv/01-14-18/bv_result_epoch0.json'
result_list = json.load(open(result_file, "r", encoding='utf-8'))
predicts = []
answers = []
for each in result_list:
    # import pdb; pdb.set_trace()
    predicts.append(" ".join(jieba.cut(each["pred_caption"].replace(' ', ''))))
    answers.append(" ".join(jieba.cut(each["gold_caption"].replace(' ', ''))))
evaluator = language_evaluation.CocoEvaluator(verbose=False)
results = evaluator.run_evaluation(predicts, answers)
results['BLEU'] = (results['Bleu_1'] + results['Bleu_2'] +
                   results['Bleu_3'] + results['Bleu_4']) / 4.0
del results['Bleu_1']
del results['Bleu_2']
del results['Bleu_3']
del results['Bleu_4']
print(len(result_list), results)


predicts = []
answers = []
for each in result_list:
    # import pdb; pdb.set_trace()
    predicts.append(each["pred_caption"])
    answers.append(each["gold_caption"])
evaluator = language_evaluation.CocoEvaluator(verbose=False)
results = evaluator.run_evaluation(predicts, answers)
results['BLEU'] = (results['Bleu_1'] + results['Bleu_2'] +
                   results['Bleu_3'] + results['Bleu_4']) / 4.0
del results['Bleu_1']
del results['Bleu_2']
del results['Bleu_3']
del results['Bleu_4']
print(len(result_list), results)
