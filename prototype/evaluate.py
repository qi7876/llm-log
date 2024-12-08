import re


class Evaluate:
    def __init__(self):
        self._res_list = []

    def _put_res(self, true_label: bool, predict_label: int):
        self._res_list.append([true_label, predict_label])

    def extract_and_record_result(self, label: str, output: str, log_type: str, output_pattern: str):
        if label is None or output is None:
            raise Exception("Label or Output is none.")
        match log_type:
            case 'BGL':
                if label == '-':
                    true_label = False
                else:
                    true_label = True
            case _:
                raise Exception("Unknown Log Type.")
        if re.search(output_pattern, output):
            match_res = re.search(output_pattern, output).groups()[0]
            match match_res:
                case 'Yes' | 'yes':
                    predict_label = 1
                case 'No' | 'no':
                    predict_label = 0
                case 'Uncertain' | 'uncertain':
                    predict_label = -1
                case _:
                    raise Exception('Unknown predict result.')
        else:
            raise Exception('Not a right LLM output.')
        self._put_res(true_label, predict_label)

    def get_evaluate_data(self):
        TP, FP, TN, FN, PU, NU = 0, 0, 0, 0, 0, 0
        for data in self._res_list:
            match data:
                case [True, 1]:
                    TP += 1
                case [True, 0]:
                    FN += 1
                case [False, 1]:
                    FP += 1
                case [False, 0]:
                    TN += 1
                case [True, -1]:
                    PU += 1
                case [False, -1]:
                    NU += 1

        print(f"Test is over. TP:{TP} FP:{FP} TN:{TN} FN:{FN} PU:{PU} NU:{NU}")
        if TP != 0:
            Precision = float(TP) / float(TP + FP)
            Recall = float(TP) / float(TP + FN)
            print(f"Precision:{Precision} Recall:{Recall}")

