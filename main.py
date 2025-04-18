import pickle
import pandas as pd
import json as j
from llm import PromptTemplate, OpenaiLLM
from metric import BNGMetrics
import os


def load_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def preprocess_AdventureWork_1():
    df = load_pickle("./dataset/AdventureWork_1/gold.pkl")
    grouped = df.groupby(['Table'])
    json_structs = [
        {
            "dataset_name": "AdventureWork_1",
            "table_name": group_df['Table'].iloc[0],
            "technical_name": group_df['COLUMN_NAME_1'].tolist(),  # crypted name
            "gt_label": group_df['GT_LABEL_1'].tolist()
        }
        for key, group_df in grouped
    ]
    for item in json_structs:
        aliases = " | ".join(item["technical_name"])
        item["query"] = f"As abbreviations of column names from a table, {aliases} stand for"
    assert (len(df) == sum(len(item["gt_label"]) for item in json_structs))
    return json_structs


def preprocess_AdventureWork_2():
    df = load_pickle("./dataset/AdventureWork_2/gold.pkl")
    grouped = df.groupby(['Table'])
    json_structs = [
        {
            "dataset_name": "AdventureWork_2",
            "table_name": group_df['Table'].iloc[0],
            "technical_name": group_df['COLUMN_NAME_2'].tolist(),
            "gt_label": group_df['GT_LABEL_2'].tolist()
        }
        for key, group_df in grouped
    ]
    for item in json_structs:
        aliases = " | ".join(item["technical_name"])
        item["query"] = f"As abbreviations of column names from a table, {aliases} stand for"
    assert(len(df) == sum(len(item["gt_label"]) for item in json_structs))
    return json_structs

# def preprocess_EDI_demo():
    # df = load_pickle("./dataset/AdventureWork_2/gold.pkl")
    # grouped = df.groupby(['Table'])
    # json_structs = [
    #     {
    #         "dataset_name": "AdventureWork_2",
    #         "table_name": group_df['Table'].iloc[0],
    #         "gt_label": group_df['GT_LABEL_2'].tolist()
    #     }
    #     for key, group_df in grouped
    # ]

def extract_answer(raw_answer_str: str, sep_token: str):
    processed_str = raw_answer_str.strip("").split(".")[0]
    answer_list = [_ans.strip("") for _ans in processed_str.split(sep_token)]
    return answer_list


if __name__ == "__main__":
    json_adv1 = preprocess_AdventureWork_1()
    # json_adv2 = preprocess_AdventureWork_2()
    # json_edi = preprocess_EDI_demo()
    json_total = json_adv1 #+ json_adv2
    # json_string = json.dumps(json_total, indent=2)
    # print(json_string)

    print("============ Start Feeding Data to GPT ============")

    temp_prompt = PromptTemplate()
    model_name = "gpt-3.5-turbo"
    model = OpenaiLLM("gpt-3.5-turbo")
    num_examples = sum([len(ele["gt_label"]) for ele in json_total])
    all_table_results = []

    for _idx, json in enumerate(json_total):
        prompt = (
            temp_prompt.demos() + json["query"]
        )
        print(prompt)
        x_list, y_list = json["technical_name"], json["gt_label"]
        raw_answer = model(prompt, temperature=0.0, max_tokens=1024)

        answers = extract_answer(raw_answer, temp_prompt.sep_token())
        if len(answers) != len(x_list):
            y_pred_list = [" "] * len(x_list)
            print("Error! The extracted answers are not correct.")
        else:
            y_pred_list = answers
            for _x, _pred, _y in zip(x_list, y_pred_list, y_list):
                print(f"{_x}\t-->\t{_pred}\t(label={_y})")

        # save the prediction and table information for each input query name
        table_result = []
        for _x, _y, _pred in zip(
            x_list, y_list, y_pred_list
        ):
            table_result.append(
                [
                    str(_idx),
                    str(json["dataset_name"]),
                    str(json["table_name"]),
                    str(_x),
                    str(_y),
                    str(_pred),
                ]
            )
            all_table_results += table_result

    pred_df = pd.DataFrame(
        all_table_results,
        columns=[
            "idx",
            "dataset_name",
            "table_name",
            "technical_name",
            "gt_label",
            "prediction",
        ],
    )
    print("=================================================")
    print("pred_df:")
    print(pred_df)
    print("=================================================")
    metric_names = ["squad"]
    metric_generator = BNGMetrics(metric_names)

    individual_scores = metric_generator.compute_scores(
        predictions=pred_df["prediction"], references=pred_df["gt_label"], level="individual"
    )
    pred_df["exact-match"] = individual_scores["individual_squad-em"]
    pred_df["f1"] = individual_scores["individual_squad-f1"]

    # save the results
    save_res = {
        "squad-em": individual_scores["squad-em"],
        "squad-f1": individual_scores["squad-f1"],
        "squad-pm": individual_scores["squad-pm"],
        "model-name": model_name,
        "total-num-example": num_examples,
        "demo": temp_prompt.demos(),
    }
    print(save_res)
    print(j.dumps(save_res, indent=4))

    # save results and configures
    save_dir = os.path.join('outputs', "{}-results".format(model_name))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f"{model_name}_results.json"), "w") as fo:
        j.dump(save_res, fo, indent=4)
    # save individual prediction results
    pred_df.to_csv(os.path.join(save_dir, f"{model_name}_predictions.csv"))

