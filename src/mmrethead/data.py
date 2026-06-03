import os
import re
import json
import random
from functools import partial
from datasets import load_dataset
from torch.utils.data import Dataset
from .data_transforms import build_haystack_variant_config, apply_haystack_composition_variant
from .utils import calculate_metrics, parse_output

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pipline
# 1. template 2. load data 3. max test_sample sampling 4. update context 5. post process (later)
# field
# 1. context 2. question 3. answer 4. image_list


def default_post_process(output, example, metrics=None, prefix=None):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    prediction = output["output"]
    answer = example["answer"]
    mets = calculate_metrics(prediction, answer, metrics)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction, prefix=prefix)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer, metrics)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    return mets, {"parsed_output": parsed_pred}


### instruction\n\nDocument (Title: {title}): {text}\n\nDocument (Title: {title}): {text}\n\nQuestion: {question}
def load_vrag(args, path, max_test_samples=None):
    """
    Load the data for vision rag
    """
    user_template = "Use the given documents to write a concise and short answer to the question about the entity shown in the image. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = sorted(set(data["id"]), key=lambda x: int(x.split("-")[1]))
        key_list = key_list[:min(max_test_samples, len(key_list))]
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    passage_template = "Document (Title: {title}): {text}"

    def update(sample):
        doc_span = [passage_template.format(**c) for c in sample["ctxs"]] # document-level spans
        passage_text = "\n\n".join(doc_span)

        gold_span = [passage_template.format(**c) for c in sample["positive_ctxs"]] # gold document-level spans
        gold_span = [{"type": "text", "text": gs} for gs in gold_span]

        question = "<image>" + sample["question"]
        query_span = {"type": "text-image", "text": question, "image": [sample["image"]], "img_idx": [0]}
        return {"doc_span": doc_span,
                "gold_span": gold_span,
                "query_span": query_span,
                "context": passage_text,
                "question": question,
                "image_list": [os.path.join(args.image_file_root, sample["image"])]}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def load_nq_data(args, path, max_test_samples=None):
    """
    Load the NQ retrieval-head data.
    """
    user_template = "Here are some paragraphs:\n\n{context}\n\nPlease find information that are relevant to the following query in the paragraphs above. Query: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    with open(path, "r") as f:
        data_instances = json.load(f)
        if max_test_samples is not None:
            data_instances = data_instances[:max_test_samples]

    qrel_path = os.path.join(args.test_file_root, 'nq_label.json')
    with open(qrel_path, "r") as f:
        qrels = json.load(f)

    passage_template = "Paragraph (Title: {title}): {paragraph_text}"

    def update(sample):
        query = sample["question"]
        doc_span = sample["paragraphs"]
        for p in doc_span:
            p["paragraph_text"] = p["paragraph_text"].strip()

        gold_doc_list = qrels[sample['idx']]
        gold_doc_list = [str(d) for d, rel in gold_doc_list.items() if float(rel) > 0]
        gold_span = [p for p in doc_span if p["idx"] in gold_doc_list]

        doc_span = [passage_template.format(**c) for c in doc_span] # document-level spans
        passage_text = "\n\n".join(doc_span)

        gold_span = [passage_template.format(**c) for c in gold_span] # gold document-level spans
        gold_span = [{"type": "text", "text": gs} for gs in gold_span]

        query_span = {"type": "text", "text": query, "image": [], "img_idx": []}
        return {"doc_span": doc_span,
                "gold_span": gold_span,
                "query_span": query_span,
                "context": passage_text,
                "question": query, 
                "image_list": [], 
                "idx": sample["idx"]}

    new_data_instances = []
    for sample in data_instances:
        new_sample = update(sample)
        if not new_sample["gold_span"]: # remove samples without gold span
            continue

        new_data_instances.append(new_sample)
    data_instances = new_data_instances

    return {
        "data": data_instances,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}
def load_mm_niah_text(args, path, max_test_samples=None):
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = sorted(set(data["id"]), key=lambda x: int(x.split("-")[-1]))
        key_list = key_list[:min(max_test_samples, len(key_list))]
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    variant_cfg = build_haystack_variant_config(args)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        paragraph_list = sample["ctxs"]
        doc_span = [p["text"] for p in paragraph_list] # document-level spans
        passage_text = " ".join(doc_span)

        gold_span = sample["positive_ctxs"] # gold document-level spans

        query_span = {"type": "text", "text": sample["question"], "image": [], "img_idx": []}
        image_list = [os.path.join(args.image_file_root, image) for image in sample["image_list"]]
        return {"doc_span": doc_span,
                "gold_span": gold_span,
                "query_span": query_span,
                "context": passage_text,
                "image_list": image_list}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}"
# used for multiple choices problems in mm-niah-image
def load_mm_niah_image_mc(args, path, max_test_samples=None):
    user_template = "You are given interleaved text and images. Please answer the question with the option's letter (A, B, etc.) based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = sorted(set(data["id"]), key=lambda x: int(x.split("-")[-1]))
        key_list = key_list[:min(max_test_samples, len(key_list))]
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    variant_cfg = build_haystack_variant_config(args)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        # doc span
        paragraph_list = sample["ctxs"]
        doc_span = [p["text"] for p in paragraph_list] # document-level spans
        passage_text = " ".join(doc_span)

        # add choices image to image_list
        image_list = sample["image_list"] + sample["choices_image"]
        image_len, image_len_with_choices = len(sample["image_list"]), len(image_list)

        # gold document-level spans (haystack image)
        haystack_image_list = [os.path.join("mm-niah", i["image"]) for i in sample["positive_ctxs"]]
        haystack_image_idx_list = [[idx, img] for idx, img in enumerate(image_list) if img in haystack_image_list]
        gold_span = [{"type": "image",
                      "image": img,
                      "img_idx": img_idx} for img_idx, img in haystack_image_idx_list]

        # add choices to question
        question = sample["question"]
        question += "".join([f"\n{chr(c_idx + ord('A'))}. <image>" for c_idx in range(len(sample["choices_image"]))])

        choices_img_indices = list(range(image_len, image_len_with_choices))
        query_span = {"type": "text-image", "text": question, "image": sample["choices_image"],
                      "img_idx": choices_img_indices}

        # add current image root
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]

        return {"doc_span": doc_span, # this doc_span is all text
                "gold_span": gold_span, # this gold_span is only of image idx
                "query_span": query_span,
                "context": passage_text,
                "image_list": image_list,
                "question": question}
    
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": partial(default_post_process, metrics="mc_acc", prefix=system_template)
    }

# retrieval identical image with yes/no answer
def load_mm_niah_image_yn(args, path, max_test_samples=None):
    user_template = "You are given a document consisting of interleaved text and images. Your task is to determine if the specific image provided in the question appears anywhere in the document. Respond with only 'Yes' or 'No'. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = sorted(set(data["id"]), key=lambda x: int(x.split("-")[-1]))
        key_list = key_list[:min(max_test_samples, len(key_list))]
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    variant_cfg = build_haystack_variant_config(args)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        # get answer image
        answer_image = sample["choices_image"][sample["answer"]]

        # add choices image to image_list and revise its needle
        image_list = sample["image_list"] + [answer_image]
        image_len, image_len_with_choices = len(sample["image_list"]), len(image_list)
        
        needle_image_list = [os.path.join("mm-niah", i["image"]) for i in sample["positive_ctxs"]]
        needle_image_idx_list = [[idx, img] for idx, img in enumerate(image_list) if img in needle_image_list]
        assert len(needle_image_idx_list) == 1, f"needle_image_idx_list: {needle_image_idx_list}"
        image_list[needle_image_idx_list[0][0]] = answer_image
        needle_image_idx_list[0][1] = answer_image

        # doc span
        paragraph_list = sample["ctxs"]
        doc_span = [p["text"] for p in paragraph_list] # document-level spans
        passage_text = " ".join(doc_span)

        # gold document-level spans (haystack image)
        gold_span = [{"type": "image",
                    "image": img,
                    "img_idx": img_idx} for img_idx, img in needle_image_idx_list]

        question = "Does the following image appear anywhere in the document? <image>"
        answer_img_indices = list(range(image_len, image_len_with_choices))
        query_span = {"type": "text-image", "text": question, "image": [answer_image],
                    "img_idx": answer_img_indices}

        # add current image root
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]

        return {"doc_span": doc_span, # this doc_span is all text
                "gold_span": gold_span, # this gold_span is only of image idx
                "query_span": query_span,
                "context": passage_text,
                "image_list": image_list,
                "question": question}
    data = data.map(update, num_proc=args.preprocessing_num_workers, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def sanitize_filename(text):
    clean_text = text.rstrip('.')
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    clean_text = clean_text.lower()
    clean_text = clean_text.replace(' ', '_')
    return clean_text


# instruction\n\nparagraph\n\nparagraph\n\nQuestion: {question}
def load_mm_niah_text_needle_image(args, path, max_test_samples=None):
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        key_list = sorted(set(data["id"]), key=lambda x: int(x.split("-")[-1]))
        key_list = key_list[:min(max_test_samples, len(key_list))]
        data = data.filter(lambda x: x["id"] in key_list, num_proc=args.preprocessing_num_workers)

    variant_cfg = build_haystack_variant_config(args)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        # replace text needle with image needle
        paragraph_list, needle = sample["ctxs"], sample["positive_ctxs"]
        assert len(needle) == 1
        needle = needle[0]
        needle_image = "mm-niah/text_needle_image/" + sanitize_filename(needle["text"]) + ".png"

        needle_image_idx = 0
        for idx, p in enumerate(paragraph_list):
            if p["type"] == "image":
                needle_image_idx += 1
                continue

            if "has_answer" in p and p["has_answer"]:
                assert needle["text"] in p["text"]
                break
        p["text"] = p["text"].replace(needle["text"], "").strip()
        p["len"] -= needle["len"]
        p.pop("has_answer")
        paragraph_list.insert(idx + 1, {"type": "image", "text": "<image>", "has_answer": True})
        image_list = sample["image_list"]
        image_list.insert(needle_image_idx, needle_image)

        doc_span = [p["text"] for p in paragraph_list] # document-level spans
        passage_text = " ".join(doc_span)

        gold_span = [{"type": "image", 
                     "image": needle_image, 
                     "img_idx": needle_image_idx}]

        query_span = {"type": "text", "text": sample["question"], "image": [], "img_idx": []}
        image_list = [os.path.join(args.image_file_root, image) for image in image_list]
        return {"doc_span": doc_span,
                "gold_span": gold_span,
                "query_span": query_span,
                "context": passage_text,
                "image_list": image_list}
    data = data.map(update, num_proc=args.preprocessing_num_workers, load_from_cache_file=False, remove_columns=["ctxs"])

    return {
        "data": data,
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
    }


def load_doc_qa(args, path, max_test_samples=None):
    user_template = "You are given a document with text and images, and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'Not answerable.' Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    item_template = "Document {doc_id:.15} (page {page_id}): <image>"
    system_template = "Answer:"
    prompt_template = user_template + "\n" + system_template

    path = os.path.join(args.test_file_root, path)
    data = load_dataset("json", data_files=path)["train"]

    if max_test_samples is not None:
        data = data.shuffle(seed=args.seed).select(range(min(len(data), max_test_samples)))

    def page_id_from_path(image_path):
        match = re.search(r"_page(\d+)\.jpg$", image_path)
        if match is None:
            raise ValueError(f"Could not parse page id from {image_path}")
        return match.group(1)

    def answer_page_paths(sample):
        ans_page_path_list = sample.get("ans_page_path_list")
        if ans_page_path_list:
            return ans_page_path_list

        answer_page_ids = {str(page_id) for page_id in sample.get("ans_page_list", [])}
        return [image_path for image_path in sample["page_list"] if page_id_from_path(image_path) in answer_page_ids]

    def update(sample):
        # document-level spans
        image_list = sample["page_list"]
        page_prompt_list = [item_template.format(
            doc_id=image_path.split("/")[-2], page_id=image_path.split("page")[-1].split(".")[0]) for image_path in image_list]
        passage_text = "\n\n".join(page_prompt_list)

        # gold document-level spans
        ans_page_path_list = answer_page_paths(sample)
        ans_page_idx_list = [[idx, img] for idx, img in enumerate(image_list) if img in ans_page_path_list]
        gold_span = [{"type": "image",
                      "image": img,
                      "img_idx": img_idx,
                      "bbox": None} for img_idx, img in ans_page_idx_list] # TODO bounding box annotation

        question = sample["question"]
        doc_id = sample["doc_name"]
        question = "Based on Document {doc_id:.15}, answer the following question. ".format(doc_id=doc_id) + question
        query_span = {"type": "text", "text": question}

        image_list = [os.path.join(args.image_file_root, image) for image in image_list]

        return {"doc_span": page_prompt_list,
                "gold_span": gold_span,
                "query_span": query_span,
                "context": passage_text,
                "image_list": image_list,
                "question": question}

    data = data.map(update, num_proc=args.preprocessing_num_workers)

    def dcoqa_post_process(output, example):
        """
        Returns: metrics (dict) and additional info to update the original sample with (dict)
        """
        prediction = output["output"]
        answer = [example["answer"], example["answer_format"]]
        parsed_pred = parse_output(prediction, prefix=system_template)
        if parsed_pred is None:
            parsed_pred = prediction
        mets = calculate_metrics(parsed_pred, answer, "doc_qa")
        return mets, {"parsed_output": parsed_pred}

    return {
        "data": data,
        "needle_type": "image",
        "prompt_template": prompt_template,
        "user_template": user_template,
        "system_template": system_template,
        "post_process": dcoqa_post_process
    }


def load_data(args, dataset, path=None):
    if "infoseek" in dataset or "viquae" in dataset:
        data = load_vrag(args, path, max_test_samples=args.max_test_samples)
    elif "nq_data" in dataset:
        data = load_nq_data(args, path, max_test_samples=args.max_test_samples)
    elif "vh_single" in dataset or "vh_multi" in dataset:
        data = load_visual_haystack(args, path, max_test_samples=args.max_test_samples)
    elif "mm_niah" in dataset:
        if "text" in path:
            data = load_mm_niah_text(args, path, max_test_samples=args.max_test_samples)
        else:
            if "retrieval" in path:
                data = load_mm_niah_image_mc(args, path, max_test_samples=args.max_test_samples)
            else:
                raise ValueError(f"Unknown dataset {dataset}")
    elif "identical_retrieval" in dataset:
        data = load_mm_niah_image_yn(args, path, max_test_samples=args.max_test_samples)
    elif "text_needle_image" in dataset:
        data = load_mm_niah_text_needle_image(args, path, max_test_samples=args.max_test_samples)
    elif any(key in dataset for key in ["longdocurl", "mmlongdoc", "slidevqa"]):
        data = load_doc_qa(args, path, max_test_samples=args.max_test_samples)
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    
    return data


class TestItemDataset(Dataset):
    def __init__(self, data, llm, processor):
        self.data = data
        self.llm = llm
        self.processor = processor

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, idx):
        inputs = self.llm.prepare_inputs(self.data["data"][idx], self.data)
        return inputs



#########################head mask code###########################
# The functions below are specific to the head mask inference task.
def head_mask_default_post_process(answer, prediction, metrics=None, prefix=None):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    mets = calculate_metrics(prediction, answer, metrics)
    # we check the metrics after parsing and take the max
    parsed_pred = parse_output(prediction, prefix=prefix)
    if parsed_pred is not None:
        new_mets = calculate_metrics(parsed_pred, answer, metrics)
        mets = {k: max(v, new_mets[k]) for k, v in mets.items()}
    mets = {k: float(v) for k, v in mets.items()}
    assert len(mets) == 1
    score = list(mets.values())[0] * 100

    return score

def select_head_mask_dataset_example(data, example_idx=-1, example_id=None):
    if example_id is not None:
        data = data.filter(lambda x: x["id"] == example_id, num_proc=8)
        return data.select([0])

    if example_idx < 0:
        example_idx = len(data) + example_idx
    return data.select([example_idx])


def select_head_mask_list_example(data_instances, key, example_idx=-1, example_id=None):
    if example_id is not None:
        for sample in data_instances:
            if sample[key] == example_id:
                return sample
        raise ValueError(f"Could not find example_id={example_id}")

    return data_instances[example_idx]


def infer_head_mask_text_dataset(task_data_path, dataset_name=None):
    if dataset_name is not None:
        name = dataset_name.lower()
    else:
        name = os.path.basename(task_data_path).lower()

    if "text_needle_image" in name:
        return "text_needle_image"
    if any(key in name for key in ["longdocurl", "mmlongdoc", "slidevqa", "doc_qa"]):
        return "doc_qa"
    if "nq" in name:
        return "nq_data"
    if any(key in name for key in ["retrieval-text", "mm_text", "mm_niah_text"]):
        return "mm_text"

    raise ValueError(f"Could not infer text head-mask dataset from {task_data_path} / {dataset_name}")


def infer_head_mask_image_dataset(task_data_path, dataset_name=None):
    if dataset_name is not None:
        name = dataset_name.lower()
    else:
        name = os.path.basename(task_data_path).lower()

    if any(key in name for key in ["viquae", "infoseek", "vrag"]):
        return "vrag"
    if "identical" in name:
        return "identical_image"
    if any(key in name for key in ["retrieval-image", "mm_image", "mm_niah_image"]):
        return "mm_image"

    raise ValueError(f"Could not infer image head-mask dataset from {task_data_path} / {dataset_name}")


def head_mask_load_mm_niah_text(task_data_path, task_image_dir, example_idx=-1, example_id=None, variant_cfg=None):
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"

    data = load_dataset("json", data_files=task_data_path)["train"]
    data = select_head_mask_dataset_example(data, example_idx=example_idx, example_id=example_id)

    # Seperate needle from haystack so build context can reinsert it during experiment
    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        paragraph_list = sample["ctxs"]
        needle = sample["positive_ctxs"]
        assert len(needle) == 1
        needle = needle[0]

        # remove needle
        remove_flag = False
        for p in paragraph_list:
            if "has_answer" in p and p["has_answer"]:
                if needle["text"] in p["text"]:
                    remove_flag = True
                p["text"] = p["text"].replace(needle["text"], "").strip()
                p["len"] -= needle["len"]
                p.pop("has_answer")

        assert remove_flag

        passage_text = " ".join([p["text"] for p in paragraph_list])
        image_list = [os.path.join(task_image_dir, image) for image in sample["image_list"]]

        return {"context": passage_text, "image_list": image_list, "needle": {"type": "text", "text": needle["text"]}}

    # add real_needle
    data = data.map(update, remove_columns=["ctxs"])

    if "retrieval" in os.path.basename(task_data_path):
        metric = "sub_em"
    else:
        raise NameError(f"Wrong mm-niah task: {os.path.basename(task_data_path)}")

    return {
        "data": data,
        "system_template": system_template,
        "user_template": user_template,
        "metric": partial(head_mask_default_post_process, metrics=metric, prefix=system_template),
    }

def head_mask_load_mm_niah_image(task_data_path, task_image_dir, example_idx=-1, example_id=None, variant_cfg=None):
    user_template = "You are given interleaved text and images. Please answer the question with the option's letter (A, B, etc.) based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"

    data = load_dataset("json", data_files=task_data_path)["train"]
    data = select_head_mask_dataset_example(data, example_idx=example_idx, example_id=example_id)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        paragraph_list = sample["ctxs"]
        needle = sample["positive_ctxs"]
        assert len(needle) == 1
        needle = needle[0]

        # Remove needle image (marked with has_ans=True)
        needle_entry = None
        for p in paragraph_list:
            if p.get("has_ans"):
                needle_entry = p
                break
        assert needle_entry is not None, "Could not find needle in context"
        paragraph_list = [p for p in paragraph_list if not p.get("has_ans")]

        # Find needle's position in image_list and remove it
        img_idx = 0
        for p in sample["ctxs"]:
            if p.get("has_ans"):
                break
            if p.get("type") == "image":
                img_idx += 1
        image_list = list(sample["image_list"])
        needle_image = image_list.pop(img_idx)

        passage_text = " ".join([p["text"] for p in paragraph_list])
        image_list = [os.path.join(task_image_dir, img) for img in image_list]

        # Build choice images into question
        question = sample["question"]
        choices_image = [os.path.join(task_image_dir, img) for img in sample["choices_image"]]
        question += "".join([f"\n{chr(c_idx + ord('A'))}. <image>" for c_idx in range(len(choices_image))])

        # Needle is the image
        needle_image_path = os.path.join(task_image_dir, needle_image)

        return {
            "context": passage_text,
            "image_list": image_list,
            "needle": {"type": "image", "image": needle_image_path},
            "question": question,
            "choices_image": choices_image,
        }


    data = data.map(update, remove_columns=["ctxs", "positive_ctxs", "needle_image_list"])

    return {
      "data": data,
      "system_template": system_template,
      "user_template": user_template,
      "metric": partial(head_mask_default_post_process, metrics="mc_acc", prefix=system_template),
    }

def head_mask_load_identical_image(task_data_path, task_image_dir, example_idx=-1, example_id=None, variant_cfg=None):
    user_template = "You are given a document consisting of interleaved text and images. Your task is to determine if the specific image provided in the question appears anywhere in the document. Respond with only 'Yes' or 'No'. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"

    data = load_dataset("json", data_files=task_data_path)["train"]
    data = select_head_mask_dataset_example(data, example_idx=example_idx, example_id=example_id)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        paragraph_list = [dict(p) for p in sample["ctxs"]]
        answer_idx = int(sample["answer"])
        needle_image_list = sample.get("needle_image_list") or []
        if len(needle_image_list) != 1:
            raise ValueError(f"Expected one identical-image query image, got: {needle_image_list}")
        answer_image = needle_image_list[0]
        expected_answer_image = sample["choices_image"][answer_idx]
        if expected_answer_image != answer_image:
            raise ValueError(
                "Identical-image query does not match the multiple-choice answer image: "
                f"{expected_answer_image} != {answer_image}"
            )

        needle_entry = None
        for paragraph in paragraph_list:
            if paragraph.get("has_ans"):
                needle_entry = paragraph
                break
        if needle_entry is None:
            raise ValueError("Could not find the identical-image needle in the selected sample.")

        paragraph_list = [p for p in paragraph_list if not p.get("has_ans")]

        image_idx = 0
        for paragraph in sample["ctxs"]:
            if paragraph.get("has_ans"):
                break
            if paragraph.get("type") == "image":
                image_idx += 1

        image_list = list(sample["image_list"])
        if image_idx >= len(image_list):
            raise ValueError("Needle image index is out of range for the selected sample.")
        image_list.pop(image_idx)

        passage_text = " ".join(paragraph["text"] for paragraph in paragraph_list)
        haystack_images = [os.path.join(task_image_dir, image) for image in image_list]
        query_image = os.path.join(task_image_dir, answer_image)

        return {
            "context": passage_text,
            "image_list": haystack_images,
            "needle": {"type": "image", "image": query_image},
            "question": "Does the following image appear anywhere in the document? <image>",
            "choices_image": [query_image],
            "answer": "Yes",
        }

    data = data.map(update, remove_columns=["ctxs", "positive_ctxs", "needle_image_list", "choices_image"])

    return {
        "data": data,
        "system_template": system_template,
        "user_template": user_template,
        "metric": partial(head_mask_default_post_process, metrics="sub_em", prefix=system_template),
    }


def head_mask_load_nq_data(task_data_path, task_image_dir, example_idx=-1, example_id=None):
    del task_image_dir

    from rouge_score import rouge_scorer

    user_template = "Here are some paragraphs:\n\n{context}\n\nPlease find information that are relevant to the following query in the paragraphs above. Query: {question}"
    system_template = "Answer:"
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    with open(task_data_path, "r", encoding="utf-8") as handle:
        data_instances = json.load(handle)
    sample = select_head_mask_list_example(data_instances, "idx", example_idx=example_idx, example_id=example_id)

    qrel_path = os.path.join(os.path.dirname(task_data_path), "nq_label.json")
    with open(qrel_path, "r", encoding="utf-8") as handle:
        qrels = json.load(handle)

    gold_doc_list = qrels[sample["idx"]]
    gold_doc_list = {str(doc_idx) for doc_idx, rel in gold_doc_list.items() if float(rel) > 0}
    if not gold_doc_list:
        raise ValueError(f"No supporting documents found for NQ sample {sample['idx']}")

    passage_template = "Paragraph (Title: {title}): {paragraph_text}"
    supporting = []
    background = []
    for paragraph in sample["paragraphs"]:
        formatted = passage_template.format(**paragraph)
        if paragraph["idx"] in gold_doc_list:
            supporting.append(formatted)
        else:
            background.append(formatted)

    if not supporting:
        raise ValueError(f"Could not map NQ qrels back to paragraph entries for {sample['idx']}")

    needle_text = "\n\n".join(supporting)
    passage_text = "\n\n".join(background)

    return {
        "data": [{
            "idx": sample["idx"],
            "question": sample["question"],
            "answer": needle_text,
            "context": passage_text,
            "image_list": [],
            "needle": {"type": "text", "text": needle_text},
        }],
        "system_template": system_template,
        "user_template": user_template,
        "metric": lambda ans, pred: scorer.score(ans, pred)["rouge1"].recall * 100,
    }


def head_mask_load_vrag(task_data_path, task_image_dir, example_idx=-1, example_id=None):
    user_template = "Use the given documents to write a concise and short answer to the question about the entity shown in the image. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    passage_template = "Document (Title: {title}): {text}"

    data = load_dataset("json", data_files=task_data_path)["train"]
    data = select_head_mask_dataset_example(data, example_idx=example_idx, example_id=example_id)

    def update(sample):
        gold_doc_ids = {ctx["doc_id"] for ctx in sample["positive_ctxs"]}
        gold_passages = [passage_template.format(**ctx) for ctx in sample["positive_ctxs"]]
        background_passages = [passage_template.format(**ctx) for ctx in sample["ctxs"] if ctx["doc_id"] not in gold_doc_ids]

        if not gold_passages:
            raise ValueError("VRAG sample has no positive passages to use as the text needle.")

        return {
            "context": "\n\n".join(background_passages),
            "image_list": [],
            "needle": {"type": "text", "text": "\n\n".join(gold_passages)},
            "question": "<image>" + sample["question"],
            "choices_image": [os.path.join(task_image_dir, sample["image"])],
            "answer": sample["answer"],
        }

    data = data.map(update, remove_columns=["ctxs", "positive_ctxs"])

    return {
        "data": data,
        "system_template": system_template,
        "user_template": user_template,
        "metric": partial(head_mask_default_post_process, metrics="sub_em", prefix=system_template),
    }


def head_mask_load_text_needle_image(task_data_path, task_image_dir, example_idx=-1, example_id=None, variant_cfg=None):
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"

    data = load_dataset("json", data_files=task_data_path)["train"]
    data = select_head_mask_dataset_example(data, example_idx=example_idx, example_id=example_id)

    def update(sample):
        sample = apply_haystack_composition_variant(sample, variant_cfg)
        paragraph_list = [dict(p) for p in sample["ctxs"]]
        needle_list = sample["positive_ctxs"]
        assert len(needle_list) == 1
        needle = dict(needle_list[0])
        needle_text = needle["text"]

        remove_flag = False
        for paragraph in paragraph_list:
            if paragraph.get("has_answer"):
                if needle_text in paragraph["text"]:
                    remove_flag = True
                paragraph["text"] = paragraph["text"].replace(needle_text, "").strip()
                if "len" in paragraph and "len" in needle:
                    paragraph["len"] -= needle["len"]
                paragraph.pop("has_answer", None)

        if not remove_flag:
            raise ValueError("Could not remove the text needle from the selected MM-NIAH sample.")

        rendered_name = f"{sanitize_filename(needle_text)}.png"
        rendered_path = os.path.join(task_image_dir, "mm-niah", "text_needle_image", rendered_name)
        if not os.path.exists(rendered_path):
            raise FileNotFoundError(f"Missing rendered text needle image: {rendered_path}")

        passage_text = " ".join(paragraph["text"] for paragraph in paragraph_list)
        image_list = [os.path.join(task_image_dir, image) for image in sample["image_list"]]

        return {
            "context": passage_text,
            "image_list": image_list,
            "needle": {"type": "image", "image": rendered_path},
        }

    data = data.map(update, remove_columns=["ctxs"])

    if "retrieval" in os.path.basename(task_data_path):
        metric = "sub_em"
    else:
        raise NameError(f"Wrong mm-niah task: {os.path.basename(task_data_path)}")

    return {
        "data": data,
        "system_template": system_template,
        "user_template": user_template,
        "metric": partial(head_mask_default_post_process, metrics=metric, prefix=system_template),
    }


def head_mask_load_doc_qa(task_data_path, task_image_dir, example_idx=-1, example_id=None):
    user_template = "You are given a document with text and images, and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'Not answerable.' Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"
    item_template = "Document {doc_id:.15} (page {page_id}): <image>"

    data = load_dataset("json", data_files=task_data_path)["train"]
    data = select_head_mask_dataset_example(data, example_idx=example_idx, example_id=example_id)

    def page_id_from_path(image_path):
        match = re.search(r"_page(\d+)\.jpg$", image_path)
        if match is None:
            raise ValueError(f"Could not parse page id from {image_path}")
        return match.group(1)

    def answer_page_paths(sample):
        ans_page_path_list = sample.get("ans_page_path_list")
        if ans_page_path_list:
            return ans_page_path_list

        answer_page_ids = {str(page_id) for page_id in sample.get("ans_page_list", [])}
        return [image_path for image_path in sample["page_list"] if page_id_from_path(image_path) in answer_page_ids]

    def update(sample):
        answer_page_set = set(answer_page_paths(sample))
        needle_image = None
        remaining_pages = []
        for image_path in sample["page_list"]:
            if needle_image is None and image_path in answer_page_set:
                needle_image = image_path
            else:
                remaining_pages.append(image_path)

        if needle_image is None:
            raise ValueError("DocQA sample has no answer page to use as the image needle.")

        page_prompt_list = [item_template.format(doc_id=image_path.split("/")[-2], page_id=page_id_from_path(image_path)) for image_path in remaining_pages]
        question = "Based on Document {doc_id:.15}, answer the following question. ".format(doc_id=sample["doc_name"]) + sample["question"]

        return {
            "context": "\n\n".join(page_prompt_list),
            "image_list": [os.path.join(task_image_dir, image) for image in remaining_pages],
            "needle": {"type": "image", "image": os.path.join(task_image_dir, needle_image)},
            "question": question,
            "answer": sample["answer"],
            "answer_format": sample["answer_format"],
        }

    data = data.map(update)

    def doc_qa_metric(answer, prediction, answer_format):
        parsed_pred = parse_output(prediction, prefix=system_template)
        if parsed_pred is None:
            parsed_pred = prediction
        score = calculate_metrics(parsed_pred, [answer, answer_format], "doc_qa")["doc_qa"]
        return float(score) * 100

    return {
        "data": data,
        "system_template": system_template,
        "user_template": user_template,
        "metric": lambda answer, prediction: doc_qa_metric(answer, prediction, data[0]["answer_format"]),
    }


def head_mask_load_text_task(task_data_path, task_image_dir, dataset_name=None, example_idx=-1, example_id=None, variant_cfg=None):
    dataset_key = infer_head_mask_text_dataset(task_data_path, dataset_name=dataset_name)
    if dataset_key == "mm_text":
        return head_mask_load_mm_niah_text(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id, variant_cfg=variant_cfg)
    if dataset_key == "text_needle_image":
        return head_mask_load_text_needle_image(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id, variant_cfg=variant_cfg)
    if dataset_key == "nq_data":
        return head_mask_load_nq_data(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id)
    if dataset_key == "doc_qa":
        return head_mask_load_doc_qa(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id)
    raise ValueError(f"Unsupported text head-mask dataset {dataset_key}")


def head_mask_load_image_task(task_data_path, task_image_dir, dataset_name=None, example_idx=-1, example_id=None, variant_cfg=None):
    dataset_key = infer_head_mask_image_dataset(task_data_path, dataset_name=dataset_name)
    if dataset_key == "mm_image":
        return head_mask_load_mm_niah_image(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id, variant_cfg=variant_cfg)
    if dataset_key == "identical_image":
        return head_mask_load_identical_image(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id, variant_cfg=variant_cfg)
    if dataset_key == "vrag":
        return head_mask_load_vrag(task_data_path, task_image_dir, example_idx=example_idx, example_id=example_id)
    raise ValueError(f"Unsupported image head-mask dataset {dataset_key}")


def head_mask_load_mm_niah_image_describe(task_data_path, task_image_dir, example_idx=-1):
    # This follows a slightly logic as MM NIAH. The needle is not within an image but within the text image haystack. 
    # MM NIAH considers the image it is pasted to as haystack as well
    user_template = "You are given interleaved text and images. Please answer the question based on the given text and images. Write your answer in the following format:\nAnswer: [answer]\n\n{context}\n\nQuestion: {question}"
    system_template = "Answer:"

    data = load_dataset("json", data_files=task_data_path)["train"]

    indices = [example_idx] # the default example_idx is -1, to use the last example. use different example from detection 

    key_list = sorted(set(data["id"]), key=lambda x: int(x.split("-")[-1]))
    key_list = [key_list[i] for i in indices]
    data = data.filter(lambda x: x["id"] in key_list, num_proc=8)
    data = data.select([0]) # only select the depth=0 example

    def update(sample):
        paragraph_list = sample["ctxs"]
        needle = sample["positive_ctxs"]
        assert len(needle) == 1
        needle = needle[0]

        # Remove needle image (marked with has_ans=True)
        needle_entry = None
        for p in paragraph_list:
            if p.get("has_ans"):
                needle_entry = p
                break
        assert needle_entry is not None, "Could not find needle in context"
        paragraph_list = [p for p in paragraph_list if not p.get("has_ans")]

        # Find needle's position in image_list and remove it
        img_idx = 0
        for p in sample["ctxs"]:
            if p.get("has_ans"):
                break
            if p.get("type") == "image":
                img_idx += 1
        image_list = list(sample["image_list"])
        needle_image = image_list.pop(img_idx)

        passage_text = " ".join([p["text"] for p in paragraph_list])
        image_list = [os.path.join(task_image_dir, img) for img in image_list]

        # Needle is the image
        needle_image_path = os.path.join(task_image_dir, needle_image)

        return {
            "context": passage_text,
            "image_list": image_list,
            "needle": {"type": "image", "image": needle_image_path},
        }


    data = data.map(update, remove_columns=["ctxs", "positive_ctxs", "needle_image_list", "choices_image"])

    from rouge_score import rouge_scorer as rs
    _scorer = rs.RougeScorer(['rouge1'], use_stemmer=True)

    return {
        "data": data,
        "system_template": system_template,
        "user_template": user_template,
        "metric": lambda ans, pred: _scorer.score(ans, pred)['rouge1'].recall * 100,
    }
