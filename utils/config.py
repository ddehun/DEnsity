TURN_TOKEN = "[SEPT]"

LM_HIDDEN_DIM_MAP = {"bert-base-uncased": 768, "gpt2": 768}


# Dataset statistics for cross-check: https://arxiv.org/pdf/1905.01969.pdf
DATASET_CONFIG = {
    "convai2": {"url": "http://parl.ai/downloads/convai2/convai2_fix_723.tgz"},  # self_original should be used
    "dailydialog": {"url": "http://yanran.li/files/ijcnlp_dailydialog.zip"},
}


DATASET_TO_KEYS = {
    "dd": {
        "num_negative": 15,
        "train_fname": "./data/dd/train/dialogues_train.txt",
        "valid_fname": "./data/dd/validation/dialogues_validation.txt",
        "test_fname": "./data/dd/test/dialogues_test.txt",
    },
    "convai2": {
        "num_negative": 19,
        "train_fname": "./data/convai2/train_self_original.txt",
        "valid_fname": "./data/convai2/valid_self_original.txt",
        "test_fname": "./data/convai2/test_self_original.txt",
    },
}
