
    #region IMPORTs
import torch
import sys
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.utils import shuffle
    #endregion

 

    #region BERT
class Bert:
    '''
        init Parameter :
        mode = 0 or 1 ( 0= FilterMode / 1 = TrainMode)
        data = if mode 0 = model_sav else csv_name 
    '''
    def __init__(self, mode, data): # 초기화
        self.MODEL_NAME = "bert-base-multilingual-cased"
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        if mode == 0:
            self.model_sav = data
        else:
            self.csv_name = data

    class SingleSentDataset(torch.utils.data.Dataset): # 데이터 전처리용 클레스
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
        
    def preprocess(self): # 학습 데이터 준비
        # 학습 데이터 준비
        data = pd.read_csv('./Dataset/'+self.csv_name)
        data = shuffle(data, random_state = 202006)

        #학습, 시험 데이터 스플릿
        train_data = data[:296]
        test_data = data[296:]
        
        tokenized_train_sentences = self.tokenizer(
            list(train_data['Text']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            )

        tokenized_test_sentences = self.tokenizer(
            list(test_data['Text']),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            )

        self.train_label = train_data['Completion'].values
        self.test_label = test_data['Completion'].values

        self.train_dataset = SingleSentDataset(tokenized_train_sentences, train_label)
        self.test_dataset = SingleSentDataset(tokenized_test_sentences, test_label)

    def train(self, model_name): # 학습
        # 하이퍼 파라미터
        training_args = TrainingArguments(
            output_dir='./',          # output directory
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./',            # directory for storing logs
            logging_steps=500,
            save_steps=500,
            save_total_limit=2
        )

        trainer = Trainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,                  # training arguments, defined above
            train_dataset=self.train_dataset,         # training dataset
        )
        trainer.train() 
        torch.save(model, "./Model/Model_sav/"+model_name)

    def compute_metrics(pred): # 테스트용 함수
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def predict_test(self): # 테스트 데이터 확인
        trainer = Trainer(
            model=torch.load("./Model/Model_sav/"+self.model_sav),
            args=training_args,
            compute_metrics=compute_metrics
        )
        trainer.evaluate(eval_dataset=self.test_dataset)

    def sentences_predict(self, sent): # predict 함수
        self.model = torch.load("./Model/Model_sav/"+self.model_sav) 
        tokenized_sent = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
        )
        with torch.no_grad():# 그라디엔트 계산 비활성화
            outputs = self.model(
                input_ids=tokenized_sent['input_ids'],
                attention_mask=tokenized_sent['attention_mask'],
                token_type_ids=tokenized_sent['token_type_ids']
            )
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits)
        return result
    #endregion



    #region TEST
if __name__ == "__main__": # 메인 테스트
    bt = Bert(0, "model_save3.pt")
    print(bt.sentences_predict("영화 개재밌어 ㅋㅋㅋㅋㅋ"))
    #endregion
