from collections import Counter
import random
from tqdm import tqdm
class CRF():
    def __init__(self,tags):
        self.weight={}
        self.tags=tags
        self.start = 'BOS'
        self.stop = 'EOS'

        pass
    def viterbi_decoder(self,sentence):

        score = [[0 for _ in range(len(self.tags))] for _ in range(len(sentence))]

        prev = [[0 for _ in range(len(self.tags))] for _ in range(len(sentence))]
        for i in range(len(sentence)):
            # 为句子第i的单词打标签self.tags[k]
            for k in range(len(self.tags)):
                max_score = float('-inf')
                max_prev = -1
                prev_tags = 1 if i == 0 else len(self.tags)
                # 如果上一个标签是
                for j in range(prev_tags):
                    prev_tag = self.tags[j] if i !=0 else self.start
                    prev_score = score[i-1][j] if i-1>=0 else 0 
                    temp_score =prev_score + self.features_scoring(self.word2features(sentence,i,self.tags[k],prev_tag))
                    if temp_score > max_score:
                        max_score = temp_score
                        max_prev = j

                score[i][k] = max_score
                prev[i][k] = max_prev
                
        tag = []
        tail_tag = 0
        max_score = float('-inf')
        for j in range(len(self.tags)):
            if score[-1][j] > max_score:
                max_score = score[-1][j]
                tail_tag = j

        tag.append(tail_tag)
        for i in range(len(sentence)-1,0,-1):
            tag.insert(0,prev[i][tail_tag])
            tail_tag = prev[i][tail_tag]
            
        tag = [self.tags[i] for i in tag]
        return tag

    def update_weights(self,sentence,true_labels,predict_labels,learning_rate):
        for i in range(len(sentence)):
            prev_lable = predict_labels[i-1] if i >=1 else self.start
            pred_features = self.word2features(sentence,i,predict_labels[i],prev_lable)

            prev_lable = true_labels[i-1] if i >=1 else self.start
            true_features = self.word2features(sentence, i, true_labels[i],prev_lable)

            union = set(pred_features)|set(true_features)
            pred_features_counter = Counter(pred_features)
            true_features_counter = Counter(true_features)
            for feature in union:
                self.weight.setdefault(feature,0)
                self.weight[feature]+= learning_rate*(true_features_counter.get(feature,0)- pred_features_counter.get(feature,0))



    def train(self, training_data, epochs=5, learning_rate=0.1):
        progress_bar = tqdm(total=epochs*len(training_data), desc='Training Progress')
        i = 0
        for epoch in range(epochs):
            random.shuffle(training_data)
            for sentences in training_data:
                true_labels = [label for word, label in sentences]
                sentence = [word for word, label in sentences]
                predicted_labels = self.viterbi_decoder(sentence)
                self.update_weights(sentence, true_labels, predicted_labels, learning_rate)
                progress_bar.update(1)


        progress_bar.close()


    def features_scoring(self,features):
        score = 0
        for feature in features:
            score = score + self.weight.get(feature,0)
        return score
    
# 定义特征模板
    def word2features(self,sent, i, current_label,previous_label):
        word = sent[i]
        # features = [
        #     'word.lower=' + word.lower(),
        #     # 'word[-3:]=' + word[-3:],
        #     # 'word[-2:]=' + word[-2:],
        #     # 'word[-1:]=' + word[-1:],
        #     # 'word.isupper=%s' % word.isupper(),
        #     # 'word.istitle=%s' % word.istitle(),
        #     # 'word.isdigit=%s' % word.isdigit(),
        # ]
    
        # for offset in [-2, -1, 0, 1, 2]:
        #     if 0 <= i + offset < len(sent) - 1:
        #         word_n = sent[i + offset]
        #         word_next = sent[i + offset + 1]
        #         features.extend([
        #             '{:02d}:{}/{}'.format(offset, word_n, word_next),
        #         ])

        
        unigram = []
        bigram = []
        unigram.append("U02:{}-{}".format(sent[i],current_label))
        unigram.append("U06:{}/{}-{}".format(sent[i-1] if i-1>=0 else self.start,sent[i],current_label))
        unigram.append("U08:{}/{}-{}".format(sent[i],"EOS" if i+1 ==len(sent) else sent[i+1],current_label))

        # bigram.append("B02:{}-{}/{}".format(sent[i],current_label,previous_label))
        bigram.append("B06:{}/{}-{}/{}".format(sent[i-1] if i-1>=0 else self.start,sent[i],current_label,previous_label))
        # bigram.append("B08:{}/{}-{}/{}".format(sent[i],"EOS" if i+1 ==len(sent) else sent[i+1],current_label,previous_label))

        return unigram+bigram

# test
train_data = []
with open("NER/English/train.txt") as f:
    temp = []
    for lines in f.readlines():
        if(len(lines)==1):

            if len(temp)>0:
                train_data.append((temp))
                temp = []
        else:
            lists = lines.split(" ")
            temp.append((lists[0],str.strip(lists[-1])))

train_data

valid_data = []
with open("NER/English/validation.txt") as f:
    temp = []
    for lines in f.readlines():
        if(len(lines)==1):

            if len(temp)>0:
                valid_data.append((temp))
                temp = []
        else:
            lists = lines.split(" ")
            temp.append((lists[0],str.strip(lists[-1])))

valid_data
import random
crf = CRF(["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC","B-MISC","I-MISC"])
words = [word for word,_ in valid_data[1]]
labels = [label for _,label in valid_data[1]]
print(words)
crf.update_weights(words,labels,random.sample(crf.tags*20,len(labels)),1)
print(words)

print(len(crf.viterbi_decoder(words)))
print(crf.viterbi_decoder(words))
print(len(labels))
