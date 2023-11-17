def get_data_from_file(file):
    data = []
    with open(file) as f:
        temp = []
        for lines in f.readlines():
            if (len(lines) == 1):

                if len(temp) > 0:
                    data.append(temp)
                    temp = []
            else:
                lists = lines.split(" ")
                temp.append((lists[0], str.strip(lists[-1])))
    if len(temp) > 0:
        data.append(temp)
        temp = []
    return data
def get_train_data(language):
    return get_data_from_file("{}/train.txt".format(language))
        
def get_valid_data(language):
    return get_data_from_file("{}/validation.txt".format(language))

