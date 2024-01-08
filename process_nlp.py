import os
import os
import json
import nltk
import pymorphy2


db_fileName = "./data_soc.json"

def add_data(text):
    import pathlib
    path = pathlib.Path(db_fileName)
    content = []
    data = get_pattern(text)
    # data = add_print_text(data)
    if path.exists():
        with open(db_fileName, "r", encoding="UTF8") as file:
            jsoncontent = file.read()
        content = json.loads(jsoncontent)
        content.append(data)
        jsonstring = json.dumps(content, ensure_ascii=False)
        with open(db_fileName, "w", encoding="UTF8") as file:
            file.write(jsonstring)
    else:
        content.append(data)
        jsonstring = json.dumps(content, ensure_ascii=False)
        with open(db_fileName, "w", encoding="UTF8") as file:
            file.write(jsonstring)
    return content


def load_db():
    import pathlib
    path = pathlib.Path(db_fileName)
    if path.exists():
        with open(db_fileName, "r", encoding="UTF8") as file:
            jsoncontent = file.read()
        content = json.loads(jsoncontent)
        return content
    else:
        return [{}]
    

def clear_db():
    import pathlib
    path = pathlib.Path(db_fileName)
    if path.exists():
        os.remove(db_fileName)


def data_proc(filename, save_filename, threshold=0):
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    text = ""
    count_messages = len(messages)
    print(count_messages)
    num = 0
    proc_messages = []  
    for m in messages:
        text = m["text"]
        print(f"{num / count_messages * 100}     {count_messages-num}     {num} / {count_messages}")
        num += 1
        if len(text) < threshold:
            continue
        line = get_pattern(text)
        line["date"] = m["date"]
        line["message_id"] = m["message_id"]
        line["user_id"] = m["user_id"]
        line["reply_message_id"] = m["reply_message_id"]
        proc_messages.append(line)
    jsonstring = json.dumps(proc_messages, ensure_ascii=False)
    with open(save_filename, "w", encoding="UTF8") as file:
        file.write(jsonstring)

def load_data_proc(filename):
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    return messages


def remove_digit(data):
    str2 = ''
    for c in data:
        if c not in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '«', '»', '–', "\""):
            str2 = str2 + c
    data = str2
    return data


def remove_punctuation(data):
    str2 = ''
    import string
    pattern = string.punctuation
    for c in data:
        if c not in pattern:
            str2 = str2 + c
        else:
            str2 = str2 + ""
    data = str2
    return data


def remove_stopwords(data):
    str2 = ''
    from nltk.corpus import stopwords
    russian_stopwords = stopwords.words("russian")
    for word in data.split():
        if word not in (russian_stopwords):
            str2 = str2 + " " + word
    data = str2
    return data


def remove_short_words(data, length=1):
    str2 = ''
    for line in data.split("\n"):
        str3 = ""
        for word in line.split():
            if len(word) > length:
                str3 += " " + word
        str2 = str2 + "\n" + str3
    data = str2
    return data


def remove_paragraf_to_lower(data):
    data = data.lower()
    data = data.replace('\n', ' ')
    return data


def remove_all(data):
    data = remove_digit(data)
    data = remove_punctuation(data)
    data = remove_stopwords(data)
    data = remove_short_words(data, length=3)
    data = remove_paragraf_to_lower(data)
    return data


def get_KeyBERT(text):
    from keyBERT import keyBERT
    kw_model = keyBERT()
    # keywords = kw_model.extract_keywords(doc)
    numOfKeywords = 20
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english',
                            use_maxsum=True, nr_candidates=20, top_n=numOfKeywords)
    l=[]
    for item in keywords:
        l.append(list(item))
    return l


def get_RAKE(text):
    from rake_nltk import Rake
    r = Rake(language="russian")
    r.extract_keywords_from_text(text)
    numOfKeywords = 20
    keywords = r.get_ranked_phrases()[:numOfKeywords]
    return keywords


def get_pattern(text):
    line = {}
    line['text'] = text.strip()
    line['remove_all'] = remove_all(text).strip()
    line['KEYWORDS'] = get_RAKE(text)
    return line


def add_print_text(data):   
    KEY_text =[]
    for item in data['KEYWORDS']:
        KEY_text.append(item[0])
    str1 = str(f"Исходный текст: {data['text']} \n\n"
            f" KEYWORDS: {KEY_text} \n\n")
    data['print_text'] = str1
    # print(str1)
    return data


def get_normal_form_mas(words):
    morph = pymorphy2.MorphAnalyzer()
    result = []
    for word in words.split():
        p = morph.parse(word)[0]
        result.append(p.normal_form)
    return result


def get_normal_form(words):
    morph = pymorphy2.MorphAnalyzer()
    p = morph.parse(words)[0]
    return p.normal_form


def load_data(filename='data.txt'):
    with open(filename, "r", encoding='utf-8') as file:
        data = file.read()
    return data

def remove_from_patterns(text, pattern):
    str2 = ''
    for c in text:
        if c not in pattern:
            str2 = str2 + c
    return str2

def display(text):
    print(text) 
    print("--------------------------------")

def remove_paragraf_and_toLower(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = ' '.join([k for k in text.split(" ") if k])
    return text


def nltk_download():
    nltk.download('stopwords')
    nltk.download('punkt')
    

def calc_intersection_list(list1, list2):
    count = 0
    for item1 in list1:
        for item2 in list2:
            count += calc_intersection_text(item1, item2)
    return count

def calc_intersection_text(text1, text2):
    count = 0
    text1 = str(text1)
    text2 = str(text2)
    for item1 in text1.split():
        for item2 in text2.split():
            if item1 == item2:
                count += 1
    return count

def calc_score(data1, data2):
    pass


def find_cl(filename):
    messages = load_data_proc(filename)
    data_cl = load_db()
    find_data = []
    for m in messages:
        item = m
        num = 0
        item["KW_COUNT"] = 0
        item["KW_NUM"] = 0
        for cl in data_cl:
            intersect_KEY = calc_intersection_list(m['KEYWORDS'], cl['KEYWORDS'])
            if intersect_KEY>item["KW_COUNT"]:
                item["KW_COUNT"] = intersect_KEY
                item["KW_NUM"] = num
            num += 1
        find_data.append(item)
    jsonstring = json.dumps(find_data, ensure_ascii=False)
    with open("./find_data.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)


def find_soc(filename, counts=3):
    messages = load_data_proc(filename)
    find_data = []
    KEY_set=set()
    for m in messages:
        KEY_set.add(m['KW_COUNT'])
    KEY_s = max(KEY_set)
    dif = KEY_s-counts
    if dif < 1:
        dif = 1
    print(dif)
    print()
    print()
    for m in messages:
        if m['KW_COUNT'] >= dif and m['KW_COUNT'] > 0:
            # m = add_print_text(m)
            find_data.append(m)                     
    jsonstring = json.dumps(find_data, ensure_ascii=False)
    with open("./find_d.json", "w", encoding="UTF8") as file:
        file.write(jsonstring)
    return jsonstring    


def convertMs2String(milliseconds):
    import datetime
    dt = datetime.datetime.fromtimestamp(milliseconds )
    return dt


def convertJsonMessages2text(filename):
    with open(filename, "r", encoding="UTF8") as file:
        content = file.read()
    messages = json.loads(content)
    text = ""
    for m in messages:
        text += f"{convertMs2String(m['date'])} {m['message_id']}  {m['user_id']} {m['reply_message_id']}  {m['text']}  <br>\n"
    return text

def find_cl_text(text, counts=3):
    filename="d:/ml/chat/andromedica1.json"
    save_filename="./data_proc.json"
    clear_db()
    add_data(text)
    data_proc(filename, save_filename, 32)
    find_cl(save_filename)
    find_data=find_soc("./find_data.json", counts)
    return find_data

if __name__ == '__main__':
    # nltk_download()
    # add_data(s1)
    s1="Доброе утро! Пусть день будет ясным, А настроение только прекрасным. Пусть пробуждение радость несет, Без суеты, без тревог и забот. Рассвет пусть уносит унылую тень, Счастливой улыбкой встречай новый день."
    s1="Привет"    # t = get_pattern(data)
    # print(t)

    # filename="d:/ml/chat/andromedica1.json"
    # save_filename="./data_proc.json"
    
    # # data_proc(filename, save_filename, 32)
    # find_cl(save_filename)
    # find_soc("./find_data.json", 4)
    
    # res = find_cl_text(s1, 2)
    # print(res)
