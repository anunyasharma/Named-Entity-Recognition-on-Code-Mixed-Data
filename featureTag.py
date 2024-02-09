import csv
def getTag():
    data = []
    sentence = 0
    with open('Twitterdata/annotatedData.csv', 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        tmp = []
        for row in reader:
            if row:
                string = "sent: " + str(sentence)
                tmp.append(string)
                tmp.append(row[0])
                if len(row) > 1 and row[1]:
                    tmp.append(row[1])
                else:
                    tmp.append("Other")
            else:
                tmp.append('')
                sentence += 1
            data.append(tmp)
            tmp = []
    csv_columns = ["Sent", "Word", "Tag"]

    with open('taggedData.csv', 'w', newline='', encoding='utf-8') as myFile:
        writer = csv.writer(myFile)
        writer.writerow(csv_columns)
        writer.writerows(data)

getTag()