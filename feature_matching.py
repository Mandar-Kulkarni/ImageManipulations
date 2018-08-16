import cv2
import numpy
import os
import csv

def feature_extract(f):


    features = f.readlines()
    desc = []
    feature_array = []
    k = 0

    for i, feature in enumerate(features):

        feature = feature.split()
        if (i == 0):
            num_of_features = int(feature[0])
        if(i >= 8*k+2 and i <= 8*k+8):
            for j in range(len(feature)):
                desc.append(int(feature[j]))

        elif(i == (8*k + 9)):
            k+=1
            feature_array.append(desc)
            desc = []
            #print('outside', len(desc), k)
    else:
        feature_array.append(desc)

    feature_array = numpy.array(feature_array)

    final_feature = numpy.reshape(feature_array, (num_of_features, 128))
    return final_feature


def read_folder():
    rootdir = "sift"
    feature_set = []
    k = 0
    fileObjs = []
    for subdir, dirs, files in os.walk(rootdir):

        for j, file in enumerate(files):

            fileObj = open(os.path.join(subdir, file), "r")

            feature_set.append(feature_extract(fileObj))

            fileObjs.append(fileObj)
            k += 1
            #print(k)
    feature_set = numpy.array(feature_set)
    #print(feature_set[0][0], file)
    header = ['ImageName', 'FirstMatchImage', 'FirstMatchScore', 'SecondMatchImage', 'SecondMarchScore', 'ThirdMatchImage', 'ThirdMatchScore',
                         'FourthMatchImage', 'FourthMatchScore', 'FifthMatchImage', 'FifthMatchScore']
    with open('output.csv', 'w') as file:
        writer1 = csv.writer(file, delimiter=',')

        writer1.writerow(header)
    file.close()
    return feature_set, fileObjs

def write_csv(file_names):

    with open('output.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(file_names)

def matches_ratio(similarities, len_a):
    return(1- float(similarities/len_a))


def flann(set_f, objs):

    for n in range(150):
        folder = []
        for m in range(150):

            write_objs = []
            similar = []
            match_similar = 0


            FLANN_INDEX_KDTREE = 0
            index = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search = dict(checks=50)   # or pass empty dictionary

            flann = cv2.FlannBasedMatcher(index, search)

            matches = flann.knnMatch(numpy.asarray(set_f[n], dtype=numpy.float32),numpy.asarray(set_f[m], dtype=numpy.float32),k=1)
            write_objs.append(str(objs[m]).split('\'')[1])

            for i in range(len(matches)):
                similar.append(matches[i][0].distance)

            threshold = 0.4*max(similar)
            print('threshold, len(similar)', threshold, len(similar))
            for i in range(len(similar)):
                if(similar[i]<threshold):
                    match_similar += 1

            print(matches_ratio(match_similar, len(similar)))
            folder.append(matches_ratio(match_similar, min(len(similar), len(set_f[m]))))


        print(folder)
        folder = numpy.asarray(folder, dtype=numpy.float32)
        five_best = folder.argsort()[-5:][::-1]
        print(five_best)
        for j in range(5):
            print(objs[j])
            write_objs.append(str(objs[j]).split('\'')[1])
            write_objs.append(int(folder[five_best[j]]*100))


        write_csv(write_objs)

set_of_features = numpy.array([])


set_of_features, objs = read_folder()

print(len(set_of_features[0]), len(set_of_features[1]))
flann(set_of_features, objs)
