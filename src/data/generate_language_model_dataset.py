import os
import numpy as np



list_selected_test_avp = [8,10,18,23]
list_selected_test_lvt = [0,6,7,13]


print('Test...')

c = 0

with open('data/interim/sequence_datasets/sequence_dataset_test.txt', 'a') as txt_file:

    # AVP Personal

    path_csv = 'data/external/AVP_Dataset/Personal'

    list_csv = []
    for path, subdirs, files in os.walk(path_csv):
        for filename in files:
            if filename.endswith('.csv'):
                list_csv.append(os.path.join(path, filename))
    list_csv = sorted(list_csv)
    list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

    list_csv = list_csv[2::5]

    for i in list_selected_test_avp:
        classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
        classes_string = ' '.join(classes_list)
        txt_file.write(classes_string)
        txt_file.write('\n')
        c += 1

    # AVP Fixed

    path_csv = 'data/external/AVP_Dataset/Fixed'

    list_csv = []
    for path, subdirs, files in os.walk(path_csv):
        for filename in files:
            if filename.endswith('.csv'):
                list_csv.append(os.path.join(path, filename))
    list_csv = sorted(list_csv)
    list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

    list_csv = list_csv[2::5]

    for i in list_selected_test_avp:
        classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
        classes_string = ' '.join(classes_list)
        txt_file.write(classes_string)
        txt_file.write('\n')
        c += 1

    # LVT with HH as HHC

    path_csv = 'data/external/LVT_Dataset'

    list_csv = []
    for path, subdirs, files in os.walk(path_csv):
        for filename in files:
            if filename.endswith('.csv'):
                list_csv.append(os.path.join(path, filename))
    list_csv = sorted(list_csv)
    list_csv = list_csv[20:]

    for i in list_selected_test_lvt:
        classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
        for n in range(len(classes_list)):
            if classes_list[n]=='Kick':
                classes_list[n] = 'kd'
            elif classes_list[n]=='Snare':
                classes_list[n] = 'sd'
            elif classes_list[n]=='HH':
                classes_list[n] = 'hhc'
            else:
                print('Warning: unknown class')
        classes_string = ' '.join(classes_list)
        txt_file.write(classes_string)
        txt_file.write('\n')
        c += 1

    # LVT with HH as HHO

    path_csv = 'data/external/LVT_Dataset'

    list_csv = []
    for path, subdirs, files in os.walk(path_csv):
        for filename in files:
            if filename.endswith('.csv'):
                list_csv.append(os.path.join(path, filename))
    list_csv = sorted(list_csv)
    list_csv = list_csv[20:]

    for i in list_selected_test_lvt:
        classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
        for n in range(len(classes_list)):
            if classes_list[n]=='Kick':
                classes_list[n] = 'kd'
            elif classes_list[n]=='Snare':
                classes_list[n] = 'sd'
            elif classes_list[n]=='HH':
                classes_list[n] = 'hho'
            else:
                print('Warning: unknown class')
        classes_string = ' '.join(classes_list)
        txt_file.write(classes_string)
        txt_file.write('\n')
        c += 1

    print(c)


print('Train-Val')

for fold in range(8):

    list_selected_validation_avp = np.array([(fold*3)+0,(fold*3)+1,(fold*3)+2])
    list_selected_validation_lvt = np.array([(fold*2)+0,(fold*2)+1])

    c = 0

    with open('data/interim/sequence_datasets/sequence_dataset_validation_' + str(fold) + '.txt', 'a') as txt_file:

        # AVP Personal

        path_csv = 'data/external/AVP_Dataset/Personal'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

        list_csv = list_csv[2::5]

        for index in sorted(list_selected_test_avp, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_avp:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # AVP Fixed

        path_csv = 'data/external/AVP_Dataset/Fixed'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

        list_csv = list_csv[2::5]

        for index in sorted(list_selected_test_avp, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_avp:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # LVT Fixed with HH as HHC

        path_csv = 'data/external/LVT_Dataset'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv = list_csv[20:]

        for index in sorted(list_selected_test_lvt, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_lvt:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                for n in range(len(classes_list)):
                    if classes_list[n]=='Kick':
                        classes_list[n] = 'kd'
                    elif classes_list[n]=='Snare':
                        classes_list[n] = 'sd'
                    elif classes_list[n]=='HH':
                        classes_list[n] = 'hhc'
                    else:
                        print('Warning: unknown class')
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # LVT Fixed with HH as HHO

        path_csv = 'data/external/LVT_Dataset'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv = list_csv[20:]

        for index in sorted(list_selected_test_lvt, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_lvt:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                for n in range(len(classes_list)):
                    if classes_list[n]=='Kick':
                        classes_list[n] = 'kd'
                    elif classes_list[n]=='Snare':
                        classes_list[n] = 'sd'
                    elif classes_list[n]=='HH':
                        classes_list[n] = 'hho'
                    else:
                        print('Warning: unknown class')
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

    print(c)

    c = 0

    with open('data/interim/sequence_datasets/sequence_dataset_train_' + str(fold) + '.txt', 'a') as txt_file:

        # AVP Personal

        path_csv = 'data/external/AVP_Dataset/Personal'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

        list_csv = list_csv[2::5]

        for index in sorted(list_selected_test_avp, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_avp:
                continue
            else:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # AVP Fixed

        path_csv = 'data/external/AVP_Dataset/Fixed'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv.sort(key = lambda f:int(''.join(filter(str.isdigit,f))))

        list_csv = list_csv[2::5]

        for index in sorted(list_selected_test_avp, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_avp:
                continue
            else:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # LVT Fixed with HH as HHC

        path_csv = 'data/external/LVT_Dataset'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv = list_csv[20:]

        for index in sorted(list_selected_test_lvt, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_lvt:
                continue
            else:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                for n in range(len(classes_list)):
                    if classes_list[n]=='Kick':
                        classes_list[n] = 'kd'
                    elif classes_list[n]=='Snare':
                        classes_list[n] = 'sd'
                    elif classes_list[n]=='HH':
                        classes_list[n] = 'hhc'
                    else:
                        print('Warning: unknown class')
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # LVT Fixed with HH as HHO

        path_csv = 'data/external/LVT_Dataset'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))
        list_csv = sorted(list_csv)
        list_csv = list_csv[20:]

        for index in sorted(list_selected_test_lvt, reverse=True):
            del list_csv[index]

        for i in range(len(list_csv)):
            if i in list_selected_validation_lvt:
                continue
            else:
                classes_list = np.loadtxt(list_csv[i], delimiter=',', usecols=1, dtype=np.unicode_)
                for n in range(len(classes_list)):
                    if classes_list[n]=='Kick':
                        classes_list[n] = 'kd'
                    elif classes_list[n]=='Snare':
                        classes_list[n] = 'sd'
                    elif classes_list[n]=='HH':
                        classes_list[n] = 'hho'
                    else:
                        print('Warning: unknown class')
                classes_string = ' '.join(classes_list)
                txt_file.write(classes_string)
                txt_file.write('\n')
                c += 1

        # BTX Fixed

        path_csv = 'data/external/Beatbox_Set'

        list_csv = []
        for path, subdirs, files in os.walk(path_csv):
            for filename in files:
                if filename.endswith('.csv'):
                    list_csv.append(os.path.join(path, filename))

        list_csv = sorted(list_csv[:14])

        for i in range(len(list_csv)):

            classes_list_pre = []
            labels = np.genfromtxt(list_csv[i], dtype='S', delimiter=',', usecols=1).tolist()
            for h in range(len(labels)):
                classes_list_pre.append(labels[h].decode("utf-8"))
            classes_list_pre = np.array(classes_list_pre)

            onsets = np.loadtxt(list_csv[i], delimiter=',', usecols=0)

            '''delete_classes = []
            for n in range(1,len(onsets)):
                #print(onsets[n])
                if onsets[n]-onsets[n-1]<0.005:
                    if classes_list[n]=='hc' or classes_list[n]=='ho':
                        delete_classes.append(n)

            print(len(classes_list_pre))
            print(len(delete_classes))
            classes_list_pre = np.delete(classes_list_pre, delete_classes)
            print(len(classes_list_pre))'''

            classes_list = []
            for n in range(len(classes_list_pre)):
                if classes_list_pre[n]=='k':
                    classes_list.append('kd')
                elif classes_list_pre[n]=='sb' or classes_list_pre[n]=='sk' or classes_list_pre[n]=='s':
                    classes_list.append('sd')
                elif classes_list_pre[n]=='hc':
                    classes_list.append('hhc')
                elif classes_list_pre[n]=='ho':
                    classes_list.append('hho')

            classes_string = ' '.join(classes_list)
            txt_file.write(classes_string)
            txt_file.write('\n')
            c += 1

    print(c)

