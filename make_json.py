import pickle
import os
import json
import glob

import torchaudio

def sp2clock(idx, sr=16000):
    sec = int(idx/16000)
    minu = int(sec//60)
    result = '{:02d}:{:02d}'.format(minu, sec-60*minu)
    return result

def make_NONE(json_output):
    for s in json_output['task2_answer'][0]:
        for idx, d in enumerate(json_output['task2_answer'][0][s]):
            for k, v in d.items():
                for classes in json_output['task2_answer'][0][s][idx][k]:
                    for c in classes:
                        if not json_output['task2_answer'][0][s][idx][k][0][c]:
                            json_output['task2_answer'][0][s][idx][k][0][c] = ['NONE']
    return json_output

# PATHS
pickle_paths = sorted([os.path.join('pickles', x) for x in glob.glob1('pickles', '*.pkl')]) # num: 15
json_output_path = 'output.json'

# define output format
json_output = {
        'task2_answer': [{
                'set_1':[
                        {'drone_1': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_2': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_3': [
                                {'M': [],
                                'W': [],
                                'C': []}]}],
                'set_2':[
                        {'drone_1': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_2': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_3': [
                                {'M': [],
                                'W': [],
                                'C': []}]}],
                'set_3':[
                        {'drone_1': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_2': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_3': [
                                {'M': [],
                                'W': [],
                                'C': []}]}],
                'set_4':[
                        {'drone_1': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_2': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_3': [
                                {'M': [],
                                'W': [],
                                'C': []}]}],
                'set_5':[
                        {'drone_1': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_2': [
                                {'M': [],
                                'W': [],
                                'C': []}]},
                        {'drone_3': [
                                {'M': [],
                                'W': [],
                                'C': []}]}]
        }]
    }

for pickle_path in pickle_paths:
    # load pickle, wav
    with open(pickle_path, 'rb') as p:
        pickle_data = pickle.load(p)
    wav_path = pickle_data['output_path']
    set_name, drone_name, _ = wav_path.split('/')[-1].split('_')
    set_num, drone_num = int(set_name[-1]), int(drone_name[-1])
    set_name = set_name[:3] + '_' + str(set_num)
    drone_name = drone_name[:5] + '_' + str(drone_num)

    wav, sr = torchaudio.load(wav_path)

    # get VAD timestamps
    timestamps = pickle_data['time']

    # iter and load wav, pass gender classification
    for ts in timestamps:
        start_ts, end_ts = ts
        start_idx, end_idx = int(start_ts*sr), int(end_ts*sr)
        wav_chunk = wav[:,start_idx:end_idx]
        
        '''
        # pass wav_chunk to model
        예를들어, 3초짜리 인풋 -> 'M', 'W', 'C' 로 나오면
        return: {'M': 3/2*sr, 'W': None, 'C': None}
        '''
        output = {'M': 1*sr, 'W': None, 'C': 2*sr} # model output example

        # append to json_output
        for class_type, class_sp_idx in output.items():
            if class_sp_idx:
                json_output['task2_answer'][0][set_name][drone_num-1][drone_name][0][class_type].append(sp2clock(start_idx+class_sp_idx))

# save json
with open(json_output_path, 'w') as j:
    data = json.dumps(make_NONE(json_output), indent=4)
    j.write(data)