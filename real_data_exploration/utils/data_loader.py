import scipy
import os
import numpy as np

def get(x):
    return x[0][0]

def found_ratname(r, ratnames_age, ratnames_old):
    for k in ratnames_age:
        if r in k : return True
    for k in ratnames_old:
        if r in k : return True
    return False

def search_struct_data(r, ratnames_age, ratnames_old):
    r = r.lower()
    if found_ratname(r, ratnames_age, ratnames_old) : return True

    if len(r.split('_'))>2:
        rat_ds = '_'.join(r.split('_')[:2])
        if found_ratname(rat_ds, ratnames_age, ratnames_old) : return True
    
    if 't' in r:
        r = r.split('t')[0]
        if found_ratname(r, ratnames_age, ratnames_old) : return True

    if found_ratname(r.split('_')[0], ratnames_age, ratnames_old):
        print(f"\t[*] Only ratname {r.split('_')[0]} (of {r}) found in processed struct data")

    return False

def load_data_struct(mat_dir: str, load_dirs: list[str], keep_full_ratname=False):
    load_dirs_path = [os.path.join(mat_dir, ld) for ld in load_dirs]

    data_ld_dict = {}
    ratnames = []

    for ld, ld_path in zip(load_dirs, load_dirs_path):
        print(ld)
        struct_data = {}
        for file in os.listdir(ld_path):
            if file.endswith('.mat'):
                name = file.split('.')[0]
                struct_data[name] = scipy.io.loadmat(os.path.join(ld_path, file))
        data_dict = {}
        for k in struct_data.keys():
            print(k)

            if not k.lower().startswith('r'):
                print(f"File {k} does not start with 'r', skipping")
                continue

            ratname = (
                '_'.join(k.split('_')[:2]).lower()
                if keep_full_ratname
                else k.split('_')[0].lower()
            )
            
            if ratname not in data_dict.keys():
                data_dict[ratname] = {}
            
            d = get(struct_data[k]['tmpS'])
            d_keys = list(d.dtype.names)

            dataset = d[d_keys.index('dataset')][0].split('_')[-1]
            ppm = d[d_keys.index('ppm')][0] if 'ppm' in d_keys else 400
            ratnames.append(f"{ratname}_{dataset}")

            ages = d[d_keys.index('age')][0] # age 40 denotes adult
            env_types = d[d_keys.index('envType')][0]
            pos = d[d_keys.index('positions')][0] if 'muessig' in ld or 'science' in ld else d[d_keys.index('posData')][0] # x, y position in pixels
            hd = d[d_keys.index('directions')][0] if 'muessig' in ld or 'science' in ld else d[d_keys.index('dirData')][0] # degrees
            speed = d[d_keys.index('speed')][0] # cm/s

            # if sample rate is not present, it is 50 Hz
            sample_rate = d[d_keys.index('sampleRate')][0] if 'sampleRate' in d_keys else 50 # Hz

            # iterate through trials
            n_trials = len(ages)
            for trial_idx in range(n_trials):
                t = {}
                age = ages[trial_idx]
                env = env_types[trial_idx][0]
                p = pos[trial_idx]
            
                if np.isnan(age) and (len(env) == 0) and (p.shape[-1] == 0):
                    continue

                age = str(int(age))
                if age == '40' : age = '100'
                if age not in data_dict[ratname].keys():
                    data_dict[ratname][age] = {}
                    data_dict[ratname][age]['trials'] = []

                t['name'] = trial_idx
                t['environment'] = 'hp' if (env == 'fam') or (env == 'hp') else ''
                t['ppm'] = ppm
                t['sample_rate'] = sample_rate[trial_idx] if 'sampleRate' in d_keys else 50
                t['x'] = p[:,0]
                t['y'] = p[:,1]
                t['speed'] = speed[trial_idx].squeeze()
                t['hd'] = hd[trial_idx].squeeze()
                t['duration'] = len(t['x'])/t['sample_rate']

                data_dict[ratname][age]['trials'].append(t)
            print(f"\t{n_trials} trial(s)")
            print()
            
        data_ld_dict[ld] = data_dict
        print('\n')

    return data_ld_dict, ratnames
