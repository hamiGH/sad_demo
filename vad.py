import os
import torch
from torch.autograd import Variable
import numpy as np
import configparser as ConfigParser
from optparse import OptionParser
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN


class Vad:
    def __init__(self, gpu_number=None):

        if gpu_number is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cuda:' + str(gpu_number) if torch.cuda.is_available() else "cpu")

        # Reading cfg file
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        options = self.read_conf(os.path.join(cur_dir, "cfg/vad.cfg"))
        #options = self.read_conf("./cfg/vad.cfg")

        pt_file = options.pt_file

        # [windowing]
        fs = int(options.fs)
        cw_len = int(options.cw_len)
        self.cw_shift = int(options.cw_shift)

        # [cnn]
        cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
        cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
        cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
        cnn_use_laynorm_inp = self.str_to_bool(options.cnn_use_laynorm_inp)
        cnn_use_batchnorm_inp = self.str_to_bool(options.cnn_use_batchnorm_inp)
        cnn_use_laynorm = list(map(self.str_to_bool, options.cnn_use_laynorm.split(',')))
        cnn_use_batchnorm = list(map(self.str_to_bool, options.cnn_use_batchnorm.split(',')))
        cnn_act = list(map(str, options.cnn_act.split(',')))
        cnn_drop = list(map(float, options.cnn_drop.split(',')))

        # [dnn]
        fc_lay = list(map(int, options.fc_lay.split(',')))
        fc_drop = list(map(float, options.fc_drop.split(',')))
        fc_use_laynorm_inp = self.str_to_bool(options.fc_use_laynorm_inp)
        fc_use_batchnorm_inp = self.str_to_bool(options.fc_use_batchnorm_inp)
        fc_use_batchnorm = list(map(self.str_to_bool, options.fc_use_batchnorm.split(',')))
        fc_use_laynorm = list(map(self.str_to_bool, options.fc_use_laynorm.split(',')))
        fc_act = list(map(str, options.fc_act.split(',')))

        # [class]
        self.class_lay = list(map(int, options.class_lay.split(',')))
        class_drop = list(map(float, options.class_drop.split(',')))
        class_use_laynorm_inp = self.str_to_bool(options.class_use_laynorm_inp)
        class_use_batchnorm_inp = self.str_to_bool(options.class_use_batchnorm_inp)
        class_use_batchnorm = list(map(self.str_to_bool, options.class_use_batchnorm.split(',')))
        class_use_laynorm = list(map(self.str_to_bool, options.class_use_laynorm.split(',')))
        class_act = list(map(str, options.class_act.split(',')))

        # Converting context and shift in samples
        self.wlen = int(fs * cw_len / 1000.00)
        self.wshift = int(fs * self.cw_shift / 1000.00)

        self.fs = fs

        # Batch_dev
        self.Batch_dev = int(options.batch_dev)

        # Feature extractor CNN
        CNN_arch = {'input_dim': self.wlen,
                    'fs': fs,
                    'cnn_N_filt': cnn_N_filt,
                    'cnn_len_filt': cnn_len_filt,
                    'cnn_max_pool_len': cnn_max_pool_len,
                    'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                    'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                    'cnn_use_laynorm': cnn_use_laynorm,
                    'cnn_use_batchnorm': cnn_use_batchnorm,
                    'cnn_act': cnn_act,
                    'cnn_drop': cnn_drop,
                    }

        self.CNN_net = CNN(CNN_arch)
        self.CNN_net.to(self.device)
        # self.CNN_net.cuda()

        # Loading label dictionary
        # lab_dict = np.load(class_dict_file).item()

        DNN1_arch = {'input_dim': self.CNN_net.out_dim,
                     'fc_lay': fc_lay,
                     'fc_drop': fc_drop,
                     'fc_use_batchnorm': fc_use_batchnorm,
                     'fc_use_laynorm': fc_use_laynorm,
                     'fc_use_laynorm_inp': fc_use_laynorm_inp,
                     'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                     'fc_act': fc_act,
                     }

        self.DNN1_net = MLP(DNN1_arch)
        self.DNN1_net.to(self.device)
        # self.DNN1_net.cuda()

        DNN2_arch = {'input_dim': fc_lay[-1],
                     'fc_lay': self.class_lay,
                     'fc_use_batchnorm': class_use_batchnorm,
                     'fc_drop': class_drop,
                     'fc_use_laynorm': class_use_laynorm,
                     'fc_use_laynorm_inp': class_use_laynorm_inp,
                     'fc_use_batchnorm_inp': class_use_batchnorm_inp,
                     'fc_act': class_act,
                     }

        self.DNN2_net = MLP(DNN2_arch)
        self.DNN2_net.to(self.device)
        # self.DNN2_net.cuda()


        if pt_file != 'none':
            checkpoint_load = torch.load(os.path.join(cur_dir, pt_file), map_location=lambda storage, loc: storage)
            self.CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
            self.DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
            self.DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

        self.CNN_net.eval()
        self.DNN1_net.eval()
        self.DNN2_net.eval()

        self.CNN_net.share_memory()
        self.DNN1_net.share_memory()
        self.DNN2_net.share_memory()

    def read_conf(self, cfg=None):

        parser = OptionParser()
        parser.add_option("--cfg")  # Mandatory
        (options, args) = parser.parse_args()

        if (cfg != None):
            options.cfg = cfg

        cfg_file = options.cfg
        Config = ConfigParser.ConfigParser()
        Config.read(cfg_file)

        # [data]
        options.pt_file = Config.get('data', 'pt_file')

        # [windowing]
        options.fs = Config.get('windowing', 'fs')
        options.cw_len = Config.get('windowing', 'cw_len')
        options.cw_shift = Config.get('windowing', 'cw_shift')

        # [cnn]
        options.cnn_N_filt = Config.get('cnn', 'cnn_N_filt')
        options.cnn_len_filt = Config.get('cnn', 'cnn_len_filt')
        options.cnn_max_pool_len = Config.get('cnn', 'cnn_max_pool_len')
        options.cnn_use_laynorm_inp = Config.get('cnn', 'cnn_use_laynorm_inp')
        options.cnn_use_batchnorm_inp = Config.get('cnn', 'cnn_use_batchnorm_inp')
        options.cnn_use_laynorm = Config.get('cnn', 'cnn_use_laynorm')
        options.cnn_use_batchnorm = Config.get('cnn', 'cnn_use_batchnorm')
        options.cnn_act = Config.get('cnn', 'cnn_act')
        options.cnn_drop = Config.get('cnn', 'cnn_drop')

        # [dnn]
        options.fc_lay = Config.get('dnn', 'fc_lay')
        options.fc_drop = Config.get('dnn', 'fc_drop')
        options.fc_use_laynorm_inp = Config.get('dnn', 'fc_use_laynorm_inp')
        options.fc_use_batchnorm_inp = Config.get('dnn', 'fc_use_batchnorm_inp')
        options.fc_use_batchnorm = Config.get('dnn', 'fc_use_batchnorm')
        options.fc_use_laynorm = Config.get('dnn', 'fc_use_laynorm')
        options.fc_act = Config.get('dnn', 'fc_act')

        # [class]
        options.class_lay = Config.get('class', 'class_lay')
        options.class_drop = Config.get('class', 'class_drop')
        options.class_use_laynorm_inp = Config.get('class', 'class_use_laynorm_inp')
        options.class_use_batchnorm_inp = Config.get('class', 'class_use_batchnorm_inp')
        options.class_use_batchnorm = Config.get('class', 'class_use_batchnorm')
        options.class_use_laynorm = Config.get('class', 'class_use_laynorm')
        options.class_act = Config.get('class', 'class_act')

        # [optimization]
        options.batch_dev = Config.get('optimization', 'batch_dev')

        return options

    def str_to_bool(self, s):
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            raise ValueError

    def run(self, signal, fs):

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono')
            signal = signal[:, 0]

        # signal = signal.astype(np.float64)
        # Signal normalization
        signal = signal / np.abs(np.max(signal))

        # signal = torch.from_numpy(signal).float().cpu().contiguous()

        # split signals into chunks
        beg_samp = 0
        end_samp = self.wlen

        if signal.shape[0] < self.wlen:
            return np.zeros(signal.shape[0]), np.zeros(signal.shape[0])

        N_fr = int((signal.shape[0] - self.wlen) / (self.wshift))

        sig_arr = np.zeros([self.Batch_dev, self.wlen])

        # pout = Variable(torch.zeros(N_fr, self.class_lay[-1]).float().cuda().contiguous())
        pout = np.zeros(N_fr + 1)
        count_fr = 0
        count_fr_tot = 0

        while end_samp < signal.shape[0]:
            sig_arr[count_fr, :] = signal[beg_samp:end_samp]

            beg_samp  = beg_samp + self.wshift
            end_samp = beg_samp + self.wlen
            count_fr = count_fr + 1
            count_fr_tot = count_fr_tot + 1
            if count_fr == self.Batch_dev:
                # inp = Variable(torch.from_numpy(sig_arr).float().cuda().contiguous())
                inp = Variable(torch.from_numpy(sig_arr).float().to(self.device).contiguous())
                # pout[count_fr_tot - self.Batch_dev:count_fr_tot, :] = self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))
                out = self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))
                out = out.squeeze()
                pout[count_fr_tot - self.Batch_dev:count_fr_tot] = out.data.cpu().numpy()

                count_fr = 0
                sig_arr = np.zeros([self.Batch_dev, self.wlen])

        if count_fr > 0:
            # inp = Variable(torch.from_numpy(sig_arr[0:count_fr]).float().cuda().contiguous())
            inp = inp = Variable(torch.from_numpy(sig_arr[0:count_fr]).float().to(self.device).contiguous())
            # pout[count_fr_tot - count_fr:count_fr_tot, :] = self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))
            out = self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))
            out = out.squeeze()
            pout[count_fr_tot - count_fr:count_fr_tot] = out.data.cpu().numpy()

        # pout = pout.squeeze()
        # pout = pout.data.cpu().numpy()

        #edit 11 april
        ##edit 4 February
        ## alpha = (self.wshift / self.wlen) * 2

        alpha = (self.wshift / (200*(self.fs/1000)))
        for i_fr in range(1, N_fr):
            pout[i_fr] =  alpha * pout[i_fr] + (1 - alpha) * pout[i_fr - 1]


        num_expand = int((self.wlen / 2) / self.wshift)
        exp_beg = np.ones(num_expand) * pout[0]
        exp_end = np.ones(num_expand) * pout[-1]
        pout_last = np.append(exp_beg, pout)
        pout_last = np.append(pout_last, exp_end)


        # getcontext().prec = 2
        pout_last = np.around(pout_last, decimals=2)

        treshold = 0.5
        pred = np.round(pout - treshold + 0.5)

        vad_out = np.ones([signal.shape[0]])
        vad_pout = np.zeros([signal.shape[0]])

        counter = np.zeros([signal.shape[0]])

        for i_fr in range(N_fr):
            start = int((self.wlen/2) + i_fr * self.wshift - (self.wshift/2))
            end = int((self.wlen / 2) + i_fr * self.wshift + (self.wshift / 2))

            # if pred[i_fr] == 1:
            #     vad_out[start:end] = np.ones([self.wshift])
            # else:
            #     vad_out[start:end] = np.zeros([self.wshift])
            # pout1 = np.exp(pout.data.cpu().numpy()[i_fr])

            #edit 11 april
            # vad_pout[start:end] = np.ones([self.wshift]) * pout[i_fr]
            # vad_out[start:end] = np.ones([self.wshift]) * pred[i_fr]
            vad_pout[i_fr * self.wshift : i_fr * self.wshift + self.wlen] += np.ones([self.wlen]) * pout[i_fr]
            counter[i_fr * self.wshift : i_fr * self.wshift + self.wlen] += np.ones([self.wlen])

        vad_pout[i_fr * self.wshift+self.wlen:] = np.ones(signal.shape[0] - (i_fr * self.wshift+self.wlen)) * pout[i_fr]

        # edit 11 april
        # idx = int((self.wlen / 2) - (self.wshift / 2))
        # vad_pout[0:idx] = np.ones([idx]) * vad_pout[idx]
        # vad_out[0:idx] = np.ones([idx]) * vad_out[idx]
        #
        # if end < vad_out.shape[0]-1:
        #     vad_pout[end:-1] = np.ones([vad_pout.shape[0]-end-1]) * vad_pout[end-1]
        #     vad_out[end:-1] = np.ones([vad_out.shape[0]-end-1]) * vad_out[end-1]
        for i in range(len(counter)):
            if counter[i] == 0:
                counter[i] = 1

        vad_pout = vad_pout / counter

        start = int(0.07 * self.fs)
        if start < len(signal):
            vad_pout = np.append(vad_pout[start:], np.ones(start)*vad_pout[-1])

        # vad_out = np.round(vad_pout - treshold + 0.5)

        # temp = vad_out - np.append(vad_out[0], vad_out[:-2])
        # indices = np.where(temp == 1)
        # w_repair = 0.1 * self.fs
        # for idx in indices:

        # newEnd = (len(vad_pout) % 80) * -1
        # if newEnd != 0:
        #     vad_pout = vad_pout[:newEnd]
        #
        # vad_pout = vad_pout.reshape(-1, 80).mean(axis=1)
        # vad_pout = np.append(vad_pout, vad_pout[-1])
        ratio = int(fs/100)
        len_vad = int((len(vad_pout) / ratio) + 1)
        if len(vad_pout) % ratio == 0:
            len_vad -= 1

        vad_res = np.zeros(len_vad)
        i = 0
        while i < len_vad-1:
            vad_res[i] = np.mean(vad_pout[i*ratio:(i+1)*ratio])
            i+=1

        if len(vad_pout) % ratio != 0:
            vad_res[-1] = np.mean(vad_pout[i*ratio:])

        # idx = np.argwhere(np.isnan(vad_res))

        # result = {}
        # result["vad_out"] = vad_res.tolist()

        # result['fs'] = 100
        # vad_json = json.dumps(result)

        # with open('w1.txt', 'w') as outfile:
        #     json.dump(result, outfile)

        return vad_res, 100
        # return vad_out, vad_pout
        # return pred, pout
